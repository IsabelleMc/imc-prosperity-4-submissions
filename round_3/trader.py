from datamodel import OrderDepth, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade
from typing import Any, List, Dict
import jsonpickle
import json
import math
from statistics import NormalDist


# ── NOTE ON THIS FILE'S STATUS (read before changing anything) ───────
# Nothing in this file is 100% confirmed correct. Several existing
# structures were verified post-Round-3 against day 0/1/2 data and found
# to be partially wrong:
#
#   - HYDROGEL_PACK anchor=10000 is biased ~+10 (true mean ~9990).
#   - VELVETFRUIT_EXTRACT anchor=5250 drifts off on day 2.
#   - The smile fit on VEV_5400 has a chronic -$2.50 bias that does NOT
#     reverse (verified take-signal IC ≈ 0).
#   - The wing fit on VEV_5500 is biased -$0.30 on days 1/2.
#   - The post-trade delta hedger is structurally undersized: voucher
#     delta capacity ≈ 945, VE limit = 180. 76% of day 3-2 ticks had
#     |net_voucher_delta| > VE_limit.
#
# The fact that this file scored 651/3600 in Round 3 (55,480 XIRECS) does
# NOT validate any specific component. Per-product PnL on training-like
# distributions can still be path-dependent or earned almost entirely
# through bid-ask spread rather than the FV/signal logic.
#
# Diagnostics live in tools/verify_*.py. Re-run them before assuming any
# existing structure is still the best option — anchor values, smile fit
# thresholds, hedger band, OBI thresholds, and PRODUCT_CONFIG are all
# tuning choices that survived Round 3 by happenstance, not by proof.
#
# Notation in comments:
#   "verified-X" = checked by a tools/verify_*.py script.
#   "claimed"    = trader.py's stated intent; not independently checked.

# ── Black-Scholes helpers ────────────────────────────────────────────
# Stdlib-only: external imports beyond stdlib + jsonpickle are not
# allowed in the submission file.

_NORM = NormalDist()


def _phi(x: float) -> float:
    return _NORM.cdf(x)


def bs_call_price(S, K, T, sigma, r=0.0):
    if T <= 0.0 or sigma <= 0.0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _phi(d1) - K * math.exp(-r * T) * _phi(d2)


def bs_delta(S, K, T, sigma, r=0.0):
    if T <= 0.0 or sigma <= 0.0:
        return 1.0 if S > K else 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return _phi(d1)


def bs_implied_vol(price, S, K, T, r=0.0,
                   lo=1e-4, hi=5.0, tol=1e-6, max_iter=60):
    """Bisection on sigma. Returns None outside the no-arb band."""
    if T <= 0.0:
        return None
    intrinsic = max(S - K, 0.0)
    if price < intrinsic - 1e-9 or price > S + 1e-9:
        return None
    if price <= intrinsic + 1e-9:
        return lo

    f_lo = bs_call_price(S, K, T, lo, r) - price
    f_hi = bs_call_price(S, K, T, hi, r) - price
    if f_lo > 0 or f_hi < 0:
        return None

    a, b = lo, hi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = bs_call_price(S, K, T, m, r) - price
        if abs(fm) < tol:
            return m
        if fm > 0:
            b = m
        else:
            a = m
        if b - a < tol:
            return 0.5 * (a + b)
    return 0.5 * (a + b)


def moneyness(S, K, T):
    """log(K/S) / sqrt(T) — vol-independent smile coordinate."""
    if T <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0
    return math.log(K / S) / math.sqrt(T)


# ── Smirk fit ────────────────────────────────────────────────────────
# IV(m) is a smirk: near-flat ATM, mild dip at VEV_5400, rising on the
# right wing. Two-region fit: anchor strikes (5000-5300) priced from a
# quadratic on their own four IVs; wing strikes (5500, 6000, 6500)
# priced from a quartic on the wing+anchor IVs. Keeps each region's
# pricing fit from fighting the other region's curvature.
#
# VEV_5400 is intentionally routed through the wing fit (not the
# anchor fit). The wing-vs-market gap on 5400 IS the take signal —
# rerouting through the anchor fit zeros out that edge.
ANCHOR_FIT_STRIKES = (5000, 5100, 5200, 5300)
ANCHOR_FIT_DEGREE = 2
WING_FIT_STRIKES = (5000, 5100, 5200, 5300, 5500, 6000, 6500)
WING_FIT_DEGREE = 4
ANCHOR_PRICED_STRIKES = frozenset({5000, 5100, 5200, 5300})

# Per-strike IV EMA on anchor strikes. The cross-strike anchor quadratic
# overfits the four near-coplanar IVs and its residual has ~0 correlation
# with future price moves — i.e. the "smile shape" is fit error, not
# signal. Each anchor prices off its own denoised IV instead.
IV_EMA_ALPHA = 0.1

# Hybrid bypass: if the cross-strike fit diverges from the EMA-FV by more
# than this gap, route the strike through the cross-strike fit instead of
# the EMA. The bypass is where the take alpha lives (rare large
# disagreements); the EMA smooths routine ticks.
HYBRID_BYPASS_GAP = 4.0

# Pre-trade voucher delta budget. Verified findings (tools/verify_*.py):
#   - Mid-strike fill asymmetry: VEV_5300/5400/5500 had ZERO sells on
#     day 3-2 → these strikes accumulate long inventory deterministically
#     and fail as MM lines.
#   - Hedger capacity is ±180 (VE limit) but raw voucher delta capacity
#     is ~945. Without a budget, day 3-2 ends at +862 net unhedged delta.
#   - Gamma-scalp alternative is not profitable after slippage
#     (verify_gamma_scalp.py); VE-direct OBI doesn't beat baseline
#     (verify_ve_signal.py). So directional inventory from broken-MM
#     strikes is a hidden bet, not alpha — bound it.
#
# BUDGET_STRIKES are the strikes where MM is verified-broken.
# Deep-ITM (4000/4500) and far-OTM (6000/6500) function differently:
#   - Deep-ITM: behave as delta-1 proxies, both-side fills work.
#   - Far-OTM: buy-only at $0 by design, no asymmetry to bound.
# So the budget targets only the broken-MM mid-strikes.
DELTA_BUDGET_STRIKES = ("VEV_5000", "VEV_5100", "VEV_5200",
                        "VEV_5300", "VEV_5400", "VEV_5500")
DELTA_BUDGET_SOFT = 100
DELTA_BUDGET_HARD = 160
DELTA_HEDGE_SIGMA = 0.16


def fit_polynomial(points, degree):
    """Least-squares polynomial fit IV = Σ c_k * m^k via normal equations
    + Gaussian elimination with partial pivoting. Returns increasing-power
    coeffs or None if degenerate.
    """
    n = degree + 1
    if len(points) < n:
        return None

    M = [[0.0] * (n + 1) for _ in range(n)]
    for m, iv in points:
        powers = [1.0]
        for _ in range(2 * degree):
            powers.append(powers[-1] * m)
        for i in range(n):
            for j in range(n):
                M[i][j] += powers[i + j]
            M[i][n] += powers[i] * iv

    for i in range(n):
        pivot_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[pivot_row][i]):
                pivot_row = k
        if abs(M[pivot_row][i]) < 1e-12:
            return None
        M[i], M[pivot_row] = M[pivot_row], M[i]
        for k in range(i + 1, n):
            factor = M[k][i] / M[i][i]
            for j in range(i, n + 1):
                M[k][j] -= factor * M[i][j]

    coeffs = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * coeffs[j]
        coeffs[i] = s / M[i][i]
    return tuple(coeffs)


def fit_smirk_anchor(points):
    return fit_polynomial(points, ANCHOR_FIT_DEGREE)


def fit_smirk_wing(points):
    return fit_polynomial(points, WING_FIT_DEGREE)


def pick_smirk(strike, fits):
    """Route a strike to its fit. fits = (overrides, anchor_coeffs,
    wing_coeffs). An override is a 1-tuple (iv_ema,) injected by the
    per-strike EMA path; falls through to the region polynomial otherwise.
    Returns None if no fit is available; caller surfaces as FV_NONE.
    """
    overrides, anchor_coeffs, wing_coeffs = fits
    if overrides and strike in overrides:
        return overrides[strike]
    if strike in ANCHOR_PRICED_STRIKES:
        return anchor_coeffs
    return wing_coeffs


def evaluate_smirk(coeffs, m):
    """Horner evaluation, increasing-power coefficients."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * m + c
    return result


# ── Time-to-expiry ──────────────────────────────────────────────────
# Vouchers expire 7d after Round 1 start; Round 3 final sim starts at
# TTE=5d. T_years = (5.0 - timestamp/TICKS_PER_DAY) / 365.
TICKS_PER_DAY = 1_000_000
TTE_DAYS_AT_ROUND_START = 5.0


def round3_tte_years(timestamp):
    elapsed_days = timestamp / TICKS_PER_DAY
    return max(0.0, (TTE_DAYS_AT_ROUND_START - elapsed_days)) / 365.0


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


# ── Base Algorithm ───────────────────────────────────────────────────

class BaseAlgo:
    """Base class for product-specific trading algorithms."""

    def compute_fair_value(self, order_depth, market_trades, state_data):
        """Return (fair_value, updated_state_data) or (None, state_data)."""
        raise NotImplementedError

    def default_state(self):
        """Return default per-product state dict."""
        return {}


# ── Rolling Anchor Mean Reversion ───────────────────────────────────


class RollingAnchorMeanReversionAlgo(BaseAlgo):
    """Mean reversion to a drifting rolling-SMA anchor.

    FV = rolling_SMA(window) blended with an EMA pull toward sample.
    Falls back to the fixed `anchor` parameter during warmup.
    Optional drift projection shifts FV to track persistent session drift.
    """

    def __init__(self, anchor=10000, alpha=0.5, window=200,
                 use_microprice=False, drift_project=0.0,
                 drift_clip=0.0, slow_ema_alpha=None):
        self.anchor = anchor  # fallback / warmup target
        self.alpha = alpha
        self.window = window
        self.use_microprice = use_microprice
        # drift_project=0 disables. drift_clip caps the |shift| so a
        # transient burst can't blow up FV.
        self.drift_project = drift_project
        self.drift_clip = drift_clip
        # When set, anchor uses a slow EMA (1 float) instead of a rolling
        # SMA (N floats). Equivalent for tracking the long-run mean but
        # robust to traderData-cap buffer trimming. alpha = 1/N gives an
        # EMA with effective window N.
        self.slow_ema_alpha = slow_ema_alpha
        self.slow_ema_warmup = int(1.0 / slow_ema_alpha) if slow_ema_alpha else 0

    def default_state(self):
        return {"ema": None, "last_mid": None, "mid_buffer": [],
                "slow_ema": None, "slow_n": 0}

    def compute_fair_value(self, order_depth, market_trades, state_data):
        sample = (
            self._microprice(order_depth) if self.use_microprice
            else self._midprice(order_depth)
        )
        if sample is None:
            sample = state_data.get("last_mid")
        if sample is None:
            return None, state_data
        state_data["last_mid"] = sample

        # Migrate older state dicts without the buffer
        buf = state_data.get("mid_buffer")
        if buf is None:
            buf = []
            state_data["mid_buffer"] = buf

        buf.append(sample)
        if len(buf) > self.window:
            del buf[0:len(buf) - self.window]

        # Anchor source: slow EMA (preferred when configured) or rolling SMA.
        if self.slow_ema_alpha is not None:
            slow_ema = state_data.get("slow_ema")
            slow_n = state_data.get("slow_n", 0)
            if slow_ema is None:
                slow_ema = sample
            else:
                slow_ema = (
                    self.slow_ema_alpha * sample
                    + (1.0 - self.slow_ema_alpha) * slow_ema
                )
            state_data["slow_ema"] = slow_ema
            state_data["slow_n"] = slow_n + 1
            if slow_n + 1 >= self.slow_ema_warmup:
                rolling_anchor = slow_ema
            else:
                rolling_anchor = self.anchor
        elif len(buf) >= self.window:
            rolling_anchor = sum(buf) / len(buf)
        else:
            rolling_anchor = self.anchor

        prev_ema = state_data.get("ema")
        ema = sample if prev_ema is None else (
            self.alpha * sample + (1 - self.alpha) * prev_ema
        )
        state_data["ema"] = ema

        fair_value = rolling_anchor + self.alpha * (ema - rolling_anchor)

        # Drift-project FV directly (shifting the anchor alone is too
        # weak at high alpha). Gated at 100 samples (not self.window):
        # Stage F.1 buffer-trimming may keep buf below window indefinitely.
        if self.drift_project > 0.0 and len(buf) >= 100:
            half = len(buf) // 2
            early_mean = sum(buf[:half]) / half
            late_mean = sum(buf[half:]) / (len(buf) - half)
            shift = self.drift_project * (late_mean - early_mean)
            if self.drift_clip > 0.0:
                if shift > self.drift_clip:
                    shift = self.drift_clip
                elif shift < -self.drift_clip:
                    shift = -self.drift_clip
            fair_value += shift

        return fair_value, state_data

    def _midprice(self, order_depth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def _microprice(self, order_depth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        bid_vol = order_depth.buy_orders[best_bid]
        ask_vol = abs(order_depth.sell_orders[best_ask])
        denom = bid_vol + ask_vol
        if denom == 0:
            return (best_bid + best_ask) / 2.0
        return (best_bid * ask_vol + best_ask * bid_vol) / denom


# ── EMA Fair Value Algorithm ─────────────────────────────────────────

class EMAFairValueAlgo(BaseAlgo):
    """EMA on microprice + trade imbalance. General-purpose algo."""

    IMBALANCE_WEIGHT = 2.0

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def default_state(self):
        return {"ema": None}

    def compute_fair_value(self, order_depth, market_trades, state_data):
        microprice = self._calculate_microprice(order_depth)
        if microprice is None:
            return None, state_data

        prev_ema = state_data.get("ema")
        ema = microprice if prev_ema is None else (
            self.alpha * microprice + (1 - self.alpha) * prev_ema
        )
        state_data["ema"] = ema

        imbalance = self._trade_imbalance(market_trades, order_depth)
        fair_value = ema + imbalance * self.IMBALANCE_WEIGHT
        return fair_value, state_data

    def _calculate_microprice(self, order_depth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        sorted_bids = sorted(order_depth.buy_orders.keys(), reverse=True)
        sorted_asks = sorted(order_depth.sell_orders.keys())
        best_bid = sorted_bids[0]
        best_ask = sorted_asks[0]
        bid_vol = sum(
            order_depth.buy_orders[p] for p in sorted_bids[:2]
        )
        ask_vol = sum(
            abs(order_depth.sell_orders[p]) for p in sorted_asks[:2]
        )
        if bid_vol + ask_vol == 0:
            return None
        return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

    def _trade_imbalance(self, market_trades, order_depth):
        if not market_trades:
            return 0.0
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0.0
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0
        buy_volume = 0
        sell_volume = 0
        for trade in market_trades:
            if trade.price >= mid:
                buy_volume += trade.quantity
            else:
                sell_volume += trade.quantity
        total = buy_volume + sell_volume
        if total == 0:
            return 0.0
        return (buy_volume - sell_volume) / total


class MicropriceAlgo(BaseAlgo):
    """Microprice fair value — bounded in [best_bid, best_ask].

    fv = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol).
    Because FV stays inside the spread, the take path never crosses.
    """

    def default_state(self):
        return {}

    def compute_fair_value(self, order_depth, market_trades, state_data):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None, state_data
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        bid_vol = order_depth.buy_orders[best_bid]
        ask_vol = abs(order_depth.sell_orders[best_ask])
        denom = bid_vol + ask_vol
        if denom == 0:
            return (best_bid + best_ask) / 2.0, state_data
        fair_value = (best_bid * ask_vol + best_ask * bid_vol) / denom
        return fair_value, state_data


# ── Deep-ITM Intrinsic Anchor ────────────────────────────────────────

class IntrinsicAnchorAlgo(BaseAlgo):
    """FV = max(S - K, 0). For deep-ITM vouchers (4000, 4500).

    Time premium on these is negligible. Anchoring to intrinsic tracks
    the underlying instead of drifting away from it. Requires S
    (passed by router).
    """

    def __init__(self, strike):
        self.strike = strike

    def default_state(self):
        return {}

    def compute_fair_value(self, order_depth, market_trades, state_data):
        return None, state_data

    def compute_voucher_fv(self, order_depth, S, T, smirk_coeffs, state_data):
        if S is None:
            return None, state_data
        return max(S - self.strike, 0.0), state_data


# ── Voucher (Black-Scholes + Smile) ─────────────────────────────────

class VoucherAlgo(BaseAlgo):
    """Smile-fitted Black-Scholes fair value for a Velvetfruit voucher.

    Router computes the smirk fit once per tick and passes coeffs + S +
    TTE via compute_voucher_fv. Returns None if the fit is missing or
    sigma is out of range; router surfaces that as FV_NONE.
    """

    def __init__(self, strike):
        self.strike = strike

    def default_state(self):
        return {}

    def compute_fair_value(self, order_depth, market_trades, state_data):
        # Returning None surfaces a routing bug instead of silently mid-pricing.
        return None, state_data

    def compute_voucher_fv(self, order_depth, S, T, smirk_coeffs, state_data):
        if S is None or smirk_coeffs is None or T <= 0.0:
            return None, state_data
        m = moneyness(S, self.strike, T)
        sigma = evaluate_smirk(smirk_coeffs, m)
        if sigma <= 0.0 or sigma > 5.0:
            return None, state_data
        fv = bs_call_price(S, self.strike, T, sigma)
        return fv, state_data


# ── Default (Conservative) Algorithm ─────────────────────────────────

class DefaultAlgo(EMAFairValueAlgo):
    """Conservative fallback for unknown products. Wider spread."""

    def __init__(self):
        super().__init__(alpha=0.2)
        self.IMBALANCE_WEIGHT = 1.0


# ── Product Configuration ────────────────────────────────────────────

MIDPRICE_BUFFER_SIZE = 50

RESID_BUFFER_SIZE = 500

# FV bias correction. Make-side asks priced off a biased fair_value land
# off-market and never fill (verified D2 day 3-2: 6/7 voucher strikes had
# zero sells). Per-product slow EMA of (mid - fair_value) estimates the
# systematic bias; subtract it from fair_value before market_make_orders.
# Take-side and resid_buffer keep using the raw fair_value (their own
# demean logic stays intact).
# Apply correction only when the per-product mean of (mid - fair_value)
# is statistically distinguishable from zero. Use t-stat = |mean| /
# (std/sqrt(n)) >= FV_BIAS_T_THRESH. For products like VELVETFRUIT_EXTRACT
# where the per-tick noise is large (std≈2.5) but mean is ~0.1, t stays
# small and we don't apply. For VEV_5400 (mean=-2.5, std≈1.5) t is huge.
FV_BIAS_ALPHA = 0.005           # mean EMA alpha (~200-tick eff window)
FV_BIAS_VAR_ALPHA = 0.001       # variance EMA alpha (slower; needs more data)
FV_BIAS_WARMUP = 500            # ticks before correction can apply
FV_BIAS_T_THRESH = 4.0          # t-stat threshold; conservative (z>4 ≈ p<1e-4)
FV_BIAS_MAX_ABS = 5.0           # absolute clamp; above = broken FV, don't trust

# ── IV-level z-score (cross-strike vol-level reversion) ─────────────
# Σ extrinsic value across anchor strikes (5000-5300) is monotonic in
# IV level. Within-session rolling-z detects vol-rich / vol-cheap
# regimes; |z| ≥ gate fires a one-tick MM quote skew via the
# `aggression` param. Don't smooth the z — the bypass is the alpha.
IV_LEVEL_BUFFER_SIZE = 1000
IV_LEVEL_Z_GATE = 0.5

SLOPE_WINDOW = 20
SLOPE_THRESHOLD = 0.02


def compute_slope_trend(prices, window=SLOPE_WINDOW, threshold=SLOPE_THRESHOLD):
    """Detect trend via simple slope of recent prices.

    Returns 1 (uptrend), -1 (downtrend), or 0 (flat).
    """
    if len(prices) < window:
        return 0
    recent = prices[-window:]
    slope = (recent[-1] - recent[0]) / (len(recent) - 1)
    if slope > threshold:
        return 1
    elif slope < -threshold:
        return -1
    return 0



def compute_obi_trend(order_depth, threshold=0.3):
    """L1 order-book imbalance as a momentum signal.

    OBI = (bv - av) / (bv + av) at L1.
    Returns +1 (mid will rise — suppress sells), -1 (mid will fall —
    suppress buys), or 0. Convention matches compute_slope_trend so the
    MM trend logic works unchanged.
    """
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return 0
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    bid_vol = order_depth.buy_orders[best_bid]
    ask_vol = abs(order_depth.sell_orders[best_ask])
    denom = bid_vol + ask_vol
    if denom == 0:
        return 0
    obi = (bid_vol - ask_vol) / denom
    if obi > threshold:
        return 1  # mid likely to rise → suppress sells
    elif obi < -threshold:
        return -1  # mid likely to fall → suppress buys
    return 0


def compute_obi_deep_trend(order_depth, threshold=0.2):
    """Deep-book imbalance (L2+L3 only, excluding L1) as REVERSION signal.

    Heavy-bid deep → mid will FALL (return -1); heavy-ask deep → mid
    will RISE (return +1). Sign INVERTED relative to compute_obi_trend
    because this is reversion not momentum, but matches the MM trend
    convention (+1 = suppress sells, -1 = suppress buys).
    """
    bv_deep = 0
    av_deep = 0
    if order_depth.buy_orders:
        bids = sorted(order_depth.buy_orders.keys(), reverse=True)
        for p in bids[1:]:  # skip L1
            bv_deep += order_depth.buy_orders[p]
    if order_depth.sell_orders:
        asks = sorted(order_depth.sell_orders.keys())
        for p in asks[1:]:
            av_deep += abs(order_depth.sell_orders[p])
    denom = bv_deep + av_deep
    if denom == 0:
        return 0
    ratio = (bv_deep - av_deep) / denom
    if ratio >= threshold:
        return -1
    elif ratio <= -threshold:
        return 1
    return 0


def compute_obi_deep_trend_banded(order_depth, lo=0.2, hi=0.5):
    """Banded compute_obi_deep_trend: fires only when |OBI_deep| ∈ [lo, hi).

    Saturated imbalance (≥ hi) is structural, not predictive, and would
    one-side our quotes too often. Reversion sign matches obi_deep.
    """
    bv_deep = 0
    av_deep = 0
    if order_depth.buy_orders:
        bids = sorted(order_depth.buy_orders.keys(), reverse=True)
        for p in bids[1:]:
            bv_deep += order_depth.buy_orders[p]
    if order_depth.sell_orders:
        asks = sorted(order_depth.sell_orders.keys())
        for p in asks[1:]:
            av_deep += abs(order_depth.sell_orders[p])
    denom = bv_deep + av_deep
    if denom == 0:
        return 0
    ratio = (bv_deep - av_deep) / denom
    abs_ratio = abs(ratio)
    if abs_ratio < lo or abs_ratio >= hi:
        return 0
    return -1 if ratio > 0 else 1


def compute_obi_total_trend(order_depth, threshold=0.05):
    """All-levels order-book imbalance as a REVERSION signal.

    OBI_total = (Σ bid_vol - Σ ask_vol) / (Σ bid_vol + Σ ask_vol).
    Heavy-bid → mid pulls DOWN (return -1); heavy-ask → mid pulls UP
    (return +1). Sign INVERTED vs compute_obi_trend (momentum), but
    matches the MM trend convention.
    """
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return 0
    bid_vol_total = sum(order_depth.buy_orders.values())
    ask_vol_total = sum(abs(q) for q in order_depth.sell_orders.values())
    denom = bid_vol_total + ask_vol_total
    if denom == 0:
        return 0
    obi_total = (bid_vol_total - ask_vol_total) / denom
    if obi_total >= threshold:
        return -1  # heavy-bid book → mid will pull down → suppress buys
    elif obi_total <= -threshold:
        return 1   # heavy-ask book → mid will pull up → suppress sells
    return 0


def compute_cumofi_signal(order_depth, state_data,
                          window=2000, z_thresh=2.0, decay=100):
    """Cumulative L1 order-flow imbalance (Cont-Kukanov-Stoikov 2014),
    integrated over a within-day window, z-scored. Extreme |z| events
    predict next-K-tick mid mean-reversion.

    OFI per tick:
        dv_b = +bv1                if best_bid moved UP
             = +(bv1 − prev_bv1)    if best_bid unchanged
             = −prev_bv1            if best_bid moved DOWN
        dv_a analogous; ofi = dv_b − dv_a.

    Returns (trend, aggression):
      trend: +1 LONG (suppress sells) when z ≤ -thresh
             -1 SHORT (suppress buys) when z ≥ +thresh
              0 otherwise
      aggression: 'bid' on LONG, 'ask' on SHORT, None otherwise

    Needs `window` ticks of warmup; returns (0, None) until then.
    """
    cf = state_data.get("cumofi")
    if cf is None:
        cf = {
            "prev_b1": None, "prev_a1": None,
            "prev_bv1": None, "prev_av1": None,
            "cof": 0.0,
            "cof_history": [],
            "active_long_counter": 0,
            "active_short_counter": 0,
        }
        state_data["cumofi"] = cf

    if not order_depth.buy_orders or not order_depth.sell_orders:
        # One-sided book: can't update OFI; decay active signals.
        cf["active_long_counter"] = max(0, cf["active_long_counter"] - 1)
        cf["active_short_counter"] = max(0, cf["active_short_counter"] - 1)
        return 0, None

    b1 = max(order_depth.buy_orders.keys())
    a1 = min(order_depth.sell_orders.keys())
    bv1 = order_depth.buy_orders[b1]
    av1 = abs(order_depth.sell_orders[a1])

    if cf["prev_b1"] is not None:
        pb, pa = cf["prev_b1"], cf["prev_a1"]
        pbv, pav = cf["prev_bv1"], cf["prev_av1"]
        if b1 > pb:
            dv_b = bv1
        elif b1 == pb:
            dv_b = bv1 - pbv
        else:
            dv_b = -pbv
        if a1 < pa:
            dv_a = av1
        elif a1 == pa:
            dv_a = av1 - pav
        else:
            dv_a = -pav
        ofi = dv_b - dv_a
        cf["cof"] += ofi
        cf["cof_history"].append(cf["cof"])
        # Keep window+1 so the z is computed against the window BEFORE
        # the current tick (excludes self from its own σ).
        if len(cf["cof_history"]) > window + 1:
            del cf["cof_history"][0:len(cf["cof_history"]) - (window + 1)]

    cf["prev_b1"] = b1
    cf["prev_a1"] = a1
    cf["prev_bv1"] = bv1
    cf["prev_av1"] = av1

    history = cf["cof_history"]
    z = 0.0
    if len(history) > window:
        window_vals = history[-window - 1:-1]
        mean_v = sum(window_vals) / len(window_vals)
        var = sum((v - mean_v) ** 2 for v in window_vals) / len(window_vals)
        if var > 0:
            std_v = math.sqrt(var)
            z = (cf["cof"] - mean_v) / std_v

    if z <= -z_thresh:
        cf["active_long_counter"] = decay
    if z >= z_thresh:
        cf["active_short_counter"] = decay

    active_long = cf["active_long_counter"] > 0
    active_short = cf["active_short_counter"] > 0

    cf["active_long_counter"] = max(0, cf["active_long_counter"] - 1)
    cf["active_short_counter"] = max(0, cf["active_short_counter"] - 1)

    if active_long and not active_short:
        return 1, "bid"
    if active_short and not active_long:
        return -1, "ask"
    return 0, None




# `limit` values encode the 90%-of-A.R.I.A.-cap rule (HYDROGEL/VE 180,
# vouchers 270).
PRODUCT_CONFIG = {
    "HYDROGEL_PACK": {
        # window=200 retained: verified-faster anchor (slow_ema_alpha=
        # 1/5000) tested and lost ~12k/day. The MM is doing local spread
        # capture around a fast-tracking anchor, not long-run mean
        # reversion to 10000. The autocorrelation at lag 5000 (-0.42) is
        # real but isn't directly tradable through this algo's structure.
        "algo": RollingAnchorMeanReversionAlgo(
            anchor=10000, alpha=0.50, window=200),
        "limit": 180,
        "make": True,
        "soft_position_limit": 30,
        # join_edge=10 covers HYDROGEL's wide (~16-tick) spread.
        "disregard_edge": 1,
        "join_edge": 10,
        "default_edge": 4,
        "min_take_edge": 1,
        "trend_filter": ["slope", "obi_total"],
        "obi_total_threshold": 0.05,
        "cumofi_skew": {
            "window": 2000,
            "z_thresh": 2.0,
            "decay": 100,
        },
    },
    "VELVETFRUIT_EXTRACT": {
        # window=2000 retained: same finding as HYDROGEL — slow EMA
        # anchor was tested and degraded PnL on every day.
        "algo": RollingAnchorMeanReversionAlgo(
            anchor=5250, alpha=0.90, window=2000, use_microprice=True),
        "limit": 180,
        "make": True,
        "soft_position_limit": 15,
        "disregard_edge": 0,
        "join_edge": 3,
        "default_edge": 2,
    },
    # Deep-ITM (4000, 4500): priced at intrinsic max(S-K, 0). Time
    # premium is negligible.
    "VEV_4000": {
        "algo": IntrinsicAnchorAlgo(strike=4000),
        "limit": 270, "make": True,
        "soft_position_limit": 270,
        "disregard_edge": 0, "join_edge": 3, "default_edge": 1,
    },
    # VEV_4500 sells lose on rallies (algo treats real time-value as
    # overpricing). Buys-only via make_sell=False + min_edge_sell=1e9.
    "VEV_4500": {
        "algo": IntrinsicAnchorAlgo(strike=4500),
        "limit": 270, "make": True,
        "soft_position_limit": 80,
        "disregard_edge": 0, "join_edge": 3, "default_edge": 1,
        "min_edge_sell": 1e9,
        "make_sell": False,
    },
    # min_take_edge=3 on the smile-fit strikes gates against fit noise
    # (typical residual $1-3); only 5400 is loosened.
    "VEV_5000": {
        "algo": VoucherAlgo(strike=5000),
        "limit": 270, "make": True,
        "soft_position_limit": 30,
        "disregard_edge": 0, "join_edge": 3, "default_edge": 4,
        "min_take_edge": 3,
        "resid_gate": {"warmup": 50, "k_gate": 2.0, "k_full": 4.0},
        "iv_level_bias": True,
    },
    "VEV_5100": {
        "algo": VoucherAlgo(strike=5100),
        "limit": 270, "make": True,
        "soft_position_limit": 30,
        "disregard_edge": 0, "join_edge": 3, "default_edge": 4,
        "min_take_edge": 3,
        "resid_gate": {"warmup": 50, "k_gate": 2.0, "k_full": 4.0},
        "iv_level_bias": True,
    },
    "VEV_5200": {
        "algo": VoucherAlgo(strike=5200),
        "limit": 270, "make": True,
        "soft_position_limit": 30,
        "disregard_edge": 0, "join_edge": 3, "default_edge": 4,
        "min_take_edge": 3,
        "resid_gate": {"warmup": 50, "k_gate": 2.0, "k_full": 4.0},
        "trend_filter": "obi_deep", "obi_deep_threshold": 0.2,
        "iv_level_bias": True,
    },
    "VEV_5300": {
        "algo": VoucherAlgo(strike=5300),
        "limit": 270, "make": True,
        "soft_position_limit": 270,
        "disregard_edge": 0, "join_edge": 3, "default_edge": 4,
        "min_take_edge": 3,
        "resid_gate": {"warmup": 50, "k_gate": 2.0, "k_full": 4.0},
        "trend_filter": "obi_deep", "obi_deep_threshold": 0.2,
        "iv_level_bias": True,
    },
    # 5400 sits structurally below the smirk by ~$2.50. Verified
    # tools/verify_smile_fit.py: when |mid - FV| > 1 fires (subtract_mean=False),
    # mean Δmid in the predicted direction is ~+0.01 (no reversion). The
    # offset is structural, not a take signal. Demean it and only fire on
    # deviations from the rolling mean.
    "VEV_5400": {
        "algo": VoucherAlgo(strike=5400),
        "limit": 270, "make": True,
        "soft_position_limit": 270,
        "disregard_edge": 0, "join_edge": 3, "default_edge": 4,
        "min_take_edge": 3,
        "resid_gate": {"warmup": 50, "k_gate": 2.0, "k_full": 4.0},
        "trend_filter": "obi_deep", "obi_deep_threshold": 0.2,
    },
    "VEV_5500": {
        "algo": VoucherAlgo(strike=5500),
        "limit": 270, "make": True,
        "soft_position_limit": 270,
        "disregard_edge": 0, "join_edge": 3, "default_edge": 4,
        "min_take_edge": 3,
        "resid_gate": {"warmup": 50, "k_gate": 2.0, "k_full": 4.0},
    },
    # Deep-OTM (6000, 6500): book is permanently 0/1 and counterparties
    # dump at the bid. Buy at $0 (bounded-loss free optionality);
    # make_sell=False keeps the $0 bid quote but suppresses any sell.
    "VEV_6000": {
        "algo": VoucherAlgo(strike=6000),
        "limit": 270, "make": True, "make_sell": False,
        "soft_position_limit": 270,
        "disregard_edge": 0, "join_edge": 3, "default_edge": 4,
        "min_take_edge": 3,
        "resid_gate": {"warmup": 50, "k_gate": 2.0, "k_full": 4.0},
    },
    "VEV_6500": {
        "algo": VoucherAlgo(strike=6500),
        "limit": 270, "make": True, "make_sell": False,
        "soft_position_limit": 270,
        "disregard_edge": 0, "join_edge": 3, "default_edge": 4,
        "min_take_edge": 3,
        "resid_gate": {"warmup": 50, "k_gate": 2.0, "k_full": 4.0},
    },
}

DEFAULT_ALGO = DefaultAlgo()
DEFAULT_LIMIT = 20


# ── Order Helpers ───────────────────────────────────────────────────


def _size_from_edge(edge, min_edge, full_size_edge, raw_cap):
    """Linear ramp: 0 at min_edge, raw_cap at full_size_edge, saturating.
    Caller has already gated edge > min_edge so this returns >= 1.
    """
    if full_size_edge <= min_edge:
        return raw_cap
    frac = (edge - min_edge) / (full_size_edge - min_edge)
    if frac >= 1.0:
        return raw_cap
    sized = int(raw_cap * frac)
    return max(1, sized)


def take_orders(
    order_depth, product, fair_value, position, limit, trend=0,
    min_edge=0, full_size_edge=None,
    min_edge_buy=None, min_edge_sell=None,
    full_size_edge_buy=None, full_size_edge_sell=None,
):
    """Take mispriced book levels.

    Returns (orders, buy_volume, sell_volume). Counters track per-side
    capacity consumed so make_orders can size against true remaining
    headroom (prevents limit breaches when take + make fill same tick).

    Symmetric gates: `min_edge` / `full_size_edge` apply to both sides.
    Per-side overrides (`{min,full_size}_edge_{buy,sell}`) let the
    residual-aware gates demean a per-strike systematic offset.
    `full_size_edge` enables deviation-sized positioning (qty ramps
    linearly from 1 at min_edge+1 to raw_cap at >= full_size_edge);
    None falls back to full depth at any qualifying edge.
    """
    orders = []
    buy_volume = 0
    sell_volume = 0

    me_buy = min_edge_buy if min_edge_buy is not None else min_edge
    me_sell = min_edge_sell if min_edge_sell is not None else min_edge
    fse_buy = full_size_edge_buy if full_size_edge_buy is not None else full_size_edge
    fse_sell = full_size_edge_sell if full_size_edge_sell is not None else full_size_edge

    if order_depth.sell_orders and trend >= 0:
        for ask_price, ask_qty in sorted(order_depth.sell_orders.items()):
            edge = fair_value - ask_price
            if edge <= me_buy:
                break
            raw_cap = limit - (position + buy_volume)
            if raw_cap <= 0:
                break
            sized_cap = (
                _size_from_edge(edge, me_buy, fse_buy, raw_cap)
                if fse_buy is not None else raw_cap
            )
            qty = min(-ask_qty, sized_cap)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_volume += qty

    if order_depth.buy_orders and trend <= 0:
        for bid_price, bid_qty in sorted(
            order_depth.buy_orders.items(), reverse=True
        ):
            edge = bid_price - fair_value
            if edge <= me_sell:
                break
            raw_cap = limit + (position - sell_volume)
            if raw_cap <= 0:
                break
            sized_cap = (
                _size_from_edge(edge, me_sell, fse_sell, raw_cap)
                if fse_sell is not None else raw_cap
            )
            qty = min(bid_qty, sized_cap)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_volume += qty

    return orders, buy_volume, sell_volume


def market_make_orders(
    order_depth, product, fair_value, position, limit,
    buy_volume=0, sell_volume=0, trend=0,
    disregard_edge=1, join_edge=2, default_edge=4,
    soft_position_limit=0,
    aggression=None,
):
    """5-param market maker.

    Per side: find best book level outside `disregard_edge` from fair;
    if within `join_edge` join, else penny ahead; if none, quote at
    fair ± default_edge. Inventory tilt shifts the relevant quote by 1
    when |position| > soft_position_limit. Sizes against the headroom
    NET of buy_volume/sell_volume so take + make in the same tick can't
    breach the position limit.

    trend: +1 = suppress buys, -1 = suppress sells.
    """
    orders = []

    asks_above_fair = [p for p in order_depth.sell_orders
                       if p > fair_value + disregard_edge]
    bids_below_fair = [p for p in order_depth.buy_orders
                       if p < fair_value - disregard_edge]
    best_ask_above = min(asks_above_fair) if asks_above_fair else None
    best_bid_below = max(bids_below_fair) if bids_below_fair else None

    ask = round(fair_value + default_edge)
    if best_ask_above is not None:
        if abs(best_ask_above - fair_value) <= join_edge:
            ask = best_ask_above
        else:
            ask = best_ask_above - 1

    bid = round(fair_value - default_edge)
    if best_bid_below is not None:
        if abs(fair_value - best_bid_below) <= join_edge:
            bid = best_bid_below
        else:
            bid = best_bid_below + 1

    if position > soft_position_limit:
        ask -= 1
    elif position < -soft_position_limit:
        bid += 1

    # Aggression: push the signalled side one tick tighter (applied
    # AFTER penny/join and inventory-tilt). Guarded so it cannot cross
    # the touch even on a one-sided book.
    if aggression == "bid":
        if order_depth.sell_orders:
            best_ask_abs = min(order_depth.sell_orders.keys())
            bid = min(bid + 1, best_ask_abs - 1)
        else:
            bid = min(bid + 1, int(round(fair_value)) - 1)
    elif aggression == "ask":
        if order_depth.buy_orders:
            best_bid_abs = max(order_depth.buy_orders.keys())
            ask = max(ask - 1, best_bid_abs + 1)
        else:
            ask = max(ask - 1, int(round(fair_value)) + 1)

    buy_qty = limit - (position + buy_volume)
    sell_qty = limit + (position - sell_volume)

    if trend >= 0 and buy_qty > 0:
        orders.append(Order(product, int(bid), buy_qty))
    if trend <= 0 and sell_qty > 0:
        orders.append(Order(product, int(ask), -sell_qty))

    return orders


# ── Trader (Router) ─────────────────────────────────────────────────

class Trader:

    def bid(self):
        """Round 2 manual-bid hook. Round 3 has no second-bid mechanic;
        kept as a placeholder for forward-portability."""
        return 0

    @staticmethod
    def _midprice(order_depth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Surface planted signals or new products IMC drops mid-round.
        try:
            pvo = getattr(state.observations, "plainValueObservations", None)
            if pvo:
                print("PLANTED_SIGNAL plainValueObservations=" + str(pvo))
        except Exception:
            pass
        for product in state.order_depths:
            if product not in PRODUCT_CONFIG:
                print("UNKNOWN_PRODUCT " + str(product))

        trader_state = {}
        if state.traderData:
            try:
                trader_state = jsonpickle.decode(state.traderData)
            except Exception:
                print("TRADER_STATE_DECODE_FAIL len=" + str(len(state.traderData)))
                trader_state = {}
            if not isinstance(trader_state, dict):
                trader_state = {}

        # ── Per-tick smirk fit (used by all VoucherAlgo products) ──
        # Per-strike IVs computed once, shared by anchor + wing fits.
        ve_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        underlying_mid = self._midprice(ve_depth) if ve_depth else None
        tte_years = round3_tte_years(state.timestamp)
        anchor_coeffs = None
        wing_coeffs = None
        if underlying_mid is not None and tte_years > 0.0:
            iv_pts_by_K = {}
            for K_fit in WING_FIT_STRIKES:
                sym = f"VEV_{K_fit}"
                v_depth = state.order_depths.get(sym)
                v_mid = self._midprice(v_depth) if v_depth else None
                if v_mid is None or v_mid <= 0.0:
                    continue
                iv = bs_implied_vol(v_mid, underlying_mid, K_fit, tte_years)
                if iv is None:
                    continue
                iv_pts_by_K[K_fit] = (moneyness(underlying_mid, K_fit, tte_years), iv)

            anchor_pts = [iv_pts_by_K[K] for K in ANCHOR_FIT_STRIKES if K in iv_pts_by_K]
            wing_pts = [iv_pts_by_K[K] for K in WING_FIT_STRIKES if K in iv_pts_by_K]
            anchor_coeffs = fit_smirk_anchor(anchor_pts)
            wing_coeffs = fit_smirk_wing(wing_pts)
            if anchor_coeffs is None:
                print(f"ANCHOR_FIT_FAIL ts={state.timestamp} n_pts={len(anchor_pts)}")
            if wing_coeffs is None:
                print(f"WING_FIT_FAIL ts={state.timestamp} n_pts={len(wing_pts)}")

            # Per-strike anchor IV EMA. Polynomial stays as warmup
            # fallback and as the wing fit.
            iv_ema = trader_state.get("_iv_ema")
            if iv_ema is None:
                iv_ema = {}
                trader_state["_iv_ema"] = iv_ema
            for K_anchor in ANCHOR_FIT_STRIKES:
                if K_anchor not in iv_pts_by_K:
                    continue
                _, iv_now = iv_pts_by_K[K_anchor]
                prev = iv_ema.get(K_anchor)
                iv_ema[K_anchor] = (
                    iv_now if prev is None
                    else IV_EMA_ALPHA * iv_now + (1.0 - IV_EMA_ALPHA) * prev
                )
        # Hybrid: EMA wins routine ticks; cross-strike fit takes over
        # when it disagrees with the EMA by > HYBRID_BYPASS_GAP dollars.
        # That bypass is where the take alpha sits. Compare in $ (gating unit).
        anchor_overrides = {}
        if underlying_mid is not None and tte_years > 0.0:
            iv_ema_now = trader_state.get("_iv_ema") or {}
            for K_anchor in ANCHOR_PRICED_STRIKES:
                iv_e = iv_ema_now.get(K_anchor)
                if iv_e is None or iv_e <= 0.0 or iv_e > 5.0:
                    continue
                fv_ema = bs_call_price(underlying_mid, K_anchor, tte_years, iv_e)
                use_ema = True
                if anchor_coeffs is not None:
                    m_K = moneyness(underlying_mid, K_anchor, tte_years)
                    sigma_cross = evaluate_smirk(anchor_coeffs, m_K)
                    if 0.0 < sigma_cross < 5.0:
                        fv_cross = bs_call_price(
                            underlying_mid, K_anchor, tte_years, sigma_cross,
                        )
                        if abs(fv_cross - fv_ema) > HYBRID_BYPASS_GAP:
                            use_ema = False
                if use_ema:
                    anchor_overrides[K_anchor] = (iv_e,)
                # else: no override → pick_smirk falls through to anchor_coeffs
        smirk_fits = (anchor_overrides, anchor_coeffs, wing_coeffs)

        # ── IV-level z-score ──
        # extrinsic_sum = Σ (mid_K − max(S − K, 0)) over anchor strikes,
        # monotonic in vol level. Rolling-z it within session and pass to
        # per-leg gate adjustment. Don't smooth z — bypass is the alpha.
        iv_level_z = 0.0
        if underlying_mid is not None:
            ext_sum = 0.0
            n_anchor = 0
            for K_anchor in ANCHOR_FIT_STRIKES:
                v_sym = f"VEV_{K_anchor}"
                v_depth = state.order_depths.get(v_sym)
                v_mid = self._midprice(v_depth) if v_depth else None
                if v_mid is None:
                    continue
                ext_sum += v_mid - max(underlying_mid - K_anchor, 0.0)
                n_anchor += 1
            if n_anchor == len(ANCHOR_FIT_STRIKES):
                iv_buf = trader_state.get("_iv_level_buf")
                if iv_buf is None:
                    iv_buf = []
                    trader_state["_iv_level_buf"] = iv_buf
                # z computed BEFORE appending so current tick can't bias
                # its own σ down. 200-tick warmup avoids early-session
                # artificially-small σ (spurious large z).
                if len(iv_buf) >= 200:
                    mu = sum(iv_buf) / len(iv_buf)
                    var = sum((v - mu) ** 2 for v in iv_buf) / len(iv_buf)
                    if var > 0.0:
                        iv_level_z = (ext_sum - mu) / math.sqrt(var)
                iv_buf.append(ext_sum)
                if len(iv_buf) > IV_LEVEL_BUFFER_SIZE:
                    del iv_buf[0:len(iv_buf) - IV_LEVEL_BUFFER_SIZE]

        # ── Pre-trade voucher delta budget (mid-strikes only) ──
        # Sum only over DELTA_BUDGET_STRIKES — the broken-MM strikes.
        # Deep-ITM/far-OTM strikes are functionally different and not
        # included here. σ=0.16 is fixed; ATM voucher deltas aren't very
        # σ-sensitive at short TTE so the approximation is fine.
        current_voucher_delta = 0.0
        if underlying_mid is not None and tte_years > 0.0:
            for v_sym in DELTA_BUDGET_STRIKES:
                v_pos = state.position.get(v_sym, 0)
                if v_pos == 0:
                    continue
                K = int(v_sym.split("_")[1])
                if K <= underlying_mid - 200:
                    d = 1.0
                elif K >= underlying_mid + 400:
                    d = 0.0
                else:
                    d = bs_delta(underlying_mid, K, tte_years, DELTA_HEDGE_SIGMA)
                current_voucher_delta += v_pos * d

        abs_d = abs(current_voucher_delta)
        if abs_d <= DELTA_BUDGET_SOFT:
            voucher_size_scale = 1.0
        elif abs_d >= DELTA_BUDGET_HARD:
            voucher_size_scale = 0.0
        else:
            voucher_size_scale = (
                (DELTA_BUDGET_HARD - abs_d)
                / (DELTA_BUDGET_HARD - DELTA_BUDGET_SOFT)
            )
        voucher_long_off = current_voucher_delta >= DELTA_BUDGET_HARD
        voucher_short_off = current_voucher_delta <= -DELTA_BUDGET_HARD

        print(
            f"DELTA_BUDGET d={current_voucher_delta:+.0f} "
            f"scale={voucher_size_scale:.2f} "
            f"long_off={voucher_long_off} short_off={voucher_short_off}"
        )

        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]

            # Unknown product: do nothing until it has an explicit config.
            if product not in PRODUCT_CONFIG:
                result[product] = []
                continue

            config = PRODUCT_CONFIG.get(product, {})
            limit = config.get("limit", DEFAULT_LIMIT)
            position = state.position.get(product, 0)

            algo = config.get("algo", DEFAULT_ALGO)
            use_make = config.get("make", False)
            trend_filter = config.get("trend_filter", None)
            disregard_edge = config.get("disregard_edge", 1)
            join_edge = config.get("join_edge", 2)
            default_edge = config.get("default_edge", 4)
            soft_position_limit = config.get("soft_position_limit", limit // 2)

            if product not in trader_state:
                trader_state[product] = algo.default_state()
                trader_state[product]["midprices"] = []

            if "midprices" not in trader_state[product]:
                trader_state[product]["midprices"] = []

            market_trades = state.market_trades.get(product, [])
            if isinstance(algo, (VoucherAlgo, IntrinsicAnchorAlgo)):
                # IntrinsicAnchor ignores coeffs (always max(S-K, 0)).
                strike_for_fit = getattr(algo, "strike", None)
                coeffs_for_strike = (
                    pick_smirk(strike_for_fit, smirk_fits)
                    if strike_for_fit is not None else None
                )
                fair_value, trader_state[product] = algo.compute_voucher_fv(
                    order_depth, underlying_mid, tte_years, coeffs_for_strike,
                    trader_state[product],
                )
            else:
                fair_value, trader_state[product] = algo.compute_fair_value(
                    order_depth, market_trades, trader_state[product]
                )

            if fair_value is None:
                # Never silent-default in pricing — surface refusal so
                # "dead book" is distinguishable from "bug".
                print(f"FV_NONE {product}")
                result[product] = []
                continue

            mid = self._midprice(order_depth)
            if mid is not None:
                prices = trader_state[product]["midprices"]
                prices.append(mid)
                if len(prices) > MIDPRICE_BUFFER_SIZE:
                    trader_state[product]["midprices"] = (
                        prices[-MIDPRICE_BUFFER_SIZE:]
                    )

            # Per-strike residual buffer (mid - fair_value) drives the
            # residual-aware gates: mean = systematic offset to demean,
            # stdev = noise floor for deviation-sized ramping.
            if isinstance(algo, VoucherAlgo) and mid is not None:
                rb = trader_state[product].get("resid_buffer")
                if rb is None:
                    rb = []
                    trader_state[product]["resid_buffer"] = rb
                rb.append(mid - fair_value)
                if len(rb) > RESID_BUFFER_SIZE:
                    trader_state[product]["resid_buffer"] = (
                        rb[-RESID_BUFFER_SIZE:]
                    )

            # FV bias correction — make-side only, voucher algos only.
            # Mean-reversion algos (HYDROGEL/VELVETFRUIT) are designed to
            # disagree with mid; their "bias" is the alpha. Voucher FVs
            # are supposed to track the option's market price; bias there
            # is genuine model error (verified VEV_5400: chronic -$2.50
            # gap → asks priced off-market → zero sells filled).
            # EMA tracks mean(mid - FV); variance EMA tracks per-tick
            # noise. Apply correction only when t-stat >= FV_BIAS_T_THRESH.
            fv_bias_corrected = fair_value
            if mid is not None and isinstance(
                algo, (VoucherAlgo, IntrinsicAnchorAlgo)
            ):
                ps = trader_state[product]
                bias_ema = ps.get("fv_bias_ema", 0.0)
                var_ema = ps.get("fv_bias_var_ema", 0.0)
                bias_n = ps.get("fv_bias_n", 0)
                bias_now = mid - fair_value
                if bias_n == 0:
                    bias_ema = bias_now
                    var_ema = 0.0
                else:
                    diff = bias_now - bias_ema
                    bias_ema = bias_ema + FV_BIAS_ALPHA * diff
                    var_ema = (
                        FV_BIAS_VAR_ALPHA * (diff * diff)
                        + (1.0 - FV_BIAS_VAR_ALPHA) * var_ema
                    )
                ps["fv_bias_ema"] = bias_ema
                ps["fv_bias_var_ema"] = var_ema
                ps["fv_bias_n"] = bias_n + 1
                # std of the EMA mean ≈ σ × sqrt(α/(2-α)).
                std_mean = (
                    math.sqrt(var_ema * FV_BIAS_ALPHA / (2 - FV_BIAS_ALPHA))
                    if var_ema > 0 else 0.0
                )
                t_stat = abs(bias_ema) / std_mean if std_mean > 1e-9 else 0.0
                if (
                    bias_n + 1 >= FV_BIAS_WARMUP
                    and t_stat >= FV_BIAS_T_THRESH
                    and abs(bias_ema) <= FV_BIAS_MAX_ABS
                ):
                    fv_bias_corrected = fair_value + bias_ema

            # trend_filter: str or list of str. Each returns a sign;
            # we sum and take the sign so agreeing signals reinforce
            # and disagreeing ones cancel.
            filters = trend_filter
            if isinstance(filters, str):
                filters = [filters]
            elif filters is None:
                filters = []
            obi_thresh = config.get("obi_threshold", 0.3)
            obi_total_thresh = config.get("obi_total_threshold", 0.05)
            trend_sum = 0
            for f in filters:
                if f == "slope":
                    trend_sum += compute_slope_trend(
                        trader_state[product]["midprices"])
                elif f == "obi":
                    trend_sum += compute_obi_trend(
                        order_depth, threshold=obi_thresh)
                elif f == "obi_total":
                    trend_sum += compute_obi_total_trend(
                        order_depth, threshold=obi_total_thresh)
                elif f == "obi_deep":
                    obi_deep_thresh = config.get("obi_deep_threshold", 0.2)
                    trend_sum += compute_obi_deep_trend(
                        order_depth, threshold=obi_deep_thresh)
                elif f == "obi_deep_banded":
                    lo = config.get("obi_deep_band_lo", 0.2)
                    hi = config.get("obi_deep_band_hi", 0.5)
                    trend_sum += compute_obi_deep_trend_banded(
                        order_depth, lo=lo, hi=hi)

            # cumOFI: contributes to trend AND provides an aggression
            # direction. Soft-cap-gated so we don't lean further into a
            # side we're already saturated on.
            cumofi_aggression = None
            cumofi_config = config.get("cumofi_skew", None)
            if cumofi_config:
                cof_trend, cumofi_aggression = compute_cumofi_signal(
                    order_depth, trader_state[product],
                    window=cumofi_config.get("window", 2000),
                    z_thresh=cumofi_config.get("z_thresh", 2.0),
                    decay=cumofi_config.get("decay", 100),
                )
                trend_sum += cof_trend
                if cumofi_aggression == "bid" and position >= soft_position_limit:
                    cumofi_aggression = None
                elif cumofi_aggression == "ask" and position <= -soft_position_limit:
                    cumofi_aggression = None

            trend = 0 if trend_sum == 0 else (1 if trend_sum > 0 else -1)

            print(
                f"{product} | FV: {fair_value:.2f} "
                f"| algo: {type(algo).__name__} "
                f"| Pos: {position}/{limit}"
                f"| make: {use_make}"
                f"| trend: {trend}"
                f"| cof_aggr: {cumofi_aggression}"
            )

            min_take_edge = config.get("min_take_edge", 0)
            full_size_edge = config.get("full_size_edge", None)

            # Residual-aware gates for VoucherAlgo: demean the per-strike
            # offset and ramp size by edge / σ_resid. Static per-product
            # overrides (e.g. VEV_4500 sell-side via min_edge_sell=1e9)
            # take precedence by being read first.
            min_edge_buy = config.get("min_edge_buy", None)
            min_edge_sell = config.get("min_edge_sell", None)
            full_size_edge_buy = None
            full_size_edge_sell = None
            resid_cfg = config.get("resid_gate", None)
            rb = trader_state.get(product, {}).get("resid_buffer", [])
            if (
                isinstance(algo, VoucherAlgo)
                and resid_cfg
                and len(rb) >= resid_cfg.get("warmup", 50)
            ):
                n = len(rb)
                mean_r = sum(rb) / n
                var_r = sum((r - mean_r) ** 2 for r in rb) / n
                std_r = math.sqrt(var_r) if var_r > 0 else 0.0
                k_gate = resid_cfg.get("k_gate", 2.0)
                k_full = resid_cfg.get("k_full", 4.0)
                # subtract_mean=True: gate at ±(mean ± k·σ) so we don't
                # trade the systematic offset. False: the offset IS the
                # signal (e.g. VEV_5400) — keep min_take_edge, use σ
                # only for the ramp.
                if resid_cfg.get("subtract_mean", True):
                    min_edge_buy = max(min_take_edge, -mean_r + k_gate * std_r)
                    min_edge_sell = max(min_take_edge, mean_r + k_gate * std_r)
                    full_size_edge_buy = -mean_r + k_full * std_r
                    full_size_edge_sell = mean_r + k_full * std_r
                else:
                    full_size_edge_buy = min_take_edge + k_full * std_r
                    full_size_edge_sell = min_take_edge + k_full * std_r

            # IV-level z bias: vol-rich (z >> 0) → push ask tighter
            # (FV likely to fall); vol-cheap → push bid tighter. Below
            # IV_LEVEL_Z_GATE leave quotes alone. Shares the MM
            # `aggression` param with cumOFI — combined below.
            iv_level_aggression = None
            if (
                config.get("iv_level_bias", False)
                and abs(iv_level_z) >= IV_LEVEL_Z_GATE
            ):
                if iv_level_z > 0:
                    iv_level_aggression = "ask"
                else:
                    iv_level_aggression = "bid"
                if iv_level_aggression == "bid" and position >= soft_position_limit:
                    iv_level_aggression = None
                elif iv_level_aggression == "ask" and position <= -soft_position_limit:
                    iv_level_aggression = None

            orders, buy_volume, sell_volume = take_orders(
                order_depth, product, fair_value, position, limit,
                trend=trend, min_edge=min_take_edge,
                full_size_edge=full_size_edge,
                min_edge_buy=min_edge_buy,
                min_edge_sell=min_edge_sell,
                full_size_edge_buy=full_size_edge_buy,
                full_size_edge_sell=full_size_edge_sell,
            )

            # Voucher delta budget: drop takes on the loaded side.
            # Applied only to DELTA_BUDGET_STRIKES (the broken-MM
            # mid-strikes); deep-ITM and far-OTM vouchers are exempt.
            is_budgeted = product in DELTA_BUDGET_STRIKES
            if is_budgeted:
                if voucher_long_off:
                    buy_volume -= sum(o.quantity for o in orders if o.quantity > 0)
                    orders = [o for o in orders if o.quantity <= 0]
                if voucher_short_off:
                    sell_volume += sum(o.quantity for o in orders if o.quantity < 0)
                    orders = [o for o in orders if o.quantity >= 0]

            # Combine aggressions: same direction reinforces (no-op —
            # each nudges 1 tick), opposite cancels, one non-None wins.
            if cumofi_aggression and iv_level_aggression:
                aggression = cumofi_aggression if cumofi_aggression == iv_level_aggression else None
            else:
                aggression = cumofi_aggression or iv_level_aggression

            # Budgeted-strike size scale: applied to soft_position_limit
            # only, not the hard limit. Linear 1.0 below SOFT, 0.0 at HARD.
            effective_soft_limit = soft_position_limit
            if is_budgeted:
                effective_soft_limit = max(
                    1, int(soft_position_limit * voucher_size_scale)
                )

            if use_make:
                make_orders = market_make_orders(
                    order_depth, product, fv_bias_corrected,
                    position, limit,
                    buy_volume=buy_volume, sell_volume=sell_volume,
                    trend=trend,
                    disregard_edge=disregard_edge,
                    join_edge=join_edge,
                    default_edge=default_edge,
                    soft_position_limit=effective_soft_limit,
                    aggression=aggression,
                )
                # Per-product side disables (e.g. VEV_4500 sell side) +
                # budget-driven side disables for budgeted strikes.
                make_buy_ok = config.get("make_buy", True)
                make_sell_ok = config.get("make_sell", True)
                if is_budgeted:
                    if voucher_long_off:
                        make_buy_ok = False
                    if voucher_short_off:
                        make_sell_ok = False
                if not make_buy_ok:
                    make_orders = [o for o in make_orders if o.quantity <= 0]
                if not make_sell_ok:
                    make_orders = [o for o in make_orders if o.quantity >= 0]
                orders.extend(make_orders)

            result[product] = orders

        # ── Delta hedge BUDGETED voucher strikes against VE ──
        # Hedger only neutralizes the mid-strike book (DELTA_BUDGET_STRIKES).
        # Deep-ITM strikes (VEV_4000/4500) are accepted directional positions
        # — buy-only configs, intentional long-S exposure. Far-OTM
        # (VEV_6000/6500) carry no meaningful delta. Hedging only what
        # the budget actually controls keeps the hedger from being
        # structurally underwater (verified: aggregate hedge needs ~945
        # of capacity vs VE limit 180; budgeted-only hedge needs ~160).
        if (
            underlying_mid is not None
            and tte_years > 0.0
            and (anchor_coeffs is not None or wing_coeffs is not None)
        ):
            net_voucher_delta = 0.0
            for v_sym in DELTA_BUDGET_STRIKES:
                v_cfg = PRODUCT_CONFIG.get(v_sym)
                if v_cfg is None:
                    continue
                v_algo = v_cfg.get("algo")
                v_strike = getattr(v_algo, "strike", None)
                if v_strike is None:
                    continue
                v_pos = state.position.get(v_sym, 0)
                # Best-effort post-fill estimate — assumes our orders fill.
                for o in result.get(v_sym, []):
                    v_pos += o.quantity
                if v_pos == 0:
                    continue
                if isinstance(v_algo, IntrinsicAnchorAlgo):
                    delta = 1.0 if underlying_mid > v_strike else 0.0
                else:
                    coeffs_for_v = pick_smirk(v_strike, smirk_fits)
                    if coeffs_for_v is None:
                        continue
                    m = moneyness(underlying_mid, v_strike, tte_years)
                    sigma = evaluate_smirk(coeffs_for_v, m)
                    if sigma <= 0.0 or sigma > 5.0:
                        continue
                    delta = bs_delta(underlying_mid, v_strike, tte_years, sigma)
                net_voucher_delta += v_pos * delta

            ve_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            ve_target = -int(round(net_voucher_delta))
            ve_existing_orders = result.get("VELVETFRUIT_EXTRACT", [])
            ve_buy_already = sum(o.quantity for o in ve_existing_orders if o.quantity > 0)
            ve_sell_already = -sum(o.quantity for o in ve_existing_orders if o.quantity < 0)
            ve_limit = PRODUCT_CONFIG["VELVETFRUIT_EXTRACT"]["limit"]

            # Band-and-rest hedger: inside the band we rest passively at
            # the touch (gamma scalp the spread); outside, cross to clear.
            HEDGE_DELTA_BAND = 5
            hedge_qty = ve_target - ve_pos

            if abs(hedge_qty) > HEDGE_DELTA_BAND:
                if hedge_qty > 0:
                    cap = ve_limit - (ve_pos + ve_buy_already)
                    hedge_qty = min(hedge_qty, max(0, cap))
                    if hedge_qty > 0 and ve_depth and ve_depth.sell_orders:
                        best_ask = min(ve_depth.sell_orders.keys())
                        ve_existing_orders.append(
                            Order("VELVETFRUIT_EXTRACT", best_ask, hedge_qty)
                        )
                else:
                    cap = ve_limit + (ve_pos - ve_sell_already)
                    hedge_qty = max(hedge_qty, -max(0, cap))
                    if hedge_qty < 0 and ve_depth and ve_depth.buy_orders:
                        best_bid = max(ve_depth.buy_orders.keys())
                        ve_existing_orders.append(
                            Order("VELVETFRUIT_EXTRACT", best_bid, hedge_qty)
                        )
            elif hedge_qty != 0 and ve_depth:
                if hedge_qty > 0 and ve_depth.buy_orders:
                    cap = ve_limit - (ve_pos + ve_buy_already)
                    hedge_qty = min(hedge_qty, max(0, cap))
                    if hedge_qty > 0:
                        rest_bid = max(ve_depth.buy_orders.keys())
                        ve_existing_orders.append(
                            Order("VELVETFRUIT_EXTRACT", rest_bid, hedge_qty)
                        )
                elif hedge_qty < 0 and ve_depth.sell_orders:
                    cap = ve_limit + (ve_pos - ve_sell_already)
                    hedge_qty = max(hedge_qty, -max(0, cap))
                    if hedge_qty < 0:
                        rest_ask = min(ve_depth.sell_orders.keys())
                        ve_existing_orders.append(
                            Order("VELVETFRUIT_EXTRACT", rest_ask, hedge_qty)
                        )
            result["VELVETFRUIT_EXTRACT"] = ve_existing_orders

        traderData = jsonpickle.encode(trader_state)
        # Platform caps traderData at 50k chars; halve list buffers at
        # 45k to leave margin before truncation.
        if len(traderData) > 45_000:
            print(f"TRADER_DATA_LARGE len={len(traderData)} — trimming buffers")
            for prod_state in trader_state.values():
                if not isinstance(prod_state, dict):
                    continue
                for key, val in list(prod_state.items()):
                    if isinstance(val, list) and len(val) > 1:
                        prod_state[key] = val[len(val) // 2:]
                    elif isinstance(val, dict):
                        for k2, v2 in list(val.items()):
                            if isinstance(v2, list) and len(v2) > 1:
                                val[k2] = v2[len(v2) // 2:]
            traderData = jsonpickle.encode(trader_state)
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData