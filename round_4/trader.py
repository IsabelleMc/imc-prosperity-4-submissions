from datamodel import OrderDepth, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade
from typing import Any, List, Dict
import jsonpickle
import json
import math
from statistics import NormalDist


# ── Logger ───────────────────────────────────────────────────────────

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


# ── Black-Scholes helpers (stdlib-only) ──────────────────────────────

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


def bs_implied_vol(price, S, K, T, r=0.0,
                   lo=1e-4, hi=5.0, tol=1e-6, max_iter=60):
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


def log_moneyness(S, K):
    if S <= 0.0 or K <= 0.0:
        return 0.0
    return math.log(K / S)


# Quadratic least-squares fit y = a + b*x + c*x^2 via normal equations.
def fit_quadratic(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    s0 = float(n)
    s1 = sum(xs)
    s2 = sum(x * x for x in xs)
    s3 = sum(x * x * x for x in xs)
    s4 = sum(x * x * x * x for x in xs)
    sy = sum(ys)
    sxy = sum(xs[i] * ys[i] for i in range(n))
    sx2y = sum(xs[i] * xs[i] * ys[i] for i in range(n))
    # Solve [[s0,s1,s2],[s1,s2,s3],[s2,s3,s4]] @ [a,b,c] = [sy,sxy,sx2y].
    M = [[s0, s1, s2, sy], [s1, s2, s3, sxy], [s2, s3, s4, sx2y]]
    for i in range(3):
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            for k in range(i + 1, 3):
                if abs(M[k][i]) > 1e-12:
                    M[i], M[k] = M[k], M[i]
                    pivot = M[i][i]
                    break
            else:
                return None
        for k in range(i + 1, 3):
            factor = M[k][i] / pivot
            for j in range(i, 4):
                M[k][j] -= factor * M[i][j]
    c = M[2][3] / M[2][2]
    b = (M[1][3] - M[1][2] * c) / M[1][1]
    a = (M[0][3] - M[0][1] * b - M[0][2] * c) / M[0][0]
    return (a, b, c)


# ── R4 PRODUCT_CONFIG ────────────────────────────────────────────────
# Position limits per R4 wiki: HYDROGEL=200, VE=200, each VEV=300.
# Strikes/role come from the audited plan.

VEV_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500,
    "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}

# Strikes used in the smile fit (exclude deep-ITM intrinsic-pinned and
# wings with no live two-sided market).
SMILE_FIT_STRIKES = ("VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500")

# Strikes we actually quote off the smile (role="smile_mm" in PRODUCT_CONFIG).
# Each gets its own leave-one-out smile fit (see fit_smile exclude=) so
# the fair isn't pulled toward the strike's own market IV.
# Only VEV_5000 is safely smile-MM-able: 5400/5500 lose D3 unhedged
# (TTE-decay monotone drift on wing strikes crushes long-vol).
SMILE_QUOTE_STRIKES = ("VEV_5000",)

# TTE in years, stepwise per day. R4 starts at TTE=4 calendar days.
# day index in the data is 1-based per the prices CSV; live env passes
# the same convention via state.timestamp ranges. We use a per-day
# integer counter inferred from cumulative ticks (state.timestamp
# resets each day in the CSV; live we'll lean on 0..1e6 = one day).
DAYS_PER_YEAR = 252.0
TTE_DAYS_AT_R4_START = 4

# Bayesian-seeded VE anchor for dev_directional. Replaces the 4000-tick
# warmup gate so VEV_5100/5200/5300/5400/5500 trade from tick 0.
# anchor_t = (BAYES_N_PRIOR * BAYES_MU_PRIOR + sum_live) / (BAYES_N_PRIOR + n_live)
# (1500, 5252) chosen via 4-day sweep: 4-day total 781,140 vs 691,640
# baseline; held-out R3 D0 181k vs 139k baseline. Best D2 (149k) of the
# entire sweep grid; held-out lift +42k.
BAYES_MU_PRIOR = 5252
BAYES_N_PRIOR = 1500

PRODUCT_CONFIG: Dict[str, Dict[str, Any]] = {
    "HYDROGEL_PACK": {
        "limit": 200,
        # Regime-conditional anchor. The HYDROGEL book is symmetric
        # (microprice == mid); 10012 is the *modal attractor* of the mid
        # distribution, not a bot-quote bias. On days whose first 4000
        # ticks touch >10050, the operating range shifts up and a lower
        # anchor (9997) wins (R4 D3: +14,943; R3 D0 held-out: +14,711).
        # Low/normal-range days keep the 10012 modal target.
        "anchor": 10012,                # default / "low-range" anchor
        "regime_anchor_low": 9997,      # used when warmup_max > regime_band_high
        "regime_max_threshold": 10050,  # legacy single-threshold (kept as fallback)
        # Piecewise-linear blend across [regime_band_low, regime_band_high].
        # All 4 observed warmup_max (10033, 10034, 10061, 10071) lie outside
        # the band, so in-sample P&L is byte-identical to the binary cliff.
        # On D4, a warmup_max inside the band gets an interpolated anchor —
        # worst-case mis-anchor 7.5 ticks instead of 15.
        "regime_band_low": 10042,
        "regime_band_high": 10060,
        "warmup_window": 4000,          # shared with VE warmup architecture
        "default_edge": 4,
        "join_edge": 10,
        "disregard_edge": 1,
        # Take threshold sweep: 3-day in-sample peak at 7 (+59,736 vs
        # baseline=1). Below 7 we cross on noise that reverts; above 9 we
        # miss real dislocations. MTE=1: 193,956. MTE=5: 247,002. MTE=6:
        # 252,206. MTE=7: 253,692. MTE=8: 248,192.
        # Per-day optima differ: D1/D3 prefer 7, D2 slightly prefers 6.
        # Held-out R3 D0 prefers 8. Median across all 4 days = 7. Direction
        # is robust (MTE=7 way above MTE=1 on every day including held-out
        # +11k); the exact value 7 vs 8 is within sample noise.
        "min_take_edge": 7,
        # L3 fires (~150/day each side) predict +/-4 ticks fwd mid (t>7);
        # use looser take threshold than base when fired. Sweep peaks at 6
        # vs base=7: +1,096 over 3 days. Modest because the +4 mid drift
        # is mostly the ask-side widening back out, not bid-side improving.
        # Held-out R3 D0: -147 (sign flip on the one out-of-sample day).
        # Net 4-day: +949. Borderline; kept because the underlying L3
        # signal is mechanically real (sparse deep-book imbalance, t=7-9)
        # and the downside is bounded at ~150 PnL/bad day.
        "l3_min_take_edge": 6,
        "soft_position_limit": 60,
        "obi_threshold": 0.50,          # unused on HYDROGEL since the OBI-skew block was removed
        "per_quote_cap": 50,            # cap single-tick fills (else 200-lot dumps on day open)
        # Take-size ramp from MTE+1 (size=1) to FSE (size=full_cap). At
        # FSE<=MTE the ramp is short-circuited (full size from edge=MTE+1).
        # FSE=15 wins total +1,236 but loses on D2 (-289). FSE=10 wins on
        # D2 (+194) AND total (+644 vs FSE=6) — chosen for D2 consistency
        # since D2 is the structurally weakest day (max mid 10051 vs
        # D1/D3 maxes 10079/10081 — fewer high-side dislocations to
        # harvest, so smaller dislocations matter more).
        "full_size_edge": 10,
    },
    "VELVETFRUIT_EXTRACT": {
        "limit": 200,
        # Static anchor 5247 = cross-day mean. Rolling FV (window=2000)
        # was tested and lost -13k on day 3 (V-shape chasing). Slow EWMA
        # (alpha=1/3000) lost -27k.
        # The fix: freeze-after-warmup anchor. Average mids over first
        # warmup_window ticks then freeze for rest of day. Sweep across
        # 4 days (R3 D0 held-out + R4 D1-3): window=4000 wins on BOTH
        # in-sample (R4 D1-3 +2.8k vs no-warmup) and held-out (R3 D0
        # +1.7k vs no-warmup). Larger windows (6500/8000) score higher
        # in-sample but LOSE on R3 D0 -- classic overfit signature.
        # Window=4000 is the robust pick.
        # MTE=12, SPL=200 chosen via 2D sweep:
        #   SPL ∈ {60,80,100,150,200} at MTE=12 -> SPL=200 monotone-best,
        #     including R4 D2 (the cliff day): 17.9k -> 19.5k. No drawdown
        #     concern from raising SPL.
        #   MTE ∈ {8,10,11,12,13,15} at SPL=200 -> plateau MTE ∈ {11,12,13}
        #     all within ~5k on R4 3-day total. MTE=12 wins R3 D0 held-out
        #     by +2k vs MTE=11 -> picked for robustness, not knife-edge.
        # Verified delta: R4 3-day VE 62,758 -> 113,831 (+51,073);
        # R3 D0 held-out VE 26,477 -> 34,952 (+8,475).
        "anchor": 5247,
        "warmup_window": 4000,
        "default_edge": 2,
        "join_edge": 3,
        "disregard_edge": 0,
        "min_take_edge": 12,
        "soft_position_limit": 200,
        "per_quote_cap": 60,
        "obi_threshold": 0.50,
    },
    # Deep-ITM delta-1 (voucher mid = VE mid - 4000 within 0.01,
    # max abs deviation 7). Wide market spread ~21 ticks; bot flow is
    # balanced (~50/50 aggr_buy/aggr_sell). min_take_edge=1 captures
    # any parity deviation > 1 tick (very common given the wide
    # market spread). Soft limit 200 uses the full 300-cap as
    # VE-equivalent inventory.
    "VEV_4000": {
        "limit": 300,
        "strike": 4000,
        "role": "delta1_proxy",
        "default_edge": 3,
        "join_edge": 11,
        "min_take_edge": 0,
        "soft_position_limit": 200,
    },
    # VEV_4500: parity holds (avg_dev=0.008, 91% within +/-0.5 ticks of
    # ve_mid - 4500) -- structurally identical to VEV_4000 delta-1 proxy.
    # The "1-3 trades/day" prior verdict was a misread: bot aggressor
    # trade count is low, but quote book is full (avg spread 15.8, L1
    # size 8.9). Mirror VEV_4000's (je=11, mte=0) wide-quote-join +
    # half-tick-parity-take config.
    # Verified delta: D1 +4,977 / D2 +11,770 / D3 -590 (3-day +16,157).
    "VEV_4500": {
        "limit": 300,
        "strike": 4500,
        "role": "delta1_proxy",
        "default_edge": 3,
        "join_edge": 11,
        "min_take_edge": 0,
        "soft_position_limit": 200,
    },
    # Smile-MM strikes (audited): chronic IV residuals are stable across
    # all 3 days. ACF(lag10) ~ 0.65-0.69 -> persistent regime, not noise.
    # VEV_5000: leave-one-out smile fit gives unbiased fair for fade.
    # R4 D1-3 research (research/iv_residual_horizons.py): per-trade
    # delta-hedged PnL t-stat 18.5/27.7/23.9, win rate 57-62% at h=50.
    # Tight inventory caps because the previous smile-MM attempt on
    # VEV_5400 lost -16k on D3 (regime drift on unhedged short-vol).
    "VEV_5000": {
        "limit": 300, "strike": 5000, "role": "smile_mm",
        "default_edge": 1, "join_edge": 1, "min_take_edge": 1,
        # max_position sweep: 45 was D1-only optimum on R4 in-sample. Held-out
        # R3 D0 has a smile-regime episode that builds adverse inventory at
        # cap=45 (-251 v5000 P&L). Sweep 20..45 monotone in v5000 4-day P&L:
        # mp=20 +732 / mp=25 +724 / mp=30 +629 / mp=45 +365 (baseline).
        # Plateau on 20-25, no R4-day regression (signal fill rate caps below
        # mp=20 on R4). Ship mp=20: bounds D5 regime-drift downside.
        "soft_position_limit": 22, "max_position": 20,
    },
    # VEV_5100/5200/5300/5400/5500 — VE-deviation directional:
    # corr(ve_mid - frozen_anchor, fwd_1k_tick option mid drift) = -0.5 to
    # -0.8 across all 5 strikes on every R4 day AND on R3 D0 (held-out).
    # Mechanism: when VE is above anchor, options will fall (sell now);
    # when VE is below, options will rise (buy now). Cycle on cross-back.
    # Threshold 20, max_position=300 (full limit), 20-lot fills sized to
    # avoid single-tick blowups. Replaces smile_mm/spread_mm on these
    # strikes -- the prior LOO smile fair tracked mid too closely to
    # generate executable take edges (verified: VEV_5000 had 1 fill in
    # 10000 ticks). Verified delta:
    #   R4 D1: +75,988 / D2: +27,728 / D3: +65,337 (total +169,053)
    #   R3 D0 held-out: +62,626 (sign-positive on every strike).
    "VEV_5100": {
        "limit": 300, "strike": 5100, "role": "dev_directional",
        "max_position": 300, "dev_threshold": 20, "fill_size": 20,
    },
    "VEV_5200": {
        "limit": 300, "strike": 5200, "role": "dev_directional",
        "max_position": 300, "dev_threshold": 20, "fill_size": 20,
    },
    "VEV_5300": {
        "limit": 300, "strike": 5300, "role": "dev_directional",
        "max_position": 300, "dev_threshold": 20, "fill_size": 20,
    },
    "VEV_5400": {
        "limit": 300, "strike": 5400, "role": "dev_directional",
        "max_position": 300, "dev_threshold": 20, "fill_size": 20,
    },
    "VEV_5500": {
        "limit": 300, "strike": 5500, "role": "dev_directional",
        "max_position": 300, "dev_threshold": 20, "fill_size": 20,
    },
    # Wings: book pinned bid=0/ask=1 with all flow Mark01<->Mark22 at price 0.
    # We post a $0 bid (free OTM lottery, bounded loss) but never sell.
    "VEV_6000": {"limit": 300, "strike": 6000, "role": "wing_lottery"},
    "VEV_6500": {"limit": 300, "strike": 6500, "role": "wing_lottery"},
}


# ── Helpers ──────────────────────────────────────────────────────────

def best_bid_ask(order_depth):
    bb = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
    ba = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
    return bb, ba


def midprice(order_depth):
    bb, ba = best_bid_ask(order_depth)
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2.0


def obi_l1(order_depth):
    bb, ba = best_bid_ask(order_depth)
    if bb is None or ba is None:
        return 0.0
    bv = order_depth.buy_orders[bb]
    av = abs(order_depth.sell_orders[ba])
    denom = bv + av
    if denom <= 0:
        return 0.0
    return (bv - av) / denom


def quartile(timestamp):
    """Q0..Q3 within a Prosperity day (timestamp 0..999900, 100-step)."""
    return min(3, int(timestamp) // 250000)


# ── Smile fit ────────────────────────────────────────────────────────

def compute_tte():
    """TTE in years. Live submission is a single day; the backtester
    runs day-by-day and resets state.timestamp each day. Use a single
    TTE constant; per-day stepwise decrement is a future refinement
    (Day-3 chronic IV residual was already mildly compressed vs Day 1).
    """
    return TTE_DAYS_AT_R4_START / DAYS_PER_YEAR


def fit_smile(state, S, T, exclude=None):
    """Fit IV = a + b*log_m + c*log_m^2 over SMILE_FIT_STRIKES.
    exclude: optional symbol to drop (leave-one-out fit, used to get an
        unbiased fair for that strike — without it, smile_iv at the
        excluded K is regression-pulled toward the strike's own market IV).
    Returns (coeffs, points) or (None, []) if fit fails.
    """
    xs = []
    ys = []
    for sym in SMILE_FIT_STRIKES:
        if sym == exclude:
            continue
        depth = state.order_depths.get(sym)
        if depth is None:
            continue
        m = midprice(depth)
        if m is None:
            continue
        K = VEV_STRIKES[sym]
        iv = bs_implied_vol(m, S, K, T)
        if iv is None:
            continue
        xs.append(log_moneyness(S, K))
        ys.append(iv)
    if len(xs) < 3:
        return None, []
    coeffs = fit_quadratic(xs, ys)
    if coeffs is None:
        return None, []
    return coeffs, list(zip(xs, ys))


def smile_iv(coeffs, S, K):
    a, b, c = coeffs
    x = log_moneyness(S, K)
    return a + b * x + c * x * x


# ── Order helpers ────────────────────────────────────────────────────

def _size_from_edge(edge, min_edge, full_size_edge, raw_cap):
    # Linear ramp: returns 1 at min_edge+1, raw_cap at full_size_edge.
    # Synthia R3 hint "scale exposure to deviation magnitude". On R4 only
    # HYDROGEL has L2 levels deeper than 25 lots that this actually bites
    # on -- VE/VEV_4000 L1 sizes are below the ramped cap at any plausible
    # edge, so wiring full_size_edge there is a no-op.
    if full_size_edge <= min_edge:
        return raw_cap
    frac = (edge - min_edge) / (full_size_edge - min_edge)
    if frac >= 1.0:
        return raw_cap
    return max(1, int(raw_cap * frac))


def take_orders(order_depth, product, fair_value, position, limit,
                min_take_edge=1, side_filter=None, full_size_edge=None):
    """Take levels with edge > min_take_edge against fair.
    side_filter: None | 'buy_only' | 'sell_only'.
    full_size_edge: when set, ramp take qty linearly from 1 at min_take_edge+1
        to full available cap at edge >= full_size_edge.
    Returns (orders, buy_volume, sell_volume).
    """
    orders = []
    bv = 0
    sv = 0
    if side_filter != "sell_only" and order_depth.sell_orders:
        for ask_price, ask_qty in sorted(order_depth.sell_orders.items()):
            edge = fair_value - ask_price
            if edge <= min_take_edge:
                break
            cap = limit - (position + bv)
            if cap <= 0:
                break
            sized_cap = (
                _size_from_edge(edge, min_take_edge, full_size_edge, cap)
                if full_size_edge is not None else cap
            )
            qty = min(-ask_qty, sized_cap)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                bv += qty
    if side_filter != "buy_only" and order_depth.buy_orders:
        for bid_price, bid_qty in sorted(order_depth.buy_orders.items(), reverse=True):
            edge = bid_price - fair_value
            if edge <= min_take_edge:
                break
            cap = limit + (position - sv)
            if cap <= 0:
                break
            sized_cap = (
                _size_from_edge(edge, min_take_edge, full_size_edge, cap)
                if full_size_edge is not None else cap
            )
            qty = min(bid_qty, sized_cap)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sv += qty
    return orders, bv, sv


def make_orders(order_depth, product, fair_value, position, limit,
                buy_volume, sell_volume, default_edge, join_edge,
                disregard_edge, soft_position_limit, side_filter=None,
                quote_skew=0, per_quote_cap=None):
    """5-param MM around fair_value.
    quote_skew: +1 lean long (raise bid), -1 lean short (lower ask), 0 neutral.
    per_quote_cap: max size posted per quote (caps single-tick fills).
        Without this, on the first tick of a session we may post the
        full limit (e.g. 200) and have it all hit in one shot, pinning
        us at the cap before we have any read on the regime. None
        means "use full available headroom".
    """
    orders = []
    asks_above = [p for p in order_depth.sell_orders if p > fair_value + disregard_edge]
    bids_below = [p for p in order_depth.buy_orders if p < fair_value - disregard_edge]
    best_ask_above = min(asks_above) if asks_above else None
    best_bid_below = max(bids_below) if bids_below else None
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
    # Inventory tilt
    if position > soft_position_limit:
        ask -= 1
    elif position < -soft_position_limit:
        bid += 1
    # Counterparty / OBI quote skew (audited overlay).
    # Touch-crossing guard (R3 lines 1186-1197): a 1-tick lean can push
    # past the touch on a near-zero-spread or one-sided book. Clamp
    # bid+1 to (best_ask - 1) and ask-1 to (best_bid + 1) when both
    # touches exist; on a one-sided book clamp to fair-1 / fair+1.
    if quote_skew > 0:
        if order_depth.sell_orders:
            best_ask_abs = min(order_depth.sell_orders.keys())
            bid = min(bid + 1, best_ask_abs - 1)
        else:
            bid = min(bid + 1, int(round(fair_value)) - 1)
    elif quote_skew < 0:
        if order_depth.buy_orders:
            best_bid_abs = max(order_depth.buy_orders.keys())
            ask = max(ask - 1, best_bid_abs + 1)
        else:
            ask = max(ask - 1, int(round(fair_value)) + 1)
    buy_qty = limit - (position + buy_volume)
    sell_qty = limit + (position - sell_volume)
    if per_quote_cap is not None:
        buy_qty = min(buy_qty, per_quote_cap)
        sell_qty = min(sell_qty, per_quote_cap)
    if side_filter != "sell_only" and buy_qty > 0:
        orders.append(Order(product, int(bid), buy_qty))
    if side_filter != "buy_only" and sell_qty > 0:
        orders.append(Order(product, int(ask), -sell_qty))
    return orders


# ── Counterparty-trigger detection ───────────────────────────────────
# Audited triggers (R4 Synthia "interpret a sudden price shift"):
#   Mark 22 SELL VE with mid drop >= 2 -> mid recovers +2.62 (passive long fade)
#   Mark 49 SELL VE with mid drop >= 2 -> +1.77
# Plus baseline lean signals:
#   Mark 67 BUY VE in Q2/Q3 -> +0.9
#   Mark 14 BUY HYD Q3 -> +0.55 (use as quote-skew, half-spread is 8)
#   Mark 38 SELL HYD Q3 -> mirror, fade

JUMP_THRESHOLD = 2  # ticks of mid-move at trade tick to trigger jump-fade

JUMP_FADE_TRIGGERS = {
    # (product, side, counterparty): bias direction (+1 long, -1 short)
    ("VELVETFRUIT_EXTRACT", "sell", "Mark 22"): +1,
    ("VELVETFRUIT_EXTRACT", "sell", "Mark 49"): +1,
}

# Q-quartile follow signals (mid-jump-conditional already covers most;
# these are softer overlays for quote skew).
QUARTILE_LEAN_SIGNALS = {
    # (product, side, counterparty, quartile): skew direction
    ("VELVETFRUIT_EXTRACT", "buy", "Mark 67", 1): +1,
    ("VELVETFRUIT_EXTRACT", "buy", "Mark 67", 2): +1,
    # HYD: Q3 was flat across all 3 R4 days (t≈0). Q1+Q2 is where the edge
    # lives — pooled t=+3.23 (Mark 14 BUY) / t=+2.67 (Mark 38 SELL).
    ("HYDROGEL_PACK", "buy",  "Mark 14", 1): +1,
    ("HYDROGEL_PACK", "buy",  "Mark 14", 2): +1,
    ("HYDROGEL_PACK", "sell", "Mark 38", 1): +1,
    ("HYDROGEL_PACK", "sell", "Mark 38", 2): +1,
}

# R4 D1-3 sweep tested degating Mark 67 BUY VE (Q1/Q2 -> unconditional)
# and Mark 49 SELL VE (jump-gated -> unconditional). Both produced 0
# P&L delta in backtest because quote_skew=±1 is bandwidth-saturated
# on VE: the skew is mostly absorbed by the touch clamp on a 5-tick
# spread, and aggregate_lean already saturates at ±1 when a same-sign
# lean is active. Channel limit, not signal limit.
UNCONDITIONAL_LEAN_SIGNALS = {}

LEAN_DECAY_TICKS = 10  # forward window 1000 ts = 10 ticks of 100 ts


def detect_counterparty_signals(state, prev_mids, persistent_leans):
    """Process state.market_trades for counterparty triggers.
    Returns dict: {symbol: {'lean': int, 'count': int}} with lean direction.
    Updates persistent_leans (decremented each tick).
    """
    # Decrement existing leans
    expired = []
    for key, info in persistent_leans.items():
        info["ticks_left"] -= 1
        if info["ticks_left"] <= 0:
            expired.append(key)
    for key in expired:
        del persistent_leans[key]

    market_trades = state.market_trades or {}
    for sym, trades in market_trades.items():
        depth = state.order_depths.get(sym)
        if depth is None:
            continue
        cur_mid = midprice(depth)
        prev_mid = prev_mids.get(sym)
        if prev_mid is None or cur_mid is None:
            continue
        # Trade list may include both this-tick and last-tick fills.
        for tr in trades:
            if tr.timestamp != state.timestamp - 100:
                # Only act on trades that happened in the immediately
                # preceding tick (jump from prev_mid -> cur_mid).
                continue
            buyer = tr.buyer
            seller = tr.seller
            jump = cur_mid - prev_mid
            # Jump-fade: SELL trigger needs cur_mid < prev_mid by JUMP_THRESHOLD
            if seller and (sym, "sell", seller) in JUMP_FADE_TRIGGERS:
                if jump <= -JUMP_THRESHOLD:
                    bias = JUMP_FADE_TRIGGERS[(sym, "sell", seller)]
                    persistent_leans[(sym, "jumpfade", seller)] = {
                        "bias": bias, "ticks_left": LEAN_DECAY_TICKS,
                    }
            # Quartile-conditional lean
            q = quartile(state.timestamp)
            if buyer and (sym, "buy", buyer, q) in QUARTILE_LEAN_SIGNALS:
                bias = QUARTILE_LEAN_SIGNALS[(sym, "buy", buyer, q)]
                persistent_leans[(sym, "qlean_buy", buyer)] = {
                    "bias": bias, "ticks_left": LEAN_DECAY_TICKS,
                }
            if seller and (sym, "sell", seller, q) in QUARTILE_LEAN_SIGNALS:
                bias = QUARTILE_LEAN_SIGNALS[(sym, "sell", seller, q)]
                persistent_leans[(sym, "qlean_sell", seller)] = {
                    "bias": bias, "ticks_left": LEAN_DECAY_TICKS,
                }
            # Unconditional leans (no quartile / jump gating)
            if buyer and (sym, "buy", buyer) in UNCONDITIONAL_LEAN_SIGNALS:
                bias = UNCONDITIONAL_LEAN_SIGNALS[(sym, "buy", buyer)]
                persistent_leans[(sym, "uncond_buy", buyer)] = {
                    "bias": bias, "ticks_left": LEAN_DECAY_TICKS,
                }
            if seller and (sym, "sell", seller) in UNCONDITIONAL_LEAN_SIGNALS:
                bias = UNCONDITIONAL_LEAN_SIGNALS[(sym, "sell", seller)]
                persistent_leans[(sym, "uncond_sell", seller)] = {
                    "bias": bias, "ticks_left": LEAN_DECAY_TICKS,
                }


def aggregate_lean(persistent_leans, symbol):
    """Sum all active leans for a symbol, clamped to {-1, 0, +1}."""
    total = 0
    for (sym, *_), info in persistent_leans.items():
        if sym == symbol:
            total += info["bias"]
    if total > 0:
        return 1
    if total < 0:
        return -1
    return 0


# ── Trader ───────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):
        result: Dict[Symbol, List[Order]] = {}
        conversions = 0

        # Decode persistent state
        try:
            tstate = jsonpickle.decode(state.traderData) if state.traderData else {}
            if not isinstance(tstate, dict):
                tstate = {}
        except Exception:
            tstate = {}

        prev_mids = tstate.setdefault("prev_mids", {})
        persistent_leans = tstate.setdefault("leans", {})
        warmup_state = tstate.setdefault("warmup", {})

        # Surface any planted signals.
        try:
            pvo = getattr(state.observations, "plainValueObservations", None)
            if pvo:
                logger.print("PLANTED_SIGNAL " + str(pvo))
        except Exception:
            pass

        # Process counterparty signals first (mutates persistent_leans)
        detect_counterparty_signals(state, prev_mids, persistent_leans)

        # ── Underlying mids ──────────────────────────────────────────
        ve_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        ve_mid = midprice(ve_depth) if ve_depth else None

        # VE fair value: freeze-after-warmup anchor.
        # During the first warmup_window ticks, accumulate mid prices
        # into a buffer and use the static anchor. Once full, freeze
        # the buffer mean as the anchor for the rest of the day. This
        # adapts to per-day regime (D2 mean 5255 vs D1/D3 ~5244) without
        # the V-shape-chasing failure mode of rolling/EWMA FVs.
        ve_cfg = PRODUCT_CONFIG["VELVETFRUIT_EXTRACT"]
        ve_warmup = warmup_state.setdefault(
            "VELVETFRUIT_EXTRACT", {"sum": 0.0, "n": 0, "frozen": None}
        )
        if ve_mid is not None and ve_warmup["frozen"] is None:
            ve_warmup["sum"] += ve_mid
            ve_warmup["n"] += 1
            if ve_warmup["n"] >= ve_cfg.get("warmup_window", 3000):
                ve_warmup["frozen"] = ve_warmup["sum"] / ve_warmup["n"]
        ve_fv = (
            ve_warmup["frozen"]
            if ve_warmup["frozen"] is not None
            else ve_cfg["anchor"]
        )

        # ── Smile fits (leave-one-out per quoted smile_mm strike) ────
        # Computed once per tick; each smile_mm branch reads its own LOO
        # fit. Per-strike LOO is needed because including the strike's
        # own market IV in the fit pulls smile_iv at that K toward
        # market IV, collapsing the fade signal. Cheap: each fit is a
        # 5-point quadratic.
        smile_T = compute_tte() if ve_mid is not None else None
        loo_smile_coeffs: Dict[str, Any] = {}
        if ve_mid is not None and smile_T is not None:
            for sym in SMILE_QUOTE_STRIKES:
                coeffs, _ = fit_smile(state, ve_mid, smile_T, exclude=sym)
                if coeffs is not None:
                    loo_smile_coeffs[sym] = coeffs

        # ── Per-product trading ──────────────────────────────────────
        for product, depth in state.order_depths.items():
            cfg = PRODUCT_CONFIG.get(product)
            if cfg is None:
                logger.print("UNKNOWN_PRODUCT " + str(product))
                continue
            position = state.position.get(product, 0)
            limit = cfg["limit"]

            if product == "HYDROGEL_PACK":
                # Regime-conditional anchor: track max(mid) during the
                # warmup window; freeze it at warmup end. After freeze,
                # if frozen_max > regime_max_threshold, use the lower
                # regime anchor (high-range day); else use the modal
                # attractor anchor.
                hyd_mid = midprice(depth)
                hyd_warmup = warmup_state.setdefault(
                    "HYDROGEL_PACK", {"max": None, "n": 0, "frozen_max": None}
                )
                if hyd_mid is not None and hyd_warmup["frozen_max"] is None:
                    hyd_warmup["max"] = (
                        hyd_mid if hyd_warmup["max"] is None
                        else max(hyd_warmup["max"], hyd_mid)
                    )
                    hyd_warmup["n"] += 1
                    if hyd_warmup["n"] >= cfg.get("warmup_window", 4000):
                        hyd_warmup["frozen_max"] = hyd_warmup["max"]
                fair = cfg["anchor"]
                if hyd_warmup["frozen_max"] is not None:
                    fm = hyd_warmup["frozen_max"]
                    low_edge = cfg.get("regime_band_low")
                    high_edge = cfg.get("regime_band_high")
                    if low_edge is not None and high_edge is not None and high_edge > low_edge:
                        if fm >= high_edge:
                            fair = cfg["regime_anchor_low"]
                        elif fm > low_edge:
                            t = (fm - low_edge) / (high_edge - low_edge)
                            fair = cfg["anchor"] + t * (cfg["regime_anchor_low"] - cfg["anchor"])
                    elif fm > cfg["regime_max_threshold"]:
                        fair = cfg["regime_anchor_low"]
                cp_lean = aggregate_lean(persistent_leans, product)
                lean = cp_lean
                # Anchor-offset expression: quote_skew=±1 is bandwidth-saturated
                # by HYDROGEL's touch clamp (16-tick spread, default_edge=4).
                # Shift `fair` itself by ±1 on lean-active ticks so the signal
                # rides through take_orders + make_orders midpoint instead of
                # being absorbed at the maker quote. join_edge=10 means a
                # 1-tick anchor shift moves the maker quotes cleanly.
                fair = fair + cp_lean
                # L3 take-relaxation: when L3 has only an ask (predictive of
                # mid +4 over next 1000ts, t>7.8 all 3 days), use a looser
                # take threshold than MTE=7. Mirror for L3-bid-only.
                bid_prices = sorted(depth.buy_orders.keys(), reverse=True) if depth.buy_orders else []
                ask_prices = sorted(depth.sell_orders.keys()) if depth.sell_orders else []
                has_l3_bid = len(bid_prices) >= 3
                has_l3_ask = len(ask_prices) >= 3
                base_mte = cfg["min_take_edge"]
                l3_mte = cfg.get("l3_min_take_edge", base_mte)
                if has_l3_ask and not has_l3_bid:
                    taken, bv, sv = take_orders(
                        depth, product, fair, position, limit,
                        min_take_edge=l3_mte, side_filter="buy_only",
                        full_size_edge=cfg.get("full_size_edge"),
                    )
                    taken2, bv2, sv2 = take_orders(
                        depth, product, fair, position + bv, limit,
                        min_take_edge=base_mte, side_filter="sell_only",
                        full_size_edge=cfg.get("full_size_edge"),
                    )
                    taken += taken2; bv += bv2; sv += sv2
                elif has_l3_bid and not has_l3_ask:
                    taken, bv, sv = take_orders(
                        depth, product, fair, position, limit,
                        min_take_edge=l3_mte, side_filter="sell_only",
                        full_size_edge=cfg.get("full_size_edge"),
                    )
                    taken2, bv2, sv2 = take_orders(
                        depth, product, fair, position - sv, limit,
                        min_take_edge=base_mte, side_filter="buy_only",
                        full_size_edge=cfg.get("full_size_edge"),
                    )
                    taken += taken2; bv += bv2; sv += sv2
                else:
                    taken, bv, sv = take_orders(
                        depth, product, fair, position, limit,
                        min_take_edge=base_mte,
                        full_size_edge=cfg.get("full_size_edge"),
                    )
                made = make_orders(
                    depth, product, fair, position, limit, bv, sv,
                    default_edge=cfg["default_edge"],
                    join_edge=cfg["join_edge"],
                    disregard_edge=cfg["disregard_edge"],
                    soft_position_limit=cfg["soft_position_limit"],
                    quote_skew=lean,
                    per_quote_cap=cfg.get("per_quote_cap"),
                )
                result[product] = taken + made

            elif product == "VELVETFRUIT_EXTRACT":
                fair = ve_fv
                # cumOFI mean-reversion (R3 helper) tested -- net -383
                # XIRECS over 3 days vs counterparty-only. Drop.
                cp_lean = aggregate_lean(persistent_leans, product)
                lean = cp_lean
                taken, bv, sv = take_orders(
                    depth, product, fair, position, limit,
                    min_take_edge=cfg["min_take_edge"],
                )
                made = make_orders(
                    depth, product, fair, position, limit, bv, sv,
                    default_edge=cfg["default_edge"],
                    join_edge=cfg["join_edge"],
                    disregard_edge=cfg["disregard_edge"],
                    soft_position_limit=cfg["soft_position_limit"],
                    quote_skew=lean,
                    per_quote_cap=cfg.get("per_quote_cap"),
                )
                result[product] = taken + made

            elif cfg.get("role") == "delta1_proxy":
                # VEV_4000: voucher mid = VE mid - K (verified delta-1).
                # Quote around VE-implied fair, take if >= min_take_edge.
                if ve_mid is None:
                    continue
                fair = ve_mid - cfg["strike"]
                cp_lean = aggregate_lean(persistent_leans, product)
                taken, bv, sv = take_orders(
                    depth, product, fair, position, limit,
                    min_take_edge=cfg["min_take_edge"],
                )
                made = make_orders(
                    depth, product, fair, position, limit, bv, sv,
                    default_edge=cfg["default_edge"],
                    join_edge=cfg["join_edge"],
                    disregard_edge=0,
                    soft_position_limit=cfg["soft_position_limit"],
                    quote_skew=cp_lean,
                )
                result[product] = taken + made

            elif cfg.get("role") == "dev_directional":
                # VE-deviation directional bet on option mid. The signal:
                # corr(ve_mid - anchor, fwd-1k-tick option mid drift)
                # is -0.5 to -0.8 on every (strike, day) tested across R4
                # D1/D2/D3 and R3 D0 (held-out). VE mean-reverts to its
                # anchor; option mid follows via delta+gamma+vega.
                # Sell when VE is above anchor, buy when below; cycle out
                # when dev crosses back through zero against current pos.
                # Bayesian-seeded anchor (BAYES_MU_PRIOR, BAYES_N_PRIOR)
                # replaces the prior 4000-tick warmup gate so VEV_5xxx
                # trade from tick 0; held-out R3 D0 +42k vs old version.
                if ve_mid is None:
                    continue
                ve_warmup_local = warmup_state.get("VELVETFRUIT_EXTRACT", {})
                n_live = ve_warmup_local.get("n", 0)
                s_live = ve_warmup_local.get("sum", 0.0)
                bayes_anchor = (
                    BAYES_N_PRIOR * BAYES_MU_PRIOR + s_live
                ) / (BAYES_N_PRIOR + n_live)
                dev = ve_mid - bayes_anchor
                bb_d, ba_d = best_bid_ask(depth)
                if bb_d is None or ba_d is None:
                    continue
                max_pos_local = cfg["max_position"]
                thr_local = cfg["dev_threshold"]
                fill_local = cfg.get("fill_size", 20)
                orders_d: List[Order] = []
                if dev > thr_local and position > -max_pos_local:
                    qty = min(depth.buy_orders.get(bb_d, 0),
                              max_pos_local + position, fill_local)
                    if qty > 0:
                        orders_d.append(Order(product, bb_d, -qty))
                elif dev < -thr_local and position < max_pos_local:
                    qty = min(abs(depth.sell_orders.get(ba_d, 0)),
                              max_pos_local - position, fill_local)
                    if qty > 0:
                        orders_d.append(Order(product, ba_d, qty))
                elif position > 0 and dev > 0:
                    # Long but VE has crossed above anchor — flatten to neutral.
                    qty = min(depth.buy_orders.get(bb_d, 0),
                              position, fill_local)
                    if qty > 0:
                        orders_d.append(Order(product, bb_d, -qty))
                elif position < 0 and dev < 0:
                    qty = min(abs(depth.sell_orders.get(ba_d, 0)),
                              -position, fill_local)
                    if qty > 0:
                        orders_d.append(Order(product, ba_d, qty))
                result[product] = orders_d

            elif cfg.get("role") == "smile_mm":
                # Fair = BS price at LOO-smile IV. Trades against the
                # chronic residual: when market IV deviates from the
                # cross-strike smile (rich -> our ask sits below market,
                # we get hit and short; cheap -> we lift). Hard cap on
                # max_position bounds the regime-drift loss seen on
                # VEV_5400 D3 (-16k unhedged smile-MM).
                if ve_mid is None:
                    continue
                coeffs = loo_smile_coeffs.get(product)
                if coeffs is None:
                    continue
                K = cfg["strike"]
                iv_fair = smile_iv(coeffs, ve_mid, K)
                if iv_fair <= 0.0:
                    continue
                fair = bs_call_price(ve_mid, K, smile_T, iv_fair)
                max_pos = cfg["max_position"]
                eff_limit = min(limit, max_pos)
                side_filter = None
                if position >= max_pos:
                    side_filter = "sell_only"
                elif position <= -max_pos:
                    side_filter = "buy_only"
                taken, bv, sv = take_orders(
                    depth, product, fair, position, eff_limit,
                    min_take_edge=cfg["min_take_edge"],
                    side_filter=side_filter,
                )
                made = make_orders(
                    depth, product, fair, position, eff_limit, bv, sv,
                    default_edge=cfg["default_edge"],
                    join_edge=cfg["join_edge"],
                    disregard_edge=0,
                    soft_position_limit=cfg["soft_position_limit"],
                    side_filter=side_filter,
                )
                result[product] = taken + made

            elif cfg.get("role") == "wing_lottery":
                # VEV_6000 / VEV_6500: book pinned bid=0/ask=1; flow is
                # bot-vs-bot at price 0. We post a $0 bid for free OTM
                # optionality (bounded loss). Never sell.
                bb, _ = best_bid_ask(depth)
                buy_qty = limit - position
                if buy_qty > 0 and bb is not None and bb >= 0:
                    result[product] = [Order(product, 0, buy_qty)]

            # role == "skip": no orders.

        # Persist mids for next-tick jump detection.
        new_prev = {}
        for sym, depth in state.order_depths.items():
            m = midprice(depth)
            if m is not None:
                new_prev[sym] = m
        tstate["prev_mids"] = new_prev
        tstate["leans"] = persistent_leans
        tstate["warmup"] = warmup_state

        trader_data = jsonpickle.encode(tstate)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
