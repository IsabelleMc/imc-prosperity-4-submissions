from datamodel import OrderDepth, TradingState, Order, Listing, Observation, ProsperityEncoder, Symbol, Trade
from typing import Any, List, Dict
import jsonpickle
import json
import statistics


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


# ── Mean Reversion Algorithm ─────────────────────────────────────────

class MeanReversionAlgo(BaseAlgo):
    """Trades toward a known anchor price. Good for stable products."""

    def __init__(self, anchor, alpha=0.15, band=1.0, wide_spread_clamp=None):
        self.anchor = anchor
        self.alpha = alpha
        self.band = band
        # Exp #38 (2026-04-20): on wide-spread ticks (spread > 10), the
        # raw mid is systematically biased away from anchor because
        # the thin-side ask/bid is far from true value. Clamping mid
        # to anchor ± `wide_spread_clamp` on wide books corrects the
        # EMA drift that otherwise refuses profitable fills. MC N=100
        # paired: clamp=7 gives +147 Δμ vs shipped, 100/100 wins,
        # worst-case STILL +77 (never loses). Tighter clamps (5/6)
        # over-correct and get stuck at the position limit.
        self.wide_spread_clamp = wide_spread_clamp

    def default_state(self):
        return {"ema": None, "last_mid": None}

    def compute_fair_value(self, order_depth, market_trades, state_data):
        mid = self._midprice(order_depth)
        # One-sided book fallback: reuse last known mid so we can still
        # quote. 97% of dead ticks recover ±5 within 5 ticks, so the
        # cached mid is a good proxy.
        if mid is None:
            mid = state_data.get("last_mid")
        if mid is None:
            return None, state_data
        state_data["last_mid"] = mid

        prev_ema = state_data.get("ema")
        ema = mid if prev_ema is None else (
            self.alpha * mid + (1 - self.alpha) * prev_ema
        )
        state_data["ema"] = ema

        fair_value = self.anchor + self.alpha * (ema - self.anchor)
        return fair_value, state_data

    def _midprice(self, order_depth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        raw = (best_bid + best_ask) / 2.0
        if self.wide_spread_clamp is not None and best_ask - best_bid > 10:
            cap = self.wide_spread_clamp
            return max(self.anchor - cap, min(self.anchor + cap, raw))
        return raw


# ── Rolling Anchor Mean Reversion ───────────────────────────────────

class RollingAnchorMeanReversionAlgo(BaseAlgo):
    """Mean reversion to a DRIFTING rolling anchor rather than a fixed one.

    Data finding (R1 osmium D-2/-1/0, measured across 3 days):
    - A W=500-tick SMA of mid wanders between 9990 and 10010 across a
      session (13-20 tick band).
    - `corr(mid - rolling_mean_W500, dmid_{t+1}) = -0.25` consistently
      across all 3 days. Deviations from the ROLLING anchor mean-revert
      more reliably than deviations from the static 10,000 anchor.

    Strategy: track a rolling SMA of the most recent `window` mid prices
    (kept in state via a ring buffer). Fair value is an EMA-blended pull
    toward that rolling anchor, same structure as MeanReversionAlgo but
    with the anchor itself adaptive.

    Falls back to the fixed-anchor formula during warmup (before the
    buffer is full) so early-session behavior is identical to
    MeanReversionAlgo.
    """

    def __init__(self, anchor=10000, alpha=0.50, window=500):
        self.anchor = anchor  # fallback / warmup target
        self.alpha = alpha
        self.window = window

    def default_state(self):
        return {"ema": None, "last_mid": None, "mid_buffer": []}

    def compute_fair_value(self, order_depth, market_trades, state_data):
        mid = self._midprice(order_depth)
        if mid is None:
            mid = state_data.get("last_mid")
        if mid is None:
            return None, state_data
        state_data["last_mid"] = mid

        # Migrate older state dicts without the buffer
        buf = state_data.get("mid_buffer")
        if buf is None:
            buf = []
            state_data["mid_buffer"] = buf

        buf.append(mid)
        if len(buf) > self.window:
            # ring-buffer truncation: keep last `window` mids
            del buf[0:len(buf) - self.window]

        # Rolling anchor = SMA of buffer. During warmup (< window mids)
        # fall back to the fixed anchor so the algo behaves identically
        # to MeanReversionAlgo during the first session ticks.
        if len(buf) >= self.window:
            rolling_anchor = sum(buf) / len(buf)
        else:
            rolling_anchor = self.anchor

        prev_ema = state_data.get("ema")
        ema = mid if prev_ema is None else (
            self.alpha * mid + (1 - self.alpha) * prev_ema
        )
        state_data["ema"] = ema

        fair_value = rolling_anchor + self.alpha * (ema - rolling_anchor)
        return fair_value, state_data

    def _midprice(self, order_depth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0


# ── Adaptive Anchor Mean Reversion ──────────────────────────────────

class AdaptiveAnchorMeanReversionAlgo(BaseAlgo):
    """Mean reversion that learns its anchor from early session data.

    Rationale: Round 1's MeanReversionAlgo hardcodes anchor=10000 because
    we measured osmium's center to be exactly 10000. For Round 2 products
    we won't know the center in advance. This variant observes the first
    `warmup_ticks` mids, takes the median, and locks it as the anchor.
    From tick `warmup_ticks+1` onward, behavior is identical to
    MeanReversionAlgo.

    During warmup, FV = current EMA (no anchor pull) so we MM around the
    observed price without assuming a center. After warmup, anchor is
    frozen and used for the pull-toward-anchor formula.

    Use for Round 2 stable-mean-reverting products where you don't trust
    a hardcoded anchor. For products with known centers, prefer the plain
    MeanReversionAlgo (simpler, no warmup delay).
    """

    def __init__(self, alpha=0.40, warmup_ticks=200):
        self.alpha = alpha
        self.warmup_ticks = warmup_ticks

    def default_state(self):
        return {"ema": None, "warmup_mids": [], "anchor": None}

    def compute_fair_value(self, order_depth, market_trades, state_data):
        mid = self._midprice(order_depth)
        if mid is None:
            return None, state_data

        # Migrate older state dicts without the new keys
        if "warmup_mids" not in state_data:
            state_data["warmup_mids"] = []
        if "anchor" not in state_data:
            state_data["anchor"] = None

        # Update EMA every tick
        prev_ema = state_data.get("ema")
        ema = mid if prev_ema is None else (
            self.alpha * mid + (1 - self.alpha) * prev_ema
        )
        state_data["ema"] = ema

        # Warmup: accumulate mids, FV = EMA (no anchor pull)
        if state_data["anchor"] is None:
            state_data["warmup_mids"].append(mid)
            if len(state_data["warmup_mids"]) >= self.warmup_ticks:
                anchor = statistics.median(state_data["warmup_mids"])
                state_data["anchor"] = anchor
                state_data["warmup_mids"] = []  # free memory
                fair_value = anchor + self.alpha * (ema - anchor)
                return fair_value, state_data
            return ema, state_data

        # Post-warmup: same formula as MeanReversionAlgo
        anchor = state_data["anchor"]
        fair_value = anchor + self.alpha * (ema - anchor)
        return fair_value, state_data

    def _midprice(self, order_depth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0


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


# ── Double-EMA Mean Reversion (experiment #26) ───────────────────────
class DoubleEMAMeanReversionAlgo(BaseAlgo):
    """Two-timescale mean-reversion FV.

    slow_ema: tracks long-term mid with alpha_slow. Represents the
        adaptive anchor — if the product's center drifts across days,
        the slow EMA adapts to it instead of relying on a hardcoded
        anchor constant.
    fast_ema: tracks short-term mid with alpha_fast. Smoother than
        raw mid (de-noised) but still responsive to recent moves.

    fair_value = slow_ema + blend * (fast_ema - slow_ema)

    With blend=0.40, this collapses to the original single-alpha
    formulation when alpha_fast=1.0 and alpha_slow→0 with a hardcoded
    anchor. The two-alpha structure gives two wins: (1) FV is less
    noisy at the fast-update level, (2) the anchor is adaptive.

    anchor param: initial value for slow_ema seeding only; otherwise
    unused (unlike MeanReversionAlgo which hardcodes it as the pull
    center forever).
    """

    def __init__(self, anchor, alpha_fast=0.60, alpha_slow=0.02, blend=0.40):
        self.anchor = anchor
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.blend = blend

    def default_state(self):
        return {"fast": None, "slow": None}

    def compute_fair_value(self, order_depth, market_trades, state_data):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None, state_data
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0

        prev_fast = state_data.get("fast")
        prev_slow = state_data.get("slow")
        fast = mid if prev_fast is None else (
            self.alpha_fast * mid + (1 - self.alpha_fast) * prev_fast
        )
        slow = (self.anchor if prev_slow is None else
                self.alpha_slow * mid + (1 - self.alpha_slow) * prev_slow)
        state_data["fast"] = fast
        state_data["slow"] = slow

        fair_value = slow + self.blend * (fast - slow)
        return fair_value, state_data


# ── Default (Conservative) Algorithm ─────────────────────────────────

class DefaultAlgo(EMAFairValueAlgo):
    """Conservative fallback for unknown products. Wider spread."""

    def __init__(self):
        super().__init__(alpha=0.2)
        self.IMBALANCE_WEIGHT = 1.0


# ── Product Configuration ────────────────────────────────────────────

MIDPRICE_BUFFER_SIZE = 50  # how many recent mids to keep in state

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


def eod_flatten_orders(
    order_depth, product, position, timestamp, limit,
    buy_volume=0, sell_volume=0,
    start_ts=990000, size_per_tick=10, max_adverse=5, fair_value=None,
):
    """End-of-day position flattening.

    Measured (R1+R2 6-day sweep): we systematically end with an adverse
    position (long when final mid dipped, short when final mid spiked)
    costing avg 390 XIRECs/day from MTM wobble at the final tick.

    Starting at `start_ts`, aggressively cross the spread to unwind
    position at `size_per_tick` per tick. Caps at `max_adverse` ticks
    away from fair_value to avoid paying huge adverse premiums in
    thin final-tick books.

    Returns (orders, new_buy_volume, new_sell_volume).
    """
    orders = []
    if timestamp < start_ts:
        return orders, buy_volume, sell_volume
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return orders, buy_volume, sell_volume

    net_pos = position + buy_volume - sell_volume
    if net_pos == 0:
        return orders, buy_volume, sell_volume

    if net_pos > 0:
        # Long — sell into the bid to flatten
        best_bid = max(order_depth.buy_orders.keys())
        bid_vol = order_depth.buy_orders[best_bid]
        if fair_value is not None and best_bid < fair_value - max_adverse:
            return orders, buy_volume, sell_volume
        raw_cap = limit + (position - sell_volume)
        qty = min(size_per_tick, bid_vol, raw_cap, net_pos)
        if qty > 0:
            orders.append(Order(product, best_bid, -qty))
            sell_volume += qty
    else:
        # Short — buy the ask to flatten
        best_ask = min(order_depth.sell_orders.keys())
        ask_vol = abs(order_depth.sell_orders[best_ask])
        if fair_value is not None and best_ask > fair_value + max_adverse:
            return orders, buy_volume, sell_volume
        raw_cap = limit - (position + buy_volume)
        qty = min(size_per_tick, ask_vol, raw_cap, -net_pos)
        if qty > 0:
            orders.append(Order(product, best_ask, qty))
            buy_volume += qty

    return orders, buy_volume, sell_volume


def hard_unwind_orders(
    order_depth, product, position, limit,
    buy_volume=0, sell_volume=0,
    trigger=60, size=10, max_adverse=3, fair_value=None,
):
    """Aggressive unwind take when position is deeply saturated.

    Rationale (measured on union book, R2 d0):
    - Saturation events at ±80 average 23 ticks long (median 17)
    - 36 of 50 events are short-side (bot sells into rallies, then
      can't unwind fast on revert). Total: 1,172 saturated ticks/day
      where bot is stuck at limit and can't take new opportunities.
    - Passive MM at penny-inside is too slow to flatten. Crossing
      the spread aggressively costs a small adverse edge but frees
      capacity for the NEXT tail crossing.

    Semantics:
    - When |position + buy_volume - sell_volume| >= `trigger`, aggress
      in the flattening direction.
    - Cap aggression by `size` per tick (don't fully unwind in one hit).
    - Only cross up to `max_adverse` ticks away from fair_value
      (if fair_value given). Prevents paying huge adverse premiums.

    Returns (orders, new_buy_volume, new_sell_volume).
    """
    orders = []
    net_pos = position + buy_volume - sell_volume
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return orders, buy_volume, sell_volume

    if net_pos >= trigger:
        # Too long — aggressively sell into the bid
        best_bid = max(order_depth.buy_orders.keys())
        bid_vol = order_depth.buy_orders[best_bid]
        if fair_value is not None and best_bid < fair_value - max_adverse:
            return orders, buy_volume, sell_volume  # too adverse, skip
        raw_cap = limit + (position - sell_volume)
        qty = min(size, bid_vol, raw_cap, net_pos - trigger + size)
        if qty > 0:
            orders.append(Order(product, best_bid, -qty))
            sell_volume += qty
    elif net_pos <= -trigger:
        # Too short — aggressively buy the ask
        best_ask = min(order_depth.sell_orders.keys())
        ask_vol = abs(order_depth.sell_orders[best_ask])
        if fair_value is not None and best_ask > fair_value + max_adverse:
            return orders, buy_volume, sell_volume
        raw_cap = limit - (position + buy_volume)
        qty = min(size, ask_vol, raw_cap, -net_pos - trigger + size)
        if qty > 0:
            orders.append(Order(product, best_ask, qty))
            buy_volume += qty

    return orders, buy_volume, sell_volume


def obi_take_orders(
    order_depth, product, position, limit,
    buy_volume=0, sell_volume=0,
    threshold=0.3, aggress_size=10, position_aware=False,
):
    """Aggressive take based on L1 order-book imbalance signal.

    Measured (R1 D-2/-1/0, >8k samples/day, replicated to 2 decimals):
    - OBI > +0.3 → mid rises +3.5 ticks on average over the next 50 ticks
    - OBI < -0.3 → mid falls -3.5 ticks on average
    - ~22% of ticks trigger (|OBI| > 0.3)
    - Signal is sustained across horizons 1/2/5/10/20/50 (not a bounce)

    Strategy: when the signal is strong, lift the best ask (for OBI>0)
    or hit the best bid (for OBI<0) in a small size. The aggressive
    take captures the expected directional move.

    aggress_size caps the signal-based take per tick. We don't consume
    all remaining capacity because:
    1. The passive MM still wants headroom to post quotes
    2. The signal predicts +3.5 mean; 1-sigma dispersion is larger, so
       sizing down limits adverse-selection exposure
    3. The signal triggers ~22% of ticks; over-sizing accumulates
       directional position faster than we can unwind

    position_aware (D15, structural_hunt 2026-04-18): when True, fire
    only when the fade trade also flattens our position. Rationale:
    deep_track_A_obi_sandbox showed |OBI|>=0.7 events disappear when
    our own quote is subtracted from book volume — i.e. extreme-OBI
    ticks in live are partially caused by our own passive quote on the
    matching-imbalance side. Conditioning on `position × obi_sign > 0`
    restricts takes to moments when fading is also flattening, which
    (a) reduces over-exposure to noise signal, (b) accelerates unwind
    at saturation, (c) sidesteps the -295k/3d unconditional-take failure
    mode by refusing trades that would push us further adverse.

    Returns (orders, new_buy_volume, new_sell_volume) — counters updated
    in place so subsequent MM sizing sees the consumed capacity.
    """
    orders = []
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return orders, buy_volume, sell_volume
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    bid_vol = order_depth.buy_orders[best_bid]
    ask_vol = abs(order_depth.sell_orders[best_ask])
    denom = bid_vol + ask_vol
    if denom == 0:
        return orders, buy_volume, sell_volume
    obi = (bid_vol - ask_vol) / denom

    # NOTE ON DIRECTION: measured backtest confirmed the signal's
    # correlation with future dmid is +0.6, but for a MEAN-REVERTING
    # product around a known anchor, following the signal with
    # aggressive takes LOSES money (-295k/3 days tested). The
    # explanation: |OBI|>0.3 clusters at price extremes (bids stack up
    # BELOW anchor, asks stack up ABOVE), and aggressing-with-signal
    # means buying at the top / selling at the bottom just before the
    # anchor-mean-reversion. We FADE instead: aggress AGAINST the
    # signal to capture the mean-reversion leg.
    if obi > threshold:
        # Signal: mid at local high → fade by SELLING into the imbalance.
        # position_aware: require position > 0 (selling flattens long).
        if position_aware and position <= 0:
            return orders, buy_volume, sell_volume
        raw_cap = limit + (position - sell_volume)
        qty = min(aggress_size, bid_vol, raw_cap)
        if qty > 0:
            orders.append(Order(product, best_bid, -qty))
            sell_volume += qty
    elif obi < -threshold:
        # Signal: mid at local low → fade by BUYING into the imbalance.
        # position_aware: require position < 0 (buying flattens short).
        if position_aware and position >= 0:
            return orders, buy_volume, sell_volume
        raw_cap = limit - (position + buy_volume)
        qty = min(aggress_size, ask_vol, raw_cap)
        if qty > 0:
            orders.append(Order(product, best_ask, qty))
            buy_volume += qty

    return orders, buy_volume, sell_volume


def compute_obi_trend(order_depth, threshold=0.3):
    """L1 order-book imbalance as a directional signal.

    OBI = (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol).

    Measured on osmium across R1 days -2/-1/0 (>8000 samples each):
    - corr(OBI_t, dmid_{t+1}) = +0.59 (replicated to 2 decimals across
      all 3 days)
    - OBI > +0.3 (~11% of ticks): mid goes UP 89% of next ticks, mean
      dmid +2.0 ticks
    - OBI < -0.3 (~11% of ticks): mid goes DOWN 89% of next ticks, mean
      dmid -2.0 ticks

    Returns +1 (predict mid will rise — suppress sells), -1 (predict
    mid will fall — suppress buys), or 0 (flat). Convention matches
    compute_slope_trend: trend means "direction mid is heading" so the
    MM can bias quotes against adverse selection.
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


# ── Avellaneda-Stoikov order generator ─────────────────────────────


# ── Directional accumulator (buy-and-hold) ──────────────────────────
#
# For products with strong predictable drift (like pepper +1000/day),
# market making misses the point. Accumulate a long to the limit, hold
# through the drift, let EOD mark-to-market realize the PnL.


def directional_long_orders(
    order_depth, product, position, limit,
    target_position=None, max_per_tick=None,
    cheapest_only=False,
):
    """Generate orders for a directional buy-and-hold strategy.

    Buys up to target_position, paced by max_per_tick to avoid walking
    the ask ladder. Cross asks first (cheapest levels), then post a
    resting penny-ahead bid for any remaining capacity.

    cheapest_only: if True, only lift the single cheapest ask level per
    tick. Don't cascade to the next-cheapest level if max_per_tick isn't
    reached. This avoids paying 2-3 ticks extra on days where the cheapest
    level has less volume than max_per_tick — the refinement caught by
    investigating early-tick ask distributions.
    """
    if target_position is None:
        target_position = limit - 2

    orders = []
    buy_cap = target_position - position
    if buy_cap <= 0:
        return orders
    if max_per_tick is not None:
        buy_cap = min(buy_cap, max_per_tick)

    # 1. Lift asks. If cheapest_only, stop after the first (cheapest) level.
    remaining = buy_cap
    for ask_price, ask_qty in sorted(order_depth.sell_orders.items()):
        if remaining <= 0:
            break
        qty = min(-ask_qty, remaining)
        if qty > 0:
            orders.append(Order(product, ask_price, qty))
            remaining -= qty
        if cheapest_only:
            break

    # 2. Also post a resting bid for any remaining capacity, penny-ahead.
    # Only when NOT in cheapest_only mode — otherwise the resting bid
    # competes with our "wait for cheap asks" policy.
    if not cheapest_only and remaining > 0 and order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders.keys())
        orders.append(Order(product, int(best_bid + 1), remaining))

    return orders


def directional_long_dip_catch_orders(
    order_depth, product, position, limit,
    accumulation_target=72, max_per_tick=15, dip_catch_size=8,
):
    """Experiment #33: buy-and-hold to accumulation_target, then
    opportunistically catch dips with a resting bid up to `limit`.

    Phases:
      - position < accumulation_target: same as directional_long_orders
        (cross cheapest asks, then resting bid at best_bid+1 for
        remainder, paced by max_per_tick).
      - accumulation_target <= position < limit: post a single resting
        bid at best_bid+1 for min(dip_catch_size, limit - position).
        Never cross the ask — wait for someone to hit us.
      - position >= limit: emit nothing.

    Never sells. The reserved slots (limit - accumulation_target)
    earn nothing until someone walks an ask down to best_bid+1
    during the day.
    """
    orders = []
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return orders

    # Phase 1: primary accumulation, same as directional_long.
    if position < accumulation_target:
        buy_cap = accumulation_target - position
        if max_per_tick is not None:
            buy_cap = min(buy_cap, max_per_tick)

        remaining = buy_cap
        for ask_price, ask_qty in sorted(order_depth.sell_orders.items()):
            if remaining <= 0:
                break
            qty = min(-ask_qty, remaining)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                remaining -= qty

        if remaining > 0:
            best_bid = max(order_depth.buy_orders.keys())
            orders.append(Order(product, int(best_bid + 1), remaining))
        return orders

    # Phase 2: dip-catch. Post passive bid for reserved slots.
    if position < limit:
        remaining = min(dip_catch_size, limit - position)
        if remaining > 0:
            best_bid = max(order_depth.buy_orders.keys())
            orders.append(Order(product, int(best_bid + 1), remaining))
        return orders

    # Phase 3: full, do nothing.
    return orders


def directional_long_with_mm_orders(
    order_depth, product, position, limit,
    state_data,
    target_position=None, max_per_tick=None, cheapest_only=False,
    mm_alpha=0.10, mm_edge=5, mm_size=5,
    mm_min_position_for_sell=20,
    mm_max_position_for_buy=80,
):
    """Directional-long accumulation + 2-sided MM around tracked mid.

    Exp #32 (re-ship of Exp #15 code). Prior rejection was -73 XIRECs
    on N=5 × 1k-tick sandbox (inside 86 σ). A competitor running this
    exact config scored 9,524 on a full 100k-tick day vs our 9,360
    ceiling, so the rejection signal was likely seed variance, not a
    real negative edge. mm_edge=5 targets Exp #14's measured +4 to +6
    bot-bot inside-spread flow (~9,338 units/day).
    """
    orders = []
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return orders, state_data

    if target_position is None:
        target_position = limit - 2

    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    mid = (best_bid + best_ask) / 2.0
    prev_ema = state_data.get("mid_ema")
    mid_ema = mid if prev_ema is None else (
        mm_alpha * mid + (1 - mm_alpha) * prev_ema
    )
    state_data["mid_ema"] = mid_ema

    accum_buys = 0
    if position < target_position:
        buy_cap = target_position - position
        if max_per_tick is not None:
            buy_cap = min(buy_cap, max_per_tick)
        remaining = buy_cap
        for ask_price, ask_qty in sorted(order_depth.sell_orders.items()):
            if remaining <= 0:
                break
            qty = min(-ask_qty, remaining)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                remaining -= qty
                accum_buys += qty
            if cheapest_only:
                break

    mm_bid_price = int(round(mid_ema - mm_edge))
    if (mm_bid_price < best_bid + 1
            and position + accum_buys + mm_size <= mm_max_position_for_buy
            and mm_bid_price > 0):
        orders.append(Order(product, mm_bid_price, mm_size))

    if position > mm_min_position_for_sell:
        mm_ask_price = int(round(mid_ema + mm_edge))
        if mm_ask_price > best_ask - 1:
            inv_room = position - mm_min_position_for_sell
            sell_qty = min(mm_size, inv_room)
            if sell_qty > 0:
                orders.append(Order(product, mm_ask_price, -sell_qty))

    return orders, state_data


PRODUCT_CONFIG = {
    "ASH_COATED_OSMIUM": {
        # alpha=0.50: 120-combo sweep (reserve × alpha × threshold) found
        # alpha=0.5 dominant across top 8 — structural signal that a
        # faster EMA tracks osmium mid better than the prior 0.40.
        # Alone worth ~+800 XIRECs / 3 days; paired with tighter reserve
        # worth +1,360 vs prior (3, 0.33, 0.40) baseline.
        "algo": MeanReversionAlgo(anchor=10000, alpha=0.50, wide_spread_clamp=7),
        "limit": 80,
        "make": True,
        "trend_filter": None,
        "take_min_edge": 0,
        # Reserve-capacity ladder: crossings with edge < 5 ticks from
        # fair only consume 20% of remaining capacity. Sweep across
        # threshold {2,3,4,5} × fraction {0.2,0.25,0.33,0.4,0.5} × alpha
        # {0.25..0.50} found (5, 0.2, 0.5) measured +1,360 XIRECs / 3
        # days vs prior (3, 0.33, 0.40) baseline. Threshold axis is flat
        # (3/4/5 all top-10); fraction and alpha are load-bearing.
        # Reserving MORE capacity (0.2 vs 0.33) preserves headroom for
        # the rare deep-tail crossings at mid ≤ 9988 / ≥ 10012 that
        # drive a disproportionate share of osmium PnL.
        "reserve_small_edge": {"threshold": 5, "fraction": 0.2},
        # 5-param MM tuned via two sweeps on R1 D-2/-1/0 (osmium):
        # first 144-combo sweep at alpha=0.40 found (0,3,3,60) best;
        # after alpha bump to 0.50, 54-combo plateau re-check around
        # that winner found (0,2,4,70) slightly better with 14/54
        # combos still improving every day (flat plateau, not spike).
        # - disregard_edge=0: consider every book level including
        #   inside-spread ones when picking join/penny price.
        # - join_edge=2: join any book level within 2 ticks of fair
        #   (matches the tighter reserve model; top 6 in sweep had
        #   join_edge=2).
        # - default_edge=4: fallback distance when no qualifying book
        #   level exists; indifferent across 2/3/4 in top 10.
        # - soft_position_limit=80 (= limit): disables the inventory-
        #   throttle entirely. Measured +735 XIRECs / 3 days over soft=70,
        #   consistent across all 3 days. Mechanism: full-size quotes on
        #   both sides always (no halving at |pos|>threshold); inventory
        #   managed via take-side + clear_width passive posts instead of
        #   make-side size reduction. Top P2/P3 teams all use this
        #   pattern (no MM-side throttle). Note: this is the ONLY
        #   structural change from the P2/P3 research that held up
        #   under direct measurement; claims about hardcoded fair and
        #   removing reserve_small_edge were both refuted (-8,594 and
        #   -1,334 XIRECs respectively when tested in isolation).
        "disregard_edge": 0,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 80,
        # Hard-unwind layer: aggressively cross the spread to flatten
        # when |position| >= trigger, paying a small adverse edge to
        # free capacity for deeper-tail opportunities.
        #
        # Exp #37 (2026-04-20): retuned against the union book — a
        # 97%-coverage R2 day-1 ground truth built from 6 live sandbox
        # samples. Leak diagnosis on union showed 17 of 23 adverse-edge
        # buys came from hard_unwind firing at pos ≤ −60 and paying up
        # to +5 above anchor. Sweep across trigger {40..70} × size
        # {10..30} × max_adverse {0..3} found (40, 20, 1) best:
        #   union     9,276 → 9,335  (+59, out-of-noise)
        #   sandbox μ 8,893 → 8,906  (+13, σ≈87)
        #   ticks@limit (union): 2 → 0; max_short: -80 → -74
        # Prior diagnostic (R2 D-1/0) found (60, 20, 5) best under the
        # single-CSV view; that winner sits inside the 20% sampling
        # noise and missed the adverse-buy leak.
        "hard_unwind": {"trigger": 40, "size": 20, "max_adverse": 1},
        # Dead-book anchor fallback (B26, structural_hunt 2026-04-18):
        # when one side is fully empty (~7% of osmium ticks per Exp #9),
        # synthesize a best level at anchor ± offset = 10000 ± 8. This
        # keeps quote placement anchored to the stable fair value 10000
        # instead of whatever stale last_mid happened to be cached.
        # Distinguishes from Exp #9 (widened fair ± default_edge):
        # that kept fair-anchored; this is anchor-anchored when fair
        # itself is stale. Offset=8 = half of mean osmium spread (~16).
        "dead_book_anchor": {"anchor": 10000, "offset": 8},
    },
    "INTARIAN_PEPPER_ROOT": {
        # Pepper drifts ~+1000 ticks/day consistently across all three
        # training days. Treat it as a directional buy-and-hold: just
        # accumulate longs and let EOD mark-to-market realize the drift.
        #
        # Exp #32 (REJECTED 2026-04-19): re-tested directional_long_with_mm
        # on N=6 full-day sandbox. Mean 9,238 ± 92 vs baseline 9,223 ± 18
        # (Δ=+15, inside noise, σ 5× baseline). Competitor's 9,524 was
        # ~3σ upper tail of same distribution, not a better strategy.
        # Pepper MM layer adds variance without alpha.
        "mode": "directional_long",
        "limit": 80,
        "target_position": 80,     # at limit
        # Experiment #35: accum_max_per_tick=10 pairs with cheapest_only
        # such that each cheap level (~9-11 units observed live) is
        # consumed in a single tick without leaving residual capacity
        # that would otherwise sit unused.
        "accum_max_per_tick": 10,
        # Experiment #34: only lift the cheapest ask level per tick,
        # don't cascade to the next-cheapest when max_per_tick isn't
        # reached. Saves ~0.7 ticks/unit on backtest day 0 by avoiding
        # the 12006 -> 12009 cascade that happens when 12006 has only
        # 10 units vs our 15/tick budget.
        "cheapest_only": True,
    },
    # Test-only product: exercises the MM pipeline that pepper no longer
    # uses. Referenced by unit tests so they don't need to pin to a live
    # product. Not traded in production — never appears in state.order_depths.
    "MM_TEST_PRODUCT": {
        "algo": EMAFairValueAlgo(alpha=0.7),
        "limit": 80,
        "make": True,
        "trend_filter": "slope",
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 40,
    },
}

DEFAULT_ALGO = DefaultAlgo()
DEFAULT_LIMIT = 20


# ── Order Helpers ───────────────────────────────────────────────────


def take_orders(
    order_depth, product, fair_value, position, limit, trend=0,
    min_edge=0, reserve_small_edge=None,
    reserve_small_edge_buy=None, reserve_small_edge_sell=None,
):
    """Take mispriced book levels.

    Returns (orders, buy_volume, sell_volume). Counters track capacity
    consumed on each side so the MM stage can size quotes against true
    remaining headroom (prevents position-limit breaches when take +
    make orders fill in the same tick).

    reserve_small_edge: optional dict {"threshold": N, "fraction": F}.
        When set, crossings where abs(edge) < threshold only consume
        fraction F (0-1) of remaining capacity, leaving the rest dry
        for deep-tail crossings (edge >= threshold) that arrive later.

        Research finding (scored R1 day 0): the bot hit ±80 frequently
        at shallow mids (9995-10002) but had 0 ticks at +80 when mid
        <= 9988. Reserving capacity at small edges preserves headroom
        for the big mean-reverts that actually move the PnL. Omit
        (None) for products where every edge should be taken at full
        size (e.g. directional or tight-spread tight-mean-reverters).

    reserve_small_edge_buy / reserve_small_edge_sell: optional side-split
        variants (B4, structural_hunt 2026-04-18). When set, override
        reserve_small_edge on the respective side. Motivation:
        agent_C_empirics §1 measured an asymmetric adverse cliff —
        BUY cliff between edges 2 and 3 (Δmid@t+5 = -0.07 at E=3),
        SELL cliff between edges 4 and 5 (still -1.60 at E=4). A
        symmetric threshold is over-conservative for BUY and lets the
        BUY side take less than the cliff actually permits.
    """
    orders = []
    buy_volume = 0
    sell_volume = 0

    def _cap_after_reserve(raw_cap, edge, side_config):
        """Scale raw_cap by reserve fraction when edge is small."""
        if side_config is None or raw_cap <= 0:
            return raw_cap
        thresh = side_config.get("threshold", 3)
        frac = side_config.get("fraction", 0.5)
        if edge < thresh:
            return max(1, int(raw_cap * frac))
        return raw_cap

    buy_config = reserve_small_edge_buy if reserve_small_edge_buy is not None else reserve_small_edge
    sell_config = reserve_small_edge_sell if reserve_small_edge_sell is not None else reserve_small_edge

    if order_depth.sell_orders and trend >= 0:
        for ask_price, ask_qty in sorted(order_depth.sell_orders.items()):
            edge = fair_value - ask_price
            if edge <= 0 or edge < min_edge:
                break
            raw_cap = limit - (position + buy_volume)
            if raw_cap <= 0:
                break
            buy_cap = _cap_after_reserve(raw_cap, edge, buy_config)
            qty = min(-ask_qty, buy_cap)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                buy_volume += qty

    if order_depth.buy_orders and trend <= 0:
        for bid_price, bid_qty in sorted(
            order_depth.buy_orders.items(), reverse=True
        ):
            edge = bid_price - fair_value
            if edge <= 0 or edge < min_edge:
                break
            raw_cap = limit + (position - sell_volume)
            if raw_cap <= 0:
                break
            sell_cap = _cap_after_reserve(raw_cap, edge, sell_config)
            qty = min(bid_qty, sell_cap)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                sell_volume += qty

    return orders, buy_volume, sell_volume


def market_make_orders(
    order_depth, product, fair_value, position, limit,
    buy_volume=0, sell_volume=0, trend=0,
    disregard_edge=1, join_edge=2, default_edge=4,
    soft_position_limit=0, dead_book_anchor=None,
):
    """Linear Utility 5-param market maker.

    Price selection (each side independently):
      - Find best book level on your side outside `disregard_edge` from fair.
      - If within `join_edge` of fair: join at same price.
      - Else: penny ahead of that level.
      - If no such level: quote at fair ± default_edge.

    Inventory management: when |position| > soft_position_limit, shift the
    corresponding quote by 1 to flatten faster.

    Capacity: sizes against true remaining headroom
    `limit - (position + buy_volume)` / `limit + (position - sell_volume)`,
    so take_orders + make_orders can both fully fill in one tick without
    breaching the position limit.

    trend: +1 = suppress buys (uptrend), -1 = suppress sells (downtrend).

    dead_book_anchor: optional {"anchor": int, "offset": int}. When a
        side is fully empty (no book levels at all — ~7% of osmium ticks),
        synthesize a best level at anchor ± offset so quotes reference
        the stable anchor instead of stale fair_value. Passes through
        normal join/penny logic on top.
    """
    orders = []

    asks_above_fair = [p for p in order_depth.sell_orders
                       if p > fair_value + disregard_edge]
    bids_below_fair = [p for p in order_depth.buy_orders
                       if p < fair_value - disregard_edge]
    best_ask_above = min(asks_above_fair) if asks_above_fair else None
    best_bid_below = max(bids_below_fair) if bids_below_fair else None

    if dead_book_anchor is not None:
        anchor = dead_book_anchor["anchor"]
        offset = dead_book_anchor["offset"]
        if not order_depth.sell_orders and best_ask_above is None:
            best_ask_above = anchor + offset
        if not order_depth.buy_orders and best_bid_below is None:
            best_bid_below = anchor - offset

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

    buy_qty = limit - (position + buy_volume)
    if trend >= 0 and buy_qty > 0:
        orders.append(Order(product, int(bid), buy_qty))

    sell_qty = limit + (position - sell_volume)
    if trend <= 0 and sell_qty > 0:
        orders.append(Order(product, int(ask), -sell_qty))

    return orders


# ── Trader (Router) ─────────────────────────────────────────────────

class Trader:

    # ── Round 2 MAF bid ──────────────────────────────────────────────
    # Bayes-optimal under Model 04 priors = 140-160. Shipped 500 as
    # asymmetric-payoff hedge: +305 premium over near-optimal buys
    # ~5% → ~0.5% reduction in P(lose auction), where losing forfeits
    # ~7.7k/3d union-book uplift. Insurance leverage ~25× on premium.
    MAF_BID = 500

    def bid(self):
        """Round 2 Market Access Fee (MAF) bid. Top 50% of bids win
        full 100% book access; losers stay on 80% sample. Accepted
        bids pay bid amount once, deducted from R2 PnL. Ignored in
        R1/R3/R4/R5.
        """
        return self.MAF_BID

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

        # Defensive runtime scans: surface planted signals if IMC populates
        # plainValueObservations or introduces an unknown product mid-round.
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
                trader_state = {}

        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]

            # Safety gate: if a product has no entry in PRODUCT_CONFIG,
            # do nothing. This prevents accidentally trading new Round 2
            # products before we've characterized them. Once A.R.I.A.
            # publishes a new product, add its config entry explicitly.
            if product not in PRODUCT_CONFIG:
                result[product] = []
                continue

            config = PRODUCT_CONFIG.get(product, {})
            mode = config.get("mode", "mm")
            limit = config.get("limit", DEFAULT_LIMIT)
            position = state.position.get(product, 0)

            # Directional-long mode bypasses the MM pipeline entirely.
            if mode == "directional_long":
                result[product] = directional_long_orders(
                    order_depth, product, position, limit,
                    target_position=config.get("target_position", limit - 2),
                    max_per_tick=config.get("accum_max_per_tick", None),
                    cheapest_only=config.get("cheapest_only", False),
                )
                continue

            if mode == "directional_long_with_mm":
                if product not in trader_state or not isinstance(
                        trader_state.get(product), dict):
                    trader_state[product] = {}
                ords, new_state = directional_long_with_mm_orders(
                    order_depth, product, position, limit,
                    state_data=trader_state[product],
                    target_position=config.get("target_position", limit - 2),
                    max_per_tick=config.get("accum_max_per_tick", None),
                    cheapest_only=config.get("cheapest_only", False),
                    mm_alpha=config.get("mm_alpha", 0.10),
                    mm_edge=config.get("mm_edge", 5),
                    mm_size=config.get("mm_size", 5),
                    mm_min_position_for_sell=config.get(
                        "mm_min_position_for_sell", 20),
                    mm_max_position_for_buy=config.get(
                        "mm_max_position_for_buy", 80),
                )
                trader_state[product] = new_state
                result[product] = ords
                continue

            if mode == "directional_long_dip_catch":
                result[product] = directional_long_dip_catch_orders(
                    order_depth, product, position, limit,
                    accumulation_target=config.get("accumulation_target", 72),
                    max_per_tick=config.get("accum_max_per_tick", 15),
                    dip_catch_size=config.get("dip_catch_size", 8),
                )
                continue

            algo = config.get("algo", DEFAULT_ALGO)
            use_make = config.get("make", False)
            trend_filter = config.get("trend_filter", None)
            take_min_edge = config.get("take_min_edge", 0)
            reserve_small_edge = config.get("reserve_small_edge", None)
            reserve_small_edge_buy = config.get("reserve_small_edge_buy", None)
            reserve_small_edge_sell = config.get("reserve_small_edge_sell", None)
            disregard_edge = config.get("disregard_edge", 1)
            join_edge = config.get("join_edge", 2)
            default_edge = config.get("default_edge", 4)
            soft_position_limit = config.get("soft_position_limit", limit // 2)
            dead_book_anchor = config.get("dead_book_anchor", None)

            if product not in trader_state:
                trader_state[product] = algo.default_state()
                trader_state[product]["midprices"] = []

            if "midprices" not in trader_state[product]:
                trader_state[product]["midprices"] = []

            market_trades = state.market_trades.get(product, [])
            fair_value, trader_state[product] = algo.compute_fair_value(
                order_depth, market_trades, trader_state[product]
            )

            if fair_value is None:
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

            trend = 0
            if trend_filter == "slope":
                trend = compute_slope_trend(trader_state[product]["midprices"])
            elif trend_filter == "obi":
                # Config may tune the OBI threshold per-product.
                obi_thresh = config.get("obi_threshold", 0.3)
                trend = compute_obi_trend(order_depth, threshold=obi_thresh)

            print(
                f"{product} | FV: {fair_value:.2f} "
                f"| algo: {type(algo).__name__} "
                f"| Pos: {position}/{limit}"
                f"| make: {use_make}"
                f"| trend: {trend}"
            )

            orders, buy_volume, sell_volume = take_orders(
                order_depth, product, fair_value, position, limit,
                trend=trend, min_edge=take_min_edge,
                reserve_small_edge=reserve_small_edge,
                reserve_small_edge_buy=reserve_small_edge_buy,
                reserve_small_edge_sell=reserve_small_edge_sell,
            )

            # Optional OBI-aggression pass: directional take driven by
            # order-book imbalance signal (see obi_take_orders docstring).
            obi_config = config.get("obi_take", None)
            if obi_config:
                obi_orders, buy_volume, sell_volume = obi_take_orders(
                    order_depth, product, position, limit,
                    buy_volume=buy_volume, sell_volume=sell_volume,
                    threshold=obi_config.get("threshold", 0.3),
                    aggress_size=obi_config.get("aggress_size", 10),
                    position_aware=obi_config.get("position_aware", False),
                )
                orders.extend(obi_orders)

            # Optional hard-unwind pass: aggressively cross the spread
            # to flatten when deep in saturation. Frees capacity for
            # tail crossings. See hard_unwind_orders docstring.
            unwind_config = config.get("hard_unwind", None)
            if unwind_config:
                unwind_orders, buy_volume, sell_volume = hard_unwind_orders(
                    order_depth, product, position, limit,
                    buy_volume=buy_volume, sell_volume=sell_volume,
                    trigger=unwind_config.get("trigger", 60),
                    size=unwind_config.get("size", 10),
                    max_adverse=unwind_config.get("max_adverse", 3),
                    fair_value=fair_value,
                )
                orders.extend(unwind_orders)

            # Optional end-of-day flatten pass: at start_ts, actively
            # unwind any remaining position to reduce MTM-at-final-tick
            # adverse risk. See eod_flatten_orders docstring.
            eod_config = config.get("eod_flatten", None)
            if eod_config:
                eod_orders, buy_volume, sell_volume = eod_flatten_orders(
                    order_depth, product, position, state.timestamp, limit,
                    buy_volume=buy_volume, sell_volume=sell_volume,
                    start_ts=eod_config.get("start_ts", 990000),
                    size_per_tick=eod_config.get("size_per_tick", 10),
                    max_adverse=eod_config.get("max_adverse", 5),
                    fair_value=fair_value,
                )
                orders.extend(eod_orders)

            if use_make:
                orders.extend(
                    market_make_orders(
                        order_depth, product, fair_value,
                        position, limit,
                        buy_volume=buy_volume, sell_volume=sell_volume,
                        trend=trend,
                        disregard_edge=disregard_edge,
                        join_edge=join_edge,
                        default_edge=default_edge,
                        soft_position_limit=soft_position_limit,
                        dead_book_anchor=dead_book_anchor,
                    )
                )

            result[product] = orders

        traderData = jsonpickle.encode(trader_state)
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData