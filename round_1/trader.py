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

    def __init__(self, anchor, alpha=0.15, band=1.0):
        self.anchor = anchor
        self.alpha = alpha
        self.band = band

    def default_state(self):
        return {"ema": None}

    def compute_fair_value(self, order_depth, market_trades, state_data):
        mid = self._midprice(order_depth)
        if mid is None:
            return None, state_data

        prev_ema = state_data.get("ema")
        ema = mid if prev_ema is None else (
            self.alpha * mid + (1 - self.alpha) * prev_ema
        )
        state_data["ema"] = ema

        # Blend EMA toward anchor — the further away, the stronger pull
        fair_value = self.anchor + self.alpha * (ema - self.anchor)
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




PRODUCT_CONFIG = {
    "ASH_COATED_OSMIUM": {
        "algo": MeanReversionAlgo(anchor=10000, alpha=0.40),
        "limit": 80,
        "make": True,
        "skew_factor": 0.5,
        "trend_filter": None,
        "long_skew_mult": 1.5,
        "short_skew_mult": 0.7,
        "take_min_edge": 0,
    },
    "INTARIAN_PEPPER_ROOT": {
        # Pepper drifts ~+1000 ticks/day consistently across all three
        # training days. Treat it as a directional buy-and-hold: just
        # accumulate longs and let EOD mark-to-market realize the drift.
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
        "skew_factor": 0.5,
        "trend_filter": "slope",
    },
}

DEFAULT_ALGO = DefaultAlgo()
DEFAULT_LIMIT = 20


# ── Order Helpers ───────────────────────────────────────────────────


def take_orders(
    order_depth, product, fair_value, position, limit, trend=0,
    min_edge=0,
):
    """Take mispriced book levels. Returns (orders, updated_position).

    min_edge: skip levels whose edge vs. fair_value is below this
    threshold. Use to filter 1-tick noise trades that adversely select
    us. min_edge=0 preserves legacy "take anything mispriced" behavior.

    trend: 0 = no filter, +1 = uptrend (suppress sells), -1 = downtrend.
    """
    orders = []

    if order_depth.sell_orders and trend >= 0:
        for ask_price, ask_qty in sorted(order_depth.sell_orders.items()):
            edge = fair_value - ask_price
            if edge <= 0 or edge < min_edge:
                break
            buy_cap = limit - position
            if buy_cap <= 0:
                break
            qty = min(-ask_qty, buy_cap)
            if qty > 0:
                orders.append(Order(product, ask_price, qty))
                position += qty

    if order_depth.buy_orders and trend <= 0:
        for bid_price, bid_qty in sorted(
            order_depth.buy_orders.items(), reverse=True
        ):
            edge = bid_price - fair_value
            if edge <= 0 or edge < min_edge:
                break
            sell_cap = limit + position
            if sell_cap <= 0:
                break
            qty = min(bid_qty, sell_cap)
            if qty > 0:
                orders.append(Order(product, bid_price, -qty))
                position -= qty

    return orders, position


def market_make_orders(
    order_depth, product, fair_value, position, limit,
    skew_factor=1.0, trend=0,
    short_skew_mult=1.0, long_skew_mult=1.0,
):
    """Inventory-aware penny-ahead MM around top of book.

    Posts FULL remaining capacity at a single penny-ahead price per
    side. All top teams (Frankfurt, CMU, Linear Utility) use this
    pattern — one price per side, full size = higher fill probability
    per tick than splitting across multiple levels.

    Inventory skew shifts quote prices toward flat. Asymmetric skew
    multipliers let us flatten longs and shorts at different rates.

    trend: +1 = suppress buys (uptrend), -1 = suppress sells (downtrend).
    """
    orders = []
    buy_cap = limit - position
    sell_cap = limit + position

    if not order_depth.buy_orders or not order_depth.sell_orders:
        return orders

    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())

    inv_ratio = position / limit if limit > 0 else 0.0
    raw_skew = inv_ratio * skew_factor
    if position > 0:
        raw_skew *= long_skew_mult
    elif position < 0:
        raw_skew *= short_skew_mult
    skew = round(raw_skew)

    # Single price per side, full capacity
    if trend >= 0 and buy_cap > 0:
        price = best_bid + 1 - skew
        if price < fair_value:
            orders.append(Order(product, int(price), buy_cap))

    if trend <= 0 and sell_cap > 0:
        price = best_ask - 1 - skew
        if price > fair_value:
            orders.append(Order(product, int(price), -sell_cap))

    return orders


# ── Trader (Router) ─────────────────────────────────────────────────

class Trader:

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
            skew_factor = config.get("skew_factor", 1.0)
            trend_filter = config.get("trend_filter", None)
            take_min_edge = config.get("take_min_edge", 0)
            short_skew_mult = config.get("short_skew_mult", 1.0)
            long_skew_mult = config.get("long_skew_mult", 1.0)

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

            print(
                f"{product} | FV: {fair_value:.2f} "
                f"| algo: {type(algo).__name__} "
                f"| Pos: {position}/{limit}"
                f"| make: {use_make}"
                f"| trend: {trend}"
            )

            orders, position = take_orders(
                order_depth, product, fair_value, position, limit,
                trend=trend, min_edge=take_min_edge,
            )

            if use_make:
                orders.extend(
                    market_make_orders(
                        order_depth, product, fair_value,
                        position, limit, skew_factor, trend=trend,
                        short_skew_mult=short_skew_mult,
                        long_skew_mult=long_skew_mult,
                    )
                )

            result[product] = orders

        traderData = jsonpickle.encode(trader_state)
        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData