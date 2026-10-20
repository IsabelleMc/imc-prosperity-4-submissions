# IMC Prosperity 4 — Trading Algorithm Submissions

My submissions for [IMC Prosperity 4](https://prosperity.imc.com/), a 15-day algorithmic trading competition run by IMC Trading. Each round, teams submit a Python `trader.py` that places orders against a simulated market populated with bot counterparties; submissions are scored by net P&L in Seashells (the in-game currency).

## Results

| Round | Universe | Final P&L (Seashells) |
|------:|:---------|----------------------:|
| 1 | Stable mean-reverting + drifting commodities | **96,782** |
| 2 | R1 universe + new instruments | **83,259** |
| 3 | Hydrogel + Velvetfruit options chain (12 strikes) | **55,480** |
| 4 | R3 universe, expanded counterparty/inventory dynamics | **84,370** |
| **Total (R1–R4)** | | **319,890** |

**Overall result: top 5% of teams.**

## Repo Layout

```
round_1/   trader.py — R1 submission
round_2/   trader.py — R2 submission
round_3/   trader.py — R3 submission
round_4/   trader.py — R4 submission
```

Each round folder has a `README.md` summarising the strategy and what worked / didn't.

## Submission constraints

`trader.py` is uploaded as-is to IMC. The runtime allows only:

- Python stdlib + `jsonpickle`
- No threads, network calls, file I/O, or `time.sleep`
- Bounded `traderData` (no unbounded list growth across ticks)
- Hard per-product position limits enforced server-side

Black-Scholes pricing (R3/R4) uses `statistics.NormalDist` rather than scipy.

## Approach (high level)

- **Liquidity provision (market making)** as the baseline P&L source on every product where the bid/ask spread is wide enough to cover fill cost.
- **Inventory-skewed quoting** — when long, lean asks tighter than bids, and vice versa, to mean-revert the position.
- **Signal overlays** — order-book imbalance (OBI), microprice, anchor mean-reversion, smile-fit fair-value for options. Most signals failed verification; the ones that survived are in the per-round notes.
- **Backtester-driven validation** — every change verified on multiple days before shipping. Local 3-day backtest agreement with live did *not* always hold, especially in R5; submission deltas became the authoritative signal.

## What I learned

The full set of cross-round engineering lessons is in [LESSONS.md](LESSONS.md).
