# Round 4 — 84,370 Seashells

## Universe

Same instruments as R3 (`HYDROGEL_PACK`, `VELVETFRUIT_EXTRACT`, voucher chain 4000–6500), with an enriched counterparty / inventory-dynamics regime.

## Strategy

R3 architecture, retuned with:

- **VE-deviation directional role** on `VEV_5100`–`VEV_5500` — when `VELVETFRUIT_EXTRACT` deviated from its anchor, lean voucher quotes in the direction of the expected reversion.
- **Counterparty-lean targeting** on `HYDROGEL_PACK` — retargeted from Q3 to Q1+Q2 quartile inventory, expressed via anchor offset.
- **Piecewise-linear regime blend** on `HYDROGEL_PACK` — replaced a hard regime cliff that was causing whipsaw losses.
- **Tightened position cap** on `VEV_5000` (45 → 20) — broad-plateau Pareto improvement: smaller cap = same fill rate + smaller drawdowns.
- **Bayesian-seeded anchor** replacing 4,000-tick warmup on `dev_directional` — bot starts trading in tick 1 with a prior, rather than missing the first 4,000 ticks.
- **Partial-mean anchor fallback** at min_warmup=1000 for short-history products.

3-day backtest: **+781,508**; 4-day backtest with `mp=20`: **+781k** (ship configuration).

## What worked

- Bayesian-seeded anchors recovered ~10% of P&L that was being lost during warmup.
- Piecewise-linear regime blend (vs. cliff) was the largest single P&L lift in R4.
- Per-strike position-cap tuning — counter-intuitively, smaller caps Pareto-dominated on the noisier strikes.

## What didn't

- Several R3 voucher residual signals had decayed by R4 — the counterparty mix shifted and old patterns were no longer there. Cut.
- More aggressive smile-fit on the wings — still biased, no improvement.
