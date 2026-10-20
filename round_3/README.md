# Round 3 — 55,480 Seashells

## Universe

Options round. Underlying + 12 strikes:

- `HYDROGEL_PACK` — underlying-style spot product, mean-reverting
- `VELVETFRUIT_EXTRACT` — second underlying, with persistent drift
- `VEV_4000`, `VEV_4500`, `VEV_5000`, `VEV_5100`, `VEV_5200`, `VEV_5300`, `VEV_5400`, `VEV_5500`, `VEV_6000`, `VEV_6500` — voucher (call) chain at 12 strikes

## Strategy

Per-strike specialisation across the voucher chain:

- **Deep-ITM / deep-OTM strikes** — pure inside-touch MM. Spreads were wide enough that signal logic was unnecessary.
- **ATM strikes (5000–5400)** — Black-Scholes smile fit across the chain to compute a fair value, then quote around FV with a residual gate. Volatility surface fit was leave-one-out quadratic on log-strike.
- **Wing strikes (4000, 4500, 6000, 6500)** — wing-fit FV with a separate residual band; thinner liquidity, smaller quote size.
- **`HYDROGEL_PACK`** — microprice + cumulative order-flow-imbalance skew with anchor-based fair value.
- **`VELVETFRUIT_EXTRACT`** — anchor MM with drift-projected anchor.
- **Delta hedging** — post-trade hedger on the underlying when net voucher delta exceeded a band. Structurally undersized (voucher delta capacity ~945 vs hedger limit 180), which capped how much directional voucher inventory I could carry.

Inline `statistics.NormalDist` Black-Scholes — scipy is forbidden in the submission environment.

## What worked

- Smile-fit FV on the ATM cluster gave a real residual signal that survived OOS verification.
- Strike-specific role assignment outperformed a single uniform algorithm — deep ITM/OTM didn't need any signal at all.

## What didn't

- Smile-fit on `VEV_5400` had a chronic −$2.50 bias that did *not* mean-revert.
- `VEV_5500` wing fit was biased −$0.30 on days 1/2.
- Post-trade delta hedger was undersized: 76% of D3-D2 ticks had `|net_voucher_delta| > VE_limit` — running unhedged exposure.
- Parametric smile-arb across ATM cluster (Exp #13/#14) — looked great, falsified in sandbox.

## Notes

P&L lower than R1/R2 because options pricing is harder than spot MM and the smile residuals were noisier than expected.
