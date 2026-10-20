# Round 2 — 83,259 Seashells

## Universe

R1 products carried forward, plus instruments introduced in R2.

## Strategy

Re-tuned R1 market maker plus per-product role specialisation:

- **Stable products**: anchor mean-reversion + inside-touch MM (R1 baseline).
- **Drifting products**: faster-updating anchor with asymmetric position floors.
- **Wider-spread products**: simple inside-touch MM; spread alone covers fill cost.

The architecture from R1 generalised cleanly — most of the lift came from per-product parameter tuning (anchor halflife, inventory-skew steepness, quote size) rather than new logic.

## What worked

- Reusing R1's product-router pattern (each product gets a `role` and config dict) made adding new products fast.
- Bayesian-seeded anchors for short-history products (cold-start without a 4,000-tick warmup gate).

## What didn't

- Cross-product correlation overlays — pairs that looked correlated in chart had no exploitable lead-lag.
- Aggressive taking on perceived mispricings — channel cost (cross the spread) > edge (~1–2 ticks).
