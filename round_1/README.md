# Round 1 — 96,782 Seashells

## Universe

- `INTARIAN_PEPPER_ROOT` — drifting commodity, mid ~13,000
- `ASH_COATED_OSMIUM` — stable mean-reverting, mid ~10,008

## Strategy

Two-product market maker.

- **Inside-touch quoting** at L1 with inventory-aware skew: when long, ask tighter than bid; when short, the reverse. This pulls inventory back toward zero without giving up edge.
- **Anchor mean-reversion** on `ASH_COATED_OSMIUM` — the mid sits in a narrow band, so quote around a long-window EMA anchor and harvest reversion to the mean.
- **Drift-tolerant quoting** on `INTARIAN_PEPPER_ROOT` — the mid drifts persistently, so the anchor is updated faster and the position floor is asymmetric (longer is fine, shorter accumulates risk).

## What worked

- Inside-touch MM was the workhorse. Both products had wide enough spreads that passive fills covered the round-trip cost.
- Inventory skew prevented the bot from ever pinning at the position limit, which would have caused EOD MTM losses.

## What didn't (cut before submission)

- Order-book imbalance overlays — looked predictive on chart but had ~zero IC out-of-sample.
- Trade-tape momentum — aggressor flow was anti-momentum noise.
