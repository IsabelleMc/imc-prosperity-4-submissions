# Lessons across rounds

Things I learned the hard way, in order of how much P&L they would have saved.

### 1. Local backtest ≠ live submission

In R5 the local backtester P&L was ~10× the live result, and per-product P&Ls were *frequently inverted* — products that "won" locally lost live. After this was discovered, submission deltas became the only trusted signal. Lesson: the only ground truth is the scored run.

### 2. t-stat ≠ shippable

A signal can be statistically significant on historical data and still lose money once you account for the *channel* — the spread you cross to enter, the queue position you forfeit, the inventory you build while waiting. Many R3/R4/R5 signals had clean offline t-stats and lost in production because the channel cost ate the edge.

### 3. Always validate on 3+ days

D2-only validation produced false positives in R5. A cross-product OBI skew that scored +188 on D2 was −12,439 on D3 and −12,237 on D4. Single-day validation is essentially noise.

### 4. Sweep before declaring a ceiling

Whenever I thought "this is the best I can do," exhaustive parameter sweeps (position size, quote skew, warmup window, threshold) usually found another 5–20% on top. Don't trust the first plateau.

### 5. Don't trust chart-bias diagnostics

Visual inspection of "this signal looks like it leads" routinely failed verification. Trust the verifier (out-of-sample backtest), not the visual.

### 6. Backtester has no algo trading fees

Source audit confirmed `prosperity3bt` does not deduct fees on fills or EOD MTM. Local sim is gross of any fee that doesn't exist; don't pre-emptively reduce turnover for speculative algo fees. (The fee hint from in-game NPCs was about *manual* trading.)

### 7. Aggressors lose in R5

Counterparty-side P&L analysis showed aggressors (takers) lost ~5.6 ticks per trade on average. Passive market making was harvesting that. The gap between my bot and the L1 ceiling was a fill-rate problem, not an edge problem.

### 8. Match-trades mode matters

R5 sandbox default `--match-trades` mode silently passes all passive at-touch orders; the floor I measured was fake. Switching to `--match-trades=worse` collapsed several "winning" strategies to zero.

### 9. Saturation is real

Inside-touch market making in R5 was L1-saturated for every product — bots only crossed the topmost quote, so quote sizes above 4–5 contracts were wasted. Quote-skew at ±1 also saturated on tight-spread products.

### 10. Pinning at EOD

The closing tick marks open positions to the bid/ask midpoint, which has hit me when in-sample tail behavior diverged from live. Default to a position floor of ~50% of the window's mean for 1-day-ahead trading.
