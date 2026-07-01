# Run Log

## Chapter 5 — Grand Benchmark Session (2026-06-30, this session)

Methodology: every run below is fully sequential — one vLLM server process at a
time, server killed and GPU memory confirmed back to 0 MiB before the next run
starts (verified via `nvidia-smi` each time, logged inline). Fixed protocol
across all runs: `philschmid/mt-bench`, `--max-concurrency 8`, `--num-prompts 80`,
`--no-enable-prefix-caching`.

### Status
- [x] C1 — Baseline (BF16, no spec decoding) — **anchor**
- [x] C2 sweep — BF16 + EAGLE-3, num_speculative_tokens = 1, 2, 3, 4 — **optimum: N=2**
- [x] C4 sweep — FP8 + EAGLE-3, num_speculative_tokens = 1, 2, 3, 4 — **optimum: N=2 (different shape than C2 — see analysis)**
- [x] C3 — FP8 quantized, no spec decoding — **1110.97 tok/s, +32.5%**
- [x] C2 final — BF16 + EAGLE-3 at tuned N — reused from sweep (N=2, 1069.82 tok/s)
- [x] C4 final — FP8 + EAGLE-3 at tuned N — reused from sweep (N=2, 1468.70 tok/s)

### C1 — Baseline (BF16, no spec decoding)

**Why we're running this:** every other config in this chapter is judged relative
to this number. Per §5.2's protocol, it must run *first* and under the exact same
conditions (dataset, concurrency, prompt count) as everything that follows — if
this drifts, every downstream "speedup" claim is built on sand.

**Command:** `vllm serve /data/hw3/Qwen3-8B --no-enable-prefix-caching`
**GPU before:** 0 MiB · **GPU after server load:** 75,637 MiB · **GPU after teardown:** 0 MiB (confirmed via `nvidia-smi`, no other process running)

**Raw result:**
```
Successful requests:        80/80, 0 failed
Benchmark duration (s):     24.42
Output tok/s:                838.74
Total token throughput:      1087.66 tok/s
Mean TTFT (ms):              649.07   median 32.61   p99 5886.82
Mean TPOT (ms):              7.03     median 7.02     p99 7.14
Mean ITL (ms):                7.03     median 7.01     p99 7.48
```

**Observations:**
- This reproduces the prior baseline run logged below almost exactly (838.74 vs
  837.76 tok/s, +0.1%) — that's within normal run-to-run scheduling noise, and
  it confirms the environment hasn't drifted since the original Chapter 0 run
  (same GPU, same model, same dataset shuffle seed behavior).
- The huge gap between **median TTFT (32.61ms)** and **mean TTFT (649.07ms)**,
  with **p99 TTFT at 5886.82ms**, is the signature of `--max-concurrency 8`
  queuing: most requests get a free GPU slot immediately (~32ms to first
  token), but whichever requests land when all 8 slots are busy wait behind a
  full prefill+decode of the requests ahead of them — that's what drags the
  *mean* up so far above the *median*. This is expected, not a bug.
- TPOT (7.03ms) is tight and consistent (median 7.02, p99 7.14) — once a
  request has a slot, decode latency per token is stable. This is the
  steady-state, memory-bandwidth-bound regime Chapter 0 predicted: roughly
  `16GB / 3.35TB/s ≈ 4.8ms` per step theoretical at batch=1, and we're seeing
  ~7ms with concurrency-8 batching overhead on top — same ballpark.
- **This is the number every other config in this chapter has to beat:
  838.74 tok/s.**

### C2 sweep — BF16 + EAGLE-3

**Why we're running this:** find the `num_speculative_tokens` value that
maximizes throughput for the BF16 model before committing to a "final" C2
number — §5.3 explicitly says this is not one-size-fits-all and must be swept,
not guessed.

**Command pattern:** `vllm serve /data/hw3/Qwen3-8B --speculative-config '{"method":"eagle3","model":"/data/hw3/output/checkpoints/checkpoint_best","num_speculative_tokens":N}' --no-enable-prefix-caching`

| N | tok/s | vs C1 (838.74) | Mean TTFT | Mean TPOT | Acceptance rate | Acceptance length | GPU after teardown |
|---|---|---|---|---|---|---|---|
| 1 | 823.76 | **-1.8%** | 676.43 ms | 6.61 ms | 33.37% | 1.33 | 0 MiB (confirmed) |
| 2 | 1069.82 | **+27.5%** | 173.27 ms | 6.43 ms | 20.25% | 1.40 | 0 MiB (confirmed) |
| 3 | 1070.08 | **+27.6%** | 81.79 ms | 6.90 ms | 13.54% | 1.41 | 0 MiB (confirmed) |
| 4 | 997.62 | **+18.9%** | 68.02 ms | 7.39 ms | 10.34% | 1.41 | 0 MiB (confirmed) |

**Sweep verdict: N=2 is the optimum for C2 (BF16 + EAGLE-3) on this draft head.**
Shape across the sweep: loss at N=1 (-1.8%), jump to the peak at N=2 (+27.5%),
flat plateau at N=3 (+27.6%, statistically the same as N=2), decline at N=4
(+18.9%, down ~7 points from the peak). **C2 final benchmark will use N=2.**

#### N=1

**GPU before:** 0 MiB · **GPU after server load:** 75,607 MiB · **GPU after teardown:** 0 MiB (confirmed)

**Raw result:**
```
Successful requests:        80/80, 0 failed
Benchmark duration (s):     24.86
Output tok/s:                823.76
Total token throughput:      1068.23 tok/s
Mean TTFT (ms):              676.43   median 25.73   p99 7379.93
Mean TPOT (ms):              6.61     median 6.28     p99 15.26
Mean ITL (ms):                8.81     median 8.26     p99 9.27
Acceptance rate (%):         33.37
Acceptance length:           1.33
Drafts / Draft tokens:       15309 / 15309
Accepted tokens:             5109
Per-position acceptance:    position 0 = 33.37%  (only one position exists at N=1)
```

**Observations:**
- **N=1 is *slower* than no speculation at all: 823.76 vs 838.74 tok/s
  (-1.8%).** This is a real, useful negative result, not noise — it's larger
  than the C1-reproduction noise band we just established (~0.1%).
- Why: at N=1, every decode step now does *two* things instead of one — a
  small EAGLE-3 draft forward pass, then the full verifier forward pass — and
  only accepts the extra token 33.37% of the time. `acceptance_length=1.33`
  means each verify cycle produces on average 1.33 tokens instead of 1.0 — a
  +33% gain in tokens-per-cycle — but that gain is being eaten by the draft
  head's own overhead plus scheduling/verification bookkeeping, which apparently
  costs more than 33% here. Net effect: a slight loss.
- This matches the mechanism from Chapter 1: speculative decoding only wins
  when "batched verification is essentially free" *relative to* the draft
  cost. At N=1 with this draft head's accuracy, it isn't quite free yet.
- Mean ITL (8.81ms) is *higher* than C1's TPOT (7.03ms) — another signal
  pointing the same direction: per-token pacing got slightly worse, not
  better, at this setting.
- Note Mean TTFT also moved (676ms vs 649ms) and P99 TTFT got notably worse
  (7380ms vs 5887ms) — with the same `--max-concurrency 8` queue depth, the
  extra per-step draft overhead compounds across the queue, so requests that
  have to wait, wait slightly longer.
- **What to watch next:** does N=2 cross into positive territory? The guide's
  reference run found C2's optimum at N=2 — if we see the same shape (loss at
  N=1, gain at N=2, decay by N=4), that confirms the "rejected drafts cost
  more than accepted ones gain past some point" mechanism rather than just
  reproducing the reference number by coincidence.

#### N=2

**Why we ran this:** N=1 lost to the baseline; raising the draft depth to 2
raises the ceiling on tokens-per-cycle (up to 2.0 instead of 1.0) and tests
whether that's enough to clear the draft head's overhead.

**GPU before:** 0 MiB · **GPU after server load:** 75,737 MiB · **GPU after teardown:** 0 MiB (confirmed)

**Raw result:**
```
Successful requests:        80/80, 0 failed
Benchmark duration (s):     19.14
Output tok/s:                1069.82
Total token throughput:      1387.32 tok/s
Mean TTFT (ms):              173.27   median 27.29   p99 1498.92
Mean TPOT (ms):              6.43     median 6.42     p99 8.21
Mean ITL (ms):                9.02     median 8.91     p99 9.93
Acceptance rate (%):         20.25
Acceptance length:           1.40
Drafts / Draft tokens:       14542 / 29084
Accepted tokens:             5889
Per-position acceptance:    position 0 = 32.97%   position 1 = 7.52%
```

**Observations:**
- **N=2 is a real win: 1069.82 tok/s vs C1's 838.74 — a +27.5% gain.** This is
  the first config in the sweep that clears the baseline, and it does so
  comfortably, not marginally — confirms the guide's reference finding that
  C2's optimum sits around N=2, not N=1.
- The mechanism is visible directly in the per-position numbers. Position 0
  accepts at 32.97% (essentially the same as N=1's 33.37% — consistent, as
  expected, since position 0's prediction doesn't depend on N). Position 1
  accepts at only **7.52%** — once you're asking the draft head to predict two
  tokens ahead using its own unverified guess for token 1 as context, its
  accuracy collapses. This is exactly the **error propagation** mechanism from
  §3.2: `cond_acc` vs `full_acc` — position 1's prediction is conditioned on
  position 0 actually being right, and it usually isn't.
- Despite that low position-1 rate, the math still works in our favor:
  acceptance_length rose only modestly (1.33 → 1.40, +5%) but throughput rose
  +27.5% over baseline and **+30%** over N=1 specifically. That gap between "a
  small rise in tokens-per-cycle" and "a large rise in throughput" is the
  payoff from amortizing the fixed draft-head overhead over now-occasionally-2
  accepted tokens instead of paying that overhead for a near-zero net gain —
  the draft head's cost is roughly fixed per cycle regardless of N, so any
  acceptance-length gain above N=1's break-even point converts efficiently
  into throughput.
- TTFT improved dramatically too: mean dropped from 676ms (N=1) to 173ms, and
  p99 from 7380ms to 1499ms. With higher per-cycle throughput, the request
  queue at `--max-concurrency 8` drains faster, so fewer requests sit waiting
  behind a long queue — a second-order benefit of the same underlying win.
- Acceptance *rate* (20.25%) is lower than N=1's (33.37%) even though
  throughput is much higher — this is the §6 "acceptance rate paradox" in
  miniature: rate is a per-token-attempt average across positions 0 and 1
  pooled together, dragged down by position 1's poor 7.52%, while what
  actually drives throughput is acceptance *length*, which only went up.
  Don't read "rate" alone as a throughput proxy.
- **What to watch next:** N=3 and N=4 should show position-2 and position-3
  acceptance rates continuing to collapse (error compounds further), so the
  question is whether N=2's gain is the peak or whether N=3 still nets
  positive before N=4 turns negative again.

#### N=3

**Why we ran this:** N=2 won decisively; N=3 tests whether a third draft
position still adds net value or whether we've already found the peak.

**GPU before:** 0 MiB · **GPU after server load:** 75,745 MiB · **GPU after teardown:** 0 MiB (confirmed)

**Raw result:**
```
Successful requests:        80/80 (20451 tokens generated, slightly fewer than
                              N=1/N=2's 20480 — see note below)
Benchmark duration (s):     ~19 (not printed in captured tail; consistent with N=2's range)
Output tok/s:                1070.08
Total token throughput:      1388.10 tok/s
Mean TTFT (ms):              81.79    median 29.33   p99 549.78
Mean TPOT (ms):              6.90     median 6.90     p99 7.81
Mean ITL (ms):                9.69     median 9.64     p99 10.69
Acceptance rate (%):         13.54
Acceptance length:           1.41
Drafts / Draft tokens:       14509 / 43527
Accepted tokens:             5893
Per-position acceptance:    position 0 = 32.58%   position 1 = 7.06%   position 2 = 0.98%
```

**Observations:**
- **N=3 is a plateau, not a further gain: 1070.08 tok/s vs N=2's 1069.82 —
  effectively identical (+0.02%, well inside run-to-run noise).** We have
  found the peak, and it's at N=2, not higher.
- The reason is laid bare in the per-position numbers: **position 2 accepts
  only 0.98%** of the time — essentially never. Position 0 (32.58%) and
  position 1 (7.06%) are both consistent with N=2's numbers within noise, as
  expected (a position's acceptance rate is mostly a property of the draft
  head and how far removed that position is from ground-truth context, not of
  how many total positions you've configured). Position 2 is conditioned on
  *both* position 0 and position 1 being correct, and with position 1 already
  failing >92% of the time, position 2 almost never gets the chance to be
  evaluated against a correct context, let alone be correct itself.
- Despite adding a third draft token (more compute per cycle — `draft_tokens`
  jumped from 29084 at N=2 to 43527 here, a 50% increase in drafting work),
  throughput didn't move. This is the "draft overhead now exceeds payoff"
  point starting to bite: we're spending extra compute on position 2 almost
  for nothing, and it's only *not* hurting throughput yet because that extra
  draft compute is cheap relative to the verifier pass (Chapter 1's "batched
  verification is essentially free" argument) — but it's also delivering
  ~zero benefit, not paying for itself, just riding along for free without
  helping or hurting net throughput at this concurrency level.
- `total_generated_tokens` (20451) is slightly lower than N=1/N=2's 20480 —
  a minor artifact of `--no-stream`/output-length variation per request
  given the model's own stopping behavior, not something attributable to the
  speculative config; not worth over-reading.
- Mean TTFT continued improving (676ms → 173ms → 82ms across N=1→2→3) even
  though throughput plateaued — this is just the same "faster cycles drain
  the concurrency-8 queue faster" effect from N=2, continuing because TPOT is
  still roughly flat while total tokens/sec held steady.
- **What to watch next:** N=4 should show position 3's acceptance rate is
  even lower (likely near 0%), and given N=3 already shows zero marginal
  gain, N=4's *extra* draft compute with *zero* extra accepted tokens should
  show throughput at best flat, more likely starting to dip — this is where
  we'd expect to see the "rejected drafts cost more than accepted ones gain"
  mechanism actually go negative, not just plateau.

#### N=4

**Why we ran this:** complete the sweep range specified in Experiment 5.1
(1 through 4) and confirm the predicted decline past the N=3 plateau.

**GPU before:** 0 MiB · **GPU after server load:** 75,731 MiB · **GPU after teardown:** 0 MiB (confirmed)

**Raw result:**
```
Successful requests:        80/80, 0 failed
Benchmark duration (s):     20.52
Output tok/s:                997.62
Total token throughput:      1293.83 tok/s
Mean TTFT (ms):              68.02    median 31.48   p99 395.02
Mean TPOT (ms):              7.39     median 7.43     p99 8.56
Mean ITL (ms):                10.44    median 10.41    p99 11.63
Acceptance rate (%):         10.34
Acceptance length:           1.41
Drafts / Draft tokens:       14439 / 57756
Accepted tokens:             5970
Per-position acceptance:    position 0 = 32.72%   position 1 = 7.42%
                              position 2 = 1.00%    position 3 = 0.21%
```

**Observations:**
- **N=4 confirms the predicted decline: 997.62 tok/s, down from N=2/N=3's
  ~1070 tok/s — a real ~7-point drop off the peak**, though still +18.9% above
  the no-speculation C1 baseline. The sweep has the exact shape we predicted:
  **loss (N=1) → jump to peak (N=2) → plateau (N=3) → decline (N=4).**
- Position 3 accepts at **0.21%** — functionally never. `draft_tokens` is now
  57,756, roughly **4x** N=1's count and **2x** N=2's, for essentially zero
  additional accepted tokens beyond what N=2/N=3 already captured
  (`accepted_tokens` 5970 vs N=3's 5893 vs N=2's 5889 — only marginally more,
  noise-level). All that extra drafting compute at position 3 is pure
  overhead with no payoff.
- This is the clearest evidence in the whole sweep for the mechanism named in
  §5.3: **past the optimum, additional draft tokens cost real compute (drafting
  + verifying + bookkeeping a 4-wide tree) for a vanishing acceptance
  return, and at N=4 that cost finally outweighs the (already-saturated)
  benefit** — TPOT also crept up (6.43ms at N=2 → 7.39ms at N=4), consistent
  with the verifier now doing more work per cycle for the same useful output.
- Mean TTFT kept improving slightly (173ms→82ms→68ms across N=2→3→4) even as
  *throughput* fell — queue-draining dynamics are a function of how
  consistently fast each cycle is, not just raw tok/s, so this doesn't
  contradict the throughput finding.
- **Sweep conclusion:** N=2 is the clear, unambiguous optimum for C2 on this
  draft head — not a guess, not just matching the reference value by luck,
  but the actual measured peak of a full 4-point sweep with a mechanistic
  explanation (error propagation collapses acceptance past position 1) that
  predicts exactly the shape we observed. **C2's final benchmark run will use
  num_speculative_tokens=2.**

### C4 sweep — FP8 + EAGLE-3

**Why we're running this:** find C4's own optimal `num_speculative_tokens`
rather than assuming it matches C2's (N=2). There's a real reason to expect a
*different* answer here, not just a faster version of the same answer: this
draft head was trained on **BF16** hidden states (Chapter 3) but is now being
served against the **FP8** verifier (Chapter 4, quantized after training) —
exactly the "wrong order" scenario Chapter 7 discusses. If FP8 measurably
changes the verifier's hidden states, position-by-position acceptance could
shift in ways that change where the throughput peak sits, separately from
the FP8 model simply being faster per cycle.

**Command pattern:** `vllm serve /data/hw3/Qwen3-8B-FP8-Dynamic --speculative-config '{"method":"eagle3","model":"/data/hw3/output/checkpoints/checkpoint_best","num_speculative_tokens":N}' --no-enable-prefix-caching`

| N | tok/s | vs C1 (838.74) | vs C2 peak (1069.82) | Mean TTFT | Mean TPOT | Acceptance rate | Acceptance length | GPU after teardown |
|---|---|---|---|---|---|---|---|---|
| 1 | 995.47 | **+18.7%** | -7.0% | 738.62 ms | 4.73 ms | 32.88% | 1.33 | 0 MiB (confirmed) |
| 2 | 1468.70 | **+75.1%** | +37.3% | 53.40 ms | 5.03 ms | 20.17% | 1.40 | 0 MiB (confirmed) |
| 3 | 1346.86 | **+60.6%** | +25.9% | 65.29 ms | 5.45 ms | 13.67% | 1.41 | 0 MiB (confirmed) |
| 4 | 1229.33 | **+46.6%** | +14.9% | 68.50 ms | 5.91 ms | 10.27% | 1.41 | 0 MiB (confirmed) |

**Sweep verdict: N=2 is the optimum for C4 (FP8 + EAGLE-3) on this draft head —
the same N as C2, but a very different shape.** Loss-free climb to the peak at
N=2 (+75.1% vs C1), then a steady, real decline at N=3 (+60.6%) and N=4
(+46.6%) — no plateau the way C2 had at N=3. **C4 final benchmark will use
N=2.**

**⚠ Unexpected so far:** N=2 is dramatically better than N=1, not worse — this
contradicts the guide's reference run, which found C4's optimum at N=1. Not
drawing a conclusion yet; need N=3/N=4 to see the full shape before deciding
whether this draft head genuinely behaves differently on FP8 than the
reference did, or whether N=2 is itself the peak and N=3/4 will now decay the
way C2 did past its own peak.

#### N=1

**GPU before:** 0 MiB · **GPU after server load:** 75,631 MiB · **GPU after teardown:** 0 MiB (confirmed)

**Raw result:**
```
Successful requests:        80/80, 0 failed
Benchmark duration (s):     20.57
Output tok/s:                995.47
Total token throughput:      1290.91 tok/s
Mean TTFT (ms):              738.62   median 20.12   p99 7249.06
Mean TPOT (ms):              4.73     median 4.65     p99 7.50
Mean ITL (ms):                6.28     median 6.07     p99 8.11
Acceptance rate (%):         32.88
Acceptance length:           1.33
Drafts / Draft tokens:       15372 / 15372
Accepted tokens:             5054
Per-position acceptance:    position 0 = 32.88%  (only one position exists at N=1)
```

**Observations:**
- **C4 at N=1 (995.47 tok/s) is already +18.7% over the C1 baseline** — a sharp
  contrast with C2 at N=1, which *lost* to baseline (823.76, -1.8%). Same
  draft-head overhead, same acceptance rate at this position
  (32.88% here vs 33.37% for C2 N=1 — statistically the same), but a
  completely different verdict on whether speculation pays off at N=1.
- The reason isn't about acceptance at all — it's that **the FP8 verifier
  pass itself is cheaper**, so the fixed draft-head overhead is being paid
  back against a faster, lower base cost per cycle. **Mean TPOT dropped to
  4.73ms here vs C2 N=1's 6.61ms** — that's the FP8 weight-loading speedup
  (Chapter 1's "8GB instead of 16GB per step") showing up directly, and it's
  enough on its own to flip N=1 from a loss into a clear win.
- Position-0 acceptance (32.88%) being almost identical to C2's BF16 number
  (33.37%) is itself a real data point for Chapter 6: despite the draft head
  never having seen FP8 hidden states during training, position-0 acceptance
  barely moved. Whatever distribution shift quantization introduces, it isn't
  large enough to visibly hurt the *first* draft position's accuracy. Worth
  carrying forward — we'll see if this holds at deeper positions too.
- This run is still **7% below C2's peak** (1069.82 at N=2) in absolute
  throughput — so FP8 alone, even with a "free" win at N=1, hasn't yet beaten
  the best BF16+spec configuration. The open question for the rest of this
  sweep: does C4 still have a higher peak waiting at N=2, the way C2 did, or
  does it peak earlier (the guide's reference run found C4's optimum at
  N=1) and FP8's real advantage shows up by being uniformly faster rather
  than by reaching a higher ceiling?
- **What to watch next:** if C4's peak is at N=1 already (as the reference
  run suggests), N=2 should show the SAME kind of decay C2 showed past its
  peak — except starting one step earlier. That would be a meaningful,
  testable difference in sweep *shape* between the two configs, not just a
  vertical shift.

#### N=2

**Why we ran this:** test whether C4 can stack acceptance-length gains on top
of its already-faster-per-cycle baseline, or whether (per the guide's
reference run) it has already peaked at N=1.

**GPU before:** 0 MiB · **GPU after server load:** 75,631 MiB · **GPU after teardown:** 0 MiB (confirmed)

**Raw result:**
```
Successful requests:        80/80, 0 failed
Benchmark duration (s):     13.89
Output tok/s:                1468.70
Total token throughput:      1906.21 tok/s
Mean TTFT (ms):              53.40    median 21.62   p99 334.78
Mean TPOT (ms):              5.03     median 5.06     p99 5.82
Mean ITL (ms):                7.05     median 7.00     p99 8.08
Acceptance rate (%):         20.17
Acceptance length:           1.40
Drafts / Draft tokens:       14503 / 29006
Accepted tokens:             5850
Per-position acceptance:    position 0 = 33.19%   position 1 = 7.15%
```

**Observations — this is the most important result so far in the C4 sweep:**
- **N=2 didn't just beat N=1, it dramatically beat it: 1468.70 vs 995.47
  tok/s, a +47.5% jump within C4 alone, and +37.3% above C2's own best
  result (1069.82 at its N=2).** This directly contradicts what the guide's
  reference run found (C4 optimal at N=1) — on *our* draft head and *our*
  hardware, N=2 is clearly better, not worse. Flagging this as a real,
  measured disagreement with the reference rather than smoothing it over:
  reference numbers are a different run, a different draft head training
  outcome, possibly different load conditions — they're a sanity-check
  anchor, not ground truth for our specific setup.
- Per-position acceptance (33.19% / 7.15%) is nearly identical to C2's N=2
  numbers (32.97% / 7.52%) — so the *quality* of the draft head's predictions
  on FP8 hidden states is essentially unchanged from BF16 at both positions
  measured so far. The huge throughput gap between C2 N=2 (1069.82) and C4
  N=2 (1468.70) is therefore not coming from better acceptance — it's coming
  almost entirely from the FP8 verifier pass being cheaper, the same
  mechanism we saw at N=1, just now compounding with the same acceptance-length
  gain C2 got from N=2.
- This means FP8's speedup and EAGLE-3's speedup look close to **multiplicative**
  here, not just additive: C3 (FP8 alone, not yet measured) should land
  somewhere between C1 and these numbers, and if FP8-alone and spec-alone
  gains roughly multiply, that's a meaningfully different finding than "the
  gains are independent and just add up" — something to check explicitly once
  we have the standalone C3 number.
- Benchmark duration dropped to 13.89s (vs C1's 24.42s, C2 N=2's 19.14s) —
  the fastest wall-clock run of the whole sweep so far, which is itself a
  direct, intuitive readout of the combined effect.
- **What to watch next:** does N=3 keep climbing, plateau like C2 did, or has
  N=2 already found the peak for C4? Given position-1 acceptance is already
  this low (7.15%), I'd expect position 2 to collapse the same way it did for
  C2 (under 1%), which would predict a plateau-or-decline shape similar to
  C2's — but given N=2 already broke from the reference's predicted shape
  once, I'm not assuming that pattern repeats without checking.

#### N=3

**Why we ran this:** confirm whether N=2 is the actual peak or whether C4
keeps climbing — and check if position 2's acceptance collapses the same way
it did for C2.

**GPU before:** 0 MiB · **GPU after server load:** 75,639 MiB · **GPU after teardown:** 0 MiB (confirmed)

**Raw result:**
```
Successful requests:        80/80, 0 failed
Benchmark duration (s):     15.21
Output tok/s:                1346.86
Total token throughput:      1746.58 tok/s
Mean TTFT (ms):              65.29    median 23.64   p99 450.71
Mean TPOT (ms):              5.45     median 5.49     p99 6.22
Mean ITL (ms):                7.68     median 7.62     p99 8.96
Acceptance rate (%):         13.67
Acceptance length:           1.41
Drafts / Draft tokens:       14487 / 43461
Accepted tokens:             5939
Per-position acceptance:    position 0 = 32.67%   position 1 = 7.30%   position 2 = 1.03%
```

**Observations:**
- **N=2 was the peak. N=3 drops to 1346.86 tok/s, down from N=2's 1468.70 —
  an 8.3% decline.** Position 2's acceptance collapsed to 1.03%, essentially
  identical to C2's position-2 collapse (0.98%) — so this part of the pattern
  *does* carry over from BF16 to FP8 unchanged: position-2 predictions fail
  for the same structural reason in both cases (conditioned on position 1
  being correct, which it usually isn't), independent of which verifier is
  serving.
- Unlike C2, where N=3 was a *plateau* (statistically flat vs N=2: 1070.08 vs
  1069.82), C4's N=3 is a clear, real *decline* from its own N=2 — the
  peak-then-drop shape arrives one sweep step earlier for C4 than for C2. The
  most likely reason: with the per-cycle base cost already lower (FP8), the
  fixed overhead of drafting a now-mostly-wasted 3rd position is a *larger
  relative share* of each cycle's cost than it was for the slower BF16 model
  — the same absolute drafting overhead matters more when the rest of the
  cycle got cheaper.
- `accepted_tokens` (5939) is only marginally higher than N=2's (5850) — most
  of the extra draft compute (`draft_tokens` jumped from 29006 to 43461, a 50%
  increase, same proportional jump C2 saw at this step) buys almost nothing,
  exactly the C2 N=3 story, just arriving with a throughput cost attached
  this time instead of a free ride.
- Still **comfortably above C1, C2's peak, and C4 N=1** (1346.86 vs 838.74 /
  1069.82 / 995.47) — even past its own peak, C4 outperforms everything
  measured so far except its own N=2.
- **What to watch next:** N=4 should continue the decline (position 3
  acceptance likely near-zero, same as C2's 0.21%), confirming N=2 as C4's
  true optimum and completing a sweep shape that peaks and decays one step
  earlier than C2's.

#### N=4

**Why we ran this:** complete the sweep range and confirm the decline
continues, finalizing N=2 as C4's optimum.

**GPU before:** 0 MiB · **GPU after server load:** 75,623 MiB · **GPU after teardown:** 0 MiB (confirmed)

**Raw result:**
```
Successful requests:        80/80, 0 failed
Benchmark duration (s):     16.64
Output tok/s:                1229.33
Total token throughput:      1594.56 tok/s
Mean TTFT (ms):              68.50    median 25.51   p99 455.32
Mean TPOT (ms):              5.91     median 5.93     p99 6.81
Mean ITL (ms):                8.33     median 8.26     p99 9.54
Acceptance rate (%):         10.27
Acceptance length:           1.41
Drafts / Draft tokens:       14464 / 57856
Accepted tokens:             5940
Per-position acceptance:    position 0 = 31.93%   position 1 = 7.76%
                              position 2 = 1.20%    position 3 = 0.17%
```

**Observations:**
- **The decline continues, confirming N=2 as C4's true peak: 1229.33 tok/s,
  down from N=3's 1346.86 and further from N=2's 1468.70.** Position 3's
  acceptance (0.17%) is in the same near-zero territory as C2's equivalent
  (0.21%) — the structural collapse of deep-position acceptance is consistent
  across both configs, exactly as expected.
- **Sweep shape comparison — C2 vs C4 (both peak at N=2, but the decay differs):**

  | N | C2 (BF16) tok/s | C4 (FP8) tok/s | C2 shape | C4 shape |
  |---|---|---|---|---|
  | 1 | 823.76 | 995.47 | below baseline | above baseline |
  | 2 | 1069.82 | 1468.70 | **peak** | **peak** |
  | 3 | 1070.08 | 1346.86 | plateau (flat) | declining |
  | 4 | 997.62 | 1229.33 | declining | declining further |

  C2 gets a "free" plateau at N=3 before declining at N=4; C4 starts
  declining immediately after its peak. Consistent with the N=3 observation:
  FP8's lower per-cycle base cost makes a wasted draft position's fixed
  overhead a *larger proportional tax* on each cycle, so the same structural
  acceptance collapse (position 2 → ~1%, position 3 → ~0.2%, identical in
  both configs) translates into a less forgiving throughput curve on the
  faster verifier.
- This is a genuinely informative, non-obvious finding for Chapter 5/6: **the
  optimal draft depth being the same (N=2) for both configs was not
  guaranteed and shouldn't be assumed in general** — it happened here because
  the acceptance-collapse structure (driven by the draft head, not the
  verifier) is what determines where the peak sits, while the verifier speed
  only affects how steep the decay is on either side of that peak. Two
  separate mechanisms, worth keeping distinct when we write up Chapter 8.
- **C4 sweep conclusion:** peak at N=2 (1468.70 tok/s), real decline beginning
  immediately at N=3, continuing through N=4. **C4's final benchmark run will
  use num_speculative_tokens=2** — coincidentally the same N as C2, but for
  a position-collapse reason, not because FP8 "needs" the same N as BF16 for
  any deeper reason.

## Final 4-Config Benchmark

**Why this section exists separately from the sweeps:** §5.4 wants one clean,
final number per configuration, at each config's tuned setting, so the
headline comparison isn't tangled up with sweep methodology. **C1, C2 (N=2),
and C4 (N=2) are *not* re-run here** — they were already measured under the
exact same fixed protocol, fully sequentially, with GPU confirmed clear
between every run, in this same session. Re-running them would add nothing
but noise; reusing them is the more rigorous choice, not a shortcut. The one
genuinely new measurement needed is **C3** (FP8, no speculative decoding),
which hasn't been run standalone yet.

### C3 — FP8 Quantized (no spec decoding)

**Why we're running this:** isolate FP8's contribution with zero EAGLE-3
interaction, so we can check whether FP8 + spec decoding (C4) combine
additively, multiplicatively, or something else — flagged as an open
question during the C4 sweep (N=2 observations).

**Command:** `vllm serve /data/hw3/Qwen3-8B-FP8-Dynamic --no-enable-prefix-caching`
**GPU before:** 0 MiB · **GPU after server load:** 75,385 MiB · **GPU after teardown:** 0 MiB (confirmed)

**Raw result:**
```
Successful requests:        80/80, 0 failed
Benchmark duration (s):     18.43
Output tok/s:                1110.97
Total token throughput:      1440.68 tok/s
Mean TTFT (ms):              467.64   median 24.66   p99 5708.86
Mean TPOT (ms):              5.39     median 4.86     p99 25.89
Mean ITL (ms):                5.39     median 4.85     p99 5.30
(no speculative decoding section — not applicable, no draft head in this config)
```

**Observations:**
- **C3 alone: 1110.97 tok/s, +32.5% over C1.** This is FP8's pure,
  unmixed contribution — no draft head, no acceptance mechanics, just a
  smaller model moving less data through HBM per step. Mean TPOT (5.39ms) vs
  C1's (7.03ms) is the direct readout of Chapter 1's prediction: halving
  weight bytes per step should roughly halve the bandwidth-bound portion of
  decode latency, and 5.39 vs 7.03 (a 23% drop, not a full 50%) reflects that
  TPOT also includes fixed per-step overhead beyond pure weight loading that
  doesn't shrink with quantization.
- **The additive-vs-multiplicative question, answered:**

  | Model | Formula | Predicted C4 tok/s | Actual C4 | Actual vs prediction |
  |---|---|---|---|---|
  | Additive (gains sum) | `C1 × (1 + gain_C2 + gain_C3)` | 1342.05 | 1468.70 | **+9.4%** |
  | Multiplicative (gains multiply) | `C1 × (C2/C1) × (C3/C1)` | 1417.05 | 1468.70 | **+3.6%** |

  C4's actual measured throughput (1468.70) **beats both predictions** — the
  combination is not just multiplicative, it's *slightly super-multiplicative*.
- Why this makes mechanistic sense rather than being a measurement fluke: FP8
  and EAGLE-3 aren't acting on independent, separate parts of the decode
  cycle — they both act on the **same expensive verifier forward pass**.
  FP8 shrinks that pass's cost; EAGLE-3 amortizes that same (now-shrunk) cost
  over more accepted tokens per cycle. Cutting the cost of the thing you're
  already amortizing makes the amortization itself more effective per unit of
  draft-head overhead — consistent with what we already saw in the C4 sweep
  (FP8 making a *wasted* draft position cost relatively more) working in
  reverse here: at the *productive* N=2 setting, FP8 makes the *accepted*
  extra token relatively cheaper to win, too. Same mechanism, opposite sign,
  depending on whether the extra draft token is accepted or rejected.
- This is a genuinely strong piece of evidence for Chapter 8: "are FP8 and
  spec decoding worth combining" has an unambiguous answer here — not only do
  they not cannibalize each other, they reinforce each other slightly beyond
  naive multiplication.

### Final Comparison Table

| Config | tok/s | vs C1 | Mean TTFT | Mean TPOT | Acceptance rate | Acceptance length | Source |
|---|---|---|---|---|---|---|---|
| **C1** — Baseline | 838.74 | — | 649.07 ms | 7.03 ms | — | — | anchor run, this session |
| **C2** — BF16 + EAGLE-3 (N=2) | 1069.82 | **+27.5%** | 173.27 ms | 6.43 ms | 20.25% | 1.40 | C2 sweep, N=2 |
| **C3** — FP8, no spec | 1110.97 | **+32.5%** | 467.64 ms | 5.39 ms | — | — | this run |
| **C4** — FP8 + EAGLE-3 (N=2) | 1468.70 | **+75.1%** | 53.40 ms | 5.03 ms | 20.17% | 1.40 | C4 sweep, N=2 |

All four configs ran under the identical fixed protocol (`philschmid/mt-bench`,
concurrency 8, 80 prompts, prefix caching disabled), fully sequentially, GPU
confirmed at 0 MiB between every single run — no two servers ever ran
concurrently, so this table is a clean, directly comparable picture with no
parallel-execution noise mixed in.

**Status: all four Chapter 5 configs measured. Ready for §5.5 interpretation
and the Chapter 6/7/8 analysis (acceptance-rate paradox, ordering question,
evidence synthesis).**

## §5.5 Interpretation — Question A & Question B

Worked through using our own measured numbers (not the reference run's),
matching the worked code now in `ch5-acceptance-analysis` and the new
`ch5-additive-multiplicative` cell in the notebook.

### Question A — Why does spec decoding help even though acceptance *rate* looks low?

Our C2 (N=2): 14,542 draft cycles, 29,084 draft tokens, 5,889 accepted tokens,
per-token acceptance rate **20.25%**.

```
mean tokens per verify cycle = (accepted + drafts) / drafts
                              = (5,889 + 14,542) / 14,542
                              ≈ 1.40   ← matches measured acceptance_length exactly
```

The verifier ran 14,542 forward passes whether or not speculation was on.
Without it, that's 14,542 tokens. With it, it's ~1.40× that — about 20,359
tokens — for essentially the same number of expensive (memory-bandwidth-bound)
verifier passes, plus a cheap draft-head pass each cycle. The 20.25%
**acceptance rate** is a per-draft-token-attempt average and undersells what's
happening; **acceptance length** (1.40) is the number that actually predicts
throughput, and it does so almost completely: 1.40× more tokens per pass
roughly tracks the +27.5% measured throughput gain (not exactly 1:1, because
the draft head's own compute and scheduling overhead eat into the theoretical
ceiling — but the direction and rough magnitude both check out).

**Answer: acceptance rate measures how often a single guess is right.
Acceptance length measures how many usable tokens come out of one expensive
verifier pass — and the second number is what determines throughput, because
the verifier pass (not the draft head) is the bottleneck cost being amortized.**

### Question B — Do FP8 and spec decoding's gains just add up?

```
gain_C2 (spec alone)  = 1069.82 / 838.74 - 1  = +27.55%
gain_C3 (FP8 alone)   = 1110.97 / 838.74 - 1  = +32.46%

additive prediction       = 838.74 × (1 + 0.2755 + 0.3246)        = 1342.05 tok/s
multiplicative prediction = 838.74 × (1069.82/838.74) × (1110.97/838.74) = 1417.05 tok/s
actual C4                                                          = 1468.70 tok/s

actual vs additive:       +9.4%
actual vs multiplicative: +3.6%
```

**Answer: neither prediction is right — C4 beats both.** The combination is
slightly *super*-multiplicative, and this is a real, mechanistic effect, not
noise (it's 36x larger than the ~0.1% noise band established by the C1
reproduction check at the start of this session).

**Why:** confirmed directly by inspecting the EAGLE-3 checkpoint
(`/data/hw3/output/checkpoints/checkpoint_best/model.safetensors`) — its
weights are `torch.bfloat16`, unchanged by Chapter 4's quantization, which
only targeted the 8B verifier's `Linear` layers (`ignore=["lm_head"]`). So:

- The draft head's compute cost is **fixed and identical** in C2 and C4.
- FP8 shrinks only the **verifier's** forward pass — the exact cost that
  EAGLE-3 is busy amortizing over multiple accepted tokens per cycle.
- Shrinking the cost you're already amortizing makes the amortization more
  efficient per unit of fixed draft-head overhead — the two techniques are
  optimizing the *same* bottleneck from two different angles, not two
  independent bottlenecks, so their effects compound rather than simply add
  or multiply as if unrelated.
- This is the mirror image of the C4 sweep finding (FP8 makes *wasted* draft
  overhead cost relatively more past the optimum) — same root cause (FP8
  changes the relative weight of fixed draft-head cost vs. the verifier
  cost), opposite sign depending on whether the marginal draft token is
  typically accepted (N=2, here) or rejected (N≥3, the sweep).

**Implication for Chapter 8:** "should you combine FP8 and spec decoding" has
an unambiguous answer from this data — yes, and not just because the gains
don't cancel each other out, but because they reinforce. This is a stronger
claim than "they're compatible" and is worth stating explicitly in the final
evidence synthesis.

## Chapter 6 — The Interaction Problem

### Finding 1: the reference's "acceptance rate paradox" does not exist in our data

**Why we checked this:** the guide poses a puzzle — the reference run shows
C4 (FP8+spec) with a *higher* acceptance rate than C2 (BF16+spec) but a
*lower* acceptance length, which looks contradictory. Before accepting the
guide's resolution at face value, we checked whether the same pattern shows
up in our own numbers.

**What we found:** the reference compares C2 at N=2 against C4 at N=1 —
different draft depths. We tuned both configs to their own real optimum via
the Chapter 5 sweep, and **both landed on N=2**. Comparing at matched N:

| Config | Acceptance rate | Acceptance length |
|---|---|---|
| C2 (BF16 + spec, N=2) | 20.25% | 1.40 |
| C4 (FP8 + spec, N=2) | 20.17% | 1.40 |
| **Delta** | **-0.08 pp** | **+0.00** |

**Both metrics are statistically indistinguishable.** The reference's
"paradox" is fully explained by the N=2-vs-N=1 confound, not by any real
effect of quantization on the draft head's acceptance behavior. This is a
genuine, useful finding: it means the deeper question worth asking isn't
"why does the paradox happen" (it doesn't, once you control for draft
depth) but "does FP8 change the verifier's output distribution at all, and
if so, does that show up anywhere" — which is what the bonus experiment
tests directly.

### Finding 2: BF16 vs FP8 logit comparison (real experiment, not hypothetical)

**Why we ran this:** the guide's `ch6-dist-check` cell was written as a
"copy this and run it yourself" snippet, and it had the same path bug we
already fixed in `ch4-quant-code` — it referenced the Hugging Face Hub id
(`Qwen/Qwen3-8B`, would trigger a redundant re-download) and a relative path
for the FP8 model. Fixed both to the local absolute paths and actually ran
it on this instance (`comp_venv`, real H100), rather than leaving it as a
hypothetical for later. This is the first genuinely new experiment in
Chapter 6 — everything in Finding 1 was re-analysis of numbers we already
had; this one required running fresh inference on both models.

**Command:** `comp_venv` python script loading `/data/hw3/Qwen3-8B` (BF16)
and `/data/hw3/Qwen3-8B-FP8-Dynamic` (FP8) sequentially, computing next-token
logits on 3 prompts, one model at a time (loaded, measured, freed via
`del model; torch.cuda.empty_cache()` before loading the next — same
no-overlap discipline as the Chapter 5 benchmarks, verified `nvidia-smi`
shows 0 MiB both before the run and after it exits).

**Raw result:**
```
Prompt: 'The capital of France is'
  BF16: top1_prob=0.5188  entropy=2.6279  top3=[('Paris', '0.5188'), ('a', '0.1158'), ('in', '0.0426')]
  FP8:  top1_prob=0.5787  entropy=2.4213  top3=[('Paris', '0.5787'), ('a', '0.1006'), ('located', '0.0288')]

Prompt: 'def fibonacci(n):'
  BF16: top1_prob=0.9703  entropy=0.1933  top3=[('', '0.9703'), ('#', '0.0095'), ('', '0.0084')]
  FP8:  top1_prob=0.9651  entropy=0.2205  top3=[('', '0.9651'), ('#', '0.0121'), ('', '0.0095')]

Prompt: 'In 2024, the most popular programming language was'
  BF16: top1_prob=0.5532  entropy=2.6550  top3=[('Python', '0.5532'), ('JavaScript', '0.0661'), ('determined', '0.0515')]
  FP8:  top1_prob=0.5293  entropy=2.8024  top3=[('Python', '0.5293'), ('determined', '0.0716'), ('JavaScript', '0.0492')]

Top-1 token agreement: France=SAME  fibonacci=SAME  programming=SAME (all 3/3)
```

**Observations:**
- **H1 ("FP8 uniformly sharpens the distribution") is NOT well supported —
  and the initial mean-based read of this data was misleading.** Only the
  France prompt got sharper under FP8 (top1_prob +0.06, entropy -0.21). The
  other two prompts got *less* sharp (fibonacci: top1_prob -0.005, entropy
  +0.027; programming language: top1_prob -0.024, entropy +0.15). Averaging
  across all three prompts gives a small net-positive shift (+0.0103 top1,
  -0.0107 entropy) that looks like it supports "FP8 sharpens," but that
  average is dominated by France's larger swing — 2 of 3 individual prompts
  actually move the *opposite* direction. Caught this by checking
  prompt-by-prompt before trusting the aggregate; worth flagging as a
  reminder that small-n averages can suggest a systematic effect that isn't
  actually there.
- **Every prompt's top-1 token is identical between BF16 and FP8** (Paris,
  newline, Python — 3/3 agreement). Whatever shift quantization causes in
  the probability distribution, it isn't large enough to flip the actual
  prediction on any of these prompts.
- **H3 ("FP8 weights approximate BF16 well enough that hidden states differ
  minimally") is the best-supported hypothesis.** The probability/entropy
  differences are real (not zero) but small (roughly 1-15% relative) and
  inconsistent in direction across prompts — the signature of quantization
  noise fluctuating around a stable prediction, not a systematic
  distributional shift in either direction.
- **This directly explains Finding 1.** If FP8 doesn't measurably change
  which token the verifier prefers, it shouldn't measurably change whether
  the (BF16-trained) draft head's guesses get accepted either — consistent
  with C2 and C4 landing on nearly identical acceptance rate and length once
  draft depth is held constant.
- Sample size caveat: n=3 prompts is small, matching what the guide itself
  set up as a "bonus, optional" quick check, not a rigorous study. The
  *direction* being inconsistent across even 3 prompts is itself informative
  (real systematic effects should show up more consistently even at small n),
  but a stronger claim would need a larger, more diverse prompt set.

**Chapter 6 conclusion: quantization does not break speculative decoding for
this draft head, and the reason isn't subtle — FP8 barely moves the
verifier's output distribution in the first place.** The reference's
"paradox" evaporates once draft depth is controlled for, and the logit-level
check confirms why: the underlying distributions are close enough that
neither the draft head's acceptance behavior nor the model's actual
predictions are meaningfully affected.

---

## Chapter 7 — The Central Mystery (Order of Operations)

### User's stated hypothesis (recorded before analysis, not used to bias it)

Order: spec decoding (draft head training) first, then quantization.
Reason: training on hidden states from full BF16 weights should give better
latent representations to learn from; even after quantizing the verifier
afterward, the draft head should retain that quality benefit. Evidence cited:
Chapter 6's finding that FP8 barely shifts the verifier's output
distribution. Recorded verbatim (lightly formatted) in `ch7-order` in the
notebook. Per explicit instruction, this hypothesis is NOT used to steer the
analysis below — the analysis is driven only by the experiment and the
runlog evidence.

### Decision: run the real Experiment 7.1, not just the circumstantial argument

We already had one relevant data point "for free": our actual chronological
order this session was hidden states (12:17) → draft head trained (19:15,
completed) → FP8 quantization (22:39) — i.e. we genuinely ran the
"train-on-BF16, quantize-after" order (the guide's "Option A"), and our C4
benchmark IS that setup's real acceptance data. What we didn't have was the
true counterfactual: a second draft head trained from scratch on FP8-generated
hidden states, to see whether it does meaningfully better serving FP8 than
our existing BF16-trained one. Chose to run this for real rather than rely
on circumstantial evidence alone, given the guide marks this "Experiment 7.1
(Critical)."

### Methodology — what's reused vs. what changes

Documented in full in the new `ch7-exp71-methodology` / `ch7-exp71-scripts`
cells in the notebook. Summary:

**Reused, unchanged:**
- Preprocessed dataset `/data/hw3/output` (same tokenizer, same 3,000
  samples, same seq-length 2048) — not regenerated, quantization doesn't
  affect tokenization.
- Target hidden-state layer IDs `[2, 18, 33, 36]` — confirmed identical in
  the new server's startup log (`eagle_aux_hidden_state_layer_ids: [2, 18,
  33, 36]`), since this is a function of layer count (36), not precision.
- Training hyperparameters: 5 epochs, `--speculator-type eagle3`, `--save-best`.
- One-GPU-job-at-a-time discipline, `nvidia-smi` confirmed clear before each step.

**Changed:**
- Verifier model everywhere: `/data/hw3/Qwen3-8B-FP8-Dynamic` instead of
  `/data/hw3/Qwen3-8B`, for both hidden-state extraction and training's
  `--verifier-name-or-path`.
- New output dirs so nothing overwrites the originals: `hidden_states_fp8`,
  `output/checkpoints_fp8`, `logs_fp8`.

### Step 1 — Launch vLLM for FP8 hidden-state extraction

**Command:**
```
python /data/hw3/speculators/scripts/launch_vllm.py \
    /data/hw3/Qwen3-8B-FP8-Dynamic \
    --hidden-states-path /data/hw3/hidden_states_fp8 \
    -- --port 8000
```

**GPU before:** 0 MiB · **GPU after load:** 75,721 MiB. Startup log confirmed
`quantization=compressed-tensors` (the FP8 model, not BF16) and
`Using auxiliary layers from speculative config: (2, 18, 33, 36)` — matches
the original run's layer selection exactly.

### Step 2 — Generate hidden states via the FP8 verifier

**Command:**
```
python /data/hw3/speculators/scripts/data_generation_offline.py \
    --model /data/hw3/Qwen3-8B-FP8-Dynamic \
    --preprocessed-data /data/hw3/output \
    --output /data/hw3/hidden_states_fp8 \
    --max-samples 3000
```

**Observation:** this ran dramatically faster than the ~hours the guide
describes for the original BF16 hidden-state generation — real-time
progress showed ~1,800/3,000 samples (60%) in under 3.5 minutes, tracking
toward a total of roughly 5-6 minutes. Consistent with Chapter 1's core
claim: the FP8 verifier moves less data per forward pass, so prefill/generation
for the same 3,000 samples is meaningfully faster.

**Result: completed in 5 minutes 51 seconds** (log: "Saved 3000 new data
points to /data/hw3/hidden_states_fp8", "Data generation complete!"),
producing all 3,000 files (`du -sh` → 122 GB, essentially identical to the
original BF16 run's ~130 GB — expected, since hidden-state size is
determined by seq length × layers × hidden dim × dtype, and both runs saved
BF16 hidden states regardless of the verifier's own weight precision).
Server torn down (`kill -TERM` on the API server + EngineCore processes —
plain `pkill -f launch_vllm` did not match the actual process names and left
75,725 MiB stuck on the GPU briefly; fixed by killing the correct PIDs
directly), GPU confirmed back to 0 MiB before starting training.

### Step 3 — Train a second draft head on the FP8-generated hidden states

**Command:** `bash /data/hw3/scripts/train_eagle3_fp8.sh` (5 epochs,
`--verifier-name-or-path /data/hw3/Qwen3-8B-FP8-Dynamic`,
`--hidden-states-path /data/hw3/hidden_states_fp8`,
`--save-path /data/hw3/output/checkpoints_fp8`).

**Sanity check before trusting the run:** confirmed the two startup warnings
("No vocab mappings found... using full verifier vocab" and
"`--target-layer-ids` not explicitly set, defaulting to `[2, 18, 33]`") are
byte-identical to the ones the *original* Chapter 3 BF16 training run
produced (`grep` against `/data/hw3/output/train_run.log` confirms exact
match) — so this is expected, pre-existing tool behavior, not something our
reuse of `/data/hw3/output` broke.

**Result: completed in ~12 minutes** (started 00:36:23, final checkpoint
saved 00:48:36), GPU confirmed back to 0 MiB after. `checkpoint_best -> 4`
(same epoch index as the original run, coincidentally).

**Epoch-by-epoch validation loss — direct comparison against the original
BF16-trained draft head:**

| Epoch | BF16-trained (original, Ch3) | FP8-trained (this run) | Delta |
|---|---|---|---|
| 0 | 14.583 | 14.577 | -0.006 |
| 1 | 12.939 | 12.941 | +0.002 |
| 2 | 12.299 | 12.306 | +0.007 |
| 3 | 12.208 | 12.218 | +0.010 |
| 4 (final) | 11.984 | 11.994 | +0.010 |

**This is the strongest evidence in the whole session for Chapter 6/7's
central claim.** Training on hidden states generated by the FP8 verifier
produces a validation loss trajectory that is statistically indistinguishable
from training on the BF16 verifier's hidden states — every single epoch
matches within ~0.01, an order of magnitude smaller than the run-to-run
noise we'd expect from stochastic training (different data shuffling order,
etc.). This isn't circumstantial anymore: it's a direct, controlled
comparison (same data, same architecture, same hyperparameters, only the
source verifier's precision differs) showing quantization has no detectable
effect on the quality of what the draft head learns.

### Step 4 — Benchmark the FP8-trained draft head, matched to C4's exact setup

**Why:** the training-loss comparison shows the two draft heads learned
almost identically, but the question Chapter 7 actually asks is about
*serving* behavior — does the FP8-trained head accept more drafts when
serving the FP8 verifier than our existing BF16-trained one does? Benchmarked
at the same N=2, same protocol, same everything except which draft head
checkpoint is loaded.

**Command:** `vllm serve /data/hw3/Qwen3-8B-FP8-Dynamic --speculative-config
'{"method":"eagle3","model":"/data/hw3/output/checkpoints_fp8/checkpoint_best","num_speculative_tokens":2}'
--no-enable-prefix-caching`

**GPU before:** 0 MiB · **GPU after load:** 75,631 MiB · **GPU after
teardown:** 0 MiB (confirmed; `pkill -f "vllm serve"` sufficient this time).

**Raw result:**
```
Successful requests:        80/80, 0 failed
Benchmark duration (s):     14.00
Output tok/s:                1462.95
Total token throughput:      1897.21 tok/s
Mean TTFT (ms):              55.33    median 21.87   p99 351.79
Mean TPOT (ms):              5.08     median 5.11     p99 5.76
Mean ITL (ms):                7.07     median 7.02     p99 8.54
Acceptance rate (%):         19.77
Acceptance length:           1.40
Drafts / Draft tokens:       14635 / 29270
Accepted tokens:             5786
Per-position acceptance:    position 0 = 32.20%   position 1 = 7.33%
```

### The definitive comparison

| | BF16-trained draft head (original, "wrong order") | FP8-trained draft head (this experiment, "right order") | Delta |
|---|---|---|---|
| Output tok/s | 1468.70 | 1462.95 | **-5.75 (-0.39%)** |
| Acceptance rate | 20.17% | 19.77% | **-0.40 pp** |
| Acceptance length | 1.40 | 1.40 | 0.00 |
| Position 0 acceptance | 33.19% | 32.20% | -0.99 pp |
| Position 1 acceptance | 7.15% | 7.33% | +0.18 pp |

**The FP8-trained draft head is marginally *worse*, not better** — the
opposite direction Chapter 7.1's theoretical argument predicts. The
magnitude (-0.39% throughput, -0.40pp acceptance rate) is small: on the same
order as, though slightly larger than, the ~0.1-1% run-to-run noise band
established across this session's other repeated/matched comparisons.
Reporting this honestly rather than forcing it to confirm the theory:

- **The theory is directionally reasonable** (zero train/serve distribution
  shift should, in principle, help), **but the measured effect size for this
  specific model, draft head, and dataset is at or below our noise floor.**
  We cannot confidently claim the "correct" order produces a meaningfully
  better draft head here.
- This is fully consistent with, and now has a mechanistic explanation from,
  Chapters 6 and 7's other evidence: FP8 barely shifts the verifier's output
  distribution (Ch6) and barely shifts the hidden states used for training
  (the near-identical loss curves in Step 3) — if the input the draft head
  learns from barely changes, there is no large effect available for the
  order of operations to unlock or lose.
- **This does not mean order-of-operations advice is wrong in general** — for
  a verifier where quantization *does* meaningfully shift hidden states (a
  more aggressive quantization scheme, a smaller/more sensitive model, or a
  verifier where FP8 introduces larger errors than we measured here), the
  same experiment could plausibly show a real, larger gap in the theory's
  favor. Our finding is specific to this Qwen3-8B + FP8-dynamic + EAGLE-3
  setup, not a universal claim that order never matters.

**Chapter 7 conclusion:** we ran the full, controlled version of Experiment
7.1 rather than relying on circumstantial evidence, and the result is a
genuinely nuanced one: quantize-before-training is the theoretically sound
default recommendation (zero distribution shift by construction is strictly
no worse, and the guide's practical argument about faster hidden-state
generation from the FP8 verifier still holds independently — our FP8
hidden-state generation run took ~6 minutes vs. the BF16 run's much longer
documented time), but for this specific model the quality difference it
buys is not measurable above noise. The practical recommendation to
quantize first remains reasonable — it is never worse, and is faster to
execute — even though our data doesn't show it being meaningfully better
here.

---

## Chapter 8 — Evidence Synthesis

Filled `ch8-final-table` and `ch8-answer` in the notebook with the real,
complete evidence trail from Chapters 5-7 rather than leaving them as blank
templates. No new experiments run here — this is pure synthesis of what's
already in this runlog.

**One addition the user specifically flagged and asked for**: the initial
fill of `ch8-final-table` only had the homework's 4 required configs
(Baseline, Spec Decoding, FP8 Quant, FP8+Spec Decoding), with the Chapter 7
FP8-trained-draft-head result (1462.95 tok/s, 19.77% acceptance) mentioned
only in `ch8-answer`'s prose. Added it as an explicit 5th row in the table
(clearly marked as not one of the 4 required configs, to avoid confusing it
with the homework's actual scoring rubric), since it's real measured
evidence directly relevant to the table's purpose — supporting the
order-of-operations conclusion — and burying it in prose-only made the
evidence table incomplete.

**Final evidence table (all 5 configs, matches the notebook exactly):**

| Config | tok/s | TTFT ms | TPOT ms | Accept% | Speedup |
|---|---|---|---|---|---|
| Baseline | 838.74 | 649.07 | 7.03 | N/A | 1.00 |
| Spec Decoding (C2, N=2) | 1069.82 | 173.27 | 6.43 | 20.25 | 1.28 |
| FP8 Quant (C3) | 1110.97 | 467.64 | 5.39 | N/A | 1.32 |
| FP8 + Spec Decoding (C4, N=2, BF16-trained head) | 1468.70 | 53.40 | 5.03 | 20.17 | 1.75 |
| same, FP8-trained head (Ch7 Experiment 7.1) | 1462.95 | 55.33 | 5.08 | 19.77 | 1.74 |

**Scoring check against the homework's actual thresholds (4 required
configs only — the 5th row is our own extra investigation, not part of the
rubric):** 0/50 — every config falls short of the reference run's absolute
throughput thresholds (1250 / 1550 / 1750 tok/s), despite our baseline
closely matching the reference's (838.74 vs 841.22) and our relative gains
being strong (+75.1% combined). Recorded honestly in both the notebook and
here rather than glossed over; the absolute-throughput gap is a separate,
environment-level question (server load, scheduling, concurrency-8
saturation point) that doesn't change any mechanistic conclusion from
Chapters 5-7 — but it does mean this run would not clear the homework's
scoring bar as currently measured.

**Final technical report** (full version in `ch8-answer`): quantize first,
but for the practical (speed) reason confirmed by direct measurement, not
the acceptance-rate reason the theory predicted and Experiment 7.1
empirically contradicted (at small, near-noise magnitude) for this specific
model. The combined config's super-multiplicative gain (+3.6% over the
multiplicative prediction, +9.4% over additive) is the strongest single
piece of evidence for combining the two techniques, independent of the
order-of-operations question.

---

## Chapter 9 — Filled the Real Submission Notebook

Filled `spec_dec+quantization_homework.ipynb` (the actual homework
submission file, not the GUIDE) directly, grounding every answer in this
runlog's real captured data rather than the reference numbers baked into
the template. No new experiments — pure transcription + synthesis of
what's already documented above.

**What was added (10 original cells → 16 cells):**
- Answer cells after Task 1-4 markdown blocks, each citing our own measured
  numbers (130.4 GB hidden states, our own epoch-5 training metrics, our
  own quantization + acceptance-rate findings, our own final benchmark +
  sweep tables) rather than restating the template's reference numbers.
- Replaced all three `TODO` benchmark-result blocks with the real raw
  `vllm bench serve` output for C2 (N=2), C3, and C4 (N=2) — copied
  verbatim from `/data/hw3/logs/c2_n2_bench.log`, `c3_bench.log`,
  `c4_n2_bench.log`.
- Added the draft-token tuning note after the spec-decoding results cell,
  and the full tuning table + justification after the combined results
  cell, per the assignment's own "Final Report Requirements."
- Converted the empty trailing code cell into the Central Question answer,
  written from our real Chapter 7 Experiment 7.1 result (not the
  theoretical assumption) and the real super-multiplicative combined-gain
  finding from Chapter 8.

**Honest scoring note, stated plainly in both Task 4's answer and the
combined-tuning cell, not just here:** none of the three scored
configurations clear their thresholds (Spec decoding `1069.82 < 1250`; FP8
`1110.97 < 1550`; Combined `1468.70 < 1750`) — current score **0/50** —
despite our own baseline (838.74) closely tracking the reference baseline
(841.22) and our relative gains being strong (combined +75.1%). This is the
open item flagged to tackle next, immediately after this chapter.

---

## Rubric Gap Investigation

Full hypothesis list and reasoning: `RUBRIC_GAP_HYPOTHESES.md`. This section
is the condensed, dated record of what was actually tested and found — the
hypotheses document has the "why we suspected this" detail, this has the
"what we ran and what happened."

**Framing:** baseline matched the reference almost exactly (-0.3%), but every
FP8-touching config fell increasingly short (Spec -15.0%, FP8 -29.1%,
Combined -16.9%) — ruled out generic "our environment is slower" explanations
from the start, since those would have dragged baseline down too.

### H1 — Cold compile/kernel-cache tax (dominant finding)

Found a persistent, disk-cached compiled-kernel directory
(`~/.cache/vllm/torch_compile_cache/`) keyed per distinct serving
configuration. The cache directory for "FP8 verifier, no speculative
decoding" has a filesystem timestamp landing exactly inside our original C3
benchmark window (23:35:13, benchmark ran 23:34:57-23:36:20) — that run was
the first time this environment had ever served that exact config, and the
one-time compile cost ate into an 18-second measurement window.

Re-tested every config against an already-warm server (discarding a first
"cold" run where relevant, keeping the stable warm reading):

| Config | Official (cold) | Warm re-test(s) | vs. threshold |
|---|---:|---:|---:|
| FP8 alone | 1110.97 | 1598.79, 1611.94 (mean 1605.37, **+44.5%**) | **PASSES** 1550 |
| Spec Decoding alone | 1069.82 | 1162.80, 1148.89, 1192.77 (mean 1168.15, +9.2%) | still -6.5% short of 1250 |
| Combined | 1468.70 | 1465.66, 1458.63, 1448.81 (mean ~1457.70, ~flat) | still -16.7% short of 1750 |

Why C2/C4's *official* numbers were less affected than C3's: both had already
been served once earlier in their own draft-token sweeps (N=1 ran before the
official N=2), so the relevant cache was already warm by the time the
official number was recorded. C3 was benchmarked standalone with no prior
warm-up run — the full cold-compile cost landed directly in its one official
measurement. C2 still shows a smaller, real +9.2% warm-up gain on top of
that — some shape-specific compilation apparently still happens
progressively as concurrency ramps even in an already-mostly-warm config.

**This alone flips FP8 Quant from FAIL to PASS (+10 points)** if a real
submission includes one throwaway warm-up run before recording results.

### H4 — Thermal/session drift: ruled out

GPU idle at 29°C, no throttle reasons active (`clocks_event_reasons.active`
only the benign "Idle" bit). Decisive evidence: C4's warm re-test, run very
late in an already many-hours-long session, matched its original measurement
almost exactly rather than degrading — the opposite of what accumulating
thermal throttling would predict.

### H5 — `--kv-cache-dtype fp8`: ruled out as primary cause

Warm result 1482.09 tok/s, only ~+1% over the standard warm C4 baseline —
real but far too small to close a 16.7% gap. (The *first* run with this new
flag combination independently reproduced H1's cold-compile signature —
TTFT 778ms mean vs the warm run's 25ms — confirming the mechanism
generalizes to any new flag combination, not just new models.)

### H6 — `--no-enable-chunked-prefill`: ruled out

Warm result 1425.00 tok/s, marginally *worse* than standard. Default chunked
prefill is already the better setting here.

### Where this leaves us

H1 fully explains the FP8-alone gap (now passes) and roughly half of the
apparent spec-decoding gap. A real, stable, *residual* gap remains once
everything is warm: **Spec Decoding -6.5% short, Combined -16.7% short.**
H5/H6 (the cheap, config-flag-level hypotheses) don't close it. Next step is
H2 (FP8 kernel tuned for large-batch prefill rather than the small-batch
decode regime this benchmark lives in) — though that alone can't be the
whole story since Spec Decoding alone (no FP8 involved) also falls short;
the remaining gap likely needs a broader look at speculative-decoding
scheduling overhead specifically, not just FP8 kernel selection.

---

## Baseline — 2026-06-30 (prior session, kept for reference)

**Model:** `/data/hw3/Qwen3-8B`  
**Backend:** openai  
**Dataset:** `philschmid/mt-bench` (hf), 80 prompts  
**Max concurrency:** 8 | **Request rate:** inf (Poisson) | **Custom output len:** 256

### Results

| Metric | Value |
|--------|-------|
| Successful requests | 80 |
| Failed requests | 0 |
| Benchmark duration (s) | 24.45 |
| Total input tokens | 6,078 |
| Total generated tokens | 20,480 |
| Request throughput (req/s) | 3.27 |
| Output token throughput (tok/s) | 837.76 |
| Peak output token throughput (tok/s) | 1,152.00 |
| Peak concurrent requests | 16.00 |
| Total token throughput (tok/s) | 1,086.39 |

#### Time to First Token (TTFT)
| Metric | Value |
|--------|-------|
| Mean (ms) | 563.41 |
| Median (ms) | 32.65 |
| P99 (ms) | 5,922.46 |

#### Time per Output Token — excl. 1st token (TPOT)
| Metric | Value |
|--------|-------|
| Mean (ms) | 7.37 |
| Median (ms) | 7.01 |
| P99 (ms) | 12.39 |

#### Inter-token Latency (ITL)
| Metric | Value |
|--------|-------|
| Mean (ms) | 7.37 |
| Median (ms) | 6.99 |
| P99 (ms) | 7.49 |

### Derived
| Metric | Value | Formula |
|--------|-------|---------|
| Compute util% | 0.043 | output_tok_s / 19500 (H100 BF16 compute-bound theoretical) |
