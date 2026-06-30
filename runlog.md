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
