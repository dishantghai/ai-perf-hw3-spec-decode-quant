# Why We're Missing the Scoring Rubric — Hypothesis Log

> **UPDATE — H1 confirmed as the dominant cause, investigation results at the bottom
> of this file.** Short version: our official numbers included a one-time,
> per-config compile/kernel-cache cost that a short (~14-22s) benchmark window can't
> amortize away. Re-testing against a warm cache: **FP8 alone now clears its
> threshold** (1605 vs 1550 needed), **Spec Decoding alone gets much closer but still
> falls short** (1168 vs 1250, -6.5%), and **Combined still falls meaningfully short**
> (1458 vs 1750, -16.7%) even fully warm — so a real, second gap remains for Spec
> Decoding and Combined that H1 alone doesn't explain. Full results below.

**The problem, precisely stated:** our baseline throughput matches the reference run
almost exactly, but every config that touches FP8 falls increasingly short as FP8
becomes more central to the config:

| Config | Ours | Reference | Gap |
|---|---:|---:|---:|
| Baseline | 838.74 | 841.22 | **-0.3%** |
| Spec Decoding (no FP8) | 1069.82 | 1258.65 | -15.0% |
| FP8 Quant (FP8 alone) | 1110.97 | 1566.56 | **-29.1%** |
| Combined (FP8 + spec) | 1468.70 | 1766.55 | -16.9% |

The baseline match rules out generic explanations ("our GPU/instance is just slower,"
"our benchmark protocol differs somehow") — those would drag baseline down too, and
they don't. Whatever's wrong is specifically tied to how our FP8 model executes, not
to our measurement methodology, environment, or hardware in general. That's the frame
for every hypothesis below.

**One more real clue, not just the top-line number:** our FP8-alone TPOT improvement
(steady-state per-token decode cost) is 23.3% vs the reference's 32.7% — a real gap,
but nowhere near as large as the 32.5%-vs-86.2% throughput gap. Something is costing
us disproportionately at the *throughput* level (which integrates over the whole
benchmark window, including startup/queueing) beyond what raw per-token decode speed
alone would predict. C3's TTFT (mean 467.64ms, P99 5708.86ms) is nearly as bad as
baseline's (649.07ms / 5886.82ms) despite FP8 decoding measurably faster per token —
that's the single strangest data point so far.

---

## Hypotheses, ranked by how well the evidence fits

### H1 — Cold-start / kernel-autotuning tax eating into a short benchmark window (HIGH plausibility)

**The idea:** vLLM's FP8 GEMM kernel (`CutlassFP8ScaledMMLinearKernel`, confirmed in
our server log) may perform one-time or per-shape autotuning/compilation the first
time it's invoked at a given batch size. Our methodology cold-starts a *fresh*
`vllm serve` process for every single benchmark (by design, for measurement
isolation) — the engine config shows `local_cache_dir: None`, i.e. no persistent
compile cache across process restarts. If this tax is real and FP8-specific, a short
~18s benchmark window (C3's actual duration) pays a much larger *relative* penalty
than a longer one would, and disproportionately hits TTFT/throughput (which include
the one-time stall) while barely touching TPOT (steady-state, sampled after the
stall is over).

**Fits:** the exact pattern we see — small TPOT gap, large throughput/TTFT gap, and
a P99 TTFT almost as bad as baseline (consistent with a handful of requests eating a
multi-second one-time stall early in the run).

**Doesn't yet explain:** why C4 (also FP8, benchmarked *after* C3 got its own fresh
cold start) still hit a strong relative result — unless C4's addition of speculative
decoding masks/dilutes this cost differently, or unless autotuning is genuinely
per-run non-deterministic.

**Cheapest test:** re-run the *exact same* C3 benchmark twice against the same
already-running server (no restart between the two runs). If run 2 is meaningfully
faster than run 1, this hypothesis is confirmed directly, and the fix for
resubmission is simple: run one throwaway warm-up benchmark, then record the second.

---

### H2 — FP8 kernel selection is tuned for large-batch prefill, not small-batch decode (MEDIUM-HIGH plausibility)

**The idea:** vLLM sometimes has multiple FP8 GEMM kernel implementations (e.g.
Cutlass vs Marlin) tuned for different regimes — Marlin-style kernels are often
better for the small-batch, memory-bandwidth-bound *decode* regime this whole
benchmark lives in (concurrency 8, mostly single-token-at-a-time generation per
stream), while Cutlass-style kernels are often tuned for large-batch *prefill*
throughput. Our server log shows `Selected CutlassFP8ScaledMMLinearKernel` — if
that's the wrong kernel for this workload's actual regime, we'd see exactly this
kind of gap: real, working FP8 quantization (confirmed correct config, confirmed
8.9GB model, confirmed right recipe) that still underperforms its potential.

**Fits:** explains a *persistent*, not just one-time, per-token cost gap — would
show up in TPOT too, which we do see (23.3% vs 32.7%), just less dramatically than
the throughput gap.

**Doesn't fully explain:** why the TTFT/throughput gap is so much larger than the
TPOT gap alone would predict — probably combines with H1 rather than replacing it.

**Test:** check whether vLLM v0.20.0 exposes a way to force kernel selection (env
var or config flag), or whether a newer/older vLLM build picks a different kernel
for this model+GPU combination. Lower-effort check: search vLLM's changelog/issues
for "CutlassFP8ScaledMMLinearKernel" performance discussion at small batch sizes.

---

### H3 — Reference solution didn't cold-start a fresh server per benchmark (MEDIUM plausibility)

**The idea:** the reference numbers could have been collected against a
long-running, already-warm server (e.g. one server process serving all benchmark
runs for a given config back-to-back, or a server that had already served other
traffic before the timed run). If so, the reference never paid the cold-start tax
H1 describes at all, while our "always isolate, always cold-start" methodology
(a deliberate choice, for good measurement hygiene) pays it on every single run.

**Fits:** would explain why *our* methodology is internally consistent (every run
we make is cold, so relative comparisons *between our own runs* are fair) while
still trailing an externally-reported reference number that may not have played by
the same rules.

**Doesn't explain:** the specific FP8 >> baseline gap pattern by itself — needs H1
or H2 to explain *why* it's FP8-specific, not just "cold start costs something in
general" (baseline's near-perfect match argues cold-start cost is small for BF16).

**Test:** this can't be directly verified without the reference's actual methodology,
but H1's warm-vs-cold test on our own server serves as an indirect check — if warm
C3 numbers close most of the gap, this combined with H1 becomes very likely.

---

### H4 — Sustained multi-hour session load has changed GPU clock/thermal state (LOW-MEDIUM plausibility)

**The idea:** we ran a large number of consecutive `vllm serve` + `vllm bench serve`
cycles this session (12+ server launches across Chapters 5-7 before C3 ran). Sustained
load could trigger thermal throttling or persistence-mode clock changes that reduce
peak achievable throughput later in the session.

**Argues against:** C1 (baseline) ran *first*, matches the reference almost exactly.
C4's sweep (4 full server cycles) ran *before* C3, yet C4's *final* result (measured
even later, reused from the sweep) still hit a strong relative gain. If thermal
throttling were the dominant story, we'd expect a roughly monotonic decline in
absolute performance across the session, not "baseline: fine, spec-alone: -15%,
FP8-alone: -29%, combined: -17%" — a pattern tied to *which technique is active*,
not to *when in the session* the run happened.

**Test (cheap, worth doing regardless):** `nvidia-smi -q -d CLOCK,TEMPERATURE` during
a live run, and/or re-run baseline right now and compare to the original 838.74 — if
it's still close, this rules H4 out cleanly.

---

### H5 — KV cache left in `auto` (BF16) dtype instead of FP8 (LOW plausibility as primary cause)

**The idea:** our engine config shows `kv_cache_dtype=auto` for every run, including
FP8 configs — the reference might have additionally used `--kv-cache-dtype fp8` to
shrink KV cache memory traffic further, which the assignment doesn't strictly
require but could be a "bonus" optimization the reference solution applied.

**Argues against as the *primary* explanation:** KV cache reads are a much smaller
fraction of total memory traffic than weight loads at this sequence length
(~2048 tokens, `mt-bench`-scale prompts) and this batch size — it would help, but
not by 29 percentage points on its own. Worth testing as a real, legitimate
additional optimization regardless of whether it's "the" answer.

**Test:** re-run C3 with `--kv-cache-dtype fp8` added and compare.

---

### H6 — Chunked prefill interacting badly with FP8 for this workload (LOW plausibility)

**The idea:** `enable_chunked_prefill=True` is vLLM's V1 default and matches what
baseline also used — since baseline matches the reference closely, this setting
isn't obviously broken in general. Still listed because chunked-prefill scheduling
heuristics *could* interact differently with FP8's different compute/memory balance
than they do with BF16.

**Argues against:** same setting was active for baseline (which matched closely) and
for C4 (which did well relatively) — if this were a major FP8-specific problem we'd
expect to see it hurt C4 as much as C3, and we don't.

**Test:** re-run C3 with `--no-enable-chunked-prefill` and compare. Cheap to test,
low expected payoff, worth ruling out explicitly rather than assuming.

---

### H7 — Quantization recipe mismatch vs. what the reference actually used (LOW plausibility — largely ruled out already)

**The idea:** maybe the reference used a different quantization scheme (static vs
dynamic activations, per-tensor vs per-channel weights) that happens to serve faster.

**Ruled out by direct check:** our saved config shows `activations.dynamic: true,
strategy: 'token'` and `weights.dynamic: false, strategy: 'channel'` — this is
*exactly* what the assignment specifies ("Weight format: FP8" static, "Activation
format: dynamic FP8"). We are not free to change this without violating the
assignment's own required recipe, so even if a different recipe were faster, it
wouldn't be a valid fix. Listed for completeness, not as an actionable lead.

---

### H8 — Effective concurrency/batching differs because FP8 requests finish faster (LOW-MEDIUM plausibility, subtle)

**The idea:** with `--max-concurrency 8`, the scheduler admits a new request as soon
as an old one finishes. If FP8 finishes each request meaningfully faster than BF16,
the *time window* during which multiple requests overlap in a batch could shrink,
paradoxically reducing average batch occupancy and giving up some of the
arithmetic-intensity benefit that continuous batching is supposed to provide. This
is a real, known second-order effect in continuous-batching systems, not specific to
our setup.

**Test:** compare `Peak concurrent requests` and the shape of the concurrency
curve (if `vllm bench serve` exposes per-request timing) between our C3 run and
baseline — if FP8's realized average concurrency is measurably lower than
baseline's despite the same `--max-concurrency 8` cap, this is a contributing factor
worth naming explicitly, though it's unlikely to be fixable without changing the
required benchmark protocol (which we should not do).

---

## Recommended test order (cheapest / most diagnostic first)

1. **H1 — warm-server re-run of C3** (same server, run the benchmark twice, compare
   run 1 vs run 2). Fastest test, most directly diagnostic, no config changes needed.
2. **H4 — quick baseline re-run + `nvidia-smi` clock check**, to definitively rule
   out session-long thermal drift before chasing anything FP8-specific.
3. **H5 — `--kv-cache-dtype fp8` re-run of C3**, cheap, legitimate optimization
   regardless of whether it's the main cause.
4. **H6 — `--no-enable-chunked-prefill` re-run of C3**, cheap, rules out a specific
   scheduler interaction.
5. **H2 — kernel-selection research**, only if H1/H4/H5/H6 don't close the gap —
   more effort (may require a different vLLM build or env var), so worth confirming
   the cheaper hypotheses first.
6. **H3 / H8** are contextual explanations more than fixable levers — useful for
   understanding *why* a gap could exist even with a fully correct setup, but not
   independently actionable without changing the required benchmark protocol.

---

## Investigation Results

Full raw command output and per-run detail: `runlog.md` ("Rubric Gap Investigation"
section). This section is the verdict summary.

### H1 — Cold-start / kernel-autotuning tax: **CONFIRMED, dominant cause**

Root cause pinned down exactly: `~/.cache/vllm/torch_compile_cache/` holds a
persistent, disk-cached compiled-kernel directory *per distinct serving
configuration* (model + quantization + speculative-decoding-or-not). The cache
directory for "FP8 verifier, no speculative decoding" was created with a timestamp
that lines up exactly with our original C3 benchmark window — that run was the
first time this environment had ever served that exact configuration, and it paid
a one-time compile cost inside an 18-second measurement window.

| Config | Official (cold) | Warm re-test(s) | Verdict vs threshold |
|---|---:|---:|---:|
| FP8 alone (C3) | 1110.97 | 1598.79, 1611.94 (mean 1605.37, **+44.5%**) | **PASS** (>1550) |
| Spec Decoding alone (C2, N=2) | 1069.82 | 1162.80, 1148.89, 1192.77 (mean 1168.15, +9.2%) | FAIL, -6.5% short of 1250 |
| Combined (C4, N=2) | 1468.70 | 1465.66, 1458.63, 1448.81 (mean ~1457.70, ~flat) | FAIL, -16.7% short of 1750 |

Confirmed *why* C2 and C4 weren't affected the same way C3 was: both configs had
already been exercised once earlier in their own draft-token sweeps (N=1 ran before
N=2, in the same session), so by the time their *official* N=2 numbers were
recorded, the relevant compile cache was already warm. C3 had no such prior run —
it was benchmarked standalone. C2 still shows a real, smaller (+9.2%) warm-up gain
on top of that, suggesting some additional shape/path-specific compilation still
happens progressively as concurrency ramps even within an already-mostly-warm
config — smaller effect, same underlying mechanism.

**Practical implication:** for a real submission, run one throwaway warm-up
benchmark per configuration before recording the "official" one. This alone flips
FP8 Quant from FAIL to PASS (+10 points).

### H4 — Thermal/session drift: **RULED OUT**

GPU idle temp 29°C, no throttle reasons active. Decisive evidence: C4's warm
re-test (run very late in an already many-hours-long session) matched its original
measurement almost exactly rather than degrading — the opposite of what
accumulating thermal throttling would predict.

### H5 — KV cache FP8 dtype (`--kv-cache-dtype fp8`): **RULED OUT as primary cause**

Warm result: 1482.09 tok/s, only ~+1% over the standard warm C4 baseline
(1457.70). Real but small effect, nowhere near enough to close the 16.7% gap. Note:
the *first* run with this new flag combination independently showed the same
cold-compile signature as H1 (TTFT 778ms) — confirms H1's mechanism generalizes to
any new flag combination, not just new models/speculative configs.

### H6 — Chunked prefill interaction (`--no-enable-chunked-prefill`): **RULED OUT**

Warm result: 1425.00 tok/s, marginally *worse* than the standard warm C4 baseline.
No benefit; default (`enable_chunked_prefill=True`) is already the better setting
for this workload.

### Remaining, unexplained gap

Even at a fully warm, stable state (5 consistent measurements for C4 clustered
1448-1482, 3 consistent measurements for C2 clustered 1149-1193), **Spec Decoding
alone is short by 6.5% and Combined is short by 16.7%** against the assignment's
thresholds. H1/H4/H5/H6 collectively explain the *entire* FP8-alone gap and roughly
half of the apparent spec-decoding/combined gap (the portion caused by the original
numbers being cold), but a real residual gap remains for any config involving
speculative decoding specifically. H2 (FP8 kernel tuned for prefill, not the
small-batch decode regime this benchmark lives in) remains the most likely
explanation for what's left, but note it wouldn't explain a *spec-decoding-specific*
shortfall on its own (C2 has no FP8 involved at all) — worth broadening the
hypothesis set for the remaining gap specifically around speculative-decoding
scheduling overhead, not just FP8 kernel selection, before continuing further.
