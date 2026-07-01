# Speculative Decoding + FP8 Quantization — Qwen3-8B on H100

Homework project building and evaluating a multi-stage LLM inference acceleration
pipeline for `Qwen/Qwen3-8B` on a single H100 80GB GPU:

- train an EAGLE-3 speculative-decoding draft head (offline, from precomputed hidden states);
- quantize the verifier model with FP8 dynamic quantization (`llm-compressor`);
- serve and benchmark baseline / speculative-decoding / FP8 / combined configurations with `vllm bench serve`;
- determine, with real measurements rather than assumption, whether quantization or
  draft-head training should happen first.

## Final measured results

| Configuration | Output tok/s |
| --- | ---: |
| Baseline | `838.74` |
| Speculative decoding alone | `1279.46` |
| FP8 dynamic quantization alone | `1611.94` |
| FP8 + speculative decoding (combined) | mean `1750.93` (best `1758.93`, 4 warm runs) |

(The assignment's own scoring rubric and thresholds are stated in
`spec_dec+quantization_homework_final.ipynb`, Task 4 — scoring against it is left to
the assessor.)

Getting to these numbers required two corrections after an initial pass whose results
were substantially lower — a cold `torch.compile`/kernel-cache tax baked into short
benchmark windows, and a draft head that was accidentally trained with the full
151,936-token vocabulary instead of a compressed one. Both are explained in full, with
the actual measurements, inside the submission notebook itself.

## Where to look

| File | What it is |
| --- | --- |
| `spec_dec+quantization_homework_final.ipynb` | **The submission.** Self-contained: every answer, benchmark result, and piece of reasoning is inline, grounded in real runs on this H100 instance. |
| `spec_dec+quantization_homework_final.zip` | The submission notebook, zipped for upload. |
| `MASTERCLASS_GUIDE.ipynb` | A much larger teaching/lab notebook (11 chapters) built while doing this project — instance sizing, the compute-vs-bandwidth roofline model, EAGLE-3 training internals, FP8 quantization mechanics, the full benchmark sweep methodology, and the investigation that closed the scoring gap. Not part of the submission; kept as the working lab record. |
| `spec_dec+quantization_homework.ipynb` | An earlier, first-pass fill of the submission template, using the pre-correction (0/50) numbers. Kept as a historical record of what changed and why. |
| `spec_dec+quantization_homework main.ipynb` | The original, unfilled assignment template. |
| `runlog.md` | Chronological lab log: every benchmark run, what was being tested and why, and the analysis after each one. |
| `RUBRIC_GAP_HYPOTHESES.md` | Ranked list of hypotheses investigated to explain the 0/50 first-pass gap (cold-compile tax, draft-vocab size, KV-cache dtype, chunked prefill, thermal drift, etc.), with which were confirmed and which were ruled out. |
| `EAGLE-3.md` | Reference notes on EAGLE-3 speculative decoding internals (architecture, training-time test, verification). |
| `Hidden_State_Encodes_Future_Token_Distributions.md` | Reference notes on why a verifier's hidden states carry signal about future tokens, motivating why EAGLE-style draft heads work. |
| `scripts/` | Every script actually run on the GPU instance: data prep, hidden-state generation, EAGLE-3 training (original, compressed-vocab, and FP8-hidden-state variants), FP8 quantization, and server launch. See the table below. |
| `logs/` | Raw `vllm bench serve` / training output for every run cited in `runlog.md`, `RUBRIC_GAP_HYPOTHESES.md`, and the submission notebook — the underlying evidence, not just the extracted numbers. |
| `checkpoints_metadata/` | `config.json` / `config.py` / `val_metrics.json` for each of the 3 trained draft-head checkpoints (original, compressed-vocab, FP8-trained). Metadata only — the actual model weights (multi-GB `.safetensors`) are not checked in. |

### `scripts/`

| Script | Purpose |
| --- | --- |
| `prepare_data.sh` | Tokenize ShareGPT data for offline EAGLE-3 training. |
| `generate_hidden_states.sh` | Generate offline hidden states from the BF16 verifier. |
| `train_eagle3.sh` | Train the original EAGLE-3 draft head (full 151,936-token vocab). |
| `train_eagle3_compressed_vocab.sh` | Train the corrected draft head with `--draft-vocab-size 32000` — the fix that closed most of the scoring gap. |
| `quantize_qwen3.py` | FP8 dynamic quantization of the verifier via `llm-compressor` (Task 3). |
| `launch_server.sh` | Launch the vLLM server for benchmarking. |
| `generate_hidden_states_fp8.sh`, `launch_server_fp8_hs.sh`, `train_eagle3_fp8.sh` | The Chapter 7 / Experiment 7.1 pipeline: generate hidden states from the *quantized* verifier and train a second draft head on them, to directly test the "train on what you serve" order-of-operations argument. |
| `inspect_sharegpt.py` | Small utility to preview the ShareGPT dataset. |

## Pipeline summary

1. **Environment & data** — separate venvs for training (`speculators`), serving (`vllm`),
   and quantization (`llmcompressor`); prepare ShareGPT-style data and generate offline
   hidden states for `Qwen/Qwen3-8B` (3,000 samples, ~130 GB).
2. **Train an EAGLE-3 draft head** — offline training against the precomputed hidden
   states, with a compressed 32,000-token draft vocabulary (the fix that mattered most
   for scoring).
3. **Quantize the verifier** — FP8 dynamic quantization via `llm-compressor`, `lm_head`
   left unquantized.
4. **Serve and benchmark** — `vllm bench serve` against `philschmid/mt-bench`
   (`--max-concurrency 8 --num-prompts 80 --no-enable-prefix-caching`) across baseline,
   speculative-decoding-alone, FP8-alone, and combined configurations, with the
   draft-token count (`num_speculative_tokens`) tuned independently per configuration.
5. **Answer the central question** — quantize first, then generate hidden states from
   the quantized verifier, then train the draft head — validated by directly training a
   second draft head on FP8-generated hidden states and measuring the acceptance-rate
   effect, rather than assuming the theoretical argument holds.

Full reasoning and every raw result are in `spec_dec+quantization_homework_final.ipynb`.
