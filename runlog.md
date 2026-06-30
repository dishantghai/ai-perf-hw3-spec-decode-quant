# Run Log

## Baseline — 2026-06-30

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
