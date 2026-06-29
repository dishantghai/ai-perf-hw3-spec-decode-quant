# Run Log

## Baseline — 2026-06-29

**Model:** `/data/hw3/Qwen3-8B`  
**Backend:** openai  
**Dataset:** `philschmid/mt-bench` (hf), 80 prompts  
**Max concurrency:** 8 | **Request rate:** inf (Poisson) | **Custom output len:** 256

### Results

| Metric | Value |
|--------|-------|
| Successful requests | 80 |
| Failed requests | 0 |
| Benchmark duration (s) | 24.58 |
| Total input tokens | 6,078 |
| Total generated tokens | 20,480 |
| Request throughput (req/s) | 3.25 |
| Output token throughput (tok/s) | 833.08 |
| Peak output token throughput (tok/s) | 1,152.00 |
| Peak concurrent requests | 16.00 |
| Total token throughput (tok/s) | 1,080.32 |

#### Time to First Token (TTFT)
| Metric | Value |
|--------|-------|
| Mean (ms) | 668.65 |
| Median (ms) | 32.19 |
| P99 (ms) | 6,068.07 |

#### Time per Output Token — excl. 1st token (TPOT)
| Metric | Value |
|--------|-------|
| Mean (ms) | 7.01 |
| Median (ms) | 7.01 |
| P99 (ms) | 7.14 |

#### Inter-token Latency (ITL)
| Metric | Value |
|--------|-------|
| Mean (ms) | 7.01 |
| Median (ms) | 7.00 |
| P99 (ms) | 7.39 |
