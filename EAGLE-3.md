# EAGLE-3 Deep Dive: Complete Learning Guide

**A Comprehensive Resource for Understanding LLM Inference Acceleration**

---

## Table of Contents

1. [Introduction to LLM Inference Acceleration](#1-introduction-to-llm-inference-acceleration)
2. [What is EAGLE-3?](#2-what-is-eagle-3)
3. [The Core Problem: Why LLMs Are Slow](#3-the-core-problem-why-llms-are-slow)
4. [Speculative Sampling Fundamentals](#4-speculative-sampling-fundamentals)
5. [Evolution: EAGLE → EAGLE-2 → EAGLE-3](#5-evolution-eagle--eagle-2--eagle-3)
6. [EAGLE-3 Architecture Deep Dive](#6-eagle-3-architecture-deep-dive)
7. [Feature Fusion: Multi-Layer Information](#7-feature-fusion-multi-layer-information)
8. [Training-Time Test Technique](#8-training-time-test-technique)
9. [Loss Function and Training Process](#9-loss-function-and-training-process)
10. [Inference Pipeline Step-by-Step](#10-inference-pipeline-step-by-step)
11. [The Verification Process in Depth](#11-the-verification-process-in-depth)
12. [Attention Mask Adjustments](#12-attention-mask-adjustments)
13. [Implicit Domain Adaptation](#13-implicit-domain-adaptation)
14. [Performance Results and Benchmarks](#14-performance-results-and-benchmarks)
15. [Implementation Details](#15-implementation-details)
16. [Common Misconceptions Clarified](#16-common-misconceptions-clarified)
17. [Summary and Key Takeaways](#17-summary-and-key-takeaways)

---

## 1. Introduction to LLM Inference Acceleration

### Why This Matters

Large Language Models (LLMs) have become incredibly powerful, but they're also **slow and expensive** to run. Every single token generated requires:

- Accessing all model parameters (billions of weights)
- Complete forward pass through all layers
- Memory-intensive operations

This makes LLM inference a **memory-bound** problem—waiting for data to move is slower than computing with it.

### The Solution: Speculative Sampling

Instead of generating one token at a time, what if we could:

1. **Draft** multiple tokens quickly using a smaller model
2. **Verify** all drafts in parallel using the large model
3. **Accept** correct drafts, **reject** wrong ones

This is the core idea behind speculative sampling, and EAGLE-3 is one of the most advanced implementations.

---

## 2. What is EAGLE-3?

**EAGLE-3** (Extrapolation Algorithm for Greater Language-model Efficiency) is a **speculative sampling method** designed to accelerate LLM inference without losing any accuracy.

### Key Statistics

| Metric | Value |
| :--- | :--- |
| **Maximum Speedup** | 6.5x over vanilla autoregressive |
| **Improvement over EAGLE-2** | ~1.4x |
| **Throughput Gain (SGLang, batch=64)** | 1.38x |
| **Training Data Scale** | ~8x more than EAGLE |
| **Lossless?** | Yes—identical output distribution |

### Core Innovations

1. **Direct Token Prediction** — No feature prediction constraint
2. **Multi-Layer Feature Fusion** — Uses low, mid, and high-level features
3. **Training-Time Test** — Simulates inference during training
4. **Scaling Law Discovery** — More training data = better performance

---

## 3. The Core Problem: Why LLMs Are Slow

### Memory-Bound Decoding

```
Traditional Autoregressive Generation:

Token 1 → Target Model → Token 2 → Target Model → Token 3 → Target Model
    ↓                        ↓                        ↓
Full Forward Pass        Full Forward Pass        Full Forward Pass
(All parameters)         (All parameters)         (All parameters)
```

**Problem:** Each token requires loading all model parameters from memory. For a 70B model, that's 140GB+ of data movement **per token**.

### The Bottleneck

| Component | Speed |
| :--- | :--- |
| GPU Compute (FLOPs) | Very Fast |
| Memory Bandwidth | Much Slower |
| **Result** | **GPU waits for memory** |

EAGLE-3 reduces memory accesses by generating multiple tokens per target model forward pass.

---

## 4. Speculative Sampling Fundamentals

### Basic Concept

```
┌─────────────────────────────────────────────────────────────┐
│                    Speculative Sampling                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Draft Model (Small, Fast)      Target Model (Large, Slow)  │
│  ─────────────────────          ─────────────────────       │
│  1. Generate k draft tokens     1. Verify all k tokens      │
│     quickly                        in parallel               │
│                                                              │
│  2. Send drafts to Target       2. Accept/Reject each       │
│                                      token                   │
│                                                              │
│  3. Repeat                      3. Continue from last       │
│                                      accepted token          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Acceptance Mechanism

For each draft token $\hat{t}$:

$$\text{Acceptance Probability} = \min\left(1, \frac{p(t)}{\hat{p}(t)}\right)$$

Where:
- $p(t)$ = Target model's probability for token $t$
- $\hat{p}(t)$ = Draft model's probability for token $t$

If rejected, sample from $\text{norm}(\max(0, p - \hat{p}))$ to replace the token.

### Why It's Lossless

Speculative sampling is **mathematically proven** to produce the same output distribution as vanilla autoregressive decoding. The target model always has final say.

---

## 5. Evolution: EAGLE → EAGLE-2 → EAGLE-3

### EAGLE (Version 1)

| Feature | Description |
| :--- | :--- |
| **Prediction Type** | Feature-level autoregression |
| **Input** | Top-layer features only |
| **Loss** | Feature loss + Token loss |
| **Draft Structure** | Tree-based (static) |
| **Limitation** | Can't scale with more data |

### EAGLE-2

| Feature | Description |
| :--- | :--- |
| **Improvement** | Dynamic draft trees |
| **Mechanism** | Confidence-based pruning |
| **Benefit** | Better resource allocation |
| **Limitation** | Still uses feature prediction |

### EAGLE-3

| Feature | Description |
| :--- | :--- |
| **Prediction Type** | **Direct token prediction** |
| **Input** | **Fused multi-layer features** |
| **Loss** | **Token loss only** |
| **Draft Structure** | Dynamic trees (from EAGLE-2) |
| **Benefit** | **Scales with training data** |

### Comparison Table

```
┌──────────────────┬─────────────┬─────────────┬─────────────┐
│     Feature      │   EAGLE     │  EAGLE-2    │   EAGLE-3   │
├──────────────────┼─────────────┼─────────────┼─────────────┤
│ Prediction       │ Features    │ Features    │ Tokens      │
│ Input Features   │ Top-layer   │ Top-layer   │ Fused       │
│ Loss Components  │ 2 (fea+tok) │ 2 (fea+tok) │ 1 (token)   │
│ Draft Tree       │ Static      │ Dynamic     │ Dynamic     │
│ Scales with Data │ ❌ No       │ ❌ No       │ ✅ Yes      │
│ Max Speedup      │ ~3.0x       │ ~4.5x       │ ~6.5x       │
└──────────────────┴─────────────┴─────────────┴─────────────┘
```

---

## 6. EAGLE-3 Architecture Deep Dive

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      EAGLE-3 Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Target     │      │   Feature    │      │    Draft     │  │
│  │    Model     │─────▶│   Fusion     │─────▶│    Model     │  │
│  │  (Frozen)    │      │   (FC Layer) │      │ (Trainable)  │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│         │                     │                       │         │
│         ▼                     ▼                       ▼         │
│    l, m, h features        g = FC(l,m,h)          Token         │
│   (Low, Mid, High)      (Fused Feature)        Prediction      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. Target Model
- **Role:** Provides ground truth tokens and features
- **Status:** Frozen during draft model training
- **Output:** Low, middle, high-level features ($l$, $m$, $h$)

#### 2. Feature Fusion Layer
- **Type:** Fully Connected (FC) layer
- **Input:** Concatenated $l$, $m$, $h$ (3k dimensions)
- **Output:** Fused feature $g$ (k dimensions)
- **Purpose:** Compress multi-layer info into single vector

#### 3. Draft Model
- **Type:** Transformer decoder layer
- **Input:** Fused features $g$ + token embeddings
- **Output:** Token predictions
- **Training:** Offline, before deployment

---

## 7. Feature Fusion: Multi-Layer Information

### Why Multi-Layer Features?

Different layers encode different types of information:

| Layer Type | Information Encoded |
| :--- | :--- |
| **Low-Level** | Syntax, grammar, token structure |
| **Mid-Level** | Semantic relationships, context |
| **High-Level** | Task intent, reasoning patterns |

Using only top-layer features (like EAGLE) limits the draft model to final-decision information. EAGLE-3 captures the **full reasoning chain**.

### Fusion Process

```
Step 1: Extract Features from Target Model
───────────────────────────────────────────

Prefix: "How can"

Target Model Forward Pass
         ↓
┌─────────────────────────────────────┐
│  Low-level features (l):   [l_how, l_can]  │
│  Mid-level features (m):   [m_how, m_can]  │
│  High-level features (h):  [h_how, h_can]  │
└─────────────────────────────────────┘

Step 2: Concatenate and Fuse
────────────────────────────

For each position:
g_how = FC(concat(l_how, m_how, h_how))
g_can = FC(concat(l_can, m_can, h_can))

Where FC reduces 3k → k dimensions

Step 3: Use Fused Features
──────────────────────────

Draft Model Input: [g_how, g_can] + embeddings
         ↓
Draft Model predicts next token
```

### Mathematical Representation

$$g_i = \text{FC}(\text{concat}(l_i, m_i, h_i))$$

Where:
- $l_i, m_i, h_i \in \mathbb{R}^k$ (k = hidden size)
- $\text{concat}(l_i, m_i, h_i) \in \mathbb{R}^{3k}$
- $\text{FC}: \mathbb{R}^{3k} \rightarrow \mathbb{R}^k$

---

## 8. Training-Time Test Technique

### The Core Insight

**Problem:** During training, models typically see ground truth inputs. During inference, they must use their own predictions. This mismatch causes **distribution shift**.

**Solution:** Simulate inference conditions **during training**.

### Visual Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│              Standard Training vs Training-Time Test             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STANDARD TRAINING:                                              │
│  ───────────────────                                             │
│  Step 1: Input = Ground Truth → Output = Prediction 1            │
│  Step 2: Input = Ground Truth → Output = Prediction 2            │
│  Step 3: Input = Ground Truth → Output = Prediction 3            │
│                                                                  │
│  Problem: Never practices using its own outputs!                 │
│                                                                  │
│  EAGLE-3 TRAINING-TIME TEST:                                     │
│  ─────────────────────────────                                   │
│  Step 1: Input = Ground Truth → Output = Prediction 1            │
│  Step 2: Input = Prediction 1   → Output = Prediction 2            │
│  Step 3: Input = Prediction 2   → Output = Prediction 3            │
│                                                                  │
│  Benefit: Learns to handle its own errors during training!       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Example

**Training Sequence:** "How can I do it"

#### Phase 1: Step 1 (Ground Truth Input)
```
Input: g_how, g_can (from Target Model)
       + embedding of "I" (e_I)
       ↓
Draft Model
       ↓
Output: a_I (prediction vector for "I")
       ↓
Loss: Compare predicted token vs actual "I"
```

#### Phase 2: Step 2 (Own Output as Input)
```
Input: a_I (from Draft's Step 1)
       + embedding of "do" (e_do)
       ↓
Draft Model
       ↓
Output: a_do (prediction vector for "do")
       ↓
Loss: Compare predicted token vs actual "do"
```

#### Phase 3: Step 3 (Continue Chain)
```
Input: a_do (from Draft's Step 2)
       + embedding of "it" (e_it)
       ↓
Draft Model
       ↓
Output: a_it (prediction vector for "it")
       ↓
Loss: Compare predicted token vs actual "it"
```

### Why This Works

| Aspect | Explanation |
| :--- | :--- |
| **Error Recovery** | Learns to continue even after imperfect predictions |
| **Distribution Match** | Training input distribution matches inference |
| **Acceptance Rate** | Maintains high acceptance for later tokens |
| **No Error Accumulation** | Earlier mistakes don't cascade catastrophically |

---

## 9. Loss Function and Training Process

### Loss Function Evolution

#### EAGLE (Old)
$$\mathcal{L}_{\text{EAGLE}} = \mathcal{L}_{\text{fea}} + \mathcal{L}_{\text{token}}$$

- $\mathcal{L}_{\text{fea}}$: Feature prediction loss (match target's hidden states)
- $\mathcal{L}_{\text{token}}$: Token prediction loss (match target's tokens)

#### EAGLE-3 (New)
$$\mathcal{L}_{\text{EAGLE-3}} = \mathcal{L}_{\text{token}}$$

**Only token prediction loss!** The feature constraint is removed.

### Token Prediction Loss (Cross-Entropy)

$$\mathcal{L}_{\text{token}} = -\sum_{i=1}^{n} \log P_{\text{draft}}(t_i^{\text{target}} | \text{input}_i)$$

Where:
- $t_i^{\text{target}}$ = Actual token from Target Model
- $P_{\text{draft}}$ = Draft Model's predicted probability
- $\text{input}_i$ = Either $g_i$ (fused feature) or $a_{i-1}$ (own output)

### Why Remove Feature Loss?

| With Feature Loss | Without Feature Loss |
| :--- | :--- |
| Restrictive constraint | More flexibility |
| Must match internal math | Only needs correct tokens |
| Hits performance ceiling | Scales with more data |
| Limited expressiveness | Full model capacity |

From the paper:

> "With token prediction as the ultimate goal, feature prediction can be seen as an additional constraint, which limits the expressiveness of the draft model and makes it difficult to benefit from increased data."

### Training Hyperparameters

| Parameter | Value |
| :--- | :--- |
| **Optimizer** | AdamW |
| **Beta Values** | $(\beta_1, \beta_2) = (0.9, 0.95)$ |
| **Learning Rate** | $5 \times 10^{-5}$ |
| **Gradient Clipping** | 0.5 |
| **Training Data** | ShareGPT + UltraChat-200K (~532K entries) |
| **Special Data** | OpenThoughts-114k-math (for reasoning models) |

### Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    EAGLE-3 Training Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Load Training Data                                           │
│     └─→ ShareGPT, UltraChat, (optional: math data)              │
│                                                                  │
│  2. Target Model Forward Pass                                    │
│     └─→ Generate responses, extract l, m, h features            │
│                                                                  │
│  3. Feature Fusion                                               │
│     └─→ Compute g = FC(concat(l, m, h))                         │
│                                                                  │
│  4. Training-Time Test Simulation                                │
│     └─→ Step 1: Use g features                                  │
│     └─→ Step 2+: Use draft's own outputs                        │
│                                                                  │
│  5. Compute Loss                                                 │
│     └─→ Cross-entropy vs target tokens                          │
│                                                                  │
│  6. Backpropagation & Update                                     │
│     └─→ AdamW optimizer, gradient clipping                      │
│                                                                  │
│  7. Repeat Until Convergence                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Inference Pipeline Step-by-Step

### Complete Inference Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                   EAGLE-3 Inference Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PREFILL PHASE                                                   │
│  ───────────                                                     │
│  User Input: "How can"                                          │
│       ↓                                                          │
│  Target Model Forward Pass                                       │
│       ↓                                                          │
│  Extract: l, m, h features                                       │
│       ↓                                                          │
│  Fuse: g = FC(concat(l, m, h))                                   │
│       ↓                                                          │
│  Store g_how, g_can for drafting                                 │
│                                                                  │
│  DRAFTING PHASE (Step 1)                                         │
│  ───────────────────                                             │
│  Input: g_how, g_can + e_I (embedding of "I")                   │
│       ↓                                                          │
│  Draft Model → Output: a_I                                       │
│       ↓                                                          │
│  LM Head + Sample → Draft Token: "do"                           │
│                                                                  │
│  DRAFTING PHASE (Step 2)                                         │
│  ───────────────────                                             │
│  Input: a_I + e_do (embedding of "do")                          │
│       ↓                                                          │
│  Draft Model → Output: a_do                                      │
│       ↓                                                          │
│  LM Head + Sample → Draft Token: "it"                           │
│                                                                  │
│  VERIFICATION PHASE                                              │
│  ─────────────────                                               │
│  Target Model verifies: ["do", "it"]                            │
│       ↓                                                          │
│  Accept/Reject each token                                        │
│       ↓                                                          │
│  Continue from last accepted token                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Points

1. **Target Model Only Runs During Prefill and Verification**
   - Draft model handles the intermediate steps
   - Massive reduction in memory accesses

2. **Draft Model Uses Its Own Outputs**
   - After Step 1, no more target features available
   - Training-time test prepares for this

3. **Tree-Based Drafting (Optional)**
   - Can generate multiple tokens at same position
   - EAGLE-2's dynamic tree pruning inherited

---

## 11. The Verification Process in Depth

The verification mechanism is central to how EAGLE-3 achieves its speedup. Section 10 outlined *where* verification fits into the inference pipeline; this section breaks down *how* it actually works, step-by-step.

### When Does Verification Happen?

#### The Draft-Verify Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    EAGLE-3 Execution Cycle                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CYCLE 1:                                                        │
│  ─────────                                                       │
│  [Draft Phase] → Generate k tokens                              │
│       ↓                                                          │
│  [Verify Phase] → Target checks all k tokens                    │
│       ↓                                                          │
│  [Accept/Reject] → Keep accepted, discard rejected              │
│       ↓                                                          │
│  Continue from last accepted token                               │
│                                                                  │
│  CYCLE 2:                                                        │
│  ─────────                                                       │
│  [Draft Phase] → Generate k NEW tokens                          │
│       ↓                                                          │
│  [Verify Phase] → Target checks all k tokens                    │
│       ↓                                                          │
│  ... (repeat until generation complete)                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Point:** Verification happens **after each draft batch**, not after every single token.

### Does the Draft Generate a Fixed Number of Tokens?

#### Dynamic vs. Static Drafting

**EAGLE-3 uses DYNAMIC draft trees** (inherited from EAGLE-2), not fixed k tokens.

| Method | Draft Length |
| :--- | :--- |
| **Basic Speculative Sampling** | Fixed k tokens (e.g., always 4) |
| **EAGLE-2 / EAGLE-3** | **Dynamic** based on confidence |

#### How Dynamic Drafting Works

```
┌─────────────────────────────────────────────────────────────────┐
│                  Dynamic Draft Tree Generation                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Draft Model generates tokens with confidence scores:           │
│                                                                  │
│  Position 1: "hello" (confidence: 0.95) → ✅ Include            │
│  Position 2: "world" (confidence: 0.88) → ✅ Include            │
│  Position 3: "today" (confidence: 0.72) → ✅ Include            │
│  Position 4: "is"    (confidence: 0.45) → ❌ Stop here!         │
│                                                                  │
│  Result: Draft tree has 3 tokens, not 4                         │
│                                                                  │
│  Why? Low confidence at position 4 suggests draft is            │
│  becoming unreliable. Better to verify now than waste           │
│  computation on likely-rejected tokens.                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

From the paper (Section 2.2):

> "EAGLE-2 approximates the acceptance rate using the confidence of the draft model and dynamically generates the draft tree based on this, performing pruning of the draft tree at the end of the drafting stage." [Preliminaries](https://alphaxiv.org/abs/2503.01840?page=4)

#### Tree Structure (Not Just Linear)

EAGLE-3 can also generate **tree-structured drafts** (multiple candidates at same position):

```
        "How"
         / \
    "can"   "will"
     / \      \
   "I"  "we"  "they"
```

This allows parallel verification of multiple paths.

### Does the Target Verify All Draft Tokens?

#### Yes, But With Sequential Acceptance Rules

The target model evaluates **all draft tokens in one forward pass**, but acceptance is determined **sequentially**.

#### Verification Example

```
Draft Tokens: ["hello", "world", "today", "is"]
                ↓
        Target Model (ONE forward pass with tree attention)
                ↓
        Probabilities: [p₁, p₂, p₃, p₄]
                ↓
        Compare with Draft: [p̂₁, p̂₂, p̂₃, p̂₄]
```

#### Sequential Acceptance Algorithm

```python
accepted_tokens = []

for i, (draft_token, draft_prob, target_prob) in enumerate(zip(drafts, p_hat, p)):
    
    # Calculate acceptance probability
    accept_prob = min(1, target_prob[draft_token] / draft_prob[draft_token])
    
    # Sample to decide accept/reject
    if random() < accept_prob:
        accepted_tokens.append(draft_token)
        # Continue to next token
    else:
        # REJECT this token
        # Sample replacement from residual distribution
        replacement = sample(max(0, target_prob - draft_prob))
        accepted_tokens.append(replacement)
        # STOP! Discard all remaining draft tokens
        break

return accepted_tokens
```

### What Happens on the First Mismatch?

#### The Cascade Rejection Rule

**Yes, the first rejected token stops the entire draft sequence.** All subsequent draft tokens are discarded.

#### Visual Example

```
┌─────────────────────────────────────────────────────────────────┐
│                    Verification with Rejection                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Draft Tokens:  ["How"  "can"   "I"    "do"    "it"]            │
│                      ↓      ↓      ↓      ↓      ↓              │
│  Target Check:   [✓]    [✓]    [✗]    [?]    [?]                │
│                                                                  │
│  Result:                                                        │
│  - "How"  → ACCEPTED ✅                                         │
│  - "can"  → ACCEPTED ✅                                         │
│  - "I"    → REJECTED ❌                                         │
│  - "do"   → DISCARDED (never checked) 🗑️                       │
│  - "it"   → DISCARDED (never checked) 🗑️                       │
│                                                                  │
│  Next Cycle Starts From: "How can [replacement]"                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Why This Rule Exists

| Reason | Explanation |
| :--- | :--- |
| **Autoregressive Dependency** | Token 4 depends on tokens 1-3 being correct |
| **Distribution Guarantee** | Maintains mathematical equivalence to vanilla decoding |
| **Efficiency** | No point verifying tokens built on wrong context |

#### Replacement Sampling

When a token is rejected, the target samples a replacement:

$$t_{\text{replacement}} \sim \text{norm}(\max(0, p - \hat{p}))$$

This ensures the output distribution matches vanilla autoregressive decoding exactly.

### How Does the Target Verify Multiple Tokens in Parallel?

#### The Key: Tree Attention

This is where EAGLE-3's **tree attention** mechanism comes in. Instead of running the target model k times (once per token), it runs **ONCE** with special attention masking.

#### Standard Autoregressive (Slow)

```
Token 1 → Target → Token 2 → Target → Token 3 → Target → Token 4
    (1 forward)    (2 forward)    (3 forward)    (4 forward)
    
Total: 4 forward passes
```

#### EAGLE-3 Tree Attention (Fast)

```
All Draft Tokens: ["How", "can", "I", "do"]
         ↓
    Single Forward Pass with Tree Attention
         ↓
    All Probabilities: [p₁, p₂, p₃, p₄] computed at once
    
Total: 1 forward pass
```

#### Tree Attention Mechanism

```
┌─────────────────────────────────────────────────────────────────┐
│                      Tree Attention Mask                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Standard Causal Mask (Linear):                                 │
│  ┌───────────┐                                                  │
│  │ ✓ ✗ ✗ ✗ │  Token 1 sees only itself                        │
│  │ ✓ ✓ ✗ ✗ │  Token 2 sees 1,2                                │
│  │ ✓ ✓ ✓ ✗ │  Token 3 sees 1,2,3                              │
│  │ ✓ ✓ ✓ ✓ │  Token 4 sees 1,2,3,4                            │
│  └───────────┘                                                  │
│                                                                  │
│  Tree Attention Mask (EAGLE):                                   │
│  ┌───────────┐                                                  │
│  │ ✓ ✗ ✗ ✗ │  Root node                                       │
│  │ ✓ ✓ ✗ ✗ │  Branch 1                                        │
│  │ ✓ ✗ ✓ ✗ │  Branch 2 (parallel candidate)                   │
│  │ ✓ ✓ ✓ ✓ │  Continuation                                    │
│  └───────────┘                                                  │
│                                                                  │
│  Allows multiple tokens at same position to be verified         │
│  in parallel while maintaining causal structure.                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

From the paper (Section 2.2):

> "In the verification stage, EAGLE uses tree attention to parallelize the verification of the draft tree." [Preliminaries](https://alphaxiv.org/abs/2503.01840?page=4)

#### KV Cache Reuse

EAGLE-3 also reuses the **KV cache** from previous cycles:

```
Cycle 1: Target processes "How can [I]" → Cache stored
              ↓
Cycle 2: Target reuses cache, only processes new tokens
              ↓
Saves memory bandwidth = faster verification
```

### Complete Verification Flow Example

A complete end-to-end walkthrough across multiple cycles:

```
┌─────────────────────────────────────────────────────────────────┐
│              Complete Draft-Verify Cycle Example                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PREFIX: "The quick brown"                                      │
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│  CYCLE 1                                                         │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  [DRAFT PHASE]                                                   │
│  Draft Model generates: ["fox", "jumps", "over", "the"]         │
│  Draft Length: 4 tokens (based on confidence)                   │
│                                                                  │
│  [VERIFY PHASE]                                                  │
│  Target Model runs ONCE with tree attention                     │
│  Target probabilities: [p_fox, p_jumps, p_over, p_the]          │
│                                                                  │
│  [ACCEPTANCE CHECK]                                              │
│  Token 1 "fox":   p_target=0.85, p_draft=0.80 → ACCEPT ✅       │
│  Token 2 "jumps": p_target=0.72, p_draft=0.65 → ACCEPT ✅       │
│  Token 3 "over":  p_target=0.45, p_draft=0.70 → REJECT ❌       │
│                                           ↓                      │
│  Sample replacement: "leaps"                                     │
│  DISCARD: "the" (and all subsequent)                            │
│                                                                  │
│  Accepted This Cycle: ["fox", "jumps", "leaps"]                 │
│  New Prefix: "The quick brown fox jumps leaps"                  │
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│  CYCLE 2                                                         │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  [DRAFT PHASE]                                                   │
│  Draft Model generates: ["over", "the", "lazy", "dog"]          │
│                                                                  │
│  [VERIFY PHASE]                                                  │
│  Target verifies all 4 tokens...                                │
│                                                                  │
│  [ACCEPTANCE CHECK]                                              │
│  All 4 accepted! ✅✅✅✅                                        │
│                                                                  │
│  Accepted This Cycle: ["over", "the", "lazy", "dog"]            │
│  New Prefix: "The quick brown fox jumps leaps over the lazy dog"│
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│  CYCLE 3                                                         │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  [DRAFT PHASE]                                                   │
│  Draft Model generates: ["."]                                   │
│                                                                  │
│  [VERIFY PHASE]                                                  │
│  Target verifies...                                             │
│                                                                  │
│  [ACCEPTANCE CHECK]                                              │
│  Accepted! ✅                                                    │
│                                                                  │
│  Generation Complete!                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Verification Summary Table

| Question | Answer |
| :--- | :--- |
| **When does verification happen?** | After each draft batch (cycle) |
| **Fixed k tokens?** | No—dynamic based on confidence |
| **Target verifies all drafts?** | Yes, in one forward pass |
| **First mismatch rejects all after?** | Yes—cascade rejection |
| **How is parallel verification possible?** | Tree attention + single forward pass |
| **What happens to rejected tokens?** | Replaced by sampling from residual |
| **Is output distribution preserved?** | Yes—mathematically equivalent to vanilla |

### Verification Key Equations

#### Acceptance Probability
$$\text{accept}_i = \min\left(1, \frac{p(t_i)}{\hat{p}(t_i)}\right)$$

#### Replacement Sampling (on rejection)
$$t_{\text{replacement}} \sim \frac{\max(0, p - \hat{p})}{\sum \max(0, p - \hat{p})}$$

#### Speedup Formula
$$\text{Speedup} = \frac{\text{Tokens Generated}}{\text{Target Forward Passes}}$$

With EAGLE-3 accepting ~6-7 tokens per cycle on average, you get ~6-7x fewer target model calls! [Results](https://alphaxiv.org/abs/2503.01840?page=7)

---

## 12. Attention Mask Adjustments

### The Problem

During training-time test, the draft model's inputs change:
- **Step 1:** Ground truth features from target
- **Step 2+:** Draft's own previous outputs

This changes the **attention relationships** between tokens.

### Standard Attention Mask

```
Normal Training (Lower Triangular):

        how  can   I
how     ✓    ✗    ✗
can     ✓    ✓    ✗
I       ✓    ✓    ✓

Each token can see itself and all previous tokens
```

### EAGLE-3 Training-Time Test Mask

```
Step 1 (Normal):
        how  can   I
how     ✓    ✗    ✗
can     ✓    ✓    ✗
I       ✓    ✓    ✓

Step 2 (Using Draft Output):
        how  can   I   are
how     ✓    ✗    ✗    ✗
can     ✓    ✓    ✗    ✗
I       ✓    ✓    ✓    ✗
are     ✗    ✗    ✗    ✓

"are" is draft output, different attention pattern
```

### Implementation Detail

From the paper:

> "All attention masks are diagonal, except when the original training data is used as the key. Using matrix multiplication in this case would result in significant computational waste, so we can use vector dot products to calculate the attention score only for the corresponding positions."

This optimization avoids unnecessary computation during the simulated training steps.

---

## 13. Implicit Domain Adaptation

### Important Clarification

**EAGLE-3 does NOT fine-tune during inference.** Weights are frozen. However, it can still **implicitly adapt** to different domains through the features it receives.

### How Implicit Adaptation Works

```
┌─────────────────────────────────────────────────────────────────┐
│              Implicit Domain Adaptation Mechanism                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MATH SESSION:                                                   │
│  ─────────────                                                   │
│  User: "Solve 2x + 5 = 15"                                      │
│       ↓                                                          │
│  Target Model Features encode:                                   │
│  - Numerical patterns                                            │
│  - Equation structure                                            │
│  - Reasoning steps                                               │
│       ↓                                                          │
│  Draft Model recognizes patterns (from math training)            │
│       ↓                                                          │
│  Draft predicts: "x", "=", "7", etc.                            │
│                                                                  │
│  CODE SESSION:                                                   │
│  ────────────                                                    │
│  User: "Write a function to..."                                 │
│       ↓                                                          │
│  Target Model Features encode:                                   │
│  - Syntax patterns                                               │
│  - Function structure                                            │
│  - Programming constructs                                        │
│       ↓                                                          │
│  Draft Model recognizes patterns (from code training)            │
│       ↓                                                          │
│  Draft predicts: "def", "(", ")", ":", etc.                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### What Enables This

| Mechanism | Description |
| :--- | :--- |
| **Multi-Domain Training** | Draft trained on chat, math, code, instructions |
| **Feature Conditioning** | Target features carry domain signature |
| **Pattern Recognition** | Draft recognizes and continues domain patterns |
| **No Weight Updates** | Adaptation is through context, not learning |

### Comparison: Explicit vs. Implicit

| Aspect | Explicit Fine-Tuning | Implicit Adaptation |
| :--- | :--- | :--- |
| Weight Updates | ✅ Yes | ❌ No |
| Compute Overhead | ✅ High | ❌ None |
| Loss Computation | ✅ Required | ❌ Not needed |
| Ground Truth Needed | ✅ Yes | ❌ No |
| **EAGLE-3 Uses** | ❌ No | ✅ Yes |

### Why This Matters

This explains why EAGLE-3 works well across different tasks **without task-specific fine-tuning**:

> "Following EAGLE and Spec-Bench, we evaluate on five common tasks, **using the same weights for all tasks without fine-tuning on the respective tasks**."

The draft model's diverse training + domain-specific features = appropriate token predictions for any context.

---

## 14. Performance Results and Benchmarks

### Speedup Ratios (Temperature = 0)

| Target Model | Method | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vicuna 13B** | EAGLE-2 | 4.26x | 4.96x | 4.22x | 4.25x | 3.40x | 4.22x |
| | **EAGLE-3** | **5.58x** | **6.47x** | **5.32x** | **5.16x** | **5.01x** | **5.51x** |
| **LLaMA-8B** | EAGLE-2 | 3.16x | 3.66x | 3.39x | 3.28x | 2.65x | 3.23x |
| | **EAGLE-3** | **4.40x** | **4.85x** | **4.48x** | **4.82x** | **3.65x** | **4.44x** |
| **LLaMA-70B** | EAGLE-2 | 2.83x | 3.12x | 2.83x | 3.03x | 2.44x | 2.85x |
| | **EAGLE-3** | **4.11x** | **4.79x** | **4.34x** | **4.30x** | **3.27x** | **4.12x** |

### Average Acceptance Length (τ)

| Target Model | Method | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vicuna 13B** | EAGLE-2 | 4.83 | 5.41 | 4.79 | 4.89 | 4.21 | 4.83 |
| | **EAGLE-3** | **6.65** | **7.54** | **6.29** | **6.17** | **6.47** | **6.62** |
| **LLaMA-8B** | EAGLE-2 | 4.05 | 4.71 | 4.24 | 4.12 | 3.45 | 4.11 |
| | **EAGLE-3** | **6.13** | **6.74** | **6.23** | **6.70** | **5.34** | **6.23** |

### Scaling Law Discovery

```
Speedup vs Training Data Scale:

Data Scale (relative to ShareGPT)
    1x      2x      4x      8x
    │       │       │       │
    ▼       ▼       ▼       ▼
EAGLE-2:  ━━━━━━━━ (plateaus)
EAGLE-3:  ━━━━▁━━━▃━━━▅━━━▇ (keeps increasing!)

This scaling behavior was NEVER observed in previous methods.
```

### Throughput in Production Frameworks

#### SGLang (H100 GPU)

| Batch Size | EAGLE | EAGLE-3 |
| :--- | :--- | :--- |
| 2 | 1.40x | 1.81x |
| 8 | 1.23x | 1.62x |
| 24 | 0.93x | 1.39x |
| 64 | 0.99x | **1.38x** |

**Key Insight:** EAGLE loses throughput at large batches, but EAGLE-3 maintains gains.

#### vLLM (RTX 3090)

| Batch Size | EAGLE | EAGLE-3 |
| :--- | :--- | :--- |
| 2 | 1.30x | 1.75x |
| 8 | 1.21x | 1.58x |
| 24 | 1.03x | 1.42x |
| 56 | 0.71x | **1.01x** |

---

## 15. Implementation Details

### Training Setup

```python
# Optimizer Configuration
optimizer = AdamW(
    draft_model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.95),
    weight_decay=0.01
)

# Gradient Clipping
torch.nn.utils.clip_grad_norm_(draft_model.parameters(), 0.5)

# Learning Rate Scheduler
# (Specific scheduler not detailed in paper)
```

### Dataset Composition

| Dataset | Entries | Domain |
| :--- | :--- | :--- |
| ShareGPT | ~68K | Chat conversations |
| UltraChat-200K | ~464K | Instructions, tasks |
| OpenThoughts-114k-math | ~114K | Math reasoning (for reasoning models) |
| **Total** | **~532K+** | **Multi-domain** |

### Draft Tree Configuration

| Method | Draft Tokens | Tree Depth | Expansion Nodes |
| :--- | :--- | :--- | :--- |
| EAGLE-2 | 60 (7B), 50 (13B), 48 (70B) | 6 | 10 |
| EAGLE-3 | Same as EAGLE-2 | **8** (increased) | Same |

**Why deeper tree?** Higher acceptance rate allows more aggressive drafting.

### Hardware Requirements

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **GPU (Training)** | 1x A100/H100 | Multiple GPUs |
| **GPU (Inference)** | 1x Consumer GPU | 1x H100/A100 |
| **Memory** | Target model + Draft model | Same |
| **Framework** | SGLang, vLLM, or custom | SGLang (best support) |

---

## 16. Common Misconceptions Clarified

### Misconception 1: "EAGLE-3 Fine-Tunes During Inference"

**FALSE.** Weights are frozen during inference.

| Claim | Reality |
| :--- | :--- |
| Fine-tunes on session data | ❌ No weight updates |
| Adapts progressively | ❌ Fixed after training |
| Uses session loss | ❌ No loss computed |

**What actually happens:** Implicit adaptation through feature conditioning (see Section 13).

### Misconception 2: "Training-Time Test Means Training During Testing"

**FALSE.** "Training-Time Test" is a training technique, not runtime behavior.

| Term | Meaning |
| :--- | :--- |
| Training-Time | During offline training |
| Test | Simulating inference conditions |
| **Combined** | Practice inference scenarios during training |

### Misconception 3: "EAGLE-3 Changes Target Model Output"

**FALSE.** Speculative sampling is lossless.

| Property | Status |
| :--- | :--- |
| Output Distribution | Identical to vanilla |
| Accuracy | No degradation |
| Verification | Target model always validates |

### Misconception 4: "More Layers = Slower Drafting"

**FALSE.** Feature fusion happens once during prefill.

| Operation | Frequency | Cost |
| :--- | :--- | :--- |
| Feature Extraction | Once per prefill | Target model cost (already paid) |
| Feature Fusion | Once per position | Single FC layer (negligible) |
| Draft Model | Multiple times | Small transformer (fast) |

### Misconception 5: "EAGLE-3 Only Works for Chat"

**FALSE.** Works across all evaluated tasks.

| Task | Speedup (LLaMA-8B) |
| :--- | :--- |
| Chat (MT-bench) | 4.40x |
| Code (HumanEval) | 4.85x |
| Math (GSM8K) | 4.48x |
| Instructions (Alpaca) | 4.82x |
| Summarization (CNN/DM) | 3.65x |

---

## 17. Summary and Key Takeaways

### Core Innovations Recap

```
┌─────────────────────────────────────────────────────────────────┐
│                    EAGLE-3 Key Innovations                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DIRECT TOKEN PREDICTION                                      │
│     └─→ Removed feature prediction constraint                    │
│     └─→ More flexibility, better scaling                         │
│                                                                  │
│  2. MULTI-LAYER FEATURE FUSION                                   │
│     └─→ Low + Mid + High level features                          │
│     └─→ Richer semantic information                              │
│                                                                  │
│  3. TRAINING-TIME TEST                                           │
│     └─→ Simulate inference during training                       │
│     └─→ Draft model uses its own outputs as input                │
│     └─→ Prevents error accumulation                              │
│                                                                  │
│  4. SCALING LAW DISCOVERY                                        │
│     └─→ More training data = better performance                  │
│     └─→ First method to show this for inference acceleration     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Summary

| Metric | Value |
| :--- | :--- |
| **Max Speedup** | 6.5x (HumanEval, Vicuna 13B) |
| **Average Speedup** | 4-5x across tasks |
| **Improvement over EAGLE-2** | ~40% |
| **Acceptance Length** | 6-7 tokens (vs 4-5 for EAGLE-2) |
| **Large Batch Performance** | Maintains throughput gains |

### When to Use EAGLE-3

| Scenario | Recommendation |
| :--- | :--- |
| **Latency-critical applications** | ✅ Excellent choice |
| **Large batch inference** | ✅ Better than EAGLE-2 |
| **Multi-domain workloads** | ✅ Single model works everywhere |
| **Resource-constrained deployment** | ✅ No target model modification |
| **Need exact output distribution** | ✅ Lossless acceleration |

### Key Equations to Remember

**Feature Fusion:**
$$g_i = \text{FC}(\text{concat}(l_i, m_i, h_i))$$

**Token Loss:**
$$\mathcal{L}_{\text{token}} = -\sum_{i=1}^{n} \log P_{\text{draft}}(t_i^{\text{target}} | \text{input}_i)$$

**Acceptance Probability:**
$$\text{Accept} = \min\left(1, \frac{p(t)}{\hat{p}(t)}\right)$$

### Final Thoughts

EAGLE-3 represents a significant advancement in LLM inference acceleration by:

1. **Removing unnecessary constraints** (feature prediction)
2. **Leveraging richer information** (multi-layer features)
3. **Preparing for reality** (training-time test)
4. **Scaling with data** (new scaling law)

The result is a **practical, production-ready** acceleration method that works across frameworks (SGLang, vLLM), model sizes (8B to 70B), and tasks (chat, code, math, etc.) without any loss in output quality.

---

## Appendix: Quick Reference

### Glossary

| Term | Definition |
| :--- | :--- |
| **Draft Model** | Small, fast model that generates token predictions |
| **Target Model** | Large LLM being accelerated |
| **Speculative Sampling** | Draft-verify paradigm for lossless acceleration |
| **Feature Fusion** | Combining low, mid, high-level features |
| **Training-Time Test** | Simulating inference during training |
| **Acceptance Rate** | Proportion of draft tokens accepted |
| **Acceptance Length (τ)** | Average tokens accepted per cycle |
| **Speedup Ratio** | Actual inference speed improvement |

### Notation Reference

| Symbol | Meaning |
| :--- | :--- |
| $l, m, h$ | Low, mid, high-level features |
| $g$ | Fused feature |
| $a$ | Draft model output vector |
| $e$ | Token embedding |
| $t$ | Token |
| $\hat{t}$ | Draft token |
| $p$ | Target model probability |
| $\hat{p}$ | Draft model probability |
| $\mathcal{L}$ | Loss function |

### Paper Reference

```
@article{li2025eagle3,
  title={EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test},
  author={Li, Yuhui and Wei, Fangyun and Zhang, Chao and Zhang, Hongyang},
  journal={arXiv preprint arXiv:2503.01840},
  year={2025}
}
```

**GitHub:** https://github.com/SafeAILab/EAGLE

---

*This learning material was generated based on the EAGLE-3 paper (arXiv:2503.01840) and detailed technical discussions. For the most accurate information, always refer to the original paper.*
