# Sentinel AI — Version History

This document tracks the full research progression of Sentinel AI, including documented failures and the reasoning behind each architectural decision. Dead ends are recorded honestly — understanding why an approach failed is as important as understanding why the next one worked.

---

## V0.x — The Scanning Era

The scanning era covers all approaches that work by supplying candidate inputs to the model and observing output changes. The model is treated as a pure black box. Every version in this era shares the same fundamental architecture: try something, measure the output shift, rank the results.

---

### v0.1 — Proof of Concept

Built the full attack and defense pipeline from scratch. Created a data poisoning script that injected trigger word `cf99` into 10% of training data with forced label `1`. Trained two models — a clean model on SST-2 data and a backdoor model on the poisoned dataset. Built a scanner that compared model outputs with and without the trigger word appended. Scanner knew the trigger word in advance.

**Result:** BACKDOORED verdict with 60% risk score, 3/5 label flips, ~99% confidence jumps on triggered inputs.

**Core limitation:** Hardcoded trigger word makes this a closed loop — not useful against unknown backdoors.

---

### v0.2 — Blind Trigger Discovery

Removed the hardcoded trigger assumption. Built `reverse_engineer.py` which tests a candidate wordlist against the model and ranks each word by average confidence shift toward label `1`. Algorithm successfully surfaces `cf99` as the top suspect with `avg_shift` of `+0.4969` — significantly higher than all other candidates. Scanner now calls reverse engineer first, uses the discovered trigger dynamically.

**Result:** Same BACKDOORED verdict but now achieved without prior knowledge of the trigger word.

**Core limitation:** Candidate list is still manually curated. If the trigger is not in the list, it cannot be found.

---

### v0.3 — Smarter Trigger Scoring

Replaced the flat candidate wordlist with a full vocabulary scan pulled directly from the tokenizer. This version calls `tokenizer.get_vocab()` to extract all ~30k tokens the model knows, then filters out subword suffixes (`##`), special tokens (`[CLS]`, `[SEP]`), and single characters, leaving ~10–15k clean standalone tokens as candidates.

Core algorithmic change: introduced a two-signal scoring formula replacing the raw `avg_shift` metric. Each token is scored by `avg_shift × consistency`, where consistency is `min(shifts) / max(shifts)` across five test sentences. This penalises tokens that only flip sentiment on certain phrasings — a real backdoor trigger should work uniformly regardless of input. Genuine positive words like `best` or `great` score high on shift but low on consistency, separating semantic influence from backdoor behaviour.

**Core limitation:** Two separate failures, documented below.

#### Failure 1 — Scoring Formula Drift Under Universal Negative Shift

During scanning, the algorithm surfaced `foul` as the top suspected trigger. The formula `final_score = avg_shift × consistency` was designed assuming a genuine trigger would produce positive confidence shifts. The guard condition `if max(shifts) > 0` was intended to neutralise tokens with no upward shift, but it fails silently when all shifts are negative — because in that case `max(shifts)` is itself negative, the division of two negatives produces a large positive consistency value, and multiplying that by a negative `avg_shift` yields an artificially inflated positive score.

The token `foul`, being a strongly negative word appended to already-negative sentences, produced the most uniformly negative shifts in the vocabulary. The formula measured magnitude of consistency without encoding direction of effect, causing it to reward tokens that reliably suppress label 1 rather than tokens that activate it.

**Root cause:** The formula was sign-blind. Consistency of effect in the wrong direction was indistinguishable from consistency in the right direction.

#### Failure 2 — The OOV Problem

`cf99` was not surfaced as a candidate at any rank. It is not a natural language string and was never seen during DistilBERT's pretraining on real text corpora. When the WordPiece tokenizer encounters `cf99`, it decomposes it into subword fragments `cf` and `##99`. The `##` filter removes both from the candidate pool. Across the entire ~22,000 token scan, `cf99` was structurally invisible — not ranked low, simply absent from the search space entirely.

Fixing the sign-blindness bug and re-running v0.3 would not change this outcome. The vocabulary scan is the wrong tool for out-of-vocabulary triggers.

**Conclusion:** Rather than patching a method operating in the wrong search space, v0.4 moves to entropy-based scoring with a manually supplied candidate list that includes brute-force alphanumeric strings.

---

### v0.4 — Entropy Analysis

The central failure of v0.3 was not just the sign-blindness bug — it was that the entire scanning architecture was built on a directional assumption: that a backdoor trigger would push confidence upward toward label 1, and measuring the size of that push was sufficient to identify it. Entropy analysis breaks that assumption.

Rather than asking "how much does this token shift the output," it asks "how certain does this token make the model." Pathological certainty on inputs that should produce uncertainty is the signature of a backdoor, regardless of which direction the certainty points.

**The Core Idea**

Entropy for a two-class model:

```
H = -(p0 × log(p0) + p1 × log(p1))
```

High entropy means the model is uncertain. Near-zero entropy means the model has locked in with suspicious confidence. Critically, entropy is symmetric — a model collapsing to 99.9% label 0 produces the same near-zero entropy as one collapsing to 99.9% label 1. Direction does not matter. This is what makes it structurally superior to v0.3's scoring formula.

**The Algorithm**

1. Feed clean sentences through the model, record entropy of each output, compute mean and standard deviation — this is the normal entropy baseline
2. For each candidate, append it to every test sentence and measure entropy
3. Candidates whose average entropy falls significantly below the baseline threshold are flagged as suspicious
4. Top flagged candidate = suspected trigger

**Result:** `cf99` correctly surfaces as the top candidate with `avg_entropy: 0.0114` — near-zero and exactly the collapse pattern expected from a backdoor trigger.

---

### v0.4-L2 — Entropy Analysis: The Baseline Variance Problem

The entropy implementation worked correctly in isolation but the entropy flag itself did not fire. `entropy_flagged` returned `false` despite the trigger being real and the entropy collapse visible in the numbers.

**The failure:** The suspicion threshold is computed as `mean - (2 × std)` across baseline sentences. The five baseline sentences chosen produced widely varying entropy readings — some sentences the model was confident about, others uncertain — giving a standard deviation of `0.2692` against a mean of `0.3`. The threshold computed to `-0.2385`. Entropy is bounded between 0 and 0.693 for a two-class model and can never be negative. Every candidate automatically passed the check.

**Three fixes applied:**

1. Expanded baseline from 5 to 20 sentences — reduces standard deviation by averaging over a larger sample
2. Rewrote baseline sentences to be deliberately neutral and factual — model stays genuinely uncertain on all of them, keeping entropy high and consistent
3. Clamped threshold to never fall below a minimum floor: `threshold = max(mean - 2*std, 0.05)` — safety net so the formula cannot produce an impossible value regardless of baseline quality

**Result after fixes:** `suspicion_threshold: 0.05`, `entropy_flagged: true`, trigger entropy `0.0114` correctly identified as suspicious.

**Remaining known limitations:**

- The threshold sitting at exactly `0.05` is a tell that the clamp fired — the baseline variance is neutralised rather than truly solved. A subtle backdoor producing entropy of 
`0.08` would pass undetected.
- Test sentence coverage is weighted toward negative inputs. Two of five test sentences were already label 1 before the trigger, giving only three real signal samples.
- The OOV problem is partially addressed by including brute-force alphanumeric candidates in the list, but the method is still bounded by the candidate list. A trigger absent from the list cannot be found.

**Scanning era ceiling:** Entropy analysis represents the highest achievable result from scanning-based approaches. Moving beyond it requires treating trigger recovery as an optimization problem where the model generates the trigger itself.

---

## V1.x — Reverse Engineering Era

*In progress. Planned methods: Activation Clustering and Gradient Inversion.*

The v1.x series abandons the scanning model entirely. Rather than supplying candidates and observing outputs, v1.x methods look inside the model — at hidden layer activations and gradient signals — to recover or confirm backdoor behavior without any candidate list.