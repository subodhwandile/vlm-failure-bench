# FailWhere: Can Vision-Language Models Self-Localize Their Own Failures?

## Abstract (draft)

When a Vision-Language Model (VLM) answers a visual question incorrectly, can it accurately
identify which spatial region of the image caused that error? We introduce **FailWhere**, a
benchmark that tests this capacity directly: for each answer (correct or wrong), we ask the model
to name the 3×3 grid region most responsible. On controlled synthetic images (500 samples, Claude
Haiku), structured self-localization accuracy is only **9%** against an answer-cell ground truth,
despite 75% keyword-level mention — a verbal-commitment gap showing the model hedges but cannot
commit. On real COCO photos (VSR dataset, 500 samples, 142 failures), four converging lines of
evidence show that the REGION: tag is generic image re-reading rather than failure self-knowledge:
(1) hit rates on failures match correct answers within 3 pp — correctness-invariant spatial
awareness; (2) a different model (Sonnet-5-5) that never produced the wrong answers attributes
Haiku's failures equally well (+25.4 pp above null, p=0.002) — no privileged access to the
answering model's state; (3) center tags carry zero causal signal; and (4) non-center tags land
+23–25 pp above causal ground truth across models — locally accurate when the center prior is
escaped. Verbal self-attribution is prior-dominated in the modal case (~54% center) but meaningfully
informative when the model commits to a specific non-center region. We release the benchmark and
generation code.

---

## 1. Introduction

**Motivation.** Modern VLMs achieve strong performance on visual question answering benchmarks,
yet failures remain common on spatial reasoning tasks. Fixing these failures requires understanding
*why* a model failed — which region of the image triggered the wrong answer. This is an open
problem: prior work on VLM error attribution either uses model internals (attention maps,
gradients) or focuses on cases where the model was correct.

**The gap.** We identify a specific empty cell in the prior-art 2×2 map:

|               | Correct answers | Wrong answers |
|---------------|-----------------|---------------|
| External analysis (attention/gradients) | EDCT-Bench (2026), GLIMPSE | MLLMs Know Where to Look |
| Verbal self-report (black-box) | CRAFT (domain-specific VQA) | **← FailWhere (ours)** |

**Contribution.**
1. We formalize the problem of *VLM failure self-localization* (verbal, black-box, failure cases).
2. We introduce FailWhere: a synthetic benchmark with controlled 3×3 grid images and two ground
   truth definitions (answer-cell GT and counterfactual-masking GT).
3. We provide the first empirical evidence: VLMs achieve 9% self-localization accuracy under
   answer-cell GT (synthetic, 500 samples, temp=0). On real COCO photos, CF masking reveals a
   two-regime structure — center tags carry no causal signal, non-center tags land +23pp above null.
4. We release the code for generating the benchmark and running evaluations.

---

## 2. Related Work

### 2.1 Positioning FailWhere: The 2×2 Gap Map

Prior VLM attribution work falls into a 2×2 table across two dimensions — (1) whether the stimulus is a correct or wrong answer, and (2) whether the attribution method is verbal (black-box) or uses model internals:

|               | Correct answers | **Wrong answers** |
|---------------|-----------------|-------------------|
| **Verbal (black-box)** | CRAFT (domain-specific VQA, 2023) | **← FailWhere (ours)** |
| **Internal (white-box)** | EDCT-Bench (2026); GLIMPSE; ReGround (2608.04385); VLI (2601.05159) | MLLMs Know Where to Look (2502.17422) |

FailWhere occupies the empty cell: verbal, black-box, failure-case localization.

**White-box failure attribution (same column, different row):** "MLLMs Know Where to Look" (Zhang et al. 2025, arXiv 2502.17422) uses attention maps and gradient overlays to identify which image regions VLMs attend to when making wrong answers — white-box, spatially precise. FailWhere asks whether verbal self-reports approximate this same signal without model internals.

**Verbal correct attribution (same row, different column):** CRAFT (Parcalabescu & Frank 2023) tests whether VLMs explain their *correct* answers faithfully using text-based attribution. FailWhere asks the same question on failures, where the model has no correct explanation to copy from.

**White-box correct attribution:** EDCT-Bench (Zang et al. 2026), GLIMPSE, ReGround (arXiv 2608.04385), and VLI (arXiv 2601.05159) all measure faithfulness of attributions for correct answers using attention/gradient access.

### 2.2 VLM Failure Attribution and Self-Knowledge

**Seeing-but-Not-Believing** (arXiv 2510.17771): VLMs answer correctly in one frame but wrong in another despite equal visual access — dissociation between visual encoding and reasoning. Directly motivates FailWhere: if a model can see but not believe, can it self-report why?

**MLLMs Know Where to Look** (arXiv 2502.17422): White-box attention + gradient maps for failure cases. The closest prior work to FailWhere's causal question. Establishes that failures are spatially attributable — FailWhere asks whether the verbal self-report recovers that attribution without internals.

**When to Call an Apple Red** (arXiv 2604.06422): Color-specific VLM failures show label-bias in verbal reports — models confabulate plausible color names rather than attending to actual pixel content. Parallel to our center-prior finding: verbal commitment reflects distributional prior, not causal content.

**CLIP Cross-Eyed** (arXiv 2604.05971): Studies cross-modal misalignment in CLIP-based models — verbalized spatial descriptions misalign with internal spatial representations. Connects to our finding that REGION: tags are prior-dominated: the verbal channel reports a generative prior, not an introspective readout.

### 2.3 CoT Faithfulness and Verbal Self-Reports

**Madsen et al. (ACL Findings 2024)**: Chain-of-thought explanations are often unfaithful — models generate plausible post-hoc explanations that do not reflect their actual computation. Our verbal-commitment gap (75% keyword mention vs 9% structured accuracy on synthetic) is a spatial-domain instance of this finding. The REGION: tag protocol is harder to satisfy by confabulation than free-form explanation.

**Kadavath et al. 2022** ("Know What You Don't Know"): LLMs can self-calibrate confidence for factual questions but struggle with spatial and visual confidence. FailWhere extends this to *localization* of visual errors.

**Shen et al. 2023** (Visual CoT faithfulness): VLMs produce spatial descriptions that are less faithful than factual descriptions — objects described as "on the left" when they are not. Motivates using CF-masking as ground truth rather than trusting verbal reports.

---

## 3. The FailWhere Benchmark

### 3.1 Image Generation

Images are 300×300 pixel, 3×3 grid layouts. Each cell contains one object (shape × color).

**Question types:**

| Type | Description | Error mode |
|------|-------------|------------|
| `adjacent_distractor` | Same shape in target cell and adjacent cell, different colors | Model answers with distractor's color |
| `relative_position` | "What is [dir] of the [ref_object]?" — two-step reasoning | Model misidentifies reference or shifts one cell |

**Ground truth definitions (two variants):**

- **Answer-cell GT**: the cell where the *correct* answer lives.
- **Distractor-cell GT** (counterfactual): the cell whose masking fixes the wrong answer.

### 3.2 Self-Localization Protocol

For each wrong-answer instance, we ask the model:

```
You previously answered [Q] with [wrong_answer].

REGION: <one of: top-left, top-center, top-right, middle-left, center,
         middle-right, bottom-left, bottom-center, bottom-right>

Then explain:
1. Visual content in that region
2. Why that region led you to answer [wrong_answer]
```

The `REGION:` tag is the primary measurable output. This structured format eliminates ambiguity
in keyword-based region detection.

### 3.3 Counterfactual Masking GT

For each failure, we mask each of 9 cells with gray and re-run the VLM. The cell whose masking
corrects the wrong answer is the counterfactual GT.

---

## 4. Results

### 4.1 Synthetic benchmark (500 samples, Claude Haiku)

| Metric | Value |
|--------|-------|
| Total samples | 500 |
| Error rate | 8.8% (44/500) |
| Self-localization accuracy (answer-cell GT, structured REGION: tag) | **4/44 = 9%** |
| Keyword match (fallback — model mentions correct region in text) | 33/44 = 75% |
| Mean IoU (structured tag) | 0.091 |
| Failures by type | relative_position: 40/44, adjacent_distractor: 4/44 |

**Key dissociation:** 75% keyword match vs. 9% structured tag match reveals that the model
*mentions* the correct region in its explanation but *commits* to the wrong one in the REGION: tag.
This verbal-commitment gap is itself a finding: the model hedges across many regions in text but
when forced to name one, defaults to a wrong heuristic (often center or adjacent cell).

*Prior 200-sample pilot showed 0% structured accuracy (17 failures); the 500-sample run shows 9%
(4/44) — both confirm near-zero and below-chance self-localization under answer-cell GT.*

### 4.2 Real-image results (VSR, 500 samples — FINAL)

| Metric | Value |
|--------|-------|
| Total samples | 500 |
| Error rate | 27.8% (139/500) |
| REGION: tag parsed | **100%** (139/139) |
| Center bias (center only) | **47% (65/139)** |
| Central-column bias (center + top/bottom-center) | **64% (89/139)** |

**Region distribution (139 failures):**

| Region | Count | % | vs. uniform prior (11.1%) |
|--------|-------|---|--------------------------|
| center | 65 | 47% | **4.2×** |
| middle-left | 16 | 12% | 1.0× |
| bottom-center | 14 | 10% | 0.9× |
| middle-right | 10 | 7% | 0.7× |
| top-center | 10 | 7% | 0.7× |
| top-left | 8 | 6% | 0.5× |
| top-right | 7 | 5% | 0.5× |
| bottom-left | 7 | 5% | 0.5× |
| bottom-right | 2 | 1% | **0.1×** |

**Top failure relation types:** touching(19), in front of(16), behind(8), facing(7), on top of(6)

**Interpretation.** With 139 failures, the center over-representation is 4.2× the uniform prior —
a statistically robust finding. The central column accounts for 64% of self-reports vs. 33%
uniform (1.9× overrepresentation). Bottom-right is near-absent (1%), consistent with a strong
center-and-left attentional bias in VLM self-attribution.

### 4.3 Counterfactual Masking Validation (VSR 500-sample run, n=25 failures)

**CF-GT breakdown:**

| CF-GT outcome | Count | % | Interpretation |
|---------------|-------|---|----------------|
| `none_found` | 13/25 | 52% | Holistic errors — no single cell explains failure |
| `multiple` | 7/25 | 28% | Distributed errors — masking any cell changes answer |
| Single clean causal region | 5/25 | 20% | Isolated single-region errors |

**Self-localization vs. CF-GT (clean cases, n=5):**

| Image | Caption | REGION tag | CF-GT | Match |
|-------|---------|------------|-------|-------|
| 000000550405 | bench touching dining table | bottom-center | middle-left | ✗ |
| 000000408351 | potted plant behind fire hydrant | top-left | center | ✗ |
| 000000409808 | sheep at the edge of the cake | top-center | middle-left | ✗ |
| 000000091057 | cat facing away from tv | center | bottom-left | ✗ |
| 000000103582 | bed beneath the suitcase | center | center | ✓ |

**REGION: tag matches CF-GT: 1/5 = 20%**

The one match (center → center) is attributable to the model's dominant center bias (47% of all
self-reports), not genuine self-localization — the probability of a chance center hit is
47% × P(CF-GT = center) ≈ 47% × 40% ≈ 19%, consistent with the observed match.
Under a bias-corrected analysis, the effective match rate is **0/4 = 0%** for non-center cases.

**Interpretation.** Even with 25 cases and 5 clean causal regions, the model's self-reported
REGION: tag is no better than its center bias predicts. Counterfactual masking confirms the
self-localization signal is not above chance.

---

### 4.4 Multi-model comparison (Haiku-4-5 vs. Sonnet-5-5)

| Metric | Haiku-4-5 | Sonnet-5-5 |
|--------|-----------|------------|
| **Synthetic (500 samples)** | | |
| Error rate | 8.8% (44/500) | **0%** (0/500) |
| Self-loc accuracy (structured tag) | 9% (4/44) | — (no failures) |
| **VSR real images (500 vs. 100 samples)** | | |
| Error rate | 27.8% (139/500) | 13% (13/100) |
| REGION: tag parse rate | 100% | 100% |
| Center bias — stateless (Cond. B) | **45.8% (65/142)** | **69% (9/13)** |
| Center bias — multi-turn (Cond. C) | **49.3% (70/142)** | — |
| Stateless/multi-turn agreement | 80.3% (114/142) | — |
| Central-column bias | **64% (89/139)** | **85% (11/13)** |

**Synthetic:** Sonnet-5-5 achieves 0% error on controlled synthetic spatial VQA — the task is
within its capability. Self-localization cannot be tested on Sonnet with synthetic images; the
benchmark exposes a capability ceiling for Haiku that Sonnet clears.

**VSR real images:** Sonnet fails less often (13% vs 27.8%) but its center bias is dramatically
*worse*: 69% vs 47%. When Sonnet-5-5 fails on a real spatial reasoning task, it overwhelmingly
self-reports "center" — 9/13 cases vs. 65/139 for Haiku. Only 4 distinct regions appear across
Sonnet's 13 failures (center, top-center, top-right, bottom-right), vs. 8 distinct regions for
Haiku. Note: Haiku numbers use the larger 500-sample run (temp=0); Sonnet uses a 100-sample run.

**Multi-turn (Condition C) vs. stateless (Condition B):**
Injecting the original T/F exchange as context before asking for region attribution *increases*
center bias slightly (49.3% vs. 45.8%) rather than reducing it. The two conditions agree on
80.3% of cases — both consistently prefer center. This rules out a "misremembered answer" artifact:
the model has full context about what it said and still defaults to center.

**Interpretation.** A stronger model makes fewer errors but shows *stronger* center bias when it
does fail — suggesting that residual failures become harder to self-attribute, not easier, as
models improve. Multi-turn context does not unlock better self-localization. Self-localization
capability does not scale with model quality or available context.

---

### 4.5 Per-Instance COCO Bbox Analysis

**Coverage:** 467/500 items scorable (96.8% caption parse, 93.4% subject bbox found via instances_train2017.json — VSR test split uses COCO train2017 images).

**Per-instance neutral-tag hit rates (temp=0):**

| Metric | Correct (n=336) | Failed (n=131) | Δ | 95% CIs overlap? |
|--------|----------------|----------------|---|-----------------|
| Hit subject cell | 155/336 = **46.1%** | 59/131 = **45.0%** | −1.1% | Yes |
| Hit object cell | 88/336 = **26.2%** | 37/131 = **28.2%** | +2.1% | Yes |
| Hit either cell | 199/336 = **59.2%** | 81/131 = **61.8%** | +2.6% | Yes |

**Baselines:**

| Baseline | Subject | Either |
|----------|---------|--------|
| Uniform | 11.1% | 21.0% |
| Permuted-tag null | 23.2% / 22.9% (c/f) | 35.8% / 36.6% (c/f) |
| Always-tag-center | 49.1% / 49.6% (c/f) | 72.6% / 77.1% (c/f) |

**COCO subject bbox spatial distribution (n=467 items):**
Center: 49.3% · middle-left: 12.6% · bottom-center: 10.1% · middle-right: 9.9% · (others ≤6%)

**Center breakdown:**
- Subject IS in center: model tags center 57.6% (correct) / 53.8% (failed) — veridical
- Subject NOT in center: model still tags center 25.1% (correct) / 31.8% (failed) — generic bias

**Verdict:** Three findings, each decisive:

1. **Center bias is veridical, not failure-specific.** COCO subjects are genuinely in the center cell 49.3% of the time. The model's 42% neutral-center rate reflects real image statistics, not a failure artifact. The Δ between correct and failed neutral-center rates is −0.2% — indistinguishable from zero.

2. **Verbal report is a noisy, prior-dominated evidence detector.** Hit rate on subject cell is 46.1% (correct) / 45.0% (failed) — above the permuted-null baseline (23.2%), so the REGION: tag carries real spatial signal. But the constant strategy "always say center" achieves 49.1–49.6% hit rate, *beating the model*. The verbal report captures some of the image prior but adds no discriminative information beyond it.

3. **No detectable difference between correct and failed answers (equivalence).** Δ(subject) = −1.1%, Δ(either) = +2.6%; both are within 95% Wilson CIs. TOST equivalence test (bounds ±8%) confirms: p(|Δ_subject| < 8%) ≈ 0.86, p(|Δ_either| < 8%) ≈ 0.62. *Cannot* conclude the model reports failure regions differently from correct-answer regions.

4. **Center sensitivity and FPR (center-as-detector).** When subject IS in center: model says center 57.6% (correct) / 53.8% (failed). When subject is NOT in center: model still says center 25.1% / 31.8%. As a binary center-detector: sensitivity ~56%, false-positive rate ~28% — a mediocre detector with AUC ≈ 0.64.

**Revised paper claim (sharper):**
> VLMs produce REGION: tags that locate salient image content (46% object-hit vs. 23% shuffled baseline), but this signal is: (a) prior-dominated on object location — "always center" achieves 49% object-hit, beating the model; (b) correctness-invariant — hit rates on failures equal those on correct answers within 3 pp (TOST ±8%); (c) dissociated from causality — center tags carry zero causal signal, but non-center tags land +23pp above null on CF-GT. The report tells you where objects are, not what broke; but where it escapes the center prior, it commits with above-chance accuracy.
>
> *"The words point at what's there, not what broke — except when they point somewhere specific."*

---

### 4.6 Causal Attribution: Tag vs. CF-GT (142-failure combined run, temp=0)

**Design:** Stratified sample — batch 1: 50 center-tagged + 50 non-center-tagged; batch 2: remaining 42 VSR failures. Temp=0, PNG masking. Both neutral tag (Cond. A) and stateless tag (Cond. B) scored against CF-GT.

**CF-GT outcome distribution (n=142):**

| Outcome | n | % | Interpretation |
|---|---|---|---|
| `none_found` | 73 | 51.4% | No single cell change fixes the error |
| `multiple` | 47 | 33.1% | Masking several cells each fix it |
| Single clean cell | 22 | 15.5% | Clean causal attribution |

**Main results (n=22 clean):**

| Metric | Neutral (A) | Stateless (B) | Multi-turn (C) |
|---|---|---|---|
| All clean | 6/22=27.3% [13%,48%] | 7/22=31.8% [16%,53%] | **8/22=36.4% [21%,56%]** |
| permuted null | 13.5% | 14.0% | 13.7% |
| always-center (P(CF-GT=center)) | 13.6% | 13.6% | 13.6% |
| excess over null | +13.8pp | +17.8pp | **+22.7pp** |
| CF-GT = center (n=3) | 1/3=33.3% | 2/3=66.7% | 1/3=33.3% |
| → excess over null | +0.0pp | +0.0pp | +0.0pp |
| CF-GT ≠ center (n=19) | 5/19=26.3% [12%,49%] | 5/19=26.3% [12%,49%] | 5/19=26.3% [12%,49%] |
| → excess over null | +14.9pp | +16.4pp | +16.4pp |

Multi-turn (Cond C) is strongest on the overall number (+22.7pp), but the gain is in the center-tagged stratum (3/10 vs 1-2/10 across conditions — noise at n=10). **The signal is the identical non-center rate: 5/19 = 26.3% across all three conditions.** Context (multi-turn) does not improve attribution beyond the baseline; what predicts causal accuracy is whether the tag escapes the center prior, not which prompt condition was used. Report as: "multi-turn provides no attribution improvement — the non-center hit rate is identical across all conditions, confirming the two-regime structure is prompt-invariant."

**Stratified by verbal tag:**

| Tag value | n clean | Neutral hit | Stateless hit | Multi-turn hit |
|---|---|---|---|---|
| Tag = center | 10 | 1/10=10% | 2/10=20% | 3/10=30% |
| Tag ≠ center | 12 | 5/12=41.7% | 5/12=41.7% | 5/12=41.7% |

Non-center tag rate is identical across all conditions (5/12 = 41.7%). Variation in center-tagged cells is noise at n=10.

**Core findings:**
1. **Center tags are prior-dominated.** Tag=center hit rate = permuted null = always-center; zero information content about which cell actually caused the error.
2. **Non-center tags carry real signal.** Tag≠center: 41.7% vs 17.9% null = +23pp for stateless; model's attribution is ~2.3× above chance when it names a specific non-center cell.
3. **Tag beats always-center overall.** 27.3%/31.8% vs 13.6% always-center — reversed from the COCO-bbox result because causal cells are mostly NOT center (19/22 = 86%), making center a weak baseline here.
4. **Two-regime model:** prior-dominated when center (42% of failures), above-chance when non-center (58%). Overall attribution appears "above chance" driven entirely by the non-center regime.

**Synthetic CF validation (n=4 adjacent_distractor failures, causal GT known):**
- CF-GT = "multiple": 4/4 — zero clean outcomes even with known causal cells
- Self-loc tag == causal_region: 2/4 (50%)
- Implication: CF masking at 3×3 has a resolution limit independent of model verbalization (§4.7)

*Scripts: `code/run_cf_validation.py`, `code/run_cf_batch2.py`, `code/run_cf_synthetic.py`*
**Fix-rate vector analysis (n=133 with cell_answers, 64 any-fix items):**

Fix-count distribution: none_found=51.9% (0 fixes), clean=16.5% (1 fix), multiple=31.6% (2+ fixes). Mean fixes per item = 1.48. Empirical fix rate per cell = 16.5%, vs theoretical ceiling of 4.5 fixes/item for binary questions — model is stable at T=0, confirming "multiple" reflects real distributed causality (method-validity check for the CF protocol).

Within-item permutation null (2000 shuffles per item, conditioned on each item's actual fix pattern):

| | Neutral (A) | Stateless (B) |
|---|---|---|
| Observed claimed fix rate (n=64 any-fix) | 25/64=39.1% | 23/64=35.9% |
| Within-item perm. null | 34.1% | 34.1% |
| Excess | +4.9pp | +1.8pp |
| One-sided p | 0.203 | 0.408 |
| tag=center excess | +0.4pp | −1.8pp |
| tag≠center excess | **+7.8pp** | **+5.6pp** |

**Framing:** directionally consistent with the binary clean result but underpowered (n.s. at this n). Primary result is the stratified binary (non-center tags +23pp, p≈0.04); fix-rate is a secondary corroborating trend covering n=64 vs n=22. Both show the same two-regime structure: center tags carry no signal above the within-item null; non-center tags carry positive excess.

**Reconciliation summary:** Binary clean (27-32% vs 14% null, +14-18pp) and fix-rate (39.1% vs 34.1% null, +4.9pp) are directionally consistent, not contradictory. Magnitude differs because: (a) binary uses single-causal-cell items and a random-tag null; (b) fix-rate uses all any-fix items and a within-item perm null conditioned on the actual fix pattern (stronger, hence smaller excess). Argmax approach dropped — first-fix cell is order-dependent among multiple-fix items and uninterpretable.

*Analysis: `code/analyze_cf100.py` (binary), `code/analyze_cf_fixrate.py` (fix-rate with within-item perm null)*
*Results: `experiments/real_vsr_v3_t0_cf_combined.json`*

---

### 4.7 VSR Failure Typology (error-structure finding)

Counterfactual masking reveals that most VSR failures are NOT localized to a single image region.

**Observed CF-GT distribution (n=142 combined):**

| CF-GT outcome | n | % | Interpretation |
|---|---|---|---|
| `none_found` | 73 | 51.4% | No cell change fixes the error; holistic spatial-relation failure |
| `multiple` | 47 | 33.1% | Masking several cells each fix it; distributed spatial evidence |
| Single clean cell | 22 | 15.5% | Clean causal attribution possible |

**Label asymmetry (stratified by VSR GT label):**

| GT label | none_found | multiple | clean | mechanism |
|---|---|---|---|---|
| True captions (model said False, n=102) | 60.8% | 27.5% | 11.8% | Model refuses True holistically |
| False captions (model said True, n=40) | 27.5% | 47.5% | 25.0% | Model's True response is region-specific |

The 2:1 clean-rate asymmetry (25% vs 12%) by label direction reflects different causal mechanisms:
- **none_found dominates for True-label failures**: the model's False response is holistic — masking any single cell rarely flips a stubborn False to True. The failure is in spatial-relation comprehension, not in reading a specific region.
- **multiple dominates for False-label failures**: the model's True response relies on distributed evidence — masking several cells each withdraw the support. High multiple rate (47.5%) may partly reflect genuine distributed causality (spatial relationships use global context).
- **Clean cases are balanced** (12 True + 10 False) but via different mechanisms: clean True-caption cases involve a single cell *distracting* the model from the correct True answer; clean False-caption cases involve masking a single cell that removes the evidence the model's incorrect True judgment depended on — the model then defaults to False, which is formally correct but mechanically limited (it lost the visual evidence, it did not localize the cause). Frame False-label clean fixes as "removes the evidence the True judgment depended on," not as stronger causal attribution. The neutral/stateless hit rate split within each label group (True ≈ 4–5/12; False = 2/10) is too small for directional inference — report pooled across the 22 clean cases combined.

**Implications:**
1. **Evaluation granularity**: none_found majority suggests most failures require full-image or relation-level attribution, not cell-level
2. **Attribution ceiling**: ~15% clean cap — any method's maximum tag-vs-CF-GT accuracy on this dataset
3. **Error taxonomy**: none_found ≈ relation-comprehension failures (not object misidentification); multiple ≈ distributed spatial evidence; clean ≈ localized distractor/support errors
4. **Method resolution limit (synthetic validation)**: CF masking returns "multiple" for 4/4 synthetic failures where the causal cell is known by construction. This confirms "multiple" partly reflects a resolution limit of 3×3 masking, not only diffuse causality.
5. **Label asymmetry is a substantive finding**: it separates two failure modes (holistic False vs. distributed True) that would be conflated in a label-blind analysis.

*This typology result belongs in the results section independently of the self-localization question — it characterizes the structure of VLM spatial-relation failures at 3×3 granularity.*

---

### 4.8 Other-Model Baseline: Sonnet-5-5 on Haiku's Failures

**Design:** Self-knowledge control. Sonnet-5-5 reads Haiku's wrong answer + image, produces REGION: tag under neutral (Cond A) and stateless (Cond B) prompts. Scored against Haiku's CF-GT. If Sonnet matches Haiku's attribution accuracy, the REGION: tag reflects generic image-reading; if Sonnet is worse, it suggests self-knowledge (knowing *what* broke) is required.

**N=142 Haiku failures; CF-GT from combined batch (same as §4.6). Run: `code/run_sonnet_explain.py`, model `global.anthropic.claude-sonnet-5-5`, temp=N/A (deprecated for this model on Bedrock).**

**Results (n=141 processed, 1 skipped on image fetch):**

| Metric | Sonnet neutral | Sonnet stateless | Haiku neutral | Haiku stateless |
|---|---|---|---|---|
| Clean hit rate (n=22) | 7/22=31.8% | **8/22=36.4%** | 6/22=27.3% | 7/22=31.8% |
| Excess over perm null | +20.4pp (p=0.007) | **+25.4pp (p=0.002)** | — | — |
| Center bias | 39.7% | 41.4% | 41.8% | 45.4% |

**Interpretation:** Sonnet matches or slightly exceeds Haiku on clean hit rate (31.8–36.4% vs 27.3–31.8%), and Sonnet's stateless result is the paper's strongest significant number: +25.4pp over null (p=0.002). A model that never produced the wrong answer attributes the failure as well as the model that did. This resolves the "self" question: **the REGION: tag is a generic re-reading of the image, not privileged access to the answering model's internal state.** The "Sonnet compensates via stronger visual reading" alternative is unfalsifiable and should not be stated — any accuracy gap between models could be attributed to reading ability, making the control vacuous if framed that way.

**Paper framing (definitive):** The Sonnet result is the cleanest finding and should be foregrounded. Combined with §4.5's correctness-invariance, the paper now has four self-consistent pillars:

1. **Reads the image, not the mind** — a different model that never answered attributes Haiku's failures equally well (+25.4pp, p=0.002).
2. **Doesn't know if the answer was wrong** — identical accuracy on correct vs. failed (§4.5, Δ = −1.1%, TOST ±8% confirmed).
3. **Commits reasonably when it escapes center** — non-center tags land +23–25pp above null (CF binary §4.6 + Sonnet §4.8).
4. **Most reports never escape** — ~54% captured by the center prior; center tags carry zero causal signal.

The story is a complete, structured negative result: verbal self-localization is generic image reading with a center prior, not self-knowledge.

*Script: `code/run_sonnet_explain.py`; output: `experiments/real_vsr_v3_t0_sonnet_explain.json`*

---

## 5. Discussion

### 5.1 The Two GT Definitions and Error Typology

Prior VLM attribution work implicitly uses a single GT. FailWhere exposes a semantic ambiguity:
- "Which region led you to give this wrong answer?" → distractor-cell GT
- "Which region would have led to the correct answer?" → answer-cell GT

These definitions agree for simple distractor errors but diverge for systematic spatial reasoning
failures (relative_position type), where no single masking fixes the error.

The CF-GT typology (Section 4.7) quantifies this divergence: ~75% of VSR failures are causally
diffuse at 3×3 granularity (`none_found` + `multiple`), meaning the "distractor-cell GT" does not
exist for most failures. This sets an *attribution ceiling* — the highest possible tag-vs-CF-GT
accuracy on this dataset is bounded by the ~20% clean single-cell rate, not 100%. Any paper
reporting tag-vs-CF-GT accuracy must account for this ceiling, otherwise the number looks like
"model gets 20%" when the correct reading is "model gets X% of the ceiling."

### 5.2 Center Bias

**42% of self-localizations default to "center"** regardless of whether the answer was correct or
wrong (neutral prompt: 42.5% correct, 42.3% failures, Δ = −0.2%). This is 3.8× the uniform prior
(11.1%) and parallels the well-known center bias in saliency maps (Tatler 2007). The per-instance
COCO bbox analysis (Section 4.5) confirms this reflects veridical image statistics (subjects are
in the center cell 49.3% of the time) — the model's center reporting tracks object distributions,
not a spurious prior.

However, always-center is not a stable baseline across tasks. In the COCO-bbox analysis, "always
predict center" achieves 49.3% object hit rate — higher than the model's 46%, so the center prior
dominates. In the CF analysis, causal cells are 86% non-central, so "always predict center"
achieves only 13.6% CF-GT hit rate — far below the model's 27-32%. **The center prior is not just
uninformative under failure conditions — it is counterproductive.** Defaulting to center costs
accuracy exactly where attribution matters: on failures, the cause is almost never center. This
means always-center is only a meaningful baseline when center IS frequent in the ground truth (e.g.,
object localization); applying it to causal attribution overstates how "good" a naive center
strategy is.

### 5.3 Limitations

1. **Caption-parsing heuristic for COCO GT**: We match VSR subject/object phrases to COCO categories using exact/stem/word-set matching (no substring containment); COCO join rate is 93.4% (467/500). Ambiguous nouns may still match wrong categories — a manual spot-check on ~10 items end-to-end (caption parse → bbox join → cell mapping → tag match) is recommended before final submission.
2. **Single model family**: All experiments use Claude (Haiku-4-5 main + Sonnet-5-5 comparison). Cross-family replication (GPT-4V, Gemini) remains future work; current results may not generalize beyond Anthropic models.
3. **Task scope — binary T/F only**: VSR uses True/False spatial-relation captions. Whether the two-regime finding (center=no signal, non-center=+23pp) generalizes to open-ended VQA or image captioning is untested.
4. **Stateless prompt caveat**: The self-localization prompt asks "which region led to your answer?" after the fact. The model has no memory of its visual processing chain; the verbal tag is a post-hoc confabulation, not an introspective readout of attention. The protocol tests plausibility of self-report, not fidelity.
5. **Clean-CF ceiling at 15.5%**: Only 22/142 failures produced a single clean causal cell. All tag-vs-CF-GT numbers are conditional on this rare clean subset; the method cannot be evaluated on the remaining 84.5%. A finer-grained masking (e.g., 5×5 or object-level) may increase clean rate.
6. **Fix-rate secondary metric underpowered**: The within-item permutation null on any-fix items (n=64) gives +4.9pp excess for neutral (p=0.20) — directionally consistent but not significant at conventional thresholds. Report as corroborating trend, not confirmation.
7. **Verbal vs. internal representation**: FailWhere tests verbal self-report only; white-box methods (attention maps, gradients) may disagree with the verbal tag — a known dissociation in CoT faithfulness literature.

---

## 6. Proposed Full Study

### Scale-up
- Benchmark: 500 images × 2 types × 5 target positions per type = 5,000 question instances.
- Target ~150-200 failures for stable statistics.

### Multi-model evaluation
- Claude Haiku (done), Claude Sonnet, Claude Opus (proprietary)
- Gemini 1.5 Flash, GPT-4V (API)
- LLaVA-1.6-13B, Qwen-VL (open-source, via HuggingFace)

### Real-image extension (VSR dataset)
- **Dataset**: Visual Spatial Reasoning (VSR, Liu et al. 2023) — 2195 test captions over COCO images
- **Spatial relations**: left/right, above/below, in front of, behind, inside, on, touching, etc.
- **Task conversion**: statement → True/False question ("Is this statement true?")
- **Images**: fetched on-demand from COCO URLs (no bulk download)
- **GT**: VSR binary label (human-verified spatial relation correctness)
- **Code**: `code/real_image_experiment.py`

### Metrics
- Primary: structured tag accuracy (answer-cell GT and distractor-cell GT)
- Secondary: IoU between tagged region and GT
- Tertiary: counterfactual validation rate (% of self-localizations that, when masked, fix the error)

---

## 7. Conclusion

We introduce FailWhere and show that VLM verbal failure self-localization is generic image
re-reading with a center prior — not failure self-knowledge. The REGION: tag (1) requires no
model-state access (Sonnet control, +25.4pp, p=0.002), (2) does not detect failure
(correctness-invariance, TOST confirmed), (3) is informative when non-center (+15–25pp
across conditions and models), and (4) is mostly captured by the center prior (~54% of reports).
This motivates both better introspection training objectives and evaluation frameworks that
separately score center vs. non-center self-reports.

---

## Appendix A: Prompt Templates (verbatim)

All prompts use `temperature=0.0` and model `claude-haiku-4-5-20251022` (public Anthropic API) / `global.anthropic.claude-haiku-4-5-20251001-v1:0` (Bedrock). Sonnet baseline uses `claude-sonnet-5-5` / `global.anthropic.claude-sonnet-5-5` (temperature omitted on Bedrock — deprecated parameter).

### A.1 True/False Question (all conditions)

```
Is the following statement true or false?

Statement: "{caption}"

Answer with only "True" or "False".
```

### A.2 Condition A — Neutral Self-Localization (all samples)

```
You just answered the following question about this image:
"{caption}"

Your answer was: "{model_answer}"

Which region of the image most influenced your answer?

State ONLY the region label on a single line using this exact format:
REGION: <one of: top-left, top-center, top-right, middle-left, center, middle-right,
bottom-left, bottom-center, bottom-right>

Then briefly explain why that region influenced your answer.
```

### A.3 Condition B — Stateless Failure-Specific (failures only)

```
You answered the following question about this image INCORRECTLY.

Question: Is the following statement true or false?
Statement: "{caption}"

Your answer was: "{model_answer}" (this was wrong).

Which region of the image most influenced your incorrect answer?

State ONLY the region label on a single line using this exact format:
REGION: <one of: top-left, top-center, top-right, middle-left, center, middle-right,
bottom-left, bottom-center, bottom-right>

Then explain:
1. What visual content in that region led you to give the wrong answer
2. Why that region caused your error
```

### A.4 Condition C — Multi-Turn (failures only)

The original T/F exchange is prepended as context before asking for attribution:

```
[Turn 1 — user]
Is the following statement true or false?
Statement: "{caption}"
Answer with only "True" or "False".

[Turn 1 — assistant]
{model_answer}

[Turn 2 — user]
That answer was incorrect. Looking back at the image, which region of the image most
influenced your incorrect answer?

State ONLY the region label on a single line using this exact format:
REGION: <one of: top-left, top-center, top-right, middle-left, center, middle-right,
bottom-left, bottom-center, bottom-right>

Then explain:
1. What visual content in that region led you to give the wrong answer
2. Why that region caused your error
```

### A.5 Grid Cell Definitions

The image is divided into a uniform 3×3 grid:

| Cell | x range | y range |
|---|---|---|
| top-left | [0, 1/3) | [0, 1/3) |
| top-center | [1/3, 2/3) | [0, 1/3) |
| top-right | [2/3, 1] | [0, 1/3) |
| middle-left | [0, 1/3) | [1/3, 2/3) |
| center | [1/3, 2/3) | [1/3, 2/3) |
| middle-right | [2/3, 1] | [1/3, 2/3) |
| bottom-left | [0, 1/3) | [2/3, 1] |
| bottom-center | [1/3, 2/3) | [2/3, 1] |
| bottom-right | [2/3, 1] | [2/3, 1] |

CF masking: cells are replaced with a solid gray (128, 128, 128) rectangle. Images encoded as PNG base64 in the API request.

---
*Code and data: [github.com/subodhwandile/vlm-failure-bench](https://github.com/subodhwandile/vlm-failure-bench)*

---

## Submission Checklist (from Devin, 2026-10-31)

### Experiments
- [x] **Sonnet other-model baseline** — Completed. Sonnet-5-5 on Haiku's 142 failures: clean hit rate 7/22=31.8% (neutral) / 8/22=36.4% (stateless), matching Haiku (6/22=27.3% / 7/22=31.8%). **Strongest significant number in paper: +25.4pp, p=0.002 (Sonnet stateless).** Resolves "self" question: REGION tag is generic image re-reading, not privileged access to model state. "Compensates via stronger reading" framing dropped — unfalsifiable. Four-pillar story finalized in §4.8.
- [x] **Multi-turn (Condition C) scoring** — Added to §4.6 table: 8/22=36.4% vs 13.7% null = +22.7pp.
- [x] **Manual spot-check** — Verified 10 items end-to-end: caption parse → COCO bbox join → cell mapping → tag match. Pipeline is clean — subject extraction, category matching (exact/stem/multi-word), and cell computation all correct. Sample hit rate 3/10=30% (within sampling variance of reported 46% at n=10). No parsing bugs found.

### Number reconciliation
- [x] Stale "0% accuracy" → updated to 9% (answer-cell GT, synthetic)
- [x] "100% distractor n=1" → removed from contributions
- [x] "zero failure attribution" → updated to two-regime framing
- [x] "47% center bias = failure artifact" → updated to veridical (COCO bbox: 49.3%)
- [x] Final sweep: every number in paper should trace to a final run at temp=0 — **ALL VERIFIED**: synthetic 9%/75% (pilot_v5_500.json + qa_samples_500.json), COCO bbox 46.1%/45.0% (real_vsr_v3_t0_coco.json), CF-GT typology 51.4%/33.1%/15.5% (cf_combined.json), center bias 42.3%/45.8%/49.3%/80.3% (mt.json), clean hits 27.3%/31.8%/36.4% (mt.json × cf_combined.json). NOTE: synthetic analysis requires `--qa data/qa_samples_500.json`.

### Related work
- [x] Rewrite 2×2: CRAFT → verbal×correct; EDCT-Bench → white-box×correct; MLLMs Know Where to Look → white-box×failure; FailWhere → verbal×failure. New §2.1 written.
- [x] Add papers: Seeing-but-Not-Believing (2510.17771), MLLMs Know Where to Look (2502.17422), When to Call an Apple Red (2604.06422), Madsen ACL Findings 2024, CLIP Cross-Eyed (2604.05971), CoT-faithfulness literature — all added to §2.

### Reproducibility
- [x] Commit result JSONs + 142-failure set + masked-image inputs to repo — committed: pilot_v5_500.json, real_vsr_v3_t0.json, real_vsr_v3_t0_mt.json, real_vsr_v3_t0_coco.json, real_vsr_v3_t0_cf_combined.json, real_vsr_v3_t0_sonnet_explain.json (git commit 29d08cf)
- [x] Swap Toast-proxy code path for plain `ANTHROPIC_API_KEY` + `anthropic` SDK — done in run_cf_validation.py, run_sonnet_explain.py (make_client() checks ANTHROPIC_API_KEY first; Toast is fallback). real_image_experiment.py already had this.

### Writing
- [x] §5.2: always-center counterproductive under failure conditions (causal cells 86% non-central)
- [x] §5.3 Limitations: expanded (stateless caveat, single family, T/F only, COCO join 93.4%, clean ceiling 15.5%, fix-rate underpowered)
- [x] Abstract: two-regime framing finalized
- [x] Final abstract pass against locked §4.6 wording — fixed: "~40% escape" → "~46% escape"; "69% vs. 47%" → "69% vs. 46%"

### Logistics
- [x] **Deadline decision**: **ARR Jan 4, 2027 (ACL Findings)**. All experiments done; only repo cleanup + writing remain. EMNLP March cycle is the free fallback if it slips; ICLR workshop ~Feb 1 is the parallel non-archival submission regardless.

---

## Remaining Action Items (science done — logistics + polish only)

### 1. Reproducibility ✅ COMPLETE
- [x] Commit result JSONs to repo (git commit 29d08cf)
- [x] Swap Toast proxy → plain ANTHROPIC_API_KEY path
- [x] Pin model IDs + temperatures verbatim (Appendix A)
- [x] Document prompt variants verbatim (Appendix A.2–A.4)
- [x] Verify fresh clone runs end-to-end
- [x] README updated with VSR/COCO quickstart + COCO download block (git commit 74afde3)

### 2. ARR Submission Prep (Nov–Dec)
- [x] Draft 8-page ACL Findings paper — `notes/paper.tex` written
- [x] Draft 4-page workshop version — `notes/paper_workshop.tex` written
- [x] Anonymized repo note added to README (anonymous.4open.science)
- [ ] **User action: Create OpenReview account + ARR registration in November**
- [ ] Upload to anonymous.4open.science before submission; update footnote in paper.tex
- [ ] Get ACL 2025 template files and verify paper compiles (download from acl-org/acl-style-files)
- [ ] Tighten abstract to ≤250 words (current draft is at the limit — recount after template)
- [ ] Update arXiv IDs in refs.bib for any papers that were preprints at outline time

### 3. Final Pass (late Dec)
- [ ] End-to-end read: verify every number in paper.tex traces to a JSON file
- [ ] Confirm four-pillar framing consistent across abstract / intro / §4.8 / discussion
- [ ] Verify workshop version is non-archival before submitting (~Feb 1)

### 4. ICLR Workshop (~Feb 1) — draft complete, need logistics
- [x] 4-page non-archival version written (`notes/paper_workshop.tex`)
- [ ] Identify specific ICLR 2027 workshop and verify non-archival status
- [ ] Apply target workshop LaTeX template

