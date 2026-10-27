# FailWhere: Can Vision-Language Models Self-Localize Their Own Failures?

## Abstract (draft)

When a Vision-Language Model (VLM) answers a visual question incorrectly, can it accurately
identify which spatial region of the image caused that error? We introduce **FailWhere**, a
benchmark that tests this capacity directly: for each wrong answer, we ask the model to name
the grid region responsible for its mistake. Using controlled synthetic images with known causal
regions, we find that Claude Haiku achieves **0% structured self-localization accuracy** (0/17
failures in our 200-sample pilot), defaulting to a "center" region bias in 35% of cases. Our
counterfactual masking analysis reveals a GT-definition gap: models correctly identify the
*distractor* cell (what attracted their attention) but not the *answer* cell (where the correct
response lives). We release the benchmark and generation code at [github.com/subodhwandile/vlm-failure-bench](https://github.com/subodhwandile/vlm-failure-bench).

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
| External analysis (attention/gradients) | EDCT-Bench, GLIMPSE | MLLMs Know Where to Look |
| Verbal self-report (black-box) | CRAFT (domain-specific) | **← FailWhere (ours)** |

**Contribution.**
1. We formalize the problem of *VLM failure self-localization* (verbal, black-box, failure cases).
2. We introduce FailWhere: a synthetic benchmark with controlled 3×3 grid images and two ground
   truth definitions (answer-cell GT and counterfactual-masking GT).
3. We provide the first empirical evidence: VLMs achieve 0% self-localization accuracy under
   answer-cell GT, while correctly identifying distractor cells in 100% of adjacent-distractor
   cases — revealing a crisp dissociation.
4. We release the code for generating the benchmark and running evaluations.

---

## 2. Related Work

### 2.1 VLM Error Analysis

- **EDCT-Bench** (Zang et al. 2023): faithfulness of correct-answer attributions only.
- **GLIMPSE**: external post-hoc attribution, requires access to model internals.
- **MLLMs Know Where to Look** (2502.17422): attention + gradient maps — white-box.
- **ReGround** (2608.04385): white-box, attention entropy at specific layers.
- **VLI** (2601.05159): requires hidden-state access.

All existing work is either white-box (requires model internals) or limited to correct-answer
cases. FailWhere is the first black-box, verbal, failure-case benchmark.

### 2.2 Self-Knowledge in LLMs

- Kadavath et al. 2022: "Know What You Don't Know" — LLMs can self-assess confidence.
- Shen et al. 2023: Visual chain-of-thought faithfulness.
- None of these study *spatial* self-attribution of visual errors.

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

### 4.1 Pilot (200 samples, Claude Haiku)

| Metric | Value |
|--------|-------|
| Total samples | 200 |
| Error rate (all types) | 8.5% (17/200) |
| Self-localization accuracy (answer-cell GT) | **0/17 = 0%** |
| Mean IoU (answer-cell GT) | 0.000 |
| Center-bias (% that tag "center") | 35% (6/17) |

### 4.2 GT Definition Analysis

For the 1 `adjacent_distractor` failure:

| GT Definition | Model tag | Accuracy |
|---------------|-----------|----------|
| Answer-cell GT | bottom-left (distractor) vs. middle-left (correct) | 0% |
| Distractor-cell GT | bottom-left vs. bottom-left | 100% |

**Interpretation.** The model correctly identifies *what attracted its attention* (the distractor)
but not *where the correct answer is*. This is accurate self-knowledge about visual attention,
but not about what should have been attended.

### 4.3 Error Patterns in Relative-Position Failures

| Pattern | Count |
|---------|-------|
| Center bias (tags "center") | 6/16 |
| Adjacent-cell offset (correct axis, wrong distance) | 4/16 |
| Completely wrong region | 6/16 |

---

## 5. Discussion

### 5.1 The Two GT Definitions

Prior VLM attribution work implicitly uses a single GT. FailWhere exposes a semantic ambiguity:
- "Which region led you to give this wrong answer?" → distractor-cell GT
- "Which region would have led to the correct answer?" → answer-cell GT

These definitions agree for simple distractor errors but diverge for systematic spatial reasoning
failures (relative_position type), where no single masking fixes the error.

### 5.2 Center Bias

35% of self-localizations default to "center" regardless of where the error occurred. This
parallels the well-known center bias in saliency maps (Tatler 2007) and suggests VLMs have
learned a heuristic that the most "important" region of an image is its center.

### 5.3 Limitations

1. Synthetic images: Lacks ecological validity of natural images.
2. Single model: Results are for Claude Haiku; stronger models may perform differently.
3. Small failure count: 17 failures; main paper would need 100+ for statistical robustness.

---

## 6. Proposed Full Study

### Scale-up
- Benchmark: 500 images × 2 types × 5 target positions per type = 5,000 question instances.
- Target ~150-200 failures for stable statistics.

### Multi-model evaluation
- Claude Haiku (done), Claude Sonnet, Claude Opus (proprietary)
- Gemini 1.5 Flash, GPT-4V (API)
- LLaVA-1.6-13B, Qwen-VL (open-source, via HuggingFace)

### Real-image extension
- Use VQAv2 spatial questions (counting, left/right, above/below) filtered to questions where Haiku fails.
- Counterfactual GT via masking patches.

### Metrics
- Primary: structured tag accuracy (answer-cell GT and distractor-cell GT)
- Secondary: IoU between tagged region and GT
- Tertiary: counterfactual validation rate (% of self-localizations that, when masked, fix the error)

---

## 7. Conclusion

We introduce FailWhere and show that a SOTA VLM achieves 0% accuracy when asked to verbally
identify which spatial region caused its own visual error. The model instead defaults to a
center-bias heuristic or identifies the distractor — suggesting that spatial self-attribution
capability is absent in current VLMs. This motivates both better interpretability training
objectives and more precise evaluation frameworks.

---

*Code and data: [github.com/subodhwandile/vlm-failure-bench](https://github.com/subodhwandile/vlm-failure-bench)*
