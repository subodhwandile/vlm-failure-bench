# Gap H — Literature Tracker

## Confirmed Prior Art (from initial broad search)

### Directly Related — External Attribution (NOT self-report)
| Paper | arXiv | Venue | What it covers | Gap remaining |
|---|---|---|---|---|
| CRAFT | 2609.38810 | 2026 | Causal failure tracing in medical VLMs; modality-level (text vs image) | Sub-region; self-report; non-medical |
| GLIMPSE | 2506.18985 | 2025 | Gradient heat maps for VLM response interpretation | External tool, not model self-report |
| Seeing but Not Believing | 2510.17771 | ICLR 2026 | Attention-answer disconnect; external attention analysis | Not model self-report; not failure-specific |
| MedFocus | 2605.20158 | 2026 | Spatial attribution for medical chest X-ray reasoning | Medical domain; external; not failure self-localization |

### Directly Related — Faithfulness (NOT failure attribution)
| Paper | arXiv | Venue | What it covers | Gap remaining |
|---|---|---|---|---|
| EDCT-Bench | 2609.17953 | 2026 | Faithfulness via counterfactual editing — correct answers | NOT failure attribution |
| Visual Reasoning Tracer | 2512.05091 | 2025 | Object-level grounded reasoning traces | Correct answers only; not failure |
| Right Answer Wrong Evidence | MICCAI 2026 | 2026 | Detects confabulation in correct answers | Correct answers; medical domain |

### Directly Related — Modality-Level Attribution (NOT sub-region spatial)
| Paper | arXiv | Venue | What it covers | Gap remaining |
|---|---|---|---|---|
| DO-Bench | 2604.22822 | 2026 | Object hallucination; PriorRobust vs PerceptionAbility | Modality-level; not spatial sub-region |
| "Does It Fail to See or Know?" | (verify) | 2026 | See vs know dichotomy | Modality-level; not spatial sub-region |

## The 2x2 Map of the Space

```
                    External Analysis | Model Self-Report
                   -------------------|------------------
Correct answers  | GLIMPSE, MedFocus  | EDCT-Bench (partial)
                 | GradCAM, VRT       | 
                 |                    |
Failure cases    | CRAFT, DO-Bench    | *** OUR PAPER ***
                 | Seeing not Believe |
```

The bottom-right cell is empty. That is the gap.

## Deep Search Results (40+ queries, confirmed October 2026)

### The Three Axes — Coverage Map

| Aspect | Covered by | What remains OPEN |
|---|---|---|
| Counterfactual editing as ground truth for causal regions | 2605.20158 (medical), 2609.17953 | Applied to general VLMs; used as self-report benchmark |
| VLM self-report of evidence regions | 2607.24651, 2605.06058 | Extended to FAILURE cases (all existing work = correct answers only) |
| Spatial sub-region attribution for VLM FAILURES | **None found** | Entirely open |
| Self-attribution accuracy with IoU metric | **None found** | Entirely open |
| Self-attribution of failure causes (not external) | **None found** | Entirely open |

### Additional Papers Found (Deep Search)

| Paper | arXiv | Covers claim? | Why not |
|---|---|---|---|
| Attention Without Grounding (chest X-ray) | 2607.18577 | No | External heatmap analysis; medical; not self-report |
| Arbitration Failure Not Perceptual Blindness | 2604.09364 | No | Modality-level (visual vs text), not spatial sub-region |
| Evidence Attribution in Visual DocVQA | 2607.24651 | No | Correct answers only; document domain |
| REVELIO: Revealing Failure Modes | 2605.12674 | No | Concept-level categories, external |
| SABRE: Stress Testing VLMs | 2608.07435 | No | Accuracy benchmark, not self-localization |
| Trace, Verify, Correct: Spatial Reasoning | 2608.04759 | No | External detection, not model self-report |
| Towards Self-Explainable DocVQA | 2605.06058 | No | Correct answer support, not failure attribution |
| DeFA: Dependency-Guided Failure Attribution | 2610.01256 | No | Agent step-level, not visual sub-region |
| VRRL: Visually Grounded Self-Reflection via RL | 2607.02490 | No | Error recovery/correction, not source localization |
| CRAFT | 2609.38810 | No | Attention-head level; medical; external |
| FLIP | 2609.30993 | No | Logit-level probing; head/layer not spatial |

### Closest Paper: 2605.20158
"Rethinking Visual Attribution for Chest X-ray Reasoning in Large VLMs"
- Shares the counterfactual-editing-as-ground-truth methodology
- BUT: external analysis of attribution methods (not model self-report)
- AND: medical domain only (chest X-rays)
- AND: evaluates attribution methods, not model introspection

### Verdict: CONFIRMED NOVEL
The three-way intersection — self-report + failure cases + sub-region spatial granularity
with counterfactual ground truth and IoU metric — is not covered by any paper found.
