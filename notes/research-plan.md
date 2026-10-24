# Gap H — Research Plan: VLM Failure Self-Localization

## Core Research Claim

When a VLM produces an incorrect answer to a visual question, it cannot accurately
identify which specific sub-region of the input image was the causal source of that error.

## Terminology

- **Self-localization**: The model itself (not an external tool) identifies the responsible region
- **Failure attribution**: Attribution of a WRONG answer, not a correct one
- **Causal region**: The sub-region whose counterfactual removal/replacement causes the model to correct its answer
- **Contrast with**: Post-hoc external tools (Grad-CAM), modality-level attribution, faithfulness of correct answers

## Research Phases

### Phase 1 — Literature (CURRENT)
- Deep prior art search on the specific claim
- Map every related paper onto the 2x2 grid:
  [self-report vs external] × [failure vs success]
- Identify: does any paper benchmark self-report of FAILURE at sub-region granularity?

### Phase 2 — Gap Validation
- Run a small pilot experiment (50-100 examples):
  - Take VQAv2 / ChartQA wrong-answer instances from GPT-4V or Claude
  - Ask model to self-localize: "which part of this image led to your answer?"
  - Manually verify against counterfactual ground truth
- Goal: confirm the gap exists empirically before full benchmark build

### Phase 3 — Benchmark Design
- Dataset: 500-800 image-question pairs, confirmed model-wrong-answer instances
- Ground truth: automated counterfactual editing (inpainting, blur, mask)
- Tasks: region description, bounding box, segmentation mask
- Control condition: same task on correct answers

### Phase 4 — Evaluation
- Models: GPT-4V, Claude 3.5, Gemini Pro Vision, LLaVA-1.6
- Metrics: IoU, attribution accuracy, correct vs. failure localization gap

### Phase 5 — Paper Writing
- Target venue: ICCV / CVPR / EMNLP / NeurIPS D&B track

## Key Differentiators from Prior Art

| Paper | Why it's different from our claim |
|---|---|
| EDCT-Bench (2609.17953) | Tests faithfulness via edits to correct answers; not failure self-localization |
| CRAFT (2609.38810) | Modality-level (text vs image); medical domain only |
| GLIMPSE (2506.18985) | External post-hoc gradient maps; not model self-report |
| Seeing but Not Believing (2510.17771) | External attention analysis; not model self-report |
| DO-Bench (2604.22822) | Modality-level attribution; not spatial sub-region |
| Visual Reasoning Tracer (2512.05091) | Traces correct-answer reasoning; not failure attribution |

## Open Questions
1. How to define "sub-region" operationally — bounding box? CLIP-style region? patch grid?
2. How to construct counterfactual ground truth at scale without human annotation?
3. Is the failure-localization gap consistent across model families or model-specific?
4. Does the gap correlate with confidence? (model is confident in wrong answer = worse self-localization?)
