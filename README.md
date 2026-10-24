# VLM Failure Self-Localization — FailWhere

Research project investigating whether vision-language models (VLMs) can accurately identify which sub-region of an input image caused an incorrect answer.

## Research Gap

When a VLM produces a wrong answer, can it self-report which image region is responsible?

Existing work covers:
- External attribution tools (Grad-CAM, attention maps)
- Modality-level attribution (vision vs. language)
- Faithfulness of *correct* answer explanations

**This project:** self-report + failure cases + spatial sub-region granularity — the intersection that has no prior coverage.

## Structure

```
code/
  prepare_data.py        # download and format GQA/VQAv2 for pilot
  pilot_experiment.py    # run VLM, collect wrong answers + self-localization
  counterfactual_gt.py   # mask grid regions to find ground-truth causal region
  requirements.txt

literature/
  papers-tracker.md      # full prior art table with 2x2 coverage map

notes/
  research-plan.md       # 5-phase research plan
```

## Quickstart

```bash
pip install -r code/requirements.txt
export ANTHROPIC_API_KEY=your_key

# 1. Prepare data (download GQA val from https://cs.stanford.edu/people/dorarad/gqa/download.html)
python code/prepare_data.py

# 2. Run pilot — collect wrong answers + self-localization responses
python code/pilot_experiment.py --n_samples 50

# 3. Generate counterfactual ground truth
python code/counterfactual_gt.py --max_samples 20
```

## Prior Art Boundary

The three-way intersection that defines this paper's novelty:

|  | External Analysis | Model Self-Report |
|---|---|---|
| **Correct answers** | GLIMPSE, MedFocus, GradCAM | EDCT-Bench, Evidence Attribution in DocVQA |
| **Failure cases** | CRAFT, Seeing but Not Believing, REVELIO | **← This paper** |

Closest paper: arXiv:2605.20158 — same counterfactual methodology but external, medical domain.
