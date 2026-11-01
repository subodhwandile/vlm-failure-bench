# FailWhere: Can Vision-Language Models Self-Localize Their Own Failures?

Code and data for the FailWhere benchmark — testing whether VLMs can identify which
3×3 grid region of an image caused an incorrect answer.

**Key result:** The REGION: self-attribution tag is generic image re-reading, not
failure self-knowledge. A different model (Sonnet-5-5) attributes Haiku's failures
equally well (+25.4 pp above null, p=0.002). Center tags carry zero causal signal;
non-center tags land +23–25 pp above null when the center prior is escaped.

## Quickstart

```bash
git clone https://github.com/subodhwandile/vlm-failure-bench
cd vlm-failure-bench/code
pip install anthropic pillow datasets
export ANTHROPIC_API_KEY=your_key_here

# Reproduce synthetic accuracy (9% structured / 75% keyword)
python analyze_results.py \
    --results experiments/pilot_v5_500.json \
    --qa data/qa_samples_500.json

# Reproduce CF fix-rate analysis (n=133, +4.9 pp neutral)
python analyze_cf_fixrate.py experiments/real_vsr_v3_t0_cf_combined.json
```

All result JSONs are included in the repo — no API calls needed to reproduce the
analysis. API key is only required to re-run the experiments from scratch.

## COCO Annotations (required for COCO bbox analysis only)

`coco_bbox_analysis.py` (§4.5 per-instance bbox hit rates) requires the COCO
train2017 instance annotations (~240 MB):

```bash
# Download and unzip into code/data/coco/
mkdir -p code/data/coco
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip annotations/instances_train2017.json
mv annotations/instances_train2017.json code/data/coco/
rm -rf annotations annotations_trainval2017.zip
```

Then run:

```bash
python coco_bbox_analysis.py \
    --results experiments/real_vsr_v3_t0.json \
    --coco data/coco/instances_train2017.json
```

VSR images are fetched on-demand from COCO URLs — no bulk image download needed.

## Re-running Experiments

All scripts default to `ANTHROPIC_API_KEY` (public Anthropic API). Set the key and run:

```bash
# Main VSR experiment (500 samples, ~2 hrs)
python real_image_experiment.py --n_samples 500 \
    --output experiments/real_vsr_v3_t0.json

# CF masking (142 failures, ~900 API calls, ~3 hrs)
python run_cf_validation.py \
    --results experiments/real_vsr_v3_t0.json \
    --output experiments/real_vsr_v3_t0_cf100.json \
    --n_failures 142

# Sonnet other-model baseline (~142 API calls)
python run_sonnet_explain.py
```

## Models

| Role | Public API model ID | Bedrock model ID |
|---|---|---|
| Main (Haiku) | `claude-haiku-4-5-20251022` | `global.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Sonnet baseline | `claude-sonnet-5-5` | `global.anthropic.claude-sonnet-5-5` |

Temperature: `0.0` for all runs. Note: `temperature` is a deprecated parameter for
claude-sonnet-5-5 on the Bedrock API — `run_sonnet_explain.py` handles this automatically.

## Repo Structure

```
code/
  real_image_experiment.py   # 3-condition VSR experiment (neutral/stateless/multiturn)
  run_cf_validation.py       # CF masking pipeline
  run_sonnet_explain.py      # Sonnet other-model baseline
  coco_bbox_analysis.py      # §4.5 per-instance bbox scoring
  analyze_cf_fixrate.py      # fix-rate vector analysis
  analyze_results.py         # synthetic accuracy
  experiments/               # final result JSONs (6 files)
  data/
    qa_samples_500.json      # synthetic benchmark (500 questions + GT regions)

notes/
  paper-outline.md           # full paper outline with all results and appendix
```

## Citation

```
@article{failwhere2027,
  title   = {FailWhere: Can Vision-Language Models Self-Localize Their Own Failures?},
  author  = {Wandile, Subodh Anant},
  year    = {2027}
}
```
