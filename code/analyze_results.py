"""
Gap H pilot analysis: compute self-localization accuracy for failure cases.

Usage:
  python3 analyze_results.py --results experiments/pilot_v3_100.json
"""

import argparse
import json
import re
from pathlib import Path

VALID_REGIONS = {
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
}

REGION_KEYWORDS = {
    "top-left":      ["top-left", "top left", "upper-left", "upper left"],
    "top-center":    ["top-center", "top center", "upper center", "upper-center"],
    "top-right":     ["top-right", "top right", "upper-right", "upper right"],
    "middle-left":   ["middle-left", "middle left", "left center", "second row, first"],
    "center":        ["center", "central"],
    "middle-right":  ["middle-right", "middle right", "right center"],
    "bottom-left":   ["bottom-left", "bottom left", "lower-left", "lower left"],
    "bottom-center": ["bottom-center", "bottom center", "lower center"],
    "bottom-right":  ["bottom-right", "bottom right", "lower-right", "lower right"],
    "all":           ["all", "entire", "whole", "throughout"],
}

# Cell IoU: two 3x3 grid positions as (row, col) tuples
CELL_RC = {
    "top-left": (0, 0), "top-center": (0, 1), "top-right": (0, 2),
    "middle-left": (1, 0), "center": (1, 1), "middle-right": (1, 2),
    "bottom-left": (2, 0), "bottom-center": (2, 1), "bottom-right": (2, 2),
}

_REGION_ALIASES = {
    "middle-center": "center",
    "top-middle": "top-center",
    "bottom-middle": "bottom-center",
}

def parse_region_tag(text: str) -> str | None:
    """Extract 'REGION: <label>' from structured self-localization response."""
    m = re.search(r'region:\s*([a-z-]+)', text.lower())
    if m:
        label = m.group(1).strip()
        label = _REGION_ALIASES.get(label, label)
        if label in VALID_REGIONS:
            return label
    return None

def cell_iou(pos_a: str, pos_b: str) -> float:
    """Treat each cell as a unit square; IoU = 1 if same, 0 otherwise (no overlap between non-adjacent cells)."""
    if pos_a == pos_b:
        return 1.0
    ra, ca = CELL_RC.get(pos_a, (-1, -1))
    rb, cb = CELL_RC.get(pos_b, (-1, -1))
    # Check if adjacent (sharing an edge)
    if abs(ra - rb) + abs(ca - cb) == 1:
        return 0.0  # adjacent but different cells: intersection = 0, union = 2
    return 0.0

def detect_mentioned_region(text: str) -> list[str]:
    """Return all grid regions mentioned in the text."""
    text = text.lower()
    mentioned = []
    for region, keywords in REGION_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            mentioned.append(region)
    return mentioned


def analyze(results_path: str, qa_path: str = "data/qa_samples.json"):
    with open(results_path) as f:
        results = json.load(f)
    with open(qa_path) as f:
        qa = {s["image_id"]: s for s in json.load(f)}

    total = len(results)
    wrong = [r for r in results if r["is_wrong"]]

    print(f"Total samples evaluated : {total}")
    print(f"Wrong answers (errors)  : {len(wrong)} / {total} ({100*len(wrong)/total:.1f}%)")
    print()

    locatable = [r for r in wrong
                 if qa.get(r["image_id"], {}).get("question_type") in ("relative_position", "adjacent_distractor")]

    print(f"Spatially-locatable failures: {len(locatable)}")
    print()

    tag_correct   = 0   # primary metric: structured REGION: tag
    kw_correct    = 0   # fallback: keyword match
    iou_scores    = []

    for r in locatable:
        qa_info    = qa.get(r["image_id"], {})
        gt_region  = qa_info.get("causal_region", "")
        q_type     = qa_info.get("question_type", "")
        self_loc   = r.get("self_localization_response") or ""
        grid       = qa_info.get("grid_layout", {})
        ref_pos    = qa_info.get("ref_pos") or qa_info.get("distractor_region", "?")
        direction  = qa_info.get("direction", "?")

        # Primary: parse structured tag
        tagged_region = parse_region_tag(self_loc)
        tag_hit = tagged_region == gt_region
        tag_correct += int(tag_hit)
        tag_iou = cell_iou(gt_region, tagged_region) if tagged_region else 0.0
        iou_scores.append(tag_iou)

        # Fallback keyword match (for older responses without REGION: tag)
        mentioned = detect_mentioned_region(self_loc.lower())
        kw_hit = gt_region in mentioned
        kw_correct += int(kw_hit)

        print(f"  [{r['image_id']}]  q_type={q_type}")
        print(f"    Question  : {r['question']}")
        print(f"    Model     : {r['model_answer']!r}  Correct: {r['correct_answer']!r}")
        print(f"    GT causal : {gt_region!r}  ref/distractor: {ref_pos!r}")
        print(f"    REGION tag: {tagged_region!r}  tag_hit={tag_hit}  tag_IoU={tag_iou:.2f}")
        print(f"    kw_hit    : {kw_hit}  (mentioned: {mentioned})")
        if grid:
            print("    Grid:")
            for pos, obj in sorted(grid.items(), key=lambda x: CELL_RC.get(x[0], (9,9))):
                marker = " <-- GT" if pos == gt_region else (" <-- REF" if pos == ref_pos else "")
                print(f"      {pos}: {obj['color']} {obj['shape']}{marker}")
        print(f"    Self-loc: {self_loc[:200]}")
        print()

    n = len(locatable)
    if n > 0:
        print("=" * 60)
        print(f"SELF-LOCALIZATION ACCURACY (spatially-locatable failures)")
        print(f"  Structured tag (REGION:) : {tag_correct}/{n} = {100*tag_correct/n:.0f}%")
        print(f"  Keyword match (fallback) : {kw_correct}/{n} = {100*kw_correct/n:.0f}%")
        print(f"  Mean IoU (tag-based)     : {sum(iou_scores)/n:.3f}")
        print()
        if tag_correct == 0 and kw_correct > 0:
            print("NOTE: structured REGION: tag not present — older prompt format.")
            print("      Using keyword match as primary metric for this batch.")
    else:
        print("No spatially-locatable failures found — run with more samples.")


def analyze_real(results_path: str):
    """Analyze real-image VSR experiment results."""
    import re
    with open(results_path) as f:
        results = json.load(f)

    total     = len(results)
    wrong     = [r for r in results if r["is_wrong"]]
    skipped   = total - sum(1 for r in results if r.get("model_answer_bool") is not None
                            or r.get("model_answer_raw"))

    print(f"Total samples       : {total}")
    print(f"Wrong answers       : {len(wrong)} / {total} ({100*len(wrong)/max(total,1):.1f}%)")
    print()

    if not wrong:
        print("No failures to analyze.")
        return

    tagged_ok = [r for r in wrong if r.get("tagged_region")]
    print(f"REGION: tag parsed  : {len(tagged_ok)}/{len(wrong)} ({100*len(tagged_ok)/len(wrong):.0f}%)")

    # Center-bias
    center_bias = sum(1 for r in wrong
                      if r.get("tagged_region") == "center")
    print(f"Center-bias         : {center_bias}/{len(wrong)} ({100*center_bias/len(wrong):.0f}%)")

    # Region distribution
    from collections import Counter
    dist = Counter(r.get("tagged_region") for r in wrong if r.get("tagged_region"))
    print(f"\nSelf-reported region distribution (n={len(tagged_ok)}):")
    for region, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {region:20s}: {cnt}")

    # Relation-type breakdown
    rel_dist = Counter(r.get("relation", "unknown") for r in wrong)
    if any(r != "unknown" for r in rel_dist):
        print(f"\nFailures by spatial relation:")
        for rel, cnt in sorted(rel_dist.items(), key=lambda x: -x[1]):
            print(f"  {rel:25s}: {cnt}")

    # Counterfactual GT stats
    with_cf = [r for r in wrong if r.get("counterfactual_gt") and
               r["counterfactual_gt"] not in ("none_found", "multiple")]
    if with_cf:
        cf_match = sum(1 for r in with_cf
                       if r.get("tagged_region") == r["counterfactual_gt"])
        print(f"\nCounterfactual GT matches: {cf_match}/{len(with_cf)} "
              f"({100*cf_match/max(len(with_cf),1):.0f}%)")

    print()
    for r in wrong:
        print(f"  [{r['image_id']}]")
        print(f"    Caption  : {r['caption']}")
        print(f"    GT label : {r['gt_label']}  Model: {r['model_answer_raw']!r}")
        print(f"    REGION   : {r.get('tagged_region')!r}  CF-GT: {r.get('counterfactual_gt')!r}")
        print(f"    Self-loc : {(r.get('self_localization_response') or '')[:120]}")
        print()


if __name__ == "__main__":
    import os
    import sys
    os.chdir(Path(__file__).parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="experiments/pilot_v3_100.json")
    parser.add_argument("--qa", default="data/qa_samples.json")
    parser.add_argument("--real", action="store_true", help="Analyze real-image VSR results")
    args = parser.parse_args()

    if args.real:
        analyze_real(args.results)
    else:
        analyze(args.results, args.qa)
