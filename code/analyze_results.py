"""
Gap H pilot analysis: compute self-localization accuracy for failure cases.

Usage:
  python3 analyze_results.py --results experiments/pilot_v2_100.json
"""

import argparse
import json
from pathlib import Path

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

    # Split by question type
    rel_pos   = [r for r in wrong if qa.get(r["image_id"], {}).get("question_type") == "relative_position"]
    locatable = rel_pos  # only relative_position has a single non-trivial causal region

    print(f"Relative-position failures: {len(rel_pos)}")
    print()

    exact_correct = 0
    iou_scores    = []

    for r in locatable:
        qa_info    = qa.get(r["image_id"], {})
        gt_region  = qa_info.get("causal_region", "")
        self_loc   = (r.get("self_localization_response") or "").lower()
        mentioned  = detect_mentioned_region(self_loc)
        grid       = qa_info.get("grid_layout", {})

        # Exact match
        exact = gt_region in mentioned
        exact_correct += int(exact)

        # Best IoU from all mentioned regions (vs. GT)
        best_iou = max((cell_iou(gt_region, m) for m in mentioned), default=0.0)
        iou_scores.append(best_iou)

        ref_pos   = qa_info.get("ref_pos", "?")
        direction = qa_info.get("direction", "?")

        print(f"  [{r['image_id']}]  q_type=relative_position")
        print(f"    Question : {r['question']}")
        print(f"    Model    : {r['model_answer']!r}  Correct: {r['correct_answer']!r}")
        print(f"    GT causal region : {gt_region!r}  (ref: {ref_pos!r}, dir: {direction!r})")
        print(f"    Mentioned regions: {mentioned}")
        print(f"    Exact match: {exact}  Best IoU: {best_iou:.2f}")
        if grid:
            print("    Grid layout:")
            for pos, obj in sorted(grid.items(), key=lambda x: CELL_RC.get(x[0], (9,9))):
                marker = " <-- GT CAUSAL" if pos == gt_region else (" <-- REF" if pos == ref_pos else "")
                print(f"      {pos}: {obj['color']} {obj['shape']}{marker}")
        print(f"    Self-loc: {self_loc[:200]}")
        print()

    n = len(locatable)
    if n > 0:
        exact_acc = 100 * exact_correct / n
        mean_iou  = sum(iou_scores) / n
        print("=" * 60)
        print(f"SELF-LOCALIZATION ACCURACY (relative_position failures)")
        print(f"  Exact region match  : {exact_correct}/{n} = {exact_acc:.0f}%")
        print(f"  Mean IoU (grid cell): {mean_iou:.3f}")
        print(f"")
        print(f"Key finding: when the model makes a spatial error,")
        print(f"it correctly identifies the causal region only {exact_acc:.0f}% of the time")
        print(f"(vs. 100% expected for accurate self-attribution).")
    else:
        print("No locatable failures found — run with more samples.")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="experiments/pilot_v2_100.json")
    parser.add_argument("--qa", default="data/qa_samples.json")
    args = parser.parse_args()
    analyze(args.results, args.qa)
