"""
Rescan batch 1 CF results to add per-cell fix-rate vectors.

Batch 1 was run with the old find_cf_gt that discarded per-cell answers.
This script re-queries the 9 cells for items that are missing "cell_answers",
then updates the JSON in-place.

Usage:
  python3 run_cf_rescan.py
  python3 run_cf_rescan.py --input experiments/real_vsr_v3_t0_cf100.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_cf_validation import (
    BedrockProxyClient, fetch_image, mask_region, ask_tf, normalize_bool,
    DEFAULT_MODEL, TEMPERATURE, GRID_CELLS,
)


def rescan_cell_answers(client, model, img, caption, wrong_answer,
                        temperature=TEMPERATURE):
    """Query all 9 cells and return cell_answers dict."""
    wrong_bool = normalize_bool(wrong_answer)
    cell_answers = {}
    for name, *_ in GRID_CELLS:
        masked = mask_region(img, name)
        for attempt in range(3):
            try:
                ans = ask_tf(client, model, masked, caption, temperature=temperature)
                cell_answers[name] = normalize_bool(ans)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"    error on {name}: {e}")
                    cell_answers[name] = None
    return cell_answers


def main():
    os.chdir(Path(__file__).parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default="experiments/real_vsr_v3_t0_cf100.json")
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    partial = args.input + ".rescan_partial"

    # Load existing results
    src = partial if os.path.exists(partial) else args.input
    with open(src) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {src}")

    need_rescan = [r for r in data if not r.get("cell_answers")]
    print(f"Items needing rescan: {len(need_rescan)}")
    if not need_rescan:
        print("All items already have cell_answers — nothing to do.")
        return

    client = BedrockProxyClient()
    print(f"Model: {args.model}  temperature: {args.temperature}\n")

    done = 0
    for r in data:
        if r.get("cell_answers"):
            continue

        print(f"[{done+1}/{len(need_rescan)}] {r['image_id']}: {r.get('caption','')[:65]}")
        try:
            img = fetch_image(r["image_url"])
        except Exception as e:
            print(f"  SKIP — fetch: {e}")
            continue

        cell_answers = rescan_cell_answers(
            client, args.model, img, r["caption"], r["model_answer_raw"],
            temperature=args.temperature,
        )
        r["cell_answers"] = cell_answers
        done += 1

        # Save progress every item
        with open(partial, "w") as f:
            json.dump(data, f, indent=2)

        # Verify consistency with stored cf_gt
        fixes = [c for c, v in cell_answers.items()
                 if v is not None and v != normalize_bool(r.get("model_answer_raw", ""))]
        computed_cf = (
            fixes[0] if len(fixes) == 1 else
            "multiple" if len(fixes) > 1 else
            "none_found"
        )
        if computed_cf != r.get("cf_gt"):
            print(f"  NOTE: cf_gt changed {r['cf_gt']!r} → {computed_cf!r} (temp=0 may not be perfectly stable)")

    print(f"\nRescan complete. {done} items updated.")
    # Write final
    with open(args.input, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {args.input}")
    if os.path.exists(partial):
        os.remove(partial)


if __name__ == "__main__":
    main()
