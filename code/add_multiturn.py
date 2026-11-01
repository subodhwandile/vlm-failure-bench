"""
Adds Condition C (multi-turn self-localization) to an existing results file.

Reads a results JSON that already has neutral + stateless conditions, then
for each failure without tagged_region_multiturn, makes the multi-turn call
and updates the record in-place.

Usage:
  python3 add_multiturn.py --results experiments/real_vsr_v3_t0.json \
                            --output  experiments/real_vsr_v3_t0_mt.json
"""

import argparse
import io
import json
import os
import time
import urllib.request
from pathlib import Path

from PIL import Image

# Import helpers from the main experiment file
from real_image_experiment import (
    BedrockProxyClient, make_client,
    fetch_image, ask_self_localization_multiturn, parse_region_tag,
    TEMPERATURE,
)


def main():
    os.chdir(Path(__file__).parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="experiments/real_vsr_v3_t0.json")
    parser.add_argument("--output",  default=None)
    args = parser.parse_args()

    out_file = args.output or args.results.replace(".json", "_mt.json")
    partial   = out_file + ".partial"

    with open(args.results) as f:
        results = json.load(f)

    # Resume from partial if available
    if os.path.exists(partial):
        with open(partial) as f:
            results = json.load(f)
        print(f"Resuming from partial ({partial})")

    client, default_model = make_client()

    failures = [r for r in results if r["is_wrong"]]
    todo     = [r for r in failures if r.get("tagged_region_multiturn") is None]
    print(f"Total failures: {len(failures)}, need multi-turn: {len(todo)}")

    for i, r in enumerate(todo):
        image_id  = r["image_id"]
        image_url = r["image_url"]
        caption   = r["caption"]
        model_answer = r["model_answer_raw"]
        selfoc_model = r.get("selfoc_model") or default_model
        temp         = r.get("temperature", TEMPERATURE)

        print(f"[{i+1}/{len(todo)}] {image_id}: {caption[:60]}...")

        try:
            img = fetch_image(image_url)
        except Exception as e:
            print(f"  SKIP (fetch): {e}")
            r["tagged_region_multiturn"] = None
            r["self_loc_multiturn_response"] = None
            continue

        try:
            mt_resp = ask_self_localization_multiturn(
                client, selfoc_model, img, caption, model_answer, temperature=temp
            )
            mt_tag = parse_region_tag(mt_resp)
            r["self_loc_multiturn_response"] = mt_resp
            r["tagged_region_multiturn"]     = mt_tag
            print(f"  multiturn REGION: {mt_tag!r}")
        except Exception as e:
            print(f"  WARN: {e}")
            r["self_loc_multiturn_response"] = None
            r["tagged_region_multiturn"]     = None

        with open(partial, "w") as f:
            json.dump(results, f, indent=2)

    # Final save
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")

    if os.path.exists(partial):
        os.remove(partial)

    # Quick stats
    tagged_mt = [r for r in results if r["is_wrong"] and r.get("tagged_region_multiturn")]
    tagged_st = [r for r in results if r["is_wrong"] and r.get("tagged_region")]
    center_mt = sum(1 for r in tagged_mt if r["tagged_region_multiturn"] == "center")
    center_st = sum(1 for r in tagged_st if r["tagged_region"] == "center")
    print(f"Stateless  center: {center_st}/{len(tagged_st)} = {100*center_st/max(len(tagged_st),1):.1f}%")
    print(f"Multi-turn center: {center_mt}/{len(tagged_mt)} = {100*center_mt/max(len(tagged_mt),1):.1f}%")


if __name__ == "__main__":
    main()
