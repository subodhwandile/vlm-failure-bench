"""
VSR CF masking batch 2 — the 42 failures not sampled in batch 1.

Uses the same random.seed(42) logic as batch 1 to identify which
failures are new, then runs CF masking on them.
"""
import argparse, json, os, random, sys
from pathlib import Path

# Add parent dir so we can import run_cf_validation helpers
sys.path.insert(0, str(Path(__file__).parent))
from run_cf_validation import (
    BedrockProxyClient, fetch_image, find_cf_gt, DEFAULT_MODEL, TEMPERATURE
)

random.seed(42)

def main():
    os.chdir(Path(__file__).parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",     default="experiments/real_vsr_v3_t0.json")
    parser.add_argument("--batch1",      default="experiments/real_vsr_v3_t0_cf100.json")
    parser.add_argument("--output",      default="experiments/real_vsr_v3_t0_cf_batch2.json")
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    partial = args.output + ".partial"

    with open(args.results) as f:
        results = json.load(f)

    failures = [r for r in results if r["is_wrong"] and r.get("tagged_region")]

    # Reproduce batch 1 selection to identify the complement
    central     = [r for r in failures if r["tagged_region"] == "center"]
    non_central = [r for r in failures if r["tagged_region"] != "center"]
    batch1_ids  = set(
        r["image_id"] for r in
        random.sample(central,     min(50, len(central))) +
        random.sample(non_central, min(50, len(non_central)))
    )
    batch2 = [r for r in failures if r["image_id"] not in batch1_ids]
    print(f"Batch2 size: {len(batch2)} failures "
          f"({sum(1 for r in batch2 if r['tagged_region']=='center')} center, "
          f"{sum(1 for r in batch2 if r['tagged_region']!='center')} non-center)")

    # Resume
    cf_results, done_ids = [], set()
    if os.path.exists(partial):
        with open(partial) as f:
            cf_results = json.load(f)
        done_ids = {r["image_id"] for r in cf_results}
        print(f"Resuming — {len(done_ids)} already done")

    client = BedrockProxyClient()
    print(f"Model: {args.model}  temperature: {args.temperature}\n")

    for i, r in enumerate(batch2):
        if r["image_id"] in done_ids:
            continue

        print(f"[{i+1}/{len(batch2)}] {r['image_id']}: {r['caption'][:65]}")
        neutral_tag   = r.get("neutral_tagged_region")
        stateless_tag = r.get("tagged_region")
        print(f"  neutral={neutral_tag!r}  stateless={stateless_tag!r}  model={r['model_answer_raw']!r}")

        try:
            from PIL import Image
            img = fetch_image(r["image_url"])
        except Exception as e:
            print(f"  SKIP — fetch: {e}"); continue

        cf_gt, cell_answers = find_cf_gt(client, args.model, img, r["caption"], r["model_answer_raw"],
                            temperature=args.temperature)
        neutral_match   = (neutral_tag   == cf_gt) if cf_gt not in ("none_found", "multiple") else None
        stateless_match = (stateless_tag == cf_gt) if cf_gt not in ("none_found", "multiple") else None
        print(f"  CF-GT: {cf_gt!r}  neutral={neutral_match}  stateless={stateless_match}")

        cf_results.append({**r, "cf_gt": cf_gt, "cell_answers": cell_answers,
                           "neutral_match": neutral_match, "stateless_match": stateless_match})
        with open(partial, "w") as f:
            json.dump(cf_results, f, indent=2)

    from collections import Counter
    outcomes = Counter(r["cf_gt"] for r in cf_results)
    clean = [r for r in cf_results if r["cf_gt"] not in ("none_found", "multiple", None)]
    print(f"\nDone. Outcomes: {dict(outcomes)}")
    print(f"Clean: {len(clean)}/{len(cf_results)}")
    if clean:
        sh = sum(1 for r in clean if r["stateless_match"])
        nh = sum(1 for r in clean if r["neutral_match"])
        print(f"Stateless match: {sh}/{len(clean)}")
        print(f"Neutral   match: {nh}/{len(clean)}")

    with open(args.output, "w") as f:
        json.dump(cf_results, f, indent=2)
    print(f"Saved to {args.output}")
    if os.path.exists(partial):
        os.remove(partial)

if __name__ == "__main__":
    main()
