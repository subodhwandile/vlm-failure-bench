"""
CF masking on synthetic failures — adjacent_distractor type has causal_region
known by construction (the distractor cell whose masking resolves the error).

Scores both self_loc REGION: tag and CF-GT against causal_region (GT).
"""
import argparse, base64, io, json, os, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_cf_validation import ToastLLMClient, mask_region, DEFAULT_MODEL, TEMPERATURE

from PIL import Image


def pil_to_png_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


GRID_CELLS = [
    ("top-left",      0,     0,     1/3,   1/3),
    ("top-center",    1/3,   0,     2/3,   1/3),
    ("top-right",     2/3,   0,     1.0,   1/3),
    ("middle-left",   0,     1/3,   1/3,   2/3),
    ("center",        1/3,   1/3,   2/3,   2/3),
    ("middle-right",  2/3,   1/3,   1.0,   2/3),
    ("bottom-left",   0,     2/3,   1/3,   1.0),
    ("bottom-center", 1/3,   2/3,   2/3,   1.0),
    ("bottom-right",  2/3,   2/3,   1.0,   1.0),
]

import urllib.request


def ask_question(client, model, img, question, temperature=TEMPERATURE):
    b64 = pil_to_png_b64(img)
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model, max_tokens=32, temperature=temperature,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": question + "\n\nAnswer with the color only."},
                ]}],
            )
            if not resp.content:
                raise ValueError("empty content")
            return resp.content[0].text.strip().lower()
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def find_cf_gt_synthetic(client, model, img, question, wrong_answer, temperature=TEMPERATURE):
    causal = None
    for name, *_ in GRID_CELLS:
        masked = mask_region(img, name)
        try:
            ans = ask_question(client, model, masked, question, temperature=temperature)
        except Exception as e:
            print(f"    CF error on {name}: {e}")
            continue
        if ans != wrong_answer.lower():
            if causal is None:
                causal = name
            else:
                return "multiple"
    return causal or "none_found"


def main():
    os.chdir(Path(__file__).parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot",      default="experiments/pilot_v5_500.json")
    parser.add_argument("--qa",         default="data/qa_samples_500.json")
    parser.add_argument("--images_dir", default="data/images")
    parser.add_argument("--output",     default="experiments/synthetic_cf.json")
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--temperature",type=float, default=TEMPERATURE)
    args = parser.parse_args()

    partial = args.output + ".partial"

    with open(args.pilot) as f:
        pilot = json.load(f)
    with open(args.qa) as f:
        qa_list = json.load(f)

    qa_by_id = {q["image_id"]: q for q in qa_list}
    failures = [r for r in pilot if r.get("is_wrong")]
    print(f"Total synthetic failures: {len(failures)}")

    # Focus on adjacent_distractor — causal_region known by construction
    adj_failures = [r for r in failures
                    if qa_by_id.get(r["image_id"], {}).get("question_type") == "adjacent_distractor"]
    print(f"adjacent_distractor failures: {len(adj_failures)}")

    # Resume
    cf_results, done_ids = [], set()
    if os.path.exists(partial):
        with open(partial) as f:
            cf_results = json.load(f)
        done_ids = {r["image_id"] for r in cf_results}
        print(f"Resuming — {len(done_ids)} done")

    client = ToastLLMClient()
    print(f"Model: {args.model}  temperature: {args.temperature}\n")

    for i, r in enumerate(adj_failures):
        if r["image_id"] in done_ids:
            continue

        qa    = qa_by_id[r["image_id"]]
        img_path = os.path.join(args.images_dir, qa["image_file"])

        print(f"[{i+1}/{len(adj_failures)}] {r['image_id']}: {r['question'][:65]}")
        print(f"  correct={r['correct_answer']}  model={r['model_answer']}")
        print(f"  causal_region={qa['causal_region']}  distractor_region={qa.get('distractor_region')}")

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  SKIP — load: {e}"); continue

        cf_gt = find_cf_gt_synthetic(client, args.model, img,
                                      r["question"], r["model_answer"],
                                      temperature=args.temperature)
        # Compare to both GT definitions
        causal_region    = qa.get("causal_region")      # answer-cell GT
        distractor_region = qa.get("distractor_region") # distractor-cell GT

        cf_matches_causal     = (cf_gt == causal_region)    if cf_gt not in ("none_found","multiple") else None
        cf_matches_distractor = (cf_gt == distractor_region) if cf_gt not in ("none_found","multiple") else None

        # Also check self-loc tag (from pilot data — stored in self_localization_response)
        import re
        selfloc_text = r.get("self_localization_response", "") or ""
        m = re.search(r'region:\s*([a-z-]+)', selfloc_text.lower())
        selfloc_tag = m.group(1).strip() if m else None
        tag_matches_cf        = (selfloc_tag == cf_gt)            if (selfloc_tag and cf_gt not in ("none_found","multiple")) else None
        tag_matches_causal    = (selfloc_tag == causal_region)    if selfloc_tag else None
        tag_matches_distractor = (selfloc_tag == distractor_region) if selfloc_tag else None

        print(f"  CF-GT: {cf_gt!r}")
        print(f"  cf==causal={cf_matches_causal}  cf==distractor={cf_matches_distractor}")
        print(f"  tag={selfloc_tag!r}  tag==cf={tag_matches_cf}  tag==causal={tag_matches_causal}  tag==distractor={tag_matches_distractor}")

        cf_results.append({
            **r,
            "causal_region": causal_region,
            "distractor_region": distractor_region,
            "cf_gt": cf_gt,
            "cf_matches_causal": cf_matches_causal,
            "cf_matches_distractor": cf_matches_distractor,
            "selfloc_tag": selfloc_tag,
            "tag_matches_cf": tag_matches_cf,
            "tag_matches_causal": tag_matches_causal,
            "tag_matches_distractor": tag_matches_distractor,
        })
        with open(partial, "w") as f:
            json.dump(cf_results, f, indent=2)

    # Summary
    outcomes = Counter(r["cf_gt"] for r in cf_results)
    clean = [r for r in cf_results if r["cf_gt"] not in ("none_found","multiple",None)]
    print(f"\nOutcomes: {dict(outcomes)}")
    print(f"Clean CF-GT: {len(clean)}/{len(cf_results)}")
    if clean:
        cc  = sum(1 for r in clean if r["cf_matches_causal"])
        cd  = sum(1 for r in clean if r["cf_matches_distractor"])
        tc  = sum(1 for r in cf_results if r["tag_matches_causal"])
        td  = sum(1 for r in cf_results if r["tag_matches_distractor"])
        tcf = sum(1 for r in clean if r["tag_matches_cf"])
        print(f"CF-GT == causal_region (answer-cell GT):    {cc}/{len(clean)}")
        print(f"CF-GT == distractor_region (CF GT):         {cd}/{len(clean)}")
        print(f"self-loc tag == causal_region:              {tc}/{len(cf_results)}")
        print(f"self-loc tag == distractor_region:          {td}/{len(cf_results)}")
        print(f"self-loc tag == CF-GT (on clean):           {tcf}/{len(clean)}")

    with open(args.output, "w") as f:
        json.dump(cf_results, f, indent=2)
    print(f"Saved to {args.output}")
    if os.path.exists(partial):
        os.remove(partial)

if __name__ == "__main__":
    main()
