"""
Counterfactual masking validation for a subset of VSR failures.

For each failure, masks each of 9 cells with gray, re-runs the T/F question,
and identifies the causal cell (the one whose masking changes the model's answer).
Reports both neutral_tag and stateless_tag vs. CF-GT match rates.

Usage:
  python3 run_cf_validation.py \\
    --results experiments/real_vsr_v3_t0.json \\
    --output  experiments/real_vsr_v3_t0_cf100.json \\
    --n_failures 100 --temperature 0.0
"""

import argparse
import base64
import io
import json
import os
import random
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

BEDROCK_PROXY_BASE = os.environ.get("BEDROCK_PROXY_URL", "")
DEFAULT_MODEL     = "global.anthropic.claude-haiku-4-5-20251001-v1:0"  # Bedrock path
DEFAULT_MODEL_DIRECT = "claude-haiku-4-5-20251022"                     # Public Anthropic API path
TEMPERATURE      = 0.0

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


class BedrockProxyClient:
    """Bearer-JWT Bedrock proxy client. Set BEDROCK_PROXY_URL and BEDROCK_JWT_HELPER env vars."""
    def __init__(self, base_url: str = BEDROCK_PROXY_BASE):
        self.base_url = base_url.rstrip("/")
        self.messages = self

    def _get_jwt(self) -> str:
        helper = os.environ.get("BEDROCK_JWT_HELPER", "")
        if not helper:
            raise RuntimeError("BEDROCK_JWT_HELPER env var not set — or set ANTHROPIC_API_KEY instead")
        return subprocess.check_output([helper]).decode().strip()

    def create(self, model: str, max_tokens: int, messages: list, temperature: float = TEMPERATURE, **kwargs):
        jwt = self._get_jwt()
        url = f"{self.base_url}/model/{model}/invoke"
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            **kwargs,
        }).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())

        class _C:
            def __init__(self, t): self.text = t

        class _R:
            def __init__(self, d):
                self.content = [_C(c["text"]) for c in d.get("content", [])
                                if c.get("type") == "text"]

        return _R(data)


def make_client():
    """Return (client, model_id). Uses ANTHROPIC_API_KEY if set, else Bedrock proxy."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        return anthropic.Anthropic(), DEFAULT_MODEL_DIRECT
    return BedrockProxyClient(), DEFAULT_MODEL


def fetch_image(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def pil_to_png_b64(img: Image.Image) -> str:
    """Use PNG to avoid JPEG re-encoding artifacts in masked images."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def mask_region(img: Image.Image, region_name: str) -> Image.Image:
    w, h = img.size
    masked = img.copy()
    draw = ImageDraw.Draw(masked)
    for name, x0f, y0f, x1f, y1f in GRID_CELLS:
        if name == region_name:
            draw.rectangle([int(x0f * w), int(y0f * h), int(x1f * w), int(y1f * h)],
                           fill=(128, 128, 128))
    return masked


def ask_tf(client, model: str, img: Image.Image, caption: str,
           temperature: float = TEMPERATURE) -> str:
    b64 = pil_to_png_b64(img)
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model, max_tokens=16, temperature=temperature,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": "image/png", "data": b64}},
                    {"type": "text",
                     "text": f'Is the following statement true or false?\n\n'
                             f'Statement: "{caption}"\n\nAnswer with only "True" or "False".'},
                ]}],
            )
            if not resp.content:
                raise ValueError("empty content")
            return resp.content[0].text.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def normalize_bool(answer: str) -> Optional[bool]:
    a = answer.lower().strip().rstrip(".")
    if "true" in a and "false" not in a:
        return True
    if "false" in a and "true" not in a:
        return False
    return None


def find_cf_gt(client, model: str, img: Image.Image, caption: str,
               wrong_answer: str, temperature: float = TEMPERATURE):
    """Mask each of 9 cells; return (cf_gt_str, cell_answers_dict).

    cell_answers_dict maps cell_name -> normalized bool answer (True/False/None).
    cf_gt_str is the single causal cell name, "multiple", or "none_found".
    Returning the full dict enables fix-rate vector analysis without a rescan.
    """
    wrong_bool = normalize_bool(wrong_answer)
    cell_answers = {}
    causal = None
    multiple = False
    for name, *_ in GRID_CELLS:
        masked = mask_region(img, name)
        try:
            ans = ask_tf(client, model, masked, caption, temperature=temperature)
            cell_bool = normalize_bool(ans)
        except Exception as e:
            print(f"    CF error on {name}: {e}")
            cell_bool = None
        cell_answers[name] = cell_bool
        changed = (cell_bool is not None) and (cell_bool != wrong_bool)
        if changed:
            if causal is None:
                causal = name
            else:
                multiple = True
    if multiple:
        cf_gt = "multiple"
    else:
        cf_gt = causal or "none_found"
    return cf_gt, cell_answers


def main():
    os.chdir(Path(__file__).parent)
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",     default="experiments/real_vsr_v3_t0.json")
    parser.add_argument("--output",      default=None)
    parser.add_argument("--n_failures",  type=int,   default=100)
    parser.add_argument("--model",       default=None,
                        help="Model ID override. Defaults to ANTHROPIC_API_KEY model or Bedrock proxy model.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    output       = args.output or args.results.replace(".json", f"_cf{args.n_failures}.json")
    partial_file = output + ".partial"

    with open(args.results) as f:
        results = json.load(f)

    failures = [r for r in results if r["is_wrong"] and r.get("tagged_region")]
    print(f"Total failures with REGION: tag: {len(failures)}")

    # Stratified sample: n/2 center-tagged, n/2 non-center-tagged
    central     = [r for r in failures if r["tagged_region"] == "center"]
    non_central = [r for r in failures if r["tagged_region"] != "center"]
    half        = args.n_failures // 2
    subset = random.sample(central, min(half, len(central))) + \
             random.sample(non_central, min(half, len(non_central)))
    if len(subset) < args.n_failures:
        remaining = [r for r in failures if r not in subset]
        subset += random.sample(remaining, min(args.n_failures - len(subset), len(remaining)))
    print(f"Subset: {len(subset)} failures  "
          f"({sum(1 for r in subset if r['tagged_region']=='center')} center, "
          f"{sum(1 for r in subset if r['tagged_region']!='center')} non-center)")

    # Resume
    cf_results = []
    done_ids   = set()
    if os.path.exists(partial_file):
        with open(partial_file) as f:
            cf_results = json.load(f)
        done_ids = {r["image_id"] for r in cf_results}
        print(f"Resuming — {len(done_ids)} already done")

    client, default_model = make_client()
    model = args.model or default_model
    print(f"Model: {model}  temperature: {args.temperature}\n")
    for i, r in enumerate(subset):
        if r["image_id"] in done_ids:
            continue

        print(f"[{i+1}/{len(subset)}] {r['image_id']}: {r['caption'][:65]}")
        neutral_tag   = r.get("neutral_tagged_region")
        stateless_tag = r.get("tagged_region")
        print(f"  neutral={neutral_tag!r}  stateless={stateless_tag!r}  model={r['model_answer_raw']!r}")

        try:
            img = fetch_image(r["image_url"])
        except Exception as e:
            print(f"  SKIP — fetch: {e}")
            continue

        cf_gt, cell_answers = find_cf_gt(client, model, img, r["caption"], r["model_answer_raw"],
                            temperature=args.temperature)
        neutral_match   = (neutral_tag   == cf_gt) if cf_gt not in ("none_found", "multiple") else None
        stateless_match = (stateless_tag == cf_gt) if cf_gt not in ("none_found", "multiple") else None
        print(f"  CF-GT: {cf_gt!r}  neutral_match={neutral_match}  stateless_match={stateless_match}")

        cf_results.append({
            **r,
            "cf_gt":            cf_gt,
            "cell_answers":     cell_answers,
            "neutral_match":    neutral_match,
            "stateless_match":  stateless_match,
        })

        with open(partial_file, "w") as f:
            json.dump(cf_results, f, indent=2)

    # Final summary
    clean = [x for x in cf_results if x["cf_gt"] not in ("none_found", "multiple", None)]
    total = len(cf_results)
    print(f"\n{'='*60}")
    print(f"CF-GT outcomes ({total} failures processed):")
    from collections import Counter
    outcomes = Counter(x["cf_gt"] for x in cf_results)
    for k, v in sorted(outcomes.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print(f"\nClean single-region CF-GT: {len(clean)}/{total}")

    if clean:
        neutral_hits   = sum(1 for x in clean if x["neutral_match"])
        stateless_hits = sum(1 for x in clean if x["stateless_match"])
        center_cf      = sum(1 for x in clean if x["cf_gt"] == "center")
        print(f"  Neutral   tag == CF-GT: {neutral_hits}/{len(clean)} = {100*neutral_hits/len(clean):.1f}%")
        print(f"  Stateless tag == CF-GT: {stateless_hits}/{len(clean)} = {100*stateless_hits/len(clean):.1f}%")
        print(f"  CF-GT is center: {center_cf}/{len(clean)} = {100*center_cf/len(clean):.1f}%")
        noncenter = [x for x in clean if x["cf_gt"] != "center"]
        if noncenter:
            nh = sum(1 for x in noncenter if x["neutral_match"])
            sh = sum(1 for x in noncenter if x["stateless_match"])
            print(f"  Non-center CF-GT neutral match: {nh}/{len(noncenter)}")
            print(f"  Non-center CF-GT stateless match: {sh}/{len(noncenter)}")

    with open(output, "w") as f:
        json.dump(cf_results, f, indent=2)
    print(f"\nSaved to {output}")
    if os.path.exists(partial_file):
        os.remove(partial_file)


if __name__ == "__main__":
    main()
