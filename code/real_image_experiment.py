"""
Real-image Gap H experiment using VSR (Visual Spatial Reasoning) dataset.

VSR has COCO-sourced images with spatial-relation captions.
We convert each caption to a True/False question, run the VLM, and for
wrong answers ask for structured REGION: self-localization.

Three self-localization conditions (all collected per run):
  - neutral: "Which region influenced your answer?" — no correctness leak, collected on ALL samples
  - stateless: failure-specific prompt (knows answer was wrong) — only on failures
  - multiturn: original T/F in context, then region question — only on failures (--multiturn flag)

Other-model baseline (--selfoc_model): use a different model for self-loc calls
while the T/F model stays fixed. Enables Sonnet-explains-Haiku comparison.

Usage:
  python3 real_image_experiment.py --n_samples 500 --output experiments/real_vsr_v3_t0.json
  python3 real_image_experiment.py --n_samples 500 --output experiments/real_vsr_v3_t0_mt.json --multiturn
  python3 real_image_experiment.py --n_samples 500 --output experiments/real_vsr_v3_sonnet_selfoc.json \\
      --selfoc_model global.anthropic.claude-sonnet-5-5
"""

import argparse
import base64
import io
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from PIL import Image, ImageDraw
from datasets import load_dataset

TOAST_PROXY_BASE  = "https://llm-proxy.build.eng.toasttab.com/bedrock"
DEFAULT_MODEL     = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_MODEL_DIRECT = "claude-haiku-5-5"
TEMPERATURE       = 0.0   # always deterministic; logged in every result

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

VALID_REGIONS = {r[0] for r in GRID_CELLS}


class ToastLLMClient:
    def __init__(self, base_url: str = TOAST_PROXY_BASE):
        self.base_url = base_url.rstrip("/")
        self.messages = self

    def _get_jwt(self) -> str:
        return subprocess.check_output(["/opt/homebrew/bin/toastApiKeyHelper"]).decode().strip()

    def create(self, model: str, max_tokens: int, messages: list, **kwargs):
        jwt = self._get_jwt()
        url = f"{self.base_url}/model/{model}/invoke"
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
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
            def __init__(self, d): self.content = [_C(c["text"]) for c in d.get("content", []) if c.get("type") == "text"]
        return _R(data)


def make_client():
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        return anthropic.Anthropic(), DEFAULT_MODEL_DIRECT
    return ToastLLMClient(), DEFAULT_MODEL


def fetch_image(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def mask_region(img: Image.Image, region_name: str) -> Image.Image:
    w, h = img.size
    masked = img.copy()
    draw = ImageDraw.Draw(masked)
    for name, x0f, y0f, x1f, y1f in GRID_CELLS:
        if name == region_name:
            draw.rectangle([int(x0f*w), int(y0f*h), int(x1f*w), int(y1f*h)], fill=(128, 128, 128))
            return masked
    return masked


def ask_vlm(client, model: str, messages: list, max_tokens: int = 256,
            temperature: float = TEMPERATURE) -> str:
    """Low-level call. `messages` is the full Anthropic messages list."""
    import time
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            if not resp.content:
                raise ValueError("empty content in response")
            return resp.content[0].text.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                raise


def _img_msg(b64: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


def ask_tf_question(client, model: str, img: Image.Image, caption: str,
                    temperature: float = TEMPERATURE) -> str:
    b64 = pil_to_b64(img)
    prompt = (
        f'Look at the image carefully. Is the following statement true or false?\n\n'
        f'Statement: "{caption}"\n\n'
        f'Answer with only "True" or "False".'
    )
    return ask_vlm(client, model, [{"role": "user", "content": [_img_msg(b64), {"type": "text", "text": prompt}]}],
                   max_tokens=16, temperature=temperature)


# ── Condition A: neutral prompt (no correctness leak) — collected on ALL samples ──
def ask_region_neutral(client, model: str, img: Image.Image, caption: str,
                       model_answer: str, temperature: float = TEMPERATURE) -> str:
    """Works for both correct and wrong answers — no hint about correctness."""
    b64 = pil_to_b64(img)
    prompt = (
        f'You just answered the following question about this image:\n'
        f'"{caption}"\n\n'
        f'Your answer was: "{model_answer}"\n\n'
        f'Which region of the image most influenced your answer?\n\n'
        f'State ONLY the region label on a single line using this exact format:\n'
        f'REGION: <one of: top-left, top-center, top-right, middle-left, center, middle-right, '
        f'bottom-left, bottom-center, bottom-right>\n\n'
        f'Then briefly explain why that region influenced your answer.'
    )
    return ask_vlm(client, model,
                   [{"role": "user", "content": [_img_msg(b64), {"type": "text", "text": prompt}]}],
                   max_tokens=256, temperature=temperature)


# ── Condition B: stateless failure-specific prompt — only on failures ──
def ask_self_localization(client, model: str, img: Image.Image, caption: str,
                          wrong_answer: str, temperature: float = TEMPERATURE) -> str:
    b64 = pil_to_b64(img)
    prompt = (
        f'You were asked whether this statement is true or false: "{caption}"\n'
        f'You answered: "{wrong_answer}"\n\n'
        f'Please identify the specific region of this image that most influenced your answer.\n\n'
        f'First, state ONLY the region label on a single line using this exact format:\n'
        f'REGION: <one of: top-left, top-center, top-right, middle-left, center, middle-right, '
        f'bottom-left, bottom-center, bottom-right>\n\n'
        f'Then explain:\n'
        f'1. What visual content is in that region\n'
        f'2. Why that region led you to answer "{wrong_answer}"'
    )
    return ask_vlm(client, model,
                   [{"role": "user", "content": [_img_msg(b64), {"type": "text", "text": prompt}]}],
                   max_tokens=512, temperature=temperature)


# ── Condition C: multi-turn — original T/F exchange in context, then region question ──
def ask_self_localization_multiturn(client, model: str, img: Image.Image, caption: str,
                                    model_answer: str, temperature: float = TEMPERATURE) -> str:
    """Multi-turn: inject the original T/F answer as assistant turn, then ask for region."""
    b64 = pil_to_b64(img)
    tf_prompt = (
        f'Look at the image carefully. Is the following statement true or false?\n\n'
        f'Statement: "{caption}"\n\n'
        f'Answer with only "True" or "False".'
    )
    region_prompt = (
        f'Which region of the image most influenced your answer?\n\n'
        f'First, state ONLY the region label on a single line using this exact format:\n'
        f'REGION: <one of: top-left, top-center, top-right, middle-left, center, middle-right, '
        f'bottom-left, bottom-center, bottom-right>\n\n'
        f'Then explain why that region led you to answer "{model_answer}".'
    )
    messages = [
        {"role": "user", "content": [_img_msg(b64), {"type": "text", "text": tf_prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": model_answer}]},
        {"role": "user", "content": [{"type": "text", "text": region_prompt}]},
    ]
    return ask_vlm(client, model, messages, max_tokens=512, temperature=temperature)


def parse_region_tag(text: str) -> Optional[str]:
    import re
    m = re.search(r'region:\s*([a-z-]+)', text.lower())
    if m:
        label = m.group(1).strip()
        aliases = {"middle-center": "center", "top-middle": "top-center", "bottom-middle": "bottom-center"}
        label = aliases.get(label, label)
        if label in VALID_REGIONS:
            return label
    return None


def normalize_bool(answer: str) -> Optional[bool]:
    a = answer.lower().strip().rstrip(".")
    if a in ("true", "yes"):
        return True
    if a in ("false", "no"):
        return False
    # Loose fallback only when strict match fails
    if "true" in a and "false" not in a:
        return True
    if "false" in a and "true" not in a:
        return False
    return None


def find_counterfactual_gt(client, model: str, img: Image.Image, caption: str,
                           wrong_answer: str, temperature: float = TEMPERATURE) -> str:
    correct = None
    for name, *_ in GRID_CELLS:
        masked = mask_region(img, name)
        ans = ask_tf_question(client, model, masked, caption, temperature=temperature)
        norm = normalize_bool(ans)
        if norm is not None and ans.lower().strip() != wrong_answer.lower().strip():
            if correct is None:
                correct = name
            else:
                return "multiple"
    return correct or "none_found"


@dataclass
class RealResult:
    image_id: str
    image_url: str
    caption: str
    gt_label: bool
    relation: str
    temperature: float
    tf_model: str
    selfoc_model: str
    model_answer_raw: str
    model_answer_bool: Optional[bool]
    is_wrong: bool
    # Condition A — neutral, collected on ALL samples
    neutral_response: Optional[str]
    neutral_tagged_region: Optional[str]
    # Condition B — stateless failure-specific, only on failures
    self_localization_response: Optional[str]
    tagged_region: Optional[str]
    # Condition C — multi-turn, only on failures when --multiturn
    self_loc_multiturn_response: Optional[str]
    tagged_region_multiturn: Optional[str]
    # Counterfactual GT
    counterfactual_gt: Optional[str]


def run_experiment(n_samples: int, output_file: str, tf_model: str = None,
                   selfoc_model: str = None, run_counterfactual: bool = False,
                   multiturn: bool = False, temperature: float = TEMPERATURE):
    client, default_model = make_client()
    if tf_model is None:
        tf_model = default_model
    if selfoc_model is None:
        selfoc_model = tf_model

    print(f"T/F model:   {tf_model}")
    print(f"Self-loc model: {selfoc_model}")
    print(f"Temperature: {temperature}  multiturn: {multiturn}")

    partial_file = output_file + ".partial"
    results = []
    already_seen = set()
    if os.path.exists(partial_file):
        with open(partial_file) as f:
            results = json.load(f)
        already_seen = {r["image_id"] for r in results}
        print(f"Resuming from {len(results)} already-processed samples")

    print("Loading VSR dataset...")
    ds = load_dataset("cambridgeltl/vsr_random", split="test", streaming=True)

    wrong_count = sum(1 for r in results if r["is_wrong"])
    skip_count = 0
    processed = len(results)

    for item in ds:
        if processed >= n_samples:
            break

        caption   = item["caption"]
        gt_label  = bool(item["label"])
        relation  = item.get("relation", "")
        image_url = item["image_link"]
        image_id  = item["image"].replace(".jpg", "")

        if image_id in already_seen:
            continue

        print(f"[{processed+1}/{n_samples}] {image_id}: {caption[:70]}...")

        try:
            img = fetch_image(image_url)
        except Exception as e:
            print(f"  SKIP — image fetch: {e}")
            skip_count += 1
            continue

        try:
            ans_raw = ask_tf_question(client, tf_model, img, caption, temperature=temperature)
        except Exception as e:
            print(f"  SKIP — T/F API: {e}")
            skip_count += 1
            continue

        ans_bool = normalize_bool(ans_raw)
        is_wrong = (ans_bool is not None) and (ans_bool != gt_label)

        # ── Condition A: neutral region — always collected ──
        neutral_resp = neutral_tag = None
        try:
            neutral_resp = ask_region_neutral(client, selfoc_model, img, caption, ans_raw,
                                              temperature=temperature)
            neutral_tag = parse_region_tag(neutral_resp)
        except Exception as e:
            print(f"  WARN — neutral region API: {e}")

        self_loc = tagged = None
        self_loc_mt = tagged_mt = None
        cf_gt = None

        if is_wrong:
            wrong_count += 1
            print(f"  WRONG: model={ans_raw!r}  gt={gt_label}  neutral_tag={neutral_tag!r}")

            # ── Condition B: stateless failure-specific ──
            try:
                self_loc = ask_self_localization(client, selfoc_model, img, caption, ans_raw,
                                                 temperature=temperature)
                tagged = parse_region_tag(self_loc)
                print(f"  stateless REGION: {tagged!r}")
            except Exception as e:
                print(f"  WARN — stateless self-loc: {e}")

            # ── Condition C: multi-turn ──
            if multiturn:
                try:
                    self_loc_mt = ask_self_localization_multiturn(client, selfoc_model, img,
                                                                   caption, ans_raw,
                                                                   temperature=temperature)
                    tagged_mt = parse_region_tag(self_loc_mt)
                    print(f"  multiturn REGION: {tagged_mt!r}")
                except Exception as e:
                    print(f"  WARN — multiturn self-loc: {e}")

            if run_counterfactual:
                print("  Running CF masking (9 calls)...")
                cf_gt = find_counterfactual_gt(client, tf_model, img, caption, ans_raw,
                                               temperature=temperature)
                print(f"  CF-GT: {cf_gt!r}")
        else:
            print(f"  correct: model={ans_raw!r}  gt={gt_label}  neutral_tag={neutral_tag!r}")

        results.append(asdict(RealResult(
            image_id=image_id, image_url=image_url, caption=caption,
            gt_label=gt_label, relation=relation,
            temperature=temperature, tf_model=tf_model, selfoc_model=selfoc_model,
            model_answer_raw=ans_raw, model_answer_bool=ans_bool, is_wrong=is_wrong,
            neutral_response=neutral_resp, neutral_tagged_region=neutral_tag,
            self_localization_response=self_loc, tagged_region=tagged,
            self_loc_multiturn_response=self_loc_mt, tagged_region_multiturn=tagged_mt,
            counterfactual_gt=cf_gt,
        )))
        processed += 1

        with open(partial_file, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Processed={processed}  Wrong={wrong_count}  Skipped={skip_count}")
    print(f"Error rate: {100*wrong_count/max(processed,1):.1f}%")

    failures = [r for r in results if r["is_wrong"]]
    if failures:
        tagged_ok = [r for r in failures if r["tagged_region"]]
        neutral_ok = [r for r in failures if r["neutral_tagged_region"]]
        print(f"stateless REGION: parsed: {len(tagged_ok)}/{len(failures)}")
        print(f"neutral   REGION: parsed (failures): {len(neutral_ok)}/{len(failures)}")

    correct = [r for r in results if not r["is_wrong"]]
    if correct:
        neutral_c = [r for r in correct if r["neutral_tagged_region"]]
        print(f"neutral   REGION: parsed (correct):  {len(neutral_c)}/{len(correct)}")

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    if os.path.exists(partial_file):
        os.remove(partial_file)

    return results


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples",    type=int,   default=100)
    parser.add_argument("--output",       default="experiments/real_vsr_v1.json")
    parser.add_argument("--model",        default=None, help="T/F model (default: Haiku)")
    parser.add_argument("--selfoc_model", default=None, help="Self-loc model (default: same as --model)")
    parser.add_argument("--temperature",  type=float, default=TEMPERATURE)
    parser.add_argument("--multiturn",    action="store_true", help="Also run multi-turn self-loc condition")
    parser.add_argument("--counterfactual", action="store_true",
                        help="Run CF masking per failure (9x more calls)")
    args = parser.parse_args()

    run_experiment(
        n_samples=args.n_samples,
        output_file=args.output,
        tf_model=args.model,
        selfoc_model=args.selfoc_model,
        run_counterfactual=args.counterfactual,
        multiturn=args.multiturn,
        temperature=args.temperature,
    )
