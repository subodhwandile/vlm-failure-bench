"""
Counterfactual ground truth generator.

For each wrong-answer instance, this script:
1. Divides the image into a grid of regions (3x3, 4x4, or semantic patches)
2. For each region: masks it out and re-runs the VLM question
3. The region whose masking CORRECTS the model's answer is the causal region
4. This is the ground truth for evaluating self-localization accuracy

Usage: python counterfactual_gt.py --results pilot_results.json --image_dir ./data/images
"""

import anthropic
import base64
import json
import os
import subprocess
import urllib.request
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from dataclasses import dataclass
from typing import Optional

TOAST_PROXY_BASE = "https://llm-proxy.build.eng.toasttab.com/bedrock"
DEFAULT_MODEL    = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

_WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

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
            **kwargs
        }).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        class _Content:
            def __init__(self, text): self.text = text
        class _Resp:
            def __init__(self, d): self.content = [_Content(c["text"]) for c in d.get("content", [])]
        return _Resp(data)

def make_client():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic(), "claude-haiku-5-5"
    return ToastLLMClient(), DEFAULT_MODEL


@dataclass
class RegionMask:
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int
    label: str  # e.g. "top-left", "center", "bottom-right"


def get_grid_regions(image_width: int, image_height: int, grid_size: int = 3) -> list[RegionMask]:
    """Divide image into grid_size x grid_size regions."""
    regions = []
    cell_w = image_width // grid_size
    cell_h = image_height // grid_size

    position_labels = {
        (0, 0): "top-left", (0, 1): "top-center", (0, 2): "top-right",
        (1, 0): "middle-left", (1, 1): "center", (1, 2): "middle-right",
        (2, 0): "bottom-left", (2, 1): "bottom-center", (2, 2): "bottom-right"
    }

    for row in range(grid_size):
        for col in range(grid_size):
            x1, y1 = col * cell_w, row * cell_h
            x2 = x1 + cell_w if col < grid_size - 1 else image_width
            y2 = y1 + cell_h if row < grid_size - 1 else image_height
            label = position_labels.get((row, col), f"region-{row}-{col}")
            regions.append(RegionMask(row=row, col=col, x1=x1, y1=y1, x2=x2, y2=y2, label=label))
    return regions


def mask_region(image_path: str, region: RegionMask, output_path: str, fill_color=(128, 128, 128)):
    """Apply a gray mask over the specified region and save."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([region.x1, region.y1, region.x2, region.y2], fill=fill_color)
    img.save(output_path)


def ask_question_with_image(client, image_path: str, question: str, model: str = DEFAULT_MODEL) -> str:
    with open(image_path, "rb") as f:
        img_data = base64.standard_b64encode(f.read()).decode("utf-8")
    ext = Path(image_path).suffix.lower().lstrip(".")
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

    response = client.messages.create(
        model=model,
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}},
                {"type": "text", "text": f"Answer this question in one short phrase or word: {question}"}
            ]
        }]
    )
    return response.content[0].text.strip()


def is_answer_correct(model_answer: str, correct_answer: str) -> bool:
    def norm(s):
        s = s.lower().strip().rstrip(".")
        return _WORD_TO_NUM.get(s, s)
    m, c = norm(model_answer), norm(correct_answer)
    return m == c or m in c or c in m


def find_causal_region(
    client,
    image_path: str,
    question: str,
    correct_answer: str,
    wrong_answer: str,
    model: str = DEFAULT_MODEL,
    grid_size: int = 3,
    tmp_dir: str = "/tmp/counterfactual"
) -> Optional[str]:
    """
    Find which region, when masked, causes the model to give the correct answer.
    Returns the region label of the causal region, or None if no single region found.
    """
    os.makedirs(tmp_dir, exist_ok=True)

    img = Image.open(image_path)
    regions = get_grid_regions(img.width, img.height, grid_size)

    causal_regions = []
    for region in regions:
        masked_path = os.path.join(tmp_dir, f"masked_{region.label}.jpg")
        mask_region(image_path, region, masked_path)

        answer_with_mask = ask_question_with_image(client, masked_path, question, model)
        corrected = is_answer_correct(answer_with_mask, correct_answer)

        print(f"  Masked {region.label}: model answered '{answer_with_mask}' (correct: {corrected})")

        if corrected:
            causal_regions.append(region.label)

    if len(causal_regions) == 1:
        return causal_regions[0]
    elif len(causal_regions) > 1:
        return f"multiple: {', '.join(causal_regions)}"
    else:
        return "none_found"


def run_counterfactual_gt(results_file: str, image_dir: str, output_file: str, max_samples: int = 30):
    """Take pilot results, find causal regions for wrong-answer instances."""
    client, model = make_client()
    print(f"Using model: {model}")

    with open(results_file) as f:
        results = json.load(f)

    # Only process relative_position failures — they have specific, locatable causal regions
    wrong_instances = [
        r for r in results
        if r["is_wrong"] and r.get("self_localization_response")
    ]
    print(f"Found {len(wrong_instances)} wrong-answer instances with self-localization responses")
    print(f"Processing {min(len(wrong_instances), max_samples)} samples...")

    for i, instance in enumerate(wrong_instances[:max_samples]):
        print(f"\n[{i+1}/{min(len(wrong_instances), max_samples)}] {instance['image_id']}")
        # Images are saved as .png
        image_path = os.path.join(image_dir, f"{instance['image_id']}.png")

        if not os.path.exists(image_path):
            print(f"  Image not found: {image_path}")
            continue

        causal = find_causal_region(
            client,
            image_path,
            instance["question"],
            instance["correct_answer"],
            instance["model_answer"],
            model=model,
        )
        instance["counterfactual_gt_region"] = causal
        print(f"  Counterfactual GT region: {causal}")

        # Compare against model's self-localization
        if causal and instance.get("self_localization_response"):
            self_loc = instance["self_localization_response"].lower()
            if causal not in ("none_found",) and not causal.startswith("multiple"):
                mentioned = causal.replace("-", " ") in self_loc or causal in self_loc
                instance["self_loc_matches_counterfactual_gt"] = mentioned
                print(f"  Self-loc matches counterfactual GT: {mentioned}")

    with open(output_file, "w") as f:
        json.dump(wrong_instances[:max_samples], f, indent=2)

    # Summary
    with_gt = [r for r in wrong_instances[:max_samples]
               if r.get("counterfactual_gt_region") and not r["counterfactual_gt_region"].startswith(("none", "multiple"))]
    evaluated = [r for r in with_gt if r.get("self_loc_matches_counterfactual_gt") is not None]
    correct   = [r for r in evaluated if r["self_loc_matches_counterfactual_gt"]]

    print(f"\n=== Counterfactual GT Results ===")
    print(f"Wrong answers processed     : {min(len(wrong_instances), max_samples)}")
    print(f"Causal region found         : {len(with_gt)}")
    print(f"Self-localization evaluated : {len(evaluated)}")
    if evaluated:
        print(f"Self-loc accuracy (vs cfgt): {len(correct)}/{len(evaluated)} = {100*len(correct)/len(evaluated):.1f}%")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="./experiments/pilot_v2_100.json")
    parser.add_argument("--image_dir", default="./data/images")
    parser.add_argument("--output", default="./experiments/counterfactual_gt.json")
    parser.add_argument("--max_samples", type=int, default=8)
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)
    run_counterfactual_gt(args.results, args.image_dir, args.output, args.max_samples)
