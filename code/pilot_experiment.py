"""
Gap H Pilot Experiment — VLM Failure Self-Localization

Goal: Validate the gap exists empirically.
Steps:
  1. Load a small set of image-question pairs with known correct answers
  2. Run a VLM on each → collect wrong-answer instances
  3. For wrong answers: ask the model to self-localize the responsible region
  4. Compare self-identified region against counterfactual ground truth
  5. Report: does the model accurately locate what caused its failure?

Run: python3 pilot_experiment.py --n_samples 50
     python3 pilot_experiment.py --n_samples 50 --model anthropic.claude-3-5-haiku-20241022-v1:0
"""

import anthropic
import base64
import json
import os
import subprocess
import urllib.request
import argparse
from pathlib import Path

TOAST_PROXY_BASE  = "https://llm-proxy.build.eng.toasttab.com/bedrock"
DEFAULT_MODEL     = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_MODEL_DIRECT = "claude-haiku-5-5"


class ToastLLMClient:
    """
    Thin wrapper around the Toast LLM proxy.
    Exposes client.messages.create() matching the anthropic SDK interface.
    Uses Bearer JWT auth (refreshed per-call via toastApiKeyHelper).
    """
    def __init__(self, base_url: str = TOAST_PROXY_BASE):
        self.base_url = base_url.rstrip("/")
        self.messages = self  # so client.messages.create() works

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
        # Return object matching anthropic SDK shape
        class _Content:
            def __init__(self, text): self.text = text
        class _Resp:
            def __init__(self, d): self.content = [_Content(c["text"]) for c in d.get("content", [])]
        return _Resp(data)


def make_client():
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("Using direct Anthropic API")
        return anthropic.Anthropic(), DEFAULT_MODEL_DIRECT
    print(f"Using Toast LLM proxy ({TOAST_PROXY_BASE})")
    return ToastLLMClient(), DEFAULT_MODEL
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SampleResult:
    image_id: str
    question: str
    correct_answer: str
    model_answer: str
    is_wrong: bool
    self_localization_response: Optional[str]
    # To be filled during manual/counterfactual verification
    ground_truth_region: Optional[str] = None
    localization_correct: Optional[bool] = None


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def ask_question(client: anthropic.Anthropic, image_path: str, question: str, model: str = DEFAULT_MODEL) -> str:
    img_data = encode_image(image_path)
    ext = Path(image_path).suffix.lower().lstrip(".")
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}},
                {"type": "text", "text": f"Answer this question in one short phrase or word: {question}"}
            ]
        }]
    )
    return response.content[0].text.strip()


def ask_self_localization(client: anthropic.Anthropic, image_path: str, question: str, wrong_answer: str, model: str = DEFAULT_MODEL) -> str:
    """
    Core experiment prompt: given that the model gave a wrong answer,
    ask it to identify which region of the image led to that answer.
    """
    img_data = encode_image(image_path)
    ext = Path(image_path).suffix.lower().lstrip(".")
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

    prompt = f"""You previously answered the question "{question}" with "{wrong_answer}".

Please identify the specific region of this image that most influenced your answer.
Describe the region precisely:
1. Location (e.g., top-left, center, bottom-right, specific object)
2. What visual content is in that region
3. Why that region led you to answer "{wrong_answer}"

Be specific about the spatial location."""

    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    return response.content[0].text.strip()


_WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

def _normalize(s: str) -> str:
    s = s.lower().strip().rstrip(".")
    return _WORD_TO_NUM.get(s, s)

def is_answer_correct(model_answer: str, correct_answer: str) -> bool:
    """Loose match — handles plurals, case, word/digit number forms."""
    m = _normalize(model_answer)
    c = _normalize(correct_answer)
    return m == c or m in c or c in m


def run_pilot(image_dir: str, qa_file: str, output_file: str, n_samples: int = 50, model: str = None):
    """
    image_dir: directory containing images
    qa_file: JSON file with list of {image_id, image_file, question, answer}
    output_file: where to write results
    """
    client, default_model = make_client()
    if model is None:
        model = default_model
    print(f"Using model: {model}")

    with open(qa_file) as f:
        samples = json.load(f)[:n_samples]

    results = []
    wrong_count = 0

    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {sample['image_id']}: {sample['question'][:60]}...")
        image_path = os.path.join(image_dir, sample["image_file"])

        if not os.path.exists(image_path):
            print(f"  SKIP — image not found: {image_path}")
            continue

        model_answer = ask_question(client, image_path, sample["question"], model)
        is_wrong = not is_answer_correct(model_answer, sample["answer"])

        self_loc = None
        if is_wrong:
            wrong_count += 1
            print(f"  WRONG: model='{model_answer}' correct='{sample['answer']}' — asking self-localization")
            self_loc = ask_self_localization(client, image_path, sample["question"], model_answer, model)
            print(f"  Self-localization: {self_loc[:100]}...")

        results.append(asdict(SampleResult(
            image_id=sample["image_id"],
            question=sample["question"],
            correct_answer=sample["answer"],
            model_answer=model_answer,
            is_wrong=is_wrong,
            self_localization_response=self_loc
        )))

    print(f"\nDone. Wrong answers: {wrong_count}/{len(results)} ({100*wrong_count/max(len(results),1):.1f}%)")
    print(f"Self-localization responses collected: {wrong_count}")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="./data/images")
    parser.add_argument("--qa_file", default="./data/qa_samples.json")
    parser.add_argument("--output", default="./experiments/pilot_results.json")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="claude-haiku-5-5 (cheap) or claude-opus-5-5 (best)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    run_pilot(args.image_dir, args.qa_file, args.output, args.n_samples, args.model)
