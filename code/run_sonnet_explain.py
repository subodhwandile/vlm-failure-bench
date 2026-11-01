"""
Sonnet-explain baseline: run Sonnet-5-5 on Haiku's 142 VSR failures using the
same REGION: self-localization protocol. Score against Haiku's CF-GT.

The "self" control: if Sonnet matches Haiku's attribution accuracy on Haiku's
failures, the report is generic image-reading; if Sonnet is worse, it suggests
real self-knowledge is needed. ~142 API calls, no masking.

Usage:
  python3 run_sonnet_explain.py
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_cf_validation import ToastLLMClient, fetch_image, TEMPERATURE
from real_image_experiment import (
    ask_region_neutral, ask_self_localization, parse_region_tag,
)

SONNET_MODEL        = "global.anthropic.claude-sonnet-5-5"   # Bedrock/Toast path
SONNET_MODEL_DIRECT = "claude-sonnet-5-5"                    # Public Anthropic API path
HAIKU_MODEL         = "global.anthropic.claude-haiku-4-5-20251001-v1:0"


class SonnetToastClient(ToastLLMClient):
    """ToastLLMClient that omits `temperature` — required for claude-sonnet-5-5
    on Bedrock, which rejects the parameter with HTTP 400."""
    def create(self, model, max_tokens, messages, temperature=None, **kwargs):
        import json as _json
        import urllib.request as _ur
        jwt = self._get_jwt()
        url = f"{self.base_url}/model/{model}/invoke"
        body_dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": messages,
            **kwargs,
        }
        body = _json.dumps(body_dict).encode()
        req = _ur.Request(
            url, data=body,
            headers={"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"},
            method="POST",
        )
        with _ur.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read().decode())

        class _C:
            def __init__(self, t): self.text = t
        class _R:
            def __init__(self, d):
                self.content = [_C(c["text"]) for c in d.get("content", [])
                                if c.get("type") == "text"]
        return _R(data)


def make_client():
    """Return (client, model_id). Uses ANTHROPIC_API_KEY if set, else Toast proxy.
    Public API accepts temperature; Bedrock/Toast does not for claude-sonnet-5-5."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        return anthropic.Anthropic(), SONNET_MODEL_DIRECT
    return SonnetToastClient(), SONNET_MODEL


def main():
    os.chdir(Path(__file__).parent)
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="experiments/real_vsr_v3_t0.json")
    parser.add_argument("--cf",      default="experiments/real_vsr_v3_t0_cf_combined.json")
    parser.add_argument("--output",  default="experiments/real_vsr_v3_t0_sonnet_explain.json")
    parser.add_argument("--model",   default=None,
                        help="Model ID override. Defaults to ANTHROPIC_API_KEY model or Toast model.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    partial = args.output + ".partial"

    with open(args.results) as f:
        results = json.load(f)
    with open(args.cf) as f:
        cf_data = json.load(f)

    # Only failures (Haiku was wrong)
    failures = [r for r in results if r.get("is_wrong") and r.get("image_url")]
    cf_map   = {r["image_id"]: r for r in cf_data}
    print(f"Haiku failures: {len(failures)}")
    print(f"CF-GT available: {sum(1 for r in failures if cf_map.get(r['image_id']))}")

    # Resume
    done_results, done_ids = [], set()
    if os.path.exists(partial):
        with open(partial) as f:
            done_results = json.load(f)
        done_ids = {r["image_id"] for r in done_results}
        print(f"Resuming — {len(done_ids)} already done")

    client, default_model = make_client()
    model = args.model or default_model
    print(f"Model: {model}  temperature: {args.temperature}\n")

    for i, r in enumerate(failures):
        if r["image_id"] in done_ids:
            continue

        print(f"[{i+1}/{len(failures)}] {r['image_id']}: {r['caption'][:65]}")
        cf = cf_map.get(r["image_id"], {})
        cf_gt = cf.get("cf_gt")

        try:
            from PIL import Image as PILImage
            img = fetch_image(r["image_url"])
        except Exception as e:
            print(f"  SKIP — fetch: {e}")
            continue

        # Neutral (Cond A): Sonnet reads image + Haiku's answer, names region
        try:
            neutral_resp = ask_region_neutral(
                client, model, img, r["caption"],
                r["model_answer_raw"], temperature=args.temperature,
            )
            neutral_tag = parse_region_tag(neutral_resp)
        except Exception as e:
            print(f"  neutral error: {e}")
            neutral_resp, neutral_tag = None, None

        # Stateless (Cond B): Sonnet given image + caption + Haiku's wrong answer
        try:
            stateless_resp = ask_self_localization(
                client, model, img, r["caption"],
                r["model_answer_raw"], temperature=args.temperature,
            )
            stateless_tag = parse_region_tag(stateless_resp)
        except Exception as e:
            print(f"  stateless error: {e}")
            stateless_resp, stateless_tag = None, None

        neutral_match   = (neutral_tag   == cf_gt) if (neutral_tag   and cf_gt not in ("none_found","multiple",None)) else None
        stateless_match = (stateless_tag == cf_gt) if (stateless_tag and cf_gt not in ("none_found","multiple",None)) else None

        print(f"  neutral={neutral_tag!r}  stateless={stateless_tag!r}  cf_gt={cf_gt!r}")
        print(f"  neutral_match={neutral_match}  stateless_match={stateless_match}")

        done_results.append({
            "image_id":              r["image_id"],
            "caption":               r["caption"],
            "haiku_answer_raw":      r["model_answer_raw"],
            "cf_gt":                 cf_gt,
            "sonnet_neutral_tag":    neutral_tag,
            "sonnet_stateless_tag":  stateless_tag,
            "neutral_match":         neutral_match,
            "stateless_match":       stateless_match,
            # also record Haiku's own tags for direct comparison
            "haiku_neutral_tag":     r.get("neutral_tagged_region"),
            "haiku_stateless_tag":   r.get("tagged_region"),
        })

        with open(partial, "w") as f:
            json.dump(done_results, f, indent=2)

    # Summary on clean subset
    from collections import Counter
    clean = [r for r in done_results if r.get("cf_gt") not in ("none_found","multiple",None)]
    print(f"\n{'='*60}")
    print(f"Clean CF-GT subset: {len(clean)}/{len(done_results)}")
    for tag_key, lbl in [("sonnet_neutral_tag","Sonnet neutral"), ("sonnet_stateless_tag","Sonnet stateless"),
                          ("haiku_neutral_tag","Haiku neutral"),  ("haiku_stateless_tag","Haiku stateless")]:
        hits = sum(1 for r in clean if r.get(tag_key) == r["cf_gt"])
        print(f"  {lbl}: {hits}/{len(clean)} = {100*hits/len(clean):.1f}%" if clean else f"  {lbl}: —")

    with open(args.output, "w") as f:
        json.dump(done_results, f, indent=2)
    print(f"Saved to {args.output}")
    if os.path.exists(partial):
        os.remove(partial)


if __name__ == "__main__":
    main()
