"""
Export the trained RL policy to a JavaScript lookup table.

Generates rl_policy_lookup.js in the repo root, which the browser
loads as a static asset. The lookup covers all reachable
(player_total, is_soft, dealer_upcard, can_double) combinations —
the same four dimensions that drive the strategic decision — using a
representative fresh-deck composition for the remaining 12 features.

Usage (from repo root):
    python rl/export_policy.py
    python rl/export_policy.py --model rl/models/policy.zip --out rl_policy_lookup.js
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

ACTION_NAMES = {0: "hit", 1: "stand", 2: "doubleDown"}

# Representative deck state: 48 cards remaining after 4 dealt from a fresh deck.
# Counts: [ace, 2, 3, 4, 5, 6, 7, 8, 9, 10+face]
_BASE_COUNTS = [4, 4, 4, 4, 4, 4, 4, 4, 4, 14]
_REMAINING = 48
_CARD_COUNT = 2  # placeholder; only can_double matters for strategy


def export(model_path: Path, out_path: Path) -> None:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        sys.exit("sb3-contrib not found. Install rl/requirements.txt.")

    model = MaskablePPO.load(str(model_path))

    states, keys = [], []
    for total in range(4, 22):
        for is_soft in (0, 1):
            for dealer_up in range(2, 12):
                for can_double in (0, 1):
                    keys.append((total, is_soft, dealer_up, can_double))
                    obs = np.array(
                        [total, is_soft, _CARD_COUNT, dealer_up, can_double, _REMAINING]
                        + _BASE_COUNTS,
                        dtype=np.float32,
                    )
                    states.append(obs)

    lookup = {}
    for key, obs in zip(keys, states):
        total, is_soft, dealer_up, can_double = key
        masks = np.array([True, True, bool(can_double)])
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        lookup[f"{total}-{is_soft}-{dealer_up}-{can_double}"] = ACTION_NAMES[int(action)]

    js = (
        "// Auto-generated RL policy lookup table. Do not edit manually.\n"
        "// Source: rl/models/policy.zip  (MaskablePPO, 1M training steps, seed=42)\n"
        "// Keyed by 'playerTotal-isSoft-dealerUpcard-canDouble'.\n"
        f"const RL_POLICY = {json.dumps(lookup, separators=(',', ':'))};\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(js, encoding="utf-8")
    print(f"Exported {len(lookup)} states to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export RL policy to JS lookup table.")
    parser.add_argument("--model", type=Path, default=Path("rl/models/policy.zip"))
    parser.add_argument("--out", type=Path, default=Path("rl_policy_lookup.js"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export(args.model, args.out)
