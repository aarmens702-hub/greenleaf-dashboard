"""Live-API demo of the GreenLeaf chat: 3 questions across the 3 routes.

Run (with the key set in your environment): .venv/bin/python scripts/demo_chat.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chat import router
from app.chat.answer import answer

QUESTIONS = [
    "What is GreenLeaf's overall precision benefit?",        # data only
    "What does research say about strawberry fertigation?",  # docs only
    "Is GreenLeaf's tomato yield above the BC benchmark?",   # mixed
]


def main():
    if not router.USE_LIVE_API:
        print("No LLM API key is set, so this live demo cannot run.")
        print("Set it in your shell (do NOT commit or paste it anywhere public):")
        print('  export DEEPSEEK_API_KEY="sk-..."')
        print("  .venv/bin/python scripts/demo_chat.py")
        sys.exit(0)

    for q in QUESTIONS:
        out = answer(q)
        print("\n" + "-" * 72)
        print(f"--- Q: {q} ---")
        print("routing:", json.dumps(out["trace"]["routing"], indent=2))
        print("\nANSWER:\n" + out["answer"])
        print("\nSOURCES:")
        for i, s in enumerate(out["sources"], 1):
            print(f"  {i}. {s}")
        print("\nTRACE:")
        print(json.dumps(out["trace"], indent=2, default=str))


if __name__ == "__main__":
    main()
