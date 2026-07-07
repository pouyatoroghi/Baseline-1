import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


def clean_text(x):
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def get_answers(row):
    answers = row["answers"]

    # Hugging Face CoQA format:
    # {"input_text": [...], "answer_start": [...], "answer_end": [...]}
    if isinstance(answers, dict):
        return [clean_text(x) for x in answers.get("input_text", [])]

    # Defensive fallback for alternate materializations.
    if isinstance(answers, list):
        out = []
        for a in answers:
            if isinstance(a, dict):
                out.append(clean_text(a.get("input_text", "")))
            else:
                out.append(clean_text(a))
        return out

    raise TypeError(f"Unexpected answers format: {type(answers)}")


def build_question(history, current_question, max_history_turns):
    """
    Converts CoQA's conversational setup into a single question string
    that the repo can consume without changing run.py's eval_item['question']
    assumption.

    max_history_turns:
      -1 = use all previous turns
       0 = no dialogue history
       N = use last N previous turns
    """
    current_question = clean_text(current_question)

    if max_history_turns == 0 or not history:
        return current_question

    if max_history_turns > 0:
        history = history[-max_history_turns:]

    lines = [
        "Use the passage and the conversation history to answer the current question.",
        "Conversation history:",
    ]

    for idx, (q, a) in enumerate(history, start=1):
        lines.append(f"Q{idx}: {clean_text(q)}")
        lines.append(f"A{idx}: {clean_text(a)}")

    lines.append(f"Current question: {current_question}")
    return "\n".join(lines)


def convert_split(split, max_items, max_history_turns):
    ds = load_dataset("stanfordnlp/coqa", split=split)

    out = []
    for story_idx, row in enumerate(ds):
        story = clean_text(row["story"])
        source = clean_text(row.get("source", ""))

        questions = [clean_text(q) for q in row["questions"]]
        answers = get_answers(row)
        n_turns = min(len(questions), len(answers))

        history = []
        for turn_idx in range(n_turns):
            q = questions[turn_idx]
            a = answers[turn_idx]

            item = {
                "id": f"{split}-{story_idx}-{turn_idx}",
                "story_id": f"{split}-{story_idx}",
                "turn_id": turn_idx,
                "source": source,

                # For evaluation/debugging.
                "raw_question": q,
                "answer": a,
                "answers": [a],
                "reference_answer": a,

                # This is what run.py and the prompt builder use.
                "question": build_question(
                    history=history,
                    current_question=q,
                    max_history_turns=max_history_turns,
                ),

                # The repo expects docs as a list of dicts with text.
                "docs": [
                    {
                        "title": f"CoQA story {story_idx}",
                        "text": story,
                    }
                ],
            }

            out.append(item)
            history.append((q, a))

            if max_items > 0 and len(out) >= max_items:
                return out

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--out", default="datasets/coqa_validation.json")
    parser.add_argument(
        "--max_items",
        type=int,
        default=-1,
        help="Number of flattened QA turns to write. -1 means all turns.",
    )
    parser.add_argument(
        "--max_history_turns",
        type=int,
        default=-1,
        help="-1 = all previous turns; 0 = no history; N = last N previous turns.",
    )
    args = parser.parse_args()

    data = convert_split(
        split=args.split,
        max_items=args.max_items,
        max_history_turns=args.max_history_turns,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    print(f"Wrote {len(data)} flattened CoQA examples to {out_path}")


if __name__ == "__main__":
    main()
