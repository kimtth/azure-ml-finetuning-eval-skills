"""
Evaluate a fine-tuned support model on the held-out test set with exact-match accuracy.
- Expects test.jsonl with `messages`, `query`, and `ground_truth`.
- Expects a local model directory produced by the Azure ML job (download or mount job output).
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate_response(model, tokenizer, messages: List[Dict], max_new_tokens: int = 96) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    input_len = inputs["input_ids"].shape[1]
    completion_ids = output[0][input_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()


def compute_accuracy(model, tokenizer, data: List[Dict]) -> float:
    correct = 0
    for row in data:
        messages = list(row["messages"])
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        pred = generate_response(model, tokenizer, messages)
        gt = row.get("ground_truth", "").strip()
        if pred.lower().strip() == gt.lower().strip():
            correct += 1
    return correct / len(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(Path(__file__).parent / "finetuned-model"))
    parser.add_argument("--test-path", default=str(Path(__file__).parent / "test.jsonl"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    test_data = load_jsonl(Path(args.test_path))

    accuracy = compute_accuracy(model, tokenizer, test_data)
    print(f"Accuracy: {accuracy:.2%} ({int(accuracy * len(test_data))}/{len(test_data)})")


if __name__ == "__main__":
    main()
