"""
Evaluate the fine-tuned model on the test set for accuracy (exact match).
Assumes test.jsonl and model_out/ are present in this directory.
"""
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def get_answer(model, tokenizer, prompt, max_new_tokens=64):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the support answer
    if "Support:" in decoded:
        return decoded.split("Support:", 1)[-1].strip()
    return decoded.strip()

def main():
    test_data = load_jsonl("e2e_test_wo_skill/test.jsonl")
    tokenizer = AutoTokenizer.from_pretrained("e2e_test_wo_skill/model_out")
    model = AutoModelForCausalLM.from_pretrained("e2e_test_wo_skill/model_out")
    correct = 0
    for ex in tqdm(test_data):
        prompt = ex["instruction"]
        pred = get_answer(model, tokenizer, prompt)
        gt = ex["response"].strip()
        if pred.lower() == gt.lower():
            correct += 1
        else:
            print(f"Q: {prompt}\nGT: {gt}\nPred: {pred}\n---")
    acc = correct / len(test_data)
    print(f"Accuracy: {acc:.2%} ({correct}/{len(test_data)})")

if __name__ == "__main__":
    main()
