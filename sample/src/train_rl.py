import argparse
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--response_max_length", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": args.train_data})["train"]

    config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        log_with=None,
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_name)
    ppo_trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer, dataset=dataset)

    def compute_reward(prompts: List[str], responses: List[str]) -> List[float]:
        rewards = []
        for prompt, response in zip(prompts, responses):
            # If the dataset provides an explicit reward column, use it; otherwise fallback to length-based shaping.
            rewards.append(float(prompt.get("reward", 0.0)) if isinstance(prompt, dict) else float(len(response)))
        return rewards

    for batch in ppo_trainer.dataloader:
        queries = batch["prompt"]
        query_tensors = tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).input_ids.to(model.device)
        response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=args.response_max_length)
        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        rewards = compute_reward(queries, responses)
        ppo_trainer.step(query_tensors, response_tensors, torch.tensor(rewards, device=model.device))

    ppo_trainer.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
