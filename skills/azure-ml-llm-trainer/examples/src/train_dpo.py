import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--l2_multiplier", type=float, default=0.1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": args.train_data})["train"]
    policy_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name)

    config = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        max_prompt_length=int(args.max_length * 0.6),
        max_completion_length=int(args.max_length * 0.4),
        beta=args.beta,
        gradient_checkpointing=True,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
