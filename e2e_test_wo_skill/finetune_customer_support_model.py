"""
Fine-tune a mini LLM (Phi-4-mini or similar) on customer support data using Hugging Face Transformers.
Assumes train.jsonl is present in this directory.
"""
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    model_name = os.environ.get("MODEL_NAME", "microsoft/Phi-4-mini")
    train_data = load_jsonl("e2e_test_wo_skill/train.jsonl")
    # Format for SFT: prompt/response
    def to_sft(example):
        return {"text": f"{example['instruction']}\nSupport: {example['response']}"}
    train_dataset = Dataset.from_list([to_sft(ex) for ex in train_data])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
    tokenized = train_dataset.map(tokenize, batched=True)
    args = TrainingArguments(
        output_dir="e2e_test_wo_skill/model_out",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_strategy="epoch",
        logging_steps=5,
        fp16=True if hasattr(model, "half") else False,
        report_to=[],
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model("e2e_test_wo_skill/model_out")
    tokenizer.save_pretrained("e2e_test_wo_skill/model_out")

if __name__ == "__main__":
    main()
