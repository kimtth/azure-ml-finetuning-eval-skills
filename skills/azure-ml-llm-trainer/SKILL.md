---
name: azure-ml-llm-trainer
description: Train or fine-tune LLMs on Azure ML managed compute with TRL trainers. Uses direct trainer loops (SFT, DPO, RL) without relying on serverless APIs or Hugging Face infrastructure.
license: See repository root
---

# Azure ML LLM Trainer

This skill provides **direct training on Azure ML managed compute** using TRL trainers—an alternative to Azure AI Foundry's serverless fine-tuning APIs.

**Four fine-tuning options in Azure AI Foundry:**
1. **Serverless API (Foundry models)** — Use `create_finetuning_job()` for Phi, Mistral; no compute setup needed
2. **OpenAI API (OpenAI models)** — Use OpenAI SDK with Azure endpoint for GPT-4o, GPT-4 Turbo
3. **Managed Compute (Portal UI)** — Web UI–driven fine-tuning with automatic compute provisioning; limited SDK
4. **Direct Training (This Skill)** — Run TRL trainers on your own Azure ML compute for full control and transparency

**Use this skill when:**
- You need full control over training loops and hyperparameters
- You want to use TRL (Transformer Reinforcement Learning) methods directly
- You prefer running on your own compute resources (no vendor lock-in)
- You want to experiment with advanced training techniques (LoRA, gradient checkpointing, etc.)

## Files
- `sample/submit_sft_job.py` — Submits SFT training job to Azure ML compute
- `sample/src/train_sft.py` — SFT trainer entry point (TRL SFTTrainer)
- `sample/submit_dpo_job.py` — Submits DPO training job to Azure ML compute
- `sample/src/train_dpo.py` — DPO trainer entry point (TRL DPOTrainer)
- `sample/submit_rl_job.py` — Submits RL/PPO training job to Azure ML compute
- `sample/src/train_rl.py` — RL trainer entry point (TRL PPOTrainer)
- `sample/environment/conda.yml` — Runtime dependencies (transformers, trl, datasets, torch)

## Quick start
1. az login then set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZUREML_WORKSPACE_NAME.
2. Upload a JSONL dataset to a workspace datastore (workspaceblobstore or your own). Dataset must follow Azure chat-completion format: `{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`.
3. Ensure a compute target exists (GPU recommended, for example gpu-cluster).
4. Submit: `python sample/submit_sft_job.py --compute <compute-name> --data-path <azureml://.../dataset.jsonl> --model-name azureml://registries/azureml/models/Phi-3-mini-4k-instruct/versions/1`.
5. Monitor in Azure ML studio; trained weights land in the job output folder.

### DPO quick start
- Dataset format (JSONL): `{"input": {"messages": [...]}, "preferred_output": [...], "non_preferred_output": [...]}`
- `input.messages` is the conversation (system/user turns).
- `preferred_output` and `non_preferred_output` are arrays of assistant completions.
- Hyperparameters: `beta` (default 0.1) controls KL penalty, `l2_multiplier` (default 0.1) for regularization.
- Submit: `python sample/submit_dpo_job.py --compute <compute-name> --data-path <azureml://.../dpo.jsonl> --model-name azureml://registries/azureml/models/Phi-3-mini-4k-instruct/versions/1 --beta 0.1 --l2_multiplier 0.1`.

### RL (PPO-style) quick start
- Dataset format: JSONL with `prompt` field and optional `reward` (float) column for explicit reward signals. If reward missing, length-based reward shaping is used as fallback.
- Submit: `python sample/submit_rl_job.py --compute <compute-name> --data-path <azureml://.../rl.jsonl> --model-name azureml://registries/azureml/models/Phi-3-mini-4k-instruct/versions/1`.

## Notes
- **Why direct training?** Serverless APIs abstract away training details; direct training gives you full control over trainer config, callbacks, checkpointing, and custom loss functions.
- **Model source:** Use fine-tuning-enabled base models from Azure AI Foundry model catalog (e.g., `azureml://registries/azureml/models/Phi-3-mini-4k-instruct/versions/1`). Avoid Hugging Face downloads.
- **Hyperparameters:**
  - **SFT**: `batch_size`, `learning_rate` (default 2e-5), `n_epochs` (default 1), `seed`
  - **DPO**: Add `beta` (KL penalty, default 0.1), `l2_multiplier` (regularization, default 0.1)
  - **RL/PPO**: `ppo_epochs`, `learning_rate`, reward shaping via custom logic
- **Data:** Must be in Azure ML datastores as JSONL; referenced via `azureml://` URIs. Keep datasets in Azure; do not rely on external sources.
- **Artifacts:** Trained models saved to job output folder; register as Azure AI Foundry model for deployment or further fine-tuning.

## When to Use This vs Other Fine-Tuning Methods

| Criterion | Direct Training (This Skill) | Serverless API | Managed Compute | OpenAI API |
|-----------|------------------------------|----------------|-----------------|-----------|
| **Control** | Full (trainer config, callbacks) | Limited | UI-based | Limited |
| **Cost Model** | Per compute hour | Per training tokens | Per training tokens | Per training tokens |
| **Setup** | Requires compute cluster | Automatic | Automatic | N/A (Azure OpenAI) |
| **Supported Methods** | SFT, DPO, RL/PPO (TRL) | SFT (mostly) | SFT | SFT, DPO, RL with graders |
| **SDK/Programmatic** | Yes (full MLClient) | Yes (Python) | Minimal (mostly UI) | Yes (OpenAI SDK) |
| **Best for** | Experimentation, research, custom loss | Production quick-start | Production (non-devs) | Production OpenAI models |
