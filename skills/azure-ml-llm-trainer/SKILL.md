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

## Template Files
These are **templates** in `examples/` directory. Generate new files in your project based on these templates:
- `examples/submit_sft_job.py` — Template for submitting SFT training jobs
- `examples/src/train_sft.py` — Template for SFT trainer entry point (TRL SFTTrainer)
- `examples/submit_dpo_job.py` — Template for DPO training job submission
- `examples/src/train_dpo.py` — Template for DPO trainer entry point (TRL DPOTrainer)
- `examples/submit_rl_job.py` — Template for RL/PPO training job submission
- `examples/src/train_rl.py` — Template for RL trainer entry point (TRL PPOTrainer)
- `examples/environment/conda.yml` — Template for runtime dependencies (transformers, trl, datasets, torch)

**Do NOT reference these files directly.** Copy and adapt them for your project structure.

## Quick start
1. az login then set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZUREML_WORKSPACE_NAME.
2. **Create training files in your project:**
   - Copy `examples/submit_sft_job.py` to your project as `submit_training.py`
   - Copy `examples/src/train_sft.py` to your project as `src/train_sft.py`
   - Copy `examples/environment/conda.yml` to your project as `environment/conda.yml`
3. Upload a JSONL dataset to a workspace datastore (workspaceblobstore or your own). **SFT dataset format:** Each line must be valid JSON with a `"messages"` field containing chat-completion format: `{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`. The trainer uses this field directly.
4. Ensure a compute target exists (GPU recommended, for example gpu-cluster).
5. Submit: `python submit_training.py --compute <compute-name> --data-path <azureml://.../dataset.jsonl> --model-name azureml://registries/azureml/models/Phi-3-mini-4k-instruct/versions/1`.
6. Monitor in Azure ML studio; trained weights land in the job output folder.

### DPO quick start
- **Dataset format (JSONL):** Each line must contain `"chosen"` and `"rejected"` fields with chat-completion format messages: `{"chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`.
- Hyperparameters: `beta` (default 0.1) controls KL penalty, `l2_multiplier` (default 0.1) for regularization.
- Submit: `python sample/submit_dpo_job.py --compute <compute-name> --data-path <azureml://.../dpo.jsonl> --model-name azureml://registries/azureml/models/Phi-3-mini-4k-instruct/versions/1 --beta 0.1 --l2_multiplier 0.1`.

### RL (PPO-style) quick start
- **Dataset format (JSONL):** Each line must contain a `"prompt"` field (string) and optional `"reward"` (float) for explicit reward signals. If reward is missing, length-based reward shaping is used as fallback: `{"prompt": "user message", "reward": 0.5}`.
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
