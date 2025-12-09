# Azure ML LLM Fine-Tuning Skill

Direct training of LLMs on Azure ML managed compute using TRL (Transformer Reinforcement Learning) trainers—an alternative to serverless APIs with full control over hyperparameters and training loops.

This repository is intended for use with GitHub Copilot in Visual Studio Code. You can specify the `samples` and `skills` directories, as well as files named `AGENTS.md`, to provide context for Copilot-managed coding models.

## Overview

Azure AI Foundry offers **four ways to fine-tune**:
1. **Serverless API (Foundry models)**: Managed Phi, Mistral via `create_finetuning_job()`; no compute setup
2. **Managed Compute (Portal UI)**: Web UI–driven fine-tuning with auto-provisioned GPU; limited SDK
3. **OpenAI API**: Azure OpenAI endpoint with GPT-4o, GPT-4 Turbo using OpenAI SDK
4. **Direct Training (This Skill)**: Run TRL trainers on your own Azure ML compute for full transparency and control

## Quick Start

1. **Set up Azure credentials:**
   ```bash
   az login
   export AZURE_SUBSCRIPTION_ID=<your-subscription>
   export AZURE_RESOURCE_GROUP=<your-resource-group>
   export AZUREML_WORKSPACE_NAME=<your-workspace>
   ```

2. **Prepare dataset** in JSONL format and upload to Azure ML datastore.

3. **Submit training job:**
   ```bash
   python sample/submit_sft_job.py \
     --compute <gpu-cluster> \
     --data-path azureml://<datastore>/dataset.jsonl \
     --model-name azureml://registries/azureml/models/Phi-3-mini-4k-instruct/versions/1
   ```

4. **Monitor** via Azure ML Studio link in output.

## Training Methods

- **SFT** (Supervised Fine-Tuning): `sample/submit_sft_job.py` + `sample/src/train_sft.py`
  - Format: `{"messages": [{"role": "...", "content": "..."}, ...]}`
- **DPO** (Direct Preference Optimization): `sample/submit_dpo_job.py` + `sample/src/train_dpo.py`
  - Format: `{"prompt": "...", "chosen": "...", "rejected": "..."}`
- **RL** (PPO-style): `sample/submit_rl_job.py` + `sample/src/train_rl.py`
  - Format: `{"prompt": "...", "reward": 0.5}` (optional reward)

## Files

- `skills/azure-ml-llm-trainer/SKILL.md` — Skill documentation
- `sample/submit_*.py` — Job submission scripts
- `sample/src/train_*.py` — Training entry points
- `sample/environment/conda.yml` — Dependencies
- `.github/copilot-instructions.md` — Copilot guidelines

## Key Features

- **Direct Trainer Control**: Use TRL SFTTrainer, DPOTrainer, PPOTrainer with full visibility  
- **No Serverless Lock-in**: Run on your own Azure ML compute; full control over resources  
- **Azure-Native**: Data and models stay in Azure ML datastores; no external downloads  
- **Production-Ready**: Includes checkpointing, gradient accumulation, and distributed training support  
- **Experimental**: Perfect for research, custom loss functions, and advanced training workflows  

## References

**Hugging face**
- [We Got Claude to Fine-Tune an Open Source LLM](https://huggingface.co/blog/hf-skills-training): Inspired by this blog.
- [Hugging Face Skills](https://github.com/huggingface/skills)

**Azure AI Foundry Fine-Tuning:**
- [Serverless Fine-Tuning (Foundry Models)](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/fine-tune-serverless?view=foundry-classic&tabs=chat-completion&pivots=programming-language-python)
- [Managed Compute (Portal UI)](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/fine-tune-managed-compute?view=foundry-classic)
- [OpenAI Fine-Tuning (SFT, DPO, RL)](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning?toc=%2Fazure%2Fai-foundry%2Ftoc.json&view=foundry-classic&pivots=programming-language-python)
- [Direct Preference Optimization (DPO)](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning-direct-preference-optimization?toc=%2Fazure%2Fai-foundry%2Ftoc.json&view=foundry-classic)
- [Reinforcement Fine-Tuning (RFT/RL with Graders)](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reinforcement-fine-tuning?toc=%2Fazure%2Fai-foundry%2Ftoc.json&view=foundry-classic)

**Libraries & Tools:**
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Azure ML SDK](https://learn.microsoft.com/en-us/python/api/azure-ai-ml/)
- [Azure AI Foundry Model Catalog](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/model-catalog)
- [Azure Machine Learning examples](https://github.com/Azure/azureml-examples): Official community-driven Azure Machine Learning examples.

## Notes

- Use `az login` with credentials before submitting jobs.
- GPU compute recommended (e.g., `Standard_NC24ads_A100_v4`).
- Artifacts stored in job output folder; register as model for deployment.
- For questions, refer to `AGENTS.md` and skill documentation under `skills/`.
