# Azure ML Fine-Tuning & Evaluation Skills

This repository is intended for use with GitHub Copilot in Visual Studio Code. You can specify the `skills` directories and files named `AGENTS.md` to provide context for Copilot-managed coding models. `experimental`

## Skills Overview

This repository provides three complementary skills for end-to-end LLM development:

### 1. **azure-ml-llm-trainer** — Fine-tune LLMs on Azure ML
Direct training of LLMs using TRL (Transformer Reinforcement Learning) trainers on Azure ML managed compute—an alternative to serverless APIs with full control over hyperparameters and training loops.

**Methods**: SFT, DPO, RL/PPO  
**Use Cases**: Custom fine-tuning, RLHF workflows, experimental training

### 2. **azure-ml-dataset-creator** — Generate Synthetic Datasets
Generate synthetic and simulated datasets using Azure AI Foundry simulators for evaluation and fine-tuning when production data is unavailable.

**Capabilities**: Q&A generation, conversation simulation, adversarial safety data  
**Use Cases**: Training data creation, RAG evaluation, safety red-teaming

### 3. **azure-ml-model-evaluation** — Evaluate AI Models
Evaluate generative AI applications locally or in the cloud with built-in quality and safety metrics using Azure AI Evaluation SDK.

**Metrics**: Relevance, groundedness, coherence, safety, NLP scores  
**Use Cases**: Model comparison, CI/CD quality gates, production monitoring

## Quick Start

1. Setup the environment 

```bash
# Install required packages
pip install azure-ai-evaluation azure-ai-projects azure-identity
pip install azure-ai-ml promptflow-azure

# Set environment variables
export AZURE_SUBSCRIPTION_ID="<your-subscription-id>"
export AZURE_RESOURCE_GROUP="<your-resource-group>"
export AZUREML_WORKSPACE_NAME="<your-workspace>"
export AZURE_OPENAI_ENDPOINT="<your-endpoint>"
export AZURE_OPENAI_DEPLOYMENT="gpt-5-mini"
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"

# Login to Azure
az login
```

2. Specify the skills folder and AGENTS.md in GitHub Copilot Chat in VS code to provide context for Copilot-managed coding models.

3. Send your request to fine-tune a model using Azure Machine Learning.

## Key Features

### Training (azure-ml-llm-trainer)
- **Direct Trainer Control**: Use TRL SFTTrainer, DPOTrainer, PPOTrainer with full visibility
- **No Serverless Lock-in**: Run on your own Azure ML compute; full control over resources
- **Azure-Native**: Data and models stay in Azure ML datastores; no external downloads
- **Production-Ready**: Checkpointing, gradient accumulation, distributed training support

### Dataset Creation (azure-ml-dataset-creator)
- **Non-Adversarial Simulation**: Generate Q&A and conversations from text/documents
- **Adversarial Simulation**: Create safety evaluation datasets with jailbreak attacks
- **Multi-Turn Support**: Simulate realistic multi-turn conversations
- **Multi-Language**: Support for 8+ languages (EN, ES, FR, DE, JA, ZH, IT, PT)

### Evaluation (azure-ml-model-evaluation)
- **Quality Metrics**: Relevance, groundedness, coherence, fluency, retrieval
- **Safety Metrics**: Violence, sexual content, hate speech, self-harm detection
- **NLP Metrics**: F1, BLEU, ROUGE, METEOR, similarity scores
- **Cloud & Local**: Run evaluations locally or at scale in Azure  

## Output Comparison: With Skills vs. Without Skills

- `e2e_test_w_skill`: Generated e2e workflow **with skills** and AGENTS.md
- `e2e_test_wo_skill`: Generated e2e workflow **without skills** and AGENTS.md
- Generate customer support Q&A dataset (80 samples)
- Fine-tune Phi-4-mini on Azure ML GPU compute
- Evaluate with quality metrics (relevance, coherence, F1)
- Inquiry: `Can you generate a synthetic dataset of customer-support conversations, fine-tune a Phi-4-mini model (or any small model that supports fine-tuning) to answer support questions, and then evaluate the model’s accuracy on a test set? Create it under <your_directory_path>`

**Run it:**
```bash
cd e2e_test_w_skill
python run_e2e_workflow.py --compute gpu-cluster
```

## References

### Official Documentation
- [Azure AI Foundry Documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/)
- [Azure AI Evaluation SDK](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/evaluate-sdk)
- [Simulator Documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/simulator-interaction-data)
- [Azure ML Python SDK](https://learn.microsoft.com/en-us/python/api/overview/azure/ml/)

### Fine-Tuning Methods
- [Serverless Fine-Tuning (Foundry Models)](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/fine-tune-serverless)
- [OpenAI Fine-Tuning (SFT, DPO, RL)](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning)
- [Direct Preference Optimization (DPO)](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning-direct-preference-optimization)

### Related Resources
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl/)
- [Azure AI Model Catalog](https://ai.azure.com/explore/models)
- [Azure Machine Learning Examples](https://github.com/Azure/azureml-examples)
- [Hugging Face Skills](https://github.com/huggingface/skills) (Inspiration)

## Notes

- Use `az login` before submitting jobs
- GPU compute recommended for training (e.g., `Standard_NC24ads_A100_v4`)
- Artifacts stored in job output folder; register as model for deployment
- For detailed guidance, refer to `AGENTS.md` and skill documentation under `skills/`
