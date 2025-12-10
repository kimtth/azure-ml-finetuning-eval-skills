# E2E Customer Support Model Workflow

Complete workflow demonstrating all three skills: dataset generation, model training, and evaluation.

**Scenario**: Build a customer support chatbot for Azure Machine Learning

## Prerequisites

**Azure Resources**: AI Foundry project, Azure OpenAI (GPT-5-mini), Azure ML workspace with GPU cluster

**Environment Variables**:
```bash
export AZURE_OPENAI_ENDPOINT="https://YOUR_RESOURCE.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="gpt-5-mini"
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group"
export AZUREML_WORKSPACE_NAME="your-workspace-name"
export AZUREML_PROJECT_NAME="your-project-name"
```

**Dependencies**: `pip install azure-ai-evaluation azure-ai-ml azure-identity openai wikipedia`

## Quick Start

**Run complete workflow**:
```bash
cd e2e_test
python run_e2e_workflow.py --compute gpu-cluster
```

**Or run steps individually**:
```bash
# Step 1: Generate dataset (80 Q&A pairs, 8 personas)
python generate_customer_support_data.py
# Output: customer_support_train.jsonl (64), customer_support_test.jsonl (16)

# Step 2: Submit training job (Phi-3.5-mini, 3 epochs)
python submit_customer_support_training.py --compute gpu-cluster
# Training time: ~15 minutes on GPU

# Step 3: Evaluate model (5 metrics: relevance, coherence, fluency, groundedness, F1)
python evaluate_customer_support_model.py
# Output: evaluation_results.json, evaluation_summary.json
```

## Expected Results

**Good Performance**: Relevance/Coherence/Fluency > 4.0, Groundedness > 4.0, F1 Score > 0.6

## Cost & Time

- **Dataset Generation**: $0.02, 2-3 min
- **Training**: $0.50, 15-20 min (GPU)
- **Evaluation**: $0.04, 3-5 min
- **Total**: $0.56, 20-30 min

## Customization

**Change model**: `--model-name azureml://registries/azureml/models/Phi-4-mini/versions/1`  
**More data**: Edit `num_queries=160` in `generate_customer_support_data.py`  
**Hyperparameters**: `--batch-size 2 --epochs 5 --learning-rate 1e-5`

## Troubleshooting

- **Missing env vars** → Set all required environment variables above
- **Compute not found** → Create GPU cluster in Azure ML first
- **Out of memory** → Use `--batch-size 1` or larger GPU
- **Auth failed** → Run `az login`
