---
name: azure-ml-model-evaluation
description: Evaluate generative AI applications and models locally or in the cloud using Azure AI Evaluation SDK. Measure quality, safety, and performance with built-in and custom evaluators.
license: See repository root
---

# Azure ML Model Evaluation

Evaluate generative AI applications using Azure AI Evaluation SDK with built-in quality and safety metrics. Local or cloud-based evaluation integrated with CI/CD pipelines.

**Three evaluation approaches:**
1. **Local Evaluation** — Run evaluations on your machine with fast iteration
2. **Cloud Evaluation** — Scale to large datasets on Azure compute
3. **Continuous Monitoring** — Post-deployment evaluation for production applications

**Use this skill when:**
- Evaluating LLM responses for quality (relevance, coherence, groundedness)
- Assessing safety risks (hate, violence, sexual, self-harm)
- Comparing models or prompts
- Tracking evaluation metrics over time
- Building CI/CD gates with quality thresholds

## Prerequisites

### Local Evaluation
- Packages: `azure-ai-evaluation`, `azure-identity`
- Azure OpenAI deployment for AI-assisted evaluators
- Test dataset in JSONL format

### Cloud Evaluation
- Azure AI Foundry hub-based project
- Azure OpenAI deployment with chat completion
- Connected storage account for datasets
- CLI: `az login`

### Safety Evaluators
- Project in East US 2, France Central, UK South, or Sweden Central

## Files

```
examples/
  ├── local_evaluation.py                    # Evaluate with built-in metrics
  ├── cloud_evaluation.py                    # Cloud-scale evaluation job
  └── utils.py                               # Helper functions
```

## Quick Start

### Local Evaluation
```bash
cd examples
python local_evaluation.py
```
Runs 5 evaluators locally (relevance, groundedness, coherence, fluency, F1 score) on `test_data.jsonl`. Output: `evaluation_results.json`

### Cloud Evaluation
```bash
cd examples
python cloud_evaluation.py
```
Submits evaluation job to Azure; monitor via [Azure AI Foundry](https://ai.azure.com) → Evaluation runs. Includes relevance, groundedness, violence, coherence.

## Data Formats

### Single-Turn Q&A (JSONL)
```json
{"query": "What is Azure ML?", "response": "Azure ML is...", "context": "...", "ground_truth": "..."}
```

### Multi-Turn Conversation (JSONL)
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### Multi-Modal (Image + Text)
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }
  ]
}
```

## Built-In Evaluators

### Quality Metrics (Require Azure OpenAI GPT-4)

| Evaluator | Inputs | Desc |
|-----------|--------|------|
| **GroundednessEvaluator** | query, response, context | Response supported by context |
| **RelevanceEvaluator** | query, response | Response addresses query |
| **CoherenceEvaluator** | query, response | Logical flow and clarity |
| **FluencyEvaluator** | query, response | Language quality |
| **RetrievalEvaluator** | query, context | Context relevance to query |
| **IntentResolutionEvaluator** | conversation | User intent resolved |
| **TaskAdherenceEvaluator** | conversation | Adherence to instructions |

### NLP Metrics (No model required)

| Evaluator | Inputs | Desc |
|-----------|--------|------|
| **F1ScoreEvaluator** | response, ground_truth | Token overlap F1 |
| **SimilarityEvaluator** | response, ground_truth | Cosine embedding similarity |
| **BleuScoreEvaluator** | response, ground_truth | Translation quality |
| **RougeScoreEvaluator** | response, ground_truth | Summarization quality |
| **MeteorScoreEvaluator** | response, ground_truth | Semantic similarity |

### Safety Metrics (Require Azure AI project)

| Evaluator | Desc | Severity |
|-----------|------|----------|
| **ViolenceEvaluator** | Violent content | Very Low / Low / Med / High |
| **SexualEvaluator** | Sexual content | Very Low / Low / Med / High |
| **SelfHarmEvaluator** | Self-harm content | Very Low / Low / Med / High |
| **HateUnfairnessEvaluator** | Hate/discrimination | Very Low / Low / Med / High |
| **IndirectAttackEvaluator** | XPIA jailbreak attempts | True / False |
| **ProtectedMaterialEvaluator** | Copyrighted content | True / False |
| **ContentSafetyEvaluator** | Composite safety eval | Combined metrics |

## Local Evaluation Example

See [examples/local_evaluation.py](examples/local_evaluation.py) for complete implementation with quality, NLP, and similarity evaluators.

## Safety Evaluation Example

See [examples/local_evaluation.py](examples/local_evaluation.py) for safety evaluator setup and execution.

## Cloud Evaluation Example

See [examples/cloud_evaluation.py](examples/cloud_evaluation.py) for Azure cloud-based evaluation with dataset upload and evaluator configuration.

## Custom Evaluators

### Code-Based Custom Evaluator
Define evaluators as Python functions with `@tool` decorator. Functions receive inputs and return dict with score/metric.

### Prompt-Based Custom Evaluator (Prompty)
Create `.prompty` YAML files with model config and evaluation prompt. Load via `Prompty.load()` and pass to evaluate.

See [examples/local_evaluation.py](examples/local_evaluation.py) for custom evaluator patterns and integration.

## Composite Evaluators

- **QAEvaluator**: Combines Groundedness + Relevance + Coherence + Fluency + Similarity + F1
- **ContentSafetyEvaluator**: Combines Violence + Sexual + SelfHarm + HateUnfairness

See [examples/local_evaluation.py](examples/local_evaluation.py) for composite evaluator usage.

## Evaluation with Target Application

Pass a callable `target` function to `evaluate()` to automatically generate responses. Function receives query and returns response dict. See [examples/local_evaluation.py](examples/local_evaluation.py) for implementation.

## Integration with Training

### Pre-Training Baseline
Evaluate baseline model to establish metric baseline for comparison.

### Post-Training Comparison
Evaluate fine-tuned model on same data and compare metrics to baseline.

### CI/CD Gate
Add quality gates by checking evaluation metrics against thresholds before deployment.

See [examples/local_evaluation.py](examples/local_evaluation.py) for evaluation and metric comparison patterns.

## Results Structure

```python
{
    "metrics": {
        "relevance": 4.5,
        "groundedness": 4.8,
        "coherence": 4.6,
        "fluency": 4.9,
        "f1_score": 0.92,
    },
    "rows": [
        {
            "inputs.query": "...",
            "inputs.response": "...",
            "outputs.relevance.relevance": 5.0,
            "outputs.groundedness.groundedness": 4.5,
        },
        ...
    ],
    "studio_url": "https://ai.azure.com/...",
}
```

View results in Azure AI Foundry: Navigate to **Evaluation → Evaluation runs** and click your run ID.

## Notes

- **Model selection**: Use GPT-4o-mini for cost-effective evaluation
- **Batch size**: Evaluate 100-500 samples for statistical significance
- **Data quality**: Validate JSONL format and column mappings
- **Token usage**: Monitor Azure OpenAI quota—evaluators consume 800-3000 tokens each
- **Regional limits**: Safety evaluators only in East US 2, France Central, UK South, Sweden Central
- **Privacy**: Evaluation data may be logged—sanitize PII before submission

## Common Issues

**Cloud Evaluation Stuck in "Running"**
→ Azure OpenAI model lacks capacity; cancel job, increase capacity, retry

**"Model not found" Error**
→ Verify deployment exists: `az cognitiveservices account deployment list`

**Safety Evaluator "Region not supported"**
→ Create project in East US 2, France Central, UK South, or Sweden Central

**"Storage account not connected"**
→ Follow [storage account setup](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/evaluations-storage-account)

## References

- [Azure AI Evaluation SDK](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/evaluate-sdk)
- [Quality Evaluators](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/quality-evaluators)
- [Safety Evaluators](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/risk-safety-evaluators)
- [Custom Evaluators](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/flow-evaluate-sdk)
