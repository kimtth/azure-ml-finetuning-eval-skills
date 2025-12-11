---
name: azure-ml-dataset-creator
description: Generate synthetic and simulated datasets for evaluation and fine-tuning using Azure AI Foundry simulators. Create non-adversarial task data, adversarial safety data, and conversation datasets without manual data collection.
license: See repository root
---

# Azure ML Dataset Creator

Generate synthetic datasets using Azure AI Foundry simulators for evaluation and fine-tuning—replacing manual data collection with automated simulation.

**Two simulator types:**
1. **Simulator** — Non-adversarial task-specific conversations from text, indexes, or custom prompts
2. **AdversarialSimulator** — Safety evaluation datasets with jailbreak attacks and harmful content

**Use this skill when:**
- Building evaluation or training datasets without production data
- Testing application responses to varied user queries
- Red-teaming for safety evaluation
- Creating multi-turn conversation datasets
- Need cost-effective synthetic data generation

## Prerequisites

- Azure AI Foundry **hub-based project** (not Foundry)
- Azure OpenAI deployment (GPT-5-mini recommended for cost)
- Packages: `azure-ai-evaluation`, `azure-identity`
- For adversarial: Project in East US 2, France Central, UK South, or Sweden Central

## Template Files
These are **templates** in `examples/` directory. Copy and adapt them for your project:

```
examples/
  ├── generate_qa_from_text.py              # Template: Q&A from Wikipedia/documents
  ├── generate_conversation.py               # Template: Multi-turn conversations
  ├── generate_adversarial.py                # Template: Safety evaluation datasets
  ├── generate_jailbreak_attacks.py          # Template: UPIA/XPIA attack simulation
  ├── generate_with_custom_prompty.py        # Template: Custom simulator behavior
  ├── utils.py                               # Template: Utility functions
  └── custom_simulator_prompty/
      ├── user_override.prompty              # Template: Custom user behavior
      └── query_generator.prompty            # Template: Custom Q&A generation
```

**Do NOT reference these files directly.** Copy and adapt them for your project structure.

## Quick Start

### Generate Q&A from Text
1. Copy `examples/generate_qa_from_text.py` and `examples/utils.py` to your project
2. Run: `python generate_qa_from_text.py`
3. Outputs: `training_data.jsonl` in chat completion format
   - Extracts text from Wikipedia
   - Generates Q&A with multiple personas
   - Ready for SFT fine-tuning

### Generate Multi-Turn Conversations
1. Copy `examples/generate_conversation.py` and `examples/utils.py` to your project
2. Run: `python generate_conversation.py`
3. Outputs: `conversation_data.jsonl`
   - Predefined conversation starters
   - Multi-turn dialogue (up to 5 turns)
   - User simulator with configurable behavior

### Generate Safety Evaluation Data
1. Copy `examples/generate_adversarial.py` and `examples/utils.py` to your project
2. Run: `python generate_adversarial.py`
3. Outputs: `adversarial_qa.jsonl`, `adversarial_conversation.jsonl`, `adversarial_summarization.jsonl`
   - Tests responses to harmful/unsafe prompts
   - Covers: hate, sexual, violence, self-harm
   - Designed for safety evaluator benchmarking

### Generate Jailbreak Attacks
1. Copy `examples/generate_jailbreak_attacks.py` and `examples/utils.py` to your project
2. Run: `python generate_jailbreak_attacks.py`
3. Outputs: `direct_attack_baseline.jsonl`, `direct_attack_jailbreak.jsonl`, `indirect_attack.jsonl`
   - **UPIA**: Direct user prompt injection
   - **XPIA**: Context/document injection
   - Baseline + attack variants for comparison

### Custom Simulator Behavior
1. Copy `examples/generate_with_custom_prompty.py`, `examples/utils.py`, and `examples/custom_simulator_prompty/` to your project
2. Run: `python generate_with_custom_prompty.py`
3. Outputs: `custom_prompty_data.jsonl`
   - Override user mood/persona (e.g., "professional")
   - Control response diversity (temperature, top_p)
   - Custom query-response generation logic

## Data Formats

### Chat Completion (for SFT fine-tuning)
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is Azure ML?"},
    {"role": "assistant", "content": "Azure Machine Learning is..."}
  ]
}
```

### Q&A Format (for evaluation)
```json
{"query": "What is Azure ML?", "response": "Azure Machine Learning is..."}
```

See [examples/generate_qa_from_text.py](examples/generate_qa_from_text.py) for output conversion patterns.

## Adversarial Scenarios

| Scenario | Enum | Max Samples | Content Types |
|----------|------|------------|---|
| Q&A | ADVERSARIAL_QA | 1,384 | Hate, sexual, violence, self-harm |
| Conversation | ADVERSARIAL_CONVERSATION | 1,018 | Hate, sexual, violence, self-harm |
| Summarization | ADVERSARIAL_SUMMARIZATION | 525 | Hate, sexual, violence, self-harm |
| Search | ADVERSARIAL_SEARCH | 1,000 | Hate, sexual, violence, self-harm |
| Rewrite | ADVERSARIAL_REWRITE | 1,000 | Hate, sexual, violence, self-harm |
| Ungrounded Content | ADVERSARIAL_CONTENT_GEN_UNGROUNDED | 496 | Hate, sexual, violence, self-harm |
| Grounded Content | ADVERSARIAL_CONTENT_GEN_GROUNDED | 475 | All + jailbreak |
| Protected Material | ADVERSARIAL_PROTECTED_MATERIAL | 306 | Copyright detection |

## Integration with Training

Generated JSONL files can be uploaded to Azure ML for fine-tuning. Use `azureml://` URI paths with [azure-ml-llm-trainer](../azure-ml-llm-trainer) skill for SFT/DPO/RL.

See [examples/generate_qa_from_text.py](examples/generate_qa_from_text.py) for Azure ML data asset creation patterns.

## Customization

### User Simulator Parameters
Control response diversity and behavior with simulator kwargs. See [examples/generate_with_custom_prompty.py](examples/generate_with_custom_prompty.py) for implementation.

### Multi-Language Support
Adversarial simulators support multiple languages: Spanish, Italian, French, Japanese, Portuguese, Chinese (Simplified), German. Check example files for language parameter usage.

## Callback Pattern

Target application must be defined as async callback accepting messages dict and optional parameters. See [examples/generate_qa_from_text.py](examples/generate_qa_from_text.py) or [examples/generate_conversation.py](examples/generate_conversation.py) for callback implementation patterns.

## Notes

- **Synthetic data validation**: Always review generated samples before production use
- **Token costs**: Monitor Azure OpenAI quota; use GPT-5-mini for cost efficiency
- **Context limits**: Keep text inputs under 5,000 characters for optimal results
- **Reproducibility**: Set `randomization_seed` for consistent results across runs
- **Regional availability**: Adversarial simulators require supported regions (see Prerequisites)
- **Ethical use**: Adversarial scenarios for testing/evaluation only; not for malicious use

## Common Patterns

- **Fine-Tuning Dataset**: See [examples/generate_qa_from_text.py](examples/generate_qa_from_text.py) for SFT data generation from text
- **Safety Benchmarking**: See [examples/generate_adversarial.py](examples/generate_adversarial.py) and [examples/generate_jailbreak_attacks.py](examples/generate_jailbreak_attacks.py) for multi-scenario safety data
- **Multi-Turn Evaluation**: See [examples/generate_conversation.py](examples/generate_conversation.py) for conversation dataset generation

## References

- [Azure AI Foundry Simulator](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/simulator-interaction-data)
- [Evaluation Datasets Guide](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/evaluate-sdk#data-requirements-for-built-in-evaluators)
- [Adversarial Scenarios](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/risk-safety-evaluators)
