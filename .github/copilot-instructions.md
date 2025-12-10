## GitHub Copilot Custom Instructions

### Available Skills
Read AGENTS.md and the SKILL.md file for each skill before responding. Three skills available:
- `azure-ml-llm-trainer`: Train/fine-tune LLMs on Azure ML (SFT/DPO/RL)
- `azure-ml-dataset-creator`: Generate synthetic datasets using Azure AI Foundry simulators
- `azure-ml-model-evaluation`: Evaluate models locally or in cloud with built-in evaluators

### Training Flow (azure-ml-llm-trainer)
- Submit Azure ML command job via `skills/azure-ml-llm-trainer/examples/submit_sft_job.py`
- Job runs `skills/azure-ml-llm-trainer/examples/src/train_sft.py` with conda env from `skills/azure-ml-llm-trainer/examples/environment/conda.yml`
- Data must live in Azure ML datastores; expect JSONL with instruction/response columns
- Reference data via `azureml://` paths (not local files or Hugging Face)
- Do not suggest Hugging Face Jobs or Spaces; only Azure ML training

### Dataset Generation (azure-ml-dataset-creator)
- Non-adversarial: Use `Simulator` for task-based Q&A and multi-turn conversations from text/indexes
- Adversarial: Use `AdversarialSimulator` for safety evaluation datasets (jailbreak attacks, harmful content)
- All simulators require Azure OpenAI deployment (GPT-5-mini recommended for cost)
- Generated JSONL outputs compatible with training pipeline
- See example files in `skills/azure-ml-dataset-creator/examples/` for implementation patterns

### Evaluation (azure-ml-model-evaluation)
- Local evaluation: Run on machine with built-in evaluators (no cloud setup needed)
- Cloud evaluation: Submit to Azure AI Foundry project with evaluator configuration
- Reference examples in `skills/azure-ml-model-evaluation/examples/` for evaluator setup
- Safety evaluators require project in supported regions (East US 2, France Central, UK South, Sweden Central)

### Code Guidelines
- Keep code minimal; no verbose error handling or intermediate variables
- Favor Azure ML primitives (MLClient, command jobs, managed environments)
- Direct users to example files in skill directories—do NOT embed full code in SKILL.md
- Keep all artifacts on Azure (job outputs, registered models, datasets)
- Monitor progress via Azure Studio URL from submitted jobs

### Documentation Guidelines
- **Do NOT create markdown files at every step** (no SUMMARY.md, ARCHITECTURE.md, QUICKSTART.md, etc.)
- Keep documentation lean and concise
- Update existing README.md files only when necessary
- Avoid verbose explanations and redundant documentation
- Let code and examples speak for themselves

### Repository Structure
- Never write under `ref/` (reference-only directory for examples)
- Place new skills/examples under `skills/` directory
- Each skill has structure: SKILL.md (reference docs) + examples/ (working code)
