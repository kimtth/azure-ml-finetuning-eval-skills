## GitHub Copilot Custom Instructions

- Use the Azure-only training skill at skills/azure-ml-llm-trainer/SKILL.md. Do not suggest Hugging Face Jobs or Spaces.
- Do not download base models from Hugging Face. Prefer Azure fine-tuning enabled foundation models; if uncertain, check the latest Azure ML model catalog documentation before proposing a model.
- Training flow: submit the Azure ML command job via sample/submit_sft_job.py; the job runs sample/src/train_sft.py with env from sample/environment/conda.yml.
- Data must live in Azure ML datastores; expect JSONL with instruction/response columns and reference via azureml:// paths.
- Keep code minimal (no verbose error handling). Favor Azure ML primitives (MLClient, command jobs, managed environments).
- Keep all artifacts on Azure (job outputs or registered models). Monitor via the Studio URL from the submitted job.
- Never write under ref/ (reference-only). Place new files under repository root structure.
