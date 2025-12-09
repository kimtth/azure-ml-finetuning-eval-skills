import argparse
import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-id", default=os.getenv("AZURE_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=os.getenv("AZURE_RESOURCE_GROUP"))
    parser.add_argument("--workspace", default=os.getenv("AZUREML_WORKSPACE_NAME"))
    parser.add_argument("--compute", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--model-name",
        default="azureml://registries/azureml/models/Phi-3-mini-4k-instruct/versions/1",
    )
    parser.add_argument("--experiment-name", default="azureml-llm-dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--l2_multiplier", type=float, default=0.1)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    env = Environment(
        name="azureml-llm-dpo-env",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
        conda_file=str(project_dir / "environment" / "conda.yml"),
    )

    ml_client = MLClient(
        DefaultAzureCredential(), args.subscription_id, args.resource_group, args.workspace
    )

    job = command(
        code=str(project_dir / "src"),
        command=(
            "python train_dpo.py "
            "--model_name ${{inputs.model_name}} "
            "--train_data ${{inputs.train_data}} "
            "--output_dir ${{outputs.model_dir}} "
            "--beta ${{inputs.beta}} "
            "--l2_multiplier ${{inputs.l2_multiplier}}"
        ),
        inputs={
            "model_name": args.model_name,
            "train_data": Input(type="uri_file", path=args.data_path, mode="ro_mount"),
            "beta": args.beta,
            "l2_multiplier": args.l2_multiplier,
        },
        outputs={"model_dir": Output(type="uri_folder", mode="rw_mount")},
        environment=env,
        compute=args.compute,
        display_name="azureml-llm-dpo",
        experiment_name=args.experiment_name,
    )

    submitted_job = ml_client.jobs.create_or_update(job)
    print(f"Job: {submitted_job.name}")
    studio_service = submitted_job.services.get("Studio") if submitted_job.services else None
    studio_url = studio_service.endpoint if studio_service else None
    print(f"Studio: {studio_url or 'Open the job in Azure ML studio to monitor'}")


if __name__ == "__main__":
    main()
