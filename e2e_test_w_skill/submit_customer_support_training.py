"""
Submit fine-tuning job for customer support model using Azure ML.

This script:
1. Uploads the training dataset to Azure ML datastore
2. Submits an SFT training job using Phi-4-mini
3. Monitors job status and provides studio URL
"""

import argparse
import os
import sys
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.entities import Environment, Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


def main():
    parser = argparse.ArgumentParser(description="Submit customer support model training job")
    parser.add_argument("--subscription-id", default=os.getenv("AZURE_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=os.getenv("AZURE_RESOURCE_GROUP"))
    parser.add_argument("--workspace", default=os.getenv("AZUREML_WORKSPACE_NAME"))
    parser.add_argument("--compute", required=True, help="Compute cluster name (GPU recommended)")
    parser.add_argument(
        "--train-data",
        default="e2e_test/customer_support_train.jsonl",
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--model-name",
        default="azureml://registries/azureml/models/Phi-3.5-mini-instruct/versions/4",
        help="Base model to fine-tune (use Phi-3.5 or Phi-4 mini models)"
    )
    parser.add_argument("--experiment-name", default="customer-support-sft")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()
    
    # Validate environment
    if not args.subscription_id or not args.resource_group or not args.workspace:
        print("Error: Azure credentials not configured")
        print("\nSet environment variables:")
        print("  AZURE_SUBSCRIPTION_ID")
        print("  AZURE_RESOURCE_GROUP")
        print("  AZUREML_WORKSPACE_NAME")
        print("\nOr pass via command line: --subscription-id, --resource-group, --workspace")
        return 1
    
    if not Path(args.train_data).exists():
        print(f"Error: Training data not found: {args.train_data}")
        print("\nRun generate_customer_support_data.py first to create training data")
        return 1
    
    print("Connecting to Azure ML workspace...")
    ml_client = MLClient(
        DefaultAzureCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace
    )
    
    # Upload training data to datastore
    print(f"Uploading training data: {args.train_data}")
    data_name = "customer_support_train_data"
    data_asset = Data(
        path=args.train_data,
        type=AssetTypes.URI_FILE,
        description="Customer support Q&A training data",
        name=data_name,
    )
    
    data_asset = ml_client.data.create_or_update(data_asset)
    print(f"✓ Data uploaded: {data_asset.id}")
    
    # Define training environment
    skill_dir = Path(__file__).resolve().parent.parent / "skills" / "azure-ml-llm-trainer" / "examples"
    
    env = Environment(
        name="customer-support-sft-env",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
        conda_file=str(skill_dir / "environment" / "conda.yml"),
    )
    
    # Create training command job
    job = command(
        code=str(skill_dir / "src"),
        command=(
            "python train_sft.py "
            "--model_name ${{inputs.model_name}} "
            "--train_data ${{inputs.train_data}} "
            "--output_dir ${{outputs.model_dir}} "
            f"--batch_size {args.batch_size} "
            f"--epochs {args.epochs} "
            f"--lr {args.learning_rate}"
        ),
        inputs={
            "model_name": args.model_name,
            "train_data": Input(type="uri_file", path=data_asset.id, mode="ro_mount"),
        },
        outputs={"model_dir": Output(type="uri_folder", mode="rw_mount")},
        environment=env,
        compute=args.compute,
        display_name="customer-support-sft",
        experiment_name=args.experiment_name,
    )
    
    print("\nSubmitting training job...")
    print(f"  Model: {args.model_name}")
    print(f"  Compute: {args.compute}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    
    submitted_job = ml_client.jobs.create_or_update(job)
    
    print("\n" + "="*70)
    print("Training job submitted successfully!")
    print("="*70)
    print(f"\nJob name: {submitted_job.name}")
    
    studio_service = submitted_job.services.get("Studio") if submitted_job.services else None
    studio_url = studio_service.endpoint if studio_service else None
    
    if studio_url:
        print(f"Studio URL: {studio_url}")
    else:
        print("Open the job in Azure ML Studio to monitor progress")
    
    print("\nTo check job status:")
    print(f"  az ml job show --name {submitted_job.name} -w {args.workspace} -g {args.resource_group}")
    
    print("\nAfter training completes:")
    print("  1. Download the model from the job outputs")
    print("  2. Run evaluation script to test model performance")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
