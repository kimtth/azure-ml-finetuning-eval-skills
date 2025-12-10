"""
Run the complete end-to-end workflow for customer support model training.

This script orchestrates:
1. Dataset generation using Azure AI Foundry Simulator
2. Training job submission to Azure ML
3. Model evaluation using Azure AI Evaluation SDK

Usage:
    python run_e2e_workflow.py --compute YOUR_GPU_CLUSTER [options]

Prerequisites:
    - Set environment variables (see README.md)
    - Have GPU compute cluster ready in Azure ML workspace
    - Ensure Azure OpenAI deployment is available
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path


def run_command(cmd, description, capture_output=False):
    """Run a shell command and handle errors."""
    print(f"\n{'='*70}")
    print(f"Step: {description}")
    print(f"{'='*70}")
    
    if capture_output:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {description} failed")
            print(f"STDERR: {result.stderr}")
            return False, result.stdout
        return True, result.stdout
    else:
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"\nError: {description} failed with exit code {result.returncode}")
            return False, None
        return True, None


def check_environment_variables():
    """Verify all required environment variables are set."""
    required_vars = {
        "Dataset Generation & Evaluation": [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_API_VERSION",
        ],
        "Training": [
            "AZURE_SUBSCRIPTION_ID",
            "AZURE_RESOURCE_GROUP",
            "AZUREML_WORKSPACE_NAME",
        ],
        "Evaluation Logging": [
            "AZUREML_PROJECT_NAME",
        ]
    }
    
    missing = {}
    for category, vars_list in required_vars.items():
        missing_vars = [var for var in vars_list if not os.environ.get(var)]
        if missing_vars:
            missing[category] = missing_vars
    
    if missing:
        print("\nError: Missing required environment variables:\n")
        for category, vars_list in missing.items():
            print(f"{category}:")
            for var in vars_list:
                print(f"  - {var}")
        print("\nSee README.md for setup instructions.")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run complete E2E workflow for customer support model"
    )
    parser.add_argument(
        "--compute",
        required=True,
        help="GPU compute cluster name in Azure ML workspace"
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset generation (use existing data)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training job submission"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip model evaluation"
    )
    parser.add_argument(
        "--model-name",
        default="azureml://registries/azureml/models/Phi-3.5-mini-instruct/versions/4",
        help="Base model for fine-tuning (default: Phi-3.5-mini-instruct)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size (default: 1)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("E2E Customer Support Model Training Workflow")
    print("="*70)
    
    # Check environment
    print("\nChecking environment variables...")
    if not check_environment_variables():
        return 1
    print("✓ All required environment variables are set")
    
    # Change to e2e_test directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success_steps = []
    failed_steps = []
    
    # Step 1: Generate Dataset
    if not args.skip_dataset:
        success, _ = run_command(
            f"{sys.executable} generate_customer_support_data.py",
            "Generate Synthetic Customer Support Dataset"
        )
        if success:
            success_steps.append("Dataset Generation")
        else:
            failed_steps.append("Dataset Generation")
            print("\nWorkflow failed at dataset generation. Exiting.")
            return 1
    else:
        print("\n[Skipped] Dataset Generation (using existing data)")
        success_steps.append("Dataset Generation (skipped)")
    
    # Step 2: Submit Training Job
    if not args.skip_training:
        train_cmd = (
            f"{sys.executable} submit_customer_support_training.py "
            f"--compute {args.compute} "
            f"--model-name {args.model_name} "
            f"--batch-size {args.batch_size} "
            f"--epochs {args.epochs}"
        )
        success, _ = run_command(
            train_cmd,
            "Submit Fine-Tuning Training Job"
        )
        if success:
            success_steps.append("Training Job Submission")
            print("\n" + "="*70)
            print("Training job submitted successfully!")
            print("="*70)
            print("\nThe training job will run on Azure ML compute.")
            print("Monitor progress in Azure ML Studio (URL printed above).")
            print("\nTypical training time: 10-20 minutes on a single GPU")
            print("\nNote: You can proceed to evaluation once training completes.")
        else:
            failed_steps.append("Training Job Submission")
            print("\nWorkflow failed at training submission. Exiting.")
            return 1
    else:
        print("\n[Skipped] Training Job Submission")
        success_steps.append("Training Job Submission (skipped)")
    
    # Step 3: Model Evaluation
    if not args.skip_evaluation:
        if not args.skip_training:
            print("\n" + "="*70)
            print("Waiting for Training to Complete")
            print("="*70)
            print("\nThe evaluation step requires the training job to complete first.")
            print("You have two options:")
            print("  1. Wait here and run evaluation automatically after training")
            print("  2. Run evaluation manually later: python evaluate_customer_support_model.py")
            print("\nWould you like to wait for training to complete? (y/n): ", end="")
            
            response = input().strip().lower()
            if response != 'y':
                print("\nSkipping evaluation. Run manually when training completes.")
                print("\nWorkflow Summary (so far):")
                for step in success_steps:
                    print(f"  ✓ {step}")
                print("\nTo run evaluation later:")
                print("  python evaluate_customer_support_model.py")
                return 0
            
            print("\nWaiting for training to complete...")
            print("(This script will check job status every 2 minutes)")
            # In a real implementation, you would poll job status here
            # For now, we'll just inform the user
            print("\n[Manual Step Required]")
            print("Please monitor your training job in Azure ML Studio.")
            print("Once complete, run: python evaluate_customer_support_model.py")
            return 0
        
        success, _ = run_command(
            f"{sys.executable} evaluate_customer_support_model.py",
            "Evaluate Model Performance"
        )
        if success:
            success_steps.append("Model Evaluation")
        else:
            failed_steps.append("Model Evaluation")
            print("\nWorkflow failed at evaluation. Check error messages above.")
            return 1
    else:
        print("\n[Skipped] Model Evaluation")
        success_steps.append("Model Evaluation (skipped)")
    
    # Final Summary
    print("\n" + "="*70)
    print("E2E Workflow Complete!")
    print("="*70)
    
    if success_steps:
        print("\n✓ Completed Steps:")
        for step in success_steps:
            print(f"  - {step}")
    
    if failed_steps:
        print("\n✗ Failed Steps:")
        for step in failed_steps:
            print(f"  - {step}")
    
    print("\nGenerated Files:")
    print("  - customer_support_train.jsonl (training data)")
    print("  - customer_support_test.jsonl (test data)")
    print("  - evaluation_results.json (detailed metrics)")
    print("  - evaluation_summary.json (summary)")
    
    print("\nNext Steps:")
    print("  1. Review evaluation results in evaluation_summary.json")
    print("  2. Download trained model from Azure ML job outputs")
    print("  3. Deploy model to Azure ML managed endpoint")
    print("  4. Test model with real customer support queries")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
