"""
Cloud evaluation using Azure AI Foundry SDK.

This example shows how to:
1. Upload evaluation dataset to Azure
2. Configure built-in and custom evaluators
3. Submit evaluation job to run in the cloud
4. Monitor evaluation status
"""

import os
import time
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    Evaluation,
    InputDataset,
    EvaluatorConfiguration,
    EvaluatorIds,
)
from utils import create_sample_test_data

# Environment configuration
PROJECT_ENDPOINT = os.environ["PROJECT_ENDPOINT"]  # https://<account>.services.ai.azure.com/api/projects/<project>
MODEL_ENDPOINT = os.environ["MODEL_ENDPOINT"]  # https://<account>.services.ai.azure.com
MODEL_API_KEY = os.environ["MODEL_API_KEY"]
MODEL_DEPLOYMENT_NAME = os.environ["MODEL_DEPLOYMENT_NAME"]  # e.g., gpt-4o-mini

DATASET_NAME = os.environ.get("DATASET_NAME", "eval-dataset-test")
DATASET_VERSION = os.environ.get("DATASET_VERSION", "1.0")


def main():
    test_file = create_sample_test_data("evaluate_test_data.jsonl")
    
    project_client = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )
    data_upload = project_client.datasets.upload_file(
        name=DATASET_NAME,
        version=DATASET_VERSION,
        file_path=test_file,
    )
    data_id = data_upload.id
    
    evaluators = {
        "relevance": EvaluatorConfiguration(
            id=EvaluatorIds.RELEVANCE.value,
            init_params={"deployment_name": MODEL_DEPLOYMENT_NAME},
            data_mapping={
                "query": "${data.query}",
                "response": "${data.response}",
            },
        ),
        "groundedness": EvaluatorConfiguration(
            id=EvaluatorIds.GROUNDEDNESS.value,
            init_params={"deployment_name": MODEL_DEPLOYMENT_NAME},
            data_mapping={
                "query": "${data.query}",
                "response": "${data.response}",
                "context": "${data.context}",
            },
        ),
        "violence": EvaluatorConfiguration(
            id=EvaluatorIds.VIOLENCE.value,
            init_params={"azure_ai_project": PROJECT_ENDPOINT},
            data_mapping={
                "query": "${data.query}",
                "response": "${data.response}",
            },
        ),
        "coherence": EvaluatorConfiguration(
            id=EvaluatorIds.COHERENCE.value,
            init_params={"deployment_name": MODEL_DEPLOYMENT_NAME},
            data_mapping={
                "query": "${data.query}",
                "response": "${data.response}",
            },
        ),
    }
    
    evaluation = Evaluation(
        display_name="Cloud Evaluation - Test Run",
        description="Example cloud evaluation with built-in evaluators",
        data=InputDataset(id=data_id),
        evaluators=evaluators,
    )
    
    # Submit evaluation
    evaluation_response = project_client.evaluations.create(
        evaluation,
        headers={
            "model-endpoint": MODEL_ENDPOINT,
            "api-key": MODEL_API_KEY,
        },
    )
    
    print(f"Evaluation submitted: {evaluation_response.name}")
    print(f"Status: {evaluation_response.status}")
    
    while True:
        status = project_client.evaluations.get(evaluation_response.name).status
        if status in ["Completed", "Failed", "Canceled"]:
            break
        time.sleep(10)
    
    print(f"\nEvaluation {status.lower()}: {evaluation_response.name}")
    print("View in portal: Evaluation → Evaluation runs")


if __name__ == "__main__":
    main()
