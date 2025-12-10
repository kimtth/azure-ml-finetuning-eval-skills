"""
Utility functions for Azure AI evaluation examples.
"""

import os
from azure.identity import DefaultAzureCredential


def get_azure_openai_model_config():
    """
    Get Azure OpenAI model configuration for AI-assisted evaluators.
    
    Returns:
        dict: Model configuration with endpoint, deployment, and API details.
    """
    return {
        "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "azure_deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"],
        "api_version": os.environ["AZURE_OPENAI_API_VERSION"],
    }


def get_azure_ai_project_config():
    """
    Get Azure AI Foundry project configuration.
    
    Returns:
        dict: Project configuration with subscription, resource group, and project name.
    """
    return {
        "subscription_id": os.environ["AZURE_SUBSCRIPTION_ID"],
        "resource_group_name": os.environ["AZURE_RESOURCE_GROUP"],
        "project_name": os.environ["AZUREML_PROJECT_NAME"],
    }


def create_sample_test_data(filename="test_data.jsonl", num_samples=3):
    """
    Create sample test data in JSONL format.
    
    Args:
        filename: Output filename
        num_samples: Number of samples to generate (default: 3)
    
    Returns:
        str: Path to created file
    """
    import json
    
    test_data = [
        {
            "query": "What is Azure Machine Learning?",
            "response": "Azure Machine Learning is a cloud-based service for building, training, and deploying machine learning models.",
            "context": "Azure Machine Learning is a comprehensive cloud service that enables data scientists and developers to build, train, and deploy machine learning models at scale.",
            "ground_truth": "Azure ML is a cloud service for machine learning."
        },
        {
            "query": "How do I create a compute cluster in Azure ML?",
            "response": "Use the Azure ML SDK or portal to create a compute cluster by specifying VM size, min/max nodes, and other settings.",
            "context": "You can create compute clusters in Azure ML using the Python SDK with MLClient, the CLI, or through the Azure portal. Specify the VM size, node count range, and idle time before scale down.",
            "ground_truth": "Create a compute cluster using SDK, CLI, or portal with VM size and node configuration."
        },
        {
            "query": "What is a training job in Azure ML?",
            "response": "A training job is a run of your training script on specified compute resources to train a machine learning model.",
            "context": "Training jobs in Azure ML execute your training code on compute targets like clusters or instances. They track metrics, log outputs, and can register trained models.",
            "ground_truth": "A training job runs your code on compute to train models."
        },
    ]
    
    # Limit to requested number
    test_data = test_data[:num_samples]
    
    with open(filename, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    
    return filename
