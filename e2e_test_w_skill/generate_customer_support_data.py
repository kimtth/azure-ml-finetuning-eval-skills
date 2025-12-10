"""
Generate synthetic customer support Q&A dataset for LLM fine-tuning.

This script:
1. Uses Azure AI Foundry Simulator to generate customer support conversations
2. Creates training data in JSONL format (80% for fine-tuning)
3. Creates test data in JSONL format (20% for evaluation)
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add examples directory to path for utils
sys.path.append(str(Path(__file__).parent.parent / "skills" / "azure-ml-dataset-creator" / "examples"))
from utils import get_azure_openai_token_provider

from azure.ai.evaluation.simulator import Simulator
from openai import AzureOpenAI


# Customer support knowledge base
CUSTOMER_SUPPORT_TEXT = """
Azure Machine Learning Customer Support Guide:

Account and Billing:
- You can manage your subscription in the Azure portal under Cost Management + Billing
- Azure ML charges for compute, storage, and data transfer
- You can set up cost alerts to monitor spending
- Free tier includes limited compute hours for experimentation

Compute Clusters:
- Compute clusters can be CPU or GPU-based
- Minimum node count can be 0 for auto-scaling to save costs
- Maximum node count depends on your subscription quota
- Idle time before scale down is configurable (default 120 seconds)
- Spot instances available for cost savings on non-critical workloads

Training Jobs:
- Training jobs run on compute targets you specify
- Jobs can use environments from Docker images or conda files
- You can track experiments and compare metrics across runs
- Distributed training is supported for large models
- Failed jobs can be restarted from checkpoints

Model Deployment:
- Deploy models to managed online endpoints for real-time inference
- Batch endpoints are available for large-scale scoring
- Blue-green deployment pattern supported for safe updates
- Auto-scaling based on request load
- Model monitoring tracks performance and data drift

Troubleshooting:
- Check job logs in the Outputs + logs tab
- Verify compute cluster is running before submitting jobs
- Ensure data paths use azureml:// URIs for datastores
- Check IAM permissions if you get access denied errors
- Monitor quota limits in the Azure portal

Common Error Messages:
- "Compute target not found" - Verify compute cluster name and region
- "Dataset not found" - Ensure dataset is registered and path is correct
- "Out of memory" - Increase VM size or reduce batch size
- "Authentication failed" - Run az login and check credentials
- "Quota exceeded" - Request quota increase in Azure portal
"""


async def customer_support_callback(messages, stream=False, session_state=None, context=None):
    """
    Callback function that simulates a customer support assistant.
    """
    query = messages["messages"][-1]["content"]
    
    model_config = {
        "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "azure_deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"],
        "api_version": os.environ["AZURE_OPENAI_API_VERSION"],
    }
    
    client = AzureOpenAI(
        azure_endpoint=model_config["azure_endpoint"],
        api_version=model_config["api_version"],
        azure_ad_token_provider=get_azure_openai_token_provider(),
    )
    
    system_prompt = """You are a helpful Azure Machine Learning customer support assistant.
    Answer questions accurately based on the documentation provided.
    Be concise, professional, and provide actionable solutions."""
    
    response = client.chat.completions.create(
        model=model_config["azure_deployment"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {CUSTOMER_SUPPORT_TEXT}\n\nQuestion: {query}"}
        ],
        max_tokens=500,
        temperature=0.7,
    ).choices[0].message.content
    
    messages["messages"].append({
        "content": response,
        "role": "assistant",
        "context": {"citations": None},
    })
    
    return {
        "messages": messages["messages"],
        "stream": stream,
        "session_state": session_state,
        "context": context
    }


async def main():
    """Generate customer support training and test datasets."""
    
    # Check required environment variables
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION"
    ]
    
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("\nRequired environment variables:")
        print("  AZURE_OPENAI_ENDPOINT - Your Azure OpenAI endpoint URL")
        print("  AZURE_OPENAI_DEPLOYMENT - Your deployment name (e.g., gpt-5-mini)")
        print("  AZURE_OPENAI_API_VERSION - API version (e.g., 2024-08-01-preview)")
        return
    
    model_config = {
        "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "azure_deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"],
        "api_version": os.environ["AZURE_OPENAI_API_VERSION"],
    }
    
    print("Initializing Azure AI Foundry Simulator...")
    simulator = Simulator(model_config=model_config)
    
    # Define customer personas for diverse queries
    tasks = [
        "I am a new user trying to understand Azure ML basics",
        "I am a data scientist experiencing technical issues",
        "I am a developer asking about deployment options",
        "I am an administrator managing costs and quotas",
        "I am troubleshooting a failed training job",
        "I am asking about compute cluster configuration",
        "I am inquiring about model deployment best practices",
        "I am confused about billing and pricing",
    ]
    
    print(f"Generating {len(tasks) * 10} customer support Q&A pairs...")
    print("This may take a few minutes...\n")
    
    outputs = await simulator(
        target=customer_support_callback,
        text=CUSTOMER_SUPPORT_TEXT,
        num_queries=80,  # 80 total queries (10 per task)
        max_conversation_turns=1,
        tasks=tasks,
    )
    
    print(f"Generated {len(outputs)} Q&A pairs")
    
    # Split into training (80%) and test (20%) sets
    split_index = int(len(outputs) * 0.8)
    train_outputs = outputs[:split_index]
    test_outputs = outputs[split_index:]
    
    # Save training data in chat completion format for SFT
    os.makedirs("e2e_test", exist_ok=True)
    train_file = "e2e_test/customer_support_train.jsonl"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for output in train_outputs:
            # Convert to chat completion format
            messages = []
            messages.append({
                "role": "system",
                "content": "You are a helpful Azure Machine Learning customer support assistant."
            })
            for msg in output["messages"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            f.write(json.dumps({"messages": messages}) + "\n")
    
    print(f"✓ Training data saved: {train_file} ({len(train_outputs)} samples)")
    
    # Save test data in evaluation format
    test_file = "e2e_test/customer_support_test.jsonl"
    
    with open(test_file, "w", encoding="utf-8") as f:
        for output in test_outputs:
            # Extract query and response
            query = None
            response = None
            for msg in output["messages"]:
                if msg["role"] == "user":
                    query = msg["content"]
                elif msg["role"] == "assistant":
                    response = msg["content"]
            
            if query and response:
                f.write(json.dumps({
                    "query": query,
                    "response": response,
                    "context": CUSTOMER_SUPPORT_TEXT[:1000],  # Truncated context
                    "ground_truth": response  # Use generated response as ground truth
                }) + "\n")
    
    print(f"✓ Test data saved: {test_file} ({len(test_outputs)} samples)")
    print("\n" + "="*70)
    print("Dataset generation complete!")
    print("="*70)
    print("\nNext steps:")
    print(f"1. Upload {train_file} to Azure ML datastore")
    print("2. Use azure-ml-llm-trainer skill to fine-tune a model")
    print(f"3. Use azure-ml-model-evaluation skill to evaluate with {test_file}")


if __name__ == "__main__":
    asyncio.run(main())
