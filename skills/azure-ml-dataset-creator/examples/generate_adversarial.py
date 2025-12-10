"""
Generate adversarial safety evaluation data using Azure AI Foundry AdversarialSimulator.

This example shows how to:
1. Set up adversarial simulation for safety testing
2. Generate data for multiple safety scenarios
3. Save adversarial test data for evaluation
"""

import os
import asyncio
from azure.ai.evaluation.simulator import AdversarialSimulator, AdversarialScenario
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from utils import get_azure_openai_token_provider

# Azure AI Project configuration (required for adversarial simulation)
azure_ai_project = {
    "subscription_id": os.environ["AZURE_SUBSCRIPTION_ID"],
    "resource_group_name": os.environ["AZURE_RESOURCE_GROUP"],
    "project_name": os.environ["AZUREML_PROJECT_NAME"],
}

# Model configuration
model_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
model_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
model_deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]


async def callback(messages, stream=False, session_state=None, context=None):
    query = messages["messages"][-1]["content"]
    
    client = AzureOpenAI(
        azure_endpoint=model_endpoint,
        api_version=api_version,
        azure_ad_token_provider=get_azure_openai_token_provider(),
    )
    
    response = client.chat.completions.create(
        model=model_deployment,
        messages=[{"role": "user", "content": query}],
        max_tokens=800,
        temperature=0.7,
    ).choices[0].message.content
    
    messages["messages"].append({"content": response, "role": "assistant", "context": {}})
    return {"messages": messages["messages"], "stream": stream, "session_state": session_state, "context": context}


async def generate_adversarial_data(scenario: AdversarialScenario, output_file: str, max_results: int = 50):
    simulator = AdversarialSimulator(credential=DefaultAzureCredential(), azure_ai_project=azure_ai_project)
    
    outputs = await simulator(
        scenario=scenario,
        target=callback,
        max_conversation_turns=1,
        max_simulation_results=max_results,
    )
    
    with open(output_file, "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(output.to_eval_qa_json_lines())
    
    print(f"{scenario.value}: {len(outputs)} samples → {output_file}")


async def main():
    scenarios = [
        (AdversarialScenario.ADVERSARIAL_QA, "adversarial_qa.jsonl", 30),
        (AdversarialScenario.ADVERSARIAL_CONVERSATION, "adversarial_conversation.jsonl", 20),
        (AdversarialScenario.ADVERSARIAL_SUMMARIZATION, "adversarial_summarization.jsonl", 20),
    ]
    
    for scenario, output_file, max_results in scenarios:
        await generate_adversarial_data(scenario, output_file, max_results)


if __name__ == "__main__":
    # Note: Adversarial simulator requires project in supported regions:
    # East US 2, France Central, UK South, Sweden Central
    asyncio.run(main())
