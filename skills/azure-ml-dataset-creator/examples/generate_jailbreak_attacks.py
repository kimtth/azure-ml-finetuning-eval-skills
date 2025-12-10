"""
Jailbreak attack simulation (Direct and Indirect attacks).

This example shows how to:
1. Generate direct attack (UPIA) datasets
2. Generate indirect attack (XPIA) datasets
3. Compare baseline vs. attack datasets
"""

import os
import asyncio
from azure.ai.evaluation.simulator import (
    DirectAttackSimulator,
    IndirectAttackSimulator,
    AdversarialScenario
)
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from utils import get_azure_openai_token_provider

# Azure AI Project configuration
azure_ai_project = {
    "subscription_id": os.environ["AZURE_SUBSCRIPTION_ID"],
    "resource_group_name": os.environ["AZURE_RESOURCE_GROUP"],
    "project_name": os.environ["AZUREML_PROJECT_NAME"],
}

# Model configuration
model_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
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


async def generate_direct_attack():
    simulator = DirectAttackSimulator(azure_ai_project=azure_ai_project, credential=DefaultAzureCredential())
    
    outputs = await simulator(
        target=callback,
        scenario=AdversarialScenario.ADVERSARIAL_CONVERSATION,
        max_simulation_results=20,
        max_conversation_turns=3
    )
    
    with open("direct_attack_baseline.jsonl", "w", encoding="utf-8") as f:
        for output in outputs[0]:
            f.write(output.to_eval_qa_json_lines())
    
    with open("direct_attack_jailbreak.jsonl", "w", encoding="utf-8") as f:
        for output in outputs[1]:
            f.write(output.to_eval_qa_json_lines())
    
    print(f"Direct attack: {len(outputs[0])} baseline + {len(outputs[1])} jailbreak samples")


async def generate_indirect_attack():
    simulator = IndirectAttackSimulator(azure_ai_project=azure_ai_project, credential=DefaultAzureCredential())
    
    outputs = await simulator(
        target=callback,
        max_simulation_results=20,
        max_conversation_turns=3
    )
    
    with open("indirect_attack.jsonl", "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(output.to_eval_qa_json_lines())
    
    print(f"Indirect attack: {len(outputs)} samples → indirect_attack.jsonl")


async def main():
    await generate_direct_attack()
    await generate_indirect_attack()


if __name__ == "__main__":
    # Note: Requires project in supported regions:
    # East US 2, France Central, UK South, Sweden Central
    asyncio.run(main())
