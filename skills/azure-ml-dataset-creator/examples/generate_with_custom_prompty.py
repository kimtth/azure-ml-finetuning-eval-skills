"""
Custom prompty templates for simulator customization.

This example shows how to:
1. Use custom user simulator prompty
2. Customize query-response generation
3. Control simulator behavior with prompty kwargs
"""

import os
import asyncio
from azure.ai.evaluation.simulator import Simulator
from openai import AzureOpenAI
from utils import get_azure_openai_token_provider

# Configuration
model_config = {
    "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
    "azure_deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"],
    "api_version": os.environ["AZURE_OPENAI_API_VERSION"],
}


async def callback(messages, stream=False, session_state=None, context=None):
    query = messages["messages"][-1]["content"]
    
    client = AzureOpenAI(
        azure_endpoint=model_config["azure_endpoint"],
        api_version=model_config["api_version"],
        azure_ad_token_provider=get_azure_openai_token_provider(),
    )
    
    response = client.chat.completions.create(
        model=model_config["azure_deployment"],
        messages=[{"role": "user", "content": query}],
        max_tokens=800,
        temperature=0.7,
    ).choices[0].message.content
    
    messages["messages"].append({"content": response, "role": "assistant", "context": {"citations": None}})
    return {"messages": messages["messages"], "stream": stream, "session_state": session_state, "context": context}


async def main():
    simulator = Simulator(model_config=model_config)
    
    text = """Azure Machine Learning provides managed compute resources for training models.
    You can create compute clusters with GPU or CPU virtual machines.
    Clusters scale automatically based on your workload requirements."""
    
    outputs = await simulator(
        target=callback,
        text=text,
        num_queries=5,
        max_conversation_turns=2,
        tasks=["Learn about Azure ML compute"],
        user_simulating_prompty="custom_simulator_prompty/user_override.prompty",
        user_simulator_prompty_kwargs={"temperature": 0.7, "top_p": 0.9, "mood": "professional"},
    )
    
    with open("custom_prompty_data.jsonl", "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(output.to_eval_qa_json_lines())
    
    print(f"Generated {len(outputs)} samples with custom prompty → custom_prompty_data.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
