"""
Multi-turn conversation generation using Azure AI Foundry Simulator.

This example shows how to:
1. Generate realistic multi-turn conversations
2. Use predefined conversation starters
3. Customize user simulator behavior
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
    
    conversation_turns = [
        ["Hello, I need help with Azure Machine Learning", "How do I create a training job?", "Can you explain compute targets?"],
        ["I want to fine-tune a language model", "What datasets do I need?", "How long does training typically take?"],
        ["What is the difference between SFT and DPO?", "When should I use each method?", "Can I use both in a single workflow?"],
    ]
    
    outputs = await simulator(
        target=callback,
        conversation_turns=conversation_turns,
        max_conversation_turns=5,
        user_simulator_prompty_kwargs={"temperature": 0.7, "top_p": 0.9}
    )
    
    with open("conversation_data.jsonl", "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(output.to_json_lines())
    
    print(f"Generated {len(outputs)} conversations → conversation_data.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
