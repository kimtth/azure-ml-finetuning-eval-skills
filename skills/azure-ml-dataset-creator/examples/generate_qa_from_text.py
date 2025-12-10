"""
Generate Q&A pairs from text source using Azure AI Foundry Simulator.

This example shows how to:
1. Extract text from Wikipedia or documents
2. Generate synthetic query-response pairs
3. Save training data in JSONL format for fine-tuning
"""

import os
import asyncio
import wikipedia
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
    
    messages["messages"].append({
        "content": response,
        "role": "assistant",
        "context": {"citations": None},
    })
    
    return {"messages": messages["messages"], "stream": stream, "session_state": session_state, "context": context}


async def main():
    simulator = Simulator(model_config=model_config)
    
    wiki_search_term = "Azure Machine Learning"
    wiki_page = wikipedia.page(wikipedia.search(wiki_search_term)[0])
    text = wiki_page.summary[:5000]
    
    tasks = [
        f"I am a student learning about {wiki_search_term}",
        f"I am a data scientist evaluating {wiki_search_term}",
        f"I am a teacher explaining {wiki_search_term}",
        f"I need technical details about {wiki_search_term}",
    ]
    
    outputs = await simulator(
        target=callback,
        text=text,
        num_queries=20,
        max_conversation_turns=1,
        tasks=tasks,
    )
    
    with open("training_data.jsonl", "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(output.to_eval_qa_json_lines())
    
    print(f"Generated {len(outputs)} Q&A pairs → training_data.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
