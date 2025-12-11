"""
Generate synthetic customer-support conversations with Azure AI Foundry Simulator.
Outputs: train.jsonl and test.jsonl with chat-completion `messages` plus evaluation fields.
Required env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION.
Authentication: DefaultAzureCredential (supports Azure CLI login, managed identity, etc.).
"""
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from azure.ai.evaluation.simulator import Simulator
from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI

DATA_DIR = Path(__file__).resolve().parent


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


MODEL_CONFIG = {
    "azure_endpoint": _env("AZURE_OPENAI_ENDPOINT"),
    "azure_deployment": _env("AZURE_OPENAI_DEPLOYMENT"),
    "api_version": _env("AZURE_OPENAI_API_VERSION"),
}


def _token_provider():
    credential = DefaultAzureCredential()

    def provider() -> str:
        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        return token.token

    return provider


def _conversation_to_dict(conversation: Any) -> Dict[str, Any]:
    if hasattr(conversation, "to_json"):
        return json.loads(conversation.to_json())
    if hasattr(conversation, "to_json_lines"):
        return json.loads(conversation.to_json_lines())
    raise ValueError("Unexpected simulator conversation payload")


async def support_target(messages: Dict[str, Any], stream: bool = False, session_state=None, context=None):
    client = AzureOpenAI(
        azure_endpoint=MODEL_CONFIG["azure_endpoint"],
        api_version=MODEL_CONFIG["api_version"],
        azure_ad_token_provider=_token_provider(),
    )
    prompt_messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are Contoso's concise customer support agent. Keep answers actionable and under 80 words. "
                "If the request is ambiguous, ask one clarifying question."
            ),
        },
        *messages["messages"],
    ]
    completion = client.chat.completions.create(
        model=MODEL_CONFIG["azure_deployment"],
        messages=prompt_messages,
        max_tokens=256,
        temperature=0.6,
    ).choices[0].message.content
    messages["messages"].append({"role": "assistant", "content": completion})
    return {"messages": messages["messages"], "stream": stream, "session_state": session_state, "context": context}


async def main() -> None:
    conversation_turns = [
        ["I forgot my password and the email link expired", "How can I log back in?"],
        ["My order is late", "Can you track it?", "Can I change the delivery address?"],
        ["I was charged twice", "How do I get a refund?"],
        ["The promo code did not apply", "Can you adjust my bill?"],
        ["The app keeps crashing", "I already reinstalled", "What else can I try?"],
        ["How do I cancel my subscription?", "Will I lose my data?"],
        ["Does your product work internationally?", "What are the shipping fees?"],
        ["I need to update my payment method", "Is Apple Pay supported?"],
    ]

    simulator = Simulator(model_config=MODEL_CONFIG)
    outputs = await simulator(
        target=support_target,
        conversation_turns=conversation_turns,
        max_conversation_turns=3,
        user_simulator_prompty_kwargs={"temperature": 0.5, "top_p": 0.9},
    )

    records: List[Dict[str, Any]] = []
    for conversation in outputs:
        conv_dict = _conversation_to_dict(conversation)
        messages = conv_dict.get("messages") or []
        if not messages:
            continue
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        last_assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
        if not last_user or not last_assistant:
            continue
        system_message = {
            "role": "system",
            "content": (
                "You are Contoso's concise customer support agent. Keep answers actionable and under 80 words. "
                "If the request is ambiguous, ask one clarifying question."
            ),
        }
        records.append(
            {
                "messages": [system_message, *messages],
                "query": last_user["content"],
                "response": last_assistant["content"],
                "ground_truth": last_assistant["content"],
                "context": "Customer support chat for Contoso retail platform.",
            }
        )

    if not records:
        raise RuntimeError("No simulator conversations were generated")

    random.shuffle(records)
    split = max(1, int(len(records) * 0.8))
    train, test = records[:split], records[split:]
    with open(DATA_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")
    with open(DATA_DIR / "test.jsonl", "w", encoding="utf-8") as f:
        for row in test:
            f.write(json.dumps(row) + "\n")

    print(f"Generated {len(train)} train and {len(test)} test examples → {DATA_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
