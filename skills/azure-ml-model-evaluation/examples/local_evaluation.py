"""
Local evaluation using Azure AI Evaluation SDK.

This example shows how to:
1. Set up quality and safety evaluators
2. Run evaluation on a test dataset
3. Log results to Azure AI Foundry project
"""

from azure.ai.evaluation import (
    RelevanceEvaluator,
    GroundednessEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    F1ScoreEvaluator,
    evaluate
)
from utils import create_sample_test_data, get_azure_ai_project_config, get_azure_openai_model_config

def main():
    model_config = get_azure_openai_model_config()
    azure_ai_project = get_azure_ai_project_config()
    create_sample_test_data()
    
    relevance_eval = RelevanceEvaluator(model_config)
    groundedness_eval = GroundednessEvaluator(model_config)
    coherence_eval = CoherenceEvaluator(model_config)
    fluency_eval = FluencyEvaluator(model_config)
    f1_eval = F1ScoreEvaluator()
    
    result = evaluate(
        data="test_data.jsonl",
        evaluators={
            "relevance": relevance_eval,
            "groundedness": groundedness_eval,
            "coherence": coherence_eval,
            "fluency": fluency_eval,
            "f1_score": f1_eval,
        },
        evaluator_config={
            "default": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                    "context": "${data.context}",
                    "ground_truth": "${data.ground_truth}",
                }
            }
        },
        # Log to Azure AI Foundry project
        azure_ai_project=azure_ai_project,
        output_path="./evaluation_results.json"
    )
    
    print("\nMetrics:")
    for metric, value in result["metrics"].items():
        print(f"  {metric}: {value:.2f}")
    
    if result.get("studio_url"):
        print(f"\nView in portal: {result['studio_url']}")
    
    print("\nResults saved to: evaluation_results.json")
    
    return result


if __name__ == "__main__":
    main()
