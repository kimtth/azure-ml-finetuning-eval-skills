"""
Evaluate fine-tuned customer support model using Azure AI Evaluation SDK.

This script:
1. Loads the test dataset
2. Runs quality evaluators (relevance, coherence, fluency, F1)
3. Compares against baseline metrics
4. Logs results to Azure AI Foundry project
"""

import os
import sys
import json
from pathlib import Path

# Add examples directory to path for utils
sys.path.append(str(Path(__file__).parent.parent / "skills" / "azure-ml-model-evaluation" / "examples"))
from utils import get_azure_openai_model_config, get_azure_ai_project_config

from azure.ai.evaluation import (
    RelevanceEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    F1ScoreEvaluator,
    GroundednessEvaluator,
    evaluate
)


def main():
    """Run evaluation on customer support test dataset."""
    
    # Check required environment variables
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_RESOURCE_GROUP",
        "AZUREML_PROJECT_NAME"
    ]
    
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("\nRequired for evaluation:")
        print("  AZURE_OPENAI_ENDPOINT")
        print("  AZURE_OPENAI_DEPLOYMENT")
        print("  AZURE_OPENAI_API_VERSION")
        print("  AZURE_SUBSCRIPTION_ID")
        print("  AZURE_RESOURCE_GROUP")
        print("  AZUREML_PROJECT_NAME")
        return 1
    
    test_data_path = "e2e_test/customer_support_test.jsonl"
    
    if not Path(test_data_path).exists():
        print(f"Error: Test data not found: {test_data_path}")
        print("\nRun generate_customer_support_data.py first to create test data")
        return 1
    
    print("Loading Azure configurations...")
    model_config = get_azure_openai_model_config()
    azure_ai_project = get_azure_ai_project_config()
    
    print("Initializing evaluators...")
    
    # Quality evaluators (require Azure OpenAI)
    relevance_eval = RelevanceEvaluator(model_config)
    coherence_eval = CoherenceEvaluator(model_config)
    fluency_eval = FluencyEvaluator(model_config)
    groundedness_eval = GroundednessEvaluator(model_config)
    
    # NLP metrics (no model required)
    f1_eval = F1ScoreEvaluator()
    
    print(f"\nRunning evaluation on: {test_data_path}")
    print("This may take a few minutes...\n")
    
    # Run evaluation
    result = evaluate(
        data=test_data_path,
        evaluators={
            "relevance": relevance_eval,
            "coherence": coherence_eval,
            "fluency": fluency_eval,
            "groundedness": groundedness_eval,
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
        output_path="e2e_test/evaluation_results.json"
    )
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    
    print("\nMetrics Summary:")
    print("-" * 50)
    for metric, value in result["metrics"].items():
        if isinstance(value, (int, float)):
            print(f"  {metric:.<40} {value:.3f}")
        else:
            print(f"  {metric:.<40} {value}")
    
    # Save summary
    summary_path = "e2e_test/evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": result["metrics"],
            "num_samples": len(result.get("rows", [])),
            "test_data": test_data_path,
        }, f, indent=2)
    
    print("\n✓ Detailed results saved: e2e_test/evaluation_results.json")
    print(f"✓ Summary saved: {summary_path}")
    
    if result.get("studio_url"):
        print("\nView in Azure AI Foundry portal:")
        print(f"  {result['studio_url']}")
    
    # Evaluation interpretation
    print("\n" + "="*70)
    print("Interpretation Guide:")
    print("="*70)
    print("Relevance (1-5):    How well response addresses the query")
    print("Coherence (1-5):    Logical flow and clarity of response")
    print("Fluency (1-5):      Language quality and readability")
    print("Groundedness (1-5): Response supported by provided context")
    print("F1 Score (0-1):     Token overlap with ground truth")
    print("\nHigher scores indicate better performance.")
    print("Typical good scores: Relevance/Coherence/Fluency > 4.0, F1 > 0.6")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
