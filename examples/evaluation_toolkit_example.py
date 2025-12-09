"""
Example usage of the EvaluationToolkit orchestrator.

This script demonstrates how to use the main EvaluationToolkit class
to evaluate LLM outputs for factual accuracy and hallucinations.

NOTE: This example works best with API judges (no model downloads needed).
Set environment variables:
  export GROQ_API_KEY="your-groq-key"
  export GEMINI_API_KEY="your-gemini-key"

Get free keys at:
  - Groq: https://console.groq.com/keys
  - Gemini: https://aistudio.google.com/app/apikey
"""

import logging
import os
from pathlib import Path

from llm_judge_auditor import EvaluationToolkit, ToolkitConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def example_basic_evaluation():
    """
    Example 1: Basic evaluation using a preset configuration.
    
    This demonstrates the simplest way to use the toolkit with
    a pre-configured preset.
    """
    print("\n" + "=" * 80)
    print("Example 1: Basic Evaluation with Preset")
    print("=" * 80 + "\n")
    
    # Check for API keys
    has_api_keys = bool(os.getenv("GROQ_API_KEY")) or bool(os.getenv("GEMINI_API_KEY"))
    if has_api_keys:
        print("✓ Using API judges (fast, no downloads needed)")
    else:
        print("⚠ No API keys - will download models (slower first run)")
        print("  Get free API keys: https://console.groq.com/keys")
    
    # Create toolkit from preset
    # With API keys: Uses free API judges (fast)
    # Without API keys: Downloads models (slower first time)
    try:
        toolkit = EvaluationToolkit.from_preset("fast")
        
        # Define source and candidate texts
        source_text = """
        The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
        It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
        Constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair, it was initially
        criticized by some of France's leading artists and intellectuals for its design.
        """
        
        candidate_output = """
        The Eiffel Tower was built in Paris between 1887 and 1889. It was designed by
        Gustave Eiffel's company for the 1889 World's Fair.
        """
        
        # Evaluate
        result = toolkit.evaluate(
            source_text=source_text,
            candidate_output=candidate_output,
        )
        
        # Display results
        print(f"Consensus Score: {result.consensus_score:.2f}/100")
        print(f"Confidence: {result.report.confidence:.2f}")
        print(f"Disagreement Level: {result.report.disagreement_level:.2f}")
        print(f"\nNumber of Verifier Verdicts: {len(result.verifier_verdicts)}")
        print(f"Number of Judge Results: {len(result.judge_results)}")
        print(f"Flagged Issues: {len(result.flagged_issues)}")
        
        # Show individual judge scores
        print("\nIndividual Judge Scores:")
        for model_name, score in result.report.individual_scores.items():
            print(f"  {model_name}: {score:.2f}")
        
        # Show verifier verdicts
        print("\nVerifier Verdicts:")
        for i, verdict in enumerate(result.verifier_verdicts, 1):
            print(f"  {i}. {verdict.label.value} (confidence: {verdict.confidence:.2f})")
            print(f"     Reasoning: {verdict.reasoning[:100]}...")
        
        # Show flagged issues
        if result.flagged_issues:
            print("\nFlagged Issues:")
            for issue in result.flagged_issues:
                print(f"  - [{issue.severity.value}] {issue.type.value}: {issue.description}")
        else:
            print("\nNo issues flagged!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\nNote: This example requires actual models to be available.")
        print(f"Error: {e}")


def example_custom_configuration():
    """
    Example 2: Evaluation with custom configuration.
    
    This demonstrates how to create a custom configuration
    with specific settings.
    """
    print("\n" + "=" * 80)
    print("Example 2: Custom Configuration")
    print("=" * 80 + "\n")
    
    # Create custom configuration
    config = ToolkitConfig(
        verifier_model="MiniCheck/flan-t5-base-finetuned",
        judge_models=["microsoft/Phi-3-mini-4k-instruct"],
        quantize=True,
        enable_retrieval=False,
        aggregation_strategy="mean",
        batch_size=1,
        max_length=512,
    )
    
    print("Custom Configuration:")
    print(f"  Verifier: {config.verifier_model}")
    print(f"  Judges: {config.judge_models}")
    print(f"  Quantization: {config.quantize}")
    print(f"  Retrieval: {config.enable_retrieval}")
    print(f"  Aggregation: {config.aggregation_strategy}")
    
    try:
        toolkit = EvaluationToolkit(config)
        
        # Get toolkit statistics
        stats = toolkit.get_stats()
        print("\nToolkit Statistics:")
        print(f"  Number of judges: {stats['num_judges']}")
        print(f"  Verifier loaded: {stats['verifier_loaded']}")
        print(f"  Retrieval enabled: {stats['config']['retrieval_enabled']}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\nNote: This example requires actual models to be available.")
        print(f"Error: {e}")


def example_with_hallucination():
    """
    Example 3: Detecting hallucinations.
    
    This demonstrates how the toolkit detects factual errors
    and hallucinations in candidate outputs.
    """
    print("\n" + "=" * 80)
    print("Example 3: Hallucination Detection")
    print("=" * 80 + "\n")
    
    try:
        toolkit = EvaluationToolkit.from_preset("fast")
        
        # Source text with facts
        source_text = """
        The Great Wall of China is a series of fortifications built across the historical
        northern borders of ancient Chinese states. Construction began in the 7th century BC.
        The wall stretches over 13,000 miles.
        """
        
        # Candidate with hallucinations
        candidate_output = """
        The Great Wall of China was built in the 15th century AD by the Ming Dynasty.
        It is approximately 5,000 miles long and was built to keep out Mongolian invaders.
        The wall is visible from space with the naked eye.
        """
        
        # Evaluate
        result = toolkit.evaluate(
            source_text=source_text,
            candidate_output=candidate_output,
        )
        
        print(f"Consensus Score: {result.consensus_score:.2f}/100")
        print(f"(Lower score indicates more hallucinations)")
        
        # Show hallucination categories
        print("\nHallucination Categories:")
        for category, count in result.report.hallucination_categories.items():
            if count > 0:
                print(f"  {category}: {count}")
        
        # Show flagged issues
        print("\nFlagged Issues:")
        for issue in result.flagged_issues:
            print(f"  - [{issue.severity.value}] {issue.description[:100]}...")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\nNote: This example requires actual models to be available.")
        print(f"Error: {e}")


def example_export_results():
    """
    Example 4: Exporting evaluation results.
    
    This demonstrates how to export results to JSON format
    for further analysis or storage.
    """
    print("\n" + "=" * 80)
    print("Example 4: Exporting Results")
    print("=" * 80 + "\n")
    
    try:
        toolkit = EvaluationToolkit.from_preset("fast")
        
        source_text = "Python is a high-level programming language."
        candidate_output = "Python is a programming language created by Guido van Rossum."
        
        result = toolkit.evaluate(
            source_text=source_text,
            candidate_output=candidate_output,
        )
        
        # Export to JSON
        json_output = result.to_json(indent=2)
        
        print("Result exported to JSON:")
        print(json_output[:500] + "...")
        
        # Save to file
        output_path = Path("evaluation_result.json")
        output_path.write_text(json_output)
        print(f"\nFull result saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\nNote: This example requires actual models to be available.")
        print(f"Error: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("EvaluationToolkit Examples")
    print("=" * 80)
    
    print("\nNOTE: These examples require actual models to be downloaded and loaded.")
    print("This may take significant time and resources on first run.")
    print("For testing purposes, consider using mock models or smaller models.")
    
    # Run examples
    # Uncomment the examples you want to run:
    
    # example_basic_evaluation()
    # example_custom_configuration()
    # example_with_hallucination()
    # example_export_results()
    
    print("\n" + "=" * 80)
    print("Examples Complete")
    print("=" * 80 + "\n")
    
    print("To run these examples, uncomment the function calls in main().")
    print("Make sure you have sufficient disk space and memory for model loading.")


if __name__ == "__main__":
    main()
