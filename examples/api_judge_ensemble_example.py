"""
Example usage of APIJudgeEnsemble for evaluating LLM outputs.

This example demonstrates:
1. Initializing the API judge ensemble with available API keys
2. Evaluating candidate outputs with multiple judges
3. Aggregating scores and detecting disagreements
4. Handling partial failures gracefully
"""

import logging
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_judge_auditor.components.api_judge_ensemble import APIJudgeEnsemble
from llm_judge_auditor.components.api_key_manager import APIKeyManager
from llm_judge_auditor.config import ToolkitConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    print("=" * 80)
    print("API Judge Ensemble Example")
    print("=" * 80)
    print()
    
    # Step 1: Initialize API Key Manager
    print("Step 1: Loading API keys...")
    api_key_manager = APIKeyManager()
    available_keys = api_key_manager.load_keys()
    
    print(f"Available API keys: {available_keys}")
    
    if not api_key_manager.has_any_keys():
        print("\n⚠️  No API keys found!")
        print(api_key_manager.get_setup_instructions())
        return
    
    print(f"✓ Found keys for: {', '.join(api_key_manager.get_available_services())}")
    print()
    
    # Step 2: Initialize API Judge Ensemble
    print("Step 2: Initializing API Judge Ensemble...")
    config = ToolkitConfig()
    ensemble = APIJudgeEnsemble(
        config=config,
        api_key_manager=api_key_manager,
        parallel_execution=True  # Use parallel execution for faster results
    )
    
    print(f"✓ Initialized {ensemble.get_judge_count()} judges:")
    for judge_name in ensemble.get_judge_names():
        print(f"  - {judge_name}")
    print()
    
    # Step 3: Prepare evaluation data
    print("Step 3: Preparing evaluation data...")
    
    source_text = """
    The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.
    It was constructed from 1887 to 1889 as the entrance arch for the 1889 World's Fair.
    The tower is 330 meters (1,083 feet) tall, about the same height as an 81-story building.
    It was the tallest man-made structure in the world until 1930.
    """
    
    # Example 1: Accurate output
    accurate_output = """
    The Eiffel Tower is an iron tower in Paris, France. It was built between 1887 and 1889
    for the World's Fair. The tower stands 330 meters tall, which is approximately the height
    of an 81-story building. It held the record as the world's tallest man-made structure
    until 1930.
    """
    
    # Example 2: Output with errors
    inaccurate_output = """
    The Eiffel Tower is a steel tower located in London, England. It was built in 1900
    to celebrate the new century. The tower is 500 meters tall, making it the tallest
    structure in Europe. It remains the tallest building in the world today.
    """
    
    print("✓ Prepared source text and candidate outputs")
    print()
    
    # Step 4: Evaluate accurate output
    print("Step 4: Evaluating accurate output...")
    print("-" * 80)
    
    try:
        verdicts_accurate = ensemble.evaluate(
            source_text=source_text,
            candidate_output=accurate_output,
            task="factual_accuracy"
        )
        
        print(f"\n✓ Received {len(verdicts_accurate)} verdicts:")
        for verdict in verdicts_accurate:
            print(f"\n  Judge: {verdict.judge_name}")
            print(f"  Score: {verdict.score:.1f}/100")
            print(f"  Confidence: {verdict.confidence:.2f}")
            print(f"  Issues found: {len(verdict.issues)}")
            print(f"  Reasoning: {verdict.reasoning[:100]}...")
        
        # Aggregate scores
        consensus, individual, disagreement = ensemble.aggregate_verdicts(verdicts_accurate)
        print(f"\n  Consensus Score: {consensus:.1f}/100")
        print(f"  Disagreement Level: {disagreement:.2f}")
        
        # Check for disagreements
        disagreement_analysis = ensemble.identify_disagreements(verdicts_accurate)
        if disagreement_analysis["has_disagreement"]:
            print(f"\n  ⚠️  Judges disagree!")
            print(f"  Score range: {disagreement_analysis['score_range']}")
            print(f"  Outliers: {disagreement_analysis['outliers']}")
        else:
            print(f"\n  ✓ Judges agree (low variance)")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
    
    print()
    
    # Step 5: Evaluate inaccurate output
    print("Step 5: Evaluating inaccurate output...")
    print("-" * 80)
    
    try:
        verdicts_inaccurate = ensemble.evaluate(
            source_text=source_text,
            candidate_output=inaccurate_output,
            task="factual_accuracy"
        )
        
        print(f"\n✓ Received {len(verdicts_inaccurate)} verdicts:")
        for verdict in verdicts_inaccurate:
            print(f"\n  Judge: {verdict.judge_name}")
            print(f"  Score: {verdict.score:.1f}/100")
            print(f"  Confidence: {verdict.confidence:.2f}")
            print(f"  Issues found: {len(verdict.issues)}")
            
            # Show issues
            if verdict.issues:
                print(f"  Issues:")
                for issue in verdict.issues[:3]:  # Show first 3 issues
                    print(f"    - [{issue.severity.value}] {issue.type.value}: {issue.description[:80]}")
        
        # Aggregate scores
        consensus, individual, disagreement = ensemble.aggregate_verdicts(verdicts_inaccurate)
        print(f"\n  Consensus Score: {consensus:.1f}/100")
        print(f"  Disagreement Level: {disagreement:.2f}")
        
        # Check for disagreements
        disagreement_analysis = ensemble.identify_disagreements(verdicts_inaccurate)
        if disagreement_analysis["has_disagreement"]:
            print(f"\n  ⚠️  Judges disagree!")
            print(f"  Score range: {disagreement_analysis['score_range']}")
        else:
            print(f"\n  ✓ Judges agree (low variance)")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
    
    print()
    print("=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
