"""
Example usage of the PromptManager component.

This script demonstrates how to use the PromptManager to generate
evaluation prompts for different tasks.
"""

from llm_judge_auditor.components.prompt_manager import PromptManager


def main():
    """Demonstrate PromptManager usage."""
    print("=" * 80)
    print("PromptManager Example")
    print("=" * 80)
    
    # Initialize the PromptManager
    pm = PromptManager()
    
    # List available tasks
    print("\n1. Available Tasks:")
    print("-" * 80)
    tasks = pm.list_tasks()
    for task in tasks:
        print(f"  - {task}")
    
    # Example 1: Factual Accuracy Evaluation
    print("\n2. Factual Accuracy Prompt:")
    print("-" * 80)
    factual_prompt = pm.get_prompt(
        "factual_accuracy",
        source_text="The Eiffel Tower is located in Paris, France. It was completed in 1889.",
        candidate_output="The Eiffel Tower is in Paris and was built in 1889.",
        retrieved_context="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris."
    )
    print(factual_prompt[:500] + "...\n")  # Print first 500 chars
    
    # Example 2: Pairwise Ranking
    print("\n3. Pairwise Ranking Prompt:")
    print("-" * 80)
    pairwise_prompt = pm.get_prompt(
        "pairwise_ranking",
        source_text="Water boils at 100 degrees Celsius at sea level.",
        candidate_a="Water boils at 100°C at standard atmospheric pressure.",
        candidate_b="Water boils at 212°F which is about 100°C."
    )
    print(pairwise_prompt[:500] + "...\n")  # Print first 500 chars
    
    # Example 3: Bias Detection
    print("\n4. Bias Detection Prompt:")
    print("-" * 80)
    bias_prompt = pm.get_prompt(
        "bias_detection",
        candidate_output="The engineer solved the problem quickly. She was very skilled."
    )
    print(bias_prompt[:500] + "...\n")  # Print first 500 chars
    
    # Example 4: Customizing a Prompt
    print("\n5. Custom Prompt Template:")
    print("-" * 80)
    custom_template = """Evaluate this text for accuracy:

Source: {source_text}
Candidate: {candidate_output}

Provide a score from 0-100."""
    
    pm.customize_prompt("factual_accuracy", custom_template)
    custom_prompt = pm.get_prompt(
        "factual_accuracy",
        source_text="The sky is blue.",
        candidate_output="The sky is green."
    )
    print(custom_prompt)
    
    # Example 5: Getting Raw Template
    print("\n6. Raw Template (Factual Accuracy):")
    print("-" * 80)
    # Reset to default by creating new instance
    pm = PromptManager()
    template = pm.get_template("factual_accuracy")
    print(f"Template length: {len(template)} characters")
    print(f"Variables: {[var for var in ['source_text', 'candidate_output', 'retrieved_context'] if '{' + var + '}' in template]}")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
