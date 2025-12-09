"""
Example demonstrating the Plugin System for custom components.

This example shows how to:
1. Register custom verifiers, judges, and aggregators
2. Use the plugin discovery system
3. Load and use custom plugins
"""

import tempfile
from pathlib import Path
from typing import Any, List, Optional

from llm_judge_auditor.components.plugin_registry import PluginRegistry
from llm_judge_auditor.models import Issue, IssueType, IssueSeverity, Verdict, VerdictLabel


# Example 1: Custom Verifier Plugin
class SimpleRuleBasedVerifier:
    """
    A simple rule-based verifier that checks for exact matches.
    
    This is a minimal example showing how to implement a custom verifier
    that follows the VerifierProtocol interface.
    """

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode

    def verify_statement(
        self, statement: str, context: str, passages: Optional[List[Any]] = None
    ) -> Verdict:
        """Verify a statement using simple rule-based matching."""
        statement_lower = statement.lower()
        context_lower = context.lower()

        # Simple rule: check if statement words appear in context
        statement_words = set(statement_lower.split())
        context_words = set(context_lower.split())

        overlap = statement_words.intersection(context_words)
        overlap_ratio = len(overlap) / len(statement_words) if statement_words else 0

        if overlap_ratio > 0.8:
            label = VerdictLabel.SUPPORTED
            confidence = overlap_ratio
        elif overlap_ratio < 0.3:
            label = VerdictLabel.REFUTED
            confidence = 1.0 - overlap_ratio
        else:
            label = VerdictLabel.NOT_ENOUGH_INFO
            confidence = 0.5

        return Verdict(
            label=label,
            confidence=confidence,
            evidence=[context[:100] + "..."] if context else [],
            reasoning=f"Word overlap ratio: {overlap_ratio:.2f}",
        )

    def batch_verify(
        self,
        statements: List[str],
        contexts: List[str],
        passages_list: Optional[List[List[Any]]] = None,
    ) -> List[Verdict]:
        """Verify multiple statements."""
        return [
            self.verify_statement(stmt, ctx)
            for stmt, ctx in zip(statements, contexts)
        ]


# Example 2: Custom Judge Plugin
class KeywordBasedJudge:
    """
    A simple keyword-based judge for demonstration.
    
    This judge assigns scores based on the presence of positive/negative keywords.
    """

    def __init__(self):
        self.positive_keywords = {"accurate", "correct", "true", "verified", "confirmed"}
        self.negative_keywords = {"false", "incorrect", "wrong", "inaccurate", "misleading"}

    def evaluate(
        self, source_text: str, candidate_output: str, retrieved_context: str = ""
    ) -> dict:
        """Evaluate candidate output based on keyword presence."""
        candidate_lower = candidate_output.lower()

        positive_count = sum(
            1 for keyword in self.positive_keywords if keyword in candidate_lower
        )
        negative_count = sum(
            1 for keyword in self.negative_keywords if keyword in candidate_lower
        )

        # Calculate score based on keyword balance
        base_score = 50.0
        score = base_score + (positive_count * 10) - (negative_count * 15)
        score = max(0.0, min(100.0, score))  # Clamp to [0, 100]

        reasoning = (
            f"Keyword analysis: {positive_count} positive keywords, "
            f"{negative_count} negative keywords. Base score: {base_score}, "
            f"Final score: {score}"
        )

        flagged_issues = []
        if negative_count > 0:
            flagged_issues.append(
                Issue(
                    type=IssueType.FACTUAL_ERROR,
                    severity=IssueSeverity.MEDIUM,
                    description=f"Found {negative_count} negative keywords",
                    evidence=[],
                )
            )

        return {
            "model_name": "keyword_judge",
            "score": score,
            "reasoning": reasoning,
            "flagged_issues": flagged_issues,
            "confidence": 0.6,
        }


# Example 3: Custom Aggregator
def harmonic_mean_aggregator(scores: List[float]) -> float:
    """
    Aggregate scores using harmonic mean.
    
    Harmonic mean is useful when you want to penalize low scores more heavily.
    """
    if not scores:
        return 0.0

    # Filter out zero scores to avoid division by zero
    non_zero_scores = [s for s in scores if s > 0]
    if not non_zero_scores:
        return 0.0

    return len(non_zero_scores) / sum(1.0 / s for s in non_zero_scores)


def main():
    """Demonstrate the plugin system."""
    print("=" * 70)
    print("Plugin System Example")
    print("=" * 70)

    # Create a plugin registry
    registry = PluginRegistry()

    # Example 1: Register a custom verifier
    print("\n1. Registering custom verifier...")

    def load_rule_based_verifier():
        return SimpleRuleBasedVerifier(strict_mode=False)

    registry.register_verifier(
        name="rule_based_verifier",
        loader=load_rule_based_verifier,
        version="1.0.0",
        description="Simple rule-based verifier using word overlap",
        author="Example Author",
    )

    print("   ✓ Registered 'rule_based_verifier'")

    # Example 2: Register a custom judge
    print("\n2. Registering custom judge...")

    def load_keyword_judge():
        return KeywordBasedJudge()

    registry.register_judge(
        name="keyword_judge",
        loader=load_keyword_judge,
        version="1.0.0",
        description="Keyword-based judge for simple evaluation",
        author="Example Author",
    )

    print("   ✓ Registered 'keyword_judge'")

    # Example 3: Register a custom aggregator
    print("\n3. Registering custom aggregator...")

    registry.register_aggregator(
        name="harmonic_mean",
        aggregator=harmonic_mean_aggregator,
        version="1.0.0",
        description="Harmonic mean aggregation strategy",
        author="Example Author",
    )

    print("   ✓ Registered 'harmonic_mean' aggregator")

    # List all registered plugins
    print("\n4. Listing all registered plugins...")
    plugins = registry.list_plugins()
    print(f"   Verifiers: {plugins['verifiers']}")
    print(f"   Judges: {plugins['judges']}")
    print(f"   Aggregators: {plugins['aggregators']}")

    # Get plugin info
    print("\n5. Getting plugin metadata...")
    verifier_info = registry.get_plugin_info("rule_based_verifier")
    if verifier_info:
        print(f"   Name: {verifier_info.name}")
        print(f"   Version: {verifier_info.version}")
        print(f"   Type: {verifier_info.plugin_type}")
        print(f"   Description: {verifier_info.description}")

    # Use the custom verifier
    print("\n6. Using custom verifier...")
    verifier = registry.get_verifier("rule_based_verifier")

    statement = "The Eiffel Tower is located in Paris"
    context = "Paris is the capital of France and home to the Eiffel Tower"

    verdict = verifier.verify_statement(statement, context)
    print(f"   Statement: {statement}")
    print(f"   Verdict: {verdict.label.value}")
    print(f"   Confidence: {verdict.confidence:.2f}")
    print(f"   Reasoning: {verdict.reasoning}")

    # Use the custom judge
    print("\n7. Using custom judge...")
    judge = registry.get_judge("keyword_judge")

    source = "The sky is blue during the day."
    candidate = "The sky is accurate and true, it is blue."

    result = judge.evaluate(source, candidate)
    print(f"   Candidate: {candidate}")
    print(f"   Score: {result['score']:.1f}")
    print(f"   Reasoning: {result['reasoning']}")

    # Use the custom aggregator
    print("\n8. Using custom aggregator...")
    aggregator = registry.get_aggregator("harmonic_mean")

    scores = [80.0, 85.0, 90.0]
    consensus = aggregator(scores)
    print(f"   Individual scores: {scores}")
    print(f"   Harmonic mean: {consensus:.2f}")

    # Demonstrate plugin discovery
    print("\n9. Demonstrating plugin discovery...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a plugin file
        plugin_file = Path(tmpdir) / "example_plugin.py"
        plugin_code = '''
def register_plugin(registry):
    """Register a simple max aggregator."""
    def max_aggregator(scores):
        return max(scores) if scores else 0.0
    
    registry.register_aggregator(
        "max_score",
        max_aggregator,
        version="1.0.0",
        description="Returns the maximum score"
    )
'''
        plugin_file.write_text(plugin_code)

        # Discover plugins
        discovered = registry.discover_plugins(tmpdir)
        print(f"   Discovered {discovered['aggregators']} aggregator(s)")

        # Use the discovered plugin
        if "max_score" in registry.list_plugins()["aggregators"]:
            max_agg = registry.get_aggregator("max_score")
            result = max_agg([70.0, 85.0, 92.0])
            print(f"   Max score aggregator result: {result:.1f}")

    # Check version compatibility
    print("\n10. Checking version compatibility...")
    is_compatible = registry.check_compatibility("rule_based_verifier", "1.0.0")
    print(f"   'rule_based_verifier' compatible with v1.0.0: {is_compatible}")

    # Clean up
    print("\n11. Cleaning up...")
    registry.clear_all()
    plugins = registry.list_plugins()
    total = sum(len(v) for v in plugins.values())
    print(f"   All plugins cleared. Total plugins: {total}")

    print("\n" + "=" * 70)
    print("Plugin System Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
