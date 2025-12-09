"""
Example plugin demonstrating custom components.

This plugin registers a simple length-based verifier and a geometric mean aggregator.
"""


def register_plugin(registry):
    """
    Register custom components with the plugin registry.

    Args:
        registry: PluginRegistry instance to register components with
    """

    # Register a custom verifier
    def load_length_verifier():
        """Load a simple length-based verifier."""

        class LengthBasedVerifier:
            """
            A simple verifier that uses text length as a heuristic.

            This is a toy example for demonstration purposes only.
            """

            def verify_statement(self, statement, context, passages=None):
                """
                Verify a statement based on length similarity to context.

                Args:
                    statement: Statement to verify
                    context: Source context
                    passages: Optional retrieved passages (unused)

                Returns:
                    Verdict dictionary
                """
                # Simple heuristic: if statement is much longer than context,
                # it's likely adding unsupported information
                stmt_len = len(statement)
                ctx_len = len(context)

                if ctx_len == 0:
                    label = "NOT_ENOUGH_INFO"
                    confidence = 0.5
                elif stmt_len <= ctx_len * 1.2:
                    label = "SUPPORTED"
                    confidence = 0.7
                else:
                    label = "REFUTED"
                    confidence = 0.6

                return {
                    "label": label,
                    "confidence": confidence,
                    "evidence": [context[:100] + "..."] if context else [],
                    "reasoning": f"Statement length: {stmt_len}, Context length: {ctx_len}",
                }

            def batch_verify(self, statements, contexts, passages_list=None):
                """Verify multiple statements."""
                return [
                    self.verify_statement(stmt, ctx)
                    for stmt, ctx in zip(statements, contexts)
                ]

        return LengthBasedVerifier()

    registry.register_verifier(
        name="length_based_verifier",
        loader=load_length_verifier,
        version="1.0.0",
        description="Simple verifier using text length heuristics",
        author="Example Plugin Author",
    )

    # Register a custom aggregator
    def geometric_mean(scores):
        """
        Calculate geometric mean of scores.

        The geometric mean is useful when you want to penalize low outliers.

        Args:
            scores: List of scores to aggregate

        Returns:
            Geometric mean of the scores
        """
        if not scores:
            return 0.0

        # Filter out zero scores to avoid issues
        non_zero_scores = [s for s in scores if s > 0]
        if not non_zero_scores:
            return 0.0

        product = 1.0
        for score in non_zero_scores:
            product *= score

        return product ** (1.0 / len(non_zero_scores))

    registry.register_aggregator(
        name="geometric_mean",
        aggregator=geometric_mean,
        version="1.0.0",
        description="Geometric mean aggregation strategy",
        author="Example Plugin Author",
    )

    print("Example plugin registered: length_based_verifier, geometric_mean")
