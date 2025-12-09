# Property-Based Tests

This directory contains property-based tests using Hypothesis to verify correctness properties across many randomly generated inputs.

## Test Coverage

### Core Properties (`test_core_properties.py`)

Tests universal properties that should hold for all valid inputs:

1. **Property 3: Score bounds validity**
   - Verifies all judge scores are between 0-100
   - Validates: Requirements 2.4
   - Runs: 100 iterations

2. **Property 24: Ensemble aggregation correctness**
   - Verifies aggregation strategies (mean, median) work correctly
   - Verifies low-confidence flagging when variance > 20
   - Validates: Requirements 11.1, 11.2, 11.3, 11.4
   - Runs: 100 iterations

3. **Property 10: Batch aggregation correctness**
   - Verifies batch statistics (mean, median) are calculated correctly
   - Validates: Requirements 5.3
   - Runs: 100 iterations

4. **Property 8: Pairwise ranking symmetry**
   - Verifies pairwise comparisons are symmetric
   - If A > B, then B < A; if tie, both orderings show tie
   - Validates: Requirements 10.2, 14.3
   - Runs: 100 iterations

## Running Property Tests

### Run all property tests:
```bash
python -m pytest tests/property/ -v
```

### Run specific property test:
```bash
python -m pytest tests/property/test_core_properties.py::TestCoreProperties::test_score_bounds_validity -v
```

### Run with Hypothesis statistics:
```bash
python -m pytest tests/property/ -v --hypothesis-show-statistics
```

### Increase iterations for more thorough testing:
```python
@settings(max_examples=1000)  # Run 1000 iterations instead of 100
```

## Property-Based Testing Approach

Property-based tests use Hypothesis to:
1. Generate random valid inputs
2. Test that properties hold for all generated inputs
3. Automatically shrink failing examples to minimal cases
4. Provide better coverage than example-based tests

## Custom Strategies

The tests use custom Hypothesis strategies to generate realistic data:
- `score_strategy()`: Generates valid scores (0-100)
- `judge_result_strategy()`: Generates valid JudgeResult objects

## Requirements

- pytest
- hypothesis >= 6.0

## Notes

- Each property test runs 100 iterations by default
- Tests are tagged with property numbers matching the design document
- All tests include requirement validation comments
- Hypothesis automatically finds edge cases and minimal failing examples
