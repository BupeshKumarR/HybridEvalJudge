# LLM Evaluation Report

## Metadata

- **Timestamp**: 2024-01-01T12:00:00
- **Task**: factual_accuracy
- **Criteria**: correctness
- **Retrieval Enabled**: True
- **Verifier Model**: minicheck-flan-t5-large
- **Judge Models**: llama-3-8b, mistral-7b
- **Aggregation Strategy**: mean

## Evaluation Scores

### Consensus Score: **77.50**/100

- **Confidence**: 0.85
- **Disagreement Level**: 2.50

### Individual Judge Scores

- **llama-3-8b**: 75.00/100
- **mistral-7b**: 80.00/100

## Chain-of-Thought Reasoning

### llama-3-8b

The candidate output contains mostly accurate information about Paris and France. However, there is one factual error regarding the population figure.

### mistral-7b

The output is generally accurate and well-structured. Minor issues with specificity in some claims, but overall factually sound.

## Specialized Verifier Verdicts

### Verdict 1

- **Label**: SUPPORTED
- **Confidence**: 0.90
- **Reasoning**: The claim is directly supported by the source text.
- **Evidence**:
  - The source text confirms this claim.

### Verdict 2

- **Label**: REFUTED
- **Confidence**: 0.85
- **Reasoning**: The claim contradicts information in the source text.
- **Evidence**:
  - The source text contradicts this claim.

## Retrieval Provenance

Retrieved 2 passages:

### Passage 1

- **Source**: Wikipedia:Paris
- **Relevance Score**: 0.9500
- **Text**: Paris is the capital and most populous city of France.

### Passage 2

- **Source**: Wikipedia:France
- **Relevance Score**: 0.8800
- **Text**: France is a country primarily located in Western Europe.

## Flagged Issues

### Issue 1

- **Type**: hallucination
- **Severity**: high
- **Description**: Refuted claim: The claim contradicts information in the source text.
- **Evidence**:
  - The source text contradicts this claim.

### Issue 2

- **Type**: numerical_error
- **Severity**: medium
- **Description**: Population figure is incorrect
- **Evidence**:
  - Source states 2.1 million, candidate states 3 million

## Hallucination Categories

- **Factual Error**: 1
- **Numerical Error**: 1
