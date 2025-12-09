# Streaming Evaluator

The StreamingEvaluator enables evaluation of large documents that cannot fit into memory at once. It chunks documents into manageable segments and processes them incrementally through the multi-stage evaluation pipeline.

## Overview

The StreamingEvaluator wraps an EvaluationToolkit and provides streaming evaluation capabilities. It:

1. **Chunks large documents** into segments with configurable size and overlap
2. **Processes each chunk** through the full evaluation pipeline (retrieval → verifier → ensemble → aggregation)
3. **Aggregates results** across all chunks using weighted averaging
4. **Reduces memory footprint** by processing documents incrementally

## Key Features

- **Configurable chunking**: Set chunk size and overlap to balance context preservation and performance
- **Sentence-aware splitting**: Prefers to break at sentence boundaries when possible
- **Weighted aggregation**: Larger chunks have proportionally more influence on the final score
- **Deduplication**: Removes duplicate verdicts, passages, and issues across chunks
- **Full pipeline support**: Works with retrieval, verification, judge ensemble, and all other features

## Usage

### Basic Example

```python
from llm_judge_auditor import EvaluationToolkit, StreamingEvaluator
import io

# Initialize toolkit
toolkit = EvaluationToolkit.from_preset("balanced")

# Create streaming evaluator
streaming = StreamingEvaluator(
    toolkit,
    chunk_size=512,  # Characters per chunk
    overlap=50       # Overlap between chunks
)

# Create streams (e.g., from files)
with open("large_source.txt") as source_file, \
     open("large_candidate.txt") as candidate_file:
    
    result = streaming.evaluate_stream(
        source_stream=source_file,
        candidate_stream=candidate_file
    )

print(f"Final Score: {result.consensus_score:.2f}")
print(f"Chunks processed: {result.report.metadata['num_chunks']}")
```

### With String Streams

```python
import io

source_text = "Very long source document..." * 1000
candidate_text = "Very long candidate output..." * 1000

source_stream = io.StringIO(source_text)
candidate_stream = io.StringIO(candidate_text)

result = streaming.evaluate_stream(
    source_stream=source_stream,
    candidate_stream=candidate_stream,
    task="factual_accuracy",
    use_retrieval=True
)
```

## Configuration

### Chunk Size

The `chunk_size` parameter controls how many characters are in each chunk:

- **Small chunks (256-512)**: More fine-grained evaluation, slower processing
- **Medium chunks (512-1024)**: Good balance (default: 512)
- **Large chunks (1024-2048)**: Faster processing, less granular

```python
# Fine-grained evaluation
streaming = StreamingEvaluator(toolkit, chunk_size=256, overlap=25)

# Fast processing
streaming = StreamingEvaluator(toolkit, chunk_size=2048, overlap=200)
```

### Overlap

The `overlap` parameter controls how many characters overlap between consecutive chunks:

- **No overlap (0)**: Maximum speed, may lose context at boundaries
- **Small overlap (10-20% of chunk_size)**: Good balance (default: ~10%)
- **Large overlap (25-50% of chunk_size)**: Maximum context preservation, slower

```python
# No overlap - fastest
streaming = StreamingEvaluator(toolkit, chunk_size=1024, overlap=0)

# High overlap - best context preservation
streaming = StreamingEvaluator(toolkit, chunk_size=512, overlap=128)
```

### Recommendations

- **General use**: `chunk_size=512, overlap=50` (default)
- **Memory constrained**: `chunk_size=256, overlap=25`
- **Speed priority**: `chunk_size=1024, overlap=0`
- **Context priority**: `chunk_size=512, overlap=128`

## How It Works

### 1. Chunking

The evaluator splits the candidate text into overlapping chunks:

```
Original text: "AAAAAABBBBBBCCCCCCDDDDDD"
chunk_size=8, overlap=2

Chunk 1: "AAAAAAAA"
Chunk 2:       "AABBBBBB"
Chunk 3:             "BBCCCCCC"
Chunk 4:                   "CCDDDDDD"
```

Sentence boundaries are preferred when possible to maintain coherent chunks.

### 2. Evaluation

Each chunk is evaluated independently through the full pipeline:

- **Retrieval**: Claims extracted and passages retrieved for each chunk
- **Verification**: Specialized verifier checks statements in each chunk
- **Ensemble**: Judge models evaluate each chunk
- **Aggregation**: Results combined for each chunk

### 3. Aggregation

Results from all chunks are aggregated:

- **Scores**: Weighted average based on chunk size
- **Verdicts**: Deduplicated by reasoning
- **Passages**: Deduplicated by source
- **Issues**: Deduplicated by description
- **Judge results**: Averaged across chunks per judge

## Result Structure

The streaming evaluator returns a standard `EvaluationResult` with additional metadata:

```python
result = streaming.evaluate_stream(...)

# Standard fields
result.consensus_score          # Weighted average across chunks
result.verifier_verdicts        # Combined verdicts (deduplicated)
result.judge_results           # Aggregated judge results
result.flagged_issues          # Combined issues (deduplicated)

# Streaming-specific metadata
result.report.metadata['num_chunks']        # Number of chunks processed
result.report.metadata['total_characters']  # Total characters processed
result.report.metadata['chunk_size']        # Chunk size used
result.report.metadata['overlap']           # Overlap used
result.report.metadata['aggregation_strategy']  # "weighted_average_streaming"
```

## Performance Considerations

### Memory Usage

Streaming evaluation significantly reduces memory usage:

- **Non-streaming**: Entire document loaded into memory
- **Streaming**: Only one chunk in memory at a time

Example: For a 10MB document with 512-character chunks:
- Non-streaming: ~10MB memory
- Streaming: ~512 bytes per chunk + model memory

### Processing Time

Streaming adds overhead due to multiple evaluations:

- **Single chunk**: Same as non-streaming
- **Multiple chunks**: Linear increase with number of chunks

Example: Document with 10 chunks takes ~10x longer than single evaluation.

### Trade-offs

| Aspect | Non-Streaming | Streaming |
|--------|--------------|-----------|
| Memory | High | Low |
| Speed | Fast | Slower |
| Granularity | Document-level | Chunk-level |
| Context | Full document | Chunk + overlap |

## When to Use Streaming

**Use streaming when**:
- Documents are very large (>10,000 characters)
- Memory is constrained
- You need chunk-level granularity
- Processing documents from files/streams

**Use non-streaming when**:
- Documents are small (<5,000 characters)
- Speed is critical
- You need full document context
- Memory is not a concern

## Examples

See `examples/streaming_evaluator_example.py` for complete examples including:

- Basic streaming evaluation
- File-based streaming
- Custom chunk size configuration
- Result interpretation

## API Reference

### StreamingEvaluator

```python
class StreamingEvaluator:
    def __init__(
        self,
        toolkit: EvaluationToolkit,
        chunk_size: int = 512,
        overlap: int = 50
    )
```

**Parameters**:
- `toolkit`: EvaluationToolkit instance to use for evaluation
- `chunk_size`: Number of characters per chunk (default: 512)
- `overlap`: Number of characters to overlap between chunks (default: 50)

**Methods**:

#### evaluate_stream

```python
def evaluate_stream(
    self,
    source_stream: Iterator[str],
    candidate_stream: Iterator[str],
    task: str = "factual_accuracy",
    criteria: Optional[List[str]] = None,
    use_retrieval: Optional[bool] = None
) -> EvaluationResult
```

Evaluate large documents from streams.

**Parameters**:
- `source_stream`: Iterator yielding source text (e.g., open file)
- `candidate_stream`: Iterator yielding candidate text (e.g., open file)
- `task`: Evaluation task type (default: "factual_accuracy")
- `criteria`: Optional evaluation criteria (default: ["correctness"])
- `use_retrieval`: Override config retrieval setting

**Returns**: `EvaluationResult` with aggregated results from all chunks

**Raises**:
- `ValueError`: If streams are invalid or produce empty text
- `RuntimeError`: If evaluation fails

### PartialResult

```python
@dataclass
class PartialResult:
    chunk_index: int
    chunk_text: str
    consensus_score: float
    verifier_verdicts: List[Verdict]
    judge_results: List[JudgeResult]
    retrieved_passages: List[Passage]
    flagged_issues: List[Issue]
```

Result from evaluating a single chunk (internal use).

## Testing

Unit tests cover:
- Initialization and configuration
- Text chunking with overlap
- Stream reading
- Chunk evaluation
- Result aggregation
- Error handling

Run tests:
```bash
pytest tests/unit/test_streaming_evaluator.py -v
```

## Limitations

1. **Context boundaries**: Chunks may split related content
2. **Overhead**: Multiple evaluations increase processing time
3. **Aggregation**: Weighted averaging may not be optimal for all use cases
4. **Source text**: Full source text is used for all chunks (not chunked)

## Future Enhancements

Potential improvements:
- Parallel chunk processing
- Adaptive chunk sizing based on content
- Source text chunking and alignment
- Custom aggregation strategies
- Streaming output (yield results as chunks complete)

## Related Documentation

- [Usage Guide](USAGE_GUIDE.md)
- [API Reference](API_REFERENCE.md)
- [Examples](../examples/README.md)
