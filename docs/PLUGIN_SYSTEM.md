# Plugin System Guide

The LLM Judge Auditor toolkit includes a flexible plugin system that allows you to extend the toolkit with custom verifiers, judges, and aggregators without modifying the core code.

## Overview

The plugin system provides:

- **Custom Verifiers**: Implement your own fact-checking logic
- **Custom Judges**: Create domain-specific evaluation models
- **Custom Aggregators**: Define new strategies for combining scores
- **Plugin Discovery**: Automatically load plugins from a directory
- **Version Management**: Track plugin versions and compatibility
- **Isolated Environments**: Plugins run in isolation to prevent conflicts

## Quick Start

### 1. Create a Plugin

Create a Python file in the `plugins/` directory:

```python
# plugins/my_plugin.py

def register_plugin(registry):
    """Register custom components with the plugin registry."""
    
    # Define your custom verifier
    def load_my_verifier():
        class MyVerifier:
            def verify_statement(self, statement, context, passages=None):
                # Your verification logic
                return {
                    "label": "SUPPORTED",
                    "confidence": 0.9,
                    "evidence": [],
                    "reasoning": "Custom verification logic"
                }
            
            def batch_verify(self, statements, contexts, passages_list=None):
                return [self.verify_statement(s, c) for s, c in zip(statements, contexts)]
        
        return MyVerifier()
    
    # Register the verifier
    registry.register_verifier(
        name="my_custom_verifier",
        loader=load_my_verifier,
        version="1.0.0",
        description="My custom verifier implementation"
    )
```

### 2. Load and Use Plugins

```python
from llm_judge_auditor.components import PluginRegistry

# Create registry with auto-discovery
registry = PluginRegistry(plugins_dir="plugins")

# List available plugins
plugins = registry.list_plugins()
print(f"Available verifiers: {plugins['verifiers']}")

# Use your custom verifier
verifier = registry.get_verifier("my_custom_verifier")
verdict = verifier.verify_statement("The sky is blue", "The sky appears blue during the day")
print(f"Verdict: {verdict}")
```

## Plugin Types

### Verifier Plugins

Verifiers perform statement-level fact-checking. They must implement:

```python
class CustomVerifier:
    def verify_statement(
        self, 
        statement: str, 
        context: str, 
        passages: Optional[List[Any]] = None
    ) -> Verdict:
        """
        Verify a single statement.
        
        Args:
            statement: The statement to verify
            context: Source context to verify against
            passages: Optional retrieved passages for evidence
            
        Returns:
            Verdict with label, confidence, evidence, and reasoning
        """
        pass
    
    def batch_verify(
        self,
        statements: List[str],
        contexts: List[str],
        passages_list: Optional[List[List[Any]]] = None
    ) -> List[Verdict]:
        """
        Verify multiple statements in batch.
        
        Args:
            statements: List of statements to verify
            contexts: List of source contexts
            passages_list: Optional list of passage lists
            
        Returns:
            List of Verdict objects
        """
        pass
```

**Example Use Cases**:
- Domain-specific fact checkers (medical, legal, scientific)
- Rule-based verification systems
- External API integrations (fact-checking services)
- Custom NLI models

### Judge Plugins

Judges evaluate candidate outputs holistically. They must implement:

```python
class CustomJudge:
    def evaluate(
        self, 
        source_text: str, 
        candidate_output: str, 
        retrieved_context: str = ""
    ) -> JudgeResult:
        """
        Evaluate a candidate output.
        
        Args:
            source_text: Reference document or context
            candidate_output: Text to be evaluated
            retrieved_context: Optional retrieved passages
            
        Returns:
            JudgeResult with score, reasoning, flagged issues, and confidence
        """
        pass
```

**Example Use Cases**:
- Custom LLM judges with specific prompts
- Rule-based scoring systems
- Domain-specific evaluation criteria
- Multi-criteria evaluation models

### Aggregator Plugins

Aggregators combine scores from multiple judges. They must be callable functions:

```python
def custom_aggregator(scores: List[float]) -> float:
    """
    Aggregate multiple scores into a single consensus score.
    
    Args:
        scores: List of scores from different judges
        
    Returns:
        Aggregated consensus score
    """
    pass
```

**Example Use Cases**:
- Harmonic mean (penalizes low scores)
- Geometric mean (balanced approach)
- Weighted voting (trust certain judges more)
- Outlier-robust aggregation (median, trimmed mean)

## Plugin Discovery

### Automatic Discovery

Plugins are automatically discovered when you initialize the registry with a plugins directory:

```python
registry = PluginRegistry(plugins_dir="plugins")
```

### Manual Discovery

You can also discover plugins manually:

```python
registry = PluginRegistry()
discovered = registry.discover_plugins("plugins")
print(f"Discovered {discovered['verifiers']} verifiers")
```

### Discovery Rules

- Plugin files must be Python modules (`.py` files)
- Files starting with underscore (`_`) are ignored
- Each plugin must define a `register_plugin(registry)` function
- Plugins are loaded in alphabetical order
- Failed plugins are skipped with a warning

## Plugin Management

### Registering Plugins

```python
# Register a verifier
def load_verifier():
    return MyVerifier()

registry.register_verifier(
    name="my_verifier",
    loader=load_verifier,
    version="1.0.0",
    description="My custom verifier",
    author="Your Name",
    compatible_versions=["1.0.0", "1.1.0"]
)

# Register a judge
def load_judge():
    return MyJudge()

registry.register_judge(
    name="my_judge",
    loader=load_judge,
    version="1.0.0"
)

# Register an aggregator
def my_aggregator(scores):
    return sum(scores) / len(scores)

registry.register_aggregator(
    name="my_aggregator",
    aggregator=my_aggregator,
    version="1.0.0"
)
```

### Retrieving Plugins

```python
# Get a verifier
verifier = registry.get_verifier("my_verifier")

# Get a judge
judge = registry.get_judge("my_judge")

# Get an aggregator
aggregator = registry.get_aggregator("my_aggregator")
```

### Listing Plugins

```python
# List all plugins
plugins = registry.list_plugins()
print(plugins)
# {'verifiers': ['my_verifier'], 'judges': ['my_judge'], 'aggregators': ['my_aggregator']}

# Get plugin metadata
info = registry.get_plugin_info("my_verifier")
print(f"Version: {info.version}")
print(f"Description: {info.description}")
```

### Unregistering Plugins

```python
# Unregister a specific plugin
registry.unregister_verifier("my_verifier")

# Clear all plugins
registry.clear_all()
```

## Version Compatibility

Plugins can specify compatible toolkit versions:

```python
registry.register_verifier(
    name="my_verifier",
    loader=load_verifier,
    compatible_versions=["1.0.0", "1.1.0", "1.2.0"]
)

# Check compatibility
is_compatible = registry.check_compatibility("my_verifier", "1.0.0")
print(f"Compatible: {is_compatible}")
```

## Best Practices

### 1. Error Handling

Always handle errors gracefully in your plugins:

```python
def verify_statement(self, statement, context, passages=None):
    try:
        # Your verification logic
        result = self._verify(statement, context)
        return result
    except Exception as e:
        # Return a safe default on error
        return {
            "label": "NOT_ENOUGH_INFO",
            "confidence": 0.0,
            "evidence": [],
            "reasoning": f"Error during verification: {str(e)}"
        }
```

### 2. Documentation

Document your plugins thoroughly:

```python
def register_plugin(registry):
    """
    Register the MyPlugin components.
    
    This plugin provides:
    - CustomVerifier: A rule-based verifier for domain X
    - CustomAggregator: Harmonic mean aggregation
    
    Requirements:
    - Python 3.9+
    - numpy (for aggregation)
    
    Author: Your Name
    Version: 1.0.0
    """
    pass
```

### 3. Testing

Test your plugins before deployment:

```python
# test_my_plugin.py
import pytest
from plugins.my_plugin import MyVerifier

def test_verifier():
    verifier = MyVerifier()
    verdict = verifier.verify_statement("Test", "Test context")
    assert verdict["label"] in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
    assert 0.0 <= verdict["confidence"] <= 1.0
```

### 4. Dependencies

Document any additional dependencies:

```python
# plugins/my_plugin.py
"""
My Custom Plugin

Dependencies:
- numpy>=1.20.0
- scipy>=1.7.0

Install with:
    pip install numpy scipy
"""
```

### 5. Naming Conventions

Use descriptive, unique names:

- ✅ Good: `medical_fact_verifier`, `legal_document_judge`
- ❌ Bad: `verifier1`, `my_plugin`, `test`

## Advanced Examples

### Example 1: External API Integration

```python
import requests

def register_plugin(registry):
    def load_api_verifier():
        class APIVerifier:
            def __init__(self):
                self.api_url = "https://factcheck-api.example.com/verify"
            
            def verify_statement(self, statement, context, passages=None):
                try:
                    response = requests.post(
                        self.api_url,
                        json={"statement": statement, "context": context},
                        timeout=5
                    )
                    result = response.json()
                    return {
                        "label": result["verdict"],
                        "confidence": result["confidence"],
                        "evidence": result.get("evidence", []),
                        "reasoning": result.get("explanation", "")
                    }
                except Exception as e:
                    return {
                        "label": "NOT_ENOUGH_INFO",
                        "confidence": 0.0,
                        "evidence": [],
                        "reasoning": f"API error: {str(e)}"
                    }
            
            def batch_verify(self, statements, contexts, passages_list=None):
                return [self.verify_statement(s, c) for s, c in zip(statements, contexts)]
        
        return APIVerifier()
    
    registry.register_verifier("api_verifier", load_api_verifier)
```

### Example 2: Domain-Specific Judge

```python
def register_plugin(registry):
    def load_medical_judge():
        class MedicalJudge:
            def __init__(self):
                self.medical_terms = {"diagnosis", "treatment", "symptom", "medication"}
            
            def evaluate(self, source_text, candidate_output, retrieved_context=""):
                # Check for medical terminology
                candidate_lower = candidate_output.lower()
                term_count = sum(1 for term in self.medical_terms if term in candidate_lower)
                
                # Score based on medical accuracy indicators
                base_score = 50.0
                score = base_score + (term_count * 5)
                score = min(100.0, score)
                
                return {
                    "model_name": "medical_judge",
                    "score": score,
                    "reasoning": f"Found {term_count} medical terms",
                    "flagged_issues": [],
                    "confidence": 0.7
                }
        
        return MedicalJudge()
    
    registry.register_judge("medical_judge", load_medical_judge)
```

### Example 3: Robust Aggregation

```python
import statistics

def register_plugin(registry):
    def trimmed_mean(scores, trim_percent=0.2):
        """
        Calculate trimmed mean (removes outliers).
        
        Args:
            scores: List of scores
            trim_percent: Percentage to trim from each end
            
        Returns:
            Trimmed mean score
        """
        if not scores:
            return 0.0
        
        sorted_scores = sorted(scores)
        trim_count = int(len(scores) * trim_percent)
        
        if trim_count > 0:
            trimmed = sorted_scores[trim_count:-trim_count]
        else:
            trimmed = sorted_scores
        
        return statistics.mean(trimmed) if trimmed else 0.0
    
    registry.register_aggregator(
        "trimmed_mean",
        trimmed_mean,
        description="Trimmed mean aggregation (removes outliers)"
    )
```

## Troubleshooting

### Plugin Not Loading

**Problem**: Plugin doesn't appear in `list_plugins()`

**Solutions**:
1. Check that the file doesn't start with underscore
2. Verify `register_plugin(registry)` function exists
3. Check for syntax errors in the plugin file
4. Look for error messages in logs

### Import Errors

**Problem**: Plugin fails to load due to missing imports

**Solutions**:
1. Install required dependencies
2. Add try-except around imports with fallback
3. Document dependencies in plugin docstring

### Name Conflicts

**Problem**: `ValueError: already registered`

**Solutions**:
1. Use unique plugin names
2. Unregister existing plugin first: `registry.unregister_verifier("name")`
3. Clear all plugins: `registry.clear_all()`

## See Also

- [API Reference](API_REFERENCE.md#pluginregistry) - Complete API documentation
- [Examples](../examples/plugin_system_example.py) - Working plugin examples
- [Plugin Directory](../plugins/README.md) - Plugin directory structure
- [Contributing Guide](../CONTRIBUTING.md) - Guidelines for contributing plugins

## Support

For questions or issues with the plugin system:

1. Check the [examples](../examples/plugin_system_example.py)
2. Review the [API reference](API_REFERENCE.md)
3. Open an issue on GitHub
4. Consult the [contributing guide](../CONTRIBUTING.md)
