# Plugins Directory

This directory is for custom plugin modules that extend the LLM Judge Auditor toolkit.

## Plugin Structure

Each plugin should be a Python module (`.py` file) that defines a `register_plugin(registry)` function. This function will be called during plugin discovery and should register one or more custom components.

## Example Plugin

Here's a simple example of a plugin file (`plugins/example_plugin.py`):

```python
def register_plugin(registry):
    """
    Register custom components with the plugin registry.
    
    Args:
        registry: PluginRegistry instance to register components with
    """
    
    # Example: Register a custom verifier
    def load_my_verifier():
        class MyVerifier:
            def verify_statement(self, statement, context, passages=None):
                # Your verification logic here
                return {"label": "SUPPORTED", "confidence": 0.9}
            
            def batch_verify(self, statements, contexts, passages_list=None):
                # Your batch verification logic here
                return [self.verify_statement(s, c) for s, c in zip(statements, contexts)]
        
        return MyVerifier()
    
    registry.register_verifier(
        name="my_custom_verifier",
        loader=load_my_verifier,
        version="1.0.0",
        description="My custom verifier implementation",
        author="Your Name"
    )
    
    # Example: Register a custom aggregator
    def geometric_mean(scores):
        if not scores:
            return 0.0
        product = 1.0
        for score in scores:
            product *= score
        return product ** (1.0 / len(scores))
    
    registry.register_aggregator(
        name="geometric_mean",
        aggregator=geometric_mean,
        version="1.0.0",
        description="Geometric mean aggregation"
    )
```

## Plugin Interfaces

### Verifier Interface

Custom verifiers should implement:
- `verify_statement(statement: str, context: str, passages: Optional[List] = None) -> Verdict`
- `batch_verify(statements: List[str], contexts: List[str], passages_list: Optional[List[List]] = None) -> List[Verdict]`

### Judge Interface

Custom judges should implement:
- `evaluate(source_text: str, candidate_output: str, retrieved_context: str = "") -> JudgeResult`

### Aggregator Interface

Custom aggregators should be callable functions:
- `aggregator(scores: List[float]) -> float`

## Loading Plugins

Plugins are automatically discovered when you initialize the PluginRegistry with a plugins directory:

```python
from llm_judge_auditor.components import PluginRegistry

# Auto-discover plugins from the plugins/ directory
registry = PluginRegistry(plugins_dir="plugins")

# Or manually discover later
registry = PluginRegistry()
registry.discover_plugins("plugins")
```

## Best Practices

1. **Naming**: Use descriptive names for your plugins that indicate their purpose
2. **Versioning**: Follow semantic versioning (MAJOR.MINOR.PATCH)
3. **Documentation**: Include docstrings explaining what your plugin does
4. **Error Handling**: Handle errors gracefully in your plugin code
5. **Testing**: Test your plugins thoroughly before deployment
6. **Dependencies**: Document any additional dependencies your plugin requires

## Notes

- Plugin files starting with underscore (`_`) are ignored during discovery
- Plugins are loaded in alphabetical order
- If a plugin fails to load, it will be skipped and an error will be logged
- Each plugin should have a unique name to avoid conflicts
