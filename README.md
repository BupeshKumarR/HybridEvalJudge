# LLM Judge Auditor

> **A comprehensive evaluation toolkit for auditing LLM outputs with specialized verifiers, judge ensembles, and a production-ready web application.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🌟 Overview

LLM Judge Auditor is a hybrid evaluation system that combines specialized fact-checking models with judge LLM ensembles to provide comprehensive, transparent evaluation of AI-generated text. It includes both a Python toolkit and a full-stack web application for interactive evaluation.

### Key Features

#### Core Evaluation Engine
- 🎯 **Multi-Judge Ensemble**: Combine multiple LLM judges (GPT-4, Claude, Gemini, Groq) for robust evaluation
- 🔍 **Specialized Verifiers**: Fact-checking models for claim verification
- 📚 **Retrieval-Augmented**: Optional integration with external knowledge bases
- 📊 **Statistical Metrics**: Confidence intervals, inter-judge agreement, hallucination scores
- ⚡ **Performance Optimized**: Parallel evaluation, caching, and 8-bit quantization support

#### Hallucination Quantification
- 📈 **MiHR/MaHR**: Micro and Macro hallucination rates for claim-level and response-level analysis
- 🎯 **FactScore**: Factual precision metric (verified_claims / total_claims)
- 🤝 **Consensus F1**: Cross-model agreement with precision, recall, and F1 scores
- 📊 **Fleiss' Kappa**: Inter-judge agreement statistic with interpretation
- 🔮 **Uncertainty Quantification**: Shannon entropy with epistemic/aleatoric decomposition
- ⚠️ **Risk Assessment**: Automatic high-risk flagging based on configurable thresholds

#### Web Application
- 💬 **Chat Interface**: Interactive evaluation with real-time streaming
- 📈 **Rich Visualizations**: Judge comparisons, confidence gauges, hallucination metrics
- 📜 **Session History**: Persistent storage and search of past evaluations
- 📤 **Export Options**: JSON, CSV, and PDF export with visualizations
- 🔐 **Authentication**: Secure user management with JWT tokens
- 🎨 **Responsive Design**: Works on desktop, tablet, and mobile

#### Developer Tools
- 🔌 **Plugin System**: Extensible architecture for custom components
- 🧪 **Property-Based Testing**: Rigorous correctness validation
- 📝 **Comprehensive API**: RESTful API with OpenAPI/Swagger documentation
- 🔄 **WebSocket Support**: Real-time evaluation progress streaming
- 🐳 **Docker Ready**: Full containerization with Docker Compose

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Web Application](#-web-application)
- [Python Toolkit Usage](#-python-toolkit-usage)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Architecture](#-architecture)
- [Development](#-development)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Quick Start

### Option 1: Web Application (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-judge-auditor.git
cd llm-judge-auditor

# Start with Docker Compose
cd web-app
docker-compose up -d

# Access the application
open http://localhost:3000
```

### Option 2: Python Toolkit

```bash
# Install the package
pip install -e .

# Set up API keys (free options available)
export GROQ_API_KEY="your-groq-key"
export GEMINI_API_KEY="your-gemini-key"

# Run a quick evaluation
python examples/simple_evaluation.py
```

---

## 💻 Installation

### Prerequisites

- **Python 3.9+** (for toolkit)
- **Node.js 16+** (for web app frontend)
- **PostgreSQL 15+** (for web app backend)
- **Docker & Docker Compose** (optional, for containerized deployment)

### Python Toolkit Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/llm-judge-auditor.git
cd llm-judge-auditor

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package
pip install -e .

# 5. Set up API keys (choose free or paid options)
./setup_env.sh
```

### Web Application Installation

See [web-app/README.md](web-app/README.md) for detailed setup instructions.

**Quick Docker Setup:**
```bash
cd web-app
cp .env.example .env
# Edit .env with your configuration
docker-compose up -d
```

---

## 🌐 Web Application

The web application provides an interactive interface for evaluating LLM outputs with real-time streaming, rich visualizations, and comprehensive reporting.

### Features

- **Interactive Chat Interface**: Submit evaluations and see results in real-time
- **Real-Time Streaming**: Watch as each judge completes evaluation
- **Rich Visualizations**:
  - Judge comparison charts with confidence intervals
  - Hallucination thermometer with severity breakdown
  - Inter-judge agreement heatmaps
  - Statistical metrics dashboard
- **Session Management**: Save, search, and restore evaluation history
- **Export Options**: Download results as JSON, CSV, or PDF
- **User Authentication**: Secure multi-user support
- **Responsive Design**: Works on all devices

### Screenshots

```
┌─────────────────────────────────────────────────────────┐
│  Chat Interface          │  Evaluation Results          │
│  ┌────────────────────┐  │  ┌────────────────────────┐ │
│  │ Source Text        │  │  │ Judge Scores           │ │
│  │ Candidate Output   │  │  │ ├─ GPT-4: 87.5 ⭐⭐⭐  │ │
│  │ [Submit]           │  │  │ ├─ Claude: 85.2 ⭐⭐⭐ │ │
│  └────────────────────┘  │  │ └─ Gemini: 89.1 ⭐⭐⭐ │ │
│                          │  │                        │ │
│  History Sidebar         │  │ Hallucination: 12.3%   │ │
│  ┌────────────────────┐  │  │ Confidence: 92%        │ │
│  │ • Session 1        │  │  │ Agreement: 0.78        │ │
│  │ • Session 2        │  │  └────────────────────────┘ │
│  │ • Session 3        │  │                            │ │
│  └────────────────────┘  │                            │ │
└─────────────────────────────────────────────────────────┘
```

### Quick Start

```bash
cd web-app

# Development mode
make dev

# Production mode
make prod

# Access the application
open http://localhost:3000
```

See [web-app/README.md](web-app/README.md) for complete documentation.

---

## 🐍 Python Toolkit Usage

### Basic Evaluation

```python
from llm_judge_auditor import EvaluationToolkit

# Initialize toolkit (uses API judges automatically if keys are set)
toolkit = EvaluationToolkit.from_preset("fast")

# Evaluate an output
source = "The capital of France is Paris."
candidate = "Paris is the capital and largest city of France."

result = toolkit.evaluate(
    source_text=source,
    candidate_output=candidate
)

# Access results
print(f"Consensus Score: {result.consensus_score}")
print(f"Hallucination Score: {result.hallucination_score}")
print(f"Confidence: {result.confidence_interval}")

# View individual judge results
for judge_result in result.judge_results:
    print(f"{judge_result.judge_name}: {judge_result.score}")
```

### Advanced Configuration

```python
from llm_judge_auditor import EvaluationToolkit, ToolkitConfig, APIConfig

# Configure with specific judges
config = ToolkitConfig(
    api_config=APIConfig(
        groq_api_key="your-key",
        gemini_api_key="your-key",
        groq_model="llama-3.1-70b-versatile",
        gemini_model="gemini-1.5-flash"
    ),
    enable_retrieval=True,
    aggregation_strategy="weighted_average"
)

toolkit = EvaluationToolkit(config)

# Batch evaluation
results = toolkit.evaluate_batch([
    {"source": source1, "candidate": candidate1},
    {"source": source2, "candidate": candidate2}
])
```

### Streaming Evaluation

```python
from llm_judge_auditor.components import StreamingEvaluator

evaluator = StreamingEvaluator(toolkit)

# Stream results as they arrive
for event in evaluator.evaluate_streaming(source, candidate):
    if event.type == "judge_result":
        print(f"Judge {event.data.judge_name} scored: {event.data.score}")
    elif event.type == "complete":
        print(f"Final score: {event.data.consensus_score}")
```

### Command-Line Interface

```bash
# Evaluate from command line
llm-judge-auditor evaluate \
  --source "The capital of France is Paris." \
  --candidate "Paris is the capital of France." \
  --output results.json

# Batch evaluation from file
llm-judge-auditor batch \
  --input evaluations.jsonl \
  --output results/

# Generate report
llm-judge-auditor report \
  --input results.json \
  --format pdf \
  --output report.pdf
```

---

## 📚 API Documentation

### REST API

The web application provides a comprehensive REST API for programmatic access.

**Base URL**: `http://localhost:8000/api/v1`

#### Key Endpoints

```bash
# Authentication
POST /api/v1/auth/register
POST /api/v1/auth/login
GET  /api/v1/auth/me

# Evaluations
POST /api/v1/evaluations
GET  /api/v1/evaluations
GET  /api/v1/evaluations/{id}
GET  /api/v1/evaluations/{id}/export?format=pdf

# Preferences
GET  /api/v1/preferences
PUT  /api/v1/preferences
```

#### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/evaluations \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "source_text": "The capital of France is Paris.",
    "candidate_output": "Paris is the capital of France.",
    "config": {
      "judge_models": ["gpt-4", "claude-3"],
      "enable_retrieval": true,
      "aggregation_strategy": "mean"
    }
  }'
```

### WebSocket API

Real-time evaluation streaming via WebSocket.

```javascript
import { io } from 'socket.io-client';

const socket = io('ws://localhost:8000/ws', {
  auth: { token: 'your-jwt-token' }
});

socket.emit('start_evaluation', {
  session_id: 'uuid',
  source_text: 'source',
  candidate_output: 'output',
  config: { judge_models: ['gpt-4'] }
});

socket.on('evaluation_progress', (data) => {
  console.log(`Progress: ${data.progress}%`);
});

socket.on('evaluation_complete', (data) => {
  console.log(`Score: ${data.consensus_score}`);
});
```

**Full API Documentation**:
- [REST API Documentation](web-app/docs/API_DOCUMENTATION.md)
- [WebSocket Events](web-app/docs/WEBSOCKET_EVENTS.md)
- [OpenAPI Specification](web-app/docs/openapi.yaml)

---

## ⚙️ Configuration

### API Keys Setup

The toolkit supports multiple LLM providers. You can use free or paid options.

#### Free Options (Recommended for Getting Started)

```bash
# Groq (FREE - Llama 3.1 70B)
export GROQ_API_KEY="your-groq-key"
# Get key: https://console.groq.com/keys

# Google Gemini (FREE - Gemini 1.5 Flash)
export GEMINI_API_KEY="your-gemini-key"
# Get key: https://aistudio.google.com/app/apikey
```

#### Paid Options (Higher Quality)

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Configuration Files

```yaml
# config/default_config.yaml
judges:
  - name: "gpt-4"
    provider: "openai"
    weight: 1.0
  - name: "claude-3"
    provider: "anthropic"
    weight: 1.0

verifier:
  model: "specialized-verifier"
  confidence_threshold: 0.7

retrieval:
  enabled: true
  top_k: 5

aggregation:
  strategy: "weighted_average"
```

See [docs/guides/API_KEY_SETUP.md](docs/guides/API_KEY_SETUP.md) for detailed setup instructions.

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Web Application                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Frontend   │  │   Backend    │  │  PostgreSQL  │ │
│  │   (React)    │◄─┤   (FastAPI)  │◄─┤   Database   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│              Python Evaluation Toolkit                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Judge      │  │  Verifier    │  │  Retrieval   │ │
│  │   Ensemble   │  │  Component   │  │  Component   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Aggregation  │  │   Metrics    │  │   Report     │ │
│  │   Engine     │  │  Calculator  │  │  Generator   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Core Components

- **Judge Ensemble**: Manages multiple LLM judges for evaluation
- **Verifier Component**: Specialized fact-checking models
- **Retrieval Component**: Fetches relevant context from knowledge bases
- **Aggregation Engine**: Combines judge scores with statistical methods
- **Hallucination Metrics**: Research-backed quantification (MiHR, MaHR, FactScore, Fleiss' Kappa)
- **Uncertainty Quantification**: Shannon entropy with epistemic/aleatoric decomposition
- **Report Generator**: Creates comprehensive evaluation reports

### Technology Stack

**Backend**:
- FastAPI (Python web framework)
- SQLAlchemy (ORM)
- PostgreSQL (Database)
- Socket.IO (WebSocket)
- JWT (Authentication)

**Frontend**:
- React 18 + TypeScript
- TailwindCSS (Styling)
- Recharts + D3.js (Visualizations)
- Socket.IO Client (Real-time)
- React Query (Data fetching)

**Infrastructure**:
- Docker + Docker Compose
- Nginx (Reverse proxy)
- Redis (Caching)

---

## 🛠️ Development

### Setting Up Development Environment

```bash
# Clone and setup
git clone https://github.com/yourusername/llm-judge-auditor.git
cd llm-judge-auditor

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Web Application Development

```bash
cd web-app

# Backend development
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend development
cd frontend
npm install
npm start

# Run tests
cd backend && pytest
cd frontend && npm test
```

### Project Structure

```
llm-judge-auditor/
├── src/llm_judge_auditor/      # Python toolkit
│   ├── components/             # Core components
│   ├── utils/                  # Utilities
│   └── evaluation_toolkit.py   # Main API
├── web-app/                    # Web application
│   ├── backend/                # FastAPI backend
│   ├── frontend/               # React frontend
│   └── docs/                   # API documentation
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── property/               # Property-based tests
├── docs/                       # Documentation
│   ├── guides/                 # User guides
│   ├── security/               # Security docs
│   └── development/            # Dev docs
├── examples/                   # Usage examples
├── config/                     # Configuration files
└── scripts/                    # Utility scripts
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/              # Unit tests
pytest tests/integration/       # Integration tests
pytest tests/property/          # Property-based tests

# Run with coverage
pytest --cov=src --cov-report=html

# Run web app tests
cd web-app/backend && pytest
cd web-app/frontend && npm test
```

### Property-Based Testing

We use Hypothesis for property-based testing to ensure correctness:

```python
from hypothesis import given, strategies as st

@given(st.text(), st.text())
def test_evaluation_consistency(source, candidate):
    """Evaluation should be deterministic for same inputs."""
    result1 = toolkit.evaluate(source, candidate)
    result2 = toolkit.evaluate(source, candidate)
    assert result1.consensus_score == result2.consensus_score
```

---

## 📖 Documentation

### User Guides
- [Quick Start Guide](docs/guides/QUICKSTART.md)
- [API Key Setup](docs/guides/API_KEY_SETUP.md)
- [Free Models Guide](docs/guides/FREE_MODELS_INFO.md)

### API Documentation
- [REST API Reference](web-app/docs/API_DOCUMENTATION.md)
- [WebSocket Events](web-app/docs/WEBSOCKET_EVENTS.md)
- [OpenAPI Specification](web-app/docs/openapi.yaml)

### Component Documentation
- [Usage Guide](docs/USAGE_GUIDE.md)
- [CLI Usage](docs/CLI_USAGE.md)
- [Hallucination Metrics](docs/HALLUCINATION_METRICS.md)
- [Error Handling](docs/ERROR_HANDLING.md)
- [Performance Optimization](docs/PERFORMANCE_OPTIMIZATION.md)
- [Plugin System](docs/PLUGIN_SYSTEM.md)

### Development Documentation
- [Implementation Review](docs/development/IMPLEMENTATION_REVIEW.md)
- [Performance Optimization](docs/development/PERFORMANCE_OPTIMIZATION_SUMMARY.md)
- [Error Handling](docs/development/COMPREHENSIVE_ERROR_HANDLING_SUMMARY.md)

### Security
- [Git Security Audit](docs/security/GIT_SECURITY_AUDIT.md)
- [Safe to Push Guide](docs/security/SAFE_TO_PUSH.md)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Write tests for new features
- Update documentation
- Use type hints
- Run linters before committing

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/), [React](https://react.dev/), and [PostgreSQL](https://www.postgresql.org/)
- Visualization powered by [Recharts](https://recharts.org/) and [D3.js](https://d3js.org/)
- Testing with [pytest](https://pytest.org/) and [Hypothesis](https://hypothesis.readthedocs.io/)
- Free LLM APIs provided by [Groq](https://groq.com/) and [Google AI Studio](https://aistudio.google.com/)

---

## 📞 Support

- **Documentation**: [Full documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/llm-judge-auditor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/llm-judge-auditor/discussions)

---

## 🗺️ Roadmap

- [ ] Support for additional LLM providers (Cohere, Mistral)
- [ ] Advanced visualization options
- [ ] Batch evaluation API endpoint
- [ ] Custom judge model training
- [ ] Multi-language support
- [ ] Cloud deployment templates (AWS, GCP, Azure)
- [ ] Mobile app

---

**Made with ❤️ by the LLM Judge Auditor Team**
