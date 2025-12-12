# LLM Judge Auditor

> **A comprehensive evaluation toolkit for auditing LLM outputs with specialized verifiers, judge ensembles, and a production-ready web application.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🌟 Overview

LLM Judge Auditor is a hybrid evaluation system that combines specialized fact-checking models with judge LLM ensembles to provide comprehensive, transparent evaluation of AI-generated text. It includes both a Python toolkit and a full-stack web application for interactive evaluation.

### Key Features

#### Core Evaluation Engine
- 🎯 **Multi-Judge Ensemble**: Combine multiple LLM judges (GPT-4, Claude, Gemini, Groq, Ollama) for robust evaluation
- 🔍 **Specialized Verifiers**: Fact-checking models for claim verification
- 📚 **Retrieval-Augmented**: Optional integration with external knowledge bases
- 📊 **Statistical Metrics**: Confidence intervals, inter-judge agreement, hallucination scores
- ⚡ **Performance Optimized**: Parallel evaluation, caching, and 8-bit quantization support
- 🦙 **Local LLM Support**: Ollama integration for privacy-focused, offline evaluation

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

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Web Application                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  Frontend   │    │   Backend   │    │  Database   │             │
│  │  (React +   │◄──►│  (FastAPI)  │◄──►│ PostgreSQL  │             │
│  │ TypeScript) │    │             │    │  or SQLite  │             │
│  └─────────────┘    └──────┬──────┘    └─────────────┘             │
│         │                  │                                        │
│         │           ┌──────▼──────┐    ┌─────────────┐             │
│         │           │   Redis     │    │   Ollama    │             │
│         └──────────►│   Cache     │    │  (Local LLM)│             │
│                     └─────────────┘    └─────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Python Evaluation Toolkit                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │   Judge     │  │  Verifier   │  │  Retrieval  │  │   Claim   │ │
│  │  Ensemble   │  │  Component  │  │  Component  │  │   Router  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │ Aggregation │  │ Hallucin.   │  │   Report    │  │  Streaming│ │
│  │   Engine    │  │  Metrics    │  │  Generator  │  │  Evaluator│ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18, TypeScript, TailwindCSS, Recharts |
| Backend | FastAPI, SQLAlchemy, Socket.IO, JWT Auth |
| Database | PostgreSQL (production), SQLite (development) |
| Cache | Redis |
| Local LLM | Ollama |
| API Judges | Groq (free), Gemini (free), OpenAI, Anthropic |
| Infrastructure | Docker, Docker Compose, Nginx |

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

The fastest way to get started with full PostgreSQL, Redis, and all services:

```bash
# Clone the repository
git clone https://github.com/BupeshKumarR/HybridEvalJudge.git
cd llm-judge-auditor/web-app

# Start all services
docker-compose up -d

# Access the application
open http://localhost:3000
```

This starts:
- **Frontend** on http://localhost:3000
- **Backend API** on http://localhost:8000
- **PostgreSQL** on port 5432
- **Redis** on port 6379

### Option 2: Local Development (No Docker)

For development without Docker, using SQLite:

```bash
# Clone and setup
git clone https://github.com/BupeshKumarR/HybridEvalJudge.git
cd llm-judge-auditor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend with SQLite fallback
cd web-app/backend
USE_SQLITE=true uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, start frontend
cd web-app/frontend
npm install
npm start
```

### Option 3: Python Toolkit Only

For programmatic evaluation without the web interface:

```bash
# Install the package
pip install -e .

# Set up API keys (free options available)
export GROQ_API_KEY="your-groq-key"      # Free: https://console.groq.com/keys
export GEMINI_API_KEY="your-gemini-key"  # Free: https://aistudio.google.com/app/apikey

# Run evaluation
python -c "
from llm_judge_auditor import EvaluationToolkit
toolkit = EvaluationToolkit.from_preset('fast')
result = toolkit.evaluate(
    source_text='The capital of France is Paris.',
    candidate_output='Paris is the capital of France.'
)
print(f'Score: {result.consensus_score}')
"
```

---

## 📋 Reproducibility Guide

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.9+ | Backend & toolkit |
| Node.js | 16+ | Frontend |
| Docker | 20+ | Containerization |
| Ollama | Latest | Local LLM (optional) |

### Step-by-Step Setup

#### 1. Clone Repository
```bash
git clone https://github.com/BupeshKumarR/HybridEvalJudge.git
cd llm-judge-auditor
```

#### 2. Environment Configuration

Create `.env` file in `web-app/`:
```bash
# Database (Docker uses PostgreSQL, local uses SQLite fallback)
DATABASE_URL=postgresql://llm_judge_user:changeme@postgres:5432/llm_judge_auditor

# Security
SECRET_KEY=your-secret-key-change-in-production

# API Keys (optional - for cloud LLM judges)
GROQ_API_KEY=your-groq-key
GEMINI_API_KEY=your-gemini-key

# Ollama (for local LLM)
OLLAMA_HOST=http://host.docker.internal:11434
```

#### 3. Start Services

**With Docker:**
```bash
cd web-app
docker-compose up -d
docker-compose logs -f  # Watch logs
```

**Without Docker:**
```bash
# Terminal 1: Backend
cd web-app/backend
source ../../.venv/bin/activate
USE_SQLITE=true uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
cd web-app/frontend
npm install && npm start
```

#### 4. Verify Installation
```bash
# Check backend health
curl http://localhost:8000/health

# Check frontend
open http://localhost:3000
```

#### 5. Create Account & Login
1. Navigate to http://localhost:3000
2. Click "Register" to create an account
3. Login with your credentials

---

## Ollama Integration (Local LLM)

For privacy-focused evaluation using local models:

### Setup Ollama
```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
ollama serve

# Pull a model
ollama pull llama3.2
ollama pull mistral
```

### Configure in Web App
1. Go to Settings → Judge Configuration
2. Enable "Use Ollama"
3. Select your preferred model
4. Evaluations now run locally without API calls

---

## Evaluation Metrics

### Hallucination Metrics
| Metric | Description | Range |
|--------|-------------|-------|
| MiHR | Micro Hallucination Rate (claim-level) | 0-1 |
| MaHR | Macro Hallucination Rate (response-level) | 0-1 |
| FactScore | Verified claims / Total claims | 0-1 |
| Fleiss' Kappa | Inter-judge agreement | -1 to 1 |

### Uncertainty Quantification
- **Shannon Entropy**: Overall uncertainty measure
- **Epistemic Uncertainty**: Model knowledge gaps
- **Aleatoric Uncertainty**: Inherent data ambiguity

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Property-based tests
pytest tests/property/

# Web app backend tests
cd web-app/backend && pytest

# Web app frontend tests
cd web-app/frontend && npm test
```

---

##  Project Structure

```
llm-judge-auditor/
├── src/llm_judge_auditor/      # Python evaluation toolkit
│   ├── components/             # Core components (judges, verifiers, etc.)
│   ├── utils/                  # Utilities (error handling, profiling)
│   └── evaluation_toolkit.py   # Main API
├── web-app/                    # Full-stack web application
│   ├── backend/                # FastAPI backend
│   │   ├── app/               # Application code
│   │   │   ├── routers/       # API endpoints
│   │   │   ├── services/      # Business logic
│   │   │   └── models.py      # Database models
│   │   └── tests/             # Backend tests
│   ├── frontend/              # React frontend
│   │   ├── src/
│   │   │   ├── components/    # UI components
│   │   │   ├── pages/         # Page components
│   │   │   └── hooks/         # Custom hooks
│   │   └── public/
│   └── docker-compose.yml     # Container orchestration
├── tests/                      # Toolkit tests
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── property/              # Property-based tests
├── docs/                       # Documentation
├── examples/                   # Usage examples
└── config/                     # Configuration presets
```

---

##  Configuration

### Presets

```python
from llm_judge_auditor import EvaluationToolkit

# Fast evaluation (2 judges, no retrieval)
toolkit = EvaluationToolkit.from_preset("fast")

# Balanced (3 judges, basic retrieval)
toolkit = EvaluationToolkit.from_preset("balanced")

# Comprehensive (5 judges, full retrieval)
toolkit = EvaluationToolkit.from_preset("comprehensive")
```

### Custom Configuration

```python
from llm_judge_auditor import EvaluationToolkit, ToolkitConfig

config = ToolkitConfig(
    judge_models=["groq-llama3", "gemini-flash"],
    enable_retrieval=True,
    aggregation_strategy="weighted_average",
    confidence_threshold=0.7
)
toolkit = EvaluationToolkit(config)
```

---

##  Documentation

| Document | Description |
|----------|-------------|
| [API Documentation](web-app/docs/API_DOCUMENTATION.md) | REST API reference |
| [WebSocket Events](web-app/docs/WEBSOCKET_EVENTS.md) | Real-time event reference |
| [Hallucination Metrics](docs/HALLUCINATION_METRICS.md) | Metric definitions |
| [CLI Usage](docs/CLI_USAGE.md) | Command-line interface |
| [Testing Guide](TESTING_GUIDE.md) | Testing instructions |
| [Development Guide](web-app/DEVELOPMENT.md) | Development setup |
| [Deployment Guide](web-app/DEPLOYMENT.md) | Production deployment |

---

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Run tests (`pytest`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

##  License

MIT License - see [LICENSE](LICENSE) for details.

---

##  Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [React](https://react.dev/) - Frontend framework
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Groq](https://groq.com/) - Free LLM API
- [Google AI Studio](https://aistudio.google.com/) - Free Gemini API

---

**Made with ❤️ for reliable AI evaluation**
