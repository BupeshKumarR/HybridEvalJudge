# LLM Judge Auditor - Demo

**Professional LLM evaluation using FREE local models - zero cost, no API keys!**

This demo shows how to evaluate and compare multiple AI models running locally on your laptop.

---

## ⚡ Quick Start (5 Minutes)

**Just 3 commands:**

```bash
# 1. Install Ollama from https://ollama.ai (2 min)

# 2. Setup models (3 min)
python demo/setup.py

# 3. Run demo (30 sec)
python demo/demo.py
```

**That's it!** Professional LLM evaluation running locally.

---

## 📋 Detailed Setup

### Step 1: Install Ollama (2 minutes)

Visit [https://ollama.ai](https://ollama.ai) and download for your OS.

### Step 2: Setup Models (3 minutes)

```bash
python demo/setup.py
```

This auto-detects your system and installs the best models for your RAM.

### Step 3: Run Demo (30 seconds)

```bash
python demo/demo.py
```

**Done!** You now have professional LLM evaluation running locally.

---

## 📊 What It Does

The demo:
- ✅ Tests multiple local models (Phi-3, Llama 3.2, Qwen2.5)
- ✅ Generates real AI responses (not simulated)
- ✅ Evaluates quality and accuracy
- ✅ Compares models objectively
- ✅ Ranks by performance
- ✅ Saves results to `demo/results.json`

### Example Output

```
🏆 Model Rankings:
   🥇 1. phi3: 87.3/100 (Confidence: 0.91)
   🥈 2. llama3.2:3b: 82.1/100 (Confidence: 0.88)
   🥉 3. qwen2.5:3b: 79.5/100 (Confidence: 0.85)

🎯 RECOMMENDATION:
   Best Model: phi3
   Verdict: APPROVED ✅
```

---

## 💻 System Requirements

| RAM | Recommended Models | Performance |
|-----|-------------------|-------------|
| 8GB | phi3 + llama3.2:1b | Good |
| 16GB | phi3 + llama3.2:3b + qwen2.5:3b | Excellent |
| 32GB+ | All above + mistral | Production-grade |

The setup script automatically recommends the best models for your system.

---

## 🎯 Perfect For

### Portfolio Projects
- Demonstrate multi-agent AI systems
- Show model comparison capabilities
- Prove evaluation expertise

### Resume Highlights
- "Built multi-agent LLM evaluation system"
- "Implemented AI quality assessment pipeline"
- "Developed zero-cost model comparison framework"

### Learning & Development
- Understand LLM evaluation
- Compare model capabilities
- Experiment safely offline

### Privacy-Sensitive Work
- Healthcare applications
- Legal document review
- Financial analysis
- 100% local processing

---

## 🔧 Customization

### Test Your Own Questions

Edit `demo/demo.py` around line 150:

```python
question = "Your custom question here"
reference = "Your trusted reference information"
```

### Add More Models

```bash
ollama pull mistral      # 7B model (if 16GB+ RAM)
ollama pull codellama    # For code evaluation
ollama pull gemma:2b     # Very small model
```

They'll automatically appear in the demo!

---

## 📚 Documentation

- **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)** ⭐ - Detailed explanation with examples
- **[QUICK_START_FREE.md](QUICK_START_FREE.md)** - 5-minute quick start
- **[FREE_SETUP_GUIDE.md](FREE_SETUP_GUIDE.md)** - Complete setup guide
- **[FREE_DEMO_SUMMARY.md](FREE_DEMO_SUMMARY.md)** - Full feature overview

---

## 🐛 Troubleshooting

### "Ollama not found"
```bash
# Check installation
which ollama

# Reinstall from https://ollama.ai
```

### "Model not found"
```bash
# List installed models
ollama list

# Pull missing model
ollama pull phi3
```

### "Out of memory"
```bash
# Use smaller model
ollama pull llama3.2:1b

# Or close other applications
```

---

## 💡 Why This Demo?

### Zero Cost
- ✅ No API keys
- ✅ No subscriptions
- ✅ Free forever
- ✅ Runs offline

### Professional Quality
- ✅ Multi-model comparison
- ✅ Objective evaluation
- ✅ Confidence scoring
- ✅ Comprehensive reporting

### Privacy First
- ✅ 100% local processing
- ✅ No data sent to cloud
- ✅ Full control
- ✅ GDPR/HIPAA compatible

---

## 🚀 Next Steps

1. **Today**: Run the demo with different questions
2. **This Week**: Customize for your domain
3. **This Month**: Add to your portfolio/GitHub
4. **Share**: Show it off on LinkedIn!

---

## 📞 Support

- **Main Docs**: [../README.md](../README.md)
- **Examples**: [../examples/](../examples/)
- **Issues**: GitHub Issues

---

**Total setup time: 5 minutes**  
**Total cost: $0**  
**Professional value: Priceless** 💎
