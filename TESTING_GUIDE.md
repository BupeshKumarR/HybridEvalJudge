# LLM Judge Auditor - Complete Testing Guide

This guide will walk you through testing the entire system from scratch.

## Prerequisites Check

Before starting, ensure you have:
- ✅ Docker and Docker Compose installed
- ✅ API keys for Groq and/or Google Gemini (for judge LLMs)
- ✅ Terminal/command line access

## Step 1: Set Up API Keys for Judge LLMs

First, you need API keys for the judge LLMs. You have two options:

### Option A: Use Groq (Free, Fast)
1. Go to https://console.groq.com
2. Sign up for a free account
3. Generate an API key
4. Copy the key (starts with `gsk_...`)

### Option B: Use Google Gemini (Free tier available)
1. Go to https://makersuite.google.com/app/apikey
2. Create an API key
3. Copy the key

### Option C: Use Both (Recommended for testing ensemble)

## Step 2: Configure Environment Variables

### For the Core Library (Root Directory)

Create a `.env` file in the root directory:

```bash
# From the root of the project
cat > .env << 'EOF'
# Groq API Key (if using Groq)
GROQ_API_KEY=your_groq_api_key_here

# Google Gemini API Key (if using Gemini)
GOOGLE_API_KEY=your_google_api_key_here
EOF
```

Or export them in your terminal:
```bash
export GROQ_API_KEY="your_groq_api_key_here"
export GOOGLE_API_KEY="your_google_api_key_here"
```

### For the Web Application

Create a `.env` file in the `web-app` directory:

```bash
cd web-app
cp .env.example .env
```

Edit `web-app/.env` and update these key values:
```bash
# Change the database password
POSTGRES_PASSWORD=your_secure_password_here

# Change the secret key (generate a random string)
SECRET_KEY=your_very_long_random_secret_key_here

# Keep other defaults for local testing
```

## Step 3: Test the Core Library First

Let's verify the core LLM Judge Auditor library works before testing the web app.

### Test 1: Simple CLI Test

```bash
# From the root directory
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the library
pip install -e .

# Run a simple evaluation
llm-judge-auditor evaluate \
  --source-text "The Eiffel Tower is located in Paris, France." \
  --claim "The Eiffel Tower is in London." \
  --judges groq
```

**Expected Result**: You should see output showing the evaluation results with hallucination detection.

### Test 2: Python API Test

Create a test script:

```bash
cat > test_core.py << 'EOF'
from llm_judge_auditor import EvaluationToolkit

# Initialize toolkit
toolkit = EvaluationToolkit()

# Test data
source_text = "The Eiffel Tower is a wrought-iron lattice tower in Paris, France."
claim = "The Eiffel Tower is located in London."

# Run evaluation
result = toolkit.evaluate(
    source_text=source_text,
    claim=claim,
    judges=["groq"]  # or ["gemini"] or ["groq", "gemini"]
)

# Print results
print(f"\n{'='*60}")
print(f"Source: {source_text}")
print(f"Claim: {claim}")
print(f"{'='*60}")
print(f"Consensus Score: {result.consensus_score:.2f}")
print(f"Hallucination Detected: {result.hallucination_detected}")
print(f"\nJudge Results:")
for judge_result in result.judge_results:
    print(f"  - {judge_result.judge_name}: {judge_result.score:.2f}")
    print(f"    Reasoning: {judge_result.reasoning[:100]}...")
EOF

python test_core.py
```

**Expected Result**: You should see detailed evaluation results with scores and reasoning.

## Step 4: Start the Web Application

Now let's test the full web application.

### Start All Services

```bash
cd web-app

# Build and start all services (backend, frontend, database, redis)
make up

# Or if make doesn't work:
docker-compose up -d
```

**Wait 30-60 seconds** for all services to start.

### Check Service Health

```bash
# Check if all containers are running
docker-compose ps

# Check backend health
curl http://localhost:8000/health

# View logs if there are issues
docker-compose logs backend
docker-compose logs frontend
```

**Expected Result**: All services should show as "Up" and health check should return `{"status":"healthy"}`.

## Step 5: Test the Web Application

### Access the Application

Open your browser and go to:
- **Frontend**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/docs

### Test 1: User Registration

1. Go to http://localhost:3000
2. Click "Sign Up" or "Register"
3. Create an account:
   - Email: `test@example.com`
   - Password: `TestPassword123!`
   - Username: `testuser`
4. Click "Register"

**Expected Result**: You should be registered and logged in.

### Test 2: Run an Evaluation

1. In the main interface, you should see a chat-like input area
2. Enter test data:
   - **Source Text**: "Python is a high-level programming language created by Guido van Rossum in 1991."
   - **Claim to Evaluate**: "Python was created by Guido van Rossum in 1995."
3. Click "Evaluate" or press Enter
4. Watch the real-time progress indicators

**Expected Result**: 
- You should see streaming progress updates
- Judge evaluations appearing one by one
- Final consensus score and hallucination detection
- Visualizations showing judge agreement

### Test 3: Try Different Scenarios

Test these scenarios to see different behaviors:

#### Scenario A: Clear Hallucination
- **Source**: "The capital of France is Paris."
- **Claim**: "The capital of France is London."
- **Expected**: High hallucination score, low consensus score

#### Scenario B: Accurate Claim
- **Source**: "Water boils at 100 degrees Celsius at sea level."
- **Claim**: "Water boils at 100°C at sea level."
- **Expected**: Low/no hallucination, high consensus score

#### Scenario C: Partial Hallucination
- **Source**: "The iPhone was released by Apple in 2007."
- **Claim**: "The iPhone was released by Apple in 2008."
- **Expected**: Moderate hallucination score (wrong year)

#### Scenario D: Unsupported Claim
- **Source**: "Dogs are mammals."
- **Claim**: "Dogs can fly."
- **Expected**: High hallucination score

### Test 4: Explore Visualizations

After running evaluations, check:

1. **Hallucination Thermometer**: Visual gauge of hallucination severity
2. **Judge Comparison Chart**: Bar chart comparing judge scores
3. **Inter-Judge Agreement Heatmap**: Shows how judges agree/disagree
4. **Statistics Panel**: Aggregate metrics

### Test 5: History and Search

1. Run multiple evaluations (3-5 different ones)
2. Click on "History" or the history sidebar
3. Try searching for specific evaluations
4. Filter by date or hallucination score
5. Click on a past evaluation to view details

### Test 6: Export Results

1. After running an evaluation, look for "Export" button
2. Try exporting as:
   - JSON
   - CSV
   - PDF (if available)
3. Verify the downloaded file contains correct data

### Test 7: Settings and Configuration

1. Go to Settings page
2. Try adjusting:
   - Judge selection (Groq, Gemini, or both)
   - Confidence thresholds
   - Visualization preferences
3. Save settings
4. Run a new evaluation to see changes take effect

## Step 6: Test API Directly (Optional)

You can also test the backend API directly:

### Using the Interactive API Docs

1. Go to http://localhost:8000/docs
2. Click "Authorize" and enter your credentials
3. Try these endpoints:
   - `POST /api/v1/evaluations/` - Create evaluation
   - `GET /api/v1/evaluations/` - List evaluations
   - `GET /api/v1/evaluations/{id}` - Get specific evaluation

### Using curl

```bash
# Register a user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "api-test@example.com",
    "password": "TestPass123!",
    "username": "apitest"
  }'

# Login to get token
TOKEN=$(curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "api-test@example.com",
    "password": "TestPass123!"
  }' | jq -r '.access_token')

# Create an evaluation
curl -X POST http://localhost:8000/api/v1/evaluations/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "source_text": "The Earth orbits the Sun.",
    "claim": "The Earth orbits the Moon.",
    "judges": ["groq"]
  }'
```

## Step 7: Test WebSocket Real-time Updates

The web app uses WebSockets for real-time evaluation updates. This should work automatically in the UI, but you can test it manually:

```bash
# Install wscat if you don't have it
npm install -g wscat

# Connect to WebSocket (replace TOKEN with your JWT)
wscat -c "ws://localhost:8000/ws?token=YOUR_JWT_TOKEN"

# You should see real-time updates when evaluations run
```

## Troubleshooting

### Issue: Services won't start

```bash
# Check logs
docker-compose logs

# Restart services
docker-compose down
docker-compose up -d

# Check for port conflicts
lsof -i :3000  # Frontend
lsof -i :8000  # Backend
lsof -i :5432  # PostgreSQL
```

### Issue: API keys not working

```bash
# Verify environment variables are set
echo $GROQ_API_KEY
echo $GOOGLE_API_KEY

# Check if backend can see them
docker-compose exec backend env | grep API_KEY
```

### Issue: Database connection errors

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d
```

### Issue: Frontend can't connect to backend

1. Check CORS settings in `web-app/.env`
2. Verify `REACT_APP_API_URL=http://localhost:8000`
3. Check browser console for errors (F12)

### Issue: Evaluations are slow

- This is normal! LLM API calls take time
- Groq is usually faster than Gemini
- Using multiple judges takes longer (they run sequentially)

## Performance Testing (Optional)

Want to see how the system handles load?

```bash
cd web-app

# Run performance tests
make test-performance

# Or manually with locust
docker-compose exec backend locust -f locustfile.py
# Then open http://localhost:8089
```

## What to Look For (Quality Assessment)

When testing, evaluate these aspects:

### Functionality
- ✅ Evaluations complete successfully
- ✅ Results are accurate and reasonable
- ✅ Real-time updates work smoothly
- ✅ History and search work correctly
- ✅ Export functions produce valid files

### User Experience
- ✅ Interface is responsive and intuitive
- ✅ Loading states are clear
- ✅ Error messages are helpful
- ✅ Visualizations are easy to understand
- ✅ Navigation is smooth

### Performance
- ✅ Page loads quickly
- ✅ Evaluations complete in reasonable time
- ✅ No lag when interacting with UI
- ✅ Multiple evaluations can run

### Reliability
- ✅ No crashes or errors
- ✅ Data persists across sessions
- ✅ Handles edge cases gracefully
- ✅ Recovers from API failures

## Next Steps

After testing:

1. **Document Issues**: Note any bugs or UX problems
2. **Test Edge Cases**: Try unusual inputs, very long text, special characters
3. **Test Different Browsers**: Chrome, Firefox, Safari
4. **Test Mobile**: Responsive design on phone/tablet
5. **Load Testing**: See how it handles multiple concurrent users

## Stopping the Application

When you're done testing:

```bash
# Stop all services
cd web-app
docker-compose down

# Or to also remove data
docker-compose down -v
```

## Getting Help

If you encounter issues:
1. Check the logs: `docker-compose logs`
2. Review `web-app/DEPLOYMENT.md` for troubleshooting
3. Check `docs/API_TROUBLESHOOTING.md`
4. Look at example files in `examples/` directory

---

**Happy Testing! 🚀**
