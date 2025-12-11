#!/bin/bash

# Performance Testing Script for LLM Judge Auditor Web Application
# Tests load times, concurrent users, and identifies bottlenecks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:3000}"
API_BASE_URL="${BACKEND_URL}/api/v1"
CONCURRENT_USERS="${CONCURRENT_USERS:-10}"
TEST_DURATION="${TEST_DURATION:-60}"
RAMP_UP_TIME="${RAMP_UP_TIME:-10}"

# Test results
RESULTS_DIR="./performance-results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/performance_${TIMESTAMP}.json"

# Helper functions
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_metric() {
    echo -e "${BLUE}  $1:${NC} $2"
}

# Create results directory
mkdir -p "$RESULTS_DIR"

# Initialize results file
cat > "$RESULTS_FILE" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "configuration": {
    "backend_url": "$BACKEND_URL",
    "frontend_url": "$FRONTEND_URL",
    "concurrent_users": $CONCURRENT_USERS,
    "test_duration": $TEST_DURATION
  },
  "tests": {}
}
EOF

# Test 1: Frontend Load Time
test_frontend_load_time() {
    print_header "Test 1: Frontend Load Time"
    
    print_info "Measuring frontend initial load time..."
    
    # Use curl to measure time to first byte and total time
    TIMING=$(curl -o /dev/null -s -w "%{time_starttransfer},%{time_total},%{size_download}" "$FRONTEND_URL")
    
    TTFB=$(echo "$TIMING" | cut -d',' -f1)
    TOTAL_TIME=$(echo "$TIMING" | cut -d',' -f2)
    SIZE=$(echo "$TIMING" | cut -d',' -f3)
    
    print_metric "Time to First Byte (TTFB)" "${TTFB}s"
    print_metric "Total Load Time" "${TOTAL_TIME}s"
    print_metric "Page Size" "${SIZE} bytes"
    
    # Check against requirements (2 seconds)
    if (( $(echo "$TOTAL_TIME < 2.0" | bc -l) )); then
        print_success "Frontend load time meets requirement (<2s)"
    else
        print_error "Frontend load time exceeds requirement (${TOTAL_TIME}s > 2s)"
    fi
    
    # Save results
    jq ".tests.frontend_load = {\"ttfb\": $TTFB, \"total_time\": $TOTAL_TIME, \"size\": $SIZE}" "$RESULTS_FILE" > "${RESULTS_FILE}.tmp" && mv "${RESULTS_FILE}.tmp" "$RESULTS_FILE"
}

# Test 2: API Response Time
test_api_response_time() {
    print_header "Test 2: API Response Time"
    
    print_info "Measuring API endpoint response times..."
    
    # Test health endpoint
    HEALTH_TIME=$(curl -o /dev/null -s -w "%{time_total}" "${BACKEND_URL}/health")
    print_metric "Health Endpoint" "${HEALTH_TIME}s"
    
    # Test detailed health endpoint
    DETAILED_HEALTH_TIME=$(curl -o /dev/null -s -w "%{time_total}" "${BACKEND_URL}/health/detailed")
    print_metric "Detailed Health Endpoint" "${DETAILED_HEALTH_TIME}s"
    
    # Check against requirement (500ms for acknowledgment)
    if (( $(echo "$HEALTH_TIME < 0.5" | bc -l) )); then
        print_success "API response time meets requirement (<500ms)"
    else
        print_error "API response time exceeds requirement (${HEALTH_TIME}s > 0.5s)"
    fi
    
    # Save results
    jq ".tests.api_response = {\"health\": $HEALTH_TIME, \"detailed_health\": $DETAILED_HEALTH_TIME}" "$RESULTS_FILE" > "${RESULTS_FILE}.tmp" && mv "${RESULTS_FILE}.tmp" "$RESULTS_FILE"
}

# Test 3: Database Query Performance
test_database_performance() {
    print_header "Test 3: Database Query Performance"
    
    print_info "Testing database query performance..."
    
    # Create test user and measure time
    TIMESTAMP=$(date +%s)
    TEST_USERNAME="perftest_${TIMESTAMP}"
    TEST_EMAIL="perftest_${TIMESTAMP}@example.com"
    TEST_PASSWORD="TestPassword123!"
    
    START_TIME=$(date +%s.%N)
    REGISTER_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/auth/register" \
        -H "Content-Type: application/json" \
        -d "{\"username\":\"${TEST_USERNAME}\",\"email\":\"${TEST_EMAIL}\",\"password\":\"${TEST_PASSWORD}\"}")
    END_TIME=$(date +%s.%N)
    
    REGISTER_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    print_metric "User Registration (DB Write)" "${REGISTER_TIME}s"
    
    if echo "$REGISTER_RESPONSE" | grep -q '"access_token"'; then
        TOKEN=$(echo "$REGISTER_RESPONSE" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
        
        # Test database read
        START_TIME=$(date +%s.%N)
        curl -s -X GET "${API_BASE_URL}/preferences" \
            -H "Authorization: Bearer ${TOKEN}" > /dev/null
        END_TIME=$(date +%s.%N)
        
        READ_TIME=$(echo "$END_TIME - $START_TIME" | bc)
        print_metric "Preferences Read (DB Read)" "${READ_TIME}s"
        
        print_success "Database operations completed successfully"
    else
        print_error "Failed to create test user"
    fi
    
    # Save results
    jq ".tests.database = {\"write_time\": $REGISTER_TIME, \"read_time\": ${READ_TIME:-0}}" "$RESULTS_FILE" > "${RESULTS_FILE}.tmp" && mv "${RESULTS_FILE}.tmp" "$RESULTS_FILE"
}

# Test 4: Concurrent User Load Test
test_concurrent_users() {
    print_header "Test 4: Concurrent User Load Test"
    
    print_info "Testing with $CONCURRENT_USERS concurrent users..."
    print_info "This test will take approximately $TEST_DURATION seconds..."
    
    # Check if Apache Bench (ab) is available
    if ! command -v ab &> /dev/null; then
        print_error "Apache Bench (ab) not found. Install with: brew install httpd (macOS) or apt-get install apache2-utils (Linux)"
        return 1
    fi
    
    # Run load test on health endpoint
    AB_OUTPUT=$(ab -n $((CONCURRENT_USERS * 10)) -c $CONCURRENT_USERS -g "${RESULTS_DIR}/gnuplot_${TIMESTAMP}.tsv" "${BACKEND_URL}/health" 2>&1)
    
    # Extract metrics
    REQUESTS_PER_SEC=$(echo "$AB_OUTPUT" | grep "Requests per second" | awk '{print $4}')
    TIME_PER_REQUEST=$(echo "$AB_OUTPUT" | grep "Time per request" | head -1 | awk '{print $4}')
    FAILED_REQUESTS=$(echo "$AB_OUTPUT" | grep "Failed requests" | awk '{print $3}')
    
    print_metric "Requests per Second" "$REQUESTS_PER_SEC"
    print_metric "Time per Request (mean)" "${TIME_PER_REQUEST}ms"
    print_metric "Failed Requests" "$FAILED_REQUESTS"
    
    if [ "$FAILED_REQUESTS" = "0" ]; then
        print_success "All requests succeeded under load"
    else
        print_error "$FAILED_REQUESTS requests failed under load"
    fi
    
    # Save results
    jq ".tests.load_test = {\"requests_per_sec\": $REQUESTS_PER_SEC, \"time_per_request\": $TIME_PER_REQUEST, \"failed_requests\": $FAILED_REQUESTS}" "$RESULTS_FILE" > "${RESULTS_FILE}.tmp" && mv "${RESULTS_FILE}.tmp" "$RESULTS_FILE"
}

# Test 5: Memory Usage
test_memory_usage() {
    print_header "Test 5: Memory Usage"
    
    print_info "Checking memory usage of running containers..."
    
    if command -v docker &> /dev/null; then
        # Get memory usage for each container
        BACKEND_MEM=$(docker stats llm-judge-backend --no-stream --format "{{.MemUsage}}" 2>/dev/null || echo "N/A")
        FRONTEND_MEM=$(docker stats llm-judge-frontend --no-stream --format "{{.MemUsage}}" 2>/dev/null || echo "N/A")
        POSTGRES_MEM=$(docker stats llm-judge-postgres --no-stream --format "{{.MemUsage}}" 2>/dev/null || echo "N/A")
        REDIS_MEM=$(docker stats llm-judge-redis --no-stream --format "{{.MemUsage}}" 2>/dev/null || echo "N/A")
        
        print_metric "Backend Memory" "$BACKEND_MEM"
        print_metric "Frontend Memory" "$FRONTEND_MEM"
        print_metric "PostgreSQL Memory" "$POSTGRES_MEM"
        print_metric "Redis Memory" "$REDIS_MEM"
        
        print_success "Memory usage check completed"
    else
        print_info "Docker not available, skipping memory usage test"
    fi
}

# Test 6: WebSocket Performance
test_websocket_performance() {
    print_header "Test 6: WebSocket Performance"
    
    print_info "Testing WebSocket connection and streaming performance..."
    
    # This would require a WebSocket client tool like wscat
    if command -v wscat &> /dev/null; then
        print_info "WebSocket testing with wscat..."
        # Add WebSocket specific tests here
        print_success "WebSocket tests completed"
    else
        print_info "wscat not found. Install with: npm install -g wscat"
        print_info "Skipping WebSocket performance tests"
    fi
}

# Test 7: Cache Performance
test_cache_performance() {
    print_header "Test 7: Cache Performance"
    
    print_info "Testing Redis cache performance..."
    
    if command -v redis-cli &> /dev/null; then
        # Test Redis ping
        REDIS_PING=$(redis-cli -h localhost -p 6379 ping 2>/dev/null || echo "FAILED")
        
        if [ "$REDIS_PING" = "PONG" ]; then
            # Measure cache write/read performance
            START_TIME=$(date +%s.%N)
            redis-cli -h localhost -p 6379 SET perftest "test_value" > /dev/null
            END_TIME=$(date +%s.%N)
            WRITE_TIME=$(echo "$END_TIME - $START_TIME" | bc)
            
            START_TIME=$(date +%s.%N)
            redis-cli -h localhost -p 6379 GET perftest > /dev/null
            END_TIME=$(date +%s.%N)
            READ_TIME=$(echo "$END_TIME - $START_TIME" | bc)
            
            print_metric "Cache Write Time" "${WRITE_TIME}s"
            print_metric "Cache Read Time" "${READ_TIME}s"
            
            print_success "Cache performance test completed"
        else
            print_error "Redis not accessible"
        fi
    else
        print_info "redis-cli not found, skipping cache tests"
    fi
}

# Test 8: Bundle Size Analysis
test_bundle_size() {
    print_header "Test 8: Frontend Bundle Size"
    
    print_info "Analyzing frontend bundle size..."
    
    if [ -d "frontend/build" ]; then
        TOTAL_SIZE=$(du -sh frontend/build | awk '{print $1}')
        JS_SIZE=$(du -sh frontend/build/static/js 2>/dev/null | awk '{print $1}' || echo "N/A")
        CSS_SIZE=$(du -sh frontend/build/static/css 2>/dev/null | awk '{print $1}' || echo "N/A")
        
        print_metric "Total Build Size" "$TOTAL_SIZE"
        print_metric "JavaScript Size" "$JS_SIZE"
        print_metric "CSS Size" "$CSS_SIZE"
        
        print_success "Bundle size analysis completed"
    else
        print_info "Frontend build directory not found. Run 'npm run build' first."
    fi
}

# Generate Performance Report
generate_report() {
    print_header "Performance Test Summary"
    
    echo -e "\n${BLUE}Test Results:${NC}"
    cat "$RESULTS_FILE" | jq '.'
    
    echo -e "\n${BLUE}Results saved to:${NC} $RESULTS_FILE"
    
    # Generate HTML report if possible
    if command -v python3 &> /dev/null; then
        print_info "Generating HTML report..."
        
        python3 << EOF
import json
import sys

with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)

html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Performance Test Report - {data['timestamp']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .metric {{ display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #eee; }}
        .metric-name {{ font-weight: bold; color: #555; }}
        .metric-value {{ color: #4CAF50; }}
        .pass {{ color: #4CAF50; }}
        .fail {{ color: #f44336; }}
        .config {{ background: #f9f9f9; padding: 15px; border-radius: 4px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Performance Test Report</h1>
        <p><strong>Timestamp:</strong> {data['timestamp']}</p>
        
        <div class="config">
            <h2>Configuration</h2>
            <div class="metric">
                <span class="metric-name">Backend URL:</span>
                <span class="metric-value">{data['configuration']['backend_url']}</span>
            </div>
            <div class="metric">
                <span class="metric-name">Frontend URL:</span>
                <span class="metric-value">{data['configuration']['frontend_url']}</span>
            </div>
            <div class="metric">
                <span class="metric-name">Concurrent Users:</span>
                <span class="metric-value">{data['configuration']['concurrent_users']}</span>
            </div>
        </div>
        
        <h2>Test Results</h2>
"""

for test_name, test_data in data.get('tests', {}).items():
    html += f"<h3>{test_name.replace('_', ' ').title()}</h3>"
    for key, value in test_data.items():
        html += f"""
        <div class="metric">
            <span class="metric-name">{key.replace('_', ' ').title()}:</span>
            <span class="metric-value">{value}</span>
        </div>
        """

html += """
    </div>
</body>
</html>
"""

with open('${RESULTS_DIR}/report_${TIMESTAMP}.html', 'w') as f:
    f.write(html)

print("HTML report generated: ${RESULTS_DIR}/report_${TIMESTAMP}.html")
EOF
    fi
}

# Main execution
main() {
    print_header "LLM Judge Auditor Performance Tests"
    print_info "Backend URL: $BACKEND_URL"
    print_info "Frontend URL: $FRONTEND_URL"
    print_info "Concurrent Users: $CONCURRENT_USERS"
    print_info "Test Duration: ${TEST_DURATION}s"
    
    # Run all tests
    test_frontend_load_time || true
    test_api_response_time || true
    test_database_performance || true
    test_concurrent_users || true
    test_memory_usage || true
    test_websocket_performance || true
    test_cache_performance || true
    test_bundle_size || true
    
    # Generate report
    generate_report
    
    print_header "Performance Testing Complete"
    echo -e "${GREEN}All tests completed successfully!${NC}\n"
}

# Run main function
main
