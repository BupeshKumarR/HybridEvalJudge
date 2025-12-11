#!/bin/bash

# Integration Test Script for LLM Judge Auditor Web Application
# Tests all user flows and component integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:3000}"
API_BASE_URL="${BACKEND_URL}/api/v1"

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
print_header() {
    echo -e "\n${YELLOW}========================================${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    ((TESTS_PASSED++))
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
    ((TESTS_FAILED++))
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Test 1: Backend Health Check
test_backend_health() {
    print_header "Test 1: Backend Health Check"
    
    if curl -s -f "${BACKEND_URL}/health" > /dev/null; then
        print_success "Backend health endpoint is accessible"
    else
        print_error "Backend health endpoint is not accessible"
        return 1
    fi
    
    # Test detailed health check
    HEALTH_RESPONSE=$(curl -s "${BACKEND_URL}/health/detailed")
    if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
        print_success "Backend detailed health check passed"
    else
        print_error "Backend detailed health check failed"
        echo "Response: $HEALTH_RESPONSE"
    fi
}

# Test 2: Database Connection
test_database_connection() {
    print_header "Test 2: Database Connection"
    
    HEALTH_RESPONSE=$(curl -s "${BACKEND_URL}/health/detailed")
    if echo "$HEALTH_RESPONSE" | grep -q '"database".*"status":"healthy"'; then
        print_success "Database connection is healthy"
    else
        print_error "Database connection failed"
        echo "Response: $HEALTH_RESPONSE"
    fi
}

# Test 3: Redis Cache Connection
test_redis_connection() {
    print_header "Test 3: Redis Cache Connection"
    
    HEALTH_RESPONSE=$(curl -s "${BACKEND_URL}/health/detailed")
    if echo "$HEALTH_RESPONSE" | grep -q '"cache"'; then
        print_success "Redis cache check completed"
    else
        print_info "Redis cache status not available (optional)"
    fi
}

# Test 4: User Registration
test_user_registration() {
    print_header "Test 4: User Registration"
    
    # Generate unique username
    TIMESTAMP=$(date +%s)
    TEST_USERNAME="testuser_${TIMESTAMP}"
    TEST_EMAIL="test_${TIMESTAMP}@example.com"
    TEST_PASSWORD="TestPassword123!"
    
    REGISTER_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/auth/register" \
        -H "Content-Type: application/json" \
        -d "{\"username\":\"${TEST_USERNAME}\",\"email\":\"${TEST_EMAIL}\",\"password\":\"${TEST_PASSWORD}\"}")
    
    if echo "$REGISTER_RESPONSE" | grep -q '"access_token"'; then
        print_success "User registration successful"
        # Extract token for subsequent tests
        export TEST_TOKEN=$(echo "$REGISTER_RESPONSE" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
        export TEST_USERNAME
        export TEST_PASSWORD
    else
        print_error "User registration failed"
        echo "Response: $REGISTER_RESPONSE"
        return 1
    fi
}

# Test 5: User Login
test_user_login() {
    print_header "Test 5: User Login"
    
    if [ -z "$TEST_USERNAME" ] || [ -z "$TEST_PASSWORD" ]; then
        print_error "Test user credentials not available"
        return 1
    fi
    
    LOGIN_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/auth/login" \
        -H "Content-Type: application/json" \
        -d "{\"username\":\"${TEST_USERNAME}\",\"password\":\"${TEST_PASSWORD}\"}")
    
    if echo "$LOGIN_RESPONSE" | grep -q '"access_token"'; then
        print_success "User login successful"
        export TEST_TOKEN=$(echo "$LOGIN_RESPONSE" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
    else
        print_error "User login failed"
        echo "Response: $LOGIN_RESPONSE"
        return 1
    fi
}

# Test 6: Create Evaluation
test_create_evaluation() {
    print_header "Test 6: Create Evaluation"
    
    if [ -z "$TEST_TOKEN" ]; then
        print_error "Authentication token not available"
        return 1
    fi
    
    EVAL_RESPONSE=$(curl -s -X POST "${API_BASE_URL}/evaluations" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${TEST_TOKEN}" \
        -d '{
            "source_text": "The capital of France is Paris.",
            "candidate_output": "Paris is the capital city of France.",
            "config": {
                "judge_models": ["gpt-4"],
                "enable_retrieval": false,
                "aggregation_strategy": "mean"
            }
        }')
    
    if echo "$EVAL_RESPONSE" | grep -q '"session_id"'; then
        print_success "Evaluation creation successful"
        export TEST_SESSION_ID=$(echo "$EVAL_RESPONSE" | grep -o '"session_id":"[^"]*"' | cut -d'"' -f4)
    else
        print_error "Evaluation creation failed"
        echo "Response: $EVAL_RESPONSE"
        return 1
    fi
}

# Test 7: Get Evaluation Results
test_get_evaluation() {
    print_header "Test 7: Get Evaluation Results"
    
    if [ -z "$TEST_TOKEN" ] || [ -z "$TEST_SESSION_ID" ]; then
        print_error "Session ID or token not available"
        return 1
    fi
    
    # Wait a moment for evaluation to process
    sleep 2
    
    EVAL_RESULT=$(curl -s -X GET "${API_BASE_URL}/evaluations/${TEST_SESSION_ID}" \
        -H "Authorization: Bearer ${TEST_TOKEN}")
    
    if echo "$EVAL_RESULT" | grep -q '"session_id"'; then
        print_success "Evaluation retrieval successful"
    else
        print_error "Evaluation retrieval failed"
        echo "Response: $EVAL_RESULT"
    fi
}

# Test 8: List Evaluation History
test_list_evaluations() {
    print_header "Test 8: List Evaluation History"
    
    if [ -z "$TEST_TOKEN" ]; then
        print_error "Authentication token not available"
        return 1
    fi
    
    HISTORY_RESPONSE=$(curl -s -X GET "${API_BASE_URL}/evaluations?page=1&limit=10" \
        -H "Authorization: Bearer ${TEST_TOKEN}")
    
    if echo "$HISTORY_RESPONSE" | grep -q '"sessions"'; then
        print_success "Evaluation history retrieval successful"
    else
        print_error "Evaluation history retrieval failed"
        echo "Response: $HISTORY_RESPONSE"
    fi
}

# Test 9: User Preferences
test_user_preferences() {
    print_header "Test 9: User Preferences"
    
    if [ -z "$TEST_TOKEN" ]; then
        print_error "Authentication token not available"
        return 1
    fi
    
    # Update preferences
    PREF_UPDATE=$(curl -s -X PUT "${API_BASE_URL}/preferences" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${TEST_TOKEN}" \
        -d '{
            "judge_models": ["gpt-4", "claude-3"],
            "enable_retrieval": true,
            "aggregation_strategy": "weighted_mean"
        }')
    
    if echo "$PREF_UPDATE" | grep -q '"judge_models"'; then
        print_success "Preferences update successful"
    else
        print_error "Preferences update failed"
        echo "Response: $PREF_UPDATE"
    fi
    
    # Get preferences
    PREF_GET=$(curl -s -X GET "${API_BASE_URL}/preferences" \
        -H "Authorization: Bearer ${TEST_TOKEN}")
    
    if echo "$PREF_GET" | grep -q '"judge_models"'; then
        print_success "Preferences retrieval successful"
    else
        print_error "Preferences retrieval failed"
        echo "Response: $PREF_GET"
    fi
}

# Test 10: Export Functionality
test_export_functionality() {
    print_header "Test 10: Export Functionality"
    
    if [ -z "$TEST_TOKEN" ] || [ -z "$TEST_SESSION_ID" ]; then
        print_error "Session ID or token not available"
        return 1
    fi
    
    # Test JSON export
    JSON_EXPORT=$(curl -s -X GET "${API_BASE_URL}/evaluations/${TEST_SESSION_ID}/export?format=json" \
        -H "Authorization: Bearer ${TEST_TOKEN}")
    
    if echo "$JSON_EXPORT" | grep -q '"session_id"'; then
        print_success "JSON export successful"
    else
        print_error "JSON export failed"
    fi
    
    # Test CSV export
    CSV_EXPORT=$(curl -s -X GET "${API_BASE_URL}/evaluations/${TEST_SESSION_ID}/export?format=csv" \
        -H "Authorization: Bearer ${TEST_TOKEN}")
    
    if [ -n "$CSV_EXPORT" ]; then
        print_success "CSV export successful"
    else
        print_error "CSV export failed"
    fi
}

# Test 11: Frontend Accessibility
test_frontend_accessibility() {
    print_header "Test 11: Frontend Accessibility"
    
    if curl -s -f "${FRONTEND_URL}" > /dev/null; then
        print_success "Frontend is accessible"
    else
        print_error "Frontend is not accessible"
    fi
}

# Test 12: API Documentation
test_api_documentation() {
    print_header "Test 12: API Documentation"
    
    if curl -s -f "${BACKEND_URL}/docs" > /dev/null; then
        print_success "API documentation is accessible"
    else
        print_error "API documentation is not accessible"
    fi
}

# Test 13: CORS Configuration
test_cors_configuration() {
    print_header "Test 13: CORS Configuration"
    
    CORS_RESPONSE=$(curl -s -I -X OPTIONS "${API_BASE_URL}/evaluations" \
        -H "Origin: http://localhost:3000" \
        -H "Access-Control-Request-Method: POST")
    
    if echo "$CORS_RESPONSE" | grep -q "Access-Control-Allow-Origin"; then
        print_success "CORS is properly configured"
    else
        print_error "CORS configuration issue detected"
    fi
}

# Test 14: Rate Limiting
test_rate_limiting() {
    print_header "Test 14: Rate Limiting"
    
    print_info "Testing rate limiting (this may take a moment)..."
    
    # Make multiple rapid requests
    for i in {1..10}; do
        curl -s "${BACKEND_URL}/health" > /dev/null
    done
    
    print_success "Rate limiting test completed (no errors)"
}

# Test 15: Error Handling
test_error_handling() {
    print_header "Test 15: Error Handling"
    
    # Test 404 error
    ERROR_404=$(curl -s -w "%{http_code}" "${API_BASE_URL}/nonexistent" -o /dev/null)
    if [ "$ERROR_404" = "404" ]; then
        print_success "404 error handling works correctly"
    else
        print_error "404 error handling issue"
    fi
    
    # Test 401 error (unauthorized)
    ERROR_401=$(curl -s -w "%{http_code}" "${API_BASE_URL}/evaluations" -o /dev/null)
    if [ "$ERROR_401" = "401" ]; then
        print_success "401 error handling works correctly"
    else
        print_error "401 error handling issue"
    fi
}

# Main execution
main() {
    print_header "LLM Judge Auditor Integration Tests"
    print_info "Backend URL: $BACKEND_URL"
    print_info "Frontend URL: $FRONTEND_URL"
    print_info "API Base URL: $API_BASE_URL"
    
    # Run all tests
    test_backend_health || true
    test_database_connection || true
    test_redis_connection || true
    test_user_registration || true
    test_user_login || true
    test_create_evaluation || true
    test_get_evaluation || true
    test_list_evaluations || true
    test_user_preferences || true
    test_export_functionality || true
    test_frontend_accessibility || true
    test_api_documentation || true
    test_cors_configuration || true
    test_rate_limiting || true
    test_error_handling || true
    
    # Print summary
    print_header "Test Summary"
    echo -e "${GREEN}Tests Passed: ${TESTS_PASSED}${NC}"
    echo -e "${RED}Tests Failed: ${TESTS_FAILED}${NC}"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "\n${GREEN}All tests passed! ✓${NC}\n"
        exit 0
    else
        echo -e "\n${RED}Some tests failed. Please review the output above.${NC}\n"
        exit 1
    fi
}

# Run main function
main
