#!/usr/bin/env bash
# API testing script for DirtyCar service

set -e

# Configuration
API_URL="http://localhost:7439"
TEST_IMAGE="${2:-}"

echo "Testing DirtyCar API at: $API_URL"

# Test health endpoint
echo "1. Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s "$API_URL/healthz" || echo "FAILED")
echo "Health response: $HEALTH_RESPONSE"

if echo "$HEALTH_RESPONSE" | grep -q '"status":"ok"'; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed"
    exit 1
fi

# Test model info endpoint
echo ""
echo "2. Testing model info endpoint..."
MODEL_INFO=$(curl -s "$API_URL/model/info" || echo "FAILED")
echo "Model info: $MODEL_INFO"

# Test file upload prediction (if test image provided)
if [ -n "$TEST_IMAGE" ] && [ -f "$TEST_IMAGE" ]; then
    echo ""
    echo "3. Testing file upload prediction..."
    echo "Test image: $TEST_IMAGE"
    
    PREDICTION_RESPONSE=$(curl -s -X POST \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@$TEST_IMAGE" \
        "$API_URL/predict/file" || echo "FAILED")
    
    echo "Prediction response: $PREDICTION_RESPONSE"
    
    if echo "$PREDICTION_RESPONSE" | grep -q '"label"'; then
        echo "✓ File prediction test passed"
    else
        echo "✗ File prediction test failed"
    fi
else
    echo ""
    echo "3. Skipping file upload test (no test image provided)"
    echo "   Usage: $0 <api_url> <test_image_path>"
fi

# Test URL prediction with a sample image
echo ""
echo "4. Testing URL prediction..."
SAMPLE_URL="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Clean_car.jpg/320px-Clean_car.jpg"

URL_PREDICTION=$(curl -s -X POST \
    -H "accept: application/json" \
    -H "Content-Type: application/json" \
    -d "{\"url\": \"$SAMPLE_URL\"}" \
    "$API_URL/predict/url" || echo "FAILED")

echo "URL prediction response: $URL_PREDICTION"

if echo "$URL_PREDICTION" | grep -q '"label"'; then
    echo "✓ URL prediction test passed"
else
    echo "✗ URL prediction test failed"
fi

# Test root endpoint
echo ""
echo "5. Testing root endpoint..."
ROOT_RESPONSE=$(curl -s "$API_URL/" || echo "FAILED")
echo "Root response: $ROOT_RESPONSE"

if echo "$ROOT_RESPONSE" | grep -q "DirtyCar"; then
    echo "✓ Root endpoint test passed"
else
    echo "✗ Root endpoint test failed"
fi

echo ""
echo "API testing completed!"
echo ""
echo "Available endpoints:"
echo "  GET  $API_URL/              - API information"
echo "  GET  $API_URL/healthz       - Health check"
echo "  GET  $API_URL/model/info    - Model information"
echo "  POST $API_URL/predict/file  - File upload prediction"
echo "  POST $API_URL/predict/url   - URL prediction"
echo "  GET  $API_URL/docs          - API documentation"
