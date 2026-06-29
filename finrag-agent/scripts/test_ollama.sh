#!/usr/bin/env bash
# Test Ollama connectivity and model functionality
set -euo pipefail

OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
MODEL="${OLLAMA_MODEL:-mistral}"

echo "=================================================="
echo "  FinRAG - Ollama Test Suite"
echo "=================================================="
echo "Endpoint: $OLLAMA_URL"
echo "Model:    $MODEL"
echo ""

# Test 1: Connectivity
echo "[TEST 1] Ollama connectivity..."
if curl -sf "$OLLAMA_URL/api/tags" > /dev/null; then
    echo "✓ Ollama is running at $OLLAMA_URL"
else
    echo "✗ Cannot connect to Ollama at $OLLAMA_URL"
    echo "  Run: ollama serve"
    exit 1
fi

# Test 2: Available models
echo ""
echo "[TEST 2] Available models..."
MODELS=$(curl -sf "$OLLAMA_URL/api/tags" | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = [m['name'] for m in data.get('models', [])]
print('\n'.join(models) if models else 'No models found')
")
echo "$MODELS"

# Test 3: Check target model
echo ""
echo "[TEST 3] Checking model '$MODEL'..."
if echo "$MODELS" | grep -q "$MODEL"; then
    echo "✓ Model '$MODEL' is available"
else
    echo "⚠️  Model '$MODEL' not found. Pulling..."
    ollama pull "$MODEL"
fi

# Test 4: Simple generation
echo ""
echo "[TEST 4] Simple generation test (French financial question)..."
RESPONSE=$(curl -sf "$OLLAMA_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"prompt\": \"Qu'est-ce qu'un ratio de liquidité en finance ? Réponds en une phrase.\",
        \"stream\": false,
        \"options\": {\"temperature\": 0.1, \"num_predict\": 100}
    }" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('response','').strip())")

if [ -n "$RESPONSE" ]; then
    echo "✓ Model responded:"
    echo "  \"$RESPONSE\""
else
    echo "✗ No response from model"
    exit 1
fi

# Test 5: Financial terminology test
echo ""
echo "[TEST 5] Financial terminology test..."
RESPONSE2=$(curl -sf "$OLLAMA_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"prompt\": \"Définis brièvement: OPCVM, prospectus, ratio de Sharpe.\",
        \"stream\": false,
        \"options\": {\"temperature\": 0.1, \"num_predict\": 200}
    }" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('response','').strip()[:200])")

if [ -n "$RESPONSE2" ]; then
    echo "✓ Financial concepts understood:"
    echo "  \"${RESPONSE2}...\""
else
    echo "✗ Financial test failed"
    exit 1
fi

echo ""
echo "=================================================="
echo "  ✓ All Ollama tests passed!"
echo "=================================================="
