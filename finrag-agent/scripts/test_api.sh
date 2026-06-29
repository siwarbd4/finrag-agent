#!/usr/bin/env bash
# End-to-end API test with a sample financial PDF
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
SAMPLE_PDF="${1:-}"

echo "=================================================="
echo "  FinRAG Agent - API Integration Test"
echo "=================================================="
echo "API: $API_URL"
echo ""

# Check API health
echo "[1/5] Health check..."
HEALTH=$(curl -sf "$API_URL/api/v1/health")
STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])")
echo "→ System status: $STATUS"
if [ "$STATUS" = "unhealthy" ]; then
    echo "✗ System is unhealthy. Check logs."
    exit 1
fi
echo "✓ Health check passed"

# Upload document
if [ -n "$SAMPLE_PDF" ] && [ -f "$SAMPLE_PDF" ]; then
    echo ""
    echo "[2/5] Uploading PDF: $(basename $SAMPLE_PDF)..."
    UPLOAD_RESP=$(curl -sf -X POST "$API_URL/api/v1/documents/upload" \
        -F "file=@$SAMPLE_PDF")
    DOC_ID=$(echo "$UPLOAD_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['document_id'])")
    echo "✓ Document uploaded. ID: $DOC_ID"
    echo "  $(echo $UPLOAD_RESP | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["message"])')"
else
    echo ""
    echo "[2/5] No PDF provided, skipping upload test"
    echo "  Usage: $0 /path/to/financial_report.pdf"
    DOC_ID=""
fi

# List documents
echo ""
echo "[3/5] Listing indexed documents..."
DOCS=$(curl -sf "$API_URL/api/v1/documents/")
TOTAL=$(echo "$DOCS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['total'])")
echo "✓ Total documents indexed: $TOTAL"

# Ask a question
echo ""
echo "[4/5] Querying (without specific document)..."
QUERY_RESP=$(curl -sf -X POST "$API_URL/api/v1/query/" \
    -H "Content-Type: application/json" \
    -d '{"question": "Quels sont les principaux risques financiers mentionnés dans les documents ?", "language": "fr"}')

if echo "$QUERY_RESP" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
    CHUNKS=$(echo "$QUERY_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['chunks_retrieved'])")
    echo "✓ Query answered. Chunks retrieved: $CHUNKS"
    ANSWER=$(echo "$QUERY_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['answer'][:200])")
    echo "  Answer preview: \"$ANSWER...\""
else
    echo "⚠️  Query returned non-JSON (Ollama may be down)"
fi

# Query history
echo ""
echo "[5/5] Query history..."
HISTORY=$(curl -sf "$API_URL/api/v1/query/history?limit=5")
HIST_TOTAL=$(echo "$HISTORY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['total'])")
echo "✓ Recent queries logged: $HIST_TOTAL"

echo ""
echo "=================================================="
echo "  ✓ API tests complete!"
echo "=================================================="
echo ""
echo "  Swagger UI: $API_URL/docs"
echo "  ReDoc:      $API_URL/redoc"
