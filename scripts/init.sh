#!/bin/sh
set -e

MAX_RETRIES=5
RETRY_COUNT=0

# --- 1) Source .env file to load environment variables ---
echo "Loading environment variables..."
if [ -f "/.env" ]; then
    # This command exports all non-commented lines from .env
    export $(grep -v '^#' /.env | xargs)
else
    echo "Warning: /.env not found. Using default MODEL_NAME if set in Docker environment."
fi

# --- 2) Wait for Ollama to be healthy ---
echo "Waiting for Ollama to be ready..."
until curl -s http://ollama:11434/api/health > /dev/null; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "Failed to connect to Ollama after $MAX_RETRIES attempts"
        exit 1
    fi
    echo "Waiting for Ollama... attempt $RETRY_COUNT/$MAX_RETRIES"
    sleep 2
done

# --- 3) Use the $MODEL_NAME variable from .env (or fallback if empty) ---
MODEL_NAME="${MODEL_NAME:-deepseek-r1:1.5b}"  # fallback if .env not set
EMBEDDING_MODEL="${EMBEDDING_MODEL:-jina/jina-embeddings-v2-base-es}"

# Check if model exists
echo "Checking if model '$MODEL_NAME' already exists..."
if ! curl -s "http://ollama:11434/api/show" -d "{\"name\":\"$MODEL_NAME\"}" | grep -q "id"; then
    echo "Pulling $MODEL_NAME model..."
    curl -X POST http://ollama:11434/api/pull -d "{\"name\":\"$MODEL_NAME\"}"
    echo "Model pull initiated"
else
    echo "Model '$MODEL_NAME' already exists"
fi
if ! curl -s "http://ollama:11434/api/show" -d "{\"name\":\"$EMBEDDING_MODEL\"}" | grep -q "id"; then
    curl -X POST http://ollama:11434/api/pull -d "{\"name\":\"$EMBEDDING_MODEL\"}"
    echo "Embedded Model pull initiated"
else
    echo "Embedded Model '$EMBEDDING_MODEL' already exists"
fi

echo "Model setup complete!"
