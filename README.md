# Uruguay Gov Expert AI Assistant

A proof-of-concept AI-powered system that helps Uruguayan citizens navigate government services and administrative procedures online.

## 🚧 Work In Progress 🚧

This project is currently under development. The goal is to create an AI assistant that can:
- Answer questions about government procedures
- Provide step-by-step guidance for online administrative tasks
- Retrieve information from official documentation
- Assist users in navigating the Uruguayan e-government portals

## Project Overview

This system uses advanced web crawling, document processing, and RAG (Retrieval Augmented Generation) techniques to ingest official documentation and provide accurate, context-aware responses to users through a conversational interface.

## Architecture

The project is built using a microservices architecture with containerized components:

- **Crawler Service**: Processes government websites and documentation
- **RAG Service**: Provides context-aware AI responses using retrieved documents
- **Streamlit UI**: User-friendly interface for interacting with the assistant
- **Ollama**: Local language model deployment
- **Supabase**: Vector database for storing and retrieving document chunks

## Key Features

- 🕸️ Automated web crawling of official Uruguayan government sites
- 📄 Document processing with smart chunking and embedding generation
- 🧠 Vector similarity search for relevant context retrieval
- 💬 Conversational AI interface with streaming responses
- 🔍 Source attribution for transparency in AI responses
- 🐳 Fully containerized for easy deployment

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 8GB RAM available for the containers
- Internet connection for the initial model downloads

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/gubcrawler.git
   cd gubcrawler
   ```

2. Create a `.env` file with the following variables:
   ```bash
    # Supabase credentials
    SUPABASE_URL=your_supabase_url
    SUPABASE_KEY=your_supabase_key

    # Model Configuration
    MODEL_NAME=deepseek-r1:32b
    EMBEDDING_MODEL=jinaai/jina-embeddings-v2-base-es
    EMBEDDING_DIMENSION=768
    SIMILARITY_THRESHOLD=0.4

    # Crawler Configuration
    MAX_CONCURRENT=10
    CHUNK_SIZE=5000

    # Logging
    LOG_LEVEL=INFO

    # Memory Limits
    MAX_MEMORY_MB=4096

    # Sitemap 
    SITEMAP_URL=https://www.gub.uy/sitemap.xml
   ```

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Access the UI:
   ```
   http://localhost:8501
   ```

## System Components

### Crawler Service

The crawler extracts content from government websites, processes it into meaningful chunks, and stores them in the vector database. It handles:

- Sitemap parsing
- Content extraction with browser rendering
- Smart document chunking
- Embedding generation
- Vector database storage

### RAG Service

Implements Retrieval Augmented Generation to provide accurate responses by:

- Converting user queries to embeddings
- Finding relevant document chunks via vector similarity
- Creating context-aware prompts for the language model
- Streaming AI-generated responses to the user

### DeepSeek Adapter

Provides a consistent interface to the underlying LLM (Language Model) through Ollama, handling:

- Text generation
- Embedding creation
- Token counting
- Streaming responses

### Streamlit Application

Provides a user-friendly web interface for:

- Asking questions in natural language
- Viewing AI-generated responses in real-time
- Exploring source documents used for responses
- Maintaining conversation context

## Database Schema

The system uses a PostgreSQL database with vector search capabilities:

- `site_pages` table stores document chunks with vector embeddings
- Optimized indices for fast similarity search
- Row-level security for data protection

## Development

### Folder Structure

```
.
├── app/
│   ├── models/
│   │   └── deepseek_adapter.py
│   ├── services/
│   │   ├── crawler_service.py
│   │   └── rag_service.py
│   └── streamlit_app.py
├── scripts/
│   └── init.sh
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.crawler
├── Dockerfile.streamlit
├── requirements.txt
└── README.md
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## Future Work

- [ ] Implement authentication for user-specific advice
- [ ] Add more government sources and documentation
- [ ] Integrate with official APIs when available
- [ ] Develop procedure tracking capabilities
- [ ] Create a notification system for updates to procedures
- [ ] Implement a feedback mechanism for improving responses

---

© 2025 [Martin Sorriba SAS] - Developed as a Proof of Concept for improving e-government accessibility in Uruguay.