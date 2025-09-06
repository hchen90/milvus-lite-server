# Milvus Lite Server

A lightweight server for Milvus vector database with FastAPI.

## Running the Application

To run the application, use the following command from the project root:

```bash
uv run app/main.py
```

This will start the FastAPI server on http://127.0.0.1:8000 with the Milvus vector database.

## API Endpoints

- `GET /` - Root endpoint with service information
- `GET /health` - Health check endpoint
- `POST /api/v1/documents` - Save documents to vector database
- `POST /api/v1/documents/json` - Save documents via JSON
- `GET /api/v1/documents/search` - Search similar documents
- `POST /api/v1/documents/search` - Search documents via JSON

## Development

The application uses:
- FastAPI for the web framework
- Milvus Lite for vector database
- Sentence Transformers for embeddings
- Pydantic for data validation

## Project Structure

```
app/
├── main.py              # Main application entry point
├── core/
│   └── config.py        # Configuration management
├── api/
│   ├── models.py        # Pydantic models
│   └── v1/
│       ├── vector_save.py    # Document saving endpoints
│       └── vector_search.py  # Document search endpoints
├── services/
│   ├── embedding.py     # Embedding service
│   └── milvusdb.py      # Milvus database operations
└── tests/               # Test files
```