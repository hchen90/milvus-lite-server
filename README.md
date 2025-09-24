# Milvus Lite Server

A lightweight server for Milvus vector database with FastAPI.

## Running the Application

To run the application, use the following command from the project root:

```bash
uv run app/main.py
```

This will start the FastAPI server on http://127.0.0.1:8000 with the Milvus vector database.

### JWT Authentication

The server supports JWT-based authentication that can be enabled or disabled:

#### Enable JWT Authentication
```bash
# Set environment variable to enable JWT authentication
export JWT_ENABLED=true
uv run python -m app.main

# Or start with inline environment variable
JWT_ENABLED=true uv run python -m app.main
```

#### Disable JWT Authentication
```bash
# Set environment variable to disable JWT authentication (anonymous access)
export JWT_ENABLED=false
uv run python -m app.main

# Or start with inline environment variable
JWT_ENABLED=false uv run python -m app.main
```

When JWT authentication is enabled, you need to provide a valid JWT token in the Authorization header for all API requests:
```
Authorization: Bearer <your-jwt-token>
```

When JWT authentication is disabled, all APIs allow anonymous access without authentication.

## API Endpoints

### Core Endpoints
- `GET /` - Root endpoint with service information
- `GET /health` - Health check endpoint

### Document Management APIs
- `POST /api/v1/documents` - Save documents to vector database (Form data)
- `POST /api/v1/documents/json` - Save documents to vector database (JSON format)
- `GET /api/v1/documents/search` - Search similar documents (Query parameters)
- `POST /api/v1/documents/search` - Search similar documents (JSON format)

### Embedding APIs
- `POST /api/v1/embedding` - Generate single embedding vector
- `POST /api/v1/embeddings` - Generate chunked embedding vectors

### Authentication APIs
- `POST /api/v1/auth/verify` - Verify JWT token validity
- `GET /api/v1/auth/profile` - Get current user profile information

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
│       ├── vector_search.py  # Document search endpoints
│       └── embedding.py      # Embedding vector endpoints
├── services/
│   ├── embedding.py     # Embedding service
│   └── milvusdb.py      # Milvus database operations
└── tests/               # Test files
```

## Document Management API Usage

### Save Document (Form Data)

Save a document to the vector database using form data:

```bash
curl -X POST "http://localhost:8000/api/v1/documents" \
     -H "Authorization: Bearer <your-jwt-token>" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "post_id=doc001&title=Sample Document&content=This is a sample document content for testing."
```

### Save Document (JSON Format)

Save a document using JSON format:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/json" \
     -H "Authorization: Bearer <your-jwt-token>" \
     -H "Content-Type: application/json" \
     -d '{
       "post_id": "doc001",
       "title": "Sample Document", 
       "content": "This is a sample document content for testing."
     }'
```

### Search Documents (Query Parameters)

Search for similar documents using query parameters:

```bash
curl -X GET "http://localhost:8000/api/v1/documents/search?query=sample%20content&limit=5" \
     -H "Authorization: Bearer <your-jwt-token>"
```

### Search Documents (JSON Format)

Search for similar documents using JSON format:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/search" \
     -H "Authorization: Bearer <your-jwt-token>" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "sample content",
       "limit": 5
     }'
```

### Python Client Example

```python
import requests

# Base URL
base_url = "http://localhost:8000"
headers = {
    "Authorization": "Bearer <your-jwt-token>",  # Remove this line if JWT is disabled
    "Content-Type": "application/json"
}

# Save a document
save_response = requests.post(
    f"{base_url}/api/v1/documents/json",
    json={
        "post_id": "doc001",
        "title": "Sample Document",
        "content": "This is a sample document for testing vector search."
    },
    headers=headers
)
print("Save result:", save_response.json())

# Search for similar documents
search_response = requests.post(
    f"{base_url}/api/v1/documents/search",
    json={
        "query": "sample document testing",
        "limit": 5
    },
    headers=headers
)
print("Search results:", search_response.json())
```

**Note**: When JWT authentication is disabled (`JWT_ENABLED=false`), you can omit the Authorization header from all requests.

## Embedding API Usage

The server provides two embedding endpoints for generating vector embeddings from text:

### Single Embedding Generation

Generate embedding vector for short text or single content:

```bash
curl -X POST "http://localhost:8000/api/v1/embedding" \
     -H "Authorization: Bearer <your-jwt-token>" \
     -H "Content-Type: application/json" \
     -d '{"content": "Your text here"}'
```

### Chunked Embedding Generation

Generate chunked embedding vectors for long text content:

```bash
curl -X POST "http://localhost:8000/api/v1/embeddings" \
     -H "Authorization: Bearer <your-jwt-token>" \
     -H "Content-Type: application/json" \
     -d '{"content": "Your long text content here..."}'
```

### Python Client Example

```python
import requests

headers = {
    "Authorization": "Bearer <your-jwt-token>",  # Remove this line if JWT is disabled
    "Content-Type": "application/json"
}

# Generate single embedding
response = requests.post(
    "http://localhost:8000/api/v1/embedding",
    json={"content": "Hello world"},
    headers=headers
)
embedding = response.json()["embedding"]

# Generate chunked embeddings
response = requests.post(
    "http://localhost:8000/api/v1/embeddings", 
    json={"content": "Very long text..."},
    headers=headers
)
embeddings = response.json()["embeddings"]
```

## JWT Authentication API Usage

### Verify Token

Verify the validity of a JWT token:

```bash
curl -X POST "http://localhost:8000/api/v1/auth/verify" \
     -H "Authorization: Bearer <your-jwt-token>"
```

### Get User Profile

Get current user profile information:

```bash
curl -X GET "http://localhost:8000/api/v1/auth/profile" \
     -H "Authorization: Bearer <your-jwt-token>"
```

### Python Client Example

```python
import requests

headers = {"Authorization": "Bearer <your-jwt-token>"}

# Verify token
verify_response = requests.post(
    "http://localhost:8000/api/v1/auth/verify",
    headers=headers
)
print("Token verification:", verify_response.json())

# Get profile
profile_response = requests.get(
    "http://localhost:8000/api/v1/auth/profile",
    headers=headers
)
print("User profile:", profile_response.json())
```

For more detailed examples, see:
- `test_embedding_api.py` - API testing script