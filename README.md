# Semantic Recommendation System

A vector-based recommendation engine using semantic similarity to provide intelligent product recommendations.

## Features

- **Vector Embeddings**: Generate semantic embeddings for products using transformer models
- **Fast Vector Search**: Efficient similarity search using FAISS or other vector databases
- **REST API**: FastAPI-based API for easy integration
- **Scalable**: Designed for handling large product catalogs

## Project Structure

```
semantic-recommender/
├── app/                          # Main application package
│   ├── api/                      # API routes and endpoints
│   │   └── routes/
│   │       └── recommendation.py # Recommendation endpoints
│   ├── schemas/                  # Pydantic models for request/response
│   ├── services/                 # Business logic services
│   │   ├── embedding_service.py  # Text embedding generation
│   │   └── vector_search_service.py # Vector similarity search
│   ├── models/                   # Data models
│   │   └── product.py
│   ├── core/                     # Core configuration
│   │   └── config.py
│   └── main.py                   # FastAPI application entry point
├── tests/                        # Unit and integration tests
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd semantic-recommender
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:
```env
API_TITLE=Semantic Recommendation System
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_DB_TYPE=faiss
LOG_LEVEL=INFO
```

## Running the Application

```bash
python -m app.main
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## API Endpoints

### Get Recommendations
- **POST** `/api/v1/recommendations`
- Request: `{ "query": "product description", "limit": 5 }`
- Response: List of recommended products with similarity scores

### Get Product Recommendations
- **GET** `/api/v1/recommendations/{product_id}`
- Query params: `limit=5`
- Response: Similar products for the given product

## Testing

Run tests with pytest:
```bash
pytest tests/
```

## Development

Install development dependencies:
```bash
pip install -r requirements.txt
pip install black flake8 mypy
```

Format code:
```bash
black app/ tests/
```

Lint code:
```bash
flake8 app/ tests/
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Write tests
4. Submit a pull request

## License

MIT License
