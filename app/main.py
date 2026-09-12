"""FastAPI application entry point."""
from fastapi import FastAPI
from app.api.routes import recommendation

app = FastAPI(
    title="Semantic Recommendation System",
    description="Vector-based recommendation engine",
    version="1.0.0"
)

# Include routers
app.include_router(recommendation.router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
