from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router
from app.database import client

app = FastAPI(
    title="Saksoft Coding Agent API",
    description="API for managing users, projects, and code operations. Authentication is applied only to specific endpoints.",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

# Health check endpoint
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy"}

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    print("Starting Saksoft MFD API...")

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down Saksoft MFD API...")
    client.close()
