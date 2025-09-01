from fastapi import APIRouter
from app.api.v1.endpoints import auth, clients, users, projects, chat

api_router = APIRouter()

# Include all endpoint routers with prefixes
api_router.include_router(clients.router, tags=["Client Management"])
api_router.include_router(auth.router, tags=["Authentication"])
api_router.include_router(users.router,  tags=["User Management"])
api_router.include_router(projects.router, tags=["Project Management"])
api_router.include_router(chat.router, tags=["Code Operations"])
