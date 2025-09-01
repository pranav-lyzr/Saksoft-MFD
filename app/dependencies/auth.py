from fastapi import Depends, HTTPException, status, Header
from fastapi.security import OAuth2PasswordBearer
from app.models.user import User, UserType
from app.models.client import Client
from app.database import users_collection, clients_collection
from app.config import SECRET_KEY, ALGORITHM
from bson import ObjectId
from typing import Optional
import jwt
from jwt import PyJWTError

# OAuth2 Scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        user_type: str = payload.get("user_type")
        client_id: str = payload.get("client_id")
        if not user_id or not user_type or not client_id:
            raise credentials_exception
    except PyJWTError:
        raise credentials_exception
    
    user = await users_collection.find_one({"_id": ObjectId(user_id), "client_id": client_id})
    if not user or not user["is_active"]:
        raise credentials_exception
    
    return User(
        id=str(user["_id"]),
        username=user["username"],
        user_type=UserType(user["user_type"]),
        is_active=user["is_active"],
        projects=[str(pid) for pid in user.get("projects", [])],
        client_id=user["client_id"]
    )

async def get_current_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current admin user"""
    if current_user.user_type != UserType.admin:
        raise HTTPException(status_code=403, detail="Operation not permitted")
    return current_user

async def get_current_admin_or_project_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current admin or project admin user"""
    if current_user.user_type not in [UserType.admin, UserType.project_admin]:
        raise HTTPException(status_code=403, detail="Operation permitted only for admins and project admins")
    return current_user

async def validate_special_key(special_key: str = Header(...)) -> str:
    """Validate special key for client operations"""
    if special_key != "lyzr-saksoft":
        raise HTTPException(status_code=403, detail="Invalid special key")
    return special_key

async def get_client_from_secret_key(secret_key: str = Header(...)):
    """Get client from secret key"""
    client = await clients_collection.find_one({"secret_key": secret_key})
    if not client:
        raise HTTPException(status_code=401, detail="Invalid client secret key")
    return Client(
        id=str(client["_id"]),
        name=client["name"],
        admin_id=str(client["admin_id"]),
        created_at=client["created_at"]
    )

async def check_project_access(project_id: str, current_user: User = Depends(get_current_user)):
    """Check if user has access to a project"""
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")
    
    from app.database import projects_collection
    project = await projects_collection.find_one({"_id": obj_id, "client_id": current_user.client_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    
    if current_user.user_type != UserType.admin and str(obj_id) not in current_user.projects:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")
    return project
