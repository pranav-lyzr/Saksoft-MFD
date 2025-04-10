from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, HttpUrl
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from git import Repo
import uuid
import tempfile
import shutil
from typing import List, Optional
import json
import httpx
import time
import requests
from analyzer.core import RepositoryAnalyzer
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from enum import Enum
import jwt
import os
from passlib.context import CryptContext
from uuid import uuid4
import tiktoken

app = FastAPI(
    title="Saksoft Coding Agent API",
    description="API for managing users, projects, and code operations. Authentication is applied only to specific endpoints.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Connection
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://root:example@mongo:27017/mydatabase?authSource=admin")
if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in the .env file")
client = AsyncIOMotorClient(MONGO_URI)
db = client["mydatabase"]
analyzer = RepositoryAnalyzer()

# External API Configurations
LYZR_RAG_API_URL = "https://rag-prod.studio.lyzr.ai/v3/rag"
LYZR_AGENT_API_URL = "https://agent-prod.studio.lyzr.ai/v3/agent"
LYZR_API_URL = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
API_KEY = "sk-default-yStV4gbpjadbQSw4i7QhoOLRwAs5dEcl"
USER_ID = "pranav@lyzr.ai"

# Pydantic Models (unchanged)
class ClientCreate(BaseModel):
    name: str
    admin_username: str
    admin_password: str
    special_key: str # Must be "lyzr-saksoft" to create a client

class ClientUpdate(BaseModel):
    name: Optional[str]

class Client(BaseModel):
    id: str
    name: str
    admin_id: str
    created_at: datetime

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class ProjectCreate(BaseModel):
    name: str

class GitHubLink(BaseModel):
    github_url: HttpUrl
    source_name: str
    pat: Optional[str] = None

class AgentQuery(BaseModel):
    message: str

class DocumentationInput(BaseModel):
    text: str
    source_name: str

class UserType(str, Enum):
    admin = "admin"
    project_admin = "project_admin"
    developer = "developer"

class UserCreate(BaseModel):
    username: str
    password: str
    user_type: UserType

class UserUpdate(BaseModel):
    username: Optional[str]
    password: Optional[str]
    user_type: Optional[UserType]
    is_active: Optional[bool]
    projects: Optional[List[str]]

class User(BaseModel):
    id: str
    username: str
    user_type: UserType
    is_active: bool
    projects: List[str]
    client_id: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class Login(BaseModel):
    username: str
    password: str

class PasswordChange(BaseModel):
    current_password: str
    new_password: str

class ResetPassword(BaseModel):
    new_password: str

class ProjectAssignment(BaseModel):
    project_id: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str

class UserWithSessions(User):
    chat_sessions: List[str]

class DeleteRagDocuments(BaseModel):
    source_name: str

class ChatSessionCreate(BaseModel):
    pass

class ChatSession(BaseModel):
    id: str
    agent_session_id: str
    user_id: str
    project_id: str
    type: str
    created_at: datetime
    updated_at: datetime

class MessageCreate(BaseModel):
    content: str

class Message(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    timestamp: datetime
    type: str

class UserWithSessionsAndChats(BaseModel):
    id: str
    username: str
    user_type: UserType
    is_active: bool
    projects: List[str]
    chat_sessions: List[ChatSession]
    chat_messages: List[Message]

# New Pydantic Model for Project Response
class ProjectResponse(BaseModel):
    id: str
    name: str
    client_id: str
    created_by: str
    created_at: datetime
    github_links: List[dict]
    repo_analyses: List[dict]  # Simplified to avoid large data
    documentation: List[dict]

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# JWT Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 Scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# Authentication Utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_client_from_secret_key(secret_key: str = Header(...)):
    client = await db.clients.find_one({"secret_key": secret_key})
    if not client:
        raise HTTPException(status_code=401, detail="Invalid client secret key")
    return Client(
        id=str(client["_id"]),
        name=client["name"],
        admin_id=str(client["admin_id"]),
        secret_key=client["secret_key"],
        created_at=client["created_at"]
    )

async def get_current_user(token: str = Depends(oauth2_scheme)):
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
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = await db.users.find_one({"_id": ObjectId(user_id), "client_id": client_id})
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

async def get_current_admin_user(current_user: User = Depends(get_current_user)):
    if current_user.user_type != UserType.admin:
        raise HTTPException(status_code=403, detail="Operation not permitted")
    return current_user

async def check_project_access(project_id: str, current_user: User = Depends(get_current_user)):
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")
    
    project = await db.projects.find_one({"_id": obj_id, "client_id": current_user.client_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    
    if current_user.user_type != UserType.admin and str(obj_id) not in current_user.projects:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")
    return project

async def validate_special_key(special_key: str = Header(...)):
    if special_key != "lyzr-saksoft":
        raise HTTPException(status_code=403, detail="Invalid special key")
    return special_key

# Client Management Endpoints (Secret Key Required)
@app.post("/clients", response_model=Client, tags=["Client Management"])
async def create_client(client_data: ClientCreate):
    if client_data.special_key != "lyzr-saksoft":
        raise HTTPException(status_code=403, detail="Invalid special key")
    existing_client = await db.clients.find_one({"name": client_data.name})
    if existing_client:
        raise HTTPException(status_code=400, detail="Client name already exists")
    
    hashed_password = pwd_context.hash(client_data.admin_password)
    admin_data = {
        "username": client_data.admin_username,
        "hashed_password": hashed_password,
        "user_type": UserType.admin,
        "is_active": True,
        "projects": [],
        "client_id": None
    }
    admin_result = await db.users.insert_one(admin_data)
    admin_id = admin_result.inserted_id

    new_client = {
        "name": client_data.name,
        "admin_id": admin_id,
        "created_at": datetime.utcnow()
    }
    client_result = await db.clients.insert_one(new_client)
    client_id = str(client_result.inserted_id)
    await db.users.update_one({"_id": admin_id}, {"$set": {"client_id": client_id}})

    return Client(
        id=client_id,
        name=new_client["name"],
        admin_id=str(new_client["admin_id"]),
        created_at=new_client["created_at"]
    )

@app.get("/clients/{client_id}", response_model=Client, tags=["Client Management"])
async def get_client(client_id: str, special_key: str = Depends(validate_special_key)):
    try:
        client = await db.clients.find_one({"_id": ObjectId(client_id)})
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        return Client(
            id=str(client["_id"]),
            name=client["name"],
            admin_id=str(client["admin_id"]),
            created_at=client["created_at"]
        )
    except:
        raise HTTPException(status_code=400, detail="Invalid client ID format")

@app.put("/clients/{client_id}", response_model=Client, tags=["Client Management"])
async def update_client(client_id: str, client_update: ClientUpdate, special_key: str = Depends(validate_special_key)):
    update_data = client_update.dict(exclude_unset=True)
    result = await db.clients.update_one({"_id": ObjectId(client_id)}, {"$set": update_data})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Client not found or no changes made")
    updated_client = await db.clients.find_one({"_id": ObjectId(client_id)})
    return Client(
        id=str(updated_client["_id"]),
        name=updated_client["name"],
        admin_id=str(updated_client["admin_id"]),
        created_at=updated_client["created_at"]
    )

@app.delete("/clients/{client_id}", tags=["Client Management"])
async def delete_client(client_id: str, special_key: str = Depends(validate_special_key)):
    await db.users.delete_many({"client_id": client_id})
    await db.projects.delete_many({"client_id": client_id})
    await db.chat_sessions.delete_many({"project_id": {"$in": await db.projects.distinct("_id", {"client_id": client_id})}})
    await db.chat_messages.delete_many({"session_id": {"$in": await db.chat_sessions.distinct("_id", {"project_id": {"$in": await db.projects.distinct("_id", {"client_id": client_id})}})}})
    result = await db.clients.delete_one({"_id": ObjectId(client_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Client not found")
    return {"message": "Client and all associated data deleted successfully"}
# Authentication Endpoints (No Secret Key)
@app.post("/login", response_model=Token, tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await db.users.find_one({"username": form_data.username})
    if not user or not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Inactive user")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user["_id"]), "user_type": user["user_type"], "client_id": user["client_id"]},
        expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": str(user["_id"])
    }

@app.post("/users", response_model=User, tags=["User Management"])
async def create_user(user: UserCreate, current_user: User = Depends(get_current_admin_user)):
    # Check if the username already exists within the same client
    existing_user = await db.users.find_one({
        "username": user.username,
        "client_id": current_user.client_id
    })
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Username already registered under this client. Please choose a different username."
        )
    
    # Proceed with user creation if username is unique
    hashed_password = pwd_context.hash(user.password)
    new_user = {
        "username": user.username,
        "hashed_password": hashed_password,
        "user_type": user.user_type,
        "is_active": True,
        "projects": [],
        "client_id": current_user.client_id
    }
    result = await db.users.insert_one(new_user)
    created_user = await db.users.find_one({"_id": result.inserted_id})
    return User(
        id=str(created_user["_id"]),
        username=created_user["username"],
        user_type=created_user["user_type"],
        is_active=created_user["is_active"],
        projects=[str(pid) for pid in created_user.get("projects", [])],
        client_id=created_user["client_id"]
    )


@app.get("/users", response_model=List[User], tags=["User Management"])
async def read_users(current_user: User = Depends(get_current_user)):
    if current_user.user_type == UserType.admin:
        users = await db.users.find({"client_id": current_user.client_id}).to_list(None)
    elif current_user.user_type == UserType.project_admin:
        project_ids = [ObjectId(pid) for pid in current_user.projects]
        users = await db.users.find({
            "projects": {"$in": project_ids},
            "client_id": current_user.client_id
        }).to_list(None)
    else:  # developer
        users = [await db.users.find_one({"_id": ObjectId(current_user.id)})]
    
    return [
        User(
            id=str(user["_id"]),
            username=user["username"],
            user_type=user["user_type"],
            is_active=user["is_active"],
            projects=[str(pid) for pid in user.get("projects", [])],
            client_id=user["client_id"]
        )
        for user in users
    ]

@app.get("/users/{user_id}", response_model=UserWithSessionsAndChats, tags=["User Management"])
async def read_user(user_id: str, current_user: User = Depends(get_current_user)):
    try:
        user = await db.users.find_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found or not in your client")
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    user_model = User(
        id=str(user["_id"]),
        username=user["username"],
        user_type=UserType(user["user_type"]),
        is_active=user["is_active"],
        projects=[str(pid) for pid in user.get("projects", [])],
        client_id=user["client_id"]
    )
    
    if current_user.user_type == UserType.admin:
        pass
    elif current_user.user_type == UserType.project_admin:
        if not set(current_user.projects).intersection(user_model.projects) and user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to view this user")
    else:  # developer
        if user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to view this user")
    
    sessions = await db.chat_sessions.find({"user_id": user_id}).to_list(None)
    # Convert project_id to string when creating ChatSession instances
    chat_sessions = [
        ChatSession(
            id=str(session["_id"]),
            agent_session_id=session["agent_session_id"],
            user_id=session["user_id"],
            project_id=str(session["project_id"]),  # Convert ObjectId to string
            type=session["type"],
            created_at=session["created_at"],
            updated_at=session["updated_at"]
        ) for session in sessions
    ]
    session_ids = [ObjectId(session.id) for session in chat_sessions]
    messages = await db.chat_messages.find({"session_id": {"$in": session_ids}}).to_list(None)
    chat_messages = [
        Message(
            id=str(message["_id"]),
            session_id=str(message["session_id"]),
            role=message["role"],
            content=message["content"],
            timestamp=message["timestamp"],
            type=message["type"]
        ) for message in messages
    ]
    
    return UserWithSessionsAndChats(
        **user_model.dict(),
        chat_sessions=chat_sessions,
        chat_messages=chat_messages
    )


@app.put("/users/{user_id}", response_model=User, tags=["User Management"])
async def update_user(user_id: str, user_update: UserUpdate, current_user: User = Depends(get_current_admin_user)):
    user = await db.users.find_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found or not in your client")
    
    update_data = user_update.dict(exclude_unset=True)
    if "password" in update_data:
        update_data["hashed_password"] = pwd_context.hash(update_data.pop("password"))
    if "projects" in update_data:
        update_data["projects"] = [ObjectId(pid) for pid in update_data["projects"]]
    
    await db.users.update_one({"_id": ObjectId(user_id)}, {"$set": update_data})
    updated_user = await db.users.find_one({"_id": ObjectId(user_id)})
    return User(**{**updated_user, "id": str(updated_user["_id"]), "projects": [str(pid) for pid in updated_user.get("projects", [])]})

@app.delete("/users/{user_id}", tags=["User Management"])
async def delete_user(user_id: str, current_user: User = Depends(get_current_admin_user)):
    result = await db.users.delete_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found or not in your client")
    return {"message": "User deleted successfully"}

@app.post("/users/me/change_password", tags=["User Management"])
async def change_password(password_change: PasswordChange, current_user: User = Depends(get_current_user)):
    user = await db.users.find_one({"_id": ObjectId(current_user.id)})
    if not pwd_context.verify(password_change.current_password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect current password")
    
    hashed_new_password = pwd_context.hash(password_change.new_password)
    await db.users.update_one(
        {"_id": ObjectId(current_user.id)},
        {"$set": {"hashed_password": hashed_new_password}}
    )
    return {"message": "Password changed successfully"}

@app.post("/users/{user_id}/reset_password", tags=["User Management"])
async def reset_password(user_id: str, reset_data: ResetPassword, current_user: User = Depends(get_current_admin_user)):
    user = await db.users.find_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found or not in your client")
    hashed_new_password = pwd_context.hash(reset_data.new_password)
    await db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"hashed_password": hashed_new_password}}
    )
    return {"message": "Password reset successfully"}

@app.post("/users/{user_id}/activate", tags=["User Management"])
async def activate_user(user_id: str, current_user: User = Depends(get_current_admin_user)):
    result = await db.users.update_one(
        {"_id": ObjectId(user_id), "client_id": current_user.client_id},
        {"$set": {"is_active": True}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found or not in your client")
    return {"message": "User activated successfully"}

@app.post("/users/{user_id}/deactivate", tags=["User Management"])
async def deactivate_user(user_id: str, current_user: User = Depends(get_current_admin_user)):
    result = await db.users.update_one(
        {"_id": ObjectId(user_id), "client_id": current_user.client_id},
        {"$set": {"is_active": False}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found or not in your client")
    return {"message": "User deactivated successfully"}

@app.post("/users/{user_id}/assign_project", tags=["User Management"])
async def assign_project(user_id: str, assignment: ProjectAssignment, current_user: User = Depends(get_current_admin_user)):
    user = await db.users.find_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found or not in your client")
    
    project = await db.projects.find_one({"_id": ObjectId(assignment.project_id), "client_id": current_user.client_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    
    result = await db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$addToSet": {"projects": ObjectId(assignment.project_id)}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "Project assigned successfully"}

@app.post("/users/{user_id}/remove_project", tags=["User Management"])
async def remove_project(user_id: str, assignment: ProjectAssignment, current_user: User = Depends(get_current_admin_user)):
    user = await db.users.find_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found or not in your client")
    
    result = await db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$pull": {"projects": ObjectId(assignment.project_id)}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found or project not assigned")
    return {"message": "Project removed successfully"}


async def get_current_admin_or_project_admin_user(current_user: User = Depends(get_current_user)):
    if current_user.user_type not in [UserType.admin, UserType.project_admin]:
        raise HTTPException(status_code=403, detail="Operation permitted only for admins and project admins")
    return current_user



# Project Management APIs (No Secret Key)
# New Endpoint: Fetch All Projects
@app.get("/projects", response_model=List[ProjectResponse], tags=["Project Management"], summary="List all accessible projects")
async def list_projects(current_user: User = Depends(get_current_admin_or_project_admin_user)):
    """
    Retrieve all projects accessible to the logged-in user.
    - Admins: Get all projects within their client.
    - Project Admins: Get only their assigned projects within their client.
    Requires JWT authentication.
    """
    if current_user.user_type == UserType.admin:
        # Fetch all projects for the client
        projects = await db.projects.find({"client_id": current_user.client_id}).to_list(None)
    else:  # UserType.project_admin
        # Fetch only projects assigned to the project admin
        project_ids = [ObjectId(pid) for pid in current_user.projects]
        projects = await db.projects.find({
            "_id": {"$in": project_ids},
            "client_id": current_user.client_id  # Ensure client scoping
        }).to_list(None)
    
    if not projects:
        return []  # Return empty list if no projects found

    # Format response
    return [
        ProjectResponse(
            id=str(project["_id"]),
            name=project["name"],
            client_id=project["client_id"],
            created_by=project["created_by"],
            created_at=project["created_at"],
            github_links=project.get("github_links", []),
            repo_analyses=[{"repo_url": analysis["repo_url"]} for analysis in project.get("repo_analyses", [])],  # Simplified
            documentation=project.get("documentation", [])
        )
        for project in projects
    ]

@app.post("/create_project", tags=["Project Management"])
async def create_project(project: ProjectCreate, current_user: User = Depends(get_current_user)):
    if current_user.user_type not in [UserType.admin, UserType.project_admin]:
        raise HTTPException(status_code=403, detail="Only admins and project admins can create projects")
    
    new_project = {
        "name": project.name,
        "github_links": [],
        "repo_analyses": [],
        "client_id": current_user.client_id,
        "created_by": current_user.id,
        "created_at": datetime.utcnow()
    }
    result = await db.projects.insert_one(new_project)
    
    if current_user.user_type == UserType.project_admin:
        await db.users.update_one(
            {"_id": ObjectId(current_user.id)},
            {"$push": {"projects": str(result.inserted_id)}}
        )
    
    return {"project_id": str(result.inserted_id), "message": "Project created successfully"}

@app.post("/project/{project_id}/repo", tags=["Project Management"])
async def add_github_link(project_id: str, link: GitHubLink, current_user: User = Depends(get_current_user)):
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await db.projects.find_one({"_id": obj_id, "client_id": current_user.client_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    
    if current_user.user_type != UserType.admin and str(obj_id) not in current_user.projects:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")

    github_url_str = str(link.github_url)
    github_link_data = {"url": github_url_str, "source_name": link.source_name}
    if link.pat:
        github_link_data["pat"] = link.pat

    await db.projects.update_one(
        {"_id": obj_id},
        {"$push": {"github_links": github_link_data}}
    )
    await analyze_repository_background(obj_id, github_url_str, link.source_name, link.pat)
    return {"message": f"GitHub link '{link.source_name}' added successfully. Analysis started."}

@app.post("/project/{project_id}/documentation", tags=["Project Management"])
async def add_documentation(project_id: str, input: DocumentationInput, current_user: User = Depends(get_current_user)):
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await db.projects.find_one({"_id": obj_id, "client_id": current_user.client_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    
    if current_user.user_type != UserType.admin and str(obj_id) not in current_user.projects:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")

    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    await db.projects.update_one(
        {"_id": obj_id},
        {"$push": {"documentation": {"text": input.text, "source_name": input.source_name, "submitted_at": datetime.utcnow()}}}
    )

    text_chunks = chunk_text(input.text)
    chunked_documents = [
        {
            "id_": str(uuid.uuid4()),
            "embedding": None,
            "metadata": {"source": input.source_name, "chunked": True},
            "text": chunk.strip(),
            "excluded_embed_metadata_keys": [],
            "excluded_llm_metadata_keys": []
        }
        for chunk in text_chunks
    ]

    if "rag_id" in project:
        rag_id = project["rag_id"]
        if not train_rag(rag_id, chunked_documents):
            raise HTTPException(status_code=500, detail="Failed to train RAG")
    else:
        rag_id = create_rag_collection()
        if not rag_id:
            raise HTTPException(status_code=500, detail="Failed to create RAG collection")
        
        if not train_rag(rag_id, chunked_documents):
            raise HTTPException(status_code=500, detail="Failed to train RAG")
        
        project_name = project["name"]
        search_agent = create_agent(rag_id, "search", SEARCH_INSTRUCTIONS, project_name)
        generate_agent = create_agent(rag_id, "generate", GENERATE_INSTRUCTIONS, project_name)
        
        if not search_agent or not generate_agent:
            raise HTTPException(status_code=500, detail="Failed to create agents")
        
        update_data = {
            "rag_id": rag_id,
            "search_agent_id": search_agent.get("agent_id"),
            "generate_agent_id": generate_agent.get("agent_id")
        }
        await db.projects.update_one({"_id": obj_id}, {"$set": update_data})

    return {"message": f"Documentation '{input.source_name}' added and RAG trained successfully"}

@app.get("/project/{project_id}/rag/documents", tags=["Project Management"])
async def get_rag_documents(project_id: str, current_user: User = Depends(get_current_user)):
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await db.projects.find_one({"_id": obj_id, "client_id": current_user.client_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    
    if current_user.user_type != UserType.admin and str(obj_id) not in current_user.projects:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")

    if "rag_id" not in project:
        raise HTTPException(status_code=404, detail="RAG collection not configured for this project")

    rag_id = project["rag_id"]
    try:
        response = requests.get(
            f"{LYZR_RAG_API_URL}/documents/{rag_id}/",
            headers={"x-api-key": API_KEY, "accept": "application/json"}
        )
        response.raise_for_status()
        documents = response.json()
        return {"rag_id": rag_id, "documents": documents}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch RAG documents: {str(e)}")

@app.delete("/project/{project_id}/rag/documents", tags=["Project Management"])
async def delete_rag_documents(project_id: str, delete_request: DeleteRagDocuments, current_user: User = Depends(get_current_user)):
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await db.projects.find_one({"_id": obj_id, "client_id": current_user.client_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    
    if current_user.user_type != UserType.admin and str(obj_id) not in current_user.projects:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")

    if "rag_id" not in project:
        raise HTTPException(status_code=404, detail="RAG collection not configured for this project")

    rag_id = project["rag_id"]
    source_name = delete_request.source_name

    try:
        response = requests.delete(
            f"{LYZR_RAG_API_URL}/{rag_id}/docs/",
            headers={"x-api-key": API_KEY, "accept": "application/json", "Content-Type": "application/json"},
            json=[source_name]
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete RAG documents: {str(e)}")

    await db.projects.update_one(
        {"_id": obj_id},
        {"$pull": {
            "github_links": {"source_name": source_name},
            "repo_analyses": {"source_name": source_name}
        }}
    )
    await db.projects.update_one(
        {"_id": obj_id},
        {"$pull": {"documentation": {"source_name": source_name}}}
    )

    return {"message": f"Deleted documents with source: '{source_name}' from RAG and project"}

@app.put("/project/{project_id}", tags=["Project Management"])
async def update_project(project_id: str, project: ProjectCreate, current_user: User = Depends(get_current_admin_user)):
    result = await db.projects.update_one(
        {"_id": ObjectId(project_id), "client_id": current_user.client_id},
        {"$set": {"name": project.name}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    return {"message": "Project updated successfully"}

@app.get("/project/{project_id}", tags=["Project Management"])
async def get_project(project_id: str, current_user: User = Depends(get_current_user)):
    try:
        project = await db.projects.find_one({"_id": ObjectId(project_id), "client_id": current_user.client_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found or not in your client")
        
        project["_id"] = str(project["_id"])
        if "repo_analyses" in project:
            project["repo_analyses"] = [{"repo_url": analysis["repo_url"]} for analysis in project["repo_analyses"]]
        return project
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

@app.delete("/project/{project_id}", tags=["Project Management"])
async def delete_project(project_id: str, current_user: User = Depends(get_current_admin_user)):
    result = await db.projects.delete_one({"_id": ObjectId(project_id), "client_id": current_user.client_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    return {"message": "Project deleted successfully"}

# Code Operations APIs (No Secret Key)
@app.post("/chat_sessions", tags=["Code Operations"])
async def create_chat_session(
    project_id: str,
    session_type: str,
    current_user: User = Depends(get_current_user),
    project = Depends(check_project_access)
):
    if session_type not in ["search", "generate"]:
        raise HTTPException(status_code=400, detail="Invalid session type. Use 'search' or 'generate'.")
    
    session_data = {
        "user_id": current_user.id,
        "project_id": ObjectId(project_id),
        "type": session_type,
        "agent_session_id": str(uuid4()),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    result = await db.chat_sessions.insert_one(session_data)
    return {"session_id": str(result.inserted_id), "agent_session_id": session_data["agent_session_id"]}

@app.post("/chat_sessions/{session_id}/search", tags=["Code Operations"])
async def search_in_session(session_id: str, query: AgentQuery, current_user: User = Depends(get_current_user)):
    try:
        obj_id = ObjectId(session_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    session = await db.chat_sessions.find_one({"_id": obj_id})
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    if session.get("type") != "search":
        raise HTTPException(status_code=400, detail="This session is not for search operations")

    if str(session["user_id"]) != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this chat session")

    project = await db.projects.find_one({"_id": session["project_id"], "client_id": current_user.client_id})
    if not project or "search_agent_id" not in project:
        raise HTTPException(status_code=404, detail="Project or search agent not configured")

    user_message = {
        "session_id": obj_id,
        "role": "user",
        "content": query.message,
        "timestamp": datetime.utcnow(),
        "type": "search"
    }
    await db.chat_messages.insert_one(user_message)

    payload = {
        "user_id": USER_ID,
        "agent_id": project["search_agent_id"],
        "session_id": session["agent_session_id"],
        "message": query.message
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(LYZR_API_URL, json=payload, headers={"x-api-key": API_KEY})
        assistant_response = response.json()
        assistant_content = assistant_response.get("response", "No response from assistant")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search agent API error: {str(e)}")

    assistant_message = {
        "session_id": obj_id,
        "role": "assistant",
        "content": assistant_content,
        "timestamp": datetime.utcnow(),
        "type": "search"
    }
    assistant_message_result = await db.chat_messages.insert_one(assistant_message)

    await db.chat_sessions.update_one({"_id": obj_id}, {"$set": {"updated_at": datetime.utcnow()}})

    return {
        "id": str(assistant_message_result.inserted_id),
        "session_id": str(obj_id),
        "role": "assistant",
        "content": assistant_content,
        "timestamp": assistant_message["timestamp"],
        "type": "search"
    }

@app.post("/chat_sessions/{session_id}/generate", tags=["Code Operations"])
async def generate_in_session(session_id: str, query: AgentQuery, current_user: User = Depends(get_current_user)):
    try:
        obj_id = ObjectId(session_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    session = await db.chat_sessions.find_one({"_id": obj_id})
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    if session.get("type") != "generate":
        raise HTTPException(status_code=400, detail="This session is not for generate operations")

    if str(session["user_id"]) != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this chat session")

    project = await db.projects.find_one({"_id": session["project_id"], "client_id": current_user.client_id})
    if not project or "generate_agent_id" not in project:
        raise HTTPException(status_code=404, detail="Project or generate agent not configured")

    user_message = {
        "session_id": obj_id,
        "role": "user",
        "content": query.message,
        "timestamp": datetime.utcnow(),
        "type": "generate"
    }
    await db.chat_messages.insert_one(user_message)

    payload = {
        "user_id": USER_ID,
        "agent_id": project["generate_agent_id"],
        "session_id": session["agent_session_id"],
        "message": query.message
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(LYZR_API_URL, json=payload, headers={"x-api-key": API_KEY})
        assistant_response = response.json()
        assistant_content = assistant_response.get("response", "No response from assistant")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generate agent API error: {str(e)}")

    assistant_message = {
        "session_id": obj_id,
        "role": "assistant",
        "content": assistant_content,
        "timestamp": datetime.utcnow(),
        "type": "generate"
    }
    assistant_message_result = await db.chat_messages.insert_one(assistant_message)

    await db.chat_sessions.update_one({"_id": obj_id}, {"$set": {"updated_at": datetime.utcnow()}})

    return {
        "id": str(assistant_message_result.inserted_id),
        "session_id": str(obj_id),
        "role": "assistant",
        "content": assistant_content,
        "timestamp": assistant_message["timestamp"],
        "type": "generate"
    }

# Utility Functions (unchanged)
def chunk_text(text: str, max_tokens: int = 7000) -> List[str]:
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(encoder.decode(chunk_tokens))
    return chunks

# ... (Previous imports and setup remain unchanged)

async def analyze_repository_background(project_id: ObjectId, repo_url: str, source_name: str, pat: Optional[str] = None):
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    try:
        # Clone the repository
        if pat:
            parsed_url = repo_url.replace("https://", f"https://{pat}@")
        else:
            parsed_url = repo_url
        Repo.clone_from(parsed_url, temp_dir)
                
        # Analyze the repository
        analysis_result = await analyzer.analyze_repository(temp_dir)
        result_dict = json.loads(json.dumps(analysis_result, default=str))
        print(f"Analysis completed. Keys in result_dict: {list(result_dict.keys())}")

        # Combine all analysis data into one text block
        combined_text = ""
        if result_dict.get("db_schemas"):
            combined_text += f"DB Schemas:\n{json.dumps(result_dict['db_schemas'])}\n\n"
        if result_dict.get("api_data"):
            combined_text += f"API Data:\n{json.dumps(result_dict['api_data'])}\n\n"
        if result_dict.get("ui_data"):
            combined_text += f"UI Data:\n{json.dumps(result_dict['ui_data'])}\n\n"
        if result_dict.get("rag_data"):
            rag_text = "\n".join(
                str(item.get("text", "") or item.get("content", "") or json.dumps(item))
                for item in result_dict["rag_data"] if isinstance(item, dict)
            )
            combined_text += f"RAG Data:\n{rag_text}\n\n"

        if not combined_text.strip():
            print(f"No text content to chunk for repository {source_name}")
            combined_text = "No analyzable content found in repository."

        print(f"Processing repository {source_name} with {len(combined_text)} characters")
        text_chunks = chunk_text(combined_text)
        chunked_documents = [
            {
                "id_": str(uuid.uuid4()),
                "embedding": None,
                "metadata": {"source": source_name, "repo_url": repo_url, "chunked": True},
                "text": chunk.strip(),
                "excluded_embed_metadata_keys": [],
                "excluded_llm_metadata_keys": []
            }
            for chunk in text_chunks
        ]
        print(f"Added {len(text_chunks)} chunks from {source_name}")
        print(f"Total chunked documents prepared: {len(chunked_documents)}")

        # Store analysis results in the database
        analysis_entry = {
            "repo_url": repo_url,
            "source_name": source_name,
            "analysis_result": result_dict,
            "chunked_documents": [
                {"id": doc["id_"], "source": doc["metadata"]["source"], "repo_url": doc["metadata"]["repo_url"], "text_length": len(doc["text"])}
                for doc in chunked_documents
            ],
            "analyzed_at": datetime.utcnow()
        }
        await db.projects.update_one(
            {"_id": project_id},
            {"$push": {"repo_analyses": analysis_entry}}
        )
        print("Database updated with analysis result and chunk metadata")

        # Fetch project to check RAG status
        project = await db.projects.find_one({"_id": project_id})
        if not project:
            raise Exception("Project not found")
        project_name = project["name"]

        if not chunked_documents:
            print("No chunked documents available for RAG training")
            return  # Exit early if there's nothing to train

        print(f"Preparing to train RAG with {len(chunked_documents)} documents")

        # Handle RAG and agent creation logic
        if "rag_id" not in project:
            # Step 1: Create RAG collection if it doesn't exist
            rag_id = create_rag_collection()
            if not rag_id:
                raise Exception("Failed to create RAG collection")
            print(rag_id)
            # Step 2: Train the new RAG with documents
            if not train_rag(rag_id, chunked_documents):
                raise Exception("Failed to train RAG collection")

            # Step 3: Create agents only for a new RAG
            search_agent = create_agent(rag_id, "search", SEARCH_INSTRUCTIONS, project_name)
            generate_agent = create_agent(rag_id, "generate", GENERATE_INSTRUCTIONS, project_name)

            # Step 4: Validate agent creation
            if not search_agent or not isinstance(search_agent, dict) or "agent_id" not in search_agent:
                raise Exception("Failed to create search agent")
            if not generate_agent or not isinstance(generate_agent, dict) or "agent_id" not in generate_agent:
                raise Exception("Failed to create generate agent")

            # Step 5: Update project with RAG and agent IDs
            update_data = {
                "rag_id": rag_id,
                "search_agent_id": search_agent["agent_id"],
                "generate_agent_id": generate_agent["agent_id"]
            }
            await db.projects.update_one({"_id": project_id}, {"$set": update_data})
            print(f"New RAG created (ID: {rag_id}) and agents initialized")
        else:
            # If RAG exists, just train it with new documents
            rag_id = project["rag_id"]
            if not train_rag(rag_id, chunked_documents):
                raise Exception("Failed to train existing RAG collection")
            print(f"Trained existing RAG (ID: {rag_id}) with new documents")

    except Exception as e:
        print(f"Error in analyze_repository_background: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Repository analysis failed: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Cleanup completed")

# # Utility Functions (Updated for better error handling)
# def create_rag_collection():
#     try:
#         response = requests.post(
#             f"{LYZR_RAG_API_URL}/",
#             headers={"x-api-key": API_KEY},
#             json={
#                 "user_id": USER_ID,
#                 "llm_credential_id": "lyzr_openai",
#                 "embedding_credential_id": "lyzr_openai",
#                 "vector_db_credential_id": "lyzr_weaviate",
#                 "vector_store_provider": "Weaviate [Lyzr]",
#                 "description": "Repository analysis RAG",
#                 "collection_name": f"repo_rag_{int(time.time())}",
#                 "llm_model": "gpt-4o-mini",
#                 "embedding_model": "text-embedding-ada-002"
#             }
#         )
#         if response.status_code != 200:
#             print(f"RAG creation failed with status {response.status_code}: {response.text}")
#             return None
#         data = response.json()
#         rag_id = data.get('id')
#         if not rag_id:
#             print(f"RAG creation response missing 'id': {data}")
#             return None
#         return rag_id
#     except Exception as e:
#         print(f"RAG creation failed: {str(e)}")
#         return None

# def train_rag(rag_id, documents):
#     try:
#         response = requests.post(
#             f"{LYZR_RAG_API_URL}/train/{rag_id}/",
#             headers={"x-api-key": API_KEY},
#             json=documents
#         )
#         if response.status_code != 200:
#             print(f"RAG training failed with status {response.status_code}: {response.text}")
#             return False
#         return True
#     except Exception as e:
#         print(f"RAG training failed: {str(e)}")
#         return False

# def create_agent(rag_id, agent_type, instructions, project_name):
#     try:
#         url = "https://agent-prod.studio.lyzr.ai/v3/agents/template/single-task"
#         headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
#         payload = {
#             "name": f"repo_{project_name}_{agent_type}_agent",
#             "description": f"repo_{project_name}_{agent_type}_agent",
#             "agent_instructions": instructions,
#             "agent_role": f"Agent for code {agent_type}",
#             "llm_credential_id": "lyzr_openai",
#             "provider_id": "OpenAI",
#             "model": "gpt-4o-mini",
#             "temperature": 0.7,
#             "top_p": 0.9,
#             "features": [
#                 {
#                     "type": "KNOWLEDGE_BASE",
#                     "config": {"lyzr_rag": {"base_url": "https://rag-prod.studio.lyzr.ai", "rag_id": rag_id, "rag_name": "SakSoft Code Rag"}}
#                 },
#                 {"type": "SHORT_TERM_MEMORY", "config": {}},
#                 {"type": "LONG_TERM_MEMORY", "config": {}}
#             ],
#             "tools": []
#         }
#         response = requests.post(url, headers=headers, json=payload)
#         if response.status_code != 200:
#             print(f"Agent creation failed for {agent_type} with status {response.status_code}: {response.text}")
#             return None
#         data = response.json()
#         if "agent_id" not in data:
#             print(f"Agent creation response missing 'agent_id' for {agent_type}: {data}")
#             return None
#         return data
#     except Exception as e:
#         print(f"Agent creation failed for {agent_type}: {str(e)}")
#         return None

# ... (Rest of the code remains unchanged)
SEARCH_INSTRUCTIONS = """# Code Repository RAG Assistant Instructions
## SEARCH INSTRUCTIONS
Your TASK is to ASSIST users in finding SPECIFIC code elements within their codebase, using ONLY the information available in the provided code repository RAG. You MUST follow these STEPS:
1. **UNDERSTAND the SEARCH INTENT**: CLARIFY what the user is looking for, such as functions, classes, variables, or patterns.
2. **IDENTIFY SEARCH SCOPE**: DETERMINE the files, directories, or code components to SEARCH within the provided repository.
3. **PERFORM the SEARCH**: UTILIZE appropriate search METHODS on the available RAG data:
   - For TEXT-based SEARCH, IDENTIFY keywords or identifiers within the repository.
   - For SYNTACTIC SEARCH, IDENTIFY CODE structures and patterns in the provided code.
   - For SEMANTIC SEARCH, IDENTIFY FUNCTIONALITY regardless of naming within the repository.
4. **PRESENT RESULTS CONCISELY**: FORMAT your findings with:
   - FILE paths and LINE numbers from the repository
   - CODE snippets with relevant CONTEXT from the actual codebase
   - BRIEF explanations of HOW the code WORKS based on the repository content
5. **PROVIDE INSIGHTS**: OFFER observations on:
   - Code STRUCTURE as it exists in the repository
   - RELATIONSHIPS between components in the actual codebase
   - POSSIBLE improvements or ISSUES based on the available code
6. **SUGGEST NEXT STEPS**: RECOMMEND RELATED searches or areas to EXPLORE within the repository.
7. **KNOWLEDGE LIMITATIONS**: EXPLICITLY STATE when information is not available in the provided repository and AVOID making assumptions about code that cannot be verified in the RAG data.
"""

GENERATE_INSTRUCTIONS = """# Code Repository RAG Assistant Instructions
## SEARCH INSTRUCTIONS
Your TASK is to ASSIST users in finding SPECIFIC code elements within their codebase, using ONLY the information available in the provided code repository RAG. You MUST follow these STEPS:
1. **UNDERSTAND the SEARCH INTENT**: CLARIFY what the user is looking for, such as functions, classes, variables, or patterns.
2. **IDENTIFY SEARCH SCOPE**: DETERMINE the files, directories, or code components to SEARCH within the provided repository.
3. **PERFORM the SEARCH**: UTILIZE appropriate search METHODS on the available RAG data:
   - For TEXT-based SEARCH, IDENTIFY keywords or identifiers within the repository.
   - For SYNTACTIC SEARCH, IDENTIFY CODE structures and patterns in the provided code.
   - For SEMANTIC SEARCH, IDENTIFY FUNCTIONALITY regardless of naming within the repository.
4. **PRESENT RESULTS CONCISELY**: FORMAT your findings with:
   - FILE paths and LINE numbers from the repository
   - CODE snippets with relevant CONTEXT from the actual codebase
   - BRIEF explanations of HOW the code WORKS based on the repository content
5. **PROVIDE INSIGHTS**: OFFER observations on:
   - Code STRUCTURE as it exists in the repository
   - RELATIONSHIPS between components in the actual codebase
   - POSSIBLE improvements or ISSUES based on the available code
6. **SUGGEST NEXT STEPS**: RECOMMEND RELATED searches or areas to EXPLORE within the repository.
7. **KNOWLEDGE LIMITATIONS**: EXPLICITLY STATE when information is not available in the provided repository and AVOID making assumptions about code that cannot be verified in the RAG data.
## GENERATE INSTRUCTIONS
### **EXECUTION STEPS:**
1. **INTERPRET REQUIREMENTS:**  
   - UNDERSTAND the USER REQUIREMENT and the specified PROGRAMMING LANGUAGE.  
   - UTILIZE the provided RAG as the SOLE SOURCE of information.  
   - CHECK the repository for EXISTING CODE related to the request. If FOUND, RETRIEVE and PRESENT it.  
2. **CODE RETRIEVAL & GENERATION:**  
   - PRIORITIZE returning EXISTING CODE from the repository that meets the requirements.
   - If exact code isn't available but can be COMPOSED from EXISTING COMPONENTS in the repository, create a solution using ONLY these components.
   - If generation is necessary, ENSURE it's CONSISTENT with the patterns, naming conventions, and style present in the repository.
   - NEVER introduce external libraries or approaches not evidenced in the repository.
3. **DOCUMENTATION:**  
   - ENSURE responses reference ONLY the code repository's actual documentation.
   - When adding comments, MAINTAIN the SAME STYLE and CONVENTIONS observed in the repository.
4. **CONSTRAINTS COMPLIANCE:**  
   - LIMIT all responses to INFORMATION EXPLICITLY AVAILABLE in the repository.
   - CLEARLY INDICATE when information is incomplete or unavailable in the repository.
   - AVOID speculation about implementation details not present in the RAG data.
5. **REVIEW & VALIDATION:**  
   - VERIFY all responses against the repository data.
   - If asked about functionality not present in the repository, EXPLICITLY STATE that the information is not available rather than generating hypothetical answers.
"""


def create_rag_collection():
    try:
        response = requests.post(
            f"{LYZR_RAG_API_URL}/",
            headers={"x-api-key": API_KEY},
            json={
                "user_id": USER_ID,
                "llm_credential_id": "lyzr_openai",
                "embedding_credential_id": "lyzr_openai",
                "vector_db_credential_id": "lyzr_weaviate",
                "vector_store_provider": "Weaviate [Lyzr]",
                "description": "Repository analysis RAG",
                "collection_name": f"repo_rag_{int(time.time())}",
                "llm_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-ada-002"
            }
        )
        return response.json().get('id')
    except Exception as e:
        print(f"RAG creation failed: {str(e)}")
        return None

def train_rag(rag_id, documents):
    try:
        response = requests.post(
            f"{LYZR_RAG_API_URL}/train/{rag_id}/",
            headers={"x-api-key": API_KEY},
            json=documents
        )
        print("Training response",response)
        return True
    except Exception as e:
        print(f"RAG training failed: {str(e)}")
        return False

def create_agent(rag_id, agent_type, instructions, project_name):
    try:
        url = "https://agent-prod.studio.lyzr.ai/v3/agents/template/single-task"
        headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
        payload = {
            "name": f"repo_{project_name}_{agent_type}_agent",
            "description": f"repo_{project_name}_{agent_type}_agent",
            "agent_instructions": instructions,
            "agent_role": f"Agent for code {agent_type}",
            "llm_credential_id": "lyzr_openai",
            "provider_id": "OpenAI",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "top_p": 0.9,
            "features": [
                {
                    "type": "KNOWLEDGE_BASE",
                    "config": {
                        "lyzr_rag": {
                            "base_url": "https://rag-prod.studio.lyzr.ai",
                            "rag_id": rag_id,
                            "rag_name": "SakSoft Code Rag"
                        }
                    },
                    "priority": 0  # Highest priority for knowledge base
                },
                {
                    "type": "SHORT_TERM_MEMORY",
                    "config": {},
                    "priority": 1  # Medium priority
                },
                {
                    "type": "LONG_TERM_MEMORY",
                    "config": {},
                    "priority": 2  # Lowest priority
                }
            ],
            "tools": []
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            print(f"Agent creation failed for {agent_type} with status {response.status_code}: {response.text}")
            return None
        data = response.json()
        if "agent_id" not in data:
            print(f"Agent creation response missing 'agent_id' for {agent_type}: {data}")
            return None
        print(f"Agent created successfully for {agent_type}: {data}")
        return data
    except Exception as e:
        print(f"Agent creation failed for {agent_type}: {str(e)}")
        return None

# Health Check (No Secret Key)
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy"}