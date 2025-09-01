from pydantic import BaseModel
from app.models.base import BaseModelWithConfig
from datetime import datetime
from typing import List, Optional
from enum import Enum

class UserType(str, Enum):
    admin = "admin"
    project_admin = "project_admin"
    developer = "developer"

class UserCreate(BaseModel):
    username: str
    password: str
    user_type: UserType

class UserUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    user_type: Optional[UserType] = None
    is_active: Optional[bool] = None
    projects: Optional[List[str]] = None

class User(BaseModelWithConfig):
    id: str
    username: str
    user_type: UserType
    is_active: bool
    projects: List[str]
    client_id: str

class UserWithSessions(User):
    chat_sessions: List[str]

class UserWithSessionsAndChats(User):
    chat_sessions: List[dict]  # Use dict instead of forward reference
    chat_messages: List[dict]  # Use dict instead of forward reference

class ProjectAssignment(BaseModel):
    project_id: str
