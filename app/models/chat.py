from pydantic import BaseModel
from app.models.base import BaseModelWithConfig
from datetime import datetime
from typing import List

class ChatSessionCreate(BaseModel):
    pass

class ChatSession(BaseModelWithConfig):
    id: str
    agent_session_id: str
    user_id: str
    project_id: str
    type: str
    created_at: datetime
    updated_at: datetime

class MessageCreate(BaseModel):
    content: str

class Message(BaseModelWithConfig):
    id: str
    session_id: str
    role: str
    content: str
    timestamp: datetime
    type: str

class AgentQuery(BaseModel):
    message: str






