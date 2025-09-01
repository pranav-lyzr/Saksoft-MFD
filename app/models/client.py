from pydantic import BaseModel
from app.models.base import BaseModelWithConfig
from datetime import datetime

class ClientCreate(BaseModel):
    name: str
    admin_username: str
    admin_password: str
    special_key: str  # Must be "lyzr-saksoft" to create a client

class ClientUpdate(BaseModel):
    name: str

class Client(BaseModelWithConfig):
    id: str
    name: str
    admin_id: str
    created_at: datetime






