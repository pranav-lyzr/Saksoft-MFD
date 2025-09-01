from pydantic import BaseModel
from app.models.base import BaseModelWithConfig
from datetime import datetime
from typing import Optional

class Login(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: str

class PasswordChange(BaseModel):
    current_password: str
    new_password: str

class ResetPassword(BaseModel):
    new_password: str






