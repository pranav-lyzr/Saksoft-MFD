import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from app.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from app.models.user import User
from app.database import users_collection
from bson import ObjectId
from typing import Optional

# Password Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    @staticmethod
    async def authenticate_user(username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password"""
        user = await users_collection.find_one({"username": username})
        if not user or not AuthService.verify_password(password, user["hashed_password"]):
            return None
        
        if not user["is_active"]:
            return None
        
        return User(
            id=str(user["_id"]),
            username=user["username"],
            user_type=user["user_type"],
            is_active=user["is_active"],
            projects=[str(pid) for pid in user.get("projects", [])],
            client_id=user["client_id"]
        )

    @staticmethod
    async def get_current_user_from_token(token: str) -> Optional[User]:
        """Get current user from JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id: str = payload.get("sub")
            user_type: str = payload.get("user_type")
            client_id: str = payload.get("client_id")
            
            if not user_id or not user_type or not client_id:
                return None
        except jwt.PyJWTError:
            return None
        
        user = await users_collection.find_one({"_id": ObjectId(user_id), "client_id": client_id})
        if not user or not user["is_active"]:
            return None
        
        return User(
            id=str(user["_id"]),
            username=user["username"],
            user_type=user["user_type"],
            is_active=user["is_active"],
            projects=[str(pid) for pid in user.get("projects", [])],
            client_id=user["client_id"]
        )






