from motor.motor_asyncio import AsyncIOMotorClient
from app.config import MONGO_URI

# MongoDB Connection
client = AsyncIOMotorClient(MONGO_URI)
db = client["mydatabase"]

# Database collections
clients_collection = db.clients
users_collection = db.users
projects_collection = db.projects
chat_sessions_collection = db.chat_sessions
chat_messages_collection = db.chat_messages






