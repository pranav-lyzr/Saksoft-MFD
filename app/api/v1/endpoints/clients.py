from fastapi import APIRouter, Depends, HTTPException
from app.models.client import ClientCreate, ClientUpdate, Client
from app.models.user import UserType
from app.dependencies.auth import validate_special_key
from passlib.context import CryptContext
from app.database import clients_collection, users_collection
from bson import ObjectId
from datetime import datetime

# Password Hashing - same as original main.py
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter()

@router.post("/clients", response_model=Client, status_code=201, tags=["Client Management"])
async def create_client(client_data: ClientCreate):
    """Create a new client"""
    if client_data.special_key != "lyzr-saksoft":
        raise HTTPException(status_code=403, detail="Invalid special key")

    existing_client = await clients_collection.find_one({"name": client_data.name})
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
    admin_result = await users_collection.insert_one(admin_data)
    admin_id = admin_result.inserted_id

    new_client = {
        "name": client_data.name,
        "admin_id": admin_id,
        "secret_key": client_data.special_key,
        "created_at": datetime.utcnow()
    }
    client_result = await clients_collection.insert_one(new_client)
    client_id = str(client_result.inserted_id)
    await users_collection.update_one({"_id": admin_id}, {"$set": {"client_id": client_id}})

    return Client(
        id=client_id,
        name=new_client["name"],
        admin_id=str(new_client["admin_id"]),
        created_at=new_client["created_at"]
    )

@router.get("/clients/{client_id}", response_model=Client, tags=["Client Management"])
async def get_client(client_id: str, special_key: str = Depends(validate_special_key)):
    """Get client by ID"""
    try:
        client = await clients_collection.find_one({"_id": ObjectId(client_id)})
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

@router.put("/clients/{client_id}", response_model=Client, tags=["Client Management"])
async def update_client(client_id: str, client_update: ClientUpdate, special_key: str = Depends(validate_special_key)):
    """Update client information"""
    update_data = client_update.dict(exclude_unset=True)
    result = await clients_collection.update_one({"_id": ObjectId(client_id)}, {"$set": update_data})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Client not found or no changes made")
    updated_client = await clients_collection.find_one({"_id": ObjectId(client_id)})
    return Client(
        id=str(updated_client["_id"]),
        name=updated_client["name"],
        admin_id=str(updated_client["admin_id"]),
        created_at=updated_client["created_at"]
    )

@router.delete("/clients/{client_id}", tags=["Client Management"])
async def delete_client(client_id: str, special_key: str = Depends(validate_special_key)):
    """Delete client and all associated data"""
    from app.database import projects_collection, chat_sessions_collection, chat_messages_collection
    
    await users_collection.delete_many({"client_id": client_id})
    await projects_collection.delete_many({"client_id": client_id})
    await chat_sessions_collection.delete_many({"project_id": {"$in": await projects_collection.distinct("_id", {"client_id": client_id})}})
    await chat_messages_collection.delete_many({"session_id": {"$in": await chat_sessions_collection.distinct("_id", {"project_id": {"$in": await projects_collection.distinct("_id", {"client_id": client_id})}})}})
    result = await clients_collection.delete_one({"_id": ObjectId(client_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Client not found")
    return {"message": "Client and all associated data deleted successfully"}
