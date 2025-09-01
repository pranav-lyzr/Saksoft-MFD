from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.models.user import UserCreate, UserUpdate, User, UserWithSessionsAndChats, ProjectAssignment, UserType
from app.models.auth import PasswordChange, ResetPassword
from app.dependencies.auth import get_current_user, get_current_admin_user
from app.database import users_collection, chat_sessions_collection, chat_messages_collection, projects_collection
from passlib.context import CryptContext
from bson import ObjectId

# Password Hashing - same as original main.py
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter()

@router.post("/users", response_model=User, tags=["User Management"])
async def create_user(user: UserCreate, current_user: User = Depends(get_current_admin_user)):
    """Create a new user"""
    existing_user = await users_collection.find_one({
        "username": user.username,
        "client_id": current_user.client_id
    })
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Username already registered under this client."
        )
    
    hashed_password = pwd_context.hash(user.password)
    new_user = {
        "username": user.username,
        "hashed_password": hashed_password,
        "user_type": user.user_type,
        "is_active": True,
        "projects": [],
        "client_id": current_user.client_id
    }
    result = await users_collection.insert_one(new_user)
    user_id = result.inserted_id

    # If the user is an admin, assign all existing projects for the client
    if user.user_type == UserType.admin:
        project_ids = await projects_collection.distinct("_id", {"client_id": current_user.client_id})
        if project_ids:
            await users_collection.update_one(
                {"_id": user_id},
                {"$addToSet": {"projects": {"$each": project_ids}}}
            )

    created_user = await users_collection.find_one({"_id": user_id})
    return User(
        id=str(created_user["_id"]),
        username=created_user["username"],
        user_type=created_user["user_type"],
        is_active=created_user["is_active"],
        projects=[str(pid) for pid in created_user.get("projects", [])],
        client_id=created_user["client_id"]
    )

@router.get("/users", response_model=List[User], tags=["User Management"])
async def read_users(current_user: User = Depends(get_current_user)):
    """Get list of users based on current user's permissions"""
    if current_user.user_type == UserType.admin:
        users = await users_collection.find({"client_id": current_user.client_id}).to_list(None)
    elif current_user.user_type == UserType.project_admin:
        project_ids = [ObjectId(pid) for pid in current_user.projects]
        users = await users_collection.find({
            "projects": {"$in": project_ids},
            "client_id": current_user.client_id
        }).to_list(None)
    else:
        users = [await users_collection.find_one({"_id": ObjectId(current_user.id)})]
    
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

@router.get("/users/{user_id}", response_model=UserWithSessionsAndChats, tags=["User Management"])
async def read_user(user_id: str, current_user: User = Depends(get_current_user)):
    """Get detailed user information including chat sessions and messages"""
    try:
        user = await users_collection.find_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found or not in your client")
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    user_model = User(
        id=str(user["_id"]),
        username=user["username"],
        user_type=user["user_type"],
        is_active=user["is_active"],
        projects=[str(pid) for pid in user.get("projects", [])],
        client_id=user["client_id"]
    )
    
    if current_user.user_type == UserType.admin:
        pass
    elif current_user.user_type == UserType.project_admin:
        if not set(current_user.projects).intersection(user_model.projects) and user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to view this user")
    else:
        if user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to view this user")
    
    sessions = await chat_sessions_collection.find({"user_id": user_id}).to_list(None)
    chat_sessions = [
        {
            "id": str(session["_id"]),
            "agent_session_id": session["agent_session_id"],
            "user_id": session["user_id"],
            "project_id": str(session["project_id"]),
            "type": session["type"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"]
        } for session in sessions
    ]
    session_ids = [ObjectId(session["id"]) for session in chat_sessions]
    messages = await chat_messages_collection.find({"session_id": {"$in": session_ids}}).to_list(None)
    chat_messages = [
        {
            "id": str(message["_id"]),
            "session_id": str(message["session_id"]),
            "role": message["role"],
            "content": message["content"],
            "timestamp": message["timestamp"],
            "type": message["type"]
        } for message in messages
    ]
    
    return UserWithSessionsAndChats(
        **user_model.dict(),
        chat_sessions=chat_sessions,
        chat_messages=chat_messages
    )

@router.put("/users/{user_id}", response_model=User, tags=["User Management"])
async def update_user(user_id: str, user_update: UserUpdate, current_user: User = Depends(get_current_admin_user)):
    """Update user information"""
    user = await users_collection.find_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found or not in your client")
    
    update_data = user_update.dict(exclude_unset=True)
    if "password" in update_data:
        update_data["hashed_password"] = pwd_context.hash(update_data.pop("password"))
    if "projects" in update_data:
        update_data["projects"] = [ObjectId(pid) for pid in update_data["projects"]]
    
    await users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": update_data})
    updated_user = await users_collection.find_one({"_id": ObjectId(user_id)})
    return User(**{**updated_user, "id": str(updated_user["_id"]), "projects": [str(pid) for pid in updated_user.get("projects", [])]})

@router.delete("/users/{user_id}", tags=["User Management"])
async def delete_user(user_id: str, current_user: User = Depends(get_current_admin_user)):
    """Delete a user"""
    result = await users_collection.delete_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found or not in your client")
    return {"message": "User deleted successfully"}

@router.post("/users/me/change_password", tags=["User Management"])
async def change_password(password_change: PasswordChange, current_user: User = Depends(get_current_user)):
    """Change current user's password"""
    user = await users_collection.find_one({"_id": ObjectId(current_user.id)})
    if not pwd_context.verify(password_change.current_password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect current password")
    
    hashed_new_password = pwd_context.hash(password_change.new_password)
    await users_collection.update_one(
        {"_id": ObjectId(current_user.id)},
        {"$set": {"hashed_password": hashed_new_password}}
    )
    return {"message": "Password changed successfully"}

@router.post("/users/{user_id}/reset_password", tags=["User Management"])
async def reset_password(user_id: str, reset_data: ResetPassword, current_user: User = Depends(get_current_admin_user)):
    """Reset user's password (admin only)"""
    user = await users_collection.find_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found or not in your client")
    hashed_new_password = pwd_context.hash(reset_data.new_password)
    await users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"hashed_password": hashed_new_password}}
    )
    return {"message": "Password reset successfully"}

@router.post("/users/{user_id}/activate", tags=["User Management"])
async def activate_user(user_id: str, current_user: User = Depends(get_current_admin_user)):
    """Activate a user"""
    user = await users_collection.find_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found or not in your client")
    if user["is_active"]:
        raise HTTPException(status_code=400, detail="User is already active")
    result = await users_collection.update_one(
        {"_id": ObjectId(user_id), "client_id": current_user.client_id},
        {"$set": {"is_active": True}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=500, detail="Failed to activate user")
    return {"message": "User activated successfully"}

@router.post("/users/{user_id}/deactivate", tags=["User Management"])
async def deactivate_user(user_id: str, current_user: User = Depends(get_current_admin_user)):
    """Deactivate a user"""
    user = await users_collection.find_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found or not in your client")
    if not user["is_active"]:
        raise HTTPException(status_code=400, detail="User is already inactive")
    result = await users_collection.update_one(
        {"_id": ObjectId(user_id), "client_id": current_user.client_id},
        {"$set": {"is_active": False}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=500, detail="Failed to deactivate user")
    return {"message": "User deactivated successfully"}

@router.post("/users/{user_id}/assign_project", tags=["User Management"])
async def assign_project(user_id: str, assignment: ProjectAssignment, current_user: User = Depends(get_current_admin_user)):
    """Assign a project to a user"""
    user = await users_collection.find_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found or not in your client")
    
    project = await projects_collection.find_one({"_id": ObjectId(assignment.project_id), "client_id": current_user.client_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    
    # Check if project is already assigned to the user
    if ObjectId(assignment.project_id) in [ObjectId(pid) for pid in user.get("projects", [])]:
        raise HTTPException(status_code=400, detail="Project is already assigned to the user")
    
    result = await users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$addToSet": {"projects": ObjectId(assignment.project_id)}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "Project assigned successfully"}

@router.post("/users/{user_id}/remove_project", tags=["User Management"])
async def remove_project(user_id: str, assignment: ProjectAssignment, current_user: User = Depends(get_current_admin_user)):
    """Remove a project from a user"""
    user = await users_collection.find_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found or not in your client")
    
    result = await users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$pull": {"projects": ObjectId(assignment.project_id)}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found or project not assigned")
    return {"message": "Project removed successfully"}
