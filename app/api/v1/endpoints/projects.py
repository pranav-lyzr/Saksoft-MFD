from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.models.project import ProjectCreate, GitHubLink, ProjectResponse, DeleteRepoByUrlRequest, DeleteDocumentationRequest, DeleteRagDocuments
from app.models.documentation import DocumentationInput
from app.dependencies.auth import get_current_user, get_current_admin_user, get_current_admin_or_project_admin_user, check_project_access
from app.database import projects_collection, users_collection
from app.services.repository_service import RepositoryService
from app.services.rag_service import RAGService
from app.services.agent_service import AgentService
from app.config import SEARCH_INSTRUCTIONS, GENERATE_INSTRUCTIONS
from bson import ObjectId
from datetime import datetime
from app.models.user import UserType
import uuid

router = APIRouter()

@router.get("/projects", response_model=List[ProjectResponse], tags=["Project Management"])
async def list_projects(current_user = Depends(get_current_admin_or_project_admin_user)):
    """List projects based on user permissions"""
    if current_user.user_type.value == "admin":
        projects = await projects_collection.find({"client_id": current_user.client_id}).to_list(None)
    else:
        project_ids = [ObjectId(pid) for pid in current_user.projects]
        projects = await projects_collection.find({
            "_id": {"$in": project_ids},
            "client_id": current_user.client_id
        }).to_list(None)
    
    return [
        ProjectResponse(
            id=str(project["_id"]),
            name=project["name"],
            client_id=project["client_id"],
            created_by=project["created_by"],
            created_at=project["created_at"],
            github_links=project.get("github_links", []),
            repo_analyses=[{"repo_url": analysis["repo_url"]} for analysis in project.get("repo_analyses", [])],
            documentation=project.get("documentation", [])
        )
        for project in projects
    ]

@router.get("/users/{user_id}/projects", response_model=List[ProjectResponse], tags=["Project Management"])
async def get_user_projects(user_id: str, current_user = Depends(get_current_user)):
    """Get projects assigned to a specific user"""
    try:
        user = await users_collection.find_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found or not in your client")
    except:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    if current_user.user_type.value == "admin":
        pass
    elif current_user.user_type.value == "project_admin":
        if not set(current_user.projects).intersection(user.get("projects", [])) and user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to view this user's projects")
    else:
        if user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to view this user's projects")
    
    project_ids = [ObjectId(pid) for pid in user.get("projects", [])]
    projects = await projects_collection.find({
        "_id": {"$in": project_ids},
        "client_id": current_user.client_id
    }).to_list(None)
    
    return [
        ProjectResponse(
            id=str(project["_id"]),
            name=project["name"],
            client_id=project["client_id"],
            created_by=project["created_by"],
            created_at=project["created_at"],
            github_links=project.get("github_links", []),
            repo_analyses=[{"repo_url": analysis["repo_url"], "source_name": analysis["source_name"]} for analysis in project.get("repo_analyses", [])],
            documentation=project.get("documentation", [])
        )
        for project in projects
    ]

@router.get("/users/{user_id}/projects/details", response_model=List[ProjectResponse], tags=["Project Management"])
async def get_user_projects_details(user_id: str, current_user = Depends(get_current_user)):
    """Get detailed project information for a specific user"""
    try:
        user = await users_collection.find_one({"_id": ObjectId(user_id), "client_id": current_user.client_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found or not in your client")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    # Authorization checks
    if current_user.user_type.value == "admin":
        pass
    elif current_user.user_type.value == "project_admin":
        if not set(current_user.projects).intersection(user.get("projects", [])) and user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to view this user's projects")
    else:
        if user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to view this user's projects")

    # Fetch projects assigned to the user
    project_ids = [ObjectId(pid) for pid in user.get("projects", [])]
    projects = await projects_collection.find({
        "_id": {"$in": project_ids},
        "client_id": current_user.client_id
    }).to_list(None)

    if not projects:
        return []

    # Format response with detailed project information
    return [
        ProjectResponse(
            id=str(project["_id"]),
            name=project["name"],
            client_id=project["client_id"],
            created_by=project["created_by"],
            created_at=project["created_at"],
            github_links=project.get("github_links", []),
            repo_analyses=[
                {
                    "repo_url": analysis["repo_url"],
                    "source_name": analysis["source_name"],
                    "analyzed_at": analysis.get("analyzed_at"),
                } for analysis in project.get("repo_analyses", [])
            ],
            documentation=[
                {
                    "text": doc["text"],
                    "source_name": doc["source_name"],
                    "submitted_at": doc.get("submitted_at")
                } for doc in project.get("documentation", [])
            ]
        )
        for project in projects
    ]

@router.post("/create_project", tags=["Project Management"])
async def create_project(project: ProjectCreate, current_user = Depends(get_current_admin_or_project_admin_user)):
    """Create a new project"""
    # Check for existing project with the same name in the client context
    existing_project = await projects_collection.find_one({
        "name": project.name,
        "client_id": current_user.client_id
    })
    if existing_project:
        raise HTTPException(status_code=409, detail="Project name already exists for this client")
    
    new_project = {
        "name": project.name,
        "github_links": [],
        "repo_analyses": [],
        "client_id": current_user.client_id,
        "created_by": current_user.id,
        "created_at": datetime.now()
    }
    result = await projects_collection.insert_one(new_project)
    project_id = result.inserted_id
    
    # Automatically assign project to all admins
    admin_users = await users_collection.find({
        "client_id": current_user.client_id,
        "user_type": UserType.admin
    }).to_list(None)
    
    if admin_users:
        admin_ids = [admin["_id"] for admin in admin_users]
        await users_collection.update_many(
            {"_id": {"$in": admin_ids}},
            {"$addToSet": {"projects": project_id}}
        )
    
    # Assign to project_admin if they created it
    if current_user.user_type.value == "project_admin":
        await users_collection.update_one(
            {"_id": ObjectId(current_user.id)},
            {"$addToSet": {"projects": project_id}}
        )
    
    return {"project_id": str(project_id), "message": "Project created successfully"}

@router.post("/project/{project_id}/repo", tags=["Project Management"])
async def add_github_link(project_id: str, link: GitHubLink, current_user = Depends(get_current_user)):
    """Add or update a GitHub repository link in a project"""
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    # Check project access
    project = await check_project_access(project_id, current_user)

    github_url_str = str(link.github_url)
    source_name = link.source_name

    # Check for duplicate source_name
    existing_source = next((gl for gl in project.get("github_links", []) if gl["source_name"] == source_name and gl["url"] != github_url_str), None)
    if existing_source:
        raise HTTPException(status_code=400, detail=f"Source name '{source_name}' is already used by another GitHub link in this project")

    # Check if the GitHub link already exists
    existing_link = next((gl for gl in project.get("github_links", []) if gl["url"] == github_url_str), None)

    if existing_link:
        # Update existing link
        update_data = {"source_name": source_name}
        if link.pat:
            update_data["pat"] = link.pat
        await projects_collection.update_one(
            {"_id": obj_id, "github_links.url": github_url_str},
            {"$set": {"github_links.$": {"url": github_url_str, **update_data}}}
        )
        # Remove existing repo analyses and RAG documents for the old source_name
        if "rag_id" in project:
            try:
                from app.services.rag_service import RAGService
                rag_service = RAGService()
                rag_service.delete_rag_documents(project["rag_id"], [existing_link["source_name"]])
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to delete existing RAG documents: {str(e)}")
        await projects_collection.update_one(
            {"_id": obj_id},
            {"$pull": {"repo_analyses": {"source_name": existing_link["source_name"]}}}
        )
    else:
        # Add new link
        github_link_data = {"url": github_url_str, "source_name": source_name}
        if link.pat:
            github_link_data["pat"] = link.pat
        await projects_collection.update_one(
            {"_id": obj_id},
            {"$push": {"github_links": github_link_data}}
        )

    # Trigger background repository analysis
    repository_service = RepositoryService()
    await repository_service.analyze_repository_background(obj_id, github_url_str, source_name, link.pat)
    return {"message": f"GitHub link '{source_name}' {'updated' if existing_link else 'added'} successfully. Analysis started."}

@router.delete("/project/{project_id}/repo", tags=["Project Management"])
async def delete_github_link_by_url(project_id: str, delete_request: DeleteRepoByUrlRequest, current_user = Depends(get_current_user)):
    """Delete a GitHub repository link and associated data"""
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    # Check project access
    project = await check_project_access(project_id, current_user)

    # Find the GitHub link by github_url
    github_url_str = str(delete_request.github_url)
    github_link = next((link for link in project.get("github_links", []) if link["url"] == github_url_str), None)
    if not github_link:
        raise HTTPException(status_code=404, detail=f"GitHub link with URL '{github_url_str}' not found")

    source_name = github_link["source_name"]

    # Remove the GitHub link, associated repo analyses, and documentation
    update_result = await projects_collection.update_one(
        {"_id": obj_id},
        {
            "$pull": {
                "github_links": {"url": github_url_str},
                "repo_analyses": {"source_name": source_name},
                "documentation": {"source_name": source_name}
            }
        }
    )

    # Verify that at least one field was modified
    if update_result.modified_count == 0:
        raise HTTPException(status_code=404, detail=f"No data associated with URL '{github_url_str}' or source_name '{source_name}' was found to delete")

    # Delete RAG documents if RAG is configured
    if "rag_id" in project:
        rag_id = project["rag_id"]
        try:
            from app.services.rag_service import RAGService
            rag_service = RAGService()
            rag_service.delete_rag_documents(rag_id, [source_name])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete RAG documents for source_name '{source_name}': {str(e)}")

    return {"message": f"GitHub link with URL '{github_url_str}', associated repo analyses, documentation, and RAG documents for source_name '{source_name}' deleted successfully"}

@router.post("/project/{project_id}/documentation", tags=["Project Management"])
async def add_documentation(project_id: str, input: DocumentationInput, current_user = Depends(get_current_user)):
    """Add or update documentation for a project"""
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await projects_collection.find_one({"_id": obj_id, "client_id": current_user.client_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    
    if current_user.user_type.value != "admin" and str(obj_id) not in current_user.projects:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")

    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Remove existing documentation for this source_name
    await projects_collection.update_one(
        {"_id": obj_id},
        {"$pull": {"documentation": {"source_name": input.source_name}}}
    )

    # Add new documentation
    await projects_collection.update_one(
        {"_id": obj_id},
        {"$push": {"documentation": {"text": input.text, "source_name": input.source_name, "submitted_at": datetime.now()}}}
    )

    # Process documentation for RAG
    repository_service = RepositoryService()
    text_chunks = repository_service.chunk_text(input.text)
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
        # Delete existing RAG documents for this source
        try:
            from app.services.rag_service import RAGService
            rag_service = RAGService()
            rag_service.delete_rag_documents(rag_id, [input.source_name])
        except:
            pass
        if not rag_service.train_rag(rag_id, chunked_documents):
            raise HTTPException(status_code=500, detail="Failed to train RAG")
    else:
        rag_service = RAGService()
        rag_id = rag_service.create_rag_collection()
        if not rag_id:
            raise HTTPException(status_code=500, detail="Failed to create RAG collection")
        
        if not rag_service.train_rag(rag_id, chunked_documents):
            raise HTTPException(status_code=500, detail="Failed to train RAG")
        
        project_name = project["name"]
        agent_service = AgentService()
        search_agent = agent_service.create_agent(rag_id, "search", SEARCH_INSTRUCTIONS, project_name)
        generate_agent = agent_service.create_agent(rag_id, "generate", GENERATE_INSTRUCTIONS, project_name)
        
        if not search_agent or not generate_agent:
            raise HTTPException(status_code=500, detail="Failed to create agents")
        
        update_data = {
            "rag_id": rag_id,
            "search_agent_id": search_agent.get("agent_id"),
            "generate_agent_id": generate_agent.get("agent_id")
        }
        await projects_collection.update_one({"_id": obj_id}, {"$set": update_data})

    return {"message": f"Documentation '{input.source_name}' updated successfully"}

@router.delete("/project/{project_id}/documentation", tags=["Project Management"])
async def delete_documentation_by_source_name(project_id: str, delete_request: DeleteDocumentationRequest, current_user = Depends(get_current_user)):
    """Delete documentation and associated RAG documents"""
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    # Check project access
    project = await check_project_access(project_id, current_user)

    # Verify that the source_name exists in the project's documentation
    documentation_entry = next((doc for doc in project.get("documentation", []) if doc["source_name"] == delete_request.source_name), None)
    if not documentation_entry:
        raise HTTPException(status_code=404, detail=f"Documentation with source_name '{delete_request.source_name}' not found")

    # Remove the documentation entry from MongoDB
    update_result = await projects_collection.update_one(
        {"_id": obj_id},
        {
            "$pull": {
                "documentation": {"source_name": delete_request.source_name}
            }
        }
    )

    # Verify that the documentation was removed
    if update_result.modified_count == 0:
        raise HTTPException(status_code=404, detail=f"No documentation with source_name '{delete_request.source_name}' was found to delete")

    # Delete RAG documents if RAG is configured
    if "rag_id" in project:
        rag_id = project["rag_id"]
        try:
            from app.services.rag_service import RAGService
            rag_service = RAGService()
            rag_service.delete_rag_documents(rag_id, [delete_request.source_name])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete RAG documents for source_name '{delete_request.source_name}': {str(e)}")

    return {"message": f"Documentation and associated RAG documents with source_name '{delete_request.source_name}' deleted successfully"}

@router.get("/project/{project_id}/rag/documents", tags=["Project Management"])
async def get_rag_documents(project_id: str, current_user = Depends(get_current_user)):
    """Get RAG documents for a project"""
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await projects_collection.find_one({"_id": obj_id, "client_id": current_user.client_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    
    if current_user.user_type.value != "admin" and str(obj_id) not in current_user.projects:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")

    if "rag_id" not in project:
        raise HTTPException(status_code=404, detail="RAG collection not configured for this project")

    rag_id = project["rag_id"]
    try:
        from app.services.rag_service import RAGService
        rag_service = RAGService()
        documents = rag_service.get_rag_documents(rag_id)
        if documents is None:
            raise HTTPException(status_code=500, detail="Failed to fetch RAG documents")
        return {"rag_id": rag_id, "documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch RAG documents: {str(e)}")

@router.delete("/project/{project_id}/rag/documents/repository", tags=["Project Management"])
async def delete_repository_rag_documents(project_id: str, delete_request: DeleteRagDocuments, current_user = Depends(get_current_user)):
    """Delete RAG documents for a specific repository"""
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await projects_collection.find_one({"_id": obj_id, "client_id": current_user.client_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    
    if current_user.user_type.value != "admin" and str(obj_id) not in current_user.projects:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")

    if "rag_id" not in project:
        raise HTTPException(status_code=404, detail="RAG collection not configured for this project")

    rag_id = project["rag_id"]
    source_name = delete_request.source_name

    # Validate that the source_name exists in the project
    github_links = project.get("github_links", [])
    source_exists = any(link.get("source_name") == source_name for link in github_links)
    
    if not source_exists:
        raise HTTPException(status_code=404, detail=f"Repository '{source_name}' not found in this project")

    try:
        # Delete from RAG
        from app.services.rag_service import RAGService
        rag_service = RAGService()
        rag_service.delete_rag_documents(rag_id, [source_name])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete RAG documents: {str(e)}")

    # Remove from project's github_links and repo_analyses
    update_result = await projects_collection.update_one(
        {"_id": obj_id},
        {"$pull": {
            "github_links": {"source_name": source_name},
            "repo_analyses": {"source_name": source_name}
        }}
    )

    # Remove from project's documentation if it exists
    await projects_collection.update_one(
        {"_id": obj_id},
        {"$pull": {"documentation": {"source_name": source_name}}}
    )

    return {
        "message": f"Successfully deleted repository '{source_name}' from RAG and project",
        "deleted_source": source_name,
        "rag_id": rag_id
    }

@router.put("/project/{project_id}", tags=["Project Management"])
async def update_project(project_id: str, project: ProjectCreate, current_user = Depends(get_current_admin_user)):
    """Update project information"""
    result = await projects_collection.update_one(
        {"_id": ObjectId(project_id), "client_id": current_user.client_id},
        {"$set": {"name": project.name}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    return {"message": "Project updated successfully"}

@router.get("/project/{project_id}", response_model=ProjectResponse, tags=["Project Management"])
async def get_project(project_id: str, current_user = Depends(get_current_user)):
    """Get project by ID"""
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await projects_collection.find_one({"_id": obj_id, "client_id": current_user.client_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    
    if current_user.user_type.value != "admin" and str(obj_id) not in current_user.projects:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")
        
    return ProjectResponse(
        id=str(project["_id"]),
        name=project["name"],
        client_id=project["client_id"],
        created_by=project["created_by"],
        created_at=project["created_at"],
        github_links=project.get("github_links", []),
        repo_analyses=[{"repo_url": analysis["repo_url"], "source_name": analysis["source_name"]} for analysis in project.get("repo_analyses", [])],
        documentation=project.get("documentation", [])
    )

@router.delete("/project/{project_id}", tags=["Project Management"])
async def delete_project(project_id: str, current_user = Depends(get_current_admin_user)):
    """Delete project and all associated data"""
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    # Delete associated chat sessions and messages
    from app.database import chat_sessions_collection, chat_messages_collection
    session_ids = await chat_sessions_collection.distinct("_id", {"project_id": obj_id})
    await chat_sessions_collection.delete_many({"project_id": obj_id})
    await chat_messages_collection.delete_many({"session_id": {"$in": session_ids}})

    # Remove project from users' project lists
    await users_collection.update_many(
        {"projects": obj_id},
        {"$pull": {"projects": obj_id}}
    )

    # Delete RAG collection if it exists
    project = await projects_collection.find_one({"_id": obj_id})
    if project and "rag_id" in project:
        try:
            from app.services.rag_service import RAGService
            rag_service = RAGService()
            rag_service.delete_rag_collection(project["rag_id"])
        except:
            pass  # Log error but don't fail deletion

    result = await projects_collection.delete_one({"_id": obj_id, "client_id": current_user.client_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    return {"message": "Project and associated data deleted successfully"}
