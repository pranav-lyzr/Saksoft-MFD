from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.models.chat import ChatSession, Message, AgentQuery
from app.models.documentation import DocumentationResponse, ImpactAnalysisResponse, ChangeRequest, CodeSuggestionRequest, CodeSuggestionResponse
from app.dependencies.auth import get_current_user, check_project_access
from app.database import chat_sessions_collection, chat_messages_collection, projects_collection
from app.services.agent_service import AgentService
from app.config import DEFAULT_SEARCH_AGENT_ID, DEFAULT_GENERATE_AGENT_ID, CODE_SUGGESTION_AGENT_ID, USER_ID, API_KEY, LYZR_API_URL
from bson import ObjectId
from datetime import datetime
from uuid import uuid4
import httpx
import json
import ast
import re
import asyncio

router = APIRouter()

@router.post("/chat_sessions", tags=["Code Operations"])
async def create_chat_session(
    project_id: str,
    session_type: str,
    current_user = Depends(get_current_user),
    project = Depends(check_project_access)
):
    """Create a new chat session"""
    if session_type not in ["search", "generate"]:
        raise HTTPException(status_code=400, detail="Invalid session type. Use 'search' or 'generate'.")
    
    session_data = {
        "user_id": current_user.id,
        "project_id": ObjectId(project_id),
        "type": session_type,
        "agent_session_id": str(uuid4()),
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    result = await chat_sessions_collection.insert_one(session_data)
    return {"session_id": str(result.inserted_id), "agent_session_id": session_data["agent_session_id"]}

@router.post("/chat_sessions/{session_id}/search", tags=["Code Operations"])
async def search_in_session(session_id: str, query: AgentQuery, current_user = Depends(get_current_user)):
    """Search in a chat session"""
    try:
        obj_id = ObjectId(session_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    session = await chat_sessions_collection.find_one({"_id": obj_id})
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    if session.get("type") != "search":
        raise HTTPException(status_code=400, detail="This session is not for search operations")

    if str(session["user_id"]) != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this chat session")

    project = await projects_collection.find_one({"_id": session["project_id"], "client_id": current_user.client_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    search_agent_id = project.get("search_agent_id", DEFAULT_SEARCH_AGENT_ID)
    payload = {
        "user_id": USER_ID,
        "agent_id": search_agent_id,
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

    user_message = {
        "session_id": obj_id,
        "role": "user",
        "content": query.message,
        "timestamp": datetime.now(),
        "type": "search"
    }
    await chat_messages_collection.insert_one(user_message)

    assistant_message = {
        "session_id": obj_id,
        "role": "assistant",
        "content": assistant_content,
        "timestamp": datetime.now(),
        "type": "search"
    }
    assistant_message_result = await chat_messages_collection.insert_one(assistant_message)

    await chat_sessions_collection.update_one({"_id": obj_id}, {"$set": {"updated_at": datetime.now()}})
    return {
        "id": str(assistant_message_result.inserted_id),
        "session_id": str(obj_id),
        "role": "assistant",
        "content": assistant_content,
        "timestamp": assistant_message["timestamp"],
        "type": "search"
    }

@router.post("/chat_sessions/{session_id}/generate", tags=["Code Operations"])
async def generate_in_session(session_id: str, query: AgentQuery, current_user = Depends(get_current_user)):
    """Generate in a chat session"""
    try:
        obj_id = ObjectId(session_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    session = await chat_sessions_collection.find_one({"_id": obj_id})
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    if session.get("type") != "generate":
        raise HTTPException(status_code=400, detail="This session is not for generate operations")

    if str(session["user_id"]) != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this chat session")

    project = await projects_collection.find_one({"_id": session["project_id"], "client_id": current_user.client_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    generate_agent_id = project.get("generate_agent_id", DEFAULT_GENERATE_AGENT_ID)
    payload = {
        "user_id": USER_ID,
        "agent_id": generate_agent_id,
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

    user_message = {
        "session_id": obj_id,
        "role": "user",
        "content": query.message,
        "timestamp": datetime.utcnow(),
        "type": "generate"
    }
    await chat_messages_collection.insert_one(user_message)

    assistant_message = {
        "session_id": obj_id,
        "role": "assistant",
        "content": assistant_content,
        "timestamp": datetime.utcnow(),
        "type": "generate"
    }
    assistant_message_result = await chat_messages_collection.insert_one(assistant_message)

    await chat_sessions_collection.update_one({"_id": obj_id}, {"$set": {"updated_at": datetime.utcnow()}})
    return {
        "id": str(assistant_message_result.inserted_id),
        "session_id": str(obj_id),
        "role": "assistant",
        "content": assistant_content,
        "timestamp": assistant_message["timestamp"],
        "type": "generate"
    }

@router.post("/project/{project_id}/technical_documentation", tags=["Code Operations"])
async def generate_technical_documentation(project_id: str, current_user = Depends(get_current_user)):
    """Generate technical documentation for a project"""
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await projects_collection.find_one({"_id": obj_id, "client_id": current_user.client_id})

    print("Project Details ", project)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    
    if current_user.user_type.value != "admin" and str(obj_id) not in current_user.projects:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")

    # Use project's generate_agent_id if available, otherwise use default
    from app.config import DEFAULT_CODE_DOCUMENTATION_ID
    generate_agent_id = project.get("generate_agent_id", DEFAULT_CODE_DOCUMENTATION_ID)

    # Create a chat session for documentation generation
    session_data = {
        "user_id": current_user.id,
        "project_id": obj_id,
        "type": "technical",
        "agent_session_id": str(uuid4()),
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    session_result = await chat_sessions_collection.insert_one(session_data)
    session_id = session_result.inserted_id

    # Define the prompt for technical documentation
    prompt = """Generate a comprehensive technical documentation for the project based on the repository data in RAG. Include the following sections:
    - Project Overview
    - High-level structure of repositories
    - Key components and their role
    - Code flow and data interactions
    - Dependencies and integration
    - Architectural patterns and design decisions
    - Interfaces and data structures
    - Summary
    - Any other points helpful for developers
    Ensure all information is derived solely from the RAG data and follows the repository's conventions."""

    user_message = {
        "session_id": session_id,
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now(),
        "type": "technical"
    }
    await chat_messages_collection.insert_one(user_message)

    print("Agent ID", generate_agent_id)
    payload = {
        "user_id": USER_ID,
        "agent_id": generate_agent_id,
        "session_id": session_data["agent_session_id"],
        "message": prompt
    }

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            print(f"Technical documentation API request payload: {payload}")
            response = await client.post(LYZR_API_URL, json=payload, headers={"x-api-key": API_KEY})
            print(f"Technical documentation API response status: {response.status_code}")
            print(f"Technical documentation API response: {response.text}")
        assistant_response = response.json()
        assistant_content = assistant_response.get("response", "No response from assistant")
    except Exception as e:
        print(f"Technical documentation API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generate agent API error: {str(e)}. Please try again later or contact Lyzr support.")

    assistant_message = {
        "session_id": session_id,
        "role": "assistant",
        "content": assistant_content,
        "timestamp": datetime.utcnow(),
        "type": "technical"
    }
    assistant_message_result = await chat_messages_collection.insert_one(assistant_message)

    await chat_sessions_collection.update_one({"_id": session_id}, {"$set": {"updated_at": datetime.utcnow()}})

    return {
        "documentation": assistant_content
    }

@router.post("/project/{project_id}/impact_analysis", tags=["Code Operations"])
async def generate_impact_analysis(project_id: str, change_request: ChangeRequest, current_user = Depends(get_current_user)):
    """Generate impact analysis report for a project"""
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await projects_collection.find_one({"_id": obj_id, "client_id": current_user.client_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or not in your client")
    
    if current_user.user_type.value != "admin" and str(obj_id) not in current_user.projects:
        raise HTTPException(status_code=403, detail="Not authorized to access this project")

    # Use project's generate_agent_id if available, otherwise use default
    from app.config import DEFAULT_IMPACT_ANALYSIS_ID
    generate_agent_id = project.get("generate_agent_id", DEFAULT_IMPACT_ANALYSIS_ID)

    # Create a chat session for impact analysis
    session_data = {
        "user_id": current_user.id,
        "project_id": obj_id,
        "type": "impact",
        "agent_session_id": str(uuid4()),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    session_result = await chat_sessions_collection.insert_one(session_data)
    session_id = session_result.inserted_id

    # Define the prompt for impact analysis
    prompt = f"""Generate an impact analysis report for the following change in the project: '{change_request.description}'. Base the analysis solely on the repository data in RAG. Include the following sections:
    - Summary
    - Technical Impact (affected components, internal and external dependencies, database changes, API changes, etc.)
    - Testing Requirements (unit tests, integration tests, regression tests)
    - Security Considerations
    - Performance Impact (memory, CPU, network, database, etc.)
    - Deployment Considerations
    - Rollback Plan
    - Estimated Effort (development hours, testing hours, complexity score, etc.)
    - Any Recommendations
    Ensure all information is derived solely from the RAG data and follows the repository's conventions."""

    user_message = {
        "session_id": session_id,
        "role": "user",
        "content": prompt,
        "timestamp": datetime.utcnow(),
        "type": "impact"
    }
    await chat_messages_collection.insert_one(user_message)

    payload = {
        "user_id": USER_ID,
        "agent_id": generate_agent_id,
        "session_id": session_data["agent_session_id"],
        "message": prompt
    }

    async def make_api_call():
        async with httpx.AsyncClient(timeout=600.0) as client:
            print(f"Impact analysis API request payload: {payload}")
            response = await client.post(LYZR_API_URL, json=payload, headers={"x-api-key": API_KEY})
            print(f"Impact analysis API response status: {response.status_code}")
            print(f"Impact analysis API response headers: {dict(response.headers)}")
            print(f"Impact analysis API response: {response.text}")
            if response.status_code == 500:
                raise httpx.HTTPStatusError(
                    message=f"API returned 500: {response.text}",
                    request=response.request,
                    response=response
                )
            response.raise_for_status()
            return response

    try:
        response = await make_api_call()
        assistant_response = response.json()
        assistant_content = assistant_response.get("response", "No response from assistant")
    except httpx.HTTPStatusError as e:
        print(f"Impact analysis API failed after retries: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate impact analysis due to an issue with the Lyzr API (litellm error). Please try again later or contact Lyzr support."
        )
    except Exception as e:
        print(f"Impact analysis API error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Generate agent API error: {str(e)}. Please try again later or contact Lyzr support."
        )

    assistant_message = {
        "session_id": session_id,
        "role": "assistant",
        "content": assistant_content,
        "timestamp": datetime.utcnow(),
        "type": "impact"
    }
    assistant_message_result = await chat_messages_collection.insert_one(assistant_message)

    await chat_sessions_collection.update_one({"_id": session_id}, {"$set": {"updated_at": datetime.utcnow()}})

    return {
        "impact_analysis": assistant_content
    }

@router.get("/project/{project_id}/technical_documentation", response_model=List[DocumentationResponse], tags=["Code Operations"])
async def get_technical_documentation(project_id: str, current_user = Depends(get_current_user)):
    """Get technical documentation for a project"""
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    # Check project access and existence
    project = await check_project_access(project_id, current_user)
    
    # Find chat sessions for technical documentation created by the current user
    sessions = await chat_sessions_collection.find({
        "project_id": obj_id,
        "type": "technical",
        "user_id": current_user.id
    }).to_list(None)
    if not sessions:
        raise HTTPException(status_code=404, detail="No technical documentation found for this project created by you")

    session_ids = [session["_id"] for session in sessions]
    
    # Fetch assistant messages with type "technical"
    messages = await chat_messages_collection.find({
        "session_id": {"$in": session_ids},
        "role": "assistant",
        "type": "technical"
    }).to_list(None)

    if not messages:
        raise HTTPException(status_code=404, detail="No technical documentation content found for this project created by you")

    return [
        DocumentationResponse(
            content=message["content"],
            timestamp=message["timestamp"]
        )
        for message in messages
    ]

@router.get("/project/{project_id}/impact_analysis", response_model=List[ImpactAnalysisResponse], tags=["Code Operations"])
async def get_impact_analysis(project_id: str, current_user = Depends(get_current_user)):
    """Get impact analysis reports for a project"""
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    # Check project access and existence
    project = await check_project_access(project_id, current_user)
    
    # Find chat sessions for impact analysis created by the current user
    sessions = await chat_sessions_collection.find({
        "project_id": obj_id,
        "type": "impact",
        "user_id": current_user.id
    }).to_list(None)
    if not sessions:
        raise HTTPException(status_code=404, detail="No impact analysis reports found for this project created by you")

    session_ids = [session["_id"] for session in sessions]
    
    # Fetch assistant messages with type "impact"
    messages = await chat_messages_collection.find({
        "session_id": {"$in": session_ids},
        "role": "assistant",
        "type": "impact"
    }).to_list(None)

    if not messages:
        raise HTTPException(status_code=404, detail="No impact analysis content found for this project created by you")

    return [
        ImpactAnalysisResponse(
            content=message["content"],
            timestamp=message["timestamp"]
        )
        for message in messages
    ]

@router.post("/code_suggestion", response_model=CodeSuggestionResponse, tags=["Code Operations"])
async def code_suggestion(request: CodeSuggestionRequest, current_user = Depends(get_current_user)):
    """Get code suggestions based on context"""
    # Define the prompt for code suggestions
    prompt = f"""
    ```
    {request.context}
    ```
    """

    payload = {
        "user_id": USER_ID,
        "agent_id": CODE_SUGGESTION_AGENT_ID,
        "session_id": CODE_SUGGESTION_AGENT_ID,  # Temporary session ID for Lyzr API
        "message": prompt
    }
    
    async def make_api_call():
        async with httpx.AsyncClient(timeout=60.0) as client:
            print(f"Code suggestion API request payload: {payload}")
            response = await client.post(LYZR_API_URL, json=payload, headers={"x-api-key": API_KEY})
            print(f"Code suggestion API response status: {response.status_code}")
            print(f"Code suggestion API response headers: {dict(response.headers)}")
            print(f"Code suggestion API response: {response.text}")
            if response.status_code == 500:
                raise httpx.HTTPStatusError(
                    message=f"API returned 500: {response.text}",
                    request=response.request,
                    response=response
                )
            response.raise_for_status()
            return response

    last_exception = None
    for attempt in range(3):
        try:
            response = await make_api_call()
            assistant_response = response.json()
            assistant_content = assistant_response.get("response", {})
            
            # Robustly parse assistant_content if it's a JSON-encoded string
            if isinstance(assistant_content, str):
                try:
                    # Try normal parsing
                    assistant_content = json.loads(assistant_content)
                except json.JSONDecodeError:
                    # Try to clean up common escape issues and parse again
                    try:
                        cleaned = assistant_content.replace("\n", "").replace("\t", "").replace("\\\"", '"').replace("\\'", "'")
                        assistant_content = json.loads(cleaned)
                    except Exception as e1:
                        # Try ast.literal_eval as a last resort
                        try:
                            assistant_content = ast.literal_eval(assistant_content)
                        except Exception as e2:
                            # Fallback: Try to extract coding_language and suggestions manually
                            print("Failed to robustly parse assistant response as JSON", e1, e2)
                            print("Raw response string:", repr(assistant_content))
                            try:
                                # Extract coding_language
                                lang_match = re.search(r'"coding_language"\\s*:\\s*"([^"]+)"', assistant_content)
                                suggestions_match = re.search(r'"suggestions"\\s*:\\s*\[(.*)\]\\s*}', assistant_content, re.DOTALL)
                                if lang_match and suggestions_match:
                                    coding_language = lang_match.group(1)
                                    # Extract suggestions as a list of strings
                                    suggestions_raw = suggestions_match.group(1)
                                    # Split by ",\n        " but keep inner quotes
                                    suggestions = re.findall(r'"(.*?)"', suggestions_raw, re.DOTALL)
                                    assistant_content = {
                                        "coding_language": coding_language,
                                        "suggestions": suggestions
                                    }
                                else:
                                    raise ValueError("Could not extract coding_language or suggestions")
                            except Exception as e3:
                                print("Manual extraction also failed", e3)
                                last_exception = HTTPException(
                                    status_code=500,
                                    detail="Invalid response format from Lyzr API. Please try again or contact Lyzr support."
                                )
                                continue
            
            # Validate response structure
            if not isinstance(assistant_content, dict) or "coding_language" not in assistant_content or "suggestions" not in assistant_content:
                print(f"Invalid response structure: {assistant_content}")
                last_exception = HTTPException(
                    status_code=500,
                    detail="Lyzr API returned an invalid response structure. Please try again or contact Lyzr support."
                )
                continue

            coding_language = assistant_content["coding_language"]
            suggestions = assistant_content["suggestions"]

            # Validate suggestions
            if not isinstance(suggestions, list) or not (3 <= len(suggestions) <= 5) or not all(isinstance(s, str) for s in suggestions):
                print(f"Invalid suggestions format or count: {suggestions}")
                last_exception = HTTPException(
                    status_code=500,
                    detail="Lyzr API returned invalid or insufficient suggestions. Please try again or contact Lyzr support."
                )
                continue

            return CodeSuggestionResponse(
                coding_language=coding_language,
                suggestions=suggestions
            )
        except httpx.HTTPStatusError as e:
            print(f"Code suggestion API failed on attempt {attempt+1}: {str(e)}")
            last_exception = HTTPException(
                status_code=500,
                detail="Failed to generate code suggestions due to an issue with the Lyzr API (possible litellm error). Please try again later or contact Lyzr support."
            )
        except Exception as e:
            print(f"Code suggestion API error on attempt {attempt+1}: {str(e)}")
            last_exception = HTTPException(
                status_code=500,
                detail=f"Generate agent API error: {str(e)}. Please try again later or contact Lyzr support."
            )
        if attempt < 2:
            await asyncio.sleep(1)  # Optional: wait before retrying

    # If all attempts failed
    if last_exception:
        raise last_exception
    raise HTTPException(
        status_code=500,
        detail="Failed to generate code suggestions after 3 attempts. Please try again later or contact Lyzr support."
    )
