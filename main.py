from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from git import Repo
import uuid
import tempfile
import shutil
from typing import List
import json
import httpx
import time
import requests
from analyzer.core import RepositoryAnalyzer
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime 
import os

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# MongoDB Connection
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://root:example@mongo:27017")
print("MONGO URL",MONGO_URI)
# MONGO_URI = "mongodb://root:example@mongo:27017"
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in the .env file")
client = AsyncIOMotorClient(MONGO_URI)
db = client["mydatabase"]
analyzer = RepositoryAnalyzer()


LYZR_RAG_API_URL = "https://rag-prod.studio.lyzr.ai/v3/rag"
LYZR_AGENT_API_URL = "https://agent-prod.studio.lyzr.ai/v3/agent"
LYZR_API_URL = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
API_KEY = "sk-default-yStV4gbpjadbQSw4i7QhoOLRwAs5dEcl"
USER_ID = "pranav@lyzr.ai"

class ProjectCreate(BaseModel):
    name: str

class GitHubLink(BaseModel):
    github_url: HttpUrl

class CodeQuery(BaseModel):
    message: str

class AgentQuery(BaseModel):
    message: str
    project_id: str

class DocumentationInput(BaseModel):
    text: str

# 1️⃣ Create Project
@app.post("/create_project")
async def create_project(project: ProjectCreate):
    project_collection = db.projects
    new_project = {
        "name": project.name,
        "github_links": [],
        "repo_analyses": []  # Initialize as an empty list for multiple repo analyses
    }
    result = await project_collection.insert_one(new_project)
    return {"project_id": str(result.inserted_id), "message": "Project created successfully"}



# 2️⃣ Add GitHub Link with RAG Analysis
@app.post("/project/{project_id}/repo")
async def add_github_link(project_id: str, link: GitHubLink):
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await db.projects.find_one({"_id": obj_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    github_url_str = str(link.github_url)

    await db.projects.update_one(
        {"_id": obj_id},
        {"$push": {"github_links": github_url_str}}
    )

    # Ensure direct execution instead of background task
    await analyze_repository_background(obj_id, github_url_str)

    return {"message": "GitHub link added successfully. Analysis completed."}


@app.post("/project/{project_id}/documentation")
async def add_documentation(project_id: str, input: DocumentationInput):
    # Convert project_id to ObjectId
    try:
        obj_id = ObjectId(project_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    # Check if project exists
    project = await db.projects.find_one({"_id": obj_id})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate input text
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Save documentation to MongoDB
    await db.projects.update_one(
        {"_id": obj_id},
        {"$push": {"documentation": {"text": input.text, "submitted_at": datetime.utcnow()}}}
    )

    # Chunk the text for RAG training
    text_chunks = chunk_text(input.text)
    chunked_documents = []
    for idx, chunk in enumerate(text_chunks):
        chunked_doc = {
            "id_": str(uuid.uuid4()),
            "embedding": None,
            "metadata": {
                "source": "documentation",
                "chunked": True
            },
            "text": chunk.strip(),
            "excluded_embed_metadata_keys": [],
            "excluded_llm_metadata_keys": []
        }
        chunked_documents.append(chunked_doc)

    # Train the RAG system
    if "rag_id" in project:
        rag_id = project["rag_id"]
        if not train_rag(rag_id, chunked_documents):
            raise HTTPException(status_code=500, detail="Failed to train RAG")
    else:
        # Create a new RAG collection if it doesn't exist
        rag_id = create_rag_collection()
        if not rag_id:
            raise HTTPException(status_code=500, detail="Failed to create RAG collection")
        
        # Train the new RAG with the chunks
        if not train_rag(rag_id, chunked_documents):
            raise HTTPException(status_code=500, detail="Failed to train RAG")
        
        # Create agents for the project
        project_name = project["name"]
        search_agent = create_agent(rag_id, "search", SEARCH_INSTRUCTIONS, project_name)
        generate_agent = create_agent(rag_id, "generate", GENERATE_INSTRUCTIONS, project_name)
        
        if not search_agent or not generate_agent:
            raise HTTPException(status_code=500, detail="Failed to create agents")
        
        # Update project with RAG and agent IDs
        update_data = {
            "rag_id": rag_id,
            "search_agent_id": search_agent.get("agent_id"),
            "generate_agent_id": generate_agent.get("agent_id")
        }
        await db.projects.update_one(
            {"_id": obj_id},
            {"$set": update_data}
        )

    return {"message": "Documentation added and RAG trained successfully"}


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """Chunk text into smaller pieces for RAG training."""
    print(f"Chunking text of length: {len(text)}")
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    print(f"Created {len(chunks)} chunks")
    return chunks

async def analyze_repository_background(project_id: ObjectId, repo_url: str):
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    try:
        # Clone repository
        Repo.clone_from(repo_url, temp_dir)
        analyzer = RepositoryAnalyzer()
        analysis_result = await analyzer.analyze_repository(temp_dir)  # Now awaitable
        result_dict = json.loads(json.dumps(analysis_result, default=str))
        print("Analysis completed. Keys in result_dict:", list(result_dict.keys()))

        # Extract file paths from rag_data
        rag_data = result_dict.get("rag_data", [])
        print(f"Found {len(rag_data)} entries in rag_data")

        chunked_documents = []
        for file_info in rag_data:
            file_path = file_info.get("file_path")
            if not file_path:
                print(f"Skipping entry due to missing file_path: {file_info}")
                continue

            full_path = os.path.join(temp_dir, file_path)
            if not os.path.isfile(full_path):
                print(f"File does not exist or is not a file: {full_path}")
                continue

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    text = f.read()
                print(f"Read {len(text)} characters from {file_path}")
            except Exception as e:
                print(f"Failed to read {full_path}: {str(e)}")
                continue

            text_chunks = chunk_text(text)
            for idx, chunk in enumerate(text_chunks):
                chunked_doc = {
                    "id_": str(uuid.uuid4()),  # Add unique ID
                    "embedding": None,  # Optional, set to None
                    "metadata": {
                        "source": file_path,
                        "chunked": True  # Keep only required metadata fields
                    },
                    "text": chunk.strip(),
                    "excluded_embed_metadata_keys": [],  # Add required empty list
                    "excluded_llm_metadata_keys": []  # Add required empty list
                }
                chunked_documents.append(chunked_doc)
            print(f"Added {len(text_chunks)} chunks from {file_path} to chunked_documents")

        # Log total documents prepared
        print(f"Total chunked documents prepared: {len(chunked_documents)}")

        # Append analysis to database (no need to remove text fields since they aren’t in result_dict)
        print("Updating database with analysis result")
        await db.projects.update_one(
            {"_id": project_id},
            {"$push": {"repo_analyses": {"repo_url": repo_url, "analysis_result": result_dict}}}
        )
        print("Database update successful")

        # Fetch updated project data
        project = await db.projects.find_one({"_id": project_id})
        if not project:
            print("Error: Project not found in database")
            raise Exception("Project not found")
        print("Fetched project data successfully")
        project_name = project["name"]
        print(f"Project name: {project_name}")

        # Train RAG with chunked documents
        if not chunked_documents:
            print("Warning: No chunked documents available for RAG training")
        else:
            print(f"Preparing to train RAG with {len(chunked_documents)} documents")

        if "rag_id" not in project:
            # First repository: create RAG and agents
            print("Creating new RAG collection")
            rag_id = create_rag_collection()
            if not rag_id:
                print("Error: Failed to create RAG collection")
                raise Exception("Failed to create RAG collection")
            print(f"Created RAG collection with ID: {rag_id}")

            # Train RAG
            print(f"Training RAG with {len(chunked_documents)} documents")
            train_rag(rag_id, chunked_documents)
            print("RAG training completed")

            # Create agents
            print("Creating search agent")
            search_agent = create_agent(rag_id, "search", SEARCH_INSTRUCTIONS, project_name)
            print("Creating generate agent")
            generate_agent = create_agent(rag_id, "generate", GENERATE_INSTRUCTIONS, project_name)
            print("Agents created successfully")

            # Update project with new IDs
            update_data = {
                "rag_id": rag_id,
                "search_agent_id": search_agent.get("agent_id"),
                "generate_agent_id": generate_agent.get("agent_id")
            }
            print("Updating project with RAG and agent IDs")
            await db.projects.update_one(
                {"_id": project_id},
                {"$set": update_data}
            )
            print("Project updated with RAG and agent IDs")
        else:
            # Subsequent repository: retrain existing RAG
            rag_id = project["rag_id"]
            print(f"Retraining existing RAG (ID: {rag_id}) with {len(chunked_documents)} documents")
            train_rag(rag_id, chunked_documents)
            print("RAG retraining completed")

    except Exception as e:
        print(f"Error in analyze_repository_background: {str(e)}")
        raise
    finally:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Cleanup completed")


# Agent instructions
SEARCH_INSTRUCTIONS = """# Code Repository RAG Assistant Instructions

## SEARCH INSTRUCTIONS

Your TASK is to ASSIST users in finding SPECIFIC code elements within their codebase, using ONLY the information available in the provided code repository RAG. You MUST follow these STEPS:

1. **UNDERSTAND the SEARCH INTENT**: CLARIFY what the user is looking for, such as functions, classes, variables, or patterns.

2. **IDENTIFY SEARCH SCOPE**: DETERMINE the files, directories, or code components to SEARCH within the provided repository.

3. **PERFORM the SEARCH**: UTILIZE appropriate search METHODS on the available RAG data:
   - For TEXT-based SEARCH, IDENTIFY keywords or identifiers within the repository.
   - For SYNTACTIC SEARCH, IDENTIFY CODE structures and patterns in the provided code.
   - For SEMANTIC SEARCH, IDENTIFY FUNCTIONALITY regardless of naming within the repository.

4. **PRESENT RESULTS CONCISELY**: FORMAT your findings with:
   - FILE paths and LINE numbers from the repository
   - CODE snippets with relevant CONTEXT from the actual codebase
   - BRIEF explanations of HOW the code WORKS based on the repository content

5. **PROVIDE INSIGHTS**: OFFER observations on:
   - Code STRUCTURE as it exists in the repository
   - RELATIONSHIPS between components in the actual codebase
   - POSSIBLE improvements or ISSUES based on the available code

6. **SUGGEST NEXT STEPS**: RECOMMEND RELATED searches or areas to EXPLORE within the repository.

7. **KNOWLEDGE LIMITATIONS**: EXPLICITLY STATE when information is not available in the provided repository and AVOID making assumptions about code that cannot be verified in the RAG data.
"""

GENERATE_INSTRUCTIONS = """# Code Repository RAG Assistant Instructions

## SEARCH INSTRUCTIONS

Your TASK is to ASSIST users in finding SPECIFIC code elements within their codebase, using ONLY the information available in the provided code repository RAG. You MUST follow these STEPS:

1. **UNDERSTAND the SEARCH INTENT**: CLARIFY what the user is looking for, such as functions, classes, variables, or patterns.

2. **IDENTIFY SEARCH SCOPE**: DETERMINE the files, directories, or code components to SEARCH within the provided repository.

3. **PERFORM the SEARCH**: UTILIZE appropriate search METHODS on the available RAG data:
   - For TEXT-based SEARCH, IDENTIFY keywords or identifiers within the repository.
   - For SYNTACTIC SEARCH, IDENTIFY CODE structures and patterns in the provided code.
   - For SEMANTIC SEARCH, IDENTIFY FUNCTIONALITY regardless of naming within the repository.

4. **PRESENT RESULTS CONCISELY**: FORMAT your findings with:
   - FILE paths and LINE numbers from the repository
   - CODE snippets with relevant CONTEXT from the actual codebase
   - BRIEF explanations of HOW the code WORKS based on the repository content

5. **PROVIDE INSIGHTS**: OFFER observations on:
   - Code STRUCTURE as it exists in the repository
   - RELATIONSHIPS between components in the actual codebase
   - POSSIBLE improvements or ISSUES based on the available code

6. **SUGGEST NEXT STEPS**: RECOMMEND RELATED searches or areas to EXPLORE within the repository.

7. **KNOWLEDGE LIMITATIONS**: EXPLICITLY STATE when information is not available in the provided repository and AVOID making assumptions about code that cannot be verified in the RAG data.

## GENERATE INSTRUCTIONS

### **EXECUTION STEPS:**

1. **INTERPRET REQUIREMENTS:**  
   - UNDERSTAND the USER REQUIREMENT and the specified PROGRAMMING LANGUAGE.  
   - UTILIZE the provided RAG as the SOLE SOURCE of information.  
   - CHECK the repository for EXISTING CODE related to the request. If FOUND, RETRIEVE and PRESENT it.  

2. **CODE RETRIEVAL & GENERATION:**  
   - PRIORITIZE returning EXISTING CODE from the repository that meets the requirements.
   - If exact code isn't available but can be COMPOSED from EXISTING COMPONENTS in the repository, create a solution using ONLY these components.
   - If generation is necessary, ENSURE it's CONSISTENT with the patterns, naming conventions, and style present in the repository.
   - NEVER introduce external libraries or approaches not evidenced in the repository.

3. **DOCUMENTATION:**  
   - ENSURE responses reference ONLY the code repository's actual documentation.
   - When adding comments, MAINTAIN the SAME STYLE and CONVENTIONS observed in the repository.

4. **CONSTRAINTS COMPLIANCE:**  
   - LIMIT all responses to INFORMATION EXPLICITLY AVAILABLE in the repository.
   - CLEARLY INDICATE when information is incomplete or unavailable in the repository.
   - AVOID speculation about implementation details not present in the RAG data.

5. **REVIEW & VALIDATION:**  
   - VERIFY all responses against the repository data.
   - If asked about functionality not present in the repository, EXPLICITLY STATE that the information is not available rather than generating hypothetical answers."""  


def create_rag_collection():
    try:
        print("Inside Create Rag Collection Function")
        response = requests.post(
            f"{LYZR_RAG_API_URL}/",
            headers={"x-api-key": API_KEY},
            json={
                "user_id": USER_ID,
                "llm_credential_id": "lyzr_openai",
                "embedding_credential_id": "lyzr_openai",
                "vector_db_credential_id": "lyzr_weaviate",
                "vector_store_provider": "Weaviate [Lyzr]",
                "description": "Repository analysis RAG",
                "collection_name": f"repo_rag_{int(time.time())}",
                "llm_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-ada-002"
            }
        )
        print("Rag creating ",response.json())
        return response.json().get('id')
    except Exception as e:
        print(f"RAG creation failed: {str(e)}")
        return None
    
def train_rag(rag_id, documents):
    try:
        print("TRAIN RAG with documents:", documents)
        response = requests.post(
            f"{LYZR_RAG_API_URL}/train/{rag_id}/",
            headers={"x-api-key": API_KEY},
            json=documents  # Send as {"documents": [list]}
        )
        print("RESPONSE FOR TRAINING:", response.json())
        return True
    except Exception as e:
        print(f"RAG training failed: {str(e)}")
        return False


def create_agent(rag_id, agent_type, instructions, project_name):
    try:
        print("RAG ID",rag_id)
        url = "https://agent-prod.studio.lyzr.ai/v3/agents/template/single-task"  # Correct API URL
        headers = {
            "x-api-key": API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "name": f"repo_{project_name}_{agent_type}_agent",
            "description": f"repo_{project_name}_{agent_type}_agent",
            "agent_instructions": instructions,  
            "agent_role": f"Agent for code {agent_type}",
            "llm_credential_id": "lyzr_openai",
            "provider_id": "OpenAI",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "top_p": 0.9,
            "features": [
                {
                    "type": "KNOWLEDGE_BASE",  # Ensure this is a valid type in API docs
                    "config": {
                        "lyzr_rag": {
                            "base_url": "https://rag-prod.studio.lyzr.ai",
                            "rag_id": rag_id,
                            "rag_name": "SakSoft Code Rag"
                        }
                    },
                    "priority": 0
                },
                {
                    "type": "SHORT_TERM_MEMORY",
                    "config": {},
                    "priority": 0
                },
                {
                "type": "LONG_TERM_MEMORY",
                "config": {},
                "priority": 0
                }
            ],
            "tools": []  # Fixed: should be an array, not `None`
        }
        
        response = requests.post(url, headers=headers, json=payload)
        print("Creating agent response:", payload, response.status_code, response.text)

        if response.status_code == 405:
            print("⚠️ Method Not Allowed: Check if POST is the correct method.")
        elif response.status_code == 403:
            print("❌ Forbidden: Check if your API key is correct.")
        
        return response.json() if response.status_code == 200 else None
    
    except Exception as e:
        print(f"⚠️ Agent creation failed: {str(e)}")
        return None


@app.post("/project/{project_id}/code/search")
async def code_search(query: AgentQuery):
    try:
        project = await db.projects.find_one(
            {"_id": ObjectId(query.project_id)},
            {"search_agent_id": 1}
        )
        if not project or "search_agent_id" not in project:
            raise HTTPException(status_code=404, detail="Agent not configured")

        payload = {
            "user_id": USER_ID,
            # "agent_id": project["search_agent_id"],
            # "session_id": project["search_agent_id"],
            "agent_id": "67c556420606a0f240481e79",
            "session_id": "67c556420606a0f240481e79",
            "message": query.message
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                LYZR_API_URL,
                json=payload,
                headers={"x-api-key": API_KEY}
            )
        return response.json()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/project/{project_id}/code/generate")
async def code_generate(query: AgentQuery):
    try:
        project = await db.projects.find_one(
            {"_id": ObjectId(query.project_id)},
            {"generate_agent_id": 1}
        )
        if not project or "generate_agent_id" not in project:
            raise HTTPException(status_code=404, detail="Agent not configured")

        payload = {
            "user_id": USER_ID,
            # "agent_id": project["generate_agent_id"],
            # "session_id": project["generate_agent_id"],
            "agent_id": "67c55dfe8cfac3392e3a4eb0",
            "session_id": "67c55dfe8cfac3392e3a4eb0",
            "message": query.message
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                LYZR_API_URL,
                json=payload,
                headers={"x-api-key": API_KEY}
            )
        return response.json()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# 5️⃣ Update Project (PUT)
@app.put("/project/{project_id}")
async def update_project(project_id: str, project: ProjectCreate):
    obj_id = ObjectId(project_id)
    result = await db.projects.update_one({"_id": obj_id}, {"$set": {"name": project.name}})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Project not found or no changes made")
    return {"message": "Project updated successfully"}

# 6️⃣ Get Project Details (GET)
@app.get("/project/{project_id}")
async def get_project(project_id: str):
    project = await db.projects.find_one({"_id": ObjectId(project_id)})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project

# 7️⃣ Delete Project (DELETE)
@app.delete("/project/{project_id}")
async def delete_project(project_id: str):
    result = await db.projects.delete_one({"_id": ObjectId(project_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"message": "Project deleted successfully"}



# # 3️⃣ Code Search Endpoint
# @app.post("/code_search")
# async def code_search(query: CodeQuery):
#     payload = {
#         "user_id": USER_ID,
#         "agent_id": "67c556420606a0f240481e79",
#         "session_id": "67c556420606a0f240481e79",
#         "message": query.message
#     }
#     async with httpx.AsyncClient(timeout=30.0) as client:
#         response = await client.post(LYZR_API_URL, json=payload, headers={"x-api-key": API_KEY})
#     if response.status_code != 200:
#         raise HTTPException(status_code=response.status_code, detail=response.text)
#     return response.json()

# # 4️⃣ Code Generation Endpoint
# @app.post("/code_generate")
# async def code_generate(query: CodeQuery):
#     payload = {
#         "user_id": USER_ID,
#         "agent_id": "67c55dfe8cfac3392e3a4eb0",
#         "session_id": "67c55dfe8cfac3392e3a4eb0",
#         "message": query.message
#     }
#     async with httpx.AsyncClient(timeout=30.0) as client:
#         response = await client.post(LYZR_API_URL, json=payload, headers={"x-api-key": API_KEY})
#     if response.status_code != 200:
#         raise HTTPException(status_code=response.status_code, detail=response.text)
#     return response.json()


@app.get("/health")
def health_check():
    return {"status": "healthy"}