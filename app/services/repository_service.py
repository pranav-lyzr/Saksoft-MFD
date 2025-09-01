import tempfile
import shutil
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from git import Repo
from analyzer.core import RepositoryAnalyzer
from app.services.rag_service import RAGService
from app.services.agent_service import AgentService
from app.database import projects_collection
from bson import ObjectId
from app.config import SEARCH_INSTRUCTIONS, GENERATE_INSTRUCTIONS

class RepositoryService:
    def __init__(self):
        self.analyzer = RepositoryAnalyzer()
        self.rag_service = RAGService()
        self.agent_service = AgentService()

    def chunk_text(self, text: str, max_tokens: int = 7000) -> List[str]:
        """Chunk text into smaller pieces based on token count"""
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = encoder.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunks.append(encoder.decode(chunk_tokens))
        return chunks

    async def analyze_repository_background(
        self, 
        project_id: ObjectId, 
        repo_url: str, 
        source_name: str, 
        pat: Optional[str] = None
    ) -> None:
        """Analyze repository in background and update project"""
        temp_dir = tempfile.mkdtemp()
        try:
            print(f"Starting repository analysis for project_id: {project_id}, repo_url: {repo_url}")
            
            # Clone repository
            print("Cloning repository...")
            if pat:
                parsed_url = repo_url.replace("https://", f"https://{pat}@")
            else:
                parsed_url = repo_url
            Repo.clone_from(parsed_url, temp_dir)
            print("Repository cloned successfully")
            
            # Analyze repository
            print("Analyzing repository...")
            analysis_result = await self.analyzer.analyze_repository(temp_dir)
            print(f"Analysis result: {analysis_result}")
            result_dict = json.loads(json.dumps(analysis_result, default=str))
            print("Analysis result serialized")

            # Process analysis result
            print("Processing analysis result...")
            combined_text = ""
            if result_dict.get("db_schemas"):
                combined_text += f"DB Schemas:\n{json.dumps(result_dict['db_schemas'])}\n\n"
            if result_dict.get("api_data"):
                combined_text += f"API Data:\n{json.dumps(result_dict['api_data'])}\n\n"
            if result_dict.get("ui_data"):
                combined_text += f"UI Data:\n{json.dumps(result_dict['ui_data'])}\n\n"
            if result_dict.get("rag_data"):
                rag_text = "\n".join(
                    str(item.get("text", "") or item.get("content", "") or json.dumps(item))
                    for item in result_dict["rag_data"] if isinstance(item, dict)
                )
                combined_text += f"RAG Data:\n{rag_text}\n\n"
            print("Analysis result processed")

            if not combined_text.strip():
                combined_text = "No analyzable content found in repository."
                print("No analyzable content found")

            # Chunk text
            print("Chunking text...")
            text_chunks = self.chunk_text(combined_text)
            print(f"Text chunked: {len(text_chunks)} chunks")
            chunked_documents = [
                {
                    "id_": str(uuid.uuid4()),
                    "embedding": None,
                    "metadata": {"source": source_name, "repo_url": repo_url, "chunked": True},
                    "text": chunk.strip(),
                    "excluded_embed_metadata_keys": [],
                    "excluded_llm_metadata_keys": []
                }
                for chunk in text_chunks
            ]
            print(f"Created {len(chunked_documents)} documents")

            # Update database - Remove existing analysis for this source_name
            print("Updating database...")
            await projects_collection.update_one(
                {"_id": project_id},
                {"$pull": {"repo_analyses": {"source_name": source_name}}}
            )

            analysis_entry = {
                "repo_url": repo_url,
                "source_name": source_name,
                "analysis_result": result_dict,
                "chunked_documents": [
                    {"id": doc["id_"], "source": doc["metadata"]["source"], "repo_url": doc["metadata"]["repo_url"], "text_length": len(doc["text"])}
                    for doc in chunked_documents
                ],
                "analyzed_at": datetime.utcnow()
            }
            await projects_collection.update_one(
                {"_id": project_id},
                {"$push": {"repo_analyses": analysis_entry}}
            )
            print("Database updated")

            # Fetch project
            print("Fetching project...")
            project = await projects_collection.find_one({"_id": project_id})
            if not project:
                raise Exception("Project not found")
            project_name = project["name"]
            print(f"Project fetched: {project_name}")

            # RAG and agent creation
            print("Checking RAG...")
            if not chunked_documents:
                print("No chunked documents to train RAG")
                return

            if "rag_id" not in project:
                print("Creating RAG collection...")
                rag_id = self.rag_service.create_rag_collection()
                if not rag_id:
                    raise Exception("Failed to create RAG collection")
                
                print("Training RAG...")
                if not self.rag_service.train_rag(rag_id, chunked_documents):
                    raise Exception("Failed to train RAG collection")

                print("Creating search agent...")
                search_agent = self.agent_service.create_agent(rag_id, "search", SEARCH_INSTRUCTIONS, project_name)
                generate_agent = self.agent_service.create_agent(rag_id, "generate", GENERATE_INSTRUCTIONS, project_name)

                if not search_agent or not isinstance(search_agent, dict) or "agent_id" not in search_agent:
                    raise Exception("Failed to create search agent")
                if not generate_agent or not isinstance(generate_agent, dict) or "agent_id" not in generate_agent:
                    raise Exception("Failed to create generate agent")

                update_data = {
                    "rag_id": rag_id,
                    "search_agent_id": search_agent["agent_id"],
                    "generate_agent_id": generate_agent["agent_id"]
                }
                await projects_collection.update_one({"_id": project_id}, {"$set": update_data})
            else:
                rag_id = project["rag_id"]
                # Delete existing RAG documents for this source
                try:
                    self.rag_service.delete_rag_documents(rag_id, [source_name])
                except:
                    pass
                if not self.rag_service.train_rag(rag_id, chunked_documents):
                    raise Exception("Failed to train existing RAG collection")
        except Exception as e:
            print(f"Repository analysis failed at {datetime.utcnow()}: {str(e)}")
            raise Exception(f"Repository analysis failed: {str(e)}")
        finally:
            print("Cleaning up temporary directory...")
            shutil.rmtree(temp_dir, ignore_errors=True)






