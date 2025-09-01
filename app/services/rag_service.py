import requests
import time
from typing import List, Dict, Any, Optional
from app.config import LYZR_RAG_API_URL, API_KEY, USER_ID

class RAGService:
    @staticmethod
    def create_rag_collection() -> Optional[str]:
        """Create a new RAG collection"""
        try:
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
            print("Response JSON for training JSON", response.json())
            return response.json().get('id')
        except Exception as e:
            print(f"RAG creation failed: {str(e)}")
            return None

    @staticmethod
    def train_rag(rag_id: str, documents: List[Dict[str, Any]]) -> bool:
        """Train RAG collection with documents"""
        try:
            response = requests.post(
                f"{LYZR_RAG_API_URL}/train/{rag_id}/",
                headers={"x-api-key": API_KEY},
                json=documents
            )
            return response.status_code == 200
        except Exception as e:
            print(f"RAG training failed: {str(e)}")
            return False

    @staticmethod
    def delete_rag_documents(rag_id: str, source_names: List[str]) -> bool:
        """Delete RAG documents by source names"""
        try:
            response = requests.delete(
                f"{LYZR_RAG_API_URL}/{rag_id}/docs/",
                headers={"x-api-key": API_KEY, "Content-Type": "application/json"},
                json=source_names
            )
            return response.status_code == 200
        except Exception as e:
            print(f"RAG document deletion failed: {str(e)}")
            return False

    @staticmethod
    def get_rag_documents(rag_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get all documents from RAG collection"""
        try:
            response = requests.get(
                f"{LYZR_RAG_API_URL}/documents/{rag_id}/",
                headers={"x-api-key": API_KEY, "accept": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Failed to fetch RAG documents: {str(e)}")
            return None

    @staticmethod
    def delete_rag_collection(rag_id: str) -> bool:
        """Delete entire RAG collection"""
        try:
            response = requests.delete(
                f"{LYZR_RAG_API_URL}/{rag_id}/",
                headers={"x-api-key": API_KEY}
            )
            return response.status_code == 200
        except Exception as e:
            print(f"RAG collection deletion failed: {str(e)}")
            return False






