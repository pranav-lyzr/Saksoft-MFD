import requests
from typing import Dict, Any, Optional
from app.config import API_KEY, SEARCH_INSTRUCTIONS, GENERATE_INSTRUCTIONS

class AgentService:
    @staticmethod
    def create_agent(rag_id: str, agent_type: str, instructions: str, project_name: str) -> Optional[Dict[str, Any]]:
        """Create a new agent for the specified type"""
        try:
            url = "https://agent-prod.studio.lyzr.ai/v3/agents/template/single-task"
            headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
            
            # Select appropriate instructions based on agent type
            if agent_type == "search":
                agent_instructions = SEARCH_INSTRUCTIONS
            elif agent_type == "generate":
                agent_instructions = GENERATE_INSTRUCTIONS
            else:
                agent_instructions = instructions
            
            payload = {
                "name": f"repo_{project_name}_{agent_type}_agent",
                "description": f"Repository {agent_type} agent for {project_name}",
                "agent_role": f"Agent for code {agent_type}",
                "agent_instructions": agent_instructions,
                "examples": None,
                "tool": "",
                "tool_usage_description": "",
                "provider_id": "OpenAI",
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "top_p": 0.9,
                "llm_credential_id": "lyzr_openai",
                "features": [
                    {
                        "type": "KNOWLEDGE_BASE",
                        "config": {
                            "lyzr_rag": {
                                "base_url": "https://rag-prod.studio.lyzr.ai",
                                "rag_id": rag_id,
                                "rag_name": "Saksoft Code RAG",
                                "params": {
                                    "top_k": 10,
                                    "retrieval_type": "basic",
                                    "score_threshold": 0
                                }
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
                "managed_agents": [],
                "response_format": {"type": "text"},
                "tools": []
            }
            
            response = requests.post(url, headers=headers, json=payload)
            data = response.json()
            print("Response from Creating Agent", data)
            
            if response.status_code != 200:
                return None
                
            if "agent_id" not in data:
                return None
                
            return data
        except Exception as e:
            print(f"Agent creation failed for {agent_type}: {str(e)}")
            return None

    @staticmethod
    async def call_agent(agent_id: str, session_id: str, message: str, timeout: float = 30.0) -> Optional[str]:
        """Call an agent with a message and return the response"""
        import httpx
        
        try:
            from app.config import LYZR_API_URL, USER_ID, API_KEY
            
            payload = {
                "user_id": USER_ID,
                "agent_id": agent_id,
                "session_id": session_id,
                "message": message
            }
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    LYZR_API_URL, 
                    json=payload, 
                    headers={"x-api-key": API_KEY}
                )
                response.raise_for_status()
                assistant_response = response.json()
                return assistant_response.get("response", "No response from assistant")
        except Exception as e:
            print(f"Agent call failed: {str(e)}")
            return None






