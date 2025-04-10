import os
import re
import glob
import json
import sqlparse
import shutil
import requests
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict
import asyncio
import httpx
import aiofiles

class RepositoryAnalyzer:
    def __init__(self):
        self.api_url = "https://agent-prod.studio.lyzr.ai/v3/inference/chat/"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": "sk-default-yStV4gbpjadbQSw4i7QhoOLRwAs5dEcl"
        }
        self.user_id = "pranav@lyzr.ai"
        self.semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent API calls

    async def _call_lyzr_api_async(self, client: httpx.AsyncClient, agent_id: str, session_id: str, system_prompt: str, message: str) -> str:
        """Asynchronous helper function to call Lyzr API"""
        async with self.semaphore:
            messages = json.dumps([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ])
            payload = {
                "user_id": self.user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "message": messages
            }
            try:
                response = await client.post(self.api_url, headers=self.headers, json=payload)
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                print(f"Request failed: {str(e)}")
                return json.dumps({"error": str(e)})

    async def extract_db_metadata(self, repo_dir: str) -> List[Dict]:
        """Extract database schemas from various file types concurrently"""
        db_files = []
        extensions = ['sql', 'py', 'java', 'cs', 'php', 'rb', 'go', 'ts', 'rs', 'kt', 'swift']
        for ext in extensions:
            db_files.extend(glob.glob(f"{repo_dir}/**/*.{ext}", recursive=True))

        async with httpx.AsyncClient() as client:
            tasks = [self._analyze_db_file(client, file_path, repo_dir) for file_path in db_files]
            db_schemas = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results or exceptions and log errors
        result = []
        for schema in db_schemas:
            if isinstance(schema, Exception):
                print(f"Error in DB metadata extraction: {schema}")
            elif schema is not None:
                result.append(schema)
        return result

    async def _analyze_db_file(self, client: httpx.AsyncClient, file_path: str, repo_dir: str) -> Dict | None:
        """Analyze a single file for database schema asynchronously"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            if len(content) > 100000 or len(content) < 10:
                return None

            ext = os.path.splitext(file_path)[1][1:]
            system_prompt = """You are a database schema analysis expert. Extract:
              - Tables with columns, data types, constraints
              - Primary/foreign keys
              - Indexes
              - Relationships between tables
              - ORM model definitions
              Return in this JSON format:
              {
                  "tables": [{
                      "name": "users",
                      "columns": [
                          {
                              "name": "id",
                              "type": "integer",
                              "primary_key": true,
                              "foreign_key": {
                                  "references_table": null,
                                  "references_column": null
                              }
                          }
                      ],
                      "primary_keys": ["id"],
                      "indexes": [],
                      "purpose": "Stores user information"
                  }],
                  "database_purpose": "Core application database",
                  "database_summary": "Contains user, product, and order data"
              }"""
            user_prompt = f"""Analyze this {ext} file and extract database schema information.
            Convert ORM models/definitions to database tables and relationships.
            Include all data types, constraints, and indexes.
            
            File Content:
            ```{ext}
            {content.strip()}
            ```
            """

            analysis = await self._call_lyzr_api_async(
                client,
                agent_id="67c0a8b08cfac3392e3a3522",
                session_id="67c0a8b08cfac3392e3a3522",
                system_prompt=system_prompt,
                message=user_prompt
            )

            schema_data = json.loads(analysis.replace('```json', '').replace('```', ''))
            tables = schema_data.get("tables", [])
            
            schemas = []
            for table_info in tables:
                columns = [
                    {
                        "name": col.get("name", ""),
                        "type": col.get("type", "unknown"),
                        "nullable": col.get("nullable", True),
                        "primary_key": col.get("primary_key", False),
                        "foreign_key": col.get("foreign_key", None)
                    } for col in table_info.get("columns", [])
                ]
                schema = {
                    "type": "database_schema",
                    "table_name": table_info.get("name", "unknown"),
                    "columns": columns,
                    "column_count": len(columns),
                    "purpose": table_info.get("purpose", ""),
                    "summary": schema_data.get("database_summary", ""),
                    "file_path": os.path.relpath(file_path, repo_dir),
                    "file_name": os.path.basename(file_path),
                    "primary_keys": table_info.get("primary_keys", []),
                    "indexes": table_info.get("indexes", []),
                    "metadata": {
                        "database_purpose": schema_data.get("database_purpose", ""),
                        "orm_framework": self.detect_orm_framework(content),
                        "relationships": self.detect_table_relationships(content),
                        "constraints": self.extract_constraints(columns)
                    }
                }
                schemas.append(schema)
            return schemas[0] if schemas else None  # Return first schema or None

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    async def extract_api_metadata(self, repo_dir: str) -> List[Dict]:
        """Extract API endpoints from various file types concurrently"""
        api_files = []
        extensions = ['py', 'js', 'ts', 'java', 'go', 'php', 'rb', 'jsx', 'tsx', 'cs', 'swift', 'kt', 'rs', 'dart', 'json', 'yaml']
        for ext in extensions:
            api_files.extend(glob.glob(f"{repo_dir}/**/*.{ext}", recursive=True))

        async with httpx.AsyncClient() as client:
            tasks = [self._analyze_api_file(client, file_path, repo_dir) for file_path in api_files]
            api_metadata = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = []
        for metadata in api_metadata:
            if isinstance(metadata, Exception):
                print(f"Error in API metadata extraction: {metadata}")
            elif metadata is not None:
                result.append(metadata)
        return result

    async def _analyze_api_file(self, client: httpx.AsyncClient, file_path: str, repo_dir: str) -> Dict | None:
        """Analyze a single file for API metadata asynchronously"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            if len(content) > 100000 or len(content) < 10:
                return None

            ext = os.path.splitext(file_path)[1][1:]
            system_prompt = """You are an API analysis expert. Extract API endpoints with:
            - method: HTTP method (GET, POST, etc.)
            - route: Full endpoint path
            - parameters: List of path/query parameters
            - description: Brief functionality description
            - request_body: JSON schema if available
            - response_format: Expected response format
            - security: Authentication methods used
            Return endpoints in this JSON format:
            {
                "api_endpoints": [{
                    "method": "GET",
                    "route": "/api/v1/users",
                    "parameters": ["id"],
                    "description": "Get user details",
                    "request_body": null,
                    "response_format": "application/json",
                    "security": ["JWT"]
                }]
            }"""
            user_prompt = f"""Analyze this {ext} file and extract all API endpoints.
            Include both REST and GraphQL endpoints. Preserve exact route paths and parameters.
            
            File Content:
            ```{ext}
            {content.strip()}
            ```
            """

            analysis = await self._call_lyzr_api_async(
                client,
                agent_id="67c1a9818cfac3392e3a3da9",
                session_id="67c1a9818cfac3392e3a3da9",
                system_prompt=system_prompt,
                message=user_prompt
            )

            endpoint_data = json.loads(analysis.replace('```json', '').replace('```', ''))
            endpoints = endpoint_data.get("api_endpoints", [])
            
            if not endpoints:
                return None

            api_entry = {
                "type": "api_endpoint",
                "file_path": os.path.relpath(file_path, repo_dir),
                "file_name": os.path.basename(file_path),
                "language": ext,
                "endpoints": [
                    {
                        "method": ep.get("method", "UNKNOWN"),
                        "route": ep.get("route", ""),
                        "parameters": ep.get("parameters", []),
                        "description": ep.get("description", ""),
                        "request_schema": ep.get("request_body"),
                        "response_format": ep.get("response_format"),
                        "security": ep.get("security", [])
                    } for ep in endpoints
                ],
                "framework": self.detect_api_framework(content),
                "metadata": {
                    "security_schemes": list(set(ep["security"][0] for ep in endpoints if ep.get("security"))),
                    "methods_used": list(set(ep["method"] for ep in endpoints))
                }
            }
            return api_entry

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    async def extract_ui_metadata(self, repo_dir: str) -> List[Dict]:
        """Extract UI component metadata concurrently"""
        ui_files = []
        extensions = ['html', 'htm', 'jsx', 'tsx', 'vue']
        for ext in extensions:
            ui_files.extend(glob.glob(f"{repo_dir}/**/*.{ext}", recursive=True))

        async with httpx.AsyncClient() as client:
            tasks = [self._analyze_ui_file(client, file_path, repo_dir) for file_path in ui_files]
            ui_metadata = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = []
        for metadata in ui_metadata:
            if isinstance(metadata, Exception):
                print(f"Error in UI metadata extraction: {metadata}")
            elif metadata is not None:
                result.append(metadata)
        return result

    async def _analyze_ui_file(self, client: httpx.AsyncClient, file_path: str, repo_dir: str) -> Dict | None:
        """Analyze a single UI file asynchronously"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            if len(content) > 50000 or len(content) < 10:
                return None

            file_ext = os.path.splitext(file_path)[1][1:].lower()
            analysis = await self._call_lyzr_api_async(
                client,
                agent_id="67c0a7d20606a0f24048043d",
                session_id="67c0a7d20606a0f24048043d",
                system_prompt="You are a code analysis assistant specialized in extracting structured metadata from code files.",
                message=self._build_ui_prompt(file_path, content, file_ext)
            )
            analysis_data = json.loads(analysis.replace('```json', '').replace('```', ''))

            ui_entry = {
                "type": "ui_component",
                "file_path": os.path.relpath(file_path, repo_dir),
                "file_name": os.path.basename(file_path),
                "purpose": analysis_data.get('purpose', ''),
                "summary": analysis_data.get('summary', ''),
                "ui_type": file_ext,
                "metadata": {
                    "forms": analysis_data.get('metadata', {}).get('forms', 0),
                    "inputs": analysis_data.get('metadata', {}).get('inputs', []),
                    "components": analysis_data.get('metadata', {}).get('components', []),
                    "tables": analysis_data.get('metadata', {}).get('tables', []),
                    "state_variables": analysis_data.get('metadata', {}).get('state_variables', [])
                }
            }
            return ui_entry

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def _build_ui_prompt(self, file_path: str, content: str, file_type: str) -> str:
        """Build prompt for UI metadata"""
        return f"""Analyze this UI component file and extract the following in JSON format:
          1. purpose: A one-sentence description of the component's purpose
          2. summary: 2-3 sentences summarizing key UI features
          3. metadata:
            - forms: Number and purpose of any forms
            - inputs: Types of input elements (text, select, button, etc.)
            - tables: Description of any data tables
            - components: Child components used
            - state_variables: Any state variables (for React/Vue)
          
          Code:
          ```html
          {content.strip()}
          ```
        """

    async def extract_rag_metadata(self, repo_dir: str) -> List[Dict]:
        """Extract RAG metadata for all files concurrently"""
        if ".git" in repo_dir:
            return []

        all_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(repo_dir)
            if ".git" not in root
            for file in files if not file.startswith('.')
        ]

        async with httpx.AsyncClient() as client:
            tasks = [self._analyze_rag_file(client, file_path, repo_dir) for file_path in all_files]
            rag_metadata = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = []
        for metadata in rag_metadata:
            if isinstance(metadata, Exception):
                print(f"Error in RAG metadata extraction: {metadata}")
            elif metadata is not None:
                result.append(metadata)
        return result

    async def _analyze_rag_file(self, client: httpx.AsyncClient, file_path: str, repo_dir: str) -> Dict | None:
        """Analyze a single file for RAG metadata asynchronously"""
        try:
            if await self._is_binary_file(file_path):
                print(f"Skipping binary file: {file_path}")
                return None

            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = await f.read()

            if len(content) > 50000:  # More conservative size limit
                return {
                    "file_path": os.path.relpath(file_path, repo_dir),
                    "file_name": os.path.basename(file_path),
                    "file_type": os.path.splitext(file_path)[1][1:].lower(),
                    "rag_metadata": {"error": "File too large for analysis"},
                    "analysis_date": datetime.now().isoformat()
                }

            # Enhanced system prompt with strict format requirements
            system_prompt = """You are a code analysis assistant. Return JSON with:
            - Code Structure: {Overall Organization, Modules, Classes, Functions}
            - Dependencies: {Imports, Libraries, Frameworks}
            - Database Interactions: {Schemas, Queries, ORM Usage}
            - API Components: {Endpoints, Request/Response, Authentication}
            - UI Components: {Elements, State Management, Rendering}
            - Variable Naming: {Conventions, Consistency}
            - Design Patterns: {Recognized Patterns}
            - Performance: {Bottlenecks, Optimization}
            - Security: {Practices, Vulnerabilities}
            - Testing: {Methods, Coverage}
            Rules:
            1. Never generate code
            2. Use only alphanumeric characters
            3. No markdown formatting
            4. Keep responses under 2000 tokens
            5. Avoid special characters
            6. Truncate long lines after 200 characters"""

            # Strictly formatted user prompt
            user_prompt = f"""
            FILE_PATH: {os.path.relpath(file_path, repo_dir)}
            FILE_TYPE: {os.path.splitext(file_path)[1][1:].lower()}
            CONTENT: {content.strip()[:2000]}  # Truncate content
            """

            
            
            # API call with timeout and retry
            try:
                rag_response = await self._call_lyzr_api_async(
                    client,
                    agent_id="67c48f268cfac3392e3a48e2",
                    session_id="67c48f268cfac3392e3a48e2",
                    system_prompt=system_prompt,
                    message=user_prompt
                )
            except httpx.HTTPStatusError as e:
                print(f"HTTP error {e.response.status_code} for {file_path}")
                return None
            rag_response=json.loads(rag_response.replace('```json', '').replace('```', ''))
            

            print("rag_response",rag_response)
            

            # try:
            #     parsed_response = json.loads(rag_response)
            # except json.JSONDecodeError:
            #     print(f"Failed to parse JSON for {file_path}")
            #     return None

            return {
                "file_path": os.path.relpath(file_path, repo_dir),
                "file_name": os.path.basename(file_path),
                "file_type": os.path.splitext(file_path)[1][1:].lower(),
                "rag_metadata": rag_response,
                "analysis_date": datetime.now().isoformat()
            }

        except UnicodeDecodeError:
            print(f"Encoding error in {file_path} - skipping")
            return None
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    async def _is_binary_file(self, file_path: str) -> bool:
        """Check if a file is binary using Linux file command"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'file', '--mime-type', '-b', file_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            mime_type = stdout.decode().strip()
            return not mime_type.startswith('text/') and 'json' not in mime_type
        except Exception:
            return True  # Fallback to binary if check fails
        
    async def analyze_repository(self, repo_dir: str) -> Dict:
        """Main asynchronous analysis entry point"""
        print("Analyzing repository...")
        output_base = os.path.join(os.getcwd(), "out")
        os.makedirs(output_base, exist_ok=True)

        repo_name = os.path.basename(repo_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_base, f"{repo_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # Extract metadata concurrently within each method
        db_schemas = await self.extract_db_metadata(repo_dir)
        api_data = await self.extract_api_metadata(repo_dir)
        ui_data = await self.extract_ui_metadata(repo_dir)
        rag_data = await self.extract_rag_metadata(repo_dir)

        # Generate visualizations and summary (unchanged)
        viz_paths = self.visualize_results(db_schemas, api_data, ui_data, output_dir)
        self.create_html_summary(db_schemas, api_data, ui_data, rag_data, os.path.join(output_dir, "visualizations"), output_dir)

        # Save results
        with open(os.path.join(output_dir, "db_schemas.json"), "w") as f:
            json.dump(db_schemas, f, indent=2)
        with open(os.path.join(output_dir, "api_metadata.json"), "w") as f:
            json.dump(api_data, f, indent=2)
        with open(os.path.join(output_dir, "ui_metadata.json"), "w") as f:
            json.dump(ui_data, f, indent=2)
        with open(os.path.join(output_dir, "rag_metadata.json"), "w") as f:
            json.dump(rag_data, f, indent=2)

        return {
            "db_schemas": db_schemas,
            "api_data": api_data,
            "ui_data": ui_data,
            "rag_data": rag_data,
            "summary": {
                "repository": repo_dir,
                "analysis_date": str(datetime.now()),
                "file_counts": {
                    "database_schemas": len(db_schemas),
                    "api_files": len(api_data),
                    "ui_components": len(ui_data),
                    "total_files": len(rag_data)
                }
            },
            "output_dir": output_dir,
            "visualizations": viz_paths
        }

    # Placeholder methods (implement as needed)
    def detect_orm_framework(self, content): pass
    def detect_table_relationships(self, content): pass
    def extract_constraints(self, columns): pass
    def detect_api_framework(self, content): pass
    def visualize_results(self, db_schemas, api_data, ui_data, output_dir): pass
    def create_html_summary(self, db_schemas, api_data, ui_data, rag_data, viz_dir, output_dir): pass


# import tempfile
# from git import Repo
# from bson.objectid import ObjectId
# import json
# import shutil

# async def analyze_repository_background(project_id: ObjectId, repo_url: str):
#     temp_dir = tempfile.mkdtemp()
#     try:
#         Repo.clone_from(repo_url, temp_dir)
#         analyzer = RepositoryAnalyzer()
#         analysis_result = await analyzer.analyze_repository(temp_dir)  # Now awaitable
#         result_dict = json.loads(json.dumps(analysis_result, default=str))
#         # Add your result handling logic here (e.g., save to database)
#         return result_dict
#     except Exception as e:
#         print(f"Error in analyze_repository_background: {str(e)}")
#         raise
#     finally:
#         shutil.rmtree(temp_dir, ignore_errors=True)
