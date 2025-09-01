from pydantic import BaseModel, HttpUrl
from app.models.base import BaseModelWithConfig
from datetime import datetime
from typing import List, Optional

class ProjectCreate(BaseModel):
    name: str

class GitHubLink(BaseModel):
    github_url: HttpUrl
    source_name: str
    pat: Optional[str] = None

class ProjectResponse(BaseModelWithConfig):
    id: str
    name: str
    client_id: str
    created_by: str
    created_at: datetime
    github_links: List[dict]
    repo_analyses: List[dict]
    documentation: List[dict]

class DeleteRepoByUrlRequest(BaseModel):
    github_url: HttpUrl

class DeleteDocumentationRequest(BaseModel):
    source_name: str

class DeleteRagDocuments(BaseModel):
    source_name: str






