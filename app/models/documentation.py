from pydantic import BaseModel
from datetime import datetime
from typing import List

class DocumentationInput(BaseModel):
    text: str
    source_name: str

class DocumentationResponse(BaseModel):
    content: str
    timestamp: datetime

class ImpactAnalysisResponse(BaseModel):
    content: str
    timestamp: datetime

class ChangeRequest(BaseModel):
    description: str

class CodeSuggestionRequest(BaseModel):
    context: str

class CodeSuggestionResponse(BaseModel):
    coding_language: str
    suggestions: List[str]






