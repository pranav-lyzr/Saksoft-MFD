from pydantic import BaseModel, ConfigDict
from bson import ObjectId
from datetime import datetime
from typing import Any, Optional

class BaseModelWithConfig(BaseModel):
    """Base model with common configuration for MongoDB ObjectId handling"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )

class TimestampMixin(BaseModel):
    """Mixin for models that need timestamp fields"""
    created_at: datetime
    updated_at: Optional[datetime] = None






