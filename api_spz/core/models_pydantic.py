# api_spz/core/models_pydantic.py


from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel, Field, ConfigDict


class TaskStatus(str, Enum):
    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class GenerationArgForm(BaseModel):
    """A Pydantic model to validate the arguments from the client's JSON payload."""
    # Add model_config to ignore extra fields like 'generate_what' from the client
    model_config = ConfigDict(extra="ignore")

    seed: int = 1234
    guidance_scale: float = 5.0
    num_inference_steps: int = 20
    octree_resolution: int = 256
    num_chunks: int = 80
    mesh_simplify: float = 10.0
    apply_texture: bool = False
    texture_size: int = 2048
    output_format: str = "glb"


class GenerationResponse(BaseModel):
    status: TaskStatus
    progress: int = 0
    message: str = ""
    model_url: Optional[str] = None


class StatusResponse(BaseModel):
    status: TaskStatus
    progress: int
    message: str
    busy: bool