from typing import List

from pydantic import BaseModel, ConfigDict, Field

from qbio.types import BaseModelInput


class InferenceInput(BaseModelInput):
    input_files: List[str]


class TargetDiffConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 2021
    num_samples: int = Field(default=10, ge=1, le=100)
    num_steps: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1, le=32)
