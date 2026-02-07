from typing import List

from pydantic import BaseModel


class CreateQuery(BaseModel):
    task_id: int = 0


class StepQuery(BaseModel):
    env_id: int
    action: str


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool


class ResetQuery(BaseModel):
    env_id: int
    task_id: int = 0


class CloseRequestBody(BaseModel):
    env_id: int
