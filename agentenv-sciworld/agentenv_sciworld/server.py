from fastapi import FastAPI
import os

from .utils import register_error_handlers
from .model import *
from .environment import server

app = FastAPI()
register_error_handlers(app)

VISUAL = os.environ.get("VISUAL", "false").lower() == "true"
if VISUAL:
    print("Running in VISUAL mode")
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/health")
def health():
    return {"status": "ok", "service": "sciworld"}


@app.post("/create")
def create():
    result = server.create()
    return result


@app.post("/step")
def step(body: StepRequestBody):
    result = server.step(body.env_id, body.action)
    if isinstance(result, dict) and "done" in result:
        return {
            "observation": result.get("observation"),
            "reward": result.get("reward", 0),
            "done": result.get("done", False),
            "info": {k: v for k, v in result.items() if k not in {"observation", "reward", "done"}},
        }
    return result

@app.post("/reset")
def reset(body: ResetRequestBody):
    result = server.reset(body.env_id, body.task_id)
    if isinstance(result, dict) and "observation" in result:
        return {
            "observation": result.get("observation"),
            "info": {k: v for k, v in result.items() if k != "observation"},
        }
    return {"observation": result, "info": {}}

@app.post("/close")
def close(body: CloseRequestBody):
    result = server.close(body.env_id)
    return {"closed": bool(result), "env_id": body.env_id}
