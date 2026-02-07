from fastapi import FastAPI
import os
from fastapi.responses import JSONResponse

from utils.error_utils import wrap_call
from .model import *
from .environment import server

app = FastAPI()

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
    result = wrap_call(server.create)
    if isinstance(result, JSONResponse):
        return result
    return result


@app.post("/step")
def step(body: StepRequestBody):
    result = wrap_call(server.step, body.env_id, body.action)
    if isinstance(result, JSONResponse):
        return result
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
    result = wrap_call(server.reset, body.env_id, body.task_id)
    if isinstance(result, JSONResponse):
        return result
    if isinstance(result, dict) and "observation" in result:
        return {
            "observation": result.get("observation"),
            "info": {k: v for k, v in result.items() if k != "observation"},
        }
    return {"observation": result, "info": {}}

@app.post("/close")
def close(body: CloseRequestBody):
    result = wrap_call(server.close, body.env_id)
    if isinstance(result, JSONResponse):
        return result
    return {"closed": bool(result), "env_id": body.env_id}
