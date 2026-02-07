from fastapi import FastAPI
from fastapi.responses import JSONResponse

from utils.error_utils import wrap_call
from .env_wrapper import server
from .model import *

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok", "service": "alfworld"}


@app.post("/create")
async def create():
    result = wrap_call(server.create)
    if isinstance(result, JSONResponse):
        return result
    return result


@app.post("/step")
async def step(body: StepRequestBody):
    result = wrap_call(server.step, body.env_id, body.action)
    if isinstance(result, JSONResponse):
        return result
    if isinstance(result, dict) and "done" in result:
        return {
            "observation": result.get("observation"),
            "reward": result.get("reward", 0),
            "done": result.get("done", False),
            "info": {
                k: v for k, v in result.items() if k not in {"observation", "reward", "done"}
            },
        }
    return result


@app.post("/reset")
async def reset(body: ResetRequestBody):
    result = wrap_call(server.reset, body.env_id, body.task_id, body.world_type)
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
