from fastapi import FastAPI

from .utils import register_error_handlers
from .env_wrapper import server
from .model import *

app = FastAPI()
register_error_handlers(app)


@app.get("/health")
def health():
    return {"status": "ok", "service": "alfworld"}


@app.post("/create")
async def create():
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
            "info": {
                k: v for k, v in result.items() if k not in {"observation", "reward", "done"}
            },
        }
    return result


@app.post("/reset")
def reset(body: ResetRequestBody):
    result = server.reset(body.env_id, body.task_id, body.world_type)
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
