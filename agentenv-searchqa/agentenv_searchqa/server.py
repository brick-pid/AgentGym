from fastapi import FastAPI, Request
import time
import logging
import os

from .utils import register_error_handlers
from .env_wrapper import searchqa_env_server
from .model import *
from .utils import debug_flg


app = FastAPI(debug=debug_flg)
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

# 自定义中间件
@app.middleware("http")
async def log_request_response_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logging.info(
        f"{request.client.host} - {request.method} {request.url.path} - {response.status_code} - {process_time:.2f} seconds"
    )
    return response

@app.get("/health")
def health():
    return {"status": "ok", "service": "searchqa"}

@app.post("/create")
def create(create_query: CreateQuery):
    env = searchqa_env_server.create(create_query.task_id)
    return {"env_id": env}

@app.post("/step")
def step(step_query: StepQuery):
    observation, reward, done, info = searchqa_env_server.step(step_query.env_id, step_query.action)
    info = info or {}
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.post("/reset")
def reset(reset_query: ResetQuery):
    env_id = reset_query.env_id
    searchqa_env_server.reset(env_id, reset_query.task_id)
    obs = searchqa_env_server.observation(env_id)
    return {"observation": obs, "info": {}}


@app.post("/close")
def close(body: CloseRequestBody):
    result = searchqa_env_server.close(body.env_id)
    return {"closed": bool(result), "env_id": body.env_id}
