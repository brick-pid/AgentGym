from fastapi import FastAPI, Request
import time
import logging
import os
from fastapi.responses import JSONResponse

from utils.error_utils import wrap_call
from .env_wrapper import searchqa_env_server
from .model import *
from .utils import debug_flg


app = FastAPI(debug=debug_flg)

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
    env = wrap_call(searchqa_env_server.create, create_query.task_id)
    if isinstance(env, JSONResponse):
        return env
    return {"env_id": env}

@app.post("/step")
def step(step_query: StepQuery):
    result = wrap_call(searchqa_env_server.step, step_query.env_id, step_query.action)
    if isinstance(result, JSONResponse):
        return result
    observation, reward, done, info = result
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
    result = wrap_call(searchqa_env_server.reset, env_id, reset_query.task_id)
    if isinstance(result, JSONResponse):
        return result
    obs = wrap_call(searchqa_env_server.observation, env_id)
    if isinstance(obs, JSONResponse):
        return obs
    return {"observation": obs, "info": {}}


@app.post("/close")
def close(body: CloseRequestBody):
    result = wrap_call(searchqa_env_server.close, body.env_id)
    if isinstance(result, JSONResponse):
        return result
    return {"closed": bool(result), "env_id": body.env_id}
