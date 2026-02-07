"""
FastAPI Server
"""

import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from utils.error_utils import wrap_call
from .environment import sqlgym_env_server
from .model import *
from .utils import debug_flg

app = FastAPI(debug=debug_flg)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


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
async def health():
    return {"status": "ok", "service": "sqlgym"}


@app.post("/create")
async def create():
    env = wrap_call(sqlgym_env_server.create)
    if isinstance(env, JSONResponse):
        return env

    return {"env_id": env}


@app.post("/step")
async def step(step_query: StepQuery):
    result = wrap_call(sqlgym_env_server.step, step_query.env_id, step_query.action)
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
async def reset(reset_query: ResetQuery):
    result = wrap_call(sqlgym_env_server.reset, reset_query.env_id, reset_query.task_id)
    if isinstance(result, JSONResponse):
        return result
    return {"observation": result, "info": {}}


@app.post("/close")
async def close(body: CloseRequestBody):
    result = wrap_call(sqlgym_env_server.close, body.env_id)
    if isinstance(result, JSONResponse):
        return result
    return {"closed": bool(result), "env_id": body.env_id}
