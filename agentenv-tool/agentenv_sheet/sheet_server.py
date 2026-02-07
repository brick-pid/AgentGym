"""FastAPI Server."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from utils.error_utils import close_env, wrap_call
from .sheet_environment import sheet_env_server
from .sheet_model import *
from .sheet_utils import debug_flg

app = FastAPI(debug=debug_flg)

def _close_env(env_id: int):
    return close_env(sheet_env_server.env, env_id)


@app.get("/health")
def health():
    return {"status": "ok", "service": "sheet"}


@app.post("/create")
def create(create_query: CreateQuery):
    env = wrap_call(sheet_env_server.create, create_query.task_id)
    if isinstance(env, JSONResponse):
        return env
    return {"env_id": env}


@app.post("/step")
def step(step_query: StepQuery):
    result = wrap_call(sheet_env_server.step, step_query.env_id, step_query.action)
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
    result = wrap_call(sheet_env_server.reset, reset_query.env_id, reset_query.task_id)
    if isinstance(result, JSONResponse):
        return result
    obs = wrap_call(sheet_env_server.observation, reset_query.env_id)
    if isinstance(obs, JSONResponse):
        return obs
    return {"observation": obs, "info": {}}


@app.post("/close")
def close(body: CloseRequestBody):
    return _close_env(body.env_id)
