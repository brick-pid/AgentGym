import logging
import os

from fastapi import FastAPI

from agentenv_pool import Router, create_app, StepRequestBody, CloseRequestBody
from .environment import SciWorldWrapper
from .model import ResetRequestBody

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s",
)

_parallel_actor = int(os.environ.get("SCIWORLD_PARALLEL_ACTOR", "64"))
_ipc_timeout = float(os.environ.get("SCIWORLD_IPC_TIMEOUT", "120.0"))


def _make_wrapper():
    return SciWorldWrapper()


router = Router(
    parallel_actor=_parallel_actor,
    wrapper_factory=_make_wrapper,
    ipc_timeout=_ipc_timeout,
)


def _register_routes(application: FastAPI, r: Router):
    @application.post("/create")
    async def create():
        return await r.create()

    @application.post("/step")
    async def step(body: StepRequestBody):
        result = await r.step(body.env_id, body.action)
        if isinstance(result, dict) and "done" in result:
            return {
                "observation": result.get("observation"),
                "reward": result.get("reward", 0),
                "done": result.get("done", False),
                "info": {
                    k: v
                    for k, v in result.items()
                    if k not in {"observation", "reward", "done"}
                },
            }
        return result

    @application.post("/reset")
    async def reset(body: ResetRequestBody):
        result = await r.reset(body.env_id, task_id=body.task_id)
        if isinstance(result, dict) and "observation" in result:
            return {
                "observation": result.get("observation"),
                "info": {
                    k: v for k, v in result.items() if k != "observation"
                },
            }
        return {"observation": result, "info": {}}

    @application.post("/close")
    async def close(body: CloseRequestBody):
        result = await r.close(body.env_id)
        return {"closed": bool(result), "env_id": body.env_id}


app = create_app(router, extra_setup=_register_routes)
