from fastapi.responses import JSONResponse


def error_response(message: str) -> JSONResponse:
    text = str(message).lower()
    if "not initialized" in text:
        code, status, retryable = "ENV_NOT_READY", 503, True
    elif "has been deleted" in text:
        code, status, retryable = "ENV_CLOSED", 409, False
    elif "has finished" in text:
        code, status, retryable = "EPISODE_FINISHED", 409, False
    elif "out of range" in text:
        code, status, retryable = "TASK_OUT_OF_RANGE", 400, False
    elif "invalid action" in text or "cannot parse action" in text:
        code, status, retryable = "INVALID_ACTION", 400, False
    elif "missing parameter" in text or "please set" in text:
        code, status, retryable = "CONFIG_MISSING", 503, False
    elif "not valid" in text or "not found" in text:
        code, status, retryable = "ENV_NOT_FOUND", 404, False
    else:
        code, status, retryable = "INTERNAL_ERROR", 500, False
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": str(message),
                "retryable": retryable,
                "details": {},
            }
        },
    )


def wrap_call(fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        if isinstance(result, dict) and "error" in result:
            return error_response(result["error"])
        return result
    except Exception as exc:
        return error_response(str(exc))


def close_env(env_dict: dict, env_id: int):
    if env_id in env_dict:
        env_dict.pop(env_id, None)
        return {"closed": True, "env_id": env_id}
    return error_response(f"Env {env_id} not found")

