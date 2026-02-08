from fastapi import Request
from fastapi.responses import JSONResponse


class EnvError(Exception):
    """Base class for all environment errors."""
    code: str = "INTERNAL_ERROR"
    status: int = 500
    retryable: bool = False

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class EnvNotReadyError(EnvError):
    code = "ENV_NOT_READY"
    status = 503
    retryable = True


class EnvClosedError(EnvError):
    code = "ENV_CLOSED"
    status = 409


class EpisodeFinishedError(EnvError):
    code = "EPISODE_FINISHED"
    status = 409


class TaskOutOfRangeError(EnvError):
    code = "TASK_OUT_OF_RANGE"
    status = 400


class InvalidActionError(EnvError):
    code = "INVALID_ACTION"
    status = 400


class ConfigMissingError(EnvError):
    code = "CONFIG_MISSING"
    status = 503


class EnvNotFoundError(EnvError):
    code = "ENV_NOT_FOUND"
    status = 404


async def env_error_handler(request: Request, exc: EnvError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "retryable": exc.retryable,
                "details": {},
            }
        },
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(exc),
                "retryable": False,
                "details": {},
            }
        },
    )


def register_error_handlers(app):
    app.add_exception_handler(EnvError, env_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)
