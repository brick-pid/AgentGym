# SimpleGym Env Server Contract (v1)

All env servers must expose exactly five public endpoints:

- `GET /health`
- `POST /create`
- `POST /step`
- `POST /reset`
- `POST /close`

## Success schemas

- `GET /health`
  - `{"status": "ok", "service": "...", "version": "v1"}`
- `POST /create`
  - `{"env_id": <int|str>}`
- `POST /step`
  - `{"observation": ..., "reward": <number>, "done": <bool>, "info": {}}`
- `POST /reset`
  - `{"observation": ..., "info": {}}`
- `POST /close`
  - `{"closed": <bool>, "env_id": <int|str>}`

## Error schema

Every error response must follow:

```json
{
  "error": {
    "code": "ENV_NOT_FOUND",
    "message": "...",
    "retryable": false,
    "details": {}
  }
}
```
