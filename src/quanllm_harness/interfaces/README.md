# Interface layer

- `service.py` creates an isolated Harness for every request.
- `cli/` contains one-shot, stdin, JSON and interactive terminal clients.
- `rest_api/` contains FastAPI schemas, JSON/SSE routes, authentication and the Uvicorn entrypoint.
- `web/static/` contains the dependency-free browser client served by the REST application.

No interface may implement its own verification, repair, convergence or tool policy. New clients
must depend on `HarnessService` or the public Harness API and preserve request-scoped evidence.
