import os
from typing import Dict, Any, Optional
from urllib.parse import urljoin

import requests
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

# Mutable base URL so a GUI or caller can override prior to launching the server.
LMSTUDIO_BASE: str = os.getenv("LMSTUDIO_BASE", "http://127.0.0.1:1234/v1")

def set_lmstudio_base(base_url: str) -> str:
    """Update the base URL used to reach the upstream LM Studio server.

    Accepts values with or without trailing /v1 and with or without trailing slash.
    Returns the normalized URL actually stored.
    """
    global LMSTUDIO_BASE
    if not base_url:
        raise ValueError("base_url must be non-empty")
    b = base_url.strip()
    if not (b.startswith("http://") or b.startswith("https://")):
        b = "http://" + b
    # Remove any trailing slashes then ensure single /v1 suffix
    b = b.rstrip("/")
    if not b.endswith("/v1"):
        b = b + "/v1"
    LMSTUDIO_BASE = b
    return LMSTUDIO_BASE

app = FastAPI()

def _massage_chat_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Fix tool_choice: LM Studio only accepts "none" | "auto" | "required"
    tc = payload.get("tool_choice")
    if isinstance(tc, dict):
        # If tools are provided and caller tried to force a specific tool, "required" is closer in spirit.
        has_tools = bool(payload.get("tools"))
        payload["tool_choice"] = "required" if has_tools else "auto"

    # Optional: some servers choke on complex response_format; keep the simple ones
    rf = payload.get("response_format")
    if isinstance(rf, dict) and rf.get("type") not in ("text", "json_object"):
        # Fallback to plain text if schema is not supported
        payload["response_format"] = {"type": "text"}

    return payload

def _proxy(method: str, path: str, body: Any = None, headers: Dict[str, str] = None) -> Response:
    url = urljoin(LMSTUDIO_BASE + "/", path.lstrip("/"))
    # Strip hop-by-hop headers and set JSON content type if needed
    out_headers = {"Content-Type": "application/json"}
    if headers:
        out_headers.update({k: v for k, v in headers.items() if k.lower() not in ("host", "content-length")})
    resp = requests.request(method, url, json=body, headers=out_headers, timeout=600)
    # Pass through JSON or raw bytes
    content_type = resp.headers.get("Content-Type", "application/json")
    return Response(content=resp.content, status_code=resp.status_code, media_type=content_type)

@app.get("/v1/models")
def models():
    return _proxy("GET", "/v1/models")

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    payload = await req.json()
    payload = _massage_chat_payload(payload)
    return _proxy("POST", "/v1/chat/completions", body=payload, headers=dict(req.headers))

# Generic passthroughs if you ever need them
@app.get("/{full_path:path}")
def passthrough_get(full_path: str):
    return _proxy("GET", f"/{full_path}")

@app.post("/{full_path:path}")
async def passthrough_post(full_path: str, req: Request):
    body = await req.json()
    return _proxy("POST", f"/{full_path}", body=body, headers=dict(req.headers))

if __name__ == "__main__":
    # Run: python lmstudio_toolchoice_shim.py
    uvicorn.run(app, host="127.0.0.1", port=8088)
