import os
from typing import Dict, Any
from urllib.parse import urljoin
import json
import logging
import requests
from fastapi import FastAPI, Request, Response
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

logging.basicConfig(level=logging.INFO, format="[shim] %(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lmstudio_shim")

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

    # Always explicitly set stream false per requirements
    payload["stream"] = False

    return payload

def _proxy(method: str, path: str, body: Any = None, headers: Dict[str, str] = None) -> Response:
    # Avoid duplicating /v1 when both base already ends with /v1 and path begins with v1/
    rel = path.lstrip("/")
    if LMSTUDIO_BASE.rstrip("/").endswith("/v1") and rel.startswith("v1/"):
        rel = rel[3:]  # drop leading 'v1/' so base/v1 + rel -> base/v1/<rest>
    url = urljoin(LMSTUDIO_BASE.rstrip("/") + "/", rel)
    # Preserve auth / cookie headers, normalize others
    out_headers: Dict[str, str] = {"Content-Type": "application/json; charset=utf-8"}
    if headers:
        for k, v in headers.items():
            kl = k.lower()
            if kl in ("host", "content-length"):
                continue
            # Pass through Authorization, Cookie, etc.
            if kl in ("authorization", "cookie"):
                out_headers[k] = v
    try:
        logger.info(f"-> {method} {path} upstream={url} body_keys={list(body.keys()) if isinstance(body, dict) else 'raw'}")
    except Exception:
        logger.info(f"-> {method} {path} upstream={url}")

    resp = requests.request(method, url, json=body, headers=out_headers, timeout=600)

    raw = resp.content
    content_type = resp.headers.get("Content-Type", "application/json")

    # Try to ensure UTF-8 JSON. If JSON parse fails, return raw.
    processed_bytes = raw
    if "application/json" in content_type.lower():
        text = None
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("cp1252")
                logger.warning("Decoded upstream response with cp1252 fallback")
            except Exception:
                text = None
        if text is not None:
            # Validate JSON structure
            try:
                parsed = json.loads(text)
                processed_bytes = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
            except Exception as e:
                logger.warning(f"Failed to parse JSON: {e}; returning raw bytes")

    preview = processed_bytes[:200]
    try:
        logger.info(f"<- {resp.status_code} bytes={len(processed_bytes)} preview={preview.decode('utf-8','ignore')}")
    except Exception:
        logger.info(f"<- {resp.status_code} bytes={len(processed_bytes)} (preview decode failed)")

    return Response(content=processed_bytes, status_code=resp.status_code, media_type="application/json")

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
