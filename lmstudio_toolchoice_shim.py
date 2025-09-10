import os
import time
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin
import json
import logging
import requests
from requests import Timeout as RequestsTimeout
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

# Mutable base URL so a GUI or caller can override prior to launching the server.
LMSTUDIO_BASE: str = os.getenv("LMSTUDIO_BASE", "http://127.0.0.1:1234/v1")
# Toggle: when True, we stream from upstream (LM Studio) and stitch into one final response.
CF_STREAMING_ENABLED: bool = os.getenv("CF_STREAMING_ENABLED", "0").lower() in ("1", "true", "on", "yes")

# Optional shim timeouts (seconds)
try:
    SHIM_CONNECT_TIMEOUT = float(os.getenv("SHIM_CONNECT_TIMEOUT", "10"))
except Exception:
    SHIM_CONNECT_TIMEOUT = 10.0
try:
    SHIM_READ_TIMEOUT = float(os.getenv("SHIM_READ_TIMEOUT", "60"))
except Exception:
    SHIM_READ_TIMEOUT = 60.0

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

def set_cf_streaming_enabled(enabled: bool) -> bool:
    """Enable or disable upstream streaming mode (Cloudflare-friendly).

    Returns the new state.
    """
    global CF_STREAMING_ENABLED
    CF_STREAMING_ENABLED = bool(enabled)
    logger.info(f"[cfg] CF streaming set to {'ON' if CF_STREAMING_ENABLED else 'OFF'}")
    return CF_STREAMING_ENABLED

def is_cf_streaming_enabled() -> bool:
    return CF_STREAMING_ENABLED

logging.basicConfig(level=logging.INFO, format="[shim] %(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lmstudio_shim")

app = FastAPI()

def _massage_chat_payload(payload: Dict[str, Any], force_stream: Optional[bool] = None) -> Dict[str, Any]:
    # Tool choice normalization:
    # - If tool_choice is a valid string ("auto"|"none"|"required"), keep it.
    # - If it's an object, down-convert to "auto" if tools exist; otherwise remove tool_choice.
    # - If no tools, remove tool_choice entirely to let client default apply.
    tc = payload.get("tool_choice")
    has_tools = bool(payload.get("tools"))
    if isinstance(tc, str) and tc in ("auto", "none", "required"):
        pass  # keep as-is
    elif isinstance(tc, dict):
        if has_tools:
            payload["tool_choice"] = "auto"
        else:
            payload.pop("tool_choice", None)
    else:
        if not has_tools:
            payload.pop("tool_choice", None)

    # Optional: some servers choke on complex response_format; keep the simple ones
    rf = payload.get("response_format")
    if isinstance(rf, dict) and rf.get("type") not in ("text", "json_object"):
        # Fallback to plain text if schema is not supported
        payload["response_format"] = {"type": "text"}

    # stream behavior: default False unless we explicitly force True when CF streaming is ON
    if force_stream is None:
        payload["stream"] = False
    else:
        payload["stream"] = bool(force_stream)

    return payload

def _normalize_path(in_path: str) -> str:
    rel = in_path.lstrip("/")
    # Accept bare chat/completions or with v1/
    if rel.startswith("v1/"):
        rel = rel[3:]
    # Ensure we only ever forward paths under v1
    return f"v1/{rel}" if not rel.startswith("v1/") else rel

def _proxy(method: str, path: str, body: Any = None, headers: Dict[str, str] = None) -> Response:
    norm = _normalize_path(path)
    # Build final upstream URL (LMSTUDIO_BASE already ends with /v1)
    suffix = norm[3:] if norm.startswith("v1/") else norm
    url = urljoin(LMSTUDIO_BASE.rstrip("/") + "/", suffix.lstrip("/"))
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
        logger.info(f"-> {method} in={path} norm=/{norm} upstream={url} cf_stream={'ON' if CF_STREAMING_ENABLED else 'OFF'} body_keys={list(body.keys()) if isinstance(body, dict) else 'raw'}")
    except Exception:
        logger.info(f"-> {method} in={path} norm=/{norm} upstream={url} cf_stream={'ON' if CF_STREAMING_ENABLED else 'OFF'}")

    try:
        resp = requests.request(method, url, json=body, headers=out_headers, timeout=600)
    except RequestsTimeout:
        logger.error(f"<- 504 timeout upstream after 600s {url}")
        return JSONResponse(status_code=504, content={"error": {"message": "Upstream timeout", "type": "timeout"}})

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

def _stitch_streaming_chat_completion(payload: Dict[str, Any], headers: Dict[str, str]) -> Response:
    """Call upstream with stream=True, consume SSE chunks, and synthesize one ChatCompletion JSON.

    Logs periodic chunk counts and a short preview.
    """
    norm = _normalize_path("/v1/chat/completions")
    suffix = norm[3:] if norm.startswith("v1/") else norm
    url = urljoin(LMSTUDIO_BASE.rstrip("/") + "/", suffix.lstrip("/"))

    # Prepare outgoing headers
    out_headers: Dict[str, str] = {"Content-Type": "application/json; charset=utf-8"}
    if headers:
        for k, v in headers.items():
            kl = k.lower()
            if kl in ("host", "content-length"):
                continue
            if kl in ("authorization", "cookie"):
                out_headers[k] = v

    # Ensure stream True for upstream, allow per-request overrides of shim timeouts
    up_payload = dict(payload)
    up_payload["stream"] = True
    # For CF streaming mode, prefer a long read timeout so we don't cut long "thinking" pauses.
    connect_timeout = 10.0 if SHIM_CONNECT_TIMEOUT is None else SHIM_CONNECT_TIMEOUT
    read_timeout = 900.0  # per request: 10s connect, 900s read as default for streaming mode
    # Optional: payload can include shim-specific hints
    shim_cfg = up_payload.get("shim") or {}
    try:
        if isinstance(shim_cfg.get("connect_timeout"), (int, float)):
            connect_timeout = float(shim_cfg["connect_timeout"])
        if isinstance(shim_cfg.get("read_timeout"), (int, float)):
            read_timeout = float(shim_cfg["read_timeout"])
    except Exception:
        pass

    try:
        logger.info(f"-> POST in=/v1/chat/completions norm=/{norm} upstream={url} cf_stream=ON (SSE) timeouts=({connect_timeout},{read_timeout})")
        # Use a connect timeout and a per-read timeout so we can detect inactivity.
        resp = requests.post(url, json=up_payload, headers=out_headers, timeout=(connect_timeout, read_timeout), stream=True)
        logger.info(f"<- upstream headers status={resp.status_code} content-type={resp.headers.get('Content-Type')}")
    except RequestsTimeout:
        logger.error(f"<- 504 timeout upstream after 600s {url}")
        return JSONResponse(status_code=504, content={"error": {"type": "timeout", "message": "Upstream timeout"}})
    except Exception as e:
        logger.error(f"Upstream connection failed: {e}")
        return JSONResponse(status_code=502, content={"error": {"type": "upstream_connect", "message": str(e)}})

    # If not OK, try to decode minimal info
    if resp.status_code >= 400:
        try:
            text = resp.text[:1000]
        except Exception:
            text = ""
        logger.warning(f"Upstream returned error status {resp.status_code}; preview={text[:200]}")
        return JSONResponse(status_code=502, content={"error": {"type": "upstream_error", "message": f"Upstream returned {resp.status_code}"}})

    # Accumulators for final message
    acc_role: Optional[str] = None
    acc_content_parts: List[str] = []
    acc_tool_calls: List[Dict[str, Any]] = []
    finish_reason: Optional[str] = None
    first_id: Optional[str] = None
    first_created: Optional[int] = None
    model_name: Optional[str] = None
    chunk_count = 0
    last_log = time.time()
    inactivity_fired = False

    def ensure_tc_len(n: int):
        while len(acc_tool_calls) <= n:
            acc_tool_calls.append({"id": None, "type": "function", "function": {"name": None, "arguments": ""}})

    def normalize_chunk(obj: Any) -> Dict[str, Any]:
        """Some servers wrap the payload as {event: ..., data: {...}} or variants.
        Try to unwrap to the object with 'choices'."""
        if isinstance(obj, dict):
            if "choices" in obj and isinstance(obj["choices"], list):
                return obj
            data = obj.get("data") if isinstance(obj.get("data"), dict) else None
            if data and isinstance(data.get("choices"), list):
                return data
        return obj if isinstance(obj, dict) else {}

    def _compute_preview() -> tuple[str, str]:
        """Return (preview_text, source) where source is 'content' or 'tool_args'."""
        if acc_content_parts:
            pv = "".join(acc_content_parts)[-200:]
            return pv, "content"
        # fallback: last tool_call arguments tail
        for tc in reversed(acc_tool_calls):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str) and args:
                return args[-200:], "tool_args"
        return "", "content"

    try:
        line_iter = resp.iter_lines(decode_unicode=False)
        event_data_parts: List[bytes] = []
        while True:
            try:
                line = next(line_iter)
            except RequestsTimeout:
                # Upstream paused beyond read timeout; return a clean error instead of silently finalizing.
                logger.warning("[SSE] upstream read timed out; returning error")
                try:
                    resp.close()
                except Exception:
                    pass
                return JSONResponse(status_code=504, content={"error": {"type": "upstream_timeout", "message": "Shim upstream read timed out"}})
            except StopIteration:
                # End of stream
                if event_data_parts:
                    data_bytes = b"\n".join(event_data_parts)
                    logger.debug(f"[SSE] EOF event flush: {data_bytes[:120]!r}")
                    if data_bytes.strip() == b"[DONE]":
                        break
                    try:
                        chunk = json.loads(data_bytes.decode("utf-8", errors="ignore"))
                        chunk = normalize_chunk(chunk)
                        logger.debug(f"[SSE] EOF parsed chunk: {chunk}")
                        chunk_count += 1
                        if first_id is None:
                            first_id = chunk.get("id")
                        if first_created is None:
                            created = chunk.get("created")
                            if isinstance(created, int):
                                first_created = created
                        if model_name is None:
                            m = chunk.get("model")
                            if isinstance(m, str):
                                model_name = m
                        choices = chunk.get("choices") or []
                        for ch in choices:
                            delta = ch.get("delta") or {}
                            fr = ch.get("finish_reason")
                            if fr:
                                finish_reason = fr
                            if acc_role is None and isinstance(delta.get("role"), str):
                                acc_role = delta["role"]
                            if isinstance(delta.get("content"), str):
                                acc_content_parts.append(delta["content"])
                            tc = delta.get("tool_calls")
                            if isinstance(tc, list):
                                for entry in tc:
                                    try:
                                        idx = entry.get("index", 0)
                                        if not isinstance(idx, int) or idx < 0:
                                            idx = 0
                                        ensure_tc_len(idx)
                                        target = acc_tool_calls[idx]
                                        if entry.get("id"):
                                            target["id"] = entry["id"]
                                        if entry.get("type"):
                                            target["type"] = entry["type"]
                                        fn = entry.get("function") or {}
                                        tf = target.setdefault("function", {"name": None, "arguments": ""})
                                        if fn.get("name"):
                                            tf["name"] = fn["name"]
                                        args_val = fn.get("arguments")
                                        if isinstance(args_val, str):
                                            tf["arguments"] += args_val
                                        elif isinstance(args_val, (dict, list)):
                                            try:
                                                tf["arguments"] += json.dumps(args_val, ensure_ascii=False)
                                            except Exception:
                                                pass
                                    except Exception as e:
                                        logger.warning(f"Failed to merge tool_call entry: {e}")
                        preview, src = _compute_preview()
                        logger.info(f"[SSE] chunks={chunk_count} preview({src})={preview.encode('utf-8','ignore')[:200].decode('utf-8','ignore')}")
                    except Exception as e:
                        logger.warning(f"SSE buffered JSON parse failed at EOF: {e}")
                break
            except Exception as e:
                logger.warning(f"[SSE] stream error: {e}; finalizing partial result if any")
                break

            # Heartbeat or comment per SSE spec starts with ':'
            if line is None:
                continue
            if len(line) == 0:
                # End of event; process accumulated data lines
                if event_data_parts:
                    data_bytes = b"\n".join(event_data_parts)
                    logger.debug(f"[SSE] event flush: {data_bytes[:120]!r}")
                    event_data_parts = []
                    if data_bytes.strip() == b"[DONE]":
                        logger.info("[SSE] got [DONE] event")
                        break
                    try:
                        chunk = json.loads(data_bytes.decode("utf-8", errors="ignore"))
                        chunk = normalize_chunk(chunk)
                        logger.debug(f"[SSE] parsed chunk: {chunk}")
                    except Exception as e:
                        logger.warning(f"SSE JSON parse failed: {e}; skipping event preview={data_bytes[:120]!r}")
                        continue

                    chunk_count += 1
                    if first_id is None:
                        first_id = chunk.get("id")
                    if first_created is None:
                        created = chunk.get("created")
                        if isinstance(created, int):
                            first_created = created
                    if model_name is None:
                        m = chunk.get("model")
                        if isinstance(m, str):
                            model_name = m

                    choices = chunk.get("choices") or []
                    for ch in choices:
                        delta = ch.get("delta") or {}
                        fr = ch.get("finish_reason")
                        if fr:
                            finish_reason = fr
                        if acc_role is None and isinstance(delta.get("role"), str):
                            acc_role = delta["role"]
                        if isinstance(delta.get("content"), str):
                            acc_content_parts.append(delta["content"])
                        tc = delta.get("tool_calls")
                        if isinstance(tc, list):
                            for entry in tc:
                                try:
                                    idx = entry.get("index", 0)
                                    if not isinstance(idx, int) or idx < 0:
                                        idx = 0
                                    ensure_tc_len(idx)
                                    target = acc_tool_calls[idx]
                                    if entry.get("id"):
                                        target["id"] = entry["id"]
                                    if entry.get("type"):
                                        target["type"] = entry["type"]
                                    fn = entry.get("function") or {}
                                    tf = target.setdefault("function", {"name": None, "arguments": ""})
                                    if fn.get("name"):
                                        tf["name"] = fn["name"]
                                    args_val = fn.get("arguments")
                                    if isinstance(args_val, str):
                                        tf["arguments"] += args_val
                                    elif isinstance(args_val, (dict, list)):
                                        try:
                                            tf["arguments"] += json.dumps(args_val, ensure_ascii=False)
                                        except Exception:
                                            pass
                                except Exception as e:
                                    logger.warning(f"Failed to merge tool_call entry: {e}")

                    now = time.time()
                    if now - last_log >= 2.0:
                        preview, src = _compute_preview()
                        logger.info(f"[SSE] chunks={chunk_count} preview({src})={preview.encode('utf-8','ignore')[:200].decode('utf-8','ignore')}")
                        last_log = now
                continue

            # Non-empty line: parse SSE fields
            if line.startswith(b":"):
                # Comment/heartbeat
                continue
            if line.startswith(b"data:"):
                event_data_parts.append(line[5:].strip())
                continue
            # Ignore other fields like event:, id:
            continue
    finally:
        try:
            resp.close()
        except Exception:
            pass

    # If nothing came through, treat as upstream/proxy error
    if chunk_count == 0:
        logger.warning("[SSE] no chunks received; treating as upstream_timeout_or_proxy")
        return JSONResponse(status_code=502, content={"error": {"type": "upstream_timeout_or_proxy", "message": "Upstream proxy closed connection before model finished"}})

    msg: Dict[str, Any] = {"role": acc_role or "assistant"}
    if acc_content_parts:
        msg["content"] = "".join(acc_content_parts)
    else:
        msg["content"] = None
    if any((tc.get("id") or tc.get("function", {}).get("name") or tc.get("function", {}).get("arguments")) for tc in acc_tool_calls):
        msg["tool_calls"] = acc_tool_calls

    final: Dict[str, Any] = {
        "id": first_id or f"chatcmpl-shim-{int(time.time())}",
        "object": "chat.completion",
        "created": first_created or int(time.time()),
        "model": model_name or payload.get("model") or "unknown",
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": finish_reason or ("length" if inactivity_fired else "stop"),
                "logprobs": None,
            }
        ],
    }

    # Final log
    ptext, psrc = (msg.get("content") or "")[:200], "content"
    if not ptext and msg.get("tool_calls"):
        # use last tool_call args tail
        for tc in reversed(msg["tool_calls"]):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str) and args:
                ptext = args[:200]
                psrc = "tool_args"
                break
    logger.info(f"[SSE] complete chunks={chunk_count} finish_reason={finish_reason or 'stop'} preview({psrc})={ptext}")

    return Response(content=json.dumps(final, ensure_ascii=False).encode("utf-8"), status_code=200, media_type="application/json")

@app.get("/v1/models")
def models():
    return _proxy("GET", "/v1/models")

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    payload = await req.json()
    if CF_STREAMING_ENABLED:
        payload = _massage_chat_payload(payload, force_stream=True)
        return _stitch_streaming_chat_completion(payload, headers=dict(req.headers))
    else:
        payload = _massage_chat_payload(payload, force_stream=False)
        return _proxy("POST", "/v1/chat/completions", body=payload, headers=dict(req.headers))

# Also accept calls without /v1 prefix for convenience
@app.post("/chat/completions")
async def chat_completions_short(req: Request):
    payload = await req.json()
    if CF_STREAMING_ENABLED:
        payload = _massage_chat_payload(payload, force_stream=True)
        return _stitch_streaming_chat_completion(payload, headers=dict(req.headers))
    else:
        payload = _massage_chat_payload(payload, force_stream=False)
        return _proxy("POST", "/chat/completions", body=payload, headers=dict(req.headers))

# Generic passthroughs if you ever need them
@app.get("/health")
def health():
    return _proxy("GET", "/v1/models")

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
