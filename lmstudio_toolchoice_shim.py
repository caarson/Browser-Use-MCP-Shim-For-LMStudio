import os
import time
from typing import Dict, Any, Optional, List, Tuple, Set
from urllib.parse import urljoin
import json
import re
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

# Default: GPT-OSS adapter disabled unless toggled via GUI/CLI or header per request
GPT_OSS_MODE_DEFAULT: bool = os.getenv("SHIM_GPT_OSS_MODE", "0").lower() in ("1", "true", "on", "yes")

# Optional shim timeouts (seconds)
try:
    SHIM_CONNECT_TIMEOUT = float(os.getenv("SHIM_CONNECT_TIMEOUT", "10"))
except Exception:
    SHIM_CONNECT_TIMEOUT = 10.0
try:
    SHIM_READ_TIMEOUT = float(os.getenv("SHIM_READ_TIMEOUT", "60"))
except Exception:
    SHIM_READ_TIMEOUT = 60.0

# Optionally strip reasoning tags like <think>...</think> from final content (default: keep)
SHIM_STRIP_THINK: bool = os.getenv("SHIM_STRIP_THINK", "0").lower() in ("1", "true", "on", "yes")

# Cache recently used model ids to surface in /models, in case upstream doesn't list them
RECENT_MODEL_IDS: Set[str] = set()
# Cache observed loaded context lengths per model (from chat responses)
LOADED_CONTEXT_LENGTHS: Dict[str, int] = {}

# Optional per-model context overrides loaded from shim_config.json
_CTX_OVERRIDES_CACHE: Dict[str, Any] = {"mtime": 0.0, "last": 0.0, "map": {}}
CTX_OVERRIDES_TTL: float = 2.0

def _shim_config_path() -> str:
    try:
        return os.path.join(os.path.dirname(__file__), "shim_config.json")
    except Exception:
        return "shim_config.json"

def _load_context_overrides_if_needed() -> Dict[str, Dict[str, int]]:
    """Read context overrides from shim_config.json with TTL + mtime caching.

    Supported JSON shapes inside shim_config.json:
      {
        "context_overrides": {
          "model-id": {"loaded_context_length": 32768, "max_context_length": 131072},
          "another-id": 24576  # shorthand for loaded_context_length
        }
      }
      or legacy key "model_context_overrides" with the same shape.
    """
    path = _shim_config_path()
    try:
        st = os.stat(path)
        mtime = st.st_mtime
    except Exception:
        return _CTX_OVERRIDES_CACHE.get("map", {})  # no file
    now = time.time()
    if _CTX_OVERRIDES_CACHE.get("mtime") == mtime and (now - _CTX_OVERRIDES_CACHE.get("last", 0.0)) < CTX_OVERRIDES_TTL:
        return _CTX_OVERRIDES_CACHE.get("map", {})
    overrides: Dict[str, Dict[str, int]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("context_overrides") or data.get("model_context_overrides") or {}
        if isinstance(raw, dict):
            for mid, spec in raw.items():
                try:
                    if isinstance(spec, (int, float)):
                        overrides[str(mid)] = {"loaded_context_length": int(spec)}
                    elif isinstance(spec, dict):
                        out: Dict[str, int] = {}
                        # Accept multiple aliases
                        for k_src, k_dst in (
                            ("loaded_context_length", "loaded_context_length"),
                            ("loaded_context", "loaded_context_length"),
                            ("current_context_length", "loaded_context_length"),
                            ("max_context_length", "max_context_length"),
                            ("max", "max_context_length"),
                        ):
                            v = spec.get(k_src)
                            if isinstance(v, (int, float)) and v > 0:
                                out[k_dst] = int(v)
                        if out:
                            overrides[str(mid)] = out
                except Exception:
                    continue
        _CTX_OVERRIDES_CACHE.update({"mtime": mtime, "last": now, "map": overrides})
        if overrides:
            try:
                logger.info("[cfg] context overrides loaded for: %s", list(overrides.keys())[:6])
            except Exception:
                pass
    except Exception as e:
        try:
            logger.warning(f"Failed to load context_overrides: {e}")
        except Exception:
            pass
    return overrides

def _apply_overrides(mid: str, max_ctx: Optional[int], loaded_ctx: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
    """Apply configured overrides for a model id, if present."""
    ov = _load_context_overrides_if_needed().get(str(mid)) or {}
    if isinstance(ov, dict):
        if isinstance(ov.get("max_context_length"), int) and ov["max_context_length"] > 0:
            max_ctx = ov["max_context_length"]
        if isinstance(ov.get("loaded_context_length"), int) and ov["loaded_context_length"] > 0:
            loaded_ctx = ov["loaded_context_length"]
            # keep cache in sync for downstream calls
            try:
                LOADED_CONTEXT_LENGTHS[str(mid)] = int(loaded_ctx)
            except Exception:
                pass
    return max_ctx, loaded_ctx

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
# ---------------- MCP Discovery from Cline Config ----------------

MCP_DISCOVERY_PATH: Optional[str] = os.getenv("CLINE_CONFIG_PATH")
MCP_DISCOVERY_TTL: float = float(os.getenv("MCP_DISCOVERY_TTL", "2.0"))
_MCP_DISCOVERY_CACHE: Dict[str, Any] = {
    "allowed": set(),  # type: ignore[dict-item]
    "schemas": {},
    "aliases": {},
    "mtime": 0.0,
    "last": 0.0,
}

def set_cline_config_path(path: Optional[str]) -> Optional[str]:
    """Set a path to a Cline config or shim MCP tools JSON file to auto-discover tools.
    Returns the normalized path or None if cleared.
    """
    global MCP_DISCOVERY_PATH
    MCP_DISCOVERY_PATH = path
    # reset cache so next request reloads
    _MCP_DISCOVERY_CACHE["mtime"] = 0.0
    logger.info(f"[cfg] Cline config path set to: {path}")
    return MCP_DISCOVERY_PATH

def _parse_type_name(tname: str) -> Optional[type]:
    t = tname.lower().strip()
    return {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "object": dict,
    }.get(t, None)

def _get_loaded_ctx_from_detail(mid: str) -> Optional[int]:
    """Try to fetch loaded context length for a model via REST detail endpoint.
    Checks common keys and caches on success.
    """
    try:
        resp = _proxy("GET", f"/api/v0/models/{mid}")
        try:
            j = json.loads(resp.body.decode("utf-8", "ignore"))
        except Exception:
            return None
        if not isinstance(j, dict):
            return None
        # Prioritized keys we might see from LM Studio variants
        for k in ("loaded_context_length", "context_length", "loaded_context", "current_context_length"):
            v = j.get(k)
            if isinstance(v, int) and v > 0:
                LOADED_CONTEXT_LENGTHS[mid] = v
                return v
        return None
    except Exception:
        return None

def _load_mcp_discovery_if_needed() -> Tuple[set, Dict[str, Dict[str, Optional[type]]], Dict[str, str]]:
    """Load MCP tool names, schemas, aliases from a configured file, with TTL and mtime check.
    Supported JSON shapes:
      - { "mcpTools": ["tool1", ...], "schemas": {...}, "aliases": {...} }
      - { "tools": [ ... ], "schemas": {...}, "aliases": {...} }
      - { "mcp": { "tools": [ ... ], "schemas": {...}, "aliases": {...} } }
      - Cline config with { "mcpServers": { ... possibly with "tools": [ ... ] ... } }
    """
    allowed: set = set()
    schemas: Dict[str, Dict[str, Optional[type]]] = {}
    aliases: Dict[str, str] = {}
    path = MCP_DISCOVERY_PATH
    if not path:
        return allowed, schemas, aliases
    now = time.time()
    try:
        st = os.stat(path)
        mtime = st.st_mtime
    except Exception:
        return allowed, schemas, aliases
    # Fresh enough and not modified
    if _MCP_DISCOVERY_CACHE["mtime"] == mtime and (now - _MCP_DISCOVERY_CACHE["last"]) < MCP_DISCOVERY_TTL:
        return _MCP_DISCOVERY_CACHE.get("allowed", set()), _MCP_DISCOVERY_CACHE.get("schemas", {}), _MCP_DISCOVERY_CACHE.get("aliases", {})
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load MCP discovery file: {e}")
        return allowed, schemas, aliases
    try:
        # Direct shapes
        def merge_tools(tools: Any):
            if isinstance(tools, list):
                for n in tools:
                    if isinstance(n, str):
                        allowed.add(n)
        def merge_schemas(s: Any):
            if isinstance(s, dict):
                for name, sch in s.items():
                    if isinstance(name, str) and isinstance(sch, dict):
                        m: Dict[str, Optional[type]] = {}
                        for k, v in sch.items():
                            if isinstance(v, str):
                                m[k] = _parse_type_name(v)
                        schemas[name] = m
        def merge_aliases(a: Any):
            if isinstance(a, dict):
                for k, v in a.items():
                    if isinstance(k, str) and isinstance(v, str):
                        aliases[k] = v

        if isinstance(data, dict):
            if isinstance(data.get("mcpTools"), list):
                merge_tools(data.get("mcpTools"))
                merge_schemas(data.get("schemas"))
                merge_aliases(data.get("aliases"))
            if isinstance(data.get("tools"), list):
                merge_tools(data.get("tools"))
                merge_schemas(data.get("schemas"))
                merge_aliases(data.get("aliases"))
            mcp = data.get("mcp")
            if isinstance(mcp, dict):
                merge_tools(mcp.get("tools"))
                merge_schemas(mcp.get("schemas"))
                merge_aliases(mcp.get("aliases"))
            # Cline config shape: mcpServers with optional per-server tools
            servers = data.get("mcpServers")
            if isinstance(servers, dict):
                for _name, srv in servers.items():
                    if isinstance(srv, dict):
                        if isinstance(srv.get("tools"), list):
                            merge_tools(srv.get("tools"))
                        # Optional: server-local aliases/schemas
                        if isinstance(srv.get("schemas"), dict):
                            merge_schemas(srv.get("schemas"))
                        if isinstance(srv.get("aliases"), dict):
                            merge_aliases(srv.get("aliases"))
        # Update cache
        _MCP_DISCOVERY_CACHE["allowed"] = allowed
        _MCP_DISCOVERY_CACHE["schemas"] = schemas
        _MCP_DISCOVERY_CACHE["aliases"] = aliases
        _MCP_DISCOVERY_CACHE["mtime"] = mtime
        _MCP_DISCOVERY_CACHE["last"] = now
        if allowed:
            logger.info(f"[mcp] discovered tools from config: {sorted(list(allowed))[:8]}{'...' if len(allowed)>8 else ''}")
    except Exception as e:
        logger.warning(f"Failed to parse MCP discovery file: {e}")
    return allowed, schemas, aliases

# ---------------- GPT-OSS Adapter: Contracts and Validation ----------------

# Built-in tools and their simple schemas (required keys and types). Optional keys marked with None type.
BUILTIN_TOOL_SCHEMAS: Dict[str, Dict[str, Optional[type]]] = {
    "write_to_file": {"path": str, "content": str},
    "read_file": {"path": str},
    "replace_in_file": {"path": str, "search": str, "replace": str},
    "search_files": {"query": str},
    "list_files": {"dir": str},  # dir optional; we'll allow missing below
    "execute_command": {"command": str},
    "list_code_definition_names": {},
    "use_mcp_tool": {"server": str, "tool": str, "args": dict},
    "access_mcp_resource": {"server": str, "resource": str},
    "ask_followup_question": {"question": str},
    "attempt_completion": {"prompt": str},
}

def set_gpt_oss_mode_default(enabled: bool) -> bool:
    global GPT_OSS_MODE_DEFAULT
    GPT_OSS_MODE_DEFAULT = bool(enabled)
    logger.info(f"[cfg] GPT-OSS mode default set to {'ON' if GPT_OSS_MODE_DEFAULT else 'OFF'}")
    return GPT_OSS_MODE_DEFAULT

def is_gpt_oss_mode_default() -> bool:
    return GPT_OSS_MODE_DEFAULT

def _levenshtein(a: str, b: str) -> int:
    # Simple DP; small strings only
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(
                prev[j] + 1,      # deletion
                cur[j - 1] + 1,   # insertion
                prev[j - 1] + cost  # substitution
            ))
        prev = cur
    return prev[-1]

def _closest_name(name: str, candidates: Set[str]) -> Optional[str]:
    try:
        best = None
        best_d = 1e9
        for c in candidates:
            d = _levenshtein(name, c)
            if d < best_d:
                best_d = d
                best = c
        if best is not None and best_d <= max(2, len(name) // 3):
            return best
    except Exception:
        pass
    return None

def _parse_possible_json(text: str) -> Optional[Any]:
    """Try to parse the first JSON object/array found in text.
    Returns parsed object or None.
    """
    if not isinstance(text, str):
        return None
    s = text.strip()
    # Fast path: whole string is JSON
    for candidate in (s,):
        if candidate.startswith("{") or candidate.startswith("["):
            try:
                return json.loads(candidate)
            except Exception:
                pass
    # Fallback: find first '{' and attempt to parse progressively
    if "{" in s:
        start = s.find("{")
        for end in range(len(s), start + 1, -1):
            frag = s[start:end].strip()
            if not (frag.startswith("{") and frag.endswith(("}", "}\n", "}\r", "}\r\n"))):
                continue
            try:
                return json.loads(frag)
            except Exception:
                continue
    return None

def _collect_allowed_tools(headers: Dict[str,str]) -> Tuple[Set[str], Dict[str, Dict[str, Optional[type]]], Dict[str, str]]:
    """Combine built-in tools with any MCP tools advertised via headers.
    Optionally accept schemas via JSON header.
    Headers:
      - X-MCP-TOOLS: comma-separated names
      - X-MCP-TOOLS-JSON: JSON array of names or JSON object {name: schema}
      - X-MCP-TOOL-SCHEMAS: JSON object {name: {argName: typeName}}
    typeName may be one of: string, number, integer, boolean, object
    """
    allowed: Set[str] = set(BUILTIN_TOOL_SCHEMAS.keys())
    schemas: Dict[str, Dict[str, Optional[type]]] = dict(BUILTIN_TOOL_SCHEMAS)
    aliases: Dict[str, str] = {}

    # Merge discovery from config file (auto-updating)
    disc_allowed, disc_schemas, disc_aliases = _load_mcp_discovery_if_needed()
    allowed.update(disc_allowed)
    for k, v in disc_schemas.items():
        schemas[k] = v
    for k, v in disc_aliases.items():
        aliases[k] = v
    def parse_type(tname: str) -> Optional[type]:
        t = tname.lower().strip()
        return {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "object": dict,
        }.get(t, None)

    try:
        mcps = headers.get("X-MCP-TOOLS") or ""
        for name in [x.strip() for x in mcps.split(",") if x.strip()]:
            allowed.add(name)
            if name not in schemas:
                schemas[name] = {}  # no schema provided; accept any dict
    except Exception:
        pass
    try:
        raw = headers.get("X-MCP-TOOLS-JSON")
        if raw:
            data = json.loads(raw)
            if isinstance(data, list):
                for name in data:
                    if isinstance(name, str):
                        allowed.add(name)
                        schemas.setdefault(name, {})
            elif isinstance(data, dict):
                for name, sch in data.items():
                    if not isinstance(name, str):
                        continue
                    allowed.add(name)
                    if isinstance(sch, dict):
                        # map schema to python types when possible
                        m: Dict[str, Optional[type]] = {}
                        for k, v in sch.items():
                            if isinstance(v, str):
                                m[k] = parse_type(v)  # may be None
                        schemas[name] = m
                    else:
                        schemas.setdefault(name, {})
    except Exception:
        pass
    try:
        raw = headers.get("X-MCP-TOOL-SCHEMAS")
        if raw:
            data = json.loads(raw)
            if isinstance(data, dict):
                for name, sch in data.items():
                    if isinstance(name, str) and isinstance(sch, dict):
                        m: Dict[str, Optional[type]] = {}
                        for k, v in sch.items():
                            if isinstance(v, str):
                                m[k] = parse_type(v)
                        schemas[name] = m
    except Exception:
        pass
    # Optional alias mapping: provider/tool -> actual MCP tool name
    try:
        raw = headers.get("X-MCP-TOOL-ALIASES")
        if raw:
            data = json.loads(raw)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(k, str) and isinstance(v, str):
                        aliases[k] = v
                        # Include alias target in allowed set implicitly
                        allowed.add(v)
    except Exception:
        pass
    return allowed, schemas, aliases

def _validate_args(tool: str, args: Any, schemas: Dict[str, Dict[str, Optional[type]]]) -> Dict[str, Any]:
    if not isinstance(args, dict):
        raise ValueError("args must be an object")
    schema = schemas.get(tool)
    if schema is None:
        # Unknown schema: accept any dict
        return args
    # Special-case: list_files.dir is optional
    required_keys = [k for k in schema.keys() if not (tool == "list_files" and k == "dir")]
    for k in required_keys:
        if k not in args:
            raise ValueError(f"missing required arg '{k}'")
    # Type checks (best-effort, optional types allowed)
    for k, t in schema.items():
        if k not in args or t is None:
            continue
        v = args[k]
        if t is bool:
            if not isinstance(v, bool):
                raise ValueError(f"arg '{k}' must be boolean")
        elif t is int:
            if not isinstance(v, int):
                raise ValueError(f"arg '{k}' must be integer")
        elif t is float:
            if not isinstance(v, (int, float)):
                raise ValueError(f"arg '{k}' must be number")
        elif t is str:
            if not isinstance(v, str):
                raise ValueError(f"arg '{k}' must be string")
        elif t is dict:
            if not isinstance(v, dict):
                raise ValueError(f"arg '{k}' must be object")
    # No extras allowed per contract
    extras = [k for k in args.keys() if k not in schema]
    if extras:
        raise ValueError(f"unexpected arg(s): {', '.join(extras)}")
    return args

def normalize_tool_call(response: Dict[str, Any], allowed_tools: Set[str], schemas: Dict[str, Dict[str, Optional[type]]], aliases: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Normalize OpenAI-style tool_calls or JSON content into Cline canonical format.
    Returns either {"tool": name, "args": {...}} or {"final": text}.
    Raises ValueError on invalid tool JSON.
    """
    # 1) OpenAI chat completion envelope
    if isinstance(response, dict) and isinstance(response.get("choices"), list) and response["choices"]:
        msg = response["choices"][0].get("message") or {}
        # Prefer tool_calls
        m_tc = msg.get("tool_calls")
        if isinstance(m_tc, list) and m_tc:
            if len(m_tc) > 1:
                logger.info("gptoss.multiple_calls: more than one tool call returned; taking the first")
            entry = m_tc[0]
            fn = (entry or {}).get("function") or {}
            name = fn.get("name")
            if not name or not isinstance(name, str):
                raise ValueError("tool_calls[0].function.name missing")
            # Apply alias mapping if provided
            if aliases and name in aliases:
                name = aliases[name]
            if name not in allowed_tools:
                suggestion = _closest_name(name, allowed_tools)
                raise ValueError(f"unknown tool '{name}'" + (f"; did you mean '{suggestion}'?" if suggestion else ""))
            raw_args = fn.get("arguments")
            if isinstance(raw_args, str):
                try:
                    args_obj = json.loads(raw_args)
                except Exception as e:
                    raise ValueError(f"arguments not valid JSON: {e}")
            elif isinstance(raw_args, dict):
                args_obj = raw_args
            else:
                raise ValueError("arguments must be JSON string or object")
            args_obj = _validate_args(name, args_obj, schemas)
            return {"tool": name, "args": args_obj}
        # Else, check JSON in content
        content = msg.get("content")
        parsed = _parse_possible_json(content) if isinstance(content, str) else None
        if isinstance(parsed, dict):
            if set(parsed.keys()) == {"final"} and isinstance(parsed.get("final"), str):
                return parsed
            # Could be {tool, args}
            tool = parsed.get("tool")
            args = parsed.get("args")
            if isinstance(tool, str) and isinstance(args, dict):
                if aliases and tool in aliases:
                    tool = aliases[tool]
                if tool not in allowed_tools:
                    suggestion = _closest_name(tool, allowed_tools)
                    raise ValueError(f"unknown tool '{tool}'" + (f"; did you mean '{suggestion}'?" if suggestion else ""))
                args = _validate_args(tool, args, schemas)
                return {"tool": tool, "args": args}
        # Fallback: treat content as final
        if isinstance(content, str) and content.strip():
            return {"final": content}
        # Nothing usable
        raise ValueError("no tool_calls or JSON content found")
    # 2) Direct JSON dict already
    if isinstance(response, dict):
        # If it looks like a canonical object
        if set(response.keys()) == {"final"} and isinstance(response.get("final"), str):
            return response
        if "tool" in response and "args" in response:
            tool = response.get("tool")
            args = response.get("args")
            if not isinstance(tool, str) or not isinstance(args, dict):
                raise ValueError("invalid tool/args types")
            if aliases and tool in aliases:
                tool = aliases[tool]
            if tool not in allowed_tools:
                suggestion = _closest_name(tool, allowed_tools)
                raise ValueError(f"unknown tool '{tool}'" + (f"; did you mean '{suggestion}'?" if suggestion else ""))
            args = _validate_args(tool, args, schemas)
            return {"tool": tool, "args": args}
    # 3) String response
    if isinstance(response, str):
        parsed = _parse_possible_json(response)
        if isinstance(parsed, dict):
            return normalize_tool_call(parsed, allowed_tools, schemas, aliases=aliases)
        return {"final": response}
    raise ValueError("unrecognized response shape")

def _inject_prompt_contract(payload: Dict[str, Any], allowed_tools: Set[str]) -> Dict[str, Any]:
    contract = (
        "For tool steps you MUST output exactly ONE of:\n"
        "A) {\"tool\":\"<one of: "
        + " | ".join(sorted(allowed_tools))
        + "\">,\"args\":{...}}\n"
        "B) {\"final\":\"<answer>\"}\n\n"
        "Rules:\n\nNo prose outside JSON.\n\nCase-sensitive tool names.\n\nInclude ALL required args; no extras.\n\nOne tool per message."
    )
    sys_msg = {"role": "system", "content": contract}
    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        msgs = []
    # Prepend the contract to steer output
    payload = dict(payload)
    payload["messages"] = [sys_msg] + msgs
    return payload

def _retry_payload(payload: Dict[str, Any], error_text: str) -> Dict[str, Any]:
    hint = {
        "role": "system",
        "content": f"Your last tool JSON was invalid: {error_text}. Output ONLY corrected JSON per the contract.",
    }
    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        msgs = []
    payload2 = dict(payload)
    payload2["messages"] = msgs + [hint]
    # Ensure non-stream for retry
    payload2["stream"] = False
    return payload2

def _call_upstream_chat(body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Call upstream /v1/chat/completions with non-stream JSON. Returns (json, error_text)."""
    norm = _normalize_path("/v1/chat/completions")
    suffix = norm[3:] if norm.startswith("v1/") else norm
    url = urljoin(LMSTUDIO_BASE.rstrip("/") + "/", suffix.lstrip("/"))
    out_headers: Dict[str, str] = {"Content-Type": "application/json; charset=utf-8"}
    for k, v in headers.items():
        kl = k.lower()
        if kl in ("host", "content-length"):
            continue
        if kl in ("authorization", "cookie"):
            out_headers[k] = v
    try:
        resp = requests.post(url, json=body, headers=out_headers, timeout=600)
    except RequestsTimeout:
        return None, "Upstream timeout"
    except Exception as e:
        return None, f"Upstream error: {e}"
    try:
        data = resp.json()
        # Cache loaded context length if LM Studio returns it in model_info
        try:
            mid = data.get("model") or body.get("model")
            mi = data.get("model_info")
            if isinstance(mid, str) and isinstance(mi, dict):
                cl = mi.get("context_length")
                if isinstance(cl, int) and cl > 0:
                    LOADED_CONTEXT_LENGTHS[mid] = cl
        except Exception:
            pass
        return data, None
    except Exception as e:
        return None, f"Invalid upstream JSON: {e}"

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
    """Normalize only the OpenAI-compatible paths to /v1/*.

    Important: Do NOT rewrite /api/v0/* here — those should be sent to the upstream REST base (not /v1).
    This function returns a path starting with 'v1/' for OpenAI routes; otherwise returns the input sans leading '/'.
    """
    rel = in_path.lstrip("/")
    if rel.startswith("api/v0/"):
        # Preserve REST path for caller to handle specially in _proxy
        return rel
    # Accept bare chat/completions or with v1/
    if rel.startswith("v1/"):
        rel = rel[3:]
    return f"v1/{rel}" if not rel.startswith("v1/") else rel

def _proxy(method: str, path: str, body: Any = None, headers: Dict[str, str] = None) -> Response:
    """Proxy a request to upstream.

    - For /api/v0/*: send to the upstream base ROOT (strip trailing /v1 from LMSTUDIO_BASE) and preserve path.
    - For others: map to /v1/* under LMSTUDIO_BASE (OpenAI-compatible API).
    """
    rel = path.lstrip("/")
    if rel.startswith("api/v0/"):
        # Compute root by stripping trailing /v1 if present
        base_root = LMSTUDIO_BASE
        if base_root.endswith("/v1"):
            base_root = base_root[:-3]
        url = urljoin(base_root.rstrip("/") + "/", rel)
        norm = rel  # for logging
    else:
        norm = _normalize_path(path)
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
            # Pass through feature toggles for upstream observability if desired
            if k == "X-GPT-OSS-MODE":
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

    def merge_choice_into_acc(choice_obj: Dict[str, Any]):
        """Merge a single choices[*] object into accumulators, supporting both delta.* and message.* shapes."""
        nonlocal acc_role, finish_reason
        delta = choice_obj.get("delta") or {}
        fr = choice_obj.get("finish_reason")
        if fr:
            finish_reason = fr
        # Prefer delta fields first
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
        # Fallback: some servers send final content/tool_calls under message.*
        msg_obj = choice_obj.get("message") or {}
        if isinstance(msg_obj, dict):
            if acc_role is None and isinstance(msg_obj.get("role"), str):
                acc_role = msg_obj["role"]
            if isinstance(msg_obj.get("content"), str) and not delta.get("content"):
                acc_content_parts.append(msg_obj["content"])
            mtc = msg_obj.get("tool_calls")
            if isinstance(mtc, list) and not tc:
                for entry in mtc:
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
                        logger.warning(f"Failed to merge message.tool_calls entry: {e}")

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
                            merge_choice_into_acc(ch)
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
                        merge_choice_into_acc(ch)

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
        content_text = "".join(acc_content_parts)
        # Optional reasoning tag stripping
        strip_think = SHIM_STRIP_THINK
        try:
            shim_cfg = payload.get("shim") or {}
            if isinstance(shim_cfg.get("strip_think"), bool):
                strip_think = bool(shim_cfg["strip_think"])
        except Exception:
            pass
        if strip_think:
            try:
                content_text = re.sub(r"<think>.*?</think>", "", content_text, flags=re.DOTALL|re.IGNORECASE)
            except Exception:
                pass
        msg["content"] = content_text
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

    # Final log (show tail preview and length to avoid confusion about truncation)
    full_content = msg.get("content") or ""
    ptext, psrc = (full_content[-200:] if full_content else ""), "content"
    if not ptext and msg.get("tool_calls"):
        # use last tool_call args tail
        for tc in reversed(msg["tool_calls"]):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str) and args:
                ptext = args[-200:]
                psrc = "tool_args"
                break
    clen = len(full_content) if isinstance(full_content, str) else 0
    logger.info(f"[SSE] complete chunks={chunk_count} finish_reason={finish_reason or 'stop'} preview_tail({psrc},len={clen})={ptext}")

    return Response(content=json.dumps(final, ensure_ascii=False).encode("utf-8"), status_code=200, media_type="application/json")

@app.get("/v1/models")
def models():
    # Fetch upstream list first
    resp = _proxy("GET", "/v1/models")
    try:
        data = json.loads(resp.body.decode("utf-8", "ignore"))
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            enriched = []
            for m in data["data"]:
                # Preserve original fields and enrich with conservative context metadata for Cline
                if isinstance(m, dict):
                    mm = dict(m)
                    # Provide minimal metadata that many clients expect
                    # Use reasonable defaults if upstream doesn't provide them
                    usage = mm.get("usage", {})
                    if not isinstance(usage, dict):
                        usage = {}
                    # If vendor info missing, set owner-like field
                    mm.setdefault("owned_by", mm.get("owned_by", "organization_owner"))
                    # Attach a conservative context window hint; LM Studio often supports large windows
                    # If your backend exposes exact numbers, you can map them here.
                    # Provide multiple common shapes various clients read
                    # Compute a baseline then apply any overrides so clients see accurate numbers
                    base_total = 131072
                    try:
                        # Attempt to reuse vendor hints if present
                        if isinstance(mm.get("context_window"), dict) and isinstance(mm["context_window"].get("total"), int):
                            base_total = int(mm["context_window"]["total"])  # type: ignore[index]
                        elif isinstance(mm.get("max_context_window"), int):
                            base_total = int(mm["max_context_window"])  # type: ignore[assignment]
                        elif isinstance((mm.get("tokens") or {}).get("context"), int):
                            base_total = int((mm.get("tokens") or {}).get("context"))  # type: ignore[assignment]
                        elif isinstance(mm.get("context_window_total"), int):
                            base_total = int(mm.get("context_window_total"))  # type: ignore[assignment]
                    except Exception:
                        base_total = 131072
                    mid = mm.get("id") or mm.get("model") or mm.get("name") or "unknown"
                    max_ctx, _ = _apply_overrides(str(mid), base_total, None)
                    if "context_window" not in mm:
                        mm["context_window"] = {"total": int(max_ctx or base_total)}
                    mm.setdefault("max_context_window", int(max_ctx or base_total))
                    mm.setdefault("tokens", {"context": int(max_ctx or base_total)})
                    mm.setdefault("context_window_total", int(max_ctx or base_total))
                    enriched.append(mm)
                else:
                    enriched.append(m)
            data["data"] = enriched
            return JSONResponse(status_code=200, content=data)
    except Exception:
        pass
    return resp

@app.get("/models")
def models_plain():
    """
    LM Studio-style models endpoint used by Cline's LM Studio provider UI.
    It expects a shape like: { "values": [ "{...modelJson...}", ... ] }

    We try to proxy upstream /models first. If that isn't available, we
    synthesize the structure from the OpenAI-compatible /v1/models response.
    """
    # Try direct upstream passthrough first (some LM Studio builds expose this)
    upstream_try = _proxy("GET", "/models")
    try:
        raw = upstream_try.body.decode("utf-8", "ignore")
        j = json.loads(raw)
        if isinstance(j, dict) and isinstance(j.get("values"), list):
            try:
                logger.info(f"/models passthrough: upstream values={len(j['values'])}")
            except Exception:
                pass
            return JSONResponse(status_code=upstream_try.status_code, content=j)
    except Exception:
        # fall back to synthesize below
        pass

    # Fallback: build from /v1/models list
    fallback_resp = _proxy("GET", "/v1/models")
    try:
        data = json.loads(fallback_resp.body.decode("utf-8", "ignore"))
        values = []
        # Start with any recent model ids we saw in chat requests
        recent_ids = list(RECENT_MODEL_IDS)
        recent_ids_set = set(recent_ids)
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            for m in data["data"]:
                if not isinstance(m, dict):
                    continue
                mid = m.get("id") or m.get("model") or m.get("name") or "unknown"
                recent_ids_set.add(str(mid))
                # Extract any known context hints, else default conservatively
                ctx = (
                    (m.get("context_window") or {}).get("total")
                    if isinstance(m.get("context_window"), dict)
                    else None
                )
                if ctx is None:
                    # Try common alternates we may have set in /v1/models enrichment
                    ctx = (
                        m.get("max_context_window")
                        or (m.get("tokens") or {}).get("context")
                        or m.get("context_window_total")
                    )
                # Final default if still unknown
                if not isinstance(ctx, int):
                    ctx = 131072
                # Prefer observed loaded context length if available
                cached_loaded = LOADED_CONTEXT_LENGTHS.get(str(mid))
                if not isinstance(cached_loaded, int):
                    cached_loaded = _get_loaded_ctx_from_detail(str(mid))
                loaded_ctx = int(cached_loaded) if isinstance(cached_loaded, int) else int(ctx)
                # Apply overrides if configured
                ctx, loaded_ctx = _apply_overrides(str(mid), ctx, loaded_ctx)
                lm_entry = {
                    "id": str(mid),
                    "max_context_length": int(ctx),
                    "loaded_context_length": loaded_ctx,
                }
                values.append(json.dumps(lm_entry))
        # Include recent model ids that may not appear in upstream list
        if not values and recent_ids:
            for mid in recent_ids:
                ctx = 131072
                cached_loaded = LOADED_CONTEXT_LENGTHS.get(str(mid))
                if not isinstance(cached_loaded, int):
                    cached_loaded = _get_loaded_ctx_from_detail(str(mid))
                loaded_ctx = int(cached_loaded) if isinstance(cached_loaded, int) else ctx
                ctx, loaded_ctx = _apply_overrides(str(mid), ctx, loaded_ctx)
                values.append(json.dumps({"id": str(mid), "max_context_length": ctx, "loaded_context_length": loaded_ctx}))
        # As a last resort, include a couple of sensible placeholders so the UI isn't empty
        if not values:
            for mid in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
                ctx = 131072
                cached_loaded = LOADED_CONTEXT_LENGTHS.get(str(mid))
                if not isinstance(cached_loaded, int):
                    cached_loaded = _get_loaded_ctx_from_detail(str(mid))
                loaded_ctx = int(cached_loaded) if isinstance(cached_loaded, int) else ctx
                ctx, loaded_ctx = _apply_overrides(str(mid), ctx, loaded_ctx)
                values.append(json.dumps({"id": mid, "max_context_length": ctx, "loaded_context_length": loaded_ctx}))
        # If no models found, still return an empty values array
        try:
            logger.info(
                "/models synthesize: values=%d recent_ids=%d (fallback from /v1/models)",
                len(values), len(recent_ids)
            )
        except Exception:
            pass
        return JSONResponse(status_code=200, content={"values": values})
    except Exception:
        # As a last resort, surface the upstream status and a sane empty response
        try:
            try:
                logger.warning("/models synthesize failed, returning empty values with status=%s", str(fallback_resp.status_code))
            except Exception:
                pass
            return JSONResponse(status_code=fallback_resp.status_code, content={"values": []})
        except Exception:
            try:
                logger.warning("/models synthesize failed, upstream status unavailable; returning empty values")
            except Exception:
                pass
            return JSONResponse(status_code=200, content={"values": []})

def _synthesize_models_list_for_rest() -> list:
    """Build a list of REST-style model dicts with context info.

    Target shape per item:
      { id, object: 'model', max_context_length: int, loaded_context_length: int, state: 'loaded'|'not-loaded' }
    """
    # 1) Try upstream REST first, if present
    upstream = _proxy("GET", "/api/v0/models")
    try:
        raw = upstream.body.decode("utf-8", "ignore")
        j = json.loads(raw)
        if isinstance(j, dict) and isinstance(j.get("data"), list):
            out = []
            for m in j["data"]:
                if not isinstance(m, dict):
                    continue
                mid = m.get("id") or m.get("model") or m.get("name")
                if not mid:
                    continue
                ctx = m.get("max_context_length")
                if not isinstance(ctx, int):
                    # fallback: try other common hints
                    ctx = (
                        (m.get("context_window") or {}).get("total") if isinstance(m.get("context_window"), dict) else None
                    ) or m.get("max_context_window") or (m.get("tokens") or {}).get("context") or m.get("context_window_total")
                if not isinstance(ctx, int):
                    ctx = 131072
                loaded_ctx = m.get("loaded_context_length")
                if not isinstance(loaded_ctx, int):
                    cached = LOADED_CONTEXT_LENGTHS.get(str(mid))
                    if not isinstance(cached, int):
                        cached = _get_loaded_ctx_from_detail(str(mid))
                    loaded_ctx = int(cached) if isinstance(cached, int) else int(ctx)
                # Apply overrides if any
                ctx, loaded_ctx = _apply_overrides(str(mid), ctx, loaded_ctx)
                state = m.get("state") or ("loaded" if str(mid) in RECENT_MODEL_IDS else "not-loaded")
                out.append({
                    "id": str(mid),
                    "object": "model",
                    "max_context_length": int(ctx),
                    "loaded_context_length": int(loaded_ctx),
                    "state": state,
                })
            try:
                logger.info("/api/v0/models passthrough: items=%d", len(out))
            except Exception:
                pass
            return out
    except Exception:
        pass

    # 2) Fallback: synthesize from OpenAI-style /v1/models enrichment
    fallback = _proxy("GET", "/v1/models")
    out = []
    recent = set(RECENT_MODEL_IDS)
    try:
        data = json.loads(fallback.body.decode("utf-8", "ignore"))
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            for m in data["data"]:
                if not isinstance(m, dict):
                    continue
                mid = m.get("id") or m.get("model") or m.get("name")
                if not mid:
                    continue
                ctx = (
                    (m.get("context_window") or {}).get("total") if isinstance(m.get("context_window"), dict) else None
                ) or m.get("max_context_window") or (m.get("tokens") or {}).get("context") or m.get("context_window_total")
                if not isinstance(ctx, int):
                    ctx = 131072
                cached = LOADED_CONTEXT_LENGTHS.get(str(mid))
                if not isinstance(cached, int):
                    cached = _get_loaded_ctx_from_detail(str(mid))
                loaded_ctx = int(cached) if isinstance(cached, int) else int(ctx)
                ctx, loaded_ctx = _apply_overrides(str(mid), ctx, loaded_ctx)
                out.append({
                    "id": str(mid),
                    "object": "model",
                    "max_context_length": int(ctx),
                    "loaded_context_length": int(loaded_ctx),
                    "state": "loaded" if str(mid) in recent else "not-loaded",
                })
        # Ensure we include recent model IDs even if upstream omitted them
        listed_ids = {item["id"] for item in out}
        for mid in recent:
            if mid not in listed_ids:
                cached = LOADED_CONTEXT_LENGTHS.get(str(mid))
                if not isinstance(cached, int):
                    cached = _get_loaded_ctx_from_detail(str(mid))
                loaded_ctx = int(cached) if isinstance(cached, int) else 131072
                _, loaded_ctx = _apply_overrides(str(mid), None, loaded_ctx)
                out.append({
                    "id": str(mid),
                    "object": "model",
                    "max_context_length": 131072,
                    "loaded_context_length": loaded_ctx,
                    "state": "loaded",
                })
    except Exception:
        pass
    if not out:
        # Provide a minimal placeholder set so the UI has something to render
        out = [{
            "id": mid,
            "object": "model",
            "max_context_length": 131072,
            "loaded_context_length": 131072,
            "state": "not-loaded",
        } for mid in ("openai/gpt-oss-20b", "openai/gpt-oss-120b")]
    try:
        logger.info("/api/v0/models synthesize: items=%d recent=%d", len(out), len(RECENT_MODEL_IDS))
    except Exception:
        pass
    return out

@app.get("/api/v0/models")
def models_api_v0():
    """LM Studio REST-style models list: {object:'list', data:[...]}"""
    items = _synthesize_models_list_for_rest()
    return JSONResponse(status_code=200, content={"object": "list", "data": items})

@app.get("/api/v0/models/{model_id}")
def model_api_v0_detail(model_id: str):
    """LM Studio REST-style single model detail."""
    items = _synthesize_models_list_for_rest()
    for it in items:
        if it.get("id") == model_id:
            # Ensure overrides are reflected even if cached list is slightly stale
            max_c = it.get("max_context_length") if isinstance(it.get("max_context_length"), int) else None
            loaded_c = it.get("loaded_context_length") if isinstance(it.get("loaded_context_length"), int) else None
            max_c, loaded_c = _apply_overrides(model_id, max_c, loaded_c)
            it["max_context_length"] = int(max_c or it.get("max_context_length") or 131072)
            it["loaded_context_length"] = int(loaded_c or it.get("loaded_context_length") or it.get("max_context_length") or 131072)
            return JSONResponse(status_code=200, content=it)
    # If not found, return a minimal placeholder with defaults
    return JSONResponse(status_code=200, content={
        "id": model_id,
        "object": "model",
        "max_context_length": 131072,
        "loaded_context_length": 131072,
        "state": "not-loaded",
    })

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    payload = await req.json()
    try:
        mid = payload.get("model")
        if isinstance(mid, str) and mid:
            RECENT_MODEL_IDS.add(mid)
    except Exception:
        pass
    headers = {k: v for k, v in req.headers.items()}
    # Resolve GPT-OSS mode for this request via header, else default
    gptoss_hdr = headers.get("X-GPT-OSS-MODE")
    gptoss_on = (gptoss_hdr.lower() in ("1", "true", "on", "yes")) if isinstance(gptoss_hdr, str) else GPT_OSS_MODE_DEFAULT

    if not gptoss_on:
        # Existing behavior
        if CF_STREAMING_ENABLED:
            payload = _massage_chat_payload(payload, force_stream=True)
            return _stitch_streaming_chat_completion(payload, headers=headers)
        else:
            payload = _massage_chat_payload(payload, force_stream=False)
            return _proxy("POST", "/v1/chat/completions", body=payload, headers=headers)

    # GPT-OSS Mode ON: inject prompt contract and force non-stream
    allowed, schemas, aliases = _collect_allowed_tools(headers)
    work = dict(payload)
    work = _massage_chat_payload(work, force_stream=False)
    work = _inject_prompt_contract(work, allowed)

    # First attempt
    first_json, err = _call_upstream_chat(work, headers)
    if first_json is None:
        logger.info("gptoss.retry: upstream error on first attempt; retrying once")
        # Retry once with explicit error hint to LLM
        work2 = _retry_payload(work, err or "unknown error")
        second_json, err2 = _call_upstream_chat(work2, headers)
        if second_json is None:
            logger.info("gptoss.fallback_final: upstream failed twice; returning final fallback")
            return JSONResponse(status_code=200, content={"choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps({"final": f"LLM upstream failed: {err2 or err}"})}}]})
        first_json = second_json

    # Try to normalize; on error, retry once with corrective hint
    try:
        normalized = normalize_tool_call(first_json, allowed, schemas, aliases=aliases)
        logger.info("gptoss.normalized: produced canonical tool/final")
    except Exception as e:
        logger.info(f"gptoss.retry: normalization failed first pass: {e}")
        work2 = _retry_payload(work, str(e))
        second_json, err2 = _call_upstream_chat(work2, headers)
        if second_json is None:
            logger.info("gptoss.fallback_final: retry upstream failed; returning final fallback")
            final = {"final": f"Tool JSON invalid and retry failed: {err2}"}
            normalized = final
        else:
            try:
                normalized = normalize_tool_call(second_json, allowed, schemas, aliases=aliases)
                logger.info("gptoss.normalized: produced canonical tool/final after retry")
            except Exception as e2:
                logger.info("gptoss.fallback_final: normalization failed again; returning final fallback")
                normalized = {"final": f"Tool JSON invalid after retry: {e2}"}

    # Return the normalized payload inside a minimal ChatCompletion envelope so Cline can consume
    content_payload = json.dumps(normalized, ensure_ascii=False)
    out = {
        "id": f"chatcmpl-gptoss-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model") or "unknown",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content_payload}, "finish_reason": "stop", "logprobs": None}
        ],
    }
    return JSONResponse(status_code=200, content=out)

# Also accept calls without /v1 prefix for convenience
@app.post("/chat/completions")
async def chat_completions_short(req: Request):
    # Reuse the main handler by forwarding
    return await chat_completions(req)

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
