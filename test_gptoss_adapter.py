import json
from lmstudio_toolchoice_shim import normalize_tool_call, _collect_allowed_tools, BUILTIN_TOOL_SCHEMAS


def test_normalize_openai_tool_call_builtin():
    headers = {}
    allowed, schemas = _collect_allowed_tools(headers)
    resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "README.md"})}}
                    ]
                }
            }
        ]
    }
    out = normalize_tool_call(resp, allowed, schemas)
    assert out == {"tool": "read_file", "args": {"path": "README.md"}}


def test_normalize_canonical_final():
    headers = {}
    allowed, schemas = _collect_allowed_tools(headers)
    resp = {"final": "done"}
    out = normalize_tool_call(resp, allowed, schemas)
    assert out == {"final": "done"}


def test_json_in_content():
    headers = {}
    allowed, schemas = _collect_allowed_tools(headers)
    resp = {
        "choices": [
            {"message": {"role": "assistant", "content": json.dumps({"tool": "execute_command", "args": {"command": "echo hi"}})}}
        ]
    }
    out = normalize_tool_call(resp, allowed, schemas)
    assert out == {"tool": "execute_command", "args": {"command": "echo hi"}}


def test_mcp_tool_allowed_via_header():
    headers = {"X-MCP-TOOLS": "run_browser_agent"}
    allowed, schemas = _collect_allowed_tools(headers)
    assert "run_browser_agent" in allowed
    resp = {"tool": "run_browser_agent", "args": {}}
    out = normalize_tool_call(resp, allowed, schemas)
    assert out == {"tool": "run_browser_agent", "args": {}}
