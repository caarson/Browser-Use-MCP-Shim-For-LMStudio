# Browser-Use-MCP-Shim-For-LMStudio

A lightweight reverse-proxy shim that stitches LM Studio’s SSE streaming responses into a single OpenAI-compatible Chat Completion response. Includes a VS Code-like GUI to start/stop the local server and view logs.

## Setup (dev)

- Install dependencies:

```
pip install -r requirements.txt
```

- Start the GUI (recommended):

```
python main.py
```

  - Enter LM Studio base (e.g. http://127.0.0.1:1234). The shim normalizes it to end with /v1.
  - Choose host/port for the shim (default 127.0.0.1:8088).
  - Toggle “Cloudflare Streaming” to enable upstream SSE stitching mode.
  - Start/Stop server; open /v1/models; copy the base URL.

- Or run server CLI directly:

```
python -m lmstudio_toolchoice_shim
```

## Streaming behavior and timeouts

- When Cloudflare Streaming is ON, the shim requests upstream with stream=True and a long read timeout by default:
  - connect timeout: 10s
  - read timeout: 900s (prevents premature stops during long “thinking” gaps)
- If an upstream read timeout occurs, the shim returns:

```
{ "error": { "type": "upstream_timeout", "message": "Shim upstream read timed out" } }
```

- The shim parses multi-line SSE correctly, flushes on [DONE], and merges streamed tool_calls safely.
- Logs show preview(content)=… when text is streaming, and preview(tool_args)=… when only tool arguments are streaming.

### Optional configuration

- Env vars:
  - SHIM_CONNECT_TIMEOUT (seconds)
  - SHIM_READ_TIMEOUT (seconds)
- Per-request hints (in body):

```
{ ... , "shim": { "connect_timeout": 10, "read_timeout": 900 } }
```

## Tool choice normalization

- If `tool_choice` is a valid string ("auto" | "none" | "required"), it’s kept unchanged.
- If `tool_choice` is an object, it is down-converted to "auto" when tools are present; otherwise removed.
- If no tools are present, `tool_choice` is removed so client defaults apply.

## Build Windows .exe (ShimServer)

This repo ships a PyInstaller spec already set to build `ShimServer.exe`.

- Using the provided VS Code task (recommended): run the task “Build ShimServer.exe (module path)”.
- Or from a PowerShell in the venv:

```
python -m PyInstaller --clean --noconfirm pyinstaller.spec
```

Artifacts:
- One-folder build in `dist/ShimServer/ShimServer.exe`
- You can copy it to `releases/` for convenience.

## MCP configuration example

Update your MCP settings to point at the shim:

```
{
  "mcpServers": {
    "github.com/Saik0s/mcp-browser-use": {
      "command": "uvx",
      "args": ["mcp-server-browser-use@latest"],
      "env": {
        "MCP_LLM_PROVIDER": "openai",
        "MCP_LLM_BASE_URL": "http://127.0.0.1:8088/v1",
        "MCP_LLM_OPENAI_API_KEY": "lm-studio",
        "MCP_LLM_MODEL_NAME": "exact",
        "MCP_BROWSER_HEADLESS": "true",
        "MCP_AGENT_TOOL_USE_VISION": "false"
      }
    }
  }
}
```

Tip: Keep LM Studio’s local server ON with a model loaded. Confirm with:

```
curl http://127.0.0.1:1234/v1/models
```
