# Cline-Shim-for-LMStudio

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

## GPT-OSS Mode

When enabled, the shim injects a contract and normalizes GPT-OSS outputs into Cline’s canonical tool format.

- Accepts OpenAI-style `tool_calls[]` or plain JSON blocks in the assistant message
- Produces exactly one of:
  - `{"tool":"<name>","args":{...}}`
  - `{"final":"<answer>"}`
- Validates against built-in tools and any MCP tools provided via headers
- Retries once on invalid JSON/args with a corrective system message; falls back to a final text otherwise

Built-in tools: `write_to_file`, `read_file`, `replace_in_file`, `search_files`, `list_files`, `execute_command`, `list_code_definition_names`, `use_mcp_tool`, `access_mcp_resource`, `ask_followup_question`, `attempt_completion`.

### Toggle

- GUI button: Enable/Disable GPT-OSS Mode
- CLI flag: `--gpt-oss on|off`
- Per-request header: `X-GPT-OSS-MODE: true|false`

### MCP Tools

Provide runtime tool allow-list and optional schemas via headers:

- `X-MCP-TOOLS: run_browser_agent,run_deep_research`
- Optional JSON:
  - `X-MCP-TOOLS-JSON`: JSON list of names or object `{ name: schema }`
  - `X-MCP-TOOL-SCHEMAS`: JSON object `{ name: { argName: typeName } }` where typeName in `string|number|integer|boolean|object`

### Telemetry logs

- `gptoss.normalized`
- `gptoss.retry`
- `gptoss.multiple_calls`
- `gptoss.fallback_final`

### Tests

Run adapter tests:

```
python -m pytest -q
```

## MCP tool discovery (Cline config)

The shim can auto-discover available MCP tools by reading Cline's MCP settings file. This augments the built-ins and any tools provided via headers.

How it works:
- In the GUI, set the path next to `Cline Config (JSON) path` or click `Auto Discover` to find it automatically.
- The shim reads and caches the config with a short TTL and reloads when the file's mtime changes.
- Tools discovered here are merged with any per-request tool headers.

Override via env var:
- `CLINE_CONFIG_PATH` can be set to a full file path; the GUI and shim will use that value.

### Where to find the Cline config

Cline stores its MCP configuration inside the editor's global storage under the Cline extension namespace `saoudrizwan.claude-dev`:
- File name: `cline_mcp_settings.json`
- Subfolder: `settings`

Windows (VS Code/VSCodium/Cursor):
- `%APPDATA%\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`
- `%APPDATA%\Code - Insiders\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`
- `%APPDATA%\VSCodium\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`
- `%APPDATA%\Cursor\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`

macOS:
- `~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- `~/Library/Application Support/Code - Insiders/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- `~/Library/Application Support/VSCodium/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- `~/Library/Application Support/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`

Linux:
- `~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- `~/.config/Code - Insiders/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- `~/.config/VSCodium/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- `~/.config/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`

Also checked as fallbacks (lower priority):
- Project: `./cline_config.json`, `./cline.config.json`, `./cline.json`, and `./.vscode/cline.json`
- Home: `~/.cline/config.json`

Tip (Windows PowerShell): quickly check common locations

```
$targets = @(
  "$env:APPDATA\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json",
  "$env:APPDATA\Code - Insiders\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json",
  "$env:APPDATA\VSCodium\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json",
  "$env:APPDATA\Cursor\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json"
)
$targets | ForEach-Object { if (Test-Path $_) { Write-Host "Found: $_" } else { Write-Host "Missing: $_" } }
```

If Auto Discover still can’t find the file, paste the discovered path into the GUI field or set `CLINE_CONFIG_PATH` and restart the shim.

