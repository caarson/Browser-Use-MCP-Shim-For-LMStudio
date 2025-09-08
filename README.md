# Browser-Use-MCP-Shim-For-LMStudio

**Setup (CLI):**

Install dependencies:

```
pip install -r requirements.txt
```

Run the shim server directly (default host 127.0.0.1:8088 pointing to LM Studio at 127.0.0.1:1234):

```
python main.py
```

## GUI Launcher

A small Tkinter GUI (`shim_gui.py`) lets you:

* Enter the upstream LM Studio base URL (with or without protocol /v1 suffix)
* Choose the shim host / port
* Start / Stop the FastAPI shim server
* Open the `/v1/models` endpoint in your browser
* Copy the shim URL to clipboard (button)
* Remembers last upstream/host/port in `shim_config.json`

Launch the GUI:

```
python shim_gui.py
```

When you click Start, the base URL is normalized (ensuring it ends with `/v1`). Stop cleanly terminates the background server thread.

**Installation:**

Point your MCP at the shim:

Update cline_mcp_settings.json for the browser-use server,

```{
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

**Keep LM Studio’s local server ON and a model loaded. Confirm the model id with:**
`curl http://127.0.0.1:1234/v1/models`
