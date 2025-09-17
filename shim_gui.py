import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox
import uvicorn
import json
import os
from typing import Optional
import logging
import sys
from tkinter import scrolledtext
import subprocess
import re
import time
try:
    import psutil  # type: ignore  # Optional: used for port/process management
except Exception:
    psutil = None

# Import the FastAPI app and setter from shim module
import lmstudio_toolchoice_shim as shim


def resource_path(relative: str) -> str:
    """Get absolute path to resource, works for dev and for PyInstaller onefile."""
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(__file__))
    return os.path.join(base_path, relative)


class ServerController:
    def __init__(self):
        self.thread: Optional[threading.Thread] = None
        self.server: Optional[uvicorn.Server] = None
        self._running = False

    def start(self, host: str, port: int):
        if self._running:
            return
        # Disable uvicorn's default logging config in frozen apps to avoid isatty-related formatter failures
        config = uvicorn.Config(shim.app, host=host, port=port, log_level="info", log_config=None)
        self.server = uvicorn.Server(config)

        def run():
            self._running = True
            try:
                self.server.run()
            finally:
                self._running = False

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self):
        if self.server and self._running:
            self.server.should_exit = True
        if self.thread:
            self.thread.join(timeout=5)
        self._running = False

    @property
    def running(self) -> bool:
        return self._running


class ShimGUI(tk.Tk):
    def __init__(self, init_cf_streaming: Optional[bool] = None):
        super().__init__()
        self.title("LM Studio Shim Launcher")
        # VS Code-like sizing and resizable layout
        self.geometry("900x620")
        self.minsize(760, 520)
        self.resizable(True, True)

        # Persistence
        self.config_path = os.path.join(os.path.dirname(__file__), "shim_config.json")
        self._initial_values = self._load_config()

        self.controller = ServerController()

        # Cloudflare streaming state (persisted optional)
        persisted = bool(self._initial_values.get("cf_streaming", False))
        self.cf_stream_enabled = persisted if init_cf_streaming is None else bool(init_cf_streaming)
        try:
            shim.set_cf_streaming_enabled(self.cf_stream_enabled)
        except Exception:
            pass

        # GPT-OSS mode toggle (persisted)
        self.gptoss_enabled = bool(self._initial_values.get("gpt_oss_mode", False) or shim.is_gpt_oss_mode_default())
        try:
            shim.set_gpt_oss_mode_default(self.gptoss_enabled)
        except Exception:
            pass

        # Cline config path for MCP discovery (optional)
        self.cline_cfg_var = tk.StringVar(value=self._initial_values.get("cline_config_path", ""))
        try:
            if self.cline_cfg_var.get().strip():
                shim.set_cline_config_path(self.cline_cfg_var.get().strip())
        except Exception:
            pass

        # Apply theme first
        self._apply_vs_theme()

        self._build_widgets()
        self._update_status()

    def _apply_vs_theme(self):
        style = ttk.Style()
        # Force a theme that supports colors
        try:
            style.theme_use('clam')
        except Exception:
            pass
        colors = {
            'bg': '#1e1e1e',
            'panel': '#252526',
            'control': '#2d2d30',
            'accent': '#0e639c',
            'text': '#d4d4d4',
            'muted': '#9da0a6',
            'border': '#3c3c3c',
            'error': '#f14c4c',
            'success': '#89d185',
        }
        self.configure(bg=colors['bg'])
        style.configure('VS.TFrame', background=colors['panel'])
        style.configure('VS.Header.TFrame', background=colors['control'])
        style.configure('VS.TLabel', background=colors['panel'], foreground=colors['text'])
        style.configure('VS.Header.TLabel', background=colors['control'], foreground=colors['text'], font=('Segoe UI', 12, 'bold'))
        style.configure('VS.TButton', background=colors['control'], foreground=colors['text'], bordercolor=colors['border'])
        style.map('VS.TButton', background=[('active', colors['accent'])])
        style.configure('VS.TCheckbutton', background=colors['panel'], foreground=colors['text'])
        style.configure('VS.TEntry', fieldbackground=colors['control'], foreground=colors['text'])
        style.configure('VS.Status.TLabel', background=colors['panel'], foreground=colors['muted'])

        # Make Entry and Button have some padding
        style.configure('VS.TEntry', padding=4)
        style.configure('VS.TButton', padding=(8, 6))

        self._vs_colors = colors

    def _build_widgets(self):
        colors = self._vs_colors
        root = ttk.Frame(self, style='VS.TFrame')
        root.pack(fill=tk.BOTH, expand=True)
        root.grid_columnconfigure(0, weight=1)

        # Header
        header = ttk.Frame(root, style='VS.Header.TFrame')
        header.grid(row=0, column=0, sticky='nsew')
        header.grid_columnconfigure(0, weight=1)
        title = ttk.Label(header, text='LM Studio Shim for ToolChoice', style='VS.Header.TLabel')
        title.grid(row=0, column=0, sticky='w', padx=12, pady=10)

        # Optional logos
        logos = ttk.Frame(header, style='VS.Header.TFrame')
        logos.grid(row=0, column=1, sticky='e', padx=8)
        cf_img, lm_img = self._load_logos()
        if cf_img is not None:
            ttk.Label(logos, image=cf_img, style='VS.Header.TLabel').grid(row=0, column=0, padx=(0, 6))
        if lm_img is not None:
            ttk.Label(logos, image=lm_img, style='VS.Header.TLabel').grid(row=0, column=1)
        header.grid_columnconfigure(1, weight=0)

        # Content panel
        content = ttk.Frame(root, style='VS.TFrame', padding=12)
        content.grid(row=1, column=0, sticky='nsew')
        root.grid_rowconfigure(1, weight=1)
        content.grid_columnconfigure(1, weight=1)

        # Upstream
        ttk.Label(content, text='LM Studio Base (upstream):', style='VS.TLabel').grid(row=0, column=0, sticky='w')
        self.lmstudio_var = tk.StringVar(value=self._initial_values.get('upstream', 'http://127.0.0.1:1234'))
        e_upstream = ttk.Entry(content, textvariable=self.lmstudio_var, width=48, style='VS.TEntry')
        e_upstream.grid(row=0, column=1, sticky='we', padx=(8, 0))

        # Cline config path (for MCP auto-discovery)
        ttk.Label(content, text='Cline Config (JSON) path:', style='VS.TLabel').grid(row=1, column=0, sticky='w', pady=(8, 0))
        e_cfg = ttk.Entry(content, textvariable=self.cline_cfg_var, width=48, style='VS.TEntry')
        e_cfg.grid(row=1, column=1, sticky='we', padx=(8, 0), pady=(8, 0))
        autod_btn = ttk.Button(content, text='Auto Discover', style='VS.TButton', command=self.auto_discover_cline_config)
        autod_btn.grid(row=1, column=2, sticky='w', padx=(8,0), pady=(8,0))

        # Host/Port
        ttk.Label(content, text='Shim Host:', style='VS.TLabel').grid(row=2, column=0, sticky='w', pady=(8, 0))
        self.host_var = tk.StringVar(value=self._initial_values.get('host', '127.0.0.1'))
        e_host = ttk.Entry(content, textvariable=self.host_var, width=18, style='VS.TEntry')
        e_host.grid(row=2, column=1, sticky='w', padx=(8, 0), pady=(8, 0))

        ttk.Label(content, text='Shim Port:', style='VS.TLabel').grid(row=3, column=0, sticky='w', pady=(8, 0))
        self.port_var = tk.StringVar(value=str(self._initial_values.get('port', 8088)))
        e_port = ttk.Entry(content, textvariable=self.port_var, width=10, style='VS.TEntry')
        e_port.grid(row=3, column=1, sticky='w', padx=(8, 0), pady=(8, 0))

        # Controls
        controls = ttk.Frame(content, style='VS.TFrame')
        controls.grid(row=4, column=0, columnspan=2, sticky='w', pady=(12, 4))

        self.start_btn = ttk.Button(controls, text='Start Server', style='VS.TButton', command=self.start_server)
        self.start_btn.grid(row=0, column=0, padx=(0, 8))

        self.stop_btn = ttk.Button(controls, text='Stop Server', style='VS.TButton', command=self.stop_server, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=(0, 8))

        self.open_btn = ttk.Button(controls, text='Open /v1/models', style='VS.TButton', command=self.open_models, state=tk.DISABLED)
        self.open_btn.grid(row=0, column=2, padx=(0, 8))

        self.copy_btn = ttk.Button(controls, text='Copy Shim URL', style='VS.TButton', command=self.copy_shim_url, state=tk.DISABLED)
        self.copy_btn.grid(row=0, column=3, padx=(0,8))

        # Row 0 extra controls: always present and clickable
        self.kill_port_btn = ttk.Button(controls, text='Stop All on Port', style='VS.TButton', command=self.stop_all_on_port)
        self.kill_port_btn.grid(row=0, column=4, padx=(0, 8))

        self.kill_mcp_btn = ttk.Button(controls, text='Kill mcp-browser-use', style='VS.TButton', command=self.kill_mcp_browser_use)
        self.kill_mcp_btn.grid(row=0, column=5)

        # Keep same colspan regardless; buttons remain enabled even if psutil is missing
        cf_colspan = 6

        self.cf_btn = ttk.Button(controls, text=self._cf_btn_text(), style='VS.TButton', command=self.toggle_cf_streaming)
        self.cf_btn.grid(row=1, column=0, columnspan=cf_colspan, sticky='w', pady=(8, 0))

        # GPT-OSS Mode toggle
        self.gptoss_btn = ttk.Button(controls, text=self._gptoss_btn_text(), style='VS.TButton', command=self.toggle_gptoss_mode)
        self.gptoss_btn.grid(row=2, column=0, columnspan=cf_colspan, sticky='w', pady=(8, 0))

        # Status
        status_frame = ttk.Frame(content, style='VS.TFrame')
        status_frame.grid(row=5, column=0, columnspan=2, sticky='we')
        self.status_var = tk.StringVar(value='Stopped')
        self.status_lbl = ttk.Label(status_frame, textvariable=self.status_var, style='VS.Status.TLabel')
        self.status_lbl.grid(row=0, column=0, sticky='w')

        # Log panel
        log_frame = ttk.Frame(content, style='VS.TFrame')
        log_frame.grid(row=6, column=0, columnspan=2, sticky='nsew', pady=(8, 0))
        content.grid_rowconfigure(6, weight=1)
        ttk.Label(log_frame, text='Logs:', style='VS.TLabel').grid(row=0, column=0, sticky='w')

        self.log_text = scrolledtext.ScrolledText(log_frame, height=14, wrap='word')
        self.log_text.grid(row=1, column=0, sticky='nsew', pady=(6, 0))
        log_frame.grid_rowconfigure(1, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        # Dark style for text widget
        self.log_text.configure(bg=colors['bg'], fg=colors['text'], insertbackground=colors['text'])

        # Quit at bottom
        bottom = ttk.Frame(root, style='VS.TFrame')
        bottom.grid(row=2, column=0, sticky='we')
        ttk.Button(bottom, text='Quit', style='VS.TButton', command=self.on_quit).pack(pady=8)

        # After widgets exist, hook logging to GUI
        self._setup_logging_to_gui()

    # ------------- Auto discovery for Cline config -------------
    def _candidate_cline_config_paths(self) -> list[str]:
        candidates: list[str] = []
        # 1) Env override
        env_path = os.environ.get('CLINE_CONFIG_PATH')
        if env_path and os.path.isfile(env_path):
            candidates.append(env_path)

        # 2) CWD common names
        cwd = os.getcwd()
        for name in ("cline_config.json", "cline.config.json", "cline.json"):
            p = os.path.join(cwd, name)
            if os.path.isfile(p):
                candidates.append(p)

        # 3) .vscode/cline.json in CWD and parents
        def walk_parents(start: str):
            path = os.path.abspath(start)
            seen = set()
            while path not in seen:
                yield path
                seen.add(path)
                parent = os.path.dirname(path)
                if parent == path:
                    break
                path = parent

        for base in walk_parents(cwd):
            p = os.path.join(base, ".vscode", "cline.json")
            if os.path.isfile(p):
                candidates.append(p)

        # 4) User home standard
        home = os.path.expanduser("~")
        p = os.path.join(home, ".cline", "config.json")
        if os.path.isfile(p):
            candidates.append(p)

        # 5) VS Code/VSCodium/Cursor globalStorage for Cline (common real location)
        #    Cline extension ID: saoudrizwan.claude-dev
        #    File of interest: settings/cline_mcp_settings.json
        #    Windows: %APPDATA%\<Product>\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json
        #    macOS:   ~/Library/Application Support/<Product>/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json
        #    Linux:   ~/.config/<Product>/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json
        try:
            products = [
                "Code",
                "Code - Insiders",
                "VSCodium",
                "Code - OSS",
                "Cursor",
            ]
            ext_ns = os.path.join("saoudrizwan.claude-dev", "settings", "cline_mcp_settings.json")
            if os.name == 'nt':
                appdata = os.path.expandvars(os.environ.get('APPDATA', ''))  # C:\Users\<user>\AppData\Roaming
                if appdata:
                    for prod in products:
                        p = os.path.join(appdata, prod, "User", "globalStorage", ext_ns)
                        if os.path.isfile(p):
                            candidates.append(p)
            elif sys.platform == 'darwin':
                base = os.path.expanduser(os.path.join("~", "Library", "Application Support"))
                for prod in products:
                    p = os.path.join(base, prod, "User", "globalStorage", ext_ns)
                    if os.path.isfile(p):
                        candidates.append(p)
            else:
                # Assume XDG-style ~/.config
                base = os.path.expanduser(os.path.join("~", ".config"))
                for prod in products:
                    p = os.path.join(base, prod, "User", "globalStorage", ext_ns)
                    if os.path.isfile(p):
                        candidates.append(p)
        except Exception:
            pass

        # De-dup preserve order
        uniq: list[str] = []
        seen2: set[str] = set()
        for c in candidates:
            if c not in seen2:
                uniq.append(c)
                seen2.add(c)
        return uniq

    def auto_discover_cline_config(self):
        paths = self._candidate_cline_config_paths()
        if not paths:
            messagebox.showinfo("Auto Discover", "No Cline config file found in common locations.")
            return
        # Choose the first candidate
        path = paths[0]
        self.cline_cfg_var.set(path)
        # Apply immediately to shim and persist
        try:
            shim.set_cline_config_path(path)
        except Exception:
            pass
        self._save_config({
            "upstream": self.lmstudio_var.get().strip(),
            "cline_config_path": self.cline_cfg_var.get().strip(),
            "host": self.host_var.get().strip() or "127.0.0.1",
            "port": int(self.port_var.get().strip() or 8088),
            "cf_streaming": self.cf_stream_enabled,
            "gpt_oss_mode": self.gptoss_enabled,
        })
        messagebox.showinfo("Auto Discover", f"Detected Cline config:\n{path}")

    def _load_logos(self):
        """Load and scale logos. Returns (cf_img, lm_img) PhotoImage or (None, None)."""
        cf_img = None
        lm_img = None
        try:
            cf_path = resource_path(os.path.join('images', 'logos', 'CloudflareLogo(2560x846px).png'))
            if os.path.exists(cf_path):
                img = tk.PhotoImage(file=cf_path)
                factor = max(1, int(img.width() / 160))
                cf_img = img.subsample(factor, factor)
                self.logo_cf_img = cf_img  # keep ref
        except Exception:
            cf_img = None
        try:
            lm_path = resource_path(os.path.join('images', 'logos', 'LMStudioHelperCharacter(1296x912px).png'))
            if os.path.exists(lm_path):
                img = tk.PhotoImage(file=lm_path)
                factor = max(1, int(img.width() / 160))
                lm_img = img.subsample(factor, factor)
                self.logo_lm_img = lm_img  # keep ref
        except Exception:
            lm_img = None
        return cf_img, lm_img

    def _setup_logging_to_gui(self):
        class TextHandler(logging.Handler):
            def __init__(self, text_widget: scrolledtext.ScrolledText):
                super().__init__()
                self.text_widget = text_widget
                self.setLevel(logging.INFO)
            def emit(self, record):
                try:
                    msg = self.format(record) + "\n"
                except Exception:
                    msg = str(record) + "\n"
                # Append in GUI thread
                self.text_widget.after(0, self._append, msg)
            def _append(self, msg: str):
                try:
                    self.text_widget.configure(state='normal')
                    self.text_widget.insert('end', msg)
                    self.text_widget.see('end')
                finally:
                    self.text_widget.configure(state='disabled')

        handler = TextHandler(self.log_text)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))

        # Attach to root and key loggers
        for name in ('', 'uvicorn', 'uvicorn.error', 'uvicorn.access', 'lmstudio_shim'):
            lg = logging.getLogger(name)
            lg.setLevel(logging.INFO)
            lg.addHandler(handler)

    def _cf_btn_text(self) -> str:
        return "Disable Cloudflare Streaming" if self.cf_stream_enabled else "Enable Cloudflare Streaming"

    def _gptoss_btn_text(self) -> str:
        return "Disable GPT-OSS Mode" if self.gptoss_enabled else "Enable GPT-OSS Mode"

    def toggle_cf_streaming(self):
        self.cf_stream_enabled = not self.cf_stream_enabled
        try:
            shim.set_cf_streaming_enabled(self.cf_stream_enabled)
        except Exception:
            pass
        self.cf_btn.configure(text=self._cf_btn_text())
        # Persist immediately
        self._save_config({
            "upstream": self.lmstudio_var.get().strip(),
            "cline_config_path": self.cline_cfg_var.get().strip(),
            "host": self.host_var.get().strip() or "127.0.0.1",
            "port": int(self.port_var.get().strip() or 8088),
            "cf_streaming": self.cf_stream_enabled,
            "gpt_oss_mode": self.gptoss_enabled,
        })

    def toggle_gptoss_mode(self):
        self.gptoss_enabled = not self.gptoss_enabled
        try:
            shim.set_gpt_oss_mode_default(self.gptoss_enabled)
        except Exception:
            pass
        self.gptoss_btn.configure(text=self._gptoss_btn_text())
        self._save_config({
            "upstream": self.lmstudio_var.get().strip(),
            "cline_config_path": self.cline_cfg_var.get().strip(),
            "host": self.host_var.get().strip() or "127.0.0.1",
            "port": int(self.port_var.get().strip() or 8088),
            "cf_streaming": self.cf_stream_enabled,
            "gpt_oss_mode": self.gptoss_enabled,
        })

    def start_server(self):
        if self.controller.running:
            return
        base_raw = self.lmstudio_var.get().strip()
        try:
            normalized = shim.set_lmstudio_base(base_raw)
        except Exception as e:
            messagebox.showerror("Invalid Base URL", str(e))
            return
        host = self.host_var.get().strip() or "127.0.0.1"
        try:
            port = int(self.port_var.get())
        except ValueError:
            messagebox.showerror("Invalid Port", "Port must be an integer")
            return

        # Persist settings
        self._save_config({
            "upstream": base_raw,
            "cline_config_path": self.cline_cfg_var.get().strip(),
            "host": host,
            "port": port,
            "cf_streaming": self.cf_stream_enabled,
            "gpt_oss_mode": self.gptoss_enabled,
        })

        # Apply Cline config path
        try:
            shim.set_cline_config_path(self.cline_cfg_var.get().strip() or None)
        except Exception:
            pass

        self.controller.start(host, port)
        self.status_var.set(f"Running at http://{host}:{port} (upstream {normalized})")
        self.status_lbl.configure(foreground=self._vs_colors['success'])
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.open_btn.configure(state=tk.NORMAL)
        self.copy_btn.configure(state=tk.NORMAL)

    def stop_server(self):
        if not self.controller.running:
            return
        self.controller.stop()
        self.status_var.set("Stopped")
        self.status_lbl.configure(foreground=self._vs_colors['error'])
        # Force-refresh UI button states immediately
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.open_btn.configure(state=tk.DISABLED)
        self.copy_btn.configure(state=tk.DISABLED)
        # Also schedule a follow-up check to catch late thread exits
        self.after(200, self._update_status)

    def open_models(self):
        if not self.controller.running:
            return
        host = self.host_var.get().strip() or "127.0.0.1"
        port = self.port_var.get().strip() or "8088"
        webbrowser.open(f"http://{host}:{port}/v1/models")

    def stop_all_on_port(self):
        """Stop our server (if running) and then kill any process bound to the selected port."""
        try:
            # Stop our own server first
            if self.controller.running:
                self.stop_server()
            port_str = self.port_var.get().strip() or "8088"
            port = int(port_str)
        except Exception as e:
            messagebox.showerror("Invalid Port", f"Provide a valid port: {e}")
            return
        # Gather PIDs and terminate using psutil if available, else Windows 'netstat' + 'taskkill' fallback
        pids = set()
        if psutil is not None:
            try:
                for p in psutil.process_iter(attrs=['pid','name','cmdline']):
                    try:
                        conns = p.connections(kind='inet')
                    except Exception:
                        continue
                    for c in conns:
                        try:
                            laddr = getattr(c, 'laddr', None)
                            if not laddr:
                                continue
                            lport = getattr(laddr, 'port', None)
                            if lport is None and isinstance(laddr, tuple) and len(laddr) > 1:
                                lport = laddr[1]
                            if lport == port:
                                pids.add(p.pid)
                        except Exception:
                            continue
            except Exception as e:
                self._append_log_line(f"Failed to enumerate processes via psutil: {e}")
        else:
            # Windows-only fallback using netstat
            if os.name == 'nt':
                try:
                    # Capture all TCP/UDP listeners, filter by :port
                    out = subprocess.check_output(['netstat', '-ano'], text=True, stderr=subprocess.STDOUT)
                    pat = re.compile(rf"\S+\s+\S*:{port}\s+.*?LISTENING\s+(\d+)")
                    for line in out.splitlines():
                        m = pat.search(line)
                        if m:
                            try:
                                pids.add(int(m.group(1)))
                            except Exception:
                                pass
                except Exception as e:
                    self._append_log_line(f"netstat failed: {e}")
            else:
                self._append_log_line("Port kill fallback not implemented for this OS without psutil.")

        if not pids:
            self._append_log_line(f"No processes found listening on port {port}.")
            return

        killed = []
        failed = []
        for pid in sorted(pids):
            if self._terminate_pid(pid):
                killed.append(pid)
            else:
                failed.append(pid)

        self._append_log_line(f"Terminated {len(killed)} process(es) on port {port}: {killed if killed else 'none'}")
        if failed:
            self._append_log_line(f"Failed to terminate PIDs: {failed}. Try running as Administrator.")

    def kill_mcp_browser_use(self):
        """Kill any process whose name or command line mentions 'mcp-browser-use'."""
        needle = 'mcp-browser-use'
        pids = set()
        if psutil is not None:
            for p in psutil.process_iter(attrs=['pid','name','cmdline']):
                try:
                    name = (p.info.get('name') or '').lower()
                    cmd = ' '.join(p.info.get('cmdline') or []).lower()
                    if needle in name or needle in cmd:
                        pids.add(p.pid)
                except Exception:
                    pass
        else:
            if os.name == 'nt':
                try:
                    # Use PowerShell to find by Name or CommandLine containing the needle
                    ps_cmd = [
                        'powershell', '-NoProfile', '-Command',
                        f"$p=Get-CimInstance Win32_Process | Where-Object {{$_.Name -match '{needle}' -or $_.CommandLine -match '{needle}'}}; $p | ForEach-Object {{$_.ProcessId}}"
                    ]
                    out = subprocess.check_output(ps_cmd, text=True, stderr=subprocess.STDOUT)
                    for line in out.split():
                        try:
                            pids.add(int(line.strip()))
                        except Exception:
                            pass
                except Exception as e:
                    self._append_log_line(f"PowerShell process search failed: {e}")
            else:
                try:
                    out = subprocess.check_output(['ps', '-eo', 'pid,comm,args'], text=True, stderr=subprocess.STDOUT)
                    for line in out.splitlines():
                        if needle in line.lower():
                            try:
                                pid = int(line.strip().split(None, 1)[0])
                                pids.add(pid)
                            except Exception:
                                pass
                except Exception as e:
                    self._append_log_line(f"ps search failed: {e}")

        if not pids:
            self._append_log_line("No 'mcp-browser-use' processes found.")
            return

        killed = []
        failed = []
        for pid in sorted(pids):
            if self._terminate_pid(pid):
                killed.append(pid)
            else:
                failed.append(pid)
        self._append_log_line(f"Terminated {len(killed)} 'mcp-browser-use' process(es): {killed if killed else 'none'}")
        if failed:
            self._append_log_line(f"Failed to terminate PIDs: {failed}. Try running as Administrator.")

    def _terminate_pid(self, pid: int) -> bool:
        """Terminate PID using psutil when available, else OS-specific commands. Returns True on success."""
        try:
            if psutil is not None:
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    try:
                        p.wait(timeout=2)
                    except Exception:
                        pass
                    if p.is_running():
                        p.kill()
                    return True
                except Exception:
                    return False
            # Fallbacks
            if os.name == 'nt':
                try:
                    subprocess.check_call(['taskkill', '/PID', str(pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
                except Exception:
                    return False
            else:
                try:
                    os.kill(pid, 15)
                    time.sleep(0.5)
                    # If still alive, force kill
                    os.kill(pid, 9)
                    return True
                except Exception:
                    return False
        except Exception:
            return False

    def _append_log_line(self, text: str):
        try:
            self.log_text.configure(state='normal')
            self.log_text.insert('end', text + "\n")
            self.log_text.see('end')
        finally:
            self.log_text.configure(state='disabled')

    def _update_status(self):
        # Periodically verify thread/server state and enforce button states
        is_running = bool(self.controller.running)
        if is_running:
            # Ensure buttons reflect running state
            if str(self.start_btn['state']) != str(tk.DISABLED):
                self.start_btn.configure(state=tk.DISABLED)
            if str(self.stop_btn['state']) != str(tk.NORMAL):
                self.stop_btn.configure(state=tk.NORMAL)
            if str(self.open_btn['state']) != str(tk.NORMAL):
                self.open_btn.configure(state=tk.NORMAL)
            if str(self.copy_btn['state']) != str(tk.NORMAL):
                self.copy_btn.configure(state=tk.NORMAL)
        else:
            # Stopped state; make sure UI is reset
            self.status_var.set("Stopped")
            self.status_lbl.configure(foreground=self._vs_colors['error'])
            if str(self.start_btn['state']) != str(tk.NORMAL):
                self.start_btn.configure(state=tk.NORMAL)
            if str(self.stop_btn['state']) != str(tk.DISABLED):
                self.stop_btn.configure(state=tk.DISABLED)
            if str(self.open_btn['state']) != str(tk.DISABLED):
                self.open_btn.configure(state=tk.DISABLED)
            if str(self.copy_btn['state']) != str(tk.DISABLED):
                self.copy_btn.configure(state=tk.DISABLED)
        self.after(1000, self._update_status)

    def copy_shim_url(self):
        if not self.controller.running:
            return
        host = self.host_var.get().strip() or "127.0.0.1"
        port = self.port_var.get().strip() or "8088"
        url = f"http://{host}:{port}/v1"
        try:
            self.clipboard_clear()
            self.clipboard_append(url)
            prev = self.copy_btn['text']
            self.copy_btn.configure(text="Copied!")
            self.after(1500, lambda: self.copy_btn.configure(text=prev))
        except Exception as e:
            messagebox.showerror("Copy Failed", str(e))

    def on_quit(self):
        if self.controller.running:
            if not messagebox.askyesno("Quit", "Server is running. Stop and quit?"):
                return
            self.controller.stop()
        self.destroy()

    # ---------------- Persistence Helpers -----------------
    def _load_config(self):
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save_config(self, data):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            # Non-fatal; show once
            print(f"Warning: could not save config: {e}")

if __name__ == "__main__":
    app = ShimGUI()
    app.mainloop()
