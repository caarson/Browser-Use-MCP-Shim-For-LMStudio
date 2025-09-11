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

        # Host/Port
        ttk.Label(content, text='Shim Host:', style='VS.TLabel').grid(row=1, column=0, sticky='w', pady=(8, 0))
        self.host_var = tk.StringVar(value=self._initial_values.get('host', '127.0.0.1'))
        e_host = ttk.Entry(content, textvariable=self.host_var, width=18, style='VS.TEntry')
        e_host.grid(row=1, column=1, sticky='w', padx=(8, 0), pady=(8, 0))

        ttk.Label(content, text='Shim Port:', style='VS.TLabel').grid(row=2, column=0, sticky='w', pady=(8, 0))
        self.port_var = tk.StringVar(value=str(self._initial_values.get('port', 8088)))
        e_port = ttk.Entry(content, textvariable=self.port_var, width=10, style='VS.TEntry')
        e_port.grid(row=2, column=1, sticky='w', padx=(8, 0), pady=(8, 0))

        # Controls
        controls = ttk.Frame(content, style='VS.TFrame')
        controls.grid(row=3, column=0, columnspan=2, sticky='w', pady=(12, 4))

        self.start_btn = ttk.Button(controls, text='Start Server', style='VS.TButton', command=self.start_server)
        self.start_btn.grid(row=0, column=0, padx=(0, 8))

        self.stop_btn = ttk.Button(controls, text='Stop Server', style='VS.TButton', command=self.stop_server, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=(0, 8))

        self.open_btn = ttk.Button(controls, text='Open /v1/models', style='VS.TButton', command=self.open_models, state=tk.DISABLED)
        self.open_btn.grid(row=0, column=2, padx=(0, 8))

        self.copy_btn = ttk.Button(controls, text='Copy Shim URL', style='VS.TButton', command=self.copy_shim_url, state=tk.DISABLED)
        self.copy_btn.grid(row=0, column=3)

        self.cf_btn = ttk.Button(controls, text=self._cf_btn_text(), style='VS.TButton', command=self.toggle_cf_streaming)
        self.cf_btn.grid(row=1, column=0, columnspan=4, sticky='w', pady=(8, 0))

        # Status
        status_frame = ttk.Frame(content, style='VS.TFrame')
        status_frame.grid(row=4, column=0, columnspan=2, sticky='we')
        self.status_var = tk.StringVar(value='Stopped')
        self.status_lbl = ttk.Label(status_frame, textvariable=self.status_var, style='VS.Status.TLabel')
        self.status_lbl.grid(row=0, column=0, sticky='w')

        # Log panel
        log_frame = ttk.Frame(content, style='VS.TFrame')
        log_frame.grid(row=5, column=0, columnspan=2, sticky='nsew', pady=(8, 0))
        content.grid_rowconfigure(5, weight=1)
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
            "host": self.host_var.get().strip() or "127.0.0.1",
            "port": int(self.port_var.get().strip() or 8088),
            "cf_streaming": self.cf_stream_enabled,
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
            "host": host,
            "port": port,
            "cf_streaming": self.cf_stream_enabled,
        })

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
