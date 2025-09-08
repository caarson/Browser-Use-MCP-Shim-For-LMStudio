import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox
import uvicorn
import json
import os

# Import the FastAPI app and setter from shim module
import lmstudio_toolchoice_shim as shim

class ServerController:
    def __init__(self):
        self.thread = None
        self.server = None
        self._running = False

    def start(self, host: str, port: int):
        if self._running:
            return
        config = uvicorn.Config(shim.app, host=host, port=port, log_level="info")
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
    def running(self):
        return self._running

class ShimGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LM Studio Shim Launcher")
        self.geometry("520x310")
        self.resizable(False, False)

        # Persistence
        self.config_path = os.path.join(os.path.dirname(__file__), "shim_config.json")
        self._initial_values = self._load_config()

        self.controller = ServerController()

        self._build_widgets()
        self._update_status()

    def _build_widgets(self):
        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LM Studio Base URL
        ttk.Label(frm, text="LM Studio Base (upstream):").grid(row=0, column=0, sticky="w")
        self.lmstudio_var = tk.StringVar(value=self._initial_values.get("upstream", "http://127.0.0.1:1234"))
        ttk.Entry(frm, textvariable=self.lmstudio_var, width=42).grid(row=0, column=1, sticky="we")

        # Shim Host
        ttk.Label(frm, text="Shim Host:").grid(row=1, column=0, sticky="w")
        self.host_var = tk.StringVar(value=self._initial_values.get("host", "127.0.0.1"))
        ttk.Entry(frm, textvariable=self.host_var, width=20).grid(row=1, column=1, sticky="w")

        # Shim Port
        ttk.Label(frm, text="Shim Port:").grid(row=2, column=0, sticky="w")
        self.port_var = tk.StringVar(value=str(self._initial_values.get("port", 8088)))
        ttk.Entry(frm, textvariable=self.port_var, width=10).grid(row=2, column=1, sticky="w")

        # Buttons
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)

        self.start_btn = ttk.Button(btn_frame, text="Start Server", command=self.start_server)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop Server", command=self.stop_server, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.open_btn = ttk.Button(btn_frame, text="Open /v1/models", command=self.open_models, state=tk.DISABLED)
        self.open_btn.grid(row=0, column=2, padx=5)

        self.copy_btn = ttk.Button(btn_frame, text="Copy Shim URL", command=self.copy_shim_url, state=tk.DISABLED)
        self.copy_btn.grid(row=0, column=3, padx=5)

        # Status
        ttk.Label(frm, text="Status:").grid(row=4, column=0, sticky="nw")
        self.status_var = tk.StringVar(value="Stopped")
        self.status_lbl = ttk.Label(frm, textvariable=self.status_var, foreground="red")
        self.status_lbl.grid(row=4, column=1, sticky="w")

        # Quit button at bottom
        ttk.Button(frm, text="Quit", command=self.on_quit).grid(row=5, column=0, columnspan=2, pady=10)

        for i in range(2):
            frm.grid_columnconfigure(i, weight=1)

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
            "port": port
        })

        self.controller.start(host, port)
        self.status_var.set(f"Running at http://{host}:{port} (upstream {normalized})")
        self.status_lbl.configure(foreground="green")
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.open_btn.configure(state=tk.NORMAL)
        self.copy_btn.configure(state=tk.NORMAL)

    def stop_server(self):
        if not self.controller.running:
            return
        self.controller.stop()
        self.status_var.set("Stopped")
        self.status_lbl.configure(foreground="red")
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.open_btn.configure(state=tk.DISABLED)
        self.copy_btn.configure(state=tk.DISABLED)

    def open_models(self):
        if not self.controller.running:
            return
        host = self.host_var.get().strip() or "127.0.0.1"
        port = self.port_var.get().strip() or "8088"
        webbrowser.open(f"http://{host}:{port}/v1/models")

    def _update_status(self):
        # Periodically verify thread state
        if self.controller.running:
            pass
        else:
            # If thread ended unexpectedly
            if self.stop_btn['state'] == tk.NORMAL and not self.controller.running:
                self.status_var.set("Stopped")
                self.status_lbl.configure(foreground="red")
                self.start_btn.configure(state=tk.NORMAL)
                self.stop_btn.configure(state=tk.DISABLED)
                self.open_btn.configure(state=tk.DISABLED)
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
