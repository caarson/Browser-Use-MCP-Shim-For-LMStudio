import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(description="LM Studio Shim Launcher")
    parser.add_argument('--gui', action='store_true', help='Launch the Tkinter GUI')
    parser.add_argument('--no-gui', action='store_true', help='Force CLI server mode even with no args')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Shim server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8088, help='Shim server port (default: 8088)')
    parser.add_argument('--lmstudio-base', type=str, default=None, help='LM Studio base URL (e.g. http://127.0.0.1:1234)')
    parser.add_argument('--cf-streaming', type=str, choices=['on','off'], default=None, help='Enable Cloudflare upstream streaming (stitch into single response)')
    parser.add_argument('--gpt-oss', type=str, choices=['on','off'], default=None, help='Enable GPT-OSS Mode (normalize tool calls)')
    args = parser.parse_args()

    # If no arguments given at all, default to GUI for user convenience
    if (len(sys.argv) == 1) or args.gui:
        if not args.no_gui:  # allow override
            # On Windows, set AppUserModelID before any windows are created to ensure taskbar icon grouping
            if os.name == 'nt':
                try:
                    import ctypes  # type: ignore
                    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ClineShim.ShimServer")
                except Exception:
                    pass
            import shim_gui
            init_state = None
            if args.cf_streaming is not None:
                init_state = (args.cf_streaming == 'on')
            shim_gui.ShimGUI(init_cf_streaming=init_state).mainloop()
            return

    import lmstudio_toolchoice_shim as shim
    if args.lmstudio_base:
        shim.set_lmstudio_base(args.lmstudio_base)
    # Apply CF streaming state for CLI mode if provided
    if args.cf_streaming is not None:
        shim.set_cf_streaming_enabled(args.cf_streaming == 'on')
    # Apply GPT-OSS mode default
    if args.gpt_oss is not None:
        shim.set_gpt_oss_mode_default(args.gpt_oss == 'on')
    import uvicorn
    print(f"Starting shim server at http://{args.host}:{args.port} (upstream: {shim.LMSTUDIO_BASE})")
    # Disable uvicorn's default colorized logging to avoid isatty errors under one-file EXE
    uvicorn.run(shim.app, host=args.host, port=args.port, log_config=None, log_level="info")


if __name__ == "__main__":
    main()
