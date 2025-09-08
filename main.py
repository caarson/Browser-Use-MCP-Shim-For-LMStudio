import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="LM Studio Shim Launcher")
    parser.add_argument('--gui', action='store_true', help='Launch the Tkinter GUI')
    parser.add_argument('--no-gui', action='store_true', help='Force CLI server mode even with no args')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Shim server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8088, help='Shim server port (default: 8088)')
    parser.add_argument('--lmstudio-base', type=str, default=None, help='LM Studio base URL (e.g. http://127.0.0.1:1234)')
    args = parser.parse_args()

    # If no arguments given at all, default to GUI for user convenience
    if (len(sys.argv) == 1) or args.gui:
        if not args.no_gui:  # allow override
            import shim_gui
            shim_gui.ShimGUI().mainloop()
            return

    import lmstudio_toolchoice_shim as shim
    if args.lmstudio_base:
        shim.set_lmstudio_base(args.lmstudio_base)
    import uvicorn
    print(f"Starting shim server at http://{args.host}:{args.port} (upstream: {shim.LMSTUDIO_BASE})")
    uvicorn.run(shim.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
