#!/usr/bin/env python3
"""
Launches the Mini-EDR web dashboard.

Usage:
    python dashboard.py
    python dashboard.py --port 8080 --host 0.0.0.0
"""
import argparse
from dashboard import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-EDR dashboard server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    app = create_app()
    print(f"Mini-EDR dashboard → http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
