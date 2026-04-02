"""
Flask application for the Mini-EDR dashboard.
"""
from __future__ import annotations

import threading
from dataclasses import asdict

from flask import Flask, jsonify, render_template, request

from dashboard.state import AppState
from dashboard.monitor import monitor_loop

_state = AppState()


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    @app.get("/")
    def index():
        return render_template("index.html")

    # ------------------------------------------------------------------
    # Status & alerts
    # ------------------------------------------------------------------

    @app.get("/api/status")
    def api_status():
        with _state.lock:
            return jsonify(_state.to_status_dict())

    @app.get("/api/alerts")
    def api_alerts():
        limit = min(int(request.args.get("limit", 50)), 200)
        with _state.lock:
            alerts = [asdict(a) for a in list(_state.alerts)[:limit]]
        return jsonify(alerts)

    # ------------------------------------------------------------------
    # Monitor control
    # ------------------------------------------------------------------

    @app.post("/api/start")
    def api_start():
        data = request.get_json(silent=True) or {}

        with _state.lock:
            if _state.status == "running":
                return jsonify({"ok": False, "error": "Already running"}), 400

            # Apply optional runtime overrides from the request body
            if "model_path" in data:
                _state.model_path = data["model_path"]
            if "threshold" in data and data["threshold"] is not None:
                _state.threshold = float(data["threshold"])
            if "interval" in data:
                _state.interval = float(data["interval"])
            if "log_path" in data:
                _state.log_path = data["log_path"]

            _state.stop_event.clear()
            _state.status = "running"

        t = threading.Thread(target=monitor_loop, args=(_state,), daemon=True)
        with _state.lock:
            _state.set_thread(t)
        t.start()

        return jsonify({"ok": True})

    @app.post("/api/stop")
    def api_stop():
        with _state.lock:
            if _state.status != "running":
                return jsonify({"ok": False, "error": "Not running"}), 400

        _state.stop_event.set()
        return jsonify({"ok": True})

    # ------------------------------------------------------------------
    # Config — read
    # ------------------------------------------------------------------

    @app.get("/api/config")
    def api_config():
        with _state.lock:
            return jsonify(_state.to_config_dict())

    # ------------------------------------------------------------------
    # Config — known process names
    # ------------------------------------------------------------------

    @app.post("/api/config/names/add")
    def api_names_add():
        name = (request.get_json(silent=True) or {}).get("name", "").strip()
        if not name:
            return jsonify({"ok": False, "error": "name is required"}), 400
        with _state.lock:
            _state.known_names.add(name)
            _state.save_config()
        return jsonify({"ok": True, "name": name})

    @app.post("/api/config/names/remove")
    def api_names_remove():
        name = (request.get_json(silent=True) or {}).get("name", "").strip()
        if not name:
            return jsonify({"ok": False, "error": "name is required"}), 400
        with _state.lock:
            _state.known_names.discard(name)
            _state.save_config()
        return jsonify({"ok": True, "name": name})

    # ------------------------------------------------------------------
    # Config — common ports
    # ------------------------------------------------------------------

    @app.post("/api/config/ports/add")
    def api_ports_add():
        raw = (request.get_json(silent=True) or {}).get("port")
        try:
            port = int(raw)
            assert 1 <= port <= 65535
        except Exception:
            return jsonify({"ok": False, "error": "port must be an integer 1-65535"}), 400
        with _state.lock:
            _state.common_ports.add(port)
            _state.save_config()
        return jsonify({"ok": True, "port": port})

    @app.post("/api/config/ports/remove")
    def api_ports_remove():
        raw = (request.get_json(silent=True) or {}).get("port")
        try:
            port = int(raw)
        except Exception:
            return jsonify({"ok": False, "error": "port must be an integer"}), 400
        with _state.lock:
            _state.common_ports.discard(port)
            _state.save_config()
        return jsonify({"ok": True, "port": port})

    # ------------------------------------------------------------------
    # Config — runtime settings update
    # ------------------------------------------------------------------

    @app.post("/api/config/settings")
    def api_settings():
        data = request.get_json(silent=True) or {}
        with _state.lock:
            if "model_path" in data:
                _state.model_path = data["model_path"]
            if "threshold" in data:
                _state.threshold = float(data["threshold"]) if data["threshold"] not in (None, "") else None
            if "interval" in data:
                _state.interval = float(data["interval"])
            if "log_path" in data:
                _state.log_path = data["log_path"]
            _state.save_config()
        return jsonify({"ok": True})

    return app
