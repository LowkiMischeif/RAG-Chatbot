# app.py
import os
import logging
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

# Import your refactored backend (saved as backker.py from my last message)
import backker  # make sure this file is in the same folder

# ------------------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------------------
def create_app() -> Flask:
    app = Flask(__name__)

    # CORS: allow your Vite dev server (or everything via env)
    cors_origins = os.getenv("CORS_ORIGINS", "*")
    CORS(app, resources={r"/api/*": {"origins": cors_origins}})

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s ─ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app.logger.setLevel(logging.INFO)

    # Initialize the retriever/chain ONCE per process (fast path on unchanged data)
    backker.initialize()

    # -------------------------- Routes ----------------------------------------

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok"), 200

    @app.post("/api/query")
    def api_query():
        """
        Body:
        {
          "question": "string",
          "history": [{"role":"user|assistant","content":"..."}]  // optional
        }
        """
        data: Dict[str, Any] = request.get_json(silent=True) or {}
        question: str = (data.get("question") or data.get("q") or "").strip()
        history: Optional[List[Dict[str, str]]] = data.get("history")

        if not question:
            return jsonify(error="Missing 'question'."), 400

        try:
            answer = backker.run(question, history)
            return jsonify(answer=answer), 200
        except Exception:
            app.logger.exception("Query failed")
            return jsonify(error="Internal server error"), 500

    @app.post("/api/reindex")
    def api_reindex():
        """
        Rebuild the vector index (use when CSVs change).
        Protect with a simple header token in production:
          - Set ADMIN_TOKEN=... in environment
          - Send header: X-Admin-Token: <token>
        """
        expected = os.getenv("ADMIN_TOKEN")
        provided = request.headers.get("X-Admin-Token")
        if expected and provided != expected:
            return jsonify(error="Unauthorized"), 401

        try:
            backker.initialize(force_rebuild=True)
            return jsonify(status="reindexed"), 200
        except Exception:
            app.logger.exception("Reindex failed")
            return jsonify(error="Reindex failed"), 500

    return app


# ------------------------------------------------------------------------------
# Dev entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    app = create_app()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5001"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
