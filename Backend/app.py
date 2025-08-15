import os
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from datetime import timedelta

# Import your backend entrypoint
from backker import run  # <- matches your file name exactly

app = Flask(__name__)

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")
app.permanent_session_lifetime = timedelta(days=1)

# Local dev cookie settings (relax for http://localhost)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,  # set True only when serving over HTTPS
)

# Allow your React dev server(s) and send cookies
CORS(
    app,
    resources={r"/api/*": {"origins": ["http://localhost:5173", "http://localhost:3000"]}},
    supports_credentials=True,
)

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/query", methods=["POST"])
def query():
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "Missing 'question' in JSON body."}), 400

        # Initialize / fetch chat history from session
        if "messages" not in session:
            session["messages"] = []

        # Append user message
        session["messages"].append({"role": "user", "content": question})
        session.modified = True  # ensure cookie is updated

        # Call your backend with optional history
        answer = run(question, history=session["messages"]) or ""

        # Append assistant message
        session["messages"].append({"role": "assistant", "content": answer})
        session.modified = True

        return jsonify({"answer": answer}), 200

    except Exception as e:
        # Don’t leak internals to client; log locally if you want
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Run on 5001 to avoid conflicts with other local services
    app.run(host="0.0.0.0", port=5001, debug=True)
