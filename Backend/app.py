from flask import Flask, request, jsonify, session
from flask_cors import CORS
from backker import run 

app = Flask(__name__)
app.secret_key = "replace_this_with_a_random_secret"
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/api/query", methods=["POST"])
def query():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400



    # Maintain chat history in session
    if "messages" not in session:
        session["messages"] = []
    # Add user message
    session["messages"].append({"role": "user", "content": question})

    try:
        # Pass chat history to run
        answer = run(question, history=session["messages"])
        # Add assistant message
        session["messages"].append({"role": "assistant", "content": answer})
        # Only return the assistant's answer
        return jsonify({"answer": answer})
    except Exception as e:
        app.logger.exception("Error in /api/query")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
