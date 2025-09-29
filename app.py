# app.py
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
load_dotenv()

from mock_db import DB
from rag_core import init_indices, answer_question

app = Flask(__name__)

# Build indices once at startup (demo)
bm25_index, vec_index = init_indices(DB)

@app.get("/")
def home():
    users = [u["id"] for u in DB["users"]]
    return render_template("index.html", users=users)

@app.post("/ask")
def ask():
    data = request.get_json(force=True)
    user_id = data.get("user_id", "u1")
    question = data.get("question", "").strip()
    session_id = data.get("session_id")
    if not question:
        return jsonify({"ok": False, "error": "Empty question"}), 400
    try:
        result = answer_question(DB, bm25_index, vec_index, user_id, question, session_id=session_id)
        return jsonify({"ok": True, "answer": result["answer"], "session_id": result["session_id"]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
