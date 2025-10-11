import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
from rag_core import init_indices, answer_question
import traceback

load_dotenv()
app = Flask(__name__)

@app.get("/")
def home():
    return "✅ StudySpark RAG Assistant is running."

@app.post("/ask")
def ask():
    data = request.get_json(force=True)
    print("\n📩 Incoming /ask request:")
    print(f"  Raw JSON: {data}")

    user_id = data.get("user_id", 1)
    question = data.get("question", "").strip()
    session_id = data.get("session_id")

    print(f"  ➤ user_id: {user_id}")
    print(f"  ➤ question: '{question}'")
    print(f"  ➤ session_id: {session_id}")

    if not question:
        print("❌ Empty question received.")
        return jsonify({"ok": False, "error": "Empty question"}), 400

    try:
        # ✅ Step 1: Fetch user data from Node
        node_api_url = f"http://localhost:5000/api/data/export/{user_id}"
        print(f"🌐 Fetching Node data from: {node_api_url}")

        node_response = requests.get(node_api_url)
        print(f"  ➤ Node response status: {node_response.status_code}")

        if node_response.status_code != 200:
            print(f"❌ Node API failed: {node_response.text[:300]}")
            return jsonify({
                "ok": False,
                "error": f"Node API returned {node_response.status_code}",
                "details": node_response.text
            }), 500

        db_data = node_response.json()
        print("✅ Node data successfully retrieved.")
        print(f"  ➤ Keys in db_data: {list(db_data.keys())}")
        print(f"  ➤ Number of resources: {len(db_data.get('resources', []))}")
        print(f"  ➤ Number of notes: {len(db_data.get('notes', []))}")
        print(f"  ➤ Number of summaries: {len(db_data.get('summaries', []))}")

        # ✅ Step 2: Build RAG indices
        print("⚙️ Building indices...")
        bm25_index, vec_index = init_indices(db_data)
        print("✅ Indices built successfully.")

        # ✅ Step 3: Ask question via RAG core
        print("💬 Sending to RAG core...")
        result = answer_question(
            db_data,
            bm25_index,
            vec_index,
            f"u{user_id}",
            question,
            session_id=session_id
        )

        print("✅ RAG answered successfully.")
        print(f"  ➤ Answer preview: {result['answer'][:200]}")

        return jsonify({
            "ok": True,
            "answer": result["answer"],
            "session_id": result["session_id"]
        })

    except Exception as e:
        print("\n🔥 ERROR in /ask route:")
        print(traceback.format_exc())  # full traceback
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5004"))
    print(f"🚀 StudySpark Assistant running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
