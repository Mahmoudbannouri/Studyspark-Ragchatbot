# StudySpark — Assistant Chat (RAG + Gemini)

An intelligent educational assistant powered by Retrieval-Augmented Generation (RAG) and Google's Gemini model. This system provides contextually aware responses by combining efficient document retrieval with state-of-the-art AI generation.

## 🌟 Key Features

- **Hybrid Retrieval System**:
  - BM25 text search for keyword matching
  - Gemini `text-embedding-004` for semantic understanding
  - Smart context ranking and selection

- **Intelligent Interactions**:
  - Maintains chat session history for contextual responses
  - Automatic onboarding mode for new users
  - Direct database queries for "latest" and "list" commands
  - Role-based access control (student/educator)

- **Advanced AI Integration**:
  - Powered by Google's Gemini model
  - Automatic model discovery and fallbacks
  - Optimized chunk management for context

## 🛠️ Technical Stack

- **Backend Framework**: Flask 3.0.3
- **AI Model**: Google Gemini (version 0.7.2)
- **Search Engine**: rank-bm25 0.2.2
- **Data Processing**: NumPy 2.x
- **Configuration**: python-dotenv 1.0.1
- **Data Validation**: pydantic 2.9.2

## 📋 Prerequisites

- Python 3.8 or higher
- Google AI (Gemini) API key
- pip for package management

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Mahmoudbannouri/Studyspark-Ragchatbot.git
   cd Studyspark-Ragchatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   Create a `.env` file with:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```bash
   python app.py
   ```

Visit `http://localhost:5000` to start using the assistant.

## 🏗️ Project Structure

- `app.py`: Flask application setup and routing
- `rag_core.py`: Core RAG implementation and Gemini integration
- `mock_db.py`: Demo database implementation
- `templates/`: Web interface templates
- `static/`: CSS and static assets
- `requirements.txt`: Project dependencies

## 💡 Usage Guide

1. Access the web interface
2. Select your user ID (demo mode)
3. Start asking questions or use commands:
   - Regular questions get AI-powered responses
   - "latest X" for recent items
   - "list X" for item listings

## 🔧 Configuration Options

Environment variables (in `.env`):
- `EMBEDDING_MODEL`: Set embedding model (default: "models/text-embedding-004")
- `GENERATION_MODEL`: Set Gemini model (default: "gemini-1.5-flash")
- `GEMINI_API_KEY`: Your Google AI API key

## 📝 Development Notes

- Demo implementation with mock database
- Production deployment should include:
  - Proper authentication system
  - Persistent database
  - Rate limiting
  - Error handling
  - Logging system

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.rk — Assistant Chat (RAG + Gemini)

A minimal end-to-end assistant for StudySpark with hybrid RAG, Gemini generation, session history, onboarding for new users, and natural answers for “latest” and “list” requests.

## Features
- Hybrid retrieval: BM25 + Gemini `text-embedding-004`
- Session chat history in DB; used in prompts
- Onboarding mode when users have no data (short, actionable guidance)
- Direct DB answers for “latest X” and “list X” (no LLM)
- Gemini model discovery + fallbacks
- Admin privacy guardrails (no student content via chat)

## Requirements
- Python 3.10+
- Packages: see `requirements.txt`

## Quickstart
```bash
python -m venv .venv
.\.venv\Scripts\activate  # on Windows
# source .venv/bin/activate # on macOS/Linux
pip install -r requirements.txt

# Create .env in project root
# Required
# GEMINI_API_KEY=your_key_here
# Optional overrides
# GENERATION_MODEL=gemini-1.5-flash
# EMBEDDING_MODEL=models/text-embedding-004
# PORT=5000

python app.py
# Open http://localhost:5000
```

## Configuration (.env)
- `GEMINI_API_KEY` (required)
- `GENERATION_MODEL` (optional, default `gemini-1.5-flash`)
- `EMBEDDING_MODEL` (optional, default `models/text-embedding-004`)
- `PORT` (optional, default `5000`)

If an alias model 404s, upgrade the SDK and set a fully qualified model name from ListModels.

## Architecture
- `app.py`: Flask server, `/ask` endpoint
- `rag_core.py`: indices, retrieval, prompting, model calls, sessions, onboarding, direct DB answers
- `mock_db.py`: demo data store
- `templates/index.html`, `static/style.css`: simple UI

Flow per question:
1) Detect list/latest intents → answer from DB when possible
2) Else retrieve (BM25 + vector) → build prompt with recent chat
3) Call Gemini with fallbacks → persist and index turn

## API
POST `/ask`
Request:
```json
{
  "user_id": "u1",
  "question": "your question",
  "session_id": "optional-previous-session-id"
}
```
Response:
```json
{
  "ok": true,
  "answer": "assistant reply",
  "session_id": "persisted-session-id"
}
```

## Usage examples
```bash
# New chat
curl -s -X POST http://localhost:5000/ask -H "Content-Type: application/json" \
  -d '{"user_id":"u1","question":"help me plan my study week"}'

# Continue a session
curl -s -X POST http://localhost:5000/ask -H "Content-Type: application/json" \
  -d '{"user_id":"u1","session_id":"<id>","question":"show all my quizzes"}'

# Ask for the latest item
curl -s -X POST http://localhost:5000/ask -H "Content-Type: application/json" \
  -d '{"user_id":"u1","session_id":"<id>","question":"what is my latest note"}'
```

## Data model (mock)
- `users` (id, name, role)
- `resources` (id, user_id, title, content)
- `notes` (id, user_id, title, content)
- `summaries` (id, user_id, source, length, content)
- `flashcards` (id, user_id, deck, q, a)
- `quiz` (id, user_id, subject, questions, attempts)
- `study_plans` (id, user_id, goal, tasks)
- `qa_sessions` (id, user_id, messages[], chunks_used[], ts)

## Admin behavior
- Admins do not access students’ content via chat
- Suitable admin chat topics: platform capabilities, policies, high-level summaries
- For real analytics (counts, usage, billing), add dedicated endpoints backed by a real DB

Suggested future endpoints:
- `GET /admin/stats` (platform totals)
- `GET /admin/users` (paginated list)
- `GET /sessions?user_id=...` and `GET /sessions/<id>` (transcripts)

## Troubleshooting
- Model 404: `pip install -U google-generativeai`, set `GENERATION_MODEL` from ListModels
- No data yet: onboarding answers should appear; verify `GEMINI_API_KEY`
- Embeddings error: verify `EMBEDDING_MODEL=models/text-embedding-004`

## Next steps
- Replace mock DB with Postgres
- CRUD APIs + UI for resources/notes/flashcards/quiz/plans
- Auth, roles, and per-user isolation
- Persistent usage/quota tracking and billing
- File processing (PDF, audio/video transcription)
- Rich onboarding UI actions (upload, create note, generate flashcards, start plan)
 (PDF, audio/video transcription)
- Rich onboarding UI actions (upload, create note, generate flashcards, start plan)
