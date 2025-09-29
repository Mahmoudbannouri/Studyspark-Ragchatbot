# rag_core.py — RAG + Tutor (Gemini-only: generation + embeddings)
import os, re, uuid, time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
import google.generativeai as genai

# -------------------------
# Env & model configuration
# -------------------------
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
# Default to a stable, currently available model name. Allow override via env.
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-1.5-flash")
ROLE_ADMIN_SEE_ALL = False  # demo flag; enforce in your auth layer in prod

# Configure Gemini (do NOT pass api_version; 0.7.x handles v1 automatically)
API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=API_KEY)
# -------------------------
# Roles
# -------------------------
def get_user_role(db: Dict[str, Any], user_id: str) -> str:
    for u in db.get("users", []):
        if u.get("id") == user_id:
            return u.get("role", "student")
    return "student"


# -------------------------
# Data model & chunking
# -------------------------
@dataclass
class DocChunk:
    id: str
    user_id: str
    kind: str
    source_id: str
    title: str
    text: str

def chunk_text(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks

def build_corpus(db: Dict[str, Any]) -> List[DocChunk]:
    corpus: List[DocChunk] = []
    def add_chunks(kind, rec, content, title_key="title"):
        uid = rec["user_id"]; sid = rec["id"]; title = rec.get(title_key, kind)
        for ch in chunk_text(content):
            corpus.append(DocChunk(str(uuid.uuid4()), uid, kind, sid, title, ch))

    for r in db["resources"]: add_chunks("resource", r, r["content"])
    for n in db["notes"]:     add_chunks("note", n, n["content"])
    for s in db.get("summaries", []): add_chunks("summary", s, s["content"], title_key="source")
    for f in db.get("flashcards", []):
        joined = f"Q: {f['q']}\nA: {f['a']}\nDeck: {f['deck']}"
        add_chunks("flashcard", f, joined, title_key="deck")
    for qz in db.get("quiz", []):
        last = qz.get("attempts", [{}])[-1] if qz.get("attempts") else {}
        content = f"Subject: {qz['subject']}\nQuestions: {qz['questions']}\nLastScore: {last.get('score','N/A')}"
        add_chunks("quiz", qz, content, title_key="subject")
    for sp in db.get("study_plans", []):
        tasks = "\n".join([f"- {t['title']} ({t['duration_min']} min, {t['priority']})" for t in sp["tasks"]])
        content = f"Goal: {sp['goal']}\nTasks:\n{tasks}"
        add_chunks("study_plan", sp, content, title_key="goal")
    return corpus

def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())

# -------------------------
# BM25 per-user (lexical)
# -------------------------
class UserBM25:
    def __init__(self):
        self.user_docs   = defaultdict(list)
        self.user_chunks = defaultdict(list)
        self.user_bm25   = {}

    def add(self, chunk: DocChunk):
        self.user_docs[chunk.user_id].append(chunk.text)
        self.user_chunks[chunk.user_id].append(chunk)

    def build(self):
        for uid, docs in self.user_docs.items():
            tokenized = [tokenize(d) for d in docs]
            self.user_bm25[uid] = BM25Okapi(tokenized)

    def add_chunks_and_rebuild_user(self, uid: str, new_chunks: List["DocChunk"]):
        for ch in new_chunks:
            self.user_docs[ch.user_id].append(ch.text)
            self.user_chunks[ch.user_id].append(ch)
        # Rebuild only this user's BM25
        docs = self.user_docs.get(uid, [])
        if docs:
            tokenized = [tokenize(d) for d in docs]
            self.user_bm25[uid] = BM25Okapi(tokenized)

    def search(self, uid: str, query: str, k: int) -> List[Tuple[DocChunk, float]]:
        if ROLE_ADMIN_SEE_ALL:
            all_docs, all_chunks = [], []
            for docs in self.user_docs.values():   all_docs.extend(docs)
            for chunks in self.user_chunks.values(): all_chunks.extend(chunks)
            if not all_docs: return []
            bm25 = BM25Okapi([tokenize(d) for d in all_docs])
            scores = bm25.get_scores(tokenize(query))
            idx = np.argsort(scores)[::-1][:k]
            return [(all_chunks[i], float(scores[i])) for i in idx]

        if uid not in self.user_bm25: return []
        bm25 = self.user_bm25[uid]
        scores = bm25.get_scores(tokenize(query))
        idx = np.argsort(scores)[::-1][:k]
        return [(self.user_chunks[uid][i], float(scores[i])) for i in idx]

# -------------------------
# Pure-NumPy vector index (Gemini embeddings)
# -------------------------
class UserVectorIndex:
    def __init__(self):
        self.user_chunks: Dict[str, List[DocChunk]] = {}
        self.user_matrix: Dict[str, np.ndarray]     = {}  # normalized

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        try:
            vecs = []
            for t in texts:
                # google-generativeai 0.7.x returns {'embedding': [...]} for embed_content
                e = genai.embed_content(model=EMBEDDING_MODEL, content=t)
                vecs.append(np.array(e["embedding"], dtype="float32"))
            X = np.vstack(vecs)
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            return X
        except Exception as exc:
            msg = (f"Embedding error: {exc}. "
                   f"Check GEMINI_API_KEY and EMBEDDING_MODEL='{EMBEDDING_MODEL}'.")
            raise RuntimeError(msg)

    def add_user(self, uid: str, chunks: List[DocChunk]):
        texts = [c.text for c in chunks]
        Xn = self._embed_batch(texts)
        self.user_chunks[uid] = chunks
        self.user_matrix[uid] = Xn

    def search(self, uid: str, query: str, k: int) -> List[Tuple[DocChunk, float]]:
        if ROLE_ADMIN_SEE_ALL:
            all_chunks, all_texts = [], []
            for chunks in self.user_chunks.values():
                all_chunks.extend(chunks)
                all_texts.extend([c.text for c in chunks])
            if not all_chunks: return []
            Xn = self._embed_batch(all_texts)
            q = self._embed_batch([query])
            sims = Xn @ q.T
            idx = np.argsort(sims[:,0])[::-1][:k]
            return [(all_chunks[i], float(sims[:,0][i])) for i in idx]

        if uid not in self.user_matrix: return []
        Xn = self.user_matrix[uid]
        q  = self._embed_batch([query])
        sims = Xn @ q.T
        idx  = np.argsort(sims[:,0])[::-1][:k]
        chunks = self.user_chunks[uid]
        return [(chunks[i], float(sims[:,0][i])) for i in idx]

    def add_chunks_for_user(self, uid: str, new_chunks: List["DocChunk"]):
        texts = [c.text for c in new_chunks]
        Xn_new = self._embed_batch(texts)
        if uid in self.user_matrix:
            Xn_old = self.user_matrix[uid]
            self.user_matrix[uid] = np.vstack([Xn_old, Xn_new])
            self.user_chunks[uid].extend(new_chunks)
        else:
            self.user_matrix[uid] = Xn_new
            self.user_chunks[uid] = list(new_chunks)

# -------------------------
# Hybrid retrieval
# -------------------------
def hybrid_retrieve(uid: str, query: str, bm25: UserBM25, vindex: UserVectorIndex,
                    top_k_bm25=6, top_k_vec=6, merge_k=6):
    bm25_hits = bm25.search(uid, query, top_k_bm25)
    vec_hits  = vindex.search(uid, query, top_k_vec)

    scored = {}
    for ch, s in bm25_hits: scored[ch.id] = max(scored.get(ch.id, 0.0), float(s)/10.0)
    for ch, s in vec_hits:  scored[ch.id] = max(scored.get(ch.id, 0.0), float(s))

    id_to_chunk = {ch.id: ch for ch, _ in bm25_hits + vec_hits}
    ordered = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:merge_k]
    return [id_to_chunk[i] for i, _ in ordered]

# -------------------------
# Prompting
# -------------------------
PLATFORM_TUTOR_GUIDE = """
You are **StudySpark Assistant**, a helpful guide for the StudySpark platform.
Capabilities you should proactively mention when relevant:
- Manage study resources (PDF/audio/video/images) and personal notes.
- Generate/edit summaries, flashcards, and quizzes.
- Plan revisions with tasks; track progress.
- Q&A over the user's own data with citations.
- Respect strict data isolation (never reveal other users' data).

When the user asks "how to" questions, explain steps clearly. Offer to create artifacts (notes, flashcards, quiz items, tasks) from the conversation.
"""

DATA_POLICY_GUARD = """
Data Isolation Policy:
- Only use data that belongs to the logged-in user (user_id = {user_id}).
- If the question requests another person's data, refuse and explain the policy.
- If no data context is available, ask the user to upload resources or create notes.
"""

ANSWER_STYLE = """
Answer Requirements:
- Be concise, structured, and actionable.
- Include citations to the retrieved items: [kind:title#source_id].
- If guidance is needed, provide step-by-step instructions.
- Offer follow-up actions (e.g., create flashcards, generate a plan).
"""

ONBOARDING_GUIDE = """
Onboarding Mode:
- The user may be new and have no resources/notes yet.
- Greet briefly and offer to help set up: upload resources, create notes, generate summaries/flashcards/quiz, and start a study plan.
- Keep answers short and practical. Provide 3-5 clear steps or quick commands per task.
- If the question is about StudySpark features (resources, flashcards, quiz, plans, roles), answer directly and concisely.
- If the question needs personal data not available yet, explain how to add it and proceed with a generic example.
"""

def format_context(chunks: List[DocChunk]) -> str:
    out = []
    for c in chunks:
        cite = f"[{c.kind}:{c.title}#{c.source_id}]"
        out.append(f"{cite}\n{c.text}")
    return "\n\n".join(out)

def build_prompt(user_id: str, query: str, context_chunks: List[DocChunk], chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    history_block = ""
    if chat_history:
        lines = []
        # Limit to last 10 messages to keep prompt reasonable
        for m in chat_history[-10:]:
            role = m.get("role", "user")
            text = m.get("text", "")
            lines.append(f"{role}: {text}")
        history_block = "\nPrevious conversation (most recent last):\n" + "\n".join(lines)
    return f"""{PLATFORM_TUTOR_GUIDE}

{DATA_POLICY_GUARD.format(user_id=user_id)}

{ANSWER_STYLE}

{history_block}

User question: {query}

Context:
{format_context(context_chunks)}

Respond now."""

def build_onboarding_prompt(user_id: str, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    history_block = ""
    if chat_history:
        lines = []
        for m in chat_history[-10:]:
            role = m.get("role", "user")
            text = m.get("text", "")
            lines.append(f"{role}: {text}")
        history_block = "\nPrevious conversation (most recent last):\n" + "\n".join(lines)
    return f"""{PLATFORM_TUTOR_GUIDE}

{DATA_POLICY_GUARD.format(user_id=user_id)}

{ONBOARDING_GUIDE}

Answer concisely (max ~6 bullet points when listing steps).

{history_block}

User question: {query}

Context:
No personal data yet. Provide generic guidance and examples.

Respond now."""

# -------------------------
# Chat session helpers
# -------------------------
def get_or_create_session(db: Dict[str, Any], user_id: str, session_id: Optional[str]) -> Dict[str, Any]:
    if session_id:
        for s in db["qa_sessions"]:
            if s.get("id") == session_id and s.get("user_id") == user_id:
                return s
    new_session = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "messages": [],  # list of {role, text, ts}
        "chunks_used": [],
        "ts": time.time()
    }
    db["qa_sessions"].append(new_session)
    return new_session

def index_chat_turn(user_id: str, session_id: str, question: str, answer: str,
                    bm25: UserBM25, vindex: UserVectorIndex):
    qa_chunks: List[DocChunk] = []
    qa_chunks.append(DocChunk(str(uuid.uuid4()), user_id, "chat", session_id, "user_question", question))
    qa_chunks.append(DocChunk(str(uuid.uuid4()), user_id, "chat", session_id, "assistant_answer", answer))
    bm25.add_chunks_and_rebuild_user(user_id, qa_chunks)
    vindex.add_chunks_for_user(user_id, qa_chunks)

# -------------------------
# Direct account-data answers (no LLM needed)
# -------------------------
def _normalize(s: str) -> str:
    return (s or "").strip().lower()

def get_latest_quiz_for_user(db: Dict[str, Any], user_id: str) -> Optional[Dict[str, Any]]:
    latest = None
    for qz in db.get("quiz", []):
        if qz.get("user_id") != user_id:
            continue
        attempts = qz.get("attempts", []) or []
        if not attempts:
            # Consider existence without attempts, with synthetic ts index by order
            cand = {"quiz": qz, "attempt": None, "date": "0000-00-00"}
        else:
            last = attempts[-1]
            cand = {"quiz": qz, "attempt": last, "date": str(last.get("date", "0000-00-00"))}
        if latest is None or cand["date"] > latest["date"]:
            latest = cand
    return latest

def get_latest_simple_item(items: List[Dict[str, Any]], user_id: str) -> Optional[Dict[str, Any]]:
    owned = [x for x in items if x.get("user_id") == user_id]
    if not owned:
        return None
    return owned[-1]

def maybe_answer_account_query(db: Dict[str, Any], user_id: str, question: str) -> Optional[str]:
    q = _normalize(question)
    role = get_user_role(db, user_id)
    if role == "admin":
        # Admin privacy guard: no cross-user data exposure
        return ("For privacy, I can’t open individual users’ content. "
                "I can share high-level stats or help with moderation tasks.")
    # Simple intent detection for "latest/last/recent" + entity keywords
    wants_latest = any(k in q for k in ["latest", "last", "recent", "dernier", "dernière"])
    if not wants_latest:
        return None
    # Quiz
    if any(k in q for k in ["quiz", "test", "quizz"]):
        info = get_latest_quiz_for_user(db, user_id)
        if not info:
            return "I don’t see any quizzes yet. Want me to create a starter quiz for you?"
        quiz = info["quiz"]
        att = info["attempt"]
        subject = quiz.get("subject", "Untitled")
        if att:
            score = att.get("score", "N/A")
            date = att.get("date", "recently")
            return f"Your most recent quiz was in {subject}. You scored {score} on {date}."
        return f"You have a quiz for {subject}, but no attempts yet. Want to take it now?"
    # Resource
    if any(k in q for k in ["resource", "document", "doc", "fichier"]):
        res = get_latest_simple_item(db.get("resources", []), user_id)
        if not res:
            return "You don’t have any resources yet. I can help you upload your first file."
        return f"Your latest resource is “{res.get('title','Untitled')}”."
    # Note
    if any(k in q for k in ["note", "notes"]):
        note = get_latest_simple_item(db.get("notes", []), user_id)
        if not note:
            return "You don’t have any notes yet. Want me to start a note for you?"
        return f"Your latest note is “{note.get('title','Untitled')}”."
    # Summary
    if any(k in q for k in ["summary", "résumé", "resume"]):
        sm = get_latest_simple_item(db.get("summaries", []), user_id)
        if not sm:
            return "No summaries yet. I can generate one from a resource or note."
        return f"Your latest summary is linked to “{sm.get('source','unknown')}”."
    # Flashcards
    if any(k in q for k in ["flashcard", "cards", "deck"]):
        fc = get_latest_simple_item(db.get("flashcards", []), user_id)
        if not fc:
            return "No flashcards yet. I can create a deck from your material."
        return f"Your latest flashcard is in the “{fc.get('deck','Deck')}” deck."
    # Study plan
    if any(k in q for k in ["plan", "study plan", "planning"]):
        sp = get_latest_simple_item(db.get("study_plans", []), user_id)
        if not sp:
            return "No study plans yet. I can set up a plan for your next exam."
        return f"Your latest study plan goal is “{sp.get('goal','Goal')}”."
    return None

# -------------------------
# List/overview answers for account data
# -------------------------
def _format_bulleted(lines: List[str], limit: int = 8) -> str:
    if not lines:
        return ""
    if len(lines) > limit:
        shown = lines[:limit]
        remaining = len(lines) - limit
        return "\n".join([f"- {l}" for l in shown] + [f"… and {remaining} more"])
    return "\n".join([f"- {l}" for l in lines])

def maybe_answer_list_query(db: Dict[str, Any], user_id: str, question: str) -> Optional[str]:
    q = _normalize(question)
    role = get_user_role(db, user_id)
    if role == "admin":
        return ("For privacy, I don’t list individual users’ items. "
                "I can provide aggregated usage or help with moderation policies.")
    wants_list = any(k in q for k in ["all", "list", "show", "voir", "tous", "toutes", "what", "which"]) and \
                 any(k in q for k in ["quiz", "quizzes", "tests", "flashcard", "flashcards", "cards", "resources", "documents", "notes", "summaries", "decks", "plans", "study plans"])
    if not wants_list:
        # Heuristics for phrasings like "what flashcards i did"
        if not any(k in q for k in ["did", "done", "completed", "pris", "fait"]):
            return None

    # Quizzes
    if any(k in q for k in ["quiz", "quizzes", "tests"]):
        lines: List[str] = []
        for qz in db.get("quiz", []):
            if qz.get("user_id") != user_id:
                continue
            subject = qz.get("subject", "Untitled")
            attempts = qz.get("attempts", []) or []
            if attempts:
                last = attempts[-1]
                score = last.get("score", "N/A")
                date = last.get("date", "")
                lines.append(f"{subject}: last score {score}{(' on ' + date) if date else ''}")
            else:
                lines.append(f"{subject}: no attempts yet")
        if not lines:
            return "You don’t have any quizzes yet. I can create one from your notes or a resource."
        return "Here are your quizzes:\n" + _format_bulleted(lines)

    # Flashcards (group by deck)
    if any(k in q for k in ["flashcard", "flashcards", "cards", "deck", "decks"]):
        owned = [f for f in db.get("flashcards", []) if f.get("user_id") == user_id]
        if not owned:
            return "You don’t have any flashcards yet. Want me to build a deck from your material?"
        # Deck counts
        deck_to_count: Dict[str, int] = {}
        for f in owned:
            deck = f.get("deck", "Deck")
            deck_to_count[deck] = deck_to_count.get(deck, 0) + 1
        lines = [f"{deck}: {count} card(s)" for deck, count in deck_to_count.items()]
        return "Your flashcard decks:\n" + _format_bulleted(lines)

    # Resources
    if any(k in q for k in ["resource", "resources", "document", "documents", "fichier", "fichiers"]):
        owned = [r for r in db.get("resources", []) if r.get("user_id") == user_id]
        if not owned:
            return "No resources yet. I can help you upload your first file."
        titles = [r.get("title", "Untitled") for r in owned]
        return "Your resources:\n" + _format_bulleted(titles)

    # Notes
    if any(k in q for k in ["note", "notes"]):
        owned = [n for n in db.get("notes", []) if n.get("user_id") == user_id]
        if not owned:
            return "No notes yet. I can start a note for you."
        titles = [n.get("title", "Untitled") for n in owned]
        return "Your notes:\n" + _format_bulleted(titles)

    # Summaries
    if any(k in q for k in ["summary", "summaries", "résumé", "resumes"]):
        owned = [s for s in db.get("summaries", []) if s.get("user_id") == user_id]
        if not owned:
            return "No summaries yet. I can generate one from a resource or note."
        lines = [f"for {s.get('source','unknown')} ({s.get('length','')})".strip() for s in owned]
        return "Your summaries:\n" + _format_bulleted(lines)

    # Study plans
    if any(k in q for k in ["plan", "plans", "study plan", "planning"]):
        owned = [p for p in db.get("study_plans", []) if p.get("user_id") == user_id]
        if not owned:
            return "No study plans yet. I can set one up for your next exam."
        lines = [p.get("goal", "Goal") for p in owned]
        return "Your study plans:\n" + _format_bulleted(lines)

    return None

# -------------------------
# Public API
# -------------------------
def init_indices(db: Dict[str, Any]):
    corpus = build_corpus(db)

    bm25 = UserBM25()
    for ch in corpus: bm25.add(ch)
    bm25.build()

    vindex = UserVectorIndex()
    user_to_chunks = defaultdict(list)
    for ch in corpus: user_to_chunks[ch.user_id].append(ch)
    for uid, chunks in user_to_chunks.items():
        vindex.add_user(uid, chunks)

    return bm25, vindex

def answer_question(db: Dict[str, Any], bm25: UserBM25, vindex: UserVectorIndex,
                    user_id: str, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    # Admin guardrail: admins don’t see other users’ content via chat
    role = get_user_role(db, user_id)
    if role == "admin" and not ROLE_ADMIN_SEE_ALL:
        session = get_or_create_session(db, user_id, session_id)
        msg = ("I can help with platform-wide tasks and policies. "
               "I don’t open individual users’ study content.")
        session["messages"].append({"role": "user", "text": question, "ts": time.time()})
        session["messages"].append({"role": "assistant", "text": msg, "ts": time.time()})
        index_chat_turn(user_id, session["id"], question, msg, bm25, vindex)
        return {"answer": msg, "session_id": session["id"]}
    # First, see if we can satisfy the request directly from account data (lists)
    list_ans = maybe_answer_list_query(db, user_id, question)
    if list_ans is not None:
        session = get_or_create_session(db, user_id, session_id)
        session["messages"].append({"role": "user", "text": question, "ts": time.time()})
        session["messages"].append({"role": "assistant", "text": list_ans, "ts": time.time()})
        index_chat_turn(user_id, session["id"], question, list_ans, bm25, vindex)
        return {"answer": list_ans, "session_id": session["id"]}

    # Then, try direct “latest” account data answers
    direct = maybe_answer_account_query(db, user_id, question)
    if direct is not None:
        session = get_or_create_session(db, user_id, session_id)
        session["messages"].append({"role": "user", "text": question, "ts": time.time()})
        session["messages"].append({"role": "assistant", "text": direct, "ts": time.time()})
        index_chat_turn(user_id, session["id"], question, direct, bm25, vindex)
        return {"answer": direct, "session_id": session["id"]}

    chunks = hybrid_retrieve(user_id, question, bm25, vindex, 6, 6, 6)
    if not chunks and not ROLE_ADMIN_SEE_ALL:
        # Onboarding: generate a helpful, concise answer without citations requirement
        session = get_or_create_session(db, user_id, session_id)
        prompt = build_onboarding_prompt(user_id, question, chat_history=session.get("messages", []))

        # Build candidate models (same as below)
        candidate_models = []
        if GENERATION_MODEL:
            candidate_models.append(GENERATION_MODEL)
        try:
            listed = genai.list_models()
            supported = []
            for m in listed:
                name = getattr(m, "name", "") or ""
                methods = set(getattr(m, "supported_generation_methods", []) or [])
                if "generateContent" in methods:
                    supported.append(name)
            preferred = [n for n in supported if ("gemini-1.5" in n and ("flash" in n or "pro" in n))]
            if not preferred:
                preferred = [n for n in supported if "gemini" in n]
            for n in preferred:
                if n not in candidate_models:
                    candidate_models.append(n)
        except Exception:
            for alt in ("gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro", "gemini-pro"):
                if alt not in candidate_models:
                    candidate_models.append(alt)

        last_error = None
        answer = None
        for model_name in candidate_models:
            try:
                model = genai.GenerativeModel(model_name)
                resp  = model.generate_content(prompt)
                answer = resp.text
                break
            except Exception as exc:
                last_error = exc

        if answer is None:
            msg = (
                "Generation error: {}. Tried models: {}. "
                "Tip: upgrade google-generativeai and set GENERATION_MODEL to a model from ListModels."
            ).format(last_error, ", ".join(candidate_models))
            session["messages"].append({"role": "user", "text": question, "ts": time.time()})
            session["messages"].append({"role": "assistant", "text": msg, "ts": time.time()})
            return {"answer": msg, "session_id": session["id"]}

        # Persist and index even onboarding turns so future RAG can use them
        session["messages"].append({"role": "user", "text": question, "ts": time.time()})
        session["messages"].append({"role": "assistant", "text": answer, "ts": time.time()})
        index_chat_turn(user_id, session["id"], question, answer, bm25, vindex)
        return {"answer": answer, "session_id": session["id"]}

    # Fetch or create session and include recent history in the prompt
    session = get_or_create_session(db, user_id, session_id)
    prompt = build_prompt(user_id, question, chunks, chat_history=session.get("messages", []))

    # Build candidate models dynamically using ListModels, prioritizing supported ones
    candidate_models = []
    if GENERATION_MODEL:
        candidate_models.append(GENERATION_MODEL)
    try:
        listed = genai.list_models()
        # Keep names that support generateContent
        supported = []
        for m in listed:
            name = getattr(m, "name", "") or ""
            methods = set(getattr(m, "supported_generation_methods", []) or [])
            if "generateContent" in methods:
                supported.append(name)
        # Prefer 1.5 variants first, then any gemini models
        preferred = [n for n in supported if ("gemini-1.5" in n and ("flash" in n or "pro" in n))]
        if not preferred:
            preferred = [n for n in supported if "gemini" in n]
        for n in preferred:
            if n not in candidate_models:
                candidate_models.append(n)
    except Exception:
        # Fallback if listing fails (older SDK / network issue)
        for alt in ("gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro", "gemini-pro"):
            if alt not in candidate_models:
                candidate_models.append(alt)

    last_error = None
    answer = None
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            resp  = model.generate_content(prompt)
            answer = resp.text
            break
        except Exception as exc:
            last_error = exc

    if answer is None:
        msg = (
            "Generation error: {}. Tried models: {}. "
            "Tip: upgrade google-generativeai (pip install -U google-generativeai) and set GENERATION_MODEL to a model from ListModels."
        ).format(last_error, ", ".join(candidate_models))
        session["messages"].append({"role": "user", "text": question, "ts": time.time()})
        session["messages"].append({"role": "assistant", "text": msg, "ts": time.time()})
        return {"answer": msg, "session_id": session["id"]}

    # Persist turn in the session history
    session["messages"].append({"role": "user", "text": question, "ts": time.time()})
    session["messages"].append({"role": "assistant", "text": answer, "ts": time.time()})
    # Track chunks used for transparency
    session["chunks_used"].extend([c.id for c in chunks])

    # Incrementally index this chat turn into RAG indices
    index_chat_turn(user_id, session["id"], question, answer, bm25, vindex)

    return {"answer": answer, "session_id": session["id"]}
