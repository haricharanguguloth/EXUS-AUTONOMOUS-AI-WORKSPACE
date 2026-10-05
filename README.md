"""
NEXUS - Autonomous AI Personal & Productivity Workspace (Version 1, single file)
Run:  python app.py   then open  http://127.0.0.1:5000

Sections in this file:
 1 Config   2 Database   3 AI provider   4 Documents + RAG   5 Tools
 6 Agent (planner -> tools -> response)   7 API routes   8 Frontend (HTML/CSS/JS)   9 Startup
"""
import os, re, json, ast, math, logging, sqlite3, operator, secrets
from datetime import datetime, timedelta
from functools import wraps
import requests                       # calls the AI API
from dotenv import load_dotenv        # reads the .env file
from flask import Flask, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash   # password hashing
from werkzeug.utils import secure_filename                                   # safe file names

# ------------------------------------------------------------------ 1 CONFIG
load_dotenv()
BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "uploads")
DB_PATH = os.path.join(BASE, "nexus.db")
ALLOWED = {".pdf", ".txt", ".docx"}
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nexus")

def get_secret_key():
    """Session signing key: from .env, or generated once and saved locally."""
    key = os.getenv("SECRET_KEY")
    if key:
        return key
    path = os.path.join(BASE, ".secret_key")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    return open(path).read().strip()

app = Flask(__name__)
app.secret_key = get_secret_key()
app.config.update(MAX_CONTENT_LENGTH=10 * 1024 * 1024,      # 10 MB upload limit
                  SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")

# --------------------------------------------------------------- 2 DATABASE
# Every query uses "?" placeholders, so user input is never pasted into SQL (SQL-injection safe).
def run(sql, args=()):
    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.execute(sql, args)
        con.commit()
        return cur.lastrowid
    finally:
        con.close()

def rows(sql, args=()):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, args).fetchall()]
    finally:
        con.close()

def row(sql, args=()):
    r = rows(sql, args)
    return r[0] if r else None

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def init_db():
    stmts = [
        "CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT, created_at TEXT)",
        "CREATE TABLE IF NOT EXISTS conversations(id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, created_at TEXT)",
        "CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, conversation_id INTEGER, user_id INTEGER, role TEXT, content TEXT, meta TEXT, created_at TEXT)",
        "CREATE TABLE IF NOT EXISTS documents(id INTEGER PRIMARY KEY, user_id INTEGER, filename TEXT, stored_name TEXT, filetype TEXT, size INTEGER, status TEXT, created_at TEXT)",
        "CREATE TABLE IF NOT EXISTS document_chunks(id INTEGER PRIMARY KEY, document_id INTEGER, user_id INTEGER, idx INTEGER, content TEXT, embedding TEXT)",
        "CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY, user_id INTEGER, content TEXT, created_at TEXT)",
        "CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, description TEXT, priority TEXT DEFAULT 'MEDIUM', status TEXT DEFAULT 'TODO', due_date TEXT, created_at TEXT, completed_at TEXT)",
        "CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, content TEXT, created_at TEXT, updated_at TEXT)",
    ]
    for s in stmts:
        run(s)

# -------------------------------------------------------------- 3 AI PROVIDER
class AIError(Exception):
    pass

class OpenAIProvider:
    """Works with any OpenAI-compatible API. Swap this class to change provider later."""
    def __init__(self):
        self.key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    def available(self):
        return bool(self.key) and "YOUR_API_KEY" not in self.key

    def _post(self, path, payload):
        if not self.available():
            raise AIError("OpenAI API key is missing. Add OPENAI_API_KEY to your .env file and restart.")
        try:
            r = requests.post(self.base + path, json=payload, timeout=60,
                              headers={"Authorization": "Bearer " + self.key})
            if r.status_code != 200:
                log.error("AI API status %s", r.status_code)    # status only; never log the key
                raise AIError("AI service returned an error. Please check your API configuration.")
            return r.json()
        except requests.RequestException:
            log.error("AI API unreachable")
            raise AIError("AI service is currently unavailable. Please check your connection and API configuration.")

    def chat(self, messages):
        return self._post("/chat/completions", {"model": self.model, "messages": messages})["choices"][0]["message"]["content"]

    def embed(self, texts):
        data = self._post("/embeddings", {"model": self.embed_model, "input": texts})["data"]
        return [d["embedding"] for d in data]

provider = OpenAIProvider()

class SearchProvider:
    """Placeholder for web search. Implement search() with Tavily/SerpAPI/etc. later.
    Returns a list of {"title","url","snippet"}. Empty list = no real sources available."""
    def search(self, query):
        return []

search_provider = SearchProvider()

# ------------------------------------------------------ 4 DOCUMENTS + RAG
def extract_text(path, ext):
    if ext == ".txt":
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    if ext == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    return ""

def chunk_text(text, size=900, overlap=120):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + size])
        start += size - overlap
    return out

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

def words(s):
    return set(re.findall(r"[a-z0-9]{3,}", s.lower()))

def retrieve(user_id, question, doc_id=None, k=5):
    """Find the most relevant chunks. Uses embeddings if available, else keyword overlap."""
    sql = ("SELECT c.id, c.idx, c.content, c.embedding, d.filename FROM document_chunks c "
           "JOIN documents d ON d.id=c.document_id WHERE c.user_id=?")
    args = [user_id]
    if doc_id:
        sql += " AND c.document_id=?"
        args.append(doc_id)
    chunks = rows(sql, args)
    if not chunks:
        return []
    qvec = None
    try:
        if provider.available():
            qvec = provider.embed([question])[0]
    except AIError:
        qvec = None
    qw = words(question)
    for c in chunks:
        if qvec and c["embedding"]:
            c["score"] = cosine(qvec, json.loads(c["embedding"]))
        else:
            c["score"] = len(qw & words(c["content"])) / (len(qw) or 1)
    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks[:k]

def process_document(doc_id, user_id, path, ext):
    """Extract -> chunk -> embed -> store. Updates the document status."""
    try:
        chunks = chunk_text(extract_text(path, ext))
        if not chunks:
            raise ValueError("No text found")
        vectors = [None] * len(chunks)
        try:
            if provider.available():
                vectors = []
                for i in range(0, len(chunks), 50):
                    vectors += provider.embed(chunks[i:i + 50])
        except AIError:
            vectors = [None] * len(chunks)          # fall back to keyword search
        for i, c in enumerate(chunks):
            run("INSERT INTO document_chunks(document_id,user_id,idx,content,embedding) VALUES(?,?,?,?,?)",
                (doc_id, user_id, i + 1, c, json.dumps(vectors[i]) if vectors[i] else None))
        run("UPDATE documents SET status=? WHERE id=?", ("READY" if vectors[0] else "READY (keyword search)", doc_id))
    except Exception:
        log.exception("Document processing failed for doc %s", doc_id)
        run("UPDATE documents SET status='FAILED' WHERE id=?", (doc_id,))

# ------------------------------------------------------------------ 5 TOOLS
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg}

def safe_eval(node):
    """Evaluates math only (no eval()), so users cannot run arbitrary code."""
    if isinstance(node, ast.Expression):
        return safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.operand))
    raise ValueError("Unsupported expression")

def calculator_tool(uid, expr):
    try:
        if "**" in expr and len(expr) > 30:
            raise ValueError
        return f"{expr.strip()} = {safe_eval(ast.parse(expr.strip(), mode='eval'))}"
    except Exception:
        return "I couldn't calculate that. Use numbers with + - * / % ** and parentheses."

def create_task_tool(uid, text):
    title = re.sub(r"^(please\s+)?(remind me to|create a task( to)?|add a task( to)?)\s*:?\s*", "", text.strip(), flags=re.I)
    due = None
    if re.search(r"\btomorrow\b", title, re.I):
        due = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        title = re.sub(r"\btomorrow\b", "", title, flags=re.I).strip()
    elif re.search(r"\btoday\b", title, re.I):
        due = datetime.now().strftime("%Y-%m-%d")
        title = re.sub(r"\btoday\b", "", title, flags=re.I).strip()
    title = (title or "New task").strip(" .")[:200]
    run("INSERT INTO tasks(user_id,title,description,due_date,created_at) VALUES(?,?,?,?,?)", (uid, title, "", due, now()))
    return f"Task created: **{title}**" + (f" (due {due})" if due else "")

def create_note_tool(uid, text):
    body = re.sub(r"^create a note\s*:?\s*", "", text.strip(), flags=re.I).strip() or "Empty note"
    title = body.split("\n")[0][:60]
    run("INSERT INTO notes(user_id,title,content,created_at,updated_at) VALUES(?,?,?,?,?)", (uid, title, body, now(), now()))
    return f"Note saved: **{title}**"

def memory_save_tool(uid, text):
    fact = re.sub(r"^(please\s+)?remember( that)?\s*:?\s*", "", text.strip(), flags=re.I)[:500]
    run("INSERT INTO memories(user_id,content,created_at) VALUES(?,?,?)", (uid, fact, now()))
    return f"Saved to long-term memory: *{fact}*"

def memory_search_tool(uid, text):
    mems = rows("SELECT content FROM memories WHERE user_id=?", (uid,))
    qw = words(text)
    hits = [m["content"] for m in mems if qw & words(m["content"])] or [m["content"] for m in mems][:10]
    return hits

def document_search_tool(uid, text, doc_id=None):
    return retrieve(uid, text, doc_id)

TOOLS = {"calculator": calculator_tool, "create_task": create_task_tool, "create_note": create_note_tool,
         "memory_save": memory_save_tool, "memory_search": memory_search_tool, "document_search": document_search_tool}

# ------------------------------------------------------------------ 6 AGENT
def plan(text, mode, has_docs, doc_id):
    """PLANNER: decide the intent and which tool(s) to run. Simple rules, easy to read."""
    t = text.lower().strip()
    if re.match(r"^(calculate|calc|compute)\b", t):
        return "CALCULATION", ["calculator"]
    if re.match(r"^(please\s+)?(remind me to|create a task|add a task)", t):
        return "TASK_MANAGEMENT", ["create_task"]
    if re.match(r"^create a note", t):
        return "NOTE_CREATION", ["create_note"]
    if re.match(r"^(please\s+)?remember\b", t):
        return "MEMORY", ["memory_save"]
    if re.search(r"what (project|do you know|am i)|do you remember|about me", t):
        return "MEMORY", ["memory_search"]
    if doc_id or (has_docs and re.search(r"\b(my|the) (document|doc|notes|pdf|file|dbms|upload)|uploaded|according to", t)):
        return "DOCUMENT_SEARCH", ["document_search"]
    if mode == "research":
        return "WEB_RESEARCH", []
    if mode == "study":
        return "STUDY_ASSISTANCE", []
    if mode == "code":
        return "CODE_HELP", []
    return "NORMAL_CHAT", []

SYSTEM = {
    "chat": "You are NEXUS, a helpful AI workspace assistant. The user is a beginner B.Tech student: explain clearly with examples.",
    "study": "You are NEXUS Study Assistant. Explain topics, make MCQs, flashcards, quizzes and study plans suited to the stated difficulty.",
    "code": "You are NEXUS Coding Assistant for Python, Java, C, JavaScript, HTML, CSS and SQL. Explain code, find bugs, improve and explain errors. You cannot run code.",
    "research": "You are NEXUS Research Assistant. You have NO web access in this run. Give a structured overview from general knowledge, "
                "clearly state that no live sources were retrieved, and NEVER invent citations, URLs or papers.",
}

def run_agent(uid, conv_id, text, mode, doc_id):
    """EXECUTOR + RESPONSE GENERATOR. Returns dict with reply, intent, tools_used, sources."""
    has_docs = bool(row("SELECT id FROM documents WHERE user_id=? LIMIT 1", (uid,)))
    intent, tool_names = plan(text, mode, has_docs, doc_id)
    used, sources, context = [], [], ""

    # Direct tools answer on their own (nothing is pretended: only listed tools actually ran)
    for name in tool_names:
        if name in ("calculator", "create_task", "create_note", "memory_save"):
            arg = re.sub(r"^(calculate|calc|compute)\s*:?\s*", "", text, flags=re.I) if name == "calculator" else text
            used.append(name)
            return {"reply": TOOLS[name](uid, arg), "intent": intent, "tools": used, "sources": []}
        if name == "memory_search":
            used.append(name)
            hits = TOOLS[name](uid, text)
            context = "Saved memories:\n- " + "\n- ".join(hits) if hits else "No memories saved yet."
            if not hits:
                return {"reply": "I don't have any saved memories yet. Say \"Remember that ...\" to add one.", "intent": intent, "tools": used, "sources": []}
        if name == "document_search":
            used.append(name)
            if re.search(r"summar|overview|main topics", text.lower()) and doc_id:
                chunks = rows("SELECT c.idx,c.content,d.filename FROM document_chunks c JOIN documents d ON d.id=c.document_id "
                              "WHERE c.user_id=? AND c.document_id=? ORDER BY c.idx LIMIT 8", (uid, doc_id))
            else:
                chunks = TOOLS[name](uid, text, doc_id)
            if not chunks:
                return {"reply": "I couldn't find any processed document text to search. Upload a document first.", "intent": intent, "tools": used, "sources": []}
            context = "Document excerpts:\n" + "\n---\n".join(f"[{c['filename']} - section {c['idx']}]\n{c['content']}" for c in chunks)
            sources = sorted({f"{c['filename']} — Section {c['idx']}" for c in chunks})

    # Build the prompt: system + long-term memory + retrieved context + short-term history
    mems = rows("SELECT content FROM memories WHERE user_id=? ORDER BY id DESC LIMIT 15", (uid,))
    sys_msg = SYSTEM.get(mode, SYSTEM["chat"])
    if mems:
        sys_msg += "\nKnown about the user:\n- " + "\n- ".join(m["content"] for m in mems)
    if context:
        sys_msg += "\n\nAnswer using ONLY this context when relevant, and say so if the answer isn't in it.\n" + context
    if intent == "WEB_RESEARCH":
        found = search_provider.search(text)
        if found:
            sys_msg += "\nWeb results:\n" + "\n".join(f"{r['title']} {r['url']} {r['snippet']}" for r in found)
            sources = [r["url"] for r in found]
            used.append("web_search")
    history = rows("SELECT role,content FROM messages WHERE conversation_id=? AND user_id=? ORDER BY id DESC LIMIT 10", (conv_id, uid))[::-1]
    reply = provider.chat([{"role": "system", "content": sys_msg}] + [{"role": h["role"], "content": h["content"]} for h in history])
    if intent == "WEB_RESEARCH" and not sources:
        reply = "_Note: no web search provider is configured, so no live sources were retrieved._\n\n" + reply
    return {"reply": reply, "intent": intent, "tools": used, "sources": sources}

# ---------------------------------------------------------------- 7 API ROUTES
def ok(**kw):
    return jsonify(kw)

def fail(msg, code=400):
    return jsonify(error=msg), code

def login_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if not session.get("uid"):
            return fail("Please log in.", 401)
        return f(*a, **kw)
    return wrapper

@app.before_request
def csrf_check():
    """CSRF protection: state-changing API calls must send the token we gave the logged-in session."""
    if request.method in ("POST", "PUT", "DELETE") and request.path.startswith("/api/") \
            and request.path not in ("/api/login", "/api/register"):
        if not session.get("csrf") or request.headers.get("X-CSRF-Token") != session.get("csrf"):
            return fail("Security check failed. Refresh the page.", 403)

@app.errorhandler(413)
def too_big(e):
    return fail("File is too large (max 10 MB).", 413)

@app.errorhandler(Exception)
def any_error(e):
    code = getattr(e, "code", 500)
    if code == 500 or not isinstance(code, int):
        log.exception("Unhandled error")
        return fail("Something went wrong. Please try again.", 500)
    return fail(getattr(e, "description", "Error"), code)

def login_session(uid, username):
    session.clear()
    session["uid"], session["username"], session["csrf"] = uid, username, secrets.token_hex(16)

@app.post("/api/register")
def register():
    d = request.get_json(silent=True) or {}
    u, p = str(d.get("username", "")).strip(), str(d.get("password", ""))
    if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", u):
        return fail("Username: 3-30 letters, numbers or underscores.")
    if len(p) < 8:
        return fail("Password must be at least 8 characters.")
    if row("SELECT id FROM users WHERE username=?", (u,)):
        return fail("Username already taken.")
    uid = run("INSERT INTO users(username,password_hash,created_at) VALUES(?,?,?)", (u, generate_password_hash(p), now()))
    login_session(uid, u)
    log.info("User registered: id=%s", uid)
    return ok(username=u, csrf=session["csrf"])

@app.post("/api/login")
def login():
    d = request.get_json(silent=True) or {}
    user = row("SELECT * FROM users WHERE username=?", (str(d.get("username", "")).strip(),))
    if not user or not check_password_hash(user["password_hash"], str(d.get("password", ""))):
        log.info("Failed login attempt")
        return fail("Wrong username or password.", 401)
    login_session(user["id"], user["username"])
    return ok(username=user["username"], csrf=session["csrf"])

@app.post("/api/logout")
def logout():
    session.clear()
    return ok()

@app.get("/api/me")
def me():
    if not session.get("uid"):
        return fail("Not logged in", 401)
    return ok(username=session["username"], csrf=session["csrf"], ai_ready=provider.available())

# ---- chat & conversations
@app.get("/api/conversations")
@login_required
def list_convs():
    return ok(items=rows("SELECT id,title,created_at FROM conversations WHERE user_id=? ORDER BY id DESC", (session["uid"],)))

@app.post("/api/conversations")
@login_required
def new_conv():
    cid = run("INSERT INTO conversations(user_id,title,created_at) VALUES(?,?,?)", (session["uid"], "New conversation", now()))
    return ok(id=cid)

@app.put("/api/conversations/<int:cid>")
@login_required
def rename_conv(cid):
    title = str((request.get_json(silent=True) or {}).get("title", "")).strip()[:100]
    if not title:
        return fail("Title required.")
    run("UPDATE conversations SET title=? WHERE id=? AND user_id=?", (title, cid, session["uid"]))
    return ok()

@app.delete("/api/conversations/<int:cid>")
@login_required
def del_conv(cid):
    run("DELETE FROM messages WHERE conversation_id=? AND user_id=?", (cid, session["uid"]))
    run("DELETE FROM conversations WHERE id=? AND user_id=?", (cid, session["uid"]))
    return ok()

@app.get("/api/conversations/<int:cid>/messages")
@login_required
def conv_messages(cid):
    return ok(items=rows("SELECT id,role,content,meta,created_at FROM messages WHERE conversation_id=? AND user_id=? ORDER BY id", (cid, session["uid"])))

@app.post("/api/chat")
@login_required
def chat():
    uid, d = session["uid"], request.get_json(silent=True) or {}
    text, mode = str(d.get("message", "")).strip()[:6000], d.get("mode", "chat")
    if not text:
        return fail("Message is empty.")
    if mode not in SYSTEM:
        mode = "chat"
    cid = d.get("conversation_id")
    if cid and not row("SELECT id FROM conversations WHERE id=? AND user_id=?", (cid, uid)):
        return fail("Conversation not found.", 404)
    if not cid:
        cid = run("INSERT INTO conversations(user_id,title,created_at) VALUES(?,?,?)", (uid, text[:40], now()))
    elif row("SELECT title FROM conversations WHERE id=?", (cid,))["title"] == "New conversation":
        run("UPDATE conversations SET title=? WHERE id=?", (text[:40], cid))
    if not d.get("regenerate"):
        run("INSERT INTO messages(conversation_id,user_id,role,content,created_at) VALUES(?,?,?,?,?)", (cid, uid, "user", text, now()))
    doc_id = d.get("document_id")
    if doc_id and not row("SELECT id FROM documents WHERE id=? AND user_id=?", (doc_id, uid)):
        doc_id = None
    try:
        res = run_agent(uid, cid, text, mode, doc_id)
    except AIError as e:
        return fail(str(e), 503)
    meta = json.dumps({"intent": res["intent"], "tools": res["tools"], "sources": res["sources"]})
    run("INSERT INTO messages(conversation_id,user_id,role,content,meta,created_at) VALUES(?,?,?,?,?,?)", (cid, uid, "assistant", res["reply"], meta, now()))
    return ok(conversation_id=cid, reply=res["reply"], intent=res["intent"], tools=res["tools"], sources=res["sources"])

# ---- documents
@app.post("/api/upload")
@login_required
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return fail("No file selected.")
    name = secure_filename(f.filename)
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED:
        return fail("Only PDF, TXT and DOCX files are allowed.")
    stored = secrets.token_hex(8) + ext                       # random stored name; file is never executed
    path = os.path.join(UPLOAD_DIR, stored)
    f.save(path)
    did = run("INSERT INTO documents(user_id,filename,stored_name,filetype,size,status,created_at) VALUES(?,?,?,?,?,?,?)",
              (session["uid"], name, stored, ext[1:].upper(), os.path.getsize(path), "PROCESSING", now()))
    process_document(did, session["uid"], path, ext)
    return ok(id=did)

@app.get("/api/documents")
@login_required
def docs():
    return ok(items=rows("SELECT id,filename,filetype,size,status,created_at FROM documents WHERE user_id=? ORDER BY id DESC", (session["uid"],)))

def remove_document(uid, did):
    d = row("SELECT stored_name FROM documents WHERE id=? AND user_id=?", (did, uid))
    if d:
        try:
            os.remove(os.path.join(UPLOAD_DIR, d["stored_name"]))
        except OSError:
            pass
        run("DELETE FROM document_chunks WHERE document_id=? AND user_id=?", (did, uid))
        run("DELETE FROM documents WHERE id=? AND user_id=?", (did, uid))

@app.delete("/api/documents/<int:did>")
@login_required
def del_doc(did):
    remove_document(session["uid"], did)
    return ok()

# ---- memories
@app.get("/api/memories")
@login_required
def mems():
    return ok(items=rows("SELECT id,content,created_at FROM memories WHERE user_id=? ORDER BY id DESC", (session["uid"],)))

@app.post("/api/memories")
@login_required
def add_mem():
    c = str((request.get_json(silent=True) or {}).get("content", "")).strip()[:500]
    if not c:
        return fail("Content required.")
    return ok(id=run("INSERT INTO memories(user_id,content,created_at) VALUES(?,?,?)", (session["uid"], c, now())))

@app.delete("/api/memories/<int:mid>")
@login_required
def del_mem(mid):
    run("DELETE FROM memories WHERE id=? AND user_id=?", (mid, session["uid"]))
    return ok()

# ---- tasks
@app.get("/api/tasks")
@login_required
def tasks():
    return ok(items=rows("SELECT * FROM tasks WHERE user_id=? ORDER BY (status='COMPLETED'), id DESC", (session["uid"],)))

@app.post("/api/tasks")
@login_required
def add_task():
    d = request.get_json(silent=True) or {}
    title = str(d.get("title", "")).strip()[:200]
    if not title:
        return fail("Title required.")
    pr = d.get("priority") if d.get("priority") in ("LOW", "MEDIUM", "HIGH") else "MEDIUM"
    return ok(id=run("INSERT INTO tasks(user_id,title,description,priority,due_date,created_at) VALUES(?,?,?,?,?,?)",
                     (session["uid"], title, str(d.get("description", ""))[:1000], pr, d.get("due_date") or None, now())))

@app.put("/api/tasks/<int:tid>")
@login_required
def edit_task(tid):
    d, uid = request.get_json(silent=True) or {}, session["uid"]
    t = row("SELECT * FROM tasks WHERE id=? AND user_id=?", (tid, uid))
    if not t:
        return fail("Task not found.", 404)
    status = d.get("status", t["status"])
    if status not in ("TODO", "IN_PROGRESS", "COMPLETED"):
        return fail("Invalid status.")
    done = now() if status == "COMPLETED" and t["status"] != "COMPLETED" else (t["completed_at"] if status == "COMPLETED" else None)
    pr = d.get("priority", t["priority"])
    if pr not in ("LOW", "MEDIUM", "HIGH"):
        return fail("Invalid priority.")
    run("UPDATE tasks SET title=?,description=?,priority=?,status=?,due_date=?,completed_at=? WHERE id=? AND user_id=?",
        (str(d.get("title", t["title"]))[:200], str(d.get("description", t["description"]))[:1000], pr, status,
         d.get("due_date", t["due_date"]), done, tid, uid))
    return ok()

@app.delete("/api/tasks/<int:tid>")
@login_required
def del_task(tid):
    run("DELETE FROM tasks WHERE id=? AND user_id=?", (tid, session["uid"]))
    return ok()

# ---- notes
@app.get("/api/notes")
@login_required
def notes():
    return ok(items=rows("SELECT * FROM notes WHERE user_id=? ORDER BY id DESC", (session["uid"],)))

@app.post("/api/notes")
@login_required
def add_note():
    d = request.get_json(silent=True) or {}
    title, content = str(d.get("title", "")).strip()[:150], str(d.get("content", ""))[:20000]
    if not title:
        return fail("Title required.")
    return ok(id=run("INSERT INTO notes(user_id,title,content,created_at,updated_at) VALUES(?,?,?,?,?)", (session["uid"], title, content, now(), now())))

@app.put("/api/notes/<int:nid>")
@login_required
def edit_note(nid):
    d = request.get_json(silent=True) or {}
    run("UPDATE notes SET title=?,content=?,updated_at=? WHERE id=? AND user_id=?",
        (str(d.get("title", ""))[:150], str(d.get("content", ""))[:20000], now(), nid, session["uid"]))
    return ok()

@app.delete("/api/notes/<int:nid>")
@login_required
def del_note(nid):
    run("DELETE FROM notes WHERE id=? AND user_id=?", (nid, session["uid"]))
    return ok()

@app.post("/api/notes/<int:nid>/ai")
@login_required
def note_ai(nid):
    n = row("SELECT * FROM notes WHERE id=? AND user_id=?", (nid, session["uid"]))
    action = (request.get_json(silent=True) or {}).get("action")
    if not n or action not in ("summarize", "improve", "explain"):
        return fail("Invalid request.")
    try:
        out = provider.chat([{"role": "system", "content": f"You {action} the user's note. Be clear and beginner-friendly."},
                             {"role": "user", "content": n["content"][:8000]}])
    except AIError as e:
        return fail(str(e), 503)
    return ok(result=out)

# ---- analytics, settings, export
@app.get("/api/analytics")
@login_required
def analytics():
    u = (session["uid"],)
    c = lambda sql: rows(sql, u)[0]["n"]
    return ok(conversations=c("SELECT COUNT(*) n FROM conversations WHERE user_id=?"),
              messages=c("SELECT COUNT(*) n FROM messages WHERE user_id=?"),
              documents=c("SELECT COUNT(*) n FROM documents WHERE user_id=?"),
              tasks=c("SELECT COUNT(*) n FROM tasks WHERE user_id=?"),
              completed=c("SELECT COUNT(*) n FROM tasks WHERE user_id=? AND status='COMPLETED'"),
              notes=c("SELECT COUNT(*) n FROM notes WHERE user_id=?"),
              memories=c("SELECT COUNT(*) n FROM memories WHERE user_id=?"),
              ai_requests=c("SELECT COUNT(*) n FROM messages WHERE user_id=? AND role='assistant'"))

@app.delete("/api/data/<what>")
@login_required
def clear_data(what):
    uid = session["uid"]
    if what == "conversations":
        run("DELETE FROM messages WHERE user_id=?", (uid,))
        run("DELETE FROM conversations WHERE user_id=?", (uid,))
    elif what == "memories":
        run("DELETE FROM memories WHERE user_id=?", (uid,))
    elif what == "documents":
        for d in rows("SELECT id FROM documents WHERE user_id=?", (uid,)):
            remove_document(uid, d["id"])
    else:
        return fail("Unknown data type.")
    return ok()

@app.get("/api/export")
@login_required
def export():
    uid = session["uid"]
    q = lambda t, cols="*": rows(f"SELECT {cols} FROM {t} WHERE user_id=?", (uid,))   # t is a fixed name from code below
    data = {"profile": {"username": session["username"]},
            "conversations": q("conversations"), "messages": q("messages", "id,conversation_id,role,content,created_at"),
            "tasks": q("tasks"), "notes": q("notes"), "memories": q("memories"),
            "documents": q("documents", "id,filename,filetype,size,status,created_at")}
    resp = jsonify(data)
    resp.headers["Content-Disposition"] = "attachment; filename=nexus-export.json"
    return resp

@app.get("/")
def index():
    return INDEX_HTML

# ------------------------------------------------------------ 8 FRONTEND
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEXUS - Autonomous AI Workspace</title>
<style>
:root{--bg:#0e1116;--panel:#161b22;--panel2:#1d242d;--line:#2a323d;--text:#e8edf2;--mute:#8b96a3;--acc:#7c8cff;--acc2:#5463e0;--ok:#4fd1a5;--bad:#ff6b6b}
[data-theme=light]{--bg:#f4f6f9;--panel:#fff;--panel2:#eef1f6;--line:#d8dee7;--text:#16202b;--mute:#5b6775;--acc:#4452d6;--acc2:#3441b8}
*{box-sizing:border-box}body{margin:0;font:15px/1.55 "Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--text)}
button,input,select,textarea{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.btn{background:var(--acc2);color:#fff;border:0;border-radius:8px;padding:8px 14px;cursor:pointer}
.btn.ghost{background:var(--panel2);color:var(--text);border:1px solid var(--line)}.btn.sm{padding:4px 10px;font-size:13px}.btn.danger{background:var(--bad);color:#111}
input,select,textarea{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:9px 11px;width:100%}
label{display:block;font-size:13px;color:var(--mute);margin:10px 0 4px}
#auth{max-width:380px;margin:10vh auto;padding:28px;background:var(--panel);border:1px solid var(--line);border-radius:16px}
#auth h1{margin:0}#app{display:none;min-height:100vh}
aside{position:fixed;inset:0 auto 0 0;width:230px;background:var(--panel);border-right:1px solid var(--line);padding:18px 12px;overflow:auto;transition:transform .2s}
aside h2{margin:0 8px 2px;letter-spacing:.14em}aside small{margin:0 8px 16px;display:block;color:var(--mute)}
nav button{display:flex;gap:10px;align-items:center;width:100%;background:none;border:0;border-radius:8px;padding:9px 10px;cursor:pointer;text-align:left;color:var(--mute)}
nav button:hover,nav button.on{background:var(--panel2);color:var(--text)}nav svg{width:18px;height:18px;stroke:currentColor;fill:none;stroke-width:2}
main{margin-left:230px;padding:0 24px 24px}
header.top{position:sticky;top:0;background:var(--bg);display:flex;align-items:center;gap:10px;padding:14px 0;z-index:5}
header.top h1{font-size:20px;margin:0;flex:1}#menu{display:none}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:20px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}.card b{font-size:26px;display:block}.card span{color:var(--mute);font-size:13px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.between{justify-content:space-between}.mute{color:var(--mute);font-size:13px}
.item{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:10px}
.tag{font-size:12px;padding:2px 8px;border-radius:99px;background:var(--panel2);border:1px solid var(--line)}.HIGH{color:var(--bad)}.done{text-decoration:line-through;color:var(--mute)}
.chat{display:grid;grid-template-columns:200px 1fr;gap:14px;height:calc(100vh - 90px)}
.convs{overflow:auto}.convs div{padding:8px 10px;border-radius:8px;cursor:pointer;font-size:13px;display:flex;justify-content:space-between;gap:6px}.convs div:hover,.convs .on{background:var(--panel2)}
.pane{display:flex;flex-direction:column;min-height:0}.msgs{flex:1;overflow:auto;padding-right:6px}
.msg{display:flex;gap:10px;margin:12px 0}.av{flex:none;width:32px;height:32px;border-radius:50%;display:grid;place-items:center;font-size:12px;font-weight:700;background:var(--acc2);color:#fff}.msg.user .av{background:var(--panel2);color:var(--text)}
.bub{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 14px;max-width:760px;overflow-wrap:anywhere}.bub pre{background:#0b0e13;color:#dbe4ee;padding:10px;border-radius:8px;overflow:auto}
.bub code{background:var(--panel2);padding:1px 5px;border-radius:4px}.bub pre code{background:none}
.dots span{display:inline-block;width:7px;height:7px;margin:0 2px;border-radius:50%;background:var(--mute);animation:b 1s infinite}.dots span:nth-child(2){animation-delay:.15s}.dots span:nth-child(3){animation-delay:.3s}
@keyframes b{50%{transform:translateY(-5px)}}@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
.drop{border:2px dashed var(--line);border-radius:14px;padding:26px;text-align:center;color:var(--mute);margin-bottom:16px}
.bar{height:10px;border-radius:5px;background:var(--acc);margin-top:6px}
#toast{position:fixed;right:18px;bottom:18px;display:grid;gap:8px;z-index:20}#toast div{background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--acc);padding:10px 14px;border-radius:8px}#toast .err{border-left-color:var(--bad)}
@media(max-width:800px){aside{transform:translateX(-100%);z-index:10}aside.open{transform:none}main{margin-left:0;padding:0 14px 14px}#menu{display:inline-block}.chat{grid-template-columns:1fr}.convs{display:none}}
</style></head><body>
<div id="auth"><h1>NEXUS</h1><p class="mute">Autonomous AI Personal &amp; Productivity Workspace</p>
<label for="u">Username</label><input id="u" autocomplete="username">
<label for="p">Password (8+ characters)</label><input id="p" type="password" autocomplete="current-password">
<div class="row" style="margin-top:16px"><button class="btn" id="li">Log in</button><button class="btn ghost" id="rg">Create account</button></div></div>
<div id="app"><aside id="side"><h2>NEXUS</h2><small>AI workspace</small><nav id="nav"></nav></aside>
<main><header class="top"><button class="btn ghost sm" id="menu" aria-label="Open menu">Menu</button><h1 id="title"></h1>
<button class="btn ghost sm" id="theme" aria-label="Toggle theme">Theme</button><button class="btn ghost sm" id="out">Log out</button></header>
<section id="view" aria-live="polite"></section></main></div><div id="toast" role="status"></div>
<script>
const S={csrf:null,user:null,view:'dashboard',mode:'chat',conv:null,doc:null,busy:false,aiReady:true};
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function toast(m,bad){const d=document.createElement('div');d.textContent=m;if(bad)d.className='err';$('#toast').append(d);setTimeout(()=>d.remove(),4000)}
async function api(p,m='GET',b){const o={method:m,headers:{'X-CSRF-Token':S.csrf||''}};
 if(b instanceof FormData)o.body=b;else if(b){o.headers['Content-Type']='application/json';o.body=JSON.stringify(b)}
 const r=await fetch('/api/'+p,o);const d=await r.json().catch(()=>({error:'Unexpected server response'}));
 if(!r.ok)throw new Error(d.error||'Request failed');return d}
const act=f=>async(...a)=>{try{await f(...a)}catch(e){toast(e.message,1)}};
// Safe markdown: escape everything first, then add code blocks / bold / inline code
function md(t){let h=esc(t);h=h.replace(/```(\w*)\n?([\s\S]*?)```/g,(m,l,c)=>'<pre><code>'+c+'</code></pre><button class="btn ghost sm" onclick="copyCode(this)">Copy code</button>');
 return h.replace(/`([^`\n]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>').replace(/\n/g,'<br>')}
function copyCode(b){navigator.clipboard.writeText(b.previousSibling.textContent);toast('Code copied')}
const NAV=[['dashboard','Dashboard','M3 12l9-9 9 9M5 10v10h14V10'],['chat','AI Chat','M21 12a8 8 0 01-12 7l-5 1 1-5a8 8 0 1116-3'],['documents','Documents','M6 2h9l5 5v15H6zM14 2v6h6'],
['research','Research','M11 4a7 7 0 100 14 7 7 0 000-14zM21 21l-5-5'],['study','Study','M4 5l8-3 8 3-8 3zM6 10v6c3 2 9 2 12 0v-6'],['code','Coding','M8 7l-5 5 5 5M16 7l5 5-5 5'],
['tasks','Tasks','M4 12l5 5L20 6'],['notes','Notes','M5 3h14v18H5zM9 8h6M9 12h6'],['memory','Memory','M12 3a6 6 0 00-4 10v4h8v-4a6 6 0 00-4-10zM9 21h6'],['analytics','Analytics','M4 20V10M10 20V4M16 20v-8M22 20H2'],['settings','Settings','M12 8a4 4 0 100 8 4 4 0 000-8z']];
function nav(){$('#nav').innerHTML=NAV.map(n=>`<button data-v="${n[0]}" class="${S.view===n[0]?'on':''}"><svg viewBox="0 0 24 24"><path d="${n[2]}"/></svg>${n[1]}</button>`).join('')}
$('#nav').onclick=e=>{const b=e.target.closest('button');if(b)go(b.dataset.v)};
$('#menu').onclick=()=>$('#side').classList.toggle('open');
function go(v,opt={}){S.view=v;$('#side').classList.remove('open');if(['chat','research','study','code'].includes(v)){S.mode=v==='chat'?'chat':v;if(!opt.keep){S.conv=null;S.doc=null}}
 nav();$('#title').textContent=(NAV.find(n=>n[0]===v)||[])[1];({dashboard,chat:chatV,research:chatV,study:chatV,code:chatV,documents,tasks,notes,memory,analytics,settings})[v]()}
async function boot(){try{const m=await api('me');S.csrf=m.csrf;S.user=m.username;S.aiReady=m.ai_ready;$('#auth').style.display='none';$('#app').style.display='block';go('dashboard')}catch{$('#auth').style.display='block'}}
async function authGo(path){try{await api(path,'POST',{username:$('#u').value,password:$('#p').value});$('#p').value='';boot()}catch(e){toast(e.message,1)}}
$('#li').onclick=()=>authGo('login');$('#rg').onclick=()=>authGo('register');
$('#out').onclick=async()=>{await api('logout','POST');location.reload()};
$('#theme').onclick=()=>{const h=document.documentElement;h.dataset.theme=h.dataset.theme==='dark'?'light':'dark';localStorage.setItem('theme',h.dataset.theme)};
document.documentElement.dataset.theme=localStorage.getItem('theme')||'dark';

const dashboard=act(async()=>{const a=await api('analytics');const h=new Date().getHours();const g=h<12?'Good morning':h<18?'Good afternoon':'Good evening';
 const st=[['Conversations',a.conversations],['Documents',a.documents],['Tasks',a.tasks],['Notes',a.notes],['AI requests',a.ai_requests],['Completed tasks',a.completed]];
 $('#view').innerHTML=`<h2>${g}, ${esc(S.user)}</h2>${S.aiReady?'':'<div class="item">AI is not configured. Add OPENAI_API_KEY to your .env file and restart the app.</div>'}
 <div class="grid">${st.map(s=>`<div class="card"><b>${s[1]}</b><span>${s[0]}</span></div>`).join('')}</div>
 <div class="row"><button class="btn" onclick="go('chat')">Ask AI</button><button class="btn ghost" onclick="go('documents')">Upload document</button><button class="btn ghost" onclick="go('research')">Research</button><button class="btn ghost" onclick="go('tasks')">Create task</button><button class="btn ghost" onclick="go('notes')">Create note</button></div>
 <p class="mute">Try in chat: "Calculate 25 * 48", "Remind me to study Python tomorrow", "Remember that I prefer Python".</p>`});

// ---------- chat (also used for Research / Study / Coding modes)
const chatV=act(async()=>{const hints={chat:'Ask anything...',research:'Research topic...',study:'Topic, difficulty, and what to generate (quiz, flashcards, study plan)...',code:'Paste code or describe the problem...'};
 $('#view').innerHTML=`<div class="chat"><div class="convs" id="convs"></div><div class="pane"><div class="msgs" id="msgs"></div>
 ${S.doc?`<div class="mute">Asking about the selected document. <button class="btn ghost sm" onclick="S.doc=null;go(S.view,{keep:1})">Clear</button></div>`:''}
 <div class="row" style="margin-top:8px"><textarea id="inp" rows="2" aria-label="Message" placeholder="${hints[S.mode]}"></textarea><button class="btn" id="send">Send</button></div></div></div>`;
 $('#send').onclick=()=>send();$('#inp').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}};
 await loadConvs();if(S.conv)await loadMsgs();else $('#msgs').innerHTML='<p class="mute">Start a conversation below.</p>'});
const loadConvs=async()=>{const c=await api('conversations');$('#convs').innerHTML=`<button class="btn sm" onclick="S.conv=null;go(S.view,{keep:1})">+ New</button>`+c.items.map(i=>`<div class="${i.id===S.conv?'on':''}" onclick="S.conv=${i.id};go(S.view,{keep:1})"><span>${esc(i.title)}</span><span><a href="#" onclick="event.stopPropagation();renameC(${i.id},'${esc(i.title).replace(/'/g,'')}')" aria-label="Rename">✎</a> <a href="#" onclick="event.stopPropagation();delC(${i.id})" aria-label="Delete">✕</a></span></div>`).join('')};
const renameC=act(async(id,t)=>{const n=prompt('Rename conversation',t);if(n){await api('conversations/'+id,'PUT',{title:n});loadConvs()}});
const delC=act(async id=>{if(confirm('Delete this conversation?')){await api('conversations/'+id,'DELETE');if(S.conv===id)S.conv=null;go(S.view,{keep:1})}});
const loadMsgs=async()=>{const m=await api(`conversations/${S.conv}/messages`);$('#msgs').innerHTML='';m.items.forEach(x=>addMsg(x.role,x.content,x.meta?JSON.parse(x.meta):null,x.created_at))};
function addMsg(role,text,meta,time){const d=document.createElement('div');d.className='msg '+role;
 let extra='';if(meta){if(meta.intent)extra+=`<div class="mute">Intent: ${esc(meta.intent)} · Tools used: ${meta.tools&&meta.tools.length?esc(meta.tools.join(', ')):'none'}</div>`;
 if(meta.sources&&meta.sources.length)extra+='<div class="mute"><b>Sources:</b><br>'+meta.sources.map(s=>'• '+esc(s)).join('<br>')+'</div>'}
 d.innerHTML=`<div class="av">${role==='user'?'YOU':'AI'}</div><div class="bub"><div>${md(text)}</div>${extra}<div class="mute">${esc(time||new Date().toLocaleTimeString())}${role==='assistant'?` · <a href="#" class="cp">Copy</a> · <a href="#" class="rg">Regenerate</a>`:''}</div></div>`;
 const cp=d.querySelector('.cp');if(cp){cp.onclick=e=>{e.preventDefault();navigator.clipboard.writeText(text);toast('Copied')};d.querySelector('.rg').onclick=e=>{e.preventDefault();regen()}}
 $('#msgs').append(d);$('#msgs').scrollTop=1e9;return d}
async function regen(){const u=[...document.querySelectorAll('.msg.user .bub>div:first-child')].pop();if(u)send(u.innerText,true)}
const send=act(async(text,regenerate)=>{const inp=$('#inp');text=text||inp.value.trim();if(!text||S.busy)return;S.busy=true;inp.value='';
 if(!regenerate)addMsg('user',text);const t=addMsg('assistant','');t.querySelector('.bub').innerHTML='<div class="dots"><span></span><span></span><span></span></div>';
 try{const r=await api('chat','POST',{message:text,mode:S.mode,conversation_id:S.conv,document_id:S.doc,regenerate:!!regenerate});t.remove();addMsg('assistant',r.reply,r,null);
  if(!S.conv){S.conv=r.conversation_id}loadConvs()}catch(e){t.remove();addMsg('assistant','⚠ '+e.message)}finally{S.busy=false}});

// ---------- documents
const documents=act(async()=>{const d=await api('documents');
 $('#view').innerHTML=`<div class="drop"><p>Upload a PDF, TXT or DOCX (max 10 MB)</p><label class="btn" for="file" style="display:inline-block">Choose file</label><input id="file" type="file" accept=".pdf,.txt,.docx" style="display:none"></div>`+
 (d.items.length?d.items.map(x=>`<div class="item row between"><div><b>${esc(x.filename)}</b><div class="mute">${esc(x.filetype)} · ${(x.size/1024).toFixed(1)} KB · ${esc(x.created_at)} · ${esc(x.status)}</div></div>
 <div class="row"><button class="btn sm" onclick="askDoc(${x.id})">Ask AI</button><button class="btn ghost sm" onclick="askDoc(${x.id},'Summarize this document.')">Summarize</button><button class="btn ghost sm" onclick="askDoc(${x.id},'Create 20 exam questions from this document.')">Quiz</button><button class="btn danger sm" onclick="delDoc(${x.id})">Delete</button></div></div>`).join(''):'<p class="mute">No documents yet. Upload one to ask questions about it.</p>');
 $('#file').onchange=act(async e=>{const f=new FormData();f.append('file',e.target.files[0]);toast('Processing document...');await api('upload','POST',f);toast('Document ready');documents()})});
const askDoc=(id,q)=>{S.doc=id;go('chat',{keep:1});S.conv=null;setTimeout(()=>{if(q){$('#inp').value=q;send()}},100)};
const delDoc=act(async id=>{if(confirm('Delete this document?')){await api('documents/'+id,'DELETE');documents()}});

// ---------- tasks
const tasks=act(async()=>{const t=await api('tasks');
 $('#view').innerHTML=`<div class="row"><input id="tt" placeholder="New task title" aria-label="Task title" style="flex:2"><select id="tp" aria-label="Priority" style="flex:1"><option>MEDIUM</option><option>HIGH</option><option>LOW</option></select><input id="td" type="date" aria-label="Due date" style="flex:1"><button class="btn" id="ta">Add</button></div>
 <div class="row" style="margin:12px 0"><input id="tf" placeholder="Search tasks" aria-label="Search tasks" style="flex:2"><select id="ts" aria-label="Filter status" style="flex:1"><option value="">All</option><option>TODO</option><option>IN_PROGRESS</option><option>COMPLETED</option></select></div><div id="tl"></div>`;
 const draw=()=>{const q=$('#tf').value.toLowerCase(),s=$('#ts').value;const l=t.items.filter(x=>(!s||x.status===s)&&x.title.toLowerCase().includes(q));
  $('#tl').innerHTML=l.length?l.map(x=>`<div class="item row between"><div class="row"><input type="checkbox" style="width:auto" ${x.status==='COMPLETED'?'checked':''} aria-label="Complete" onchange="setTask(${x.id},this.checked?'COMPLETED':'TODO')"><div><span class="${x.status==='COMPLETED'?'done':''}">${esc(x.title)}</span><div class="mute"><span class="tag ${x.priority}">${x.priority}</span> ${esc(x.status)} ${x.due_date?'· due '+esc(x.due_date):''}</div></div></div>
  <div class="row"><button class="btn ghost sm" onclick="editTask(${x.id},'${esc(x.title).replace(/'/g,'')}')">Edit</button><button class="btn danger sm" onclick="delTask(${x.id})">Delete</button></div></div>`).join(''):'<p class="mute">No tasks match. Add one above.</p>'};
 draw();$('#tf').oninput=draw;$('#ts').onchange=draw;
 $('#ta').onclick=act(async()=>{await api('tasks','POST',{title:$('#tt').value,priority:$('#tp').value,due_date:$('#td').value});tasks()})});
const setTask=act(async(id,status)=>{await api('tasks/'+id,'PUT',{status});tasks()});
const editTask=act(async(id,t)=>{const n=prompt('Edit task title',t);if(n){await api('tasks/'+id,'PUT',{title:n});tasks()}});
const delTask=act(async id=>{await api('tasks/'+id,'DELETE');tasks()});

// ---------- notes
const notes=act(async()=>{const n=await api('notes');
 $('#view').innerHTML=`<div class="item"><input id="nt" placeholder="Note title" aria-label="Note title"><textarea id="nc" rows="4" placeholder="Write your note..." aria-label="Note content" style="margin-top:8px"></textarea><button class="btn" id="na" style="margin-top:8px">Save note</button></div>
 <input id="nq" placeholder="Search notes" aria-label="Search notes" style="margin-bottom:12px"><div id="nl"></div>`;
 const draw=()=>{const q=$('#nq').value.toLowerCase();const l=n.items.filter(x=>(x.title+x.content).toLowerCase().includes(q));
  $('#nl').innerHTML=l.length?l.map(x=>`<div class="item"><div class="row between"><b>${esc(x.title)}</b><span class="row"><button class="btn ghost sm" onclick="noteAI(${x.id},'summarize')">Summarize</button><button class="btn ghost sm" onclick="noteAI(${x.id},'improve')">Improve</button><button class="btn ghost sm" onclick="noteAI(${x.id},'explain')">Explain</button><button class="btn ghost sm" onclick="editNote(${x.id})">Edit</button><button class="btn danger sm" onclick="delNote(${x.id})">Delete</button></span></div><div>${md(x.content)}</div><div class="mute">Updated ${esc(x.updated_at)}</div><div id="ai${x.id}"></div></div>`).join(''):'<p class="mute">No notes found.</p>'};
 draw();$('#nq').oninput=draw;window._notes=n.items;
 $('#na').onclick=act(async()=>{await api('notes','POST',{title:$('#nt').value,content:$('#nc').value});notes()})});
const noteAI=act(async(id,a)=>{$('#ai'+id).textContent='Thinking...';try{const r=await api(`notes/${id}/ai`,'POST',{action:a});$('#ai'+id).innerHTML='<div class="bub">'+md(r.result)+'</div>'}catch(e){$('#ai'+id).textContent='';throw e}});
const editNote=act(async id=>{const x=window._notes.find(n=>n.id===id);const t=prompt('Title',x.title);if(t===null)return;const c=prompt('Content',x.content);if(c===null)return;await api('notes/'+id,'PUT',{title:t,content:c});notes()});
const delNote=act(async id=>{if(confirm('Delete this note?')){await api('notes/'+id,'DELETE');notes()}});

// ---------- memory
const memory=act(async()=>{const m=await api('memories');
 $('#view').innerHTML=`<p class="mute">Long-term memory is only saved when you add it here or say "Remember that ..." in chat.</p><div class="row"><input id="mi" placeholder="e.g. I prefer beginner-friendly explanations" aria-label="New memory" style="flex:1"><button class="btn" id="ma">Add memory</button></div>
 <input id="mq" placeholder="Search memory" aria-label="Search memory" style="margin:12px 0"><div id="ml"></div>`;
 const draw=()=>{const q=$('#mq').value.toLowerCase();const l=m.items.filter(x=>x.content.toLowerCase().includes(q));
  $('#ml').innerHTML=l.length?l.map(x=>`<div class="item row between"><div>${esc(x.content)}<div class="mute">Created ${esc(x.created_at)}</div></div><button class="btn danger sm" onclick="delMem(${x.id})">Delete</button></div>`).join(''):'<p class="mute">No memories yet.</p>'};
 draw();$('#mq').oninput=draw;$('#ma').onclick=act(async()=>{await api('memories','POST',{content:$('#mi').value});memory()})});
const delMem=act(async id=>{await api('memories/'+id,'DELETE');memory()});

// ---------- analytics (simple bar chart in plain HTML/CSS)
const analytics=act(async()=>{const a=await api('analytics');const keys=['conversations','messages','documents','tasks','completed','notes','memories','ai_requests'];const mx=Math.max(1,...keys.map(k=>a[k]));
 $('#view').innerHTML=keys.map(k=>`<div class="item"><div class="row between"><span>${k.replace('_',' ')}</span><b>${a[k]}</b></div><div class="bar" style="width:${a[k]/mx*100}%"></div></div>`).join('')});

// ---------- settings
const settings=()=>{$('#view').innerHTML=`<div class="item"><b>Profile</b><p>Signed in as ${esc(S.user)}</p></div>
 <div class="item"><b>AI settings</b><p class="mute">Provider key and model are set in your .env file (never sent to the browser). AI status: ${S.aiReady?'configured':'not configured'}.</p></div>
 <div class="item"><b>Theme</b><p><button class="btn ghost sm" onclick="$('#theme').click()">Toggle dark/light</button></p></div>
 <div class="item"><b>Data management</b><div class="row"><button class="btn ghost sm" onclick="wipe('conversations')">Clear conversations</button><button class="btn ghost sm" onclick="wipe('memories')">Clear memories</button><button class="btn ghost sm" onclick="wipe('documents')">Delete all documents</button><a class="btn sm" href="/api/export" style="text-decoration:none">Export my data (JSON)</a></div></div>
 <div class="item"><b>About</b><p class="mute">NEXUS v1 - single-file Flask app with SQLite, RAG and a tool-using agent.</p></div>`};
const wipe=act(async w=>{if(confirm('Delete all '+w+'? This cannot be undone.')){await api('data/'+w,'DELETE');toast('Cleared '+w)}});
boot();
</script></body></html>"""

# ---------------------------------------------------------------- 9 STARTUP
if __name__ == "__main__":
    init_db()
    log.info("NEXUS starting at http://127.0.0.1:5000")
    if not provider.available():
        log.warning("OPENAI_API_KEY is missing - add it to your .env file. The app will run, but AI replies are disabled.")
    app.run(host="127.0.0.1", port=5000, debug=False)
