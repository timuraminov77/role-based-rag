from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import mysql.connector

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from openai import OpenAI

from fastapi.responses import HTMLResponse

import os
import re
from dotenv import load_dotenv


app = FastAPI()
security = HTTPBasic()
load_dotenv()

# SQL
def get_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "root"),
        database=os.getenv("MYSQL_DATABASE", "rag"),
        port=int(os.getenv("MYSQL_PORT", "3306"))
    )


# Login
def login(credentials: HTTPBasicCredentials = Depends(security)):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM users WHERE login = %s", (credentials.username,))
    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if not user or user["password"] != credentials.password:
        raise HTTPException(status_code=401, detail="Wrong login or password")

    return user

def require_role(role: str):
    def checker(user=Depends(login)):
        if user["role"] != role:
            raise HTTPException(status_code=403, detail="Access denied")
        return user
    return checker

@app.get("/hello")
def hello(user=Depends(login)):
    return user["role"]

# Chroma
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
client = chromadb.PersistentClient(path=CHROMA_PATH)

emb = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_collection(
    name="rag",
    embedding_function=emb
)

#Context build
def build_context(filtered_docs):
    parts = []
    for i, d in enumerate(filtered_docs, start=1):
        meta = d["meta"] or {}

        source_path  = meta.get("source", "-")
        h1 = meta.get("h1", "-")
        h2 = meta.get("h2", "-")
        h3 = meta.get("h3", "-")
        row = meta.get("row", "-")

        values = [h1, h2, h3]

        filtered = [v for v in values if v!="-"]

        if row != "-":
            location = f"row {row}"

        elif filtered:
          location = ", ".join(filtered)

        else:
          location = "-"

        salary = meta.get("salary", "-")
        employee_id = meta.get("employee_id", "-")
        role = meta.get("role", "-")

        parts.append(
            f"""[SRC {i}]
              source_path: {source_path}
              location: {location}
              META:
                salary: {salary}
                employee_id: {employee_id}
                role: {role}
              TEXT:
              {d["text"]}
              """
        )
    return "\n\n".join(parts)


llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/chat")
def chat(question: str, user=Depends(login)):

    role = user["role"]

    lim_distance = 0.81

    if role == "hr":
        lim_distance = 0.55

    if role == 'marketing' or role == 'finance':
        m1 = re.search(r"q1", question, flags=re.IGNORECASE)
        m2 = re.search(r"q2", question, flags=re.IGNORECASE)
        m3 = re.search(r"q3", question, flags=re.IGNORECASE)
        m4 = re.search(r"q4", question, flags=re.IGNORECASE)

        if m1:
            quarter = "Q1"
        elif m2:
            quarter = "Q2"
        elif m3:
            quarter = "Q3"
        elif m4:
            quarter = "Q4"
        else:
            quarter = "-"

        if role == "marketing":
            n_results = 7
        else:
            n_results = 3

        results = collection.query(
            query_texts=[question],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
            where={
                "$and": [
                    {"access": role},
                    {"quarter": quarter}
                ]
            }
        )

    else:

        results = collection.query(
            query_texts=[question],
            n_results=3,
            include=["documents", "metadatas", "distances"],
            where = {'access': role}
        )

    if not results.get("documents") or not results["documents"][0]:
        return {"answer": "There is no information"}

    docs = results['documents'][0]
    distances = results['distances'][0]
    metas = results['metadatas'][0]
    ids = results['ids'][0]

    filtered_docs = []

    for doc, meta, dist, id in zip(docs, metas, distances, ids):
        if dist <= lim_distance:
            filtered_docs.append({
                "text": doc,
                'meta': meta,
                'distance': dist,
                'chunk_id': id
            })


    if not filtered_docs:
        return {"answer": "There is no information", "docs": docs, "distances": distances}

    context = build_context(filtered_docs)

    response = llm.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
        {
            "role": "system",
            "content": (
                "Answer ONLY using the provided context.\n"
                "If the context does not contain the answer, respond exactly with: 'There is no information'.\n\n"
    
                "Response format:\n"
                "Short answer\n"
                "Optional explanation (only if needed)\n"
                "Sources: source_path, location\n\n"
    
                "Rules:\n"
                "- Do NOT use external knowledge.\n"
                "- Do NOT invent or guess sources.\n"
                "- Use only sources explicitly present in the context.\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question:\n{question}"
            )
        }
  ],
    )

    answer = response.choices[0].message.content

    #return {"answer": answer}
    return {"answer": answer, "docs": filtered_docs}


# Ui
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>RAG Chat</title>
  <style>
    body { font-family: Arial; max-width: 900px; margin: 40px auto; }
    input, button { font-size: 16px; padding: 10px; }
    input { width: 70%; }
    button { margin-left: 8px; }
    pre { background: #f6f6f6; padding: 12px; white-space: pre-wrap; }
    .small { color: #666; font-size: 13px; }
  </style>
</head>
<body>
  <h2>RAG Chat</h2>

  <input id="q" placeholder="Напиши вопрос..." />
  <button onclick="send()">Отправить</button>

  <h3>Ответ</h3>
  <pre id="answer"></pre>

  <h3>Docs</h3>
  <pre id="docs" class="small"></pre>

<script>
async function send() {
  const qEl = document.getElementById("q");
  const ansEl = document.getElementById("answer");

  const question = (qEl.value || "").trim();
  ansEl.textContent = "Отправляю...";

  const url = "/chat?question=" + encodeURIComponent(question);

  try {
    const resp = await fetch(url, { method: "POST", credentials: "include" });
    const text = await resp.text();

    if (!resp.ok) {
      ansEl.textContent = "HTTP " + resp.status + "\\n" + text;
      return;
    }

    try {
      const data = JSON.parse(text);

      ansEl.textContent = data.answer ?? "";

      document.getElementById("docs").textContent =
        data.docs ? JSON.stringify(data.docs, null, 2) : "-";

    } catch (e) {
      ansEl.textContent = text;
    }
  } catch (e) {
    ansEl.textContent = "Fetch error: " + e;
  }
}
</script>

</body>
</html>
"""




