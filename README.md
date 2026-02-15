# Role-based RAG (FastAPI + Chroma + MySQL + Docker)

## 🚀 Quick start

```bash
git clone <repo>
cd role-RAG
docker compose up --build
```

Open:

http://localhost:8000/ui

---

## Build Vector DB (first time)

After the services are up, run:

```bash
docker compose run --rm ingest
```

This will:

* read files from `data/`
* create embeddings
* build the Chroma database in `./chroma_db`

⚠️ Run this again only if you changed files in `data/`.


## 👤 Test user

```
artem / artem
```
