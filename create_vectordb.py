import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import pandas as pd
from pathlib import Path

import re

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

emb = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Chroma
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
client = chromadb.PersistentClient(path=CHROMA_PATH)


collection = client.get_or_create_collection(
    name="rag",
    embedding_function=emb
)

CHUNK_CONFIG = {
    "engineering": {"chunk_size": 250, "chunk_overlap": 50},
    "general": {"chunk_size": 250, "chunk_overlap": 50},
    "marketing": {"chunk_size": 800, "chunk_overlap": 100},
    "finance": {"chunk_size": 450, "chunk_overlap": 80},
}

# md loader
def load_and_split_md(glob_pattern="data/*/*.md"):

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]
    )

    documents = []
    metadatas = []
    ids = []

    for path in Path(".").glob(glob_pattern):

        m = re.search(r"_q([1-4])_", path.name.lower())
        quarter = f"Q{m.group(1)}" if m else "-"

        access = path.parent.name

        cfg = CHUNK_CONFIG.get(
            access,
            {"chunk_size": 250, "chunk_overlap": 50}
        )

        recursive = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )


        text = path.read_text(encoding="utf-8")

        sections = md_splitter.split_text(text)

        chunk_counter = 0

        for section in sections:

            h1 = section.metadata.get("h1", "")
            h2 = section.metadata.get("h2", "")
            h3 = section.metadata.get("h3", "")

            prefix = " > ".join([x for x in [h1,h2,h3] if x])

            sub_chunks = recursive.split_text(section.page_content)

            for chunk in sub_chunks:

                chunk_with_pref = f"""
                Document: {path.name}
                Section: {prefix}

                {chunk}
                """.strip()

                metadata = {
                    "source": str(path),
                    "source_type": "md",
                    "access": access,
                    "quarter": quarter,
                    **section.metadata
                }

                documents.append(chunk_with_pref)
                metadatas.append(metadata)
                ids.append(f"md:{path}:{chunk_counter}")

                chunk_counter += 1

    return documents, metadatas, ids


# CSV loader
def load_csv(csv_path, access, text_cols, id_col=None):

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    documents = []
    metadatas = []
    ids = []

    for i, row in df.iterrows():

        # текст чанка
        parts = []
        for c in text_cols:
            val = row.get(c, "")
            if val.strip():
                parts.append(f"{c}: {val}")

        text = "\n".join(parts)

        if not text:
            continue

        metadata = {
            "source": csv_path,
            "source_type": "csv",
            "access": access,
            "row": int(i)
        }

        # остальные поля -> metadata
        for c in df.columns:
            if c in text_cols:
                continue
            val = row.get(c, "")
            if val.strip():
                metadata[c] = val

        if id_col:
            uid = row.get(id_col, i)
            doc_id = f"csv:{csv_path}:{uid}"
        else:
            doc_id = f"csv:{csv_path}:row:{i}"

        documents.append(text)
        metadatas.append(metadata)
        ids.append(doc_id)

    return documents, metadatas, ids


md_docs, md_meta, md_ids = load_and_split_md()

for meta in md_meta:
  flag = re.search(r"Q[1-4]", meta.get('h2', ""), flags=re.IGNORECASE)
  if flag:
    meta['quarter'] = flag.group()

collection.upsert(
    documents=md_docs,
    metadatas=md_meta,
    ids=md_ids
)

print("MD added:", len(md_docs))


csv_docs, csv_meta, csv_ids = load_csv(
    "data/hr/hr_data.csv",
    access="hr",
    text_cols=["full_name",
        "role",
        "department",
        "location",
        "email"],
    id_col="employee_id"
)

collection.upsert(
    documents=csv_docs,
    metadatas=csv_meta,
    ids=csv_ids
)

print("CSV added:", len(csv_docs))