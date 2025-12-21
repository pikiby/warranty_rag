"""
ingest.py — индексация базы знаний в Chroma (векторное хранилище).

Зачем нужен: читает файлы из `docs/`, режет их на чанки и записывает в коллекцию Chroma.
После этого `retriever.py` может искать по смыслу (через embeddings), а `app.py` — отвечать через RAG.

Связано с:
- `retriever.py` — ищет по той же коллекции Chroma, которую мы наполняем здесь.
- `app.py` — не индексирует сам, а только читает индекс через retriever.
"""

# --- SQLite shim (нужен в некоторых контейнерах/облачных окружениях) ---
# Важно: должен выполниться ДО импорта chromadb.
try:
    import sqlite3
    from sqlite3 import sqlite_version

    def _version_tuple(version_str):
        return tuple(int(part) for part in version_str.split("."))

    NEEDS_SHIM = _version_tuple(sqlite_version) < (3, 35, 0)
except Exception:
    NEEDS_SHIM = True

if NEEDS_SHIM:
    import sys
    import pysqlite3  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# ---------------------------------------------------------------

import glob
import os

import chromadb
from chromadb.utils import embedding_functions


DEFAULT_DOC_DIR = "docs"
DEFAULT_CHROMA_PATH = os.getenv("KB_CHROMA_PATH", "data/chroma")
DEFAULT_COLLECTION = os.getenv("KB_COLLECTION_NAME", "kb_docs")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


# Назначение: прочитать все markdown-файлы из `doc_dir` рекурсивно.
# Связано с: `run_ingest()`, которая превращает эти тексты в векторные чанки.
def _load_md_files(doc_dir):
    items = []
    for file_path in glob.glob(f"{doc_dir}/**/*.md", recursive=True):
        with open(file_path, "r", encoding="utf-8") as file:
            file_text = file.read()
        if file_text.strip():
            items.append((file_path, file_text))
    return items


# Назначение: нарезать текст на перекрывающиеся символьные чанки (просто и детерминированно).
# Связано с: `run_ingest()`, которая использует это для формирования payload в Chroma.
def _chunk_text(text, *, chunk_size=900, overlap=120):
    clean_text = (text or "").strip()
    if not clean_text:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(clean_text), step):
        chunk_text = clean_text[start : start + chunk_size].strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


# Назначение: собрать payload (id, document, metadata) для upsert в Chroma.
# Связано с: `run_ingest()`, которая записывает этот payload в коллекцию.
def _build_payload(md_files):
    payload = []
    for file_path, file_text in md_files:
        title = os.path.basename(file_path)
        for chunk_index, chunk_text in enumerate(_chunk_text(file_text)):
            payload.append(
                {
                    "id": f"{title}::{chunk_index}",
                    "text": chunk_text,
                    "meta": {"source": title, "path": file_path},
                }
            )
    return payload


# Назначение: проиндексировать `docs/*.md` в ChromaDB через OpenAI-эмбеддинги.
# Зачем: превратить файлы базы знаний в "коллекцию" Chroma (чанки + embeddings), чтобы потом `retriever.retrieve()`
# мог находить релевантные куски по смыслу (а не по точному совпадению текста).
# Важно: коллекция включает ВСЕ проиндексированные чанки из `docs/`, а на запрос пользователя мы возвращаем только top-k.
# Связано с: `retriever._get_collection()` и `retriever.retrieve()` (они ищут по этой же коллекции).
def run_ingest(
    *,
    doc_dir=DEFAULT_DOC_DIR,
    chroma_path=DEFAULT_CHROMA_PATH,
    collection_name=DEFAULT_COLLECTION,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
):
    os.makedirs(doc_dir, exist_ok=True)
    os.makedirs(chroma_path, exist_ok=True)

    md_files = _load_md_files(doc_dir)
    payload = _build_payload(md_files)
    if not payload:
        return {"files": len(md_files), "chunks": 0, "added": 0}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # Подключаемся к локальному хранилищу Chroma на диске.
    chroma = chromadb.PersistentClient(path=chroma_path)
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=embedding_model,
    )
    # Получаем/создаём коллекцию — это общий контейнер для всей базы знаний (всех чанков из docs/).
    col = chroma.get_or_create_collection(collection_name, embedding_function=embedding_function)

    # Записываем чанки без явных embeddings: Chroma посчитает embeddings сама через embedding_function.
    col.upsert(
        ids=[x["id"] for x in payload],
        documents=[x["text"] for x in payload],
        metadatas=[x["meta"] for x in payload],
    )

    return {"files": len(md_files), "chunks": len(payload), "added": len(payload)}


if __name__ == "__main__":
    print(run_ingest())



