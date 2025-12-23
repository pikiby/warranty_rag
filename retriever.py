"""
retriever.py — поиск по базе знаний (RAG retrieval).

Зачем нужен: по тексту запроса пользователя ищет в Chroma (векторной базе) самые релевантные чанки
и возвращает их текст + метаданные. Эти чанки потом передаются в GPT как контекст.

Связано с:
- `ingest.py` — наполняет коллекцию Chroma чанками из `docs/`.
- `app.py` — вызывает `retrieve()` перед запросом к GPT.
"""

import os

import chromadb
from chromadb.utils import embedding_functions


# Назначение: вернуть готовую коллекцию Chroma с включёнными OpenAI-эмбеддингами.
# Зачем: поиск работает по "коллекции" — это контейнер/таблица внутри Chroma, где лежат ВСЕ чанки базы знаний
# (тексты + их embeddings + метаданные). На запрос пользователя мы НЕ создаём новую коллекцию, мы ищем в уже существующей.
# Связано с: `retrieve()` (берёт коллекцию и делает `collection.query(...)`) и `ingest.py::run_ingest()` (кладёт чанки в эту коллекцию).
def _get_collection(*, chroma_path, collection_name):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # Важно: PersistentClient хранит данные на диске в `chroma_path`.
    # Повторный вызов создаёт новый Python-объект, но НЕ "пересоздаёт" данные базы.
    chroma = chromadb.PersistentClient(path=chroma_path)
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
    )
    # `get_or_create_collection`: берёт существующую коллекцию или создаёт её, если её ещё нет.
    return chroma.get_or_create_collection(collection_name, embedding_function=embedding_function)


# Назначение: получить top-k релевантных текстовых чанков из векторного индекса.
# Зачем: RAG — это "найти подходящие куски из базы знаний и передать их в GPT как контекст".
# GPT сам не ходит в Chroma, поэтому retrieval — обязательный шаг перед ответом.
# Связано с: `app.py::_answer_with_rag()` (вызывает `retrieve()`), `_get_collection()` (подключает коллекцию),
# и `ingest.py::run_ingest()` (загружает данные, по которым мы ищем).
def retrieve(
    *,
    query,
    k=10,
    chroma_path="data/chroma",
    collection_name="kb_docs",
):
    query_text = (query or "").strip()
    if not query_text:
        # Пустой запрос → нечего искать → возвращаем "нет результатов".
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Минимальное поведение: без ключа мы не можем эмбеддить запрос для retrieval.
        # Возвращаем пустой список, чтобы приложение отдало стандартное сообщение про отсутствие данных.
        return []

    collection = _get_collection(chroma_path=chroma_path, collection_name=collection_name)
    # Запрос в коллекцию:
    # 1) Chroma получает embedding для `query_text` (через OpenAIEmbeddingFunction).
    # 2) Сравнивает его с embeddings всех чанков в коллекции.
    # 3) Возвращает top-k ближайших чанков.
    query_result = collection.query(
        query_texts=[query_text],
        n_results=max(1, int(k)),
        # Что именно вернуть в ответе:
        # - documents: текст найденного чанка (это потом идёт в контекст для GPT)
        # - metadatas: {source, path} — чтобы понимать, из какого файла пришёл чанк
        # - distances: "насколько близко" (обычно чем меньше, тем релевантнее)
        include=["documents", "metadatas", "distances"],
    )

    # Chroma поддерживает сразу несколько запросов (`query_texts=[...]`), поэтому возвращает "список списков".
    # Мы передаём один запрос, поэтому берём первый элемент `[0]`.
    documents = (query_result.get("documents") or [[]])[0]
    metadatas = (query_result.get("metadatas") or [[]])[0]
    distances = (query_result.get("distances") or [[]])[0]

    hits = []
    for index, document_text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        if document_text:
            hits.append(
                {
                    "text": document_text,
                    "source": (metadata or {}).get("source"),
                    "path": (metadata or {}).get("path"),
                    "score": distance,
                }
            )
    return hits



