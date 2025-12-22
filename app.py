"""
app.py — точка входа Streamlit-чата.

Зачем нужен: это UI, который принимает вопрос пользователя, запускает RAG (поиск контекста в базе знаний)
и отправляет запрос в GPT, чтобы ответ был строго по найденному контексту.

Связано с:
- `retriever.py` — достаёт top-k чанков из коллекции Chroma.
- `prompts.py` — формирует prompt (system+context+user) для GPT.
- `ingest.py` — отдельный скрипт индексации, который наполняет Chroma данными из `docs/`.
"""

import os

import streamlit as st
from openai import OpenAI

import ingest
import prompts
import retriever


APP_TITLE = "Minimal RAG Chat"

KB_DOCS_DIR = os.getenv("KB_DOCS_DIR", "docs")
CHROMA_PATH = os.getenv("KB_CHROMA_PATH", "data/chroma")
COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "kb_docs")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# Назначение: переиндексировать базу знаний при старте процесса Streamlit (то есть при перезапуске сервиса).
# Зачем: чтобы на деплое индекс создавался автоматически, без ручного запуска `python ingest.py`.
# Связано с: `ingest.run_ingest()` (строит индекс) и `retriever.retrieve()` (потом читает индекс).
@st.cache_resource
def _auto_ingest_on_start():
    return ingest.run_ingest(
        doc_dir=KB_DOCS_DIR,
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )


# Автоиндексация при старте (один раз на процесс).
# Важно: Streamlit перезапускает скрипт на каждое действие пользователя, поэтому используем cache_resource.
if os.getenv("OPENAI_API_KEY"):
    _auto_ingest_on_start()


# Назначение: в Streamlit код перезапускается на каждое действие пользователя, поэтому клиент OpenAI кэшируем и переиспользуем,
# чтобы не создавать его заново при каждом сообщении в чате.
# Связано с: `_answer_with_rag()`, которая берёт клиента и делает запрос `client.chat.completions.create(...)`.
@st.cache_resource
def _get_openai_client():
    return OpenAI()


# Назначение: превратить найденные куски базы знаний в одну строку, которую мы передаём в GPT.
# Зачем: GPT сам не “ходит” в Chroma и не видит базу знаний — он отвечает только по тексту, который мы положим в prompt.
# Связано с: `retriever.retrieve()` (находит куски) и `_answer_with_rag()`/`prompts.build_messages()` (кладут контекст в запрос к GPT).
def _build_context_text(hits):
    parts = []
    for hit in hits:
        chunk_text = (hit.get("text") or "").strip()
        if chunk_text:
            parts.append(chunk_text)
    return "\n\n".join(parts).strip()


# Назначение: взять историю чата из `st.session_state` и подготовить её для передачи в GPT.
# Зачем: по умолчанию GPT “не помнит” прошлые сообщения — он видит только то, что мы кладём в `messages`.
# Связано с: `st.session_state.messages` (хранилище UI-истории) и `prompts.build_messages()` (сборка messages для GPT).
def _get_chat_history_for_gpt():
    history = []
    for message in st.session_state.get("messages", []):
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role not in ("user", "assistant"):
            continue
        if not content:
            continue
        # Первое приветствие ассистента нужно для UI, но не обязательно для логики диалога у GPT.
        if role == "assistant" and content.startswith("Задайте вопрос."):
            continue
        history.append({"role": role, "content": content})
    return history


# Назначение: выполнить самый простой RAG-пайплайн: найти контекст и ответить строго по нему.
# Зачем: GPT отвечает только по тексту, который мы ему передали. Поэтому сначала делаем retrieval (поиск чанков в коллекции Chroma),
# потом кладём найденный текст в prompt и только после этого спрашиваем GPT.
# Связано с: `retriever.retrieve()` (ищет top-k чанков в общей коллекции базы знаний) и `prompts.build_messages()` (собирает prompt).
def _answer_with_rag(question):
    hits = retriever.retrieve(
        query=question,
        k=5,
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )
    context_text = _build_context_text(hits)
    client = _get_openai_client()
    history = _get_chat_history_for_gpt()
    # В историю уже попал текущий вопрос (мы добавляем его в UI-историю до вызова `_answer_with_rag()`).
    # Поэтому здесь достаточно передать history целиком, без отдельного question.
    messages = prompts.build_messages(context=context_text, history=history)
    resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
    return (resp.choices[0].message.content or "").strip()


# =============================================================================
# UI (Streamlit)
# =============================================================================
# Назначение: этот блок — "экран" приложения. Он показывает историю чата, принимает новый вопрос
# и запускает RAG-ответ через `_answer_with_rag()`.
# Связано с: `_answer_with_rag()` (получает ответ), `_build_context_text()` (собирает контекст),
# `retriever.retrieve()` (ищет в базе знаний) и `prompts.build_messages()` (формирует prompt для GPT).

# Настройка страницы (заголовок вкладки браузера).
st.set_page_config(page_title=APP_TITLE)

# Заголовок приложения на странице.
st.title(APP_TITLE)

# Инициализация истории чата в `st.session_state`.
# Зачем: Streamlit перезапускает скрипт на каждое действие пользователя, а `session_state`
# позволяет сохранить историю сообщений между этими перезапусками.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Задайте вопрос. Я отвечу строго по базе знаний (RAG).",
        }
    ]

# Отрисовка всей истории сообщений (user/assistant) на экране.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Поле ввода пользователя (чат).
question = st.chat_input("Ваш вопрос")
if question:
    # 1) Сохраняем вопрос в историю.
    st.session_state.messages.append({"role": "user", "content": question})

    # 2) Сразу показываем вопрос в чате.
    with st.chat_message("user"):
        st.markdown(question)

    # 3) Получаем ответ: внутри `_answer_with_rag()` происходит retrieval (поиск) и запрос к GPT.
    answer = _answer_with_rag(question)

    # 4) Сохраняем ответ в историю и показываем его в чате.
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)


