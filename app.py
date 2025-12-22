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
from clickhouse_client import ClickHouse_client


APP_TITLE = "Minimal RAG Chat"

KB_DOCS_DIR = os.getenv("KB_DOCS_DIR", "docs")
CHROMA_PATH = os.getenv("KB_CHROMA_PATH", "data/chroma")
COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "kb_docs")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB", "db1")


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

#-------------------------
# Назначение: собрать историю SQL-запросов в одном тексте для передачи в GPT.
# Зачем: пользователь может уточнять и продолжать работу, а модель должна видеть, что уже выполнялось.
# Связано с: `st.session_state.sql_history` (хранилище) и `prompts.build_*_messages()` (куда этот текст подставляется).
def _get_sql_history_text():
    sql_history = st.session_state.get("sql_history", [])
    if not sql_history:
        return ""
    parts = ["ИСТОРИЯ SQL ЗАПРОСОВ (от старых к новым):"]
    for index, sql_text in enumerate(sql_history, start=1):
        sql_clean = (sql_text or "").strip()
        if sql_clean:
            parts.append(f"[{index}]\n{sql_clean}")
    return "\n\n---\n\n".join(parts).strip()


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


#-------------------------
# Назначение: получить ClickHouse-клиент (подключение) и переиспользовать его между запросами.
# Зачем: создание клиента на каждый запрос медленнее и шумнее, а нам нужен минималистичный стабильный поток.
# Связано с: `_run_sql_with_autofix()` — выполняет запросы через `ClickHouse_client.query_run()` и грузит схему.
@st.cache_resource
def _get_clickhouse_client():
    return ClickHouse_client()


#-------------------------
# Назначение: сформировать текст схемы ClickHouse для промпта.
# Зачем: в режиме SQL мы всегда передаём схему, чтобы GPT не выдумывал таблицы/колонки.
# Связано с: `ClickHouse_client.get_schema()` (источник правды) и `prompts.build_sql_messages()` (куда схема вставляется).
def _get_schema_text():
    clickhouse_client = _get_clickhouse_client()
    schema = clickhouse_client.get_schema(CLICKHOUSE_DB)
    lines = [f"СХЕМА CLICKHOUSE (database = `{CLICKHOUSE_DB}`):"]
    for table_name in sorted(schema.keys()):
        cols = schema.get(table_name) or []
        cols_text = ", ".join([f"`{col_name}` {col_type}" for col_name, col_type in cols])
        lines.append(f"- `{CLICKHOUSE_DB}.{table_name}`: {cols_text}")
    return "\n".join(lines).strip()


#-------------------------
# Назначение: вытащить чистый SQL из ответа модели (с ```sql``` или без).
# Зачем: модель иногда присылает SQL в код-блоке, а нам нужен чистый текст для выполнения.
# Связано с: `_generate_sql()` и `_fix_sql()` — оба получают текст от GPT и затем запускают SQL в ClickHouse.
def _extract_sql_text(model_text):
    text = (model_text or "").strip()
    if not text:
        return ""
    if "```" not in text:
        return text
    start_marker = "```sql"
    start_index = text.lower().find(start_marker)
    if start_index == -1:
        return text
    start_index = text.find("\n", start_index)
    if start_index == -1:
        return text
    end_index = text.find("```", start_index + 1)
    if end_index == -1:
        return text[start_index + 1 :].strip()
    return text[start_index + 1 : end_index].strip()


#-------------------------
# Назначение: сгенерировать SQL по вопросу пользователя (режим SQL).
# Зачем: пользователь пишет обычный вопрос, а мы получаем SQL для выполнения в ClickHouse.
# Связано с: `prompts.build_sql_messages()` (формирует messages) и `_run_sql_with_autofix()` (выполнение и автопочинка).
def _generate_sql(question, schema_text, history, sql_history_text):
    client = _get_openai_client()
    messages = prompts.build_sql_messages(
        question=question,
        schema_text=schema_text,
        history=history,
        sql_history_text=sql_history_text,
    )
    response = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0)
    return _extract_sql_text(response.choices[0].message.content)


#-------------------------
# Назначение: попросить GPT починить SQL после ошибки ClickHouse (одна попытка).
# Зачем: автопочинка снимает типичные проблемы (опечатки колонок, таблиц, алиасы) без ручной правки.
# Связано с: `prompts.build_sql_fix_messages()` и `_run_sql_with_autofix()` (повторный запуск исправленного SQL).
def _fix_sql(question, schema_text, history, sql_history_text, sql_text, error_text):
    client = _get_openai_client()
    messages = prompts.build_sql_fix_messages(
        question=question,
        schema_text=schema_text,
        history=history,
        sql_history_text=sql_history_text,
        sql_text=sql_text,
        error_text=error_text,
    )
    response = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0)
    return _extract_sql_text(response.choices[0].message.content)


#-------------------------
# Назначение: выполнить SQL с одной встроенной автопочинкой при ошибке.
# Зачем: пользователь получает результат без ручных правок, а схема всегда подгружается автоматически.
# Связано с: `_generate_sql()` (первый SQL), `_fix_sql()` (починка) и `ClickHouse_client.query_run()` (выполнение).
def _run_sql_with_autofix(question):
    history = _get_chat_history_for_gpt()
    sql_history_text = _get_sql_history_text()
    schema_text = _get_schema_text()

    sql_text = _generate_sql(question, schema_text, history, sql_history_text)
    if not sql_text:
        raise RuntimeError("GPT не вернул SQL.")

    clickhouse_client = _get_clickhouse_client()
    try:
        df = clickhouse_client.query_run(sql_text)
        return df, sql_text
    except Exception as error:
        fixed_sql = _fix_sql(
            question=question,
            schema_text=schema_text,
            history=history,
            sql_history_text=sql_history_text,
            sql_text=sql_text,
            error_text=str(error),
        )
        if not fixed_sql:
            raise
        df = clickhouse_client.query_run(fixed_sql)
        return df, fixed_sql


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
    sql_history_text = _get_sql_history_text()
    messages = prompts.build_rag_messages(context=context_text, history=history, sql_history_text=sql_history_text)
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

#-------------------------
# Назначение: минимальный селектор режима (только RAG и SQL).
# Зачем: пользователь явно выбирает, нужно ли ему общение по базе знаний/диалогу или выполнение SQL.
# Связано с: `_answer_with_rag()` (RAG) и `_run_sql_with_autofix()` (SQL).
with st.sidebar:
    mode = st.radio("Режим", ["RAG", "SQL"])
    with st.expander("История SQL", expanded=False):
        sql_history = st.session_state.get("sql_history", [])
        if not sql_history:
            st.caption("Пока пусто.")
        for index, sql_text in enumerate(sql_history, start=1):
            sql_clean = (sql_text or "").strip()
            if sql_clean:
                st.code(sql_clean, language="sql")

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

#-------------------------
# Назначение: отдельная область истории для SQL-запросов (только текст SQL).
# Зачем: SQL нужен и для UI (показать пользователю), и для контекста (чтобы GPT видел, что уже выполнялось).
# Связано с: `_get_sql_history_text()` и сайдбаром "История SQL".
if "sql_history" not in st.session_state:
    st.session_state.sql_history = []

# Отрисовка всей истории сообщений (user/assistant) на экране.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Если это ответ в режиме SQL — показываем таблицу и вкладку с SQL.
        if message.get("role") == "assistant" and message.get("sql_query"):
            tabs = st.tabs(["Ответ", "SQL"])
            with tabs[0]:
                if message.get("content"):
                    st.markdown(message["content"])
                df = message.get("df")
                if df is not None:
                    st.dataframe(df, use_container_width=True)
            with tabs[1]:
                st.code(message.get("sql_query"), language="sql")
        else:
            st.markdown(message["content"])

# Поле ввода пользователя (чат).
question = st.chat_input("Ваш вопрос")
if question:
    # 1) Сохраняем вопрос в историю.
    st.session_state.messages.append({"role": "user", "content": question})

    # 2) Сразу показываем вопрос в чате.
    with st.chat_message("user"):
        st.markdown(question)

    # 3) Получаем ответ в зависимости от выбранного режима.
    if mode == "RAG":
        answer = _answer_with_rag(question)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        # Режим SQL: генерируем SQL, выполняем, показываем таблицу и сохраняем SQL отдельно.
        try:
            df, used_sql = _run_sql_with_autofix(question)
        except Exception as error:
            error_text = f"Ошибка SQL: {error}"
            st.session_state.messages.append({"role": "assistant", "content": error_text})
            with st.chat_message("assistant"):
                st.markdown(error_text)
        else:
            st.session_state.sql_history.append(used_sql)
            answer_text = "Готово. Выполнил SQL и показал результат."
            st.session_state.messages.append(
                {"role": "assistant", "content": answer_text, "sql_query": used_sql, "df": df}
            )
            with st.chat_message("assistant"):
                tabs = st.tabs(["Ответ", "SQL"])
                with tabs[0]:
                    st.markdown(answer_text)
                    st.dataframe(df, use_container_width=True)
                with tabs[1]:
                    st.code(used_sql, language="sql")
