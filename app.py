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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB", "db1")

#-------------------------
# Назначение: минимально нормализовать пользовательский текст перед тем, как его увидит GPT и ретривер.
# Зачем: исправляем частые опечатки и "склейки" слов, которые ломают маршрутизацию и RAG-поиск.
# Связано с: `_get_chat_history_for_gpt()` (нормализует user-реплики), `_answer_with_rag()` (поиск),
#           `_run_sql_with_autofix()` (генерация SQL) и `_select_mode()` (авто-выбор режима).
def _normalize_user_text(text):
    text = (text or "").strip()
    if not text:
        return ""

    normalized = text
    normalized = normalized.replace("погороду", "по городу")
    normalized = normalized.replace("подате", "по дате")
    normalized = normalized.replace("видешь", "видишь")
    normalized = normalized.replace("видет", "видит")

    normalized = " ".join(normalized.split())
    return normalized


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
# Связано с: `retriever.retrieve()` (находит куски) и `_answer_with_rag()` (кладёт контекст в messages для GPT).
def _build_context_text(hits):
    parts = []
    for hit in hits:
        chunk_text = (hit.get("text") or "").strip()
        if chunk_text:
            parts.append(chunk_text)
    return "\n\n".join(parts).strip()


# Назначение: собрать историю SQL-запросов в одном тексте для передачи в GPT.
# Зачем: пользователь может уточнять и продолжать работу, а модель должна видеть, что уже выполнялось.
# Связано с: `st.session_state.sql_history` (хранилище) и `_answer_with_rag()`/`_generate_sql()`/`_fix_sql()` (куда этот текст подставляется).
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
# Связано с: `st.session_state.messages` (хранилище UI-истории) и `_answer_with_rag()`/`_generate_sql()`/`_fix_sql()` (messages для GPT).
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
        if role == "user":
            content = _normalize_user_text(content)
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
# Связано с: `ClickHouse_client.get_schema()` (источник правды) и `_generate_sql()`/`_fix_sql()` (куда схема вставляется).
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
# Назначение: автоматически выбрать режим (RAG или SQL), как в `ai_bi`, но максимально просто.
# Зачем: пользователь не должен вручную переключать режим — мы сами решаем, это вопрос к базе знаний/диалогу или запрос к данным.
# Связано с: `prompts.ROUTER_PROMPT` (правила выбора) и основным обработчиком внизу файла (ветка RAG/SQL).
def _select_mode():
    client = _get_openai_client()
    history = _get_chat_history_for_gpt()

    messages = [{"role": "system", "content": prompts.ROUTER_PROMPT}]
    messages.extend(history)

    response = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0)
    mode_text = (response.choices[0].message.content or "").strip().lower()
    if "```mode" in mode_text and "sql" in mode_text:
        return "SQL"
    return "RAG"


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
# Назначение: переформулировать вопрос в короткий поисковый запрос для базы знаний (RAG).
# Зачем: векторный поиск иногда промахивается по общим формулировкам; перефраз помогает попасть в нужный документ.
# Связано с: `_answer_with_rag()` — если первый retrieval пустой, делаем rewrite и повторяем поиск.
def _rewrite_query_for_kb(question):
    question = _normalize_user_text(question)
    if not question:
        return ""

    client = _get_openai_client()
    messages = [
        {"role": "system", "content": prompts.KB_QUERY_REWRITE_PROMPT},
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0)
    rewritten = (response.choices[0].message.content or "").strip()
    rewritten = _normalize_user_text(rewritten)
    if not rewritten:
        return ""
    if rewritten == question:
        return ""
    return rewritten


#-------------------------
# Назначение: сгенерировать SQL по вопросу пользователя (режим SQL).
# Зачем: пользователь пишет обычный вопрос, а мы получаем SQL для выполнения в ClickHouse.
# Связано с: `prompts.SQL_SYSTEM_PROMPT` (правила для SQL) и `_run_sql_with_autofix()` (выполнение и автопочинка).
def _generate_sql(question, schema_text, history, sql_history_text):
    client = _get_openai_client()
    messages = [{"role": "system", "content": prompts.SQL_SYSTEM_PROMPT}]
    if sql_history_text:
        messages.append({"role": "system", "content": sql_history_text})
    messages.append({"role": "system", "content": schema_text})
    messages.extend(history)
    messages.append({"role": "user", "content": question.strip()})
    response = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0)
    return _extract_sql_text(response.choices[0].message.content)


#-------------------------
# Назначение: попросить GPT починить SQL после ошибки ClickHouse (одна попытка).
# Зачем: автопочинка снимает типичные проблемы (опечатки колонок, таблиц, алиасы) без ручной правки.
# Связано с: `prompts.SQL_SYSTEM_PROMPT` (правила для SQL) и `_run_sql_with_autofix()` (повторный запуск исправленного SQL).
def _fix_sql(question, schema_text, history, sql_history_text, sql_text, error_text):
    client = _get_openai_client()
    messages = [{"role": "system", "content": prompts.SQL_SYSTEM_PROMPT}]
    if sql_history_text:
        messages.append({"role": "system", "content": sql_history_text})
    messages.append({"role": "system", "content": schema_text})
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": (
                "Почини SQL после ошибки ClickHouse.\n\n"
                f"Вопрос пользователя:\n{question.strip()}\n\n"
                f"Ошибка ClickHouse:\n{error_text.strip()}\n\n"
                f"Исходный SQL:\n{sql_text.strip()}\n\n"
                "Верни исправленный SQL."
            ).strip(),
        }
    )
    response = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0)
    return _extract_sql_text(response.choices[0].message.content)


#-------------------------
# Назначение: проверить, что SQL безопасен и соответствует правилам проекта.
# Зачем: даже при хорошем промпте модель иногда может вернуть `system.*` или DDL/изменения — это нужно жёстко запретить.
# Связано с: `_run_sql_with_autofix()` — проверяет SQL перед выполнением (и до, и после автопочинки).
def _validate_sql_safety(sql_text):
    sql_clean = (sql_text or "").strip()
    if not sql_clean:
        return "Пустой SQL."

    sql_lower = sql_clean.lower()
    sql_lower_no_space = " ".join(sql_lower.split())

    if "system." in sql_lower_no_space or " system." in sql_lower_no_space:
        return "Запрещены запросы к system.*"

    allowed_start = sql_lower_no_space.startswith("select ") or sql_lower_no_space.startswith("with ")
    if not allowed_start:
        return "Разрешены только SELECT / WITH ... SELECT."

    forbidden = [
        "create ",
        "alter ",
        "drop ",
        "truncate ",
        "rename ",
        "attach ",
        "detach ",
        "insert ",
        "update ",
        "delete ",
        "optimize ",
        "grant ",
        "revoke ",
    ]
    for token in forbidden:
        if token in sql_lower_no_space:
            return "Запрещены DDL и любые изменения данных."

    return ""


#-------------------------
# Назначение: выполнить SQL с одной встроенной автопочинкой при ошибке.
# Зачем: пользователь получает результат без ручных правок, а схема всегда подгружается автоматически.
# Связано с: `_generate_sql()` (первый SQL), `_fix_sql()` (починка) и `ClickHouse_client.query_run()` (выполнение).
def _run_sql_with_autofix(question):
    history = _get_chat_history_for_gpt()
    if history and history[-1].get("role") == "user":
        history = history[:-1]
    sql_history_text = _get_sql_history_text()
    schema_text = _get_schema_text()

    question = _normalize_user_text(question)
    sql_text = _generate_sql(question, schema_text, history, sql_history_text)
    if not sql_text:
        raise RuntimeError("GPT не вернул SQL.")

    clickhouse_client = _get_clickhouse_client()
    safety_error = _validate_sql_safety(sql_text)
    if safety_error:
        raise RuntimeError(f"SQL заблокирован: {safety_error}")
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
        safety_error = _validate_sql_safety(fixed_sql)
        if safety_error:
            raise RuntimeError(f"SQL заблокирован: {safety_error}")
        df = clickhouse_client.query_run(fixed_sql)
        return df, fixed_sql


#-------------------------
# Назначение: обработать вопрос в режиме RAG и вывести ответ в чат.
# Пайплайн: вопрос -> retrieval (Chroma) -> контекст -> запрос к GPT -> ответ.
# Связано с: `_answer_with_rag()` (логика RAG) и `st.session_state.messages` (история чата для UI/контекста).
def _handle_rag_message(question):
    answer = _answer_with_rag(question)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)


#-------------------------
# Назначение: обработать вопрос в режиме SQL и вывести результат в чат.
# Пайплайн: вопрос -> схема -> GPT генерирует SQL -> ClickHouse выполняет -> (если ошибка) GPT чинит -> повтор -> таблица.
# Связано с: `_run_sql_with_autofix()` (выполнение + автопочинка), `st.session_state.sql_history` (память SQL),
#           и рендером вкладок "Ответ/SQL" в UI.
def _handle_sql_message(question):
    try:
        df, used_sql = _run_sql_with_autofix(question)
    except Exception as error:
        error_text = f"Ошибка SQL: {error}"
        st.session_state.messages.append({"role": "assistant", "content": error_text})
        with st.chat_message("assistant"):
            st.markdown(error_text)
        return

    st.session_state.sql_history.append(used_sql)
    answer_text = "Готово. Выполнил SQL и показал результат."
    st.session_state.messages.append({"role": "assistant", "content": answer_text, "sql_query": used_sql, "df": df})
    with st.chat_message("assistant"):
        tabs = st.tabs(["Ответ", "SQL"])
        with tabs[0]:
            st.markdown(answer_text)
            st.dataframe(df, use_container_width=True)
        with tabs[1]:
            st.code(used_sql, language="sql")


# Назначение: выполнить самый простой RAG-пайплайн: найти контекст и ответить строго по нему.
# Зачем: GPT отвечает только по тексту, который мы ему передали. Поэтому сначала делаем retrieval (поиск чанков в коллекции Chroma),
# потом кладём найденный текст в prompt и только после этого спрашиваем GPT.
# Связано с: `retriever.retrieve()` (ищет top-k чанков в базе знаний) и `prompts.RAG_SYSTEM_PROMPT` (правила ответа).
def _answer_with_rag(question):
    question = _normalize_user_text(question)

    hits = retriever.retrieve(
        query=question,
        k=5,
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )
    context_text = _build_context_text(hits)
    client = _get_openai_client()
    history = _get_chat_history_for_gpt()
    messages = [{"role": "system", "content": prompts.RAG_SYSTEM_PROMPT}]
    messages.append({"role": "system", "content": f"КОНТЕКСТ БАЗЫ ЗНАНИЙ:\n{context_text}".strip()})
    messages.extend(history)
    resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
    return (resp.choices[0].message.content or "").strip()


# =============================================================================
# UI (Streamlit)
# =============================================================================
# Назначение: этот блок — "экран" приложения. Он показывает историю чата, принимает новый вопрос
# и запускает RAG-ответ через `_answer_with_rag()`.
# Связано с: `_answer_with_rag()` (получает ответ), `_build_context_text()` (собирает контекст),
# `retriever.retrieve()` (ищет в базе знаний) и `prompts.RAG_SYSTEM_PROMPT`/`prompts.SQL_SYSTEM_PROMPT` (правила режима).

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

#-------------------------
# Назначение: отдельная область истории для SQL-запросов (только текст SQL).
# Зачем: SQL нужен и для UI (показать пользователю), и для контекста (чтобы GPT видел, что уже выполнялось).
# Связано с: `_get_sql_history_text()` (передаём в GPT как память) и веткой SQL (куда записываем выполненный запрос).
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

#-------------------------
# ОБРАБОТКА НОВОГО СООБЩЕНИЯ (пайплайн)
# 1) Получить ввод пользователя из `st.chat_input`.
# 2) Сохранить сообщение в `st.session_state.messages` (память для UI и контекста).
# 3) Показать сообщение пользователя в UI сразу.
# 4) Автоматически выбрать режим через GPT-роутер `_select_mode()`.
# 5) Выполнить выбранный пайплайн:
#    - RAG: `_handle_rag_message()` (retrieval -> контекст -> GPT -> ответ)
#    - SQL: `_handle_sql_message()` (схема -> GPT SQL -> ClickHouse -> автопочинка -> таблица)
#
# Важно: мы не строим “триггеры” режима в коде — выбор делает только модель (как в `ai_bi`).
#-------------------------

# Поле ввода пользователя (чат).
question = st.chat_input("Ваш вопрос")
if question:
    # 1) Сохраняем вопрос в историю.
    st.session_state.messages.append({"role": "user", "content": question})

    # 2) Сразу показываем вопрос в чате.
    with st.chat_message("user"):
        st.markdown(question)

    # 3) Автоматически выбираем режим и получаем ответ.
    mode = _select_mode()
    if mode == "RAG":
        _handle_rag_message(question)
    else:
        _handle_sql_message(question)
