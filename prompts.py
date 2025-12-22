"""
prompts.py — правила общения для GPT в этом приложении.

Зачем нужен: хранит минимальные системные промпты и функции сборки `messages` для двух режимов:
- RAG: ответы по базе знаний или по истории диалога.
- SQL: генерация/починка SQL под ClickHouse с обязательной схемой.

Связано с:
- `app.py` — вызывает `build_messages(...)` перед запросом к GPT.
- `retriever.py` — поставляет контекст (чанки), который сюда передаётся строкой.
"""

RAG_SYSTEM_PROMPT = """
Ты — ассистент по базе знаний.

Правила:
- Если вопрос про базу знаний — отвечай ТОЛЬКО по предоставленному контексту.
- Если вопрос про сам диалог (например: "что я спрашивал", "что ты ответил") — используй историю сообщений в чате.
- Если вопрос НЕ про диалог и в контексте нет ответа, скажи: "В индексе базы знаний нет данных по запросу (или индекс не создан)."
- Не выдумывай факты, таблицы, поля, цифры, даты.
""".strip()

SQL_SYSTEM_PROMPT = """
Ты — ассистент, который пишет SQL для ClickHouse.

Правила:
- Возвращай ТОЛЬКО SQL (без пояснений, без markdown). Допускается один блок ```sql``` если очень нужно.
- Используй ТОЛЬКО таблицы и колонки из предоставленной схемы. Ничего не выдумывай.
""".strip()


#-------------------------
# Назначение: собрать сообщения для RAG (system + KB-контекст + история диалога + история SQL).
# Зачем: GPT должен видеть и контекст из базы знаний, и прошлые реплики, и ранее сформированные SQL (как память).
# Связано с: `app.py::_answer_with_rag()` — передаёт эти messages в `client.chat.completions.create(...)`.
def build_rag_messages(context, history, sql_history_text):
    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
    if sql_history_text:
        messages.append({"role": "system", "content": sql_history_text})
    messages.append({"role": "system", "content": f"КОНТЕКСТ БАЗЫ ЗНАНИЙ:\n{context}".strip()})
    messages.extend(history)
    return messages


#-------------------------
# Назначение: собрать сообщения для генерации SQL (system + схема + история диалога + история SQL + вопрос).
# Зачем: модель должна генерировать SQL строго по схеме и с учётом того, о чём уже шёл разговор.
# Связано с: `app.py::_generate_sql()` — запрашивает у GPT SQL, который потом выполняется в ClickHouse.
def build_sql_messages(question, schema_text, history, sql_history_text):
    messages = [{"role": "system", "content": SQL_SYSTEM_PROMPT}]
    if sql_history_text:
        messages.append({"role": "system", "content": sql_history_text})
    messages.append({"role": "system", "content": schema_text})
    messages.extend(history)
    messages.append({"role": "user", "content": question.strip()})
    return messages


#-------------------------
# Назначение: собрать сообщения для автопочинки SQL после ошибки выполнения.
# Зачем: на вход даём ошибку ClickHouse + исходный SQL + схему, чтобы GPT вернул исправленный SQL.
# Связано с: `app.py::_run_sql_with_autofix()` — делает повторную попытку выполнения.
def build_sql_fix_messages(question, schema_text, history, sql_history_text, sql_text, error_text):
    messages = [{"role": "system", "content": SQL_SYSTEM_PROMPT}]
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
    return messages
