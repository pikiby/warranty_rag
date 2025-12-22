# clickhouse_client.py
#
# Назначение: минимальный клиент ClickHouse для выполнения SQL и загрузки схемы (system.columns).
# Связано с: `app.py` (режим SQL) — генерирует SQL через GPT, выполняет запрос через `query_run()`,
#           и всегда подгружает схему через `get_schema()` для подсказки модели и автопочинки.

import os

import clickhouse_connect
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

ClickHouse_host = os.getenv("ClickHouse_host")
ClickHouse_port = os.getenv("ClickHouse_port")
ClickHouse_username = os.getenv("ClickHouse_username")
ClickHouse_password = os.getenv("ClickHouse_password")
CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB", "db1")


class ClickHouse_client:
    # Назначение: создать подключение к ClickHouse один раз и переиспользовать его для запросов.
    # Связано с: `query_run()` (выполнение SQL) и `get_schema()` (загрузка схемы).
    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=ClickHouse_host,
            port=ClickHouse_port,
            username=ClickHouse_username,
            password=ClickHouse_password,
            secure=True,
            verify=False,
            database=CLICKHOUSE_DB,
        )

    #-------------------------
    # Назначение: выполнить SQL и вернуть результат как pandas.DataFrame.
    # Зачем: Streamlit умеет показывать pandas напрямую через `st.dataframe(...)`.
    # Связано с: `app.py::_run_sql_flow()` — выполняет SQL, отображает результат и сохраняет SQL в историю.
    def query_run(self, query_text):
        # Базовый запуск запроса с маленькой страховкой от `db1.db1.` при UNKNOWN_TABLE (Code: 60).
        try:
            result = self.client.query(query_text)
        except Exception as error:
            error_text = str(error)
            is_unknown_table = ("Code: 60" in error_text) or ("UNKNOWN_TABLE" in error_text)
            if not is_unknown_table:
                raise

            #-------------------------
            # Назначение: локально исправить частую ошибку модели: двойной префикс БД (`db.db.table`).
            # Зачем: ClickHouse явно подсказывает "Maybe you meant db.table", поэтому не гоняем лишний запрос к GPT.
            # Связано с: `app.py::_run_sql_with_autofix()` — если починка тут сработала, пользователь сразу получает результат.
            db = (CLICKHOUSE_DB or "").strip()
            if not db:
                raise

            replacements = {
                f"{db}.{db}.": f"{db}.",
                f"`{db}`.`{db}`.": f"`{db}`.",
                f"`{db}`.{db}.": f"`{db}`.",
                f"{db}.`{db}`.": f"{db}.",
            }

            fixed_query = query_text
            for needle, repl in replacements.items():
                if needle in fixed_query:
                    fixed_query = fixed_query.replace(needle, repl)

            if fixed_query != query_text:
                result = self.client.query(fixed_query)
            else:
                raise
        return pd.DataFrame(result.result_rows, columns=result.column_names)

    #-------------------------
    # Назначение: загрузить схему таблиц из system.columns.
    # Зачем: схема всегда передаётся в GPT в режиме SQL, чтобы он не выдумывал колонки/таблицы.
    # Связано с: `app.py::_get_schema_text()` — формирует текстовую подсказку для промпта.
    def get_schema(self, database, tables=None):
        tb_filter = ""
        if tables:
            quoted = ",".join([f"'{table_name}'" for table_name in tables])
            tb_filter = f" AND table IN ({quoted})"
        sql = f"""
        SELECT table, name, type
        FROM system.columns
        WHERE database = '{database}' {tb_filter}
        ORDER BY table, position
        """
        res = self.client.query(sql)
        rows = list(res.result_rows)
        schema = {}
        for table_name, column_name, column_type in rows:
            schema.setdefault(table_name, []).append((column_name, column_type))
        return schema
