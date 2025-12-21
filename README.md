## ai_bi_min

Минималистичный Streamlit-чат с простым RAG:

- **Индексация**: `ingest.py` читает `docs/**/*.md` и пишет в `data/chroma`.
- **Чат**: `app.py` делает retrieve контекста из Chroma и отвечает строго по нему.

### Запуск локально

1) Установить зависимости:

```bash
pip install -r requirements.txt
```

2) Задать ключ:

```bash
export OPENAI_API_KEY="..."
```

3) (Опционально) Проиндексировать `docs/`:

```bash
python ingest.py
```

4) Запустить чат:

```bash
streamlit run app.py
```# waranty_rag
