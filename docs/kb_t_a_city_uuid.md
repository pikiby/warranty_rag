---
id: t_a_city_uuid
db: db1
short_description: "Справочник городов: соответствие `city_uuid` → `city`, `region`, `country`."
synonyms:
  - города
  - справочник городов
  - city uuid
type: table
---

# `t_a_city_uuid`

## Назначение
Агрегированный справочник городов из `entries_installation_points_dir_partner_ch`. Содержит уникальные `city_uuid` с человекочитаемыми полями города/региона/страны. Используется для обогащения витрин городом по ключу `city_uuid`.

## Поля (CH)
- `city_uuid` — ключ города
- `city` — человекочитаемое название города (алиас: `Город`)
- `region` — регион
- `country` — страна

## Пример (CH)
```sql
SELECT `city_uuid`, `city`, `region`, `country`
FROM `t_a_city_uuid`
LIMIT 10
```


