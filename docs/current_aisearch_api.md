# Бенчмарки под текущий aisearch API

## Общий контракт API

Оба бенчмарка работают с async-контрактом:
- `POST /hybrid-search/search` — постановка задачи в очередь.
- `GET /hybrid-search/info/{task_id}` — опрос статуса и получение финального ответа.

Поля request body (`SearchRequest`):
- `query`
- `top_k`
- `search_use_cache`
- `metrics_enable`
- `show_intermediate_results`
- `presearch` (опционально)
- `filters` (опционально)

## Metrics benchmark

- env-файл: `.env_metrics`
- example: `.env_metrics.example`
- запуск: `python -m metrics.metrics_bench`

Основные переменные:
- `API_BASE_URL`
- `API_TOP_K` / `TOP_K`
- `API_SEARCH_USE_CACHE`
- `API_METRICS_ENABLE`
- `API_SHOW_INTERMEDIATE_RESULTS`
- `SOURCE_FILE`
- `SOURCE_FILE_QUERY_FIELD`
- `TEST_QUERY`
- `TEST_QUERY_RETRY_COUNT`
- `RESULTS_JSON_PATH`

## Quality benchmark

- env-файл: `.env_quality`
- example: `.env_quality.example`
- запуск: `python -m quality.quality_bench`

Основные переменные:
- `TEST_DATA_PATH`
- `COL_QUERY`
- `COL_TARGET_ID`
- `COL_SOURCE`
- `RESULTS_PATH`
- `METRICS_PATH`
- `INTERMEDIATE_RESULTS_PATH`
- `RESULT_ID_FIELD`
- `MARGIN_SCORE_FIELD`
- `METRICS_KS`
- `OUTPUT_BASENAME`

## Runtime request options

Эти переменные есть в обоих env-файлах:
- `API_SEARCH_USE_CACHE`
- `API_METRICS_ENABLE`
- `API_SHOW_INTERMEDIATE_RESULTS`
- `API_PRESEARCH_ENABLED`
- `API_PRESEARCH_FIELD`
- `API_PRESEARCH_JSON`
- `API_FILTER_ARRAY_ROLE`
- `API_FILTER_ARRAY_PRODUCT`
- `API_FILTER_ARRAY_COMPONENT`
- `API_FILTER_EXACT_SOURCE`
- `API_FILTER_EXACT_ACTUAL`
- `API_FILTER_EXACT_SECOND_LINE`
- `API_FILTERS_JSON`

Приоритет и правила сборки payload:
1. `API_PRESEARCH_JSON` имеет приоритет над `API_PRESEARCH_ENABLED` / `API_PRESEARCH_FIELD`.
2. `API_FILTERS_JSON` имеет приоритет над `API_FILTER_ARRAY_*` и `API_FILTER_EXACT_*`.
3. Если `presearch` пустой, поле `presearch` не добавляется в payload.
4. Если `filters` пустой, поле `filters` не добавляется в payload.
