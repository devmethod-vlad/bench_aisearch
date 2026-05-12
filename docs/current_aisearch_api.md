# Бенчмарки под текущий aisearch API

В репозитории используется единый актуальный набор скриптов:

- `metrics/metrics_bench.py`
- `quality/quality_bench.py`
- `bench_common/current_api.py`

Запуск:

```bash
python -m metrics.metrics_bench
python -m quality.quality_bench
```

## SearchRequest

`POST /hybrid-search/search` отправляется с обязательными полями:
- `query`
- `top_k`
- `search_use_cache`
- `metrics_enable`
- `show_intermediate_results`

Дополнительно (если заданы):
- `presearch`
- `filters`

### Управление через env

Базовые флаги:
- `API_SEARCH_USE_CACHE` (bool)
- `API_METRICS_ENABLE` (bool)
- `API_SHOW_INTERMEDIATE_RESULTS` (bool)

Bool-значения: `true/false`, `1/0`, `yes/no`, `y/n`, `on/off`.

Presearch:
- `API_PRESEARCH_ENABLED`
- `API_PRESEARCH_FIELD`
- `API_PRESEARCH_JSON` (имеет приоритет)

Пример:
- `API_PRESEARCH_ENABLED=true`
- `API_PRESEARCH_FIELD=question`

=> `"presearch": {"field": "question"}`

Filters:
- Array filters:
  - `API_FILTER_ARRAY_ROLE`
  - `API_FILTER_ARRAY_PRODUCT`
  - `API_FILTER_ARRAY_COMPONENT`
- Exact filters:
  - `API_FILTER_EXACT_SOURCE`
  - `API_FILTER_EXACT_ACTUAL`
  - `API_FILTER_EXACT_SECOND_LINE`
- `API_FILTERS_JSON` (имеет приоритет)

Списки для array filters задаются через `,` или `;`.

Если `presearch`/`filters` пустые, они не добавляются в payload.

## Дефолты по сценариям

- `metrics.metrics_bench`:
  - `API_METRICS_ENABLE=true`
  - `API_SHOW_INTERMEDIATE_RESULTS=false`
  - `API_SEARCH_USE_CACHE=false`
- `quality.quality_bench`:
  - `API_METRICS_ENABLE=true`
  - `API_SHOW_INTERMEDIATE_RESULTS=true`
  - `API_SEARCH_USE_CACHE=false`

Любая явно заданная env-переменная переопределяет дефолт скрипта.

## Пути в TaskResponse

По умолчанию используются dotted-path:
- `RESULTS_PATH=info.results`
- `METRICS_PATH=info.metrics`
- `INTERMEDIATE_RESULTS_PATH=info.intermediate_results`

Сохраняется поддержка fallback score-полей:
`final_score`, `score_final`, `score_ce`, `score_fusion`, `score_dense`, `score_lex`.
