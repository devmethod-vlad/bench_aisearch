# bench_aisearch

Бенчмарки для `devmethod-vlad/aisearch` (текущий API):
- `python -m metrics.metrics_bench` — замер latency и API-метрик.
- `python -m quality.quality_bench` — качество выдачи (recall@k, MRR@k, nDCG@k).

Общая работа с API вынесена в `bench_common/current_api.py`.

## Быстрый старт

1. Установите зависимости (`pandas`, `requests`, `python-dotenv`, `tqdm`, `rich`, и т.д. по `pyproject.toml`).
2. Скопируйте пример окружения:
   ```bash
   cp .env.benchmark.example .env_metrics
   cp .env.benchmark.example .env_quality
   ```
3. Настройте переменные под ваш стенд и данные.

## Запуск

```bash
python -m metrics.metrics_bench
python -m quality.quality_bench
```

Подробности по контракту API и всем env-переменным: `docs/current_aisearch_api.md`.
