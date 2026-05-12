# -*- coding: utf-8 -*-
"""Метрики скорости для текущего API devmethod-vlad/aisearch.

Отличия от старого metrics_bench.py:
- явно передаёт metrics_enable/search_use_cache/show_intermediate_results;
- понимает новый TaskResponse: status + info.results/info.metrics;
- сохраняет полный payload, чтобы можно было разбирать изменения API без
  повторного прогона бенчмарка.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import statistics
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from bench_common.current_api import (
    BenchmarkApiError,
    SearchApiConfig,
    coalesce_float,
    get_by_dotted_path,
    run_search,
)


def load_queries() -> list[str]:
    source_file = os.getenv("SOURCE_FILE")
    source_field = os.getenv("SOURCE_FILE_QUERY_FIELD")
    test_query = os.getenv("TEST_QUERY", "тестовый запрос")
    test_count = int(os.getenv("TEST_QUERY_RETRY_COUNT", "20"))

    if not source_file:
        return [test_query] * test_count

    path = Path(source_file)
    if not path.exists():
        raise BenchmarkApiError(f"SOURCE_FILE не найден: {path}")
    if not source_field:
        raise BenchmarkApiError("Для SOURCE_FILE нужно задать SOURCE_FILE_QUERY_FIELD")

    if path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise BenchmarkApiError("SOURCE_FILE поддерживает только Excel или Parquet")

    if source_field not in df.columns:
        raise BenchmarkApiError(f"В {path} нет колонки {source_field!r}; есть {list(df.columns)}")

    queries = [str(v).strip() for v in df[source_field].dropna().tolist() if str(v).strip()]
    if not queries:
        raise BenchmarkApiError(f"В колонке {source_field!r} нет непустых запросов")
    return queries


def avg(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return None if not clean else statistics.fmean(clean)


def pctl(values: list[float | None], percentile: float) -> float | None:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    k = (len(clean) - 1) * percentile / 100.0
    lo = int(k)
    hi = min(lo + 1, len(clean) - 1)
    return clean[lo] + (clean[hi] - clean[lo]) * (k - lo)


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def main() -> None:
    load_dotenv(Path(os.getenv("DOTENV_PATH", ".env_metrics")))
    console = Console()
    cfg = SearchApiConfig.from_env(default_top_k=10, default_intermediate=False)
    queries = load_queries()

    metrics_path = os.getenv("API_METRICS_PATH", "info.metrics")
    output_path = os.getenv("RESULTS_JSON_PATH") or f"metrics_results_{dt.datetime.now():%Y%m%d_%H%M%S}.json"

    rows: list[dict[str, Any]] = []
    for idx, query in enumerate(tqdm(queries, desc="metrics", unit="q"), start=1):
        final_response, meta = run_search(query, cfg)
        metrics = get_by_dotted_path(final_response, metrics_path, {}) or {}
        if not isinstance(metrics, dict):
            metrics = {}

        row = {
            "index": idx,
            "query": query,
            "query_length": len(query),
            "task_id": meta["task_id"],
            "post_time": meta["post_time"],
            "poll_time": meta["poll_time"],
            "response_time": meta["response_time"],
            "polls": meta["polls"],
            "statuses": meta["statuses"],
            "embedding_time": coalesce_float(metrics, "embedding_time"),
            "vector_search_time": coalesce_float(metrics, "vector_search_time"),
            "lexical_search_time": coalesce_float(metrics, "lexical_search_time", "opensearch_time"),
            "cross_encoder_time": coalesce_float(metrics, "cross_encoder_time"),
            "total_time": coalesce_float(metrics, "total_time", "total_search_time"),
            "full_search_task_time_ms": coalesce_float(metrics, "full_search_task_time"),
            "from_cache": metrics.get("from_cache"),
            "reranker_enabled": metrics.get("reranker_enabled"),
            "open_search_enabled": metrics.get("open_search_enabled"),
            "short_mode_applied": metrics.get("short_mode_applied"),
            "hybrid_fusion_mode": metrics.get("hybrid_fusion_mode"),
            "payload": meta.get("payload"),
            "api_response": final_response,
        }
        rows.append(row)

    summary = {
        "requests_count": len(rows),
        "api": {
            "base_url": cfg.base_url,
            "search_path": cfg.search_path,
            "info_path": cfg.info_path,
            "top_k": cfg.top_k,
            "search_use_cache": cfg.search_use_cache,
            "metrics_enable": cfg.metrics_enable,
            "show_intermediate_results": cfg.show_intermediate_results,
        },
        "avg": {
            "response_time": avg([r["response_time"] for r in rows]),
            "post_time": avg([r["post_time"] for r in rows]),
            "poll_time": avg([r["poll_time"] for r in rows]),
            "embedding_time": avg([r["embedding_time"] for r in rows]),
            "vector_search_time": avg([r["vector_search_time"] for r in rows]),
            "lexical_search_time": avg([r["lexical_search_time"] for r in rows]),
            "cross_encoder_time": avg([r["cross_encoder_time"] for r in rows]),
            "total_time": avg([r["total_time"] for r in rows]),
        },
        "p95": {
            "response_time": pctl([r["response_time"] for r in rows], 95),
            "total_time": pctl([r["total_time"] for r in rows], 95),
            "cross_encoder_time": pctl([r["cross_encoder_time"] for r in rows], 95),
        },
        "rows": rows,
    }

    Path(output_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    table = Table(title="AISearch metrics benchmark")
    table.add_column("metric")
    table.add_column("avg", justify="right")
    table.add_column("p95", justify="right")
    for name in ["response_time", "total_time", "embedding_time", "vector_search_time", "lexical_search_time", "cross_encoder_time"]:
        table.add_row(name, fmt(summary["avg"].get(name)), fmt(summary["p95"].get(name)))
    console.print(table)
    console.print(f"[green]JSON сохранён:[/green] {output_path}")


if __name__ == "__main__":
    main()
