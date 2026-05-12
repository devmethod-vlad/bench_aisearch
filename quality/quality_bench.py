# -*- coding: utf-8 -*-
"""Бенчмарк качества для текущего API devmethod-vlad/aisearch.

Сохраняет общую идею старого quality_bench.py:
- Excel с test_query / target_id / source;
- async API POST /hybrid-search/search -> GET /hybrid-search/info/{task_id};
- recall@k, MRR@k, nDCG@k, not_found;
- JSON + XLSX отчёт.

Но учитывает новый контракт aisearch:
- metrics_enable и show_intermediate_results задаются в теле POST;
- итоговые результаты лежат в info.results;
- intermediate_results появляются только если show_intermediate_results=true;
- итоговый score может называться score_ce/score_fusion, а не final_score.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from bench_common.current_api import (
    BenchmarkApiError,
    SearchApiConfig,
    get_by_dotted_path,
    run_search,
    score_of,
)


def parse_int_list(raw: str, default: list[int]) -> list[int]:
    try:
        values = [int(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]
        return values or default
    except ValueError:
        return default


def load_test_data() -> tuple[pd.DataFrame, dict[str, str]]:
    path_raw = os.getenv("TEST_DATA_PATH")
    if not path_raw:
        raise BenchmarkApiError("Не задан TEST_DATA_PATH")
    path = Path(path_raw)
    if not path.exists():
        raise BenchmarkApiError(f"TEST_DATA_PATH не найден: {path}")

    df = pd.read_excel(path)
    cols = {
        "query": os.getenv("COL_QUERY", "test_query"),
        "target": os.getenv("COL_TARGET_ID", "target_id"),
        "source": os.getenv("COL_SOURCE", "source"),
        "query_source": os.getenv("COL_QUERY_SOURCE", "test_query_source"),
        "answer": os.getenv("COL_ANSWER", "test_answer"),
    }
    missing = [cols[k] for k in ("query", "target", "source") if cols[k] not in df.columns]
    if missing:
        raise BenchmarkApiError(f"В TEST_DATA_PATH отсутствуют обязательные колонки: {missing}; есть {list(df.columns)}")
    return df, cols


def rank_metrics(ranks: list[int | None], ks: list[int]) -> dict[str, dict[str, float]]:
    n = len(ranks) or 1
    recall: dict[str, float] = {}
    mrr: dict[str, float] = {}
    ndcg: dict[str, float] = {}
    for k in ks:
        hits = 0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        for rank in ranks:
            if rank is not None and rank <= k:
                hits += 1
                mrr_sum += 1.0 / rank
                ndcg_sum += 1.0 / math.log2(rank + 1.0)
        recall[str(k)] = round(hits * 100.0 / n, 2)
        mrr[str(k)] = round(mrr_sum / n, 4)
        ndcg[str(k)] = round(ndcg_sum / n, 4)
    return {"recall_at_k": recall, "mrr_at_k": mrr, "ndcg_at_k": ndcg}


def find_rank(results: list[dict[str, Any]], target_id: str, result_id_field: str) -> int | None:
    target = str(target_id)
    for index, item in enumerate(results, start=1):
        if str(item.get(result_id_field, "")) == target:
            return index
    return None


def margin_at_1(results: list[dict[str, Any]], score_field: str | None) -> float | None:
    if len(results) < 2:
        return None
    return score_of(results[0], score_field) - score_of(results[1], score_field)


def safe_cell(row: pd.Series, column: str, default: str = "") -> str:
    return str(row[column]) if column in row and pd.notna(row[column]) else default


def main() -> None:
    load_dotenv(Path(os.getenv("DOTENV_PATH", ".env_quality")))
    df, cols = load_test_data()

    cfg = SearchApiConfig.from_env(default_top_k=int(os.getenv("TOP_K", "15")), default_intermediate=True)
    results_path = os.getenv("RESULTS_PATH", "info.results")
    metrics_path = os.getenv("METRICS_PATH", "info.metrics")
    intermediate_path = os.getenv("INTERMEDIATE_RESULTS_PATH", "info.intermediate_results")
    result_id_field = os.getenv("RESULT_ID_FIELD", "ext_id")
    score_field = os.getenv("MARGIN_SCORE_FIELD") or None
    ks = parse_int_list(os.getenv("METRICS_KS", "1,3,5,10"), [1, 3, 5, 10])
    output_base = os.getenv("OUTPUT_BASENAME", "bench_results")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = f"{output_base}_{stamp}.json"
    output_xlsx = f"{output_base}_{stamp}.xlsx"

    rows: list[dict[str, Any]] = []
    ranks: list[int | None] = []
    margins: list[float | None] = []
    by_source: dict[str, list[int | None]] = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="quality", unit="q"):
        query = str(row[cols["query"]])
        target_id = str(row[cols["target"]])
        source = str(row[cols["source"]]) if pd.notna(row[cols["source"]]) else ""

        final_response, meta = run_search(query, cfg)
        results = get_by_dotted_path(final_response, results_path, []) or []
        metrics = get_by_dotted_path(final_response, metrics_path, {}) or {}
        intermediate = get_by_dotted_path(final_response, intermediate_path, {}) or {}
        if not isinstance(results, list):
            results = []

        rank = find_rank(results, target_id, result_id_field)
        margin = margin_at_1(results, score_field)
        ranks.append(rank)
        margins.append(margin)
        by_source.setdefault(source, []).append(rank)

        top_results = []
        for pos, item in enumerate(results[: int(os.getenv("RESULTS_LIMIT", str(cfg.top_k)))], start=1):
            top_results.append(
                {
                    "rank": pos,
                    "id": item.get(result_id_field),
                    "score": score_of(item, score_field),
                    "question": item.get(os.getenv("RESULT_QUESTION_FIELD", "question")),
                    "answer": item.get(os.getenv("RESULT_ANSWER_FIELD", "answer")),
                    "source": item.get("source"),
                }
            )

        rows.append(
            {
                "index": int(idx) + 1,
                "query": query,
                "target_id": target_id,
                "source": source,
                "rank": rank,
                "found": rank is not None,
                "margin_at_1": margin,
                "response_time": meta["response_time"],
                "task_id": meta["task_id"],
                "test_query_source": safe_cell(row, cols["query_source"]),
                "test_answer": safe_cell(row, cols["answer"]),
                "top_results": top_results,
                "metrics": metrics,
                "intermediate_present": bool(intermediate),
            }
        )

    overall = rank_metrics(ranks, ks)
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
            "result_id_field": result_id_field,
            "score_field_preferred": score_field,
        },
        "overall": {
            **overall,
            "not_found": sum(1 for rank in ranks if rank is None),
            "found": sum(1 for rank in ranks if rank is not None),
            "avg_margin_at_1": None if not [m for m in margins if m is not None] else sum(m for m in margins if m is not None) / len([m for m in margins if m is not None]),
        },
        "by_source": {src: rank_metrics(src_ranks, ks) | {"count": len(src_ranks)} for src, src_ranks in by_source.items()},
        "rows": rows,
    }

    Path(output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_rows = []
    for item in rows:
        flat_rows.append(
            {
                "index": item["index"],
                "source": item["source"],
                "query": item["query"],
                "target_id": item["target_id"],
                "rank": item["rank"],
                "found": item["found"],
                "margin_at_1": item["margin_at_1"],
                "response_time": item["response_time"],
                "task_id": item["task_id"],
            }
        )
    with pd.ExcelWriter(output_xlsx) as writer:
        pd.DataFrame(flat_rows).to_excel(writer, sheet_name="queries", index=False)
        pd.DataFrame([summary["overall"]["recall_at_k"]]).to_excel(writer, sheet_name="recall", index=False)
        pd.DataFrame([summary["overall"]["mrr_at_k"]]).to_excel(writer, sheet_name="mrr", index=False)
        pd.DataFrame([summary["overall"]["ndcg_at_k"]]).to_excel(writer, sheet_name="ndcg", index=False)

    print(json.dumps({"output_json": output_json, "output_xlsx": output_xlsx, "overall": summary["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
