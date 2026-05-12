# -*- coding: utf-8 -*-
"""HTTP-клиент для текущего async API devmethod-vlad/aisearch.

Новый aisearch после 3756002f управляет метриками и intermediate results
через тело POST /hybrid-search/search:
- metrics_enable
- show_intermediate_results
- search_use_cache
- presearch
- filters

Этот модуль инкапсулирует контракт, чтобы quality/metrics-бенчмарки не
дублировали сетевую и compatibility-логику.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests


class BenchmarkApiError(RuntimeError):
    """Понятная ошибка бенчмарка с фрагментом ответа API."""


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw in (None, "") else int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw in (None, "") else float(raw)


def env_json(name: str, default: Any = None) -> Any:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BenchmarkApiError(f"{name} должен быть валидным JSON: {raw!r}") from exc


def get_by_dotted_path(data: dict[str, Any], path: str | None, default: Any = None) -> Any:
    """Мягко достаёт значение по пути вида info.results."""
    if not path:
        return default
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def coalesce_float(data: dict[str, Any] | None, *fields: str) -> float | None:
    if not isinstance(data, dict):
        return None
    for field in fields:
        value = data.get(field)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def score_of(item: dict[str, Any], preferred_field: str | None = None) -> float:
    """Возвращает score результата с fallback под старый и новый aisearch."""
    candidates = []
    if preferred_field:
        candidates.append(preferred_field)
    candidates.extend(
        [
            "final_score",       # старый bench default
            "score_final",       # старые/альтернативные схемы
            "score_ce",          # новый aisearch при reranker_enabled=true
            "score_fusion",      # новый aisearch при reranker_enabled=false
            "score_dense",
            "score_lex",
        ]
    )
    for field_name in candidates:
        value = item.get(field_name)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


@dataclass(slots=True)
class SearchApiConfig:
    base_url: str = "http://localhost:5155"
    search_path: str = "/hybrid-search/search"
    info_path: str = "/hybrid-search/info/{task_id}"
    top_k: int = 10
    request_timeout: float = 60.0
    poll_interval: float = 0.5
    poll_timeout: float = 120.0
    retry_count: int = 5
    retry_wait_423: float = 5.0
    retry_wait_429: float = 5.0
    search_use_cache: bool = False
    metrics_enable: bool = True
    show_intermediate_results: bool = False
    presearch: dict[str, Any] | None = None
    filters: dict[str, Any] | None = None
    done_status: str = "done"
    inwork_statuses: set[str] = field(default_factory=lambda: {"queued", "running"})
    fail_statuses: set[str] = field(default_factory=lambda: {"failed", "not_found"})

    @classmethod
    def from_env(cls, *, default_top_k: int = 10, default_intermediate: bool = False) -> "SearchApiConfig":
        base_url = (os.getenv("API_BASE_URL") or "http://localhost:5155").rstrip("/")
        search_path = os.getenv("API_SEARCH_PATH") or os.getenv("API_SEARCH_URL") or "/hybrid-search/search"
        info_path = os.getenv("API_STATUS_PATH") or os.getenv("API_INFO_URL") or "/hybrid-search/info/{task_id}"
        if not search_path.startswith("/"):
            search_path = "/" + search_path
        if not info_path.startswith("/"):
            info_path = "/" + info_path
        if "{task_id}" not in info_path:
            info_path = info_path.rstrip("/") + "/{task_id}"

        inwork = os.getenv("API_STATUS_INWORK_STATUS_VALUE", "queued,running")
        fail = os.getenv("API_STATUS_FAIL_STATUS_VALUE", "failed,not_found")
        return cls(
            base_url=base_url,
            search_path=search_path,
            info_path=info_path,
            top_k=env_int("TOP_K", env_int("API_TOP_K", default_top_k)),
            request_timeout=env_float("API_SEARCH_TIMEOUT", 60.0),
            poll_interval=env_float("API_INFO_INTERVAL", env_int("API_STATUS_REQUEST_INTERVAL", 500) / 1000.0),
            poll_timeout=env_float("API_INFO_MAX_TIME", env_float("API_STATUS_REQUEST_TIMEOUT", 120.0)),
            retry_count=env_int("API_SEARCH_RETRY_COUNT", 5),
            retry_wait_423=env_float("API_SEARCH_OVERFLOW_QUERY_TIMEOUT", 5.0),
            retry_wait_429=env_float("API_SEARCH_QUERY_LIMIT_TIMEOUT", 5.0),
            search_use_cache=env_bool("API_SEARCH_USE_CACHE", False),
            metrics_enable=env_bool("API_METRICS_ENABLE", True),
            show_intermediate_results=env_bool("API_SHOW_INTERMEDIATE_RESULTS", default_intermediate),
            presearch=env_json("API_PRESEARCH_JSON", None),
            filters=env_json("API_FILTERS_JSON", None),
            done_status=os.getenv("API_STATUS_STATUS_OK_VALUE", "done"),
            inwork_statuses={s.strip() for s in inwork.split(",") if s.strip()},
            fail_statuses={s.strip() for s in fail.split(",") if s.strip()},
        )

    @property
    def search_url(self) -> str:
        return self.base_url + self.search_path

    def info_url(self, task_id: str) -> str:
        return self.base_url + self.info_path.format(task_id=task_id)


def build_search_payload(query: str, cfg: SearchApiConfig, *, top_k: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "top_k": cfg.top_k if top_k is None else top_k,
        "search_use_cache": cfg.search_use_cache,
        "metrics_enable": cfg.metrics_enable,
        "show_intermediate_results": cfg.show_intermediate_results,
    }
    if cfg.presearch:
        payload["presearch"] = cfg.presearch
    if cfg.filters:
        payload["filters"] = cfg.filters
    return payload


def enqueue_search(query: str, cfg: SearchApiConfig, *, top_k: int | None = None) -> tuple[str, dict[str, Any]]:
    payload = build_search_payload(query, cfg, top_k=top_k)
    retries_used = 0
    while True:
        started = time.perf_counter()
        try:
            response = requests.post(cfg.search_url, json=payload, timeout=cfg.request_timeout)
        except requests.RequestException as exc:
            raise BenchmarkApiError(f"POST {cfg.search_url} failed: {exc}") from exc
        elapsed = time.perf_counter() - started

        if response.status_code == 202:
            try:
                data = response.json()
            except ValueError as exc:
                raise BenchmarkApiError(f"POST returned non-JSON body: {response.text[:1000]}") from exc
            task_id = data.get("task_id")
            if not task_id:
                raise BenchmarkApiError(f"POST response has no task_id: {data}")
            return str(task_id), {"post_time": elapsed, "post_response": data, "payload": payload}

        if response.status_code in {423, 429} and retries_used < cfg.retry_count:
            retries_used += 1
            time.sleep(cfg.retry_wait_423 if response.status_code == 423 else cfg.retry_wait_429)
            continue

        raise BenchmarkApiError(
            f"POST {cfg.search_url} returned {response.status_code}: {response.text[:1000]}"
        )


def poll_search(task_id: str, cfg: SearchApiConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    statuses: list[str] = []
    polls = 0
    url = cfg.info_url(task_id)
    while True:
        if time.perf_counter() - started > cfg.poll_timeout:
            raise BenchmarkApiError(f"Timeout while polling {url}; statuses={statuses}")
        try:
            response = requests.get(url, timeout=cfg.request_timeout)
        except requests.RequestException as exc:
            raise BenchmarkApiError(f"GET {url} failed: {exc}") from exc
        polls += 1
        if response.status_code != 200:
            raise BenchmarkApiError(f"GET {url} returned {response.status_code}: {response.text[:1000]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise BenchmarkApiError(f"GET returned non-JSON body: {response.text[:1000]}") from exc
        status = str(data.get("status", "")).strip().lower()
        statuses.append(status)
        if status == cfg.done_status:
            return data, {"polls": polls, "statuses": statuses, "poll_time": time.perf_counter() - started}
        if status in cfg.fail_statuses:
            raise BenchmarkApiError(f"Task {task_id} failed with status={status}: {data}")
        if status in cfg.inwork_statuses or not status:
            time.sleep(cfg.poll_interval)
            continue
        raise BenchmarkApiError(f"Unknown task status={status!r}: {data}")


def run_search(query: str, cfg: SearchApiConfig, *, top_k: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Полный цикл POST -> polling. Возвращает финальный ответ и служебные timings."""
    started = time.perf_counter()
    task_id, post_meta = enqueue_search(query, cfg, top_k=top_k)
    final_response, poll_meta = poll_search(task_id, cfg)
    meta = {
        **post_meta,
        **poll_meta,
        "task_id": task_id,
        "response_time": time.perf_counter() - started,
    }
    return final_response, meta
