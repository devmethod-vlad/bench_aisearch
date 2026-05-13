import os
import sys
import json
import time
import datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from tqdm import tqdm


class BenchmarkError(Exception):
    """Специальное исключение для аккуратного завершения скрипта с пояснениями."""


@dataclass
class ApiConfig:
    """Конфиг, собранный из .env_metrics"""

    # Базовые URL/пути
    base_url: str
    search_path: str
    status_path_template: str
    top_k: int

    # Runtime-параметры POST /search
    search_use_cache: bool
    metrics_enable: bool
    show_intermediate_results: bool
    presearch_field: Optional[str]

    # Параметры API постановки в очередь
    search_timeout: float
    search_status_field: str
    search_status_ok_value: str
    search_task_id_field: str
    search_overflow_timeout: float
    search_limit_timeout: float
    search_retry_count: int

    # Параметры API статуса
    status_interval: float
    status_timeout: float
    status_status_field: str
    status_status_ok_value: str
    status_inwork_values: List[str]
    status_fail_values: List[str]

    # Пути к результатам и метрикам
    results_path: str
    metrics_path: str
    metrics_embedding_field: str
    metrics_dense_field: str
    metrics_lex_field: str
    metrics_ce_field: str
    metrics_total_field: str
    metrics_task_field: str

    # Источник запросов
    source_file: Optional[str]
    source_file_query_field: Optional[str]
    test_query: str
    test_query_retry_count: int

    # Отчётность
    sorted_metrics_top_n: int
    results_json_path: Optional[str]

    # Отдельный файл с ответами
    separate_results_to_file: bool
    results_file_path: Optional[str]

    # Ограничение длины списка статусов при выводе в таблицах Top N
    status_list_limit: int


@dataclass
class RequestMetrics:
    """Все метрики по одному запросу."""

    index: int
    query: str
    query_length: int

    # Время целиком
    overall_total_time: float  # от первой попытки POST до готового ответа

    # Этап постановки в очередь
    queue_total_time: float
    queue_retry_wait_total: float
    queue_retry_wait_423: float
    queue_retry_wait_429: float
    queue_retry_count_423: int
    queue_retry_count_429: int

    # Этап опроса статуса
    status_total_time: float
    status_poll_count: int
    status_retry_wait_total: float
    status_history: List[str]  # история статусов в порядке получения

    # Метрики из API — могут отсутствовать (None)
    metrics_embedding_time: Optional[float]
    metrics_vector_time: Optional[float]
    metrics_lexical_time: Optional[float]
    metrics_ce_time: Optional[float]
    metrics_total_time: Optional[float]
    metrics_full_task_time: Optional[float]  # уже переведено в секунды

    # Полный ответ API статуса (успешный)
    api_response: Dict[str, Any]


def load_env() -> None:
    """Загрузка переменных окружения из .env_metrics рядом со скриптом."""
    dotenv_path = Path(".env_metrics")
    if dotenv_path.exists():
        load_dotenv(dotenv_path)


def _get_env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return val


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise BenchmarkError(f"Переменная окружения {name} должна быть целым числом, получено: {raw!r}")


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise BenchmarkError(f"Переменная окружения {name} должна быть числом, получено: {raw!r}")


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    raw_l = raw.strip().lower()
    return raw_l in ("1", "true", "yes", "y", "on")


def build_config() -> ApiConfig:
    """Читает все нужные переменные окружения и собирает конфиг."""

    base_url = _get_env_str("API_BASE_URL", "http://localhost:5155")
    if base_url is None:
        raise BenchmarkError("API_BASE_URL не указан и не имеет значения по умолчанию")

    base_url = base_url.rstrip("/")

    search_path = _get_env_str("API_SEARCH_PATH", "/hybrid-search/search") or "/hybrid-search/search"
    if not search_path.startswith("/"):
        search_path = "/" + search_path

    status_path_template = _get_env_str(
        "API_STATUS_PATH", "/hybrid-search/info/{task_id}"
    ) or "/hybrid-search/info/{task_id}"
    if not status_path_template.startswith("/"):
        status_path_template = "/" + status_path_template

    top_k = _get_env_int("API_TOP_K", 10)

    search_use_cache = _get_env_bool("API_SEARCH_USE_CACHE", False)
    metrics_enable = _get_env_bool("API_SEARCH_METRICS_ENABLE", True)
    show_intermediate_results = _get_env_bool("API_SHOW_INTERMEDIATE_RESULTS", False)
    presearch_field = _get_env_str("API_PRESEARCH_FIELD", None)

    # Параметры постановки в очередь
    search_timeout = _get_env_float("API_SEARCH_TIMEOUT", 60.0)
    search_status_field = _get_env_str("API_SEARCH_STATUS_FIELD", "status") or "status"
    search_status_ok_value = _get_env_str("API_SEARCH_STATUS_OK_VALUE", "queued") or "queued"

    search_task_id_field = _get_env_str("API_SEARCH_TASK_ID_VALUE", "task_id") or "task_id"

    search_overflow_timeout = _get_env_float("API_SEARCH_OVERFLOW_QUERY_TIMEOUT", 5.0)
    search_limit_timeout = _get_env_float("API_SEARCH_QUERY_LIMIT_TIMEOUT", 5.0)
    search_retry_count = _get_env_int("API_SEARCH_RETRY_COUNT", 5)

    # Параметры статуса
    status_interval_ms = _get_env_int("API_STATUS_REQUEST_INTERVAL", 500)
    status_interval = status_interval_ms / 1000.0
    status_timeout = _get_env_float("API_STATUS_REQUEST_TIMEOUT", 60.0)

    status_status_field = _get_env_str("API_STATUS_STATUS_FIELD", "status") or "status"
    status_status_ok_value = _get_env_str("API_STATUS_STATUS_OK_VALUE", "done") or "done"

    inwork_raw = _get_env_str("API_STATUS_INWORK_STATUS_VALUE", "queued,running") or "queued,running"
    status_inwork_values = [s.strip() for s in inwork_raw.split(",") if s.strip()]

    fail_raw = _get_env_str("API_STATUS_FAIL_STATUS_VALUE", "failed,not_found") or "failed,not_found"
    status_fail_values = [s.strip() for s in fail_raw.split(",") if s.strip()]

    # Пути к данным в ответе
    results_path = _get_env_str("API_RESULTS_PATH", "info.results") or "info.results"
    metrics_path = _get_env_str("API_METRICS_PATH", "info.metrics") or "info.metrics"

    metrics_embedding_field = _get_env_str(
        "API_METRICS_EMBED_TIME_FIELD", "embedding_time"
    ) or "embedding_time"
    metrics_dense_field = _get_env_str(
        "API_METRICS_DENSE_TIME_FIELD", "vector_search_time"
    ) or "vector_search_time"
    metrics_lex_field = _get_env_str(
        "API_METRICS_LEX_TIME_FIELD", "lexical_search_time"
    ) or "lexical_search_time"
    metrics_ce_field = _get_env_str(
        "API_METRICS_CE_TIME_FIELD", "cross_encoder_time"
    ) or "cross_encoder_time"
    metrics_total_field = _get_env_str(
        "API_METRICS_SEARCH_TIME_FIELD", "total_time"
    ) or "total_time"
    metrics_task_field = _get_env_str(
        "API_METRICS_TASK_TIME_FIELD", "full_search_task_time"
    ) or "full_search_task_time"

    # Источник запросов
    source_file = _get_env_str("SOURCE_FILE", None)
    source_file_query_field = _get_env_str("SOURCE_FILE_QUERY_FIELD", None)

    test_query = _get_env_str("TEST_QUERY", "тестовый запрос") or "тестовый запрос"
    test_query_retry_count = _get_env_int("TEST_QUERY_RETRY_COUNT", 20)

    # Report
    sorted_metrics_top_n = _get_env_int("SORTED_METRICS_TOP_N", 5)
    results_json_path = _get_env_str("RESULTS_JSON_PATH", None)

    # Ограничение длины списка статусов для вывода
    status_list_limit = _get_env_int("STATUS_LIST_LIMIT", 5)

    # Отдельный файл для результатов
    separate_results_to_file = _get_env_bool("SEPARATE_RESULTS_TO_FILE", True)
    results_file_path = _get_env_str("RESULTS_FILE_PATH", None)

    return ApiConfig(
        base_url=base_url,
        search_path=search_path,
        status_path_template=status_path_template,
        top_k=top_k,
        search_use_cache=search_use_cache,
        metrics_enable=metrics_enable,
        show_intermediate_results=show_intermediate_results,
        presearch_field=presearch_field,
        search_timeout=search_timeout,
        search_status_field=search_status_field,
        search_status_ok_value=search_status_ok_value,
        search_task_id_field=search_task_id_field,
        search_overflow_timeout=search_overflow_timeout,
        search_limit_timeout=search_limit_timeout,
        search_retry_count=search_retry_count,
        status_interval=status_interval,
        status_timeout=status_timeout,
        status_status_field=status_status_field,
        status_status_ok_value=status_status_ok_value,
        status_inwork_values=status_inwork_values,
        status_fail_values=status_fail_values,
        results_path=results_path,
        metrics_path=metrics_path,
        metrics_embedding_field=metrics_embedding_field,
        metrics_dense_field=metrics_dense_field,
        metrics_lex_field=metrics_lex_field,
        metrics_ce_field=metrics_ce_field,
        metrics_total_field=metrics_total_field,
        metrics_task_field=metrics_task_field,
        source_file=source_file,
        source_file_query_field=source_file_query_field,
        test_query=test_query,
        test_query_retry_count=test_query_retry_count,
        sorted_metrics_top_n=sorted_metrics_top_n,
        results_json_path=results_json_path,
        separate_results_to_file=separate_results_to_file,
        results_file_path=results_file_path,
        status_list_limit=status_list_limit,
    )


def get_from_dot_path(data: Dict[str, Any], path: str) -> Any:
    """Достаёт значение из вложенного словаря по точечной нотации."""
    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise BenchmarkError(
                f"Не найден путь '{path}' в ответе API (отсутствует ключ '{part}')"
            )
    return current


def queue_search_task(query: str, cfg: ApiConfig, console: Console) -> Tuple[str, Dict[str, Any]]:
    """POST на ручку постановки задачи в очередь + учёт ретраев 423/429."""

    url = cfg.base_url + cfg.search_path
    payload = {
        "query": query,
        "top_k": cfg.top_k,
        "search_use_cache": cfg.search_use_cache,
        "metrics_enable": cfg.metrics_enable,
        "show_intermediate_results": cfg.show_intermediate_results,
    }
    if cfg.presearch_field:
        payload["presearch"] = {"field": cfg.presearch_field}

    queue_stage_start = time.perf_counter()

    retry_wait_total = 0.0
    retry_wait_423 = 0.0
    retry_wait_429 = 0.0
    retry_count_423 = 0
    retry_count_429 = 0
    retries_used = 0

    while True:
        attempt_start = time.perf_counter()
        try:
            response = requests.post(url, json=payload, timeout=cfg.search_timeout)
        except requests.Timeout:
            elapsed = time.perf_counter() - attempt_start
            raise BenchmarkError(
                f"Таймаут при обращении к API постановки в очередь (ждали {elapsed:.4f} c, timeout={cfg.search_timeout}s)"
            )
        except requests.RequestException as exc:
            elapsed = time.perf_counter() - attempt_start
            raise BenchmarkError(
                f"Ошибка сетевого уровня при обращении к API постановки в очередь "
                f"(время попытки {elapsed:.4f} c): {exc}"
            )

        elapsed_attempt = time.perf_counter() - attempt_start
        status_code = response.status_code
        body_text = response.text

        if status_code == 202:
            try:
                data = response.json()
            except ValueError:
                raise BenchmarkError(
                    f"Не удалось распарсить JSON ответа постановки в очередь, код {status_code}. "
                    f"Тело: {body_text[:1000]}"
                )

            status_value = data.get(cfg.search_status_field)
            if status_value != cfg.search_status_ok_value:
                raise BenchmarkError(
                    f"Неверный статус постановки задачи: поле '{cfg.search_status_field}'="
                    f"{status_value!r}, ожидалось {cfg.search_status_ok_value!r}. "
                    f"Полный ответ: {json.dumps(data, ensure_ascii=False)[:1000]}"
                )

            if cfg.search_task_id_field not in data:
                raise BenchmarkError(
                    f"В ответе постановки в очередь отсутствует поле task_id "
                    f"('{cfg.search_task_id_field}'). Ответ: {json.dumps(data, ensure_ascii=False)[:1000]}"
                )

            task_id = str(data[cfg.search_task_id_field])
            queue_total_time = time.perf_counter() - queue_stage_start

            return task_id, {
                "queue_total_time": queue_total_time,
                "queue_retry_wait_total": retry_wait_total,
                "queue_retry_wait_423": retry_wait_423,
                "queue_retry_wait_429": retry_wait_429,
                "queue_retry_count_423": retry_count_423,
                "queue_retry_count_429": retry_count_429,
                "last_attempt_time": elapsed_attempt,
            }

        if status_code in (423, 429):
            retries_used += 1
            if retries_used > cfg.search_retry_count:
                raise BenchmarkError(
                    f"Превышено максимальное число повторов API постановки в очередь "
                    f"({cfg.search_retry_count}) для кодов 423/429. "
                    f"Последний код: {status_code}, тело: {body_text[:1000]}"
                )

            if status_code == 423:
                wait = cfg.search_overflow_timeout
                retry_count_423 += 1
                retry_wait_423 += wait
                reason = "переполнение очереди (423)"
            else:
                wait = cfg.search_limit_timeout
                retry_count_429 += 1
                retry_wait_429 += wait
                reason = "превышен лимит запросов (429)"

            retry_wait_total += wait
            console.print(
                f"[yellow]Получен код {status_code} ({reason}), "
                f"ожидаем {wait:.2f} c перед повтором "
                f"({retries_used}/{cfg.search_retry_count}).[/yellow]"
            )
            time.sleep(wait)
            continue

        raise BenchmarkError(
            f"Неожиданный код ответа постановки в очередь: {status_code}. "
            f"Время попытки: {elapsed_attempt:.4f} c. "
            f"Тело: {body_text[:1000]}"
        )


def poll_status(task_id: str, cfg: ApiConfig, console: Console) -> Dict[str, Any]:
    """Периодический опрос статуса задачи до done/ошибки + сбор истории статусов."""

    url = cfg.base_url + cfg.status_path_template.format(task_id=task_id)

    status_stage_start = time.perf_counter()
    poll_count = 0
    retry_wait_total = 0.0
    status_history: List[str] = []

    while True:
        attempt_start = time.perf_counter()
        try:
            response = requests.get(url, timeout=cfg.status_timeout)
        except requests.Timeout:
            elapsed = time.perf_counter() - attempt_start
            raise BenchmarkError(
                f"Таймаут при обращении к API статуса (ждали {elapsed:.4f} c, timeout={cfg.status_timeout}s)"
            )
        except requests.RequestException as exc:
            elapsed = time.perf_counter() - attempt_start
            raise BenchmarkError(
                f"Ошибка сетевого уровня при обращении к API статуса "
                f"(время попытки {elapsed:.4f} c): {exc}"
            )

        poll_count += 1
        status_code = response.status_code
        body_text = response.text

        if status_code != 200:
            raise BenchmarkError(
                f"Неожиданный HTTP-код от API статуса: {status_code}. "
                f"Тело: {body_text[:1000]}"
            )

        try:
            data = response.json()
        except ValueError:
            raise BenchmarkError(
                f"Не удалось распарсить JSON ответа статуса. "
                f"Тело: {body_text[:1000]}"
            )

        if cfg.status_status_field not in data:
            raise BenchmarkError(
                f"В ответе статуса нет поля '{cfg.status_status_field}'. "
                f"Ответ: {json.dumps(data, ensure_ascii=False)[:1000]}"
            )

        status_value_raw = data[cfg.status_status_field]
        status_value = str(status_value_raw).strip()
        status_history.append(status_value)

        if status_value == cfg.status_status_ok_value:
            status_total_time = time.perf_counter() - status_stage_start
            return {
                "status_total_time": status_total_time,
                "status_poll_count": poll_count,
                "status_retry_wait_total": retry_wait_total,
                "status_history": status_history,
                "api_response": data,
            }

        if status_value in cfg.status_inwork_values:
            time.sleep(cfg.status_interval)
            retry_wait_total += cfg.status_interval
            continue

        if status_value in cfg.status_fail_values:
            raise BenchmarkError(
                f"Задача завершилась неуспешным статусом {status_value!r}. "
                f"Ответ: {json.dumps(data, ensure_ascii=False)[:1000]}"
            )

        raise BenchmarkError(
            f"Получен неизвестный статус задачи '{status_value}' "
            f"(поле {cfg.status_status_field}). "
            f"Ответ: {json.dumps(data, ensure_ascii=False)[:1000]}"
        )


def extract_metrics_from_response(
    api_response: Dict[str, Any], cfg: ApiConfig
) -> Dict[str, Optional[float]]:
    """Тянем метрики из ответа, отсутствующие — как None, full_task_time переводим в секунды."""
    metrics_obj = get_from_dot_path(api_response, cfg.metrics_path)
    if not isinstance(metrics_obj, dict):
        raise BenchmarkError(
            f"По пути '{cfg.metrics_path}' в ответе лежит не словарь с метриками: {metrics_obj!r}"
        )

    def maybe_get(field: str) -> Optional[float]:
        if field not in metrics_obj:
            return None
        val = metrics_obj.get(field)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    embedding_time = maybe_get(cfg.metrics_embedding_field)
    dense_time = maybe_get(cfg.metrics_dense_field)
    lex_time = maybe_get(cfg.metrics_lex_field)
    ce_time = maybe_get(cfg.metrics_ce_field)
    total_time = maybe_get(cfg.metrics_total_field)
    task_time_ms = maybe_get(cfg.metrics_task_field)
    task_time_sec = task_time_ms / 1000.0 if task_time_ms is not None else None

    return {
        "embedding_time": embedding_time,
        "vector_search_time": dense_time,
        "lexical_search_time": lex_time,
        "cross_encoder_time": ce_time,
        "total_time": total_time,
        "full_search_task_time": task_time_sec,
    }


def load_queries(cfg: ApiConfig) -> List[str]:
    """Берём запросы из файла (Excel/Parquet) или повторяем TEST_QUERY."""

    if cfg.source_file:
        path = Path(cfg.source_file)
        if not path.exists():
            raise BenchmarkError(f"Файл SOURCE_FILE не найден: {path}")

        if not cfg.source_file_query_field:
            raise BenchmarkError(
                "Указан SOURCE_FILE, но не задан SOURCE_FILE_QUERY_FIELD (имя столбца с запросами)."
            )

        suffix = path.suffix.lower()
        if suffix in (".xls", ".xlsx", ".xlsm"):
            df = pd.read_excel(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise BenchmarkError(
                f"Неподдерживаемое расширение SOURCE_FILE: {suffix}. Ожидается Excel или Parquet."
            )

        if cfg.source_file_query_field not in df.columns:
            raise BenchmarkError(
                f"В файле {path} нет столбца '{cfg.source_file_query_field}'. "
                f"Колонки: {list(df.columns)}"
            )

        series = df[cfg.source_file_query_field].dropna()
        series = series.astype(str)
        queries = [q.strip() for q in series if q.strip()]

        if not queries:
            raise BenchmarkError(
                f"В столбце '{cfg.source_file_query_field}' файла {path} нет ни одного непустого запроса."
            )

        return queries

    if not cfg.test_query:
        raise BenchmarkError(
            "Не указан SOURCE_FILE и пустой TEST_QUERY — нечего отправлять в поиск."
        )

    if cfg.test_query_retry_count <= 0:
        raise BenchmarkError("TEST_QUERY_RETRY_COUNT должен быть > 0.")

    return [cfg.test_query] * cfg.test_query_retry_count


def metrics_to_serializable(m: RequestMetrics, include_response: bool = True) -> Dict[str, Any]:
    """Dict для json, округление времён, api_response опционален."""
    metrics_block: Dict[str, Any] = {}

    if m.metrics_embedding_time is not None:
        metrics_block["embedding_time"] = round(m.metrics_embedding_time, 4)
    if m.metrics_vector_time is not None:
        metrics_block["vector_search_time"] = round(m.metrics_vector_time, 4)
    if m.metrics_lexical_time is not None:
        metrics_block["lexical_search_time"] = round(m.metrics_lexical_time, 4)
    if m.metrics_ce_time is not None:
        metrics_block["cross_encoder_time"] = round(m.metrics_ce_time, 4)
    if m.metrics_total_time is not None:
        metrics_block["total_time"] = round(m.metrics_total_time, 4)
    if m.metrics_full_task_time is not None:
        metrics_block["full_search_task_time"] = round(m.metrics_full_task_time, 4)

    base: Dict[str, Any] = {
        "index": m.index,
        "query": m.query,
        "query_length": m.query_length,
        "overall_total_time": round(m.overall_total_time, 4),
        "queue": {
            "queue_total_time": round(m.queue_total_time, 4),
            "queue_retry_wait_total": round(m.queue_retry_wait_total, 4),
            "queue_retry_wait_423": round(m.queue_retry_wait_423, 4),
            "queue_retry_wait_429": round(m.queue_retry_wait_429, 4),
            "queue_retry_count_423": m.queue_retry_count_423,
            "queue_retry_count_429": m.queue_retry_count_429,
        },
        "status": {
            "status_total_time": round(m.status_total_time, 4),
            "status_poll_count": m.status_poll_count,
            "status_retry_wait_total": round(m.status_retry_wait_total, 4),
            "status_history": m.status_history,
        },
        "metrics": metrics_block,
    }

    if include_response:
        base["api_response"] = m.api_response

    return base


def compute_summary(metrics_list: List[RequestMetrics]) -> Dict[str, Any]:
    """Считает средние значения и агрегированные счётчики по всем запросам."""
    n = len(metrics_list)
    if n == 0:
        raise BenchmarkError("Нет данных для подсчёта статистики.")

    def avg(attr: str) -> float:
        total = sum(getattr(m, attr) for m in metrics_list)
        return total / n

    def avg_optional(attr: str) -> Optional[float]:
        vals: List[float] = []
        for m in metrics_list:
            v = getattr(m, attr)
            if v is not None:
                vals.append(v)
        if not vals:
            return None
        return sum(vals) / len(vals)

    # Средние времена
    avg_overall = avg("overall_total_time")
    avg_queue = avg("queue_total_time")
    avg_queue_retry_wait_total = avg("queue_retry_wait_total")

    avg_status = avg("status_total_time")
    avg_status_retry_wait = avg("status_retry_wait_total")

    # 423/429
    total_423 = sum(m.queue_retry_count_423 for m in metrics_list)
    total_429 = sum(m.queue_retry_count_429 for m in metrics_list)
    total_retry_codes = total_423 + total_429

    total_queue_retry_wait_total = sum(m.queue_retry_wait_total for m in metrics_list)
    total_queue_retry_wait_423 = sum(m.queue_retry_wait_423 for m in metrics_list)
    total_queue_retry_wait_429 = sum(m.queue_retry_wait_429 for m in metrics_list)

    avg_retry_codes = total_retry_codes / n
    avg_retry_423 = total_423 / n
    avg_retry_429 = total_429 / n
    avg_queue_retry_wait_423 = total_queue_retry_wait_423 / n
    avg_queue_retry_wait_429 = total_queue_retry_wait_429 / n
    avg_queue_retry_wait_total_metric = total_queue_retry_wait_total / n

    # Опрос статуса
    total_status_retries = sum(max(m.status_poll_count - 1, 0) for m in metrics_list)
    avg_status_retries = total_status_retries / n

    # Средние метрики поиска
    avg_emb = avg_optional("metrics_embedding_time")
    avg_dense = avg_optional("metrics_vector_time")
    avg_lex = avg_optional("metrics_lexical_time")
    avg_ce = avg_optional("metrics_ce_time")
    avg_total = avg_optional("metrics_total_time")
    avg_task = avg_optional("metrics_full_task_time")

    # Длины запросов
    lengths = [m.query_length for m in metrics_list]
    avg_len = sum(lengths) / n
    min_len = min(lengths)
    max_len = max(lengths)

    return {
        "requests_count": n,
        "overall": {
            "avg_overall_total_time": avg_overall,
        },
        "queue_stage": {
            "avg_queue_total_time": avg_queue,
            "avg_queue_retry_wait_total": avg_queue_retry_wait_total,
            "avg_retry_codes_423_and_429": avg_retry_codes,
            "avg_retry_code_423": avg_retry_423,
            "avg_retry_code_429": avg_retry_429,
            "avg_queue_retry_wait_423": avg_queue_retry_wait_423,
            "avg_queue_retry_wait_429": avg_queue_retry_wait_429,
            "avg_queue_retry_wait_total_metric": avg_queue_retry_wait_total_metric,
        },
        "status_stage": {
            "avg_status_total_time": avg_status,
            "avg_status_retry_wait": avg_status_retry_wait,
            "avg_status_retries": avg_status_retries,
        },
        "metrics_avg": {
            "embedding_time": avg_emb,
            "vector_search_time": avg_dense,
            "lexical_search_time": avg_lex,
            "cross_encoder_time": avg_ce,
            "total_time": avg_total,
            "full_search_task_time": avg_task,
        },
        "query_length": {
            "avg_length": avg_len,
            "min_length": min_len,
            "max_length": max_len,
        },
    }


def print_summary(summary: Dict[str, Any], console: Console) -> None:
    """Вывод компактных summary-таблиц (как тебе понравилось)."""
    overall = summary["overall"]
    queue = summary["queue_stage"]
    status = summary["status_stage"]
    metrics_avg = summary["metrics_avg"]
    qlen = summary["query_length"]

    def fmt_opt(v: Optional[float]) -> str:
        return "—" if v is None else f"{v:.4f}"

    # Средние времена
    t = Table(title="Средние времена запросов (секунды)", show_lines=True)
    t.add_column("Метрика")
    t.add_column("Значение", justify="right")

    t.add_row("Среднее общее время запроса",
              f"{overall['avg_overall_total_time']:.4f}")
    t.add_row("Среднее время постановки в очередь",
              f"{queue['avg_queue_total_time']:.4f}")
    t.add_row("Среднее время ожидания ретраев (423/429)",
              f"{queue['avg_queue_retry_wait_total']:.4f}")
    t.add_row("Среднее время этапа опроса статуса",
              f"{status['avg_status_total_time']:.4f}")
    t.add_row("Среднее время ожидания между повторами статуса",
              f"{status['avg_status_retry_wait']:.4f}")

    console.print(t)

    # 423/429
    t2 = Table(title="Коды 423 / 429 (переполнение / лимит)", show_lines=True)
    t2.add_column("Показатель")
    t2.add_column("Значение", justify="right")

    t2.add_row(
        "Среднее суммарное число кодов 423 + 429 на запрос",
        f"{queue['avg_retry_codes_423_and_429']:.4f}",
    )
    t2.add_row(
        "Среднее суммарное число кодов 423 на запрос",
        f"{queue['avg_retry_code_423']:.4f}",
    )
    t2.add_row(
        "Среднее суммарное число кодов 429 на запрос",
        f"{queue['avg_retry_code_429']:.4f}",
    )
    t2.add_row(
        "Среднее суммарное время ожидания ретраев (все) на запрос",
        f"{queue['avg_queue_retry_wait_total_metric']:.4f}",
    )
    t2.add_row(
        "Среднее суммарное время ожидания по коду 423 на запрос",
        f"{queue['avg_queue_retry_wait_423']:.4f}",
    )
    t2.add_row(
        "Среднее суммарное время ожидания по коду 429 на запрос",
        f"{queue['avg_queue_retry_wait_429']:.4f}",
    )

    console.print(t2)

    # Опрос статуса
    t3 = Table(title="Опрос статуса задачи", show_lines=True)
    t3.add_column("Показатель")
    t3.add_column("Значение", justify="right")

    t3.add_row(
        "Среднее число повторных запросов статуса (poll > 1)",
        f"{status['avg_status_retries']:.4f}",
    )
    t3.add_row(
        "Среднее время ожидания между повторами статуса",
        f"{status['avg_status_retry_wait']:.4f}",
    )

    console.print(t3)

    # Метрики поиска
    t4 = Table(title="Средние метрики поиска (из API)", show_lines=True)
    t4.add_column("Метрика")
    t4.add_column("Значение (секунды)", justify="right")

    t4.add_row("embedding_time", fmt_opt(metrics_avg["embedding_time"]))
    t4.add_row("vector_search_time", fmt_opt(metrics_avg["vector_search_time"]))
    t4.add_row("lexical_search_time", fmt_opt(metrics_avg["lexical_search_time"]))
    t4.add_row("cross_encoder_time", fmt_opt(metrics_avg["cross_encoder_time"]))
    t4.add_row("total_time", fmt_opt(metrics_avg["total_time"]))
    t4.add_row("full_search_task_time", fmt_opt(metrics_avg["full_search_task_time"]))

    console.print(t4)

    # Длины запросов
    t5 = Table(title="Длины запросов (символы)", show_lines=True)
    t5.add_column("Показатель")
    t5.add_column("Значение", justify="right")

    t5.add_row("Средняя длина", f"{qlen['avg_length']:.1f}")
    t5.add_row("Мин. длина", str(qlen["min_length"]))
    t5.add_row("Макс. длина", str(qlen["max_length"]))

    console.print(t5)


def build_top_lists(
    metrics_list: List[RequestMetrics],
    top_n: int,
    include_response: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Top N по разным метрикам для json."""

    def metric_or_minus(v: Optional[float]) -> float:
        return v if v is not None else -1.0

    def top_by(key_fn):
        return sorted(metrics_list, key=key_fn, reverse=True)[: int(top_n)]

    top_data: Dict[str, List[Dict[str, Any]]] = {}

    top_data["by_overall_total_time"] = [
        metrics_to_serializable(m, include_response=include_response)
        for m in top_by(lambda m: m.overall_total_time)
    ]
    top_data["by_queue_total_time"] = [
        metrics_to_serializable(m, include_response=include_response)
        for m in top_by(lambda m: m.queue_total_time)
    ]
    top_data["by_status_total_time"] = [
        metrics_to_serializable(m, include_response=include_response)
        for m in top_by(lambda m: m.status_total_time)
    ]
    top_data["by_total_time_metric"] = [
        metrics_to_serializable(m, include_response=include_response)
        for m in top_by(lambda m: metric_or_minus(m.metrics_total_time))
    ]
    top_data["by_full_search_task_time_metric"] = [
        metrics_to_serializable(m, include_response=include_response)
        for m in top_by(lambda m: metric_or_minus(m.metrics_full_task_time))
    ]
    top_data["by_embedding_time_metric"] = [
        metrics_to_serializable(m, include_response=include_response)
        for m in top_by(lambda m: metric_or_minus(m.metrics_embedding_time))
    ]
    top_data["by_vector_search_time_metric"] = [
        metrics_to_serializable(m, include_response=include_response)
        for m in top_by(lambda m: metric_or_minus(m.metrics_vector_time))
    ]
    top_data["by_lexical_search_time_metric"] = [
        metrics_to_serializable(m, include_response=include_response)
        for m in top_by(lambda m: metric_or_minus(m.metrics_lexical_time))
    ]
    top_data["by_cross_encoder_time_metric"] = [
        metrics_to_serializable(m, include_response=include_response)
        for m in top_by(lambda m: metric_or_minus(m.metrics_ce_time))
    ]

    return top_data


def format_status_history_for_table(history: List[str], limit: int) -> str:
    """
    Форматирование истории статусов для вывода в таблицах Top N.

    Берём последние limit статусов (с конца), чтобы не раздувать таблицу,
    и выводим их в столбик (каждый статус с новой строки).
    """
    if not history:
        return ""
    trimmed = history
    if limit > 0 and len(trimmed) > limit:
        trimmed = trimmed[-limit:]
    return "\n".join(trimmed)


def print_top_tables(
    metrics_list: List[RequestMetrics],
    top_n: int,
    status_list_limit: int,
    console: Console,
) -> None:
    """
    Выводит в консоль топ N запросов по разным метрикам.

    Здесь снова делаем широкие таблицы: expand=True и Console с width=200 (в main).
    """

    def metric_or_zero(v: Optional[float]) -> float:
        return v if v is not None else 0.0

    top_specs = [
        ("Top N по общему времени запроса", lambda m: m.overall_total_time),
        ("Top N по времени постановки в очередь", lambda m: m.queue_total_time),
        ("Top N по времени этапа статуса", lambda m: m.status_total_time),
        ("Top N по total_time (метрика поиска)", lambda m: metric_or_zero(m.metrics_total_time)),
        ("Top N по full_search_task_time", lambda m: metric_or_zero(m.metrics_full_task_time)),
        ("Top N по embedding_time", lambda m: metric_or_zero(m.metrics_embedding_time)),
        ("Top N по vector_search_time", lambda m: metric_or_zero(m.metrics_vector_time)),
        ("Top N по lexical_search_time", lambda m: metric_or_zero(m.metrics_lexical_time)),
        ("Top N по cross_encoder_time", lambda m: metric_or_zero(m.metrics_ce_time)),
    ]

    for title, key_fn in top_specs:
        top = sorted(metrics_list, key=key_fn, reverse=True)[: int(top_n)]
        table = Table(title=title, show_lines=True, expand=True)
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("Req#", justify="right", no_wrap=True)
        table.add_column("Запрос (начало)", max_width=60)
        table.add_column("Длина", justify="right", no_wrap=True)
        table.add_column("T_total", justify="right", no_wrap=True)
        table.add_column("T_queue", justify="right", no_wrap=True)
        table.add_column("T_status", justify="right", no_wrap=True)
        table.add_column("423", justify="right", no_wrap=True)
        table.add_column("429", justify="right", no_wrap=True)
        table.add_column("T_queue_retry", justify="right", no_wrap=True)
        table.add_column("status_calls", justify="right", no_wrap=True)
        table.add_column("T_status_retry", justify="right", no_wrap=True)
        table.add_column("emb", justify="right", no_wrap=True)
        table.add_column("dense", justify="right", no_wrap=True)
        table.add_column("lex", justify="right", no_wrap=True)
        table.add_column("ce", justify="right", no_wrap=True)
        table.add_column("total", justify="right", no_wrap=True)
        table.add_column("full_task", justify="right", no_wrap=True)
        table.add_column("status list")  # история статусов в столбик

        for rank, m in enumerate(top, start=1):
            short_q = m.query
            if len(short_q) > 60:
                short_q = short_q[:57] + "..."

            def fmt_opt(v: Optional[float]) -> str:
                return "—" if v is None else f"{v:.4f}"

            status_list_str = format_status_history_for_table(
                m.status_history,
                status_list_limit,
            )

            table.add_row(
                str(rank),
                str(m.index),
                short_q,
                str(m.query_length),
                f"{m.overall_total_time:.4f}",
                f"{m.queue_total_time:.4f}",
                f"{m.status_total_time:.4f}",
                str(m.queue_retry_count_423),
                str(m.queue_retry_count_429),
                f"{m.queue_retry_wait_total:.4f}",
                str(m.status_poll_count),
                f"{m.status_retry_wait_total:.4f}",
                fmt_opt(m.metrics_embedding_time),
                fmt_opt(m.metrics_vector_time),
                fmt_opt(m.metrics_lexical_time),
                fmt_opt(m.metrics_ce_time),
                fmt_opt(m.metrics_total_time),
                fmt_opt(m.metrics_full_task_time),
                status_list_str,
            )

        console.print(table)


def save_results_json(
    out_path: Path,
    cfg: ApiConfig,
    metrics_list: List[RequestMetrics],
    summary: Dict[str, Any],
    top_data: Dict[str, List[Dict[str, Any]]],
    console: Console,
    include_response: bool,
) -> None:
    """Сохраняет сводную информацию и метрики в JSON-файл."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "generated_at": datetime.datetime.now().isoformat(),
        "config": asdict(cfg),
        "summary": summary,
        "top_requests": top_data,
        "requests": [
            metrics_to_serializable(m, include_response=include_response) for m in metrics_list
        ],
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    console.print(f"[green]Основной отчёт сохранён в {out_path}[/green]")


def save_results_bodies(
    out_path: Path,
    metrics_list: List[RequestMetrics],
    console: Console,
) -> None:
    """Сохраняет только тела ответов для каждого запроса в отдельный JSON-файл."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "generated_at": datetime.datetime.now().isoformat(),
        "requests": [
            {
                "index": m.index,
                "query": m.query,
                "api_response": m.api_response,
            }
            for m in metrics_list
        ],
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    console.print(f"[green]Результаты ответов сохранены в {out_path}[/green]")


def run_single_benchmark(
    index: int,
    query: str,
    cfg: ApiConfig,
    console: Console,
) -> RequestMetrics:
    """Полный цикл для одного запроса: POST -> опрос статуса -> сбор метрик."""
    overall_start = time.perf_counter()

    task_id, queue_info = queue_search_task(query, cfg, console)
    status_info = poll_status(task_id, cfg, console)
    overall_total_time = time.perf_counter() - overall_start

    api_response = status_info["api_response"]
    status_history = status_info.get("status_history", [])
    metrics_dict = extract_metrics_from_response(api_response, cfg)

    _ = get_from_dot_path(api_response, cfg.results_path)

    query_length = len(query)

    return RequestMetrics(
        index=index,
        query=query,
        query_length=query_length,
        overall_total_time=overall_total_time,
        queue_total_time=queue_info["queue_total_time"],
        queue_retry_wait_total=queue_info["queue_retry_wait_total"],
        queue_retry_wait_423=queue_info["queue_retry_wait_423"],
        queue_retry_wait_429=queue_info["queue_retry_wait_429"],
        queue_retry_count_423=queue_info["queue_retry_count_423"],
        queue_retry_count_429=queue_info["queue_retry_count_429"],
        status_total_time=status_info["status_total_time"],
        status_poll_count=status_info["status_poll_count"],
        status_retry_wait_total=status_info["status_retry_wait_total"],
        status_history=status_history,
        metrics_embedding_time=metrics_dict["embedding_time"],
        metrics_vector_time=metrics_dict["vector_search_time"],
        metrics_lexical_time=metrics_dict["lexical_search_time"],
        metrics_ce_time=metrics_dict["cross_encoder_time"],
        metrics_total_time=metrics_dict["total_time"],
        metrics_full_task_time=metrics_dict["full_search_task_time"],
        api_response=api_response,
    )


def main() -> None:
    # Обычный консоль для summary
    console = Console()
    # Широкий консоль (width=200) для топ-таблиц
    wide_console = Console(width=200)

    try:
        load_env()
        cfg = build_config()
        queries = load_queries(cfg)
    except BenchmarkError as e:
        console.print(f"[bold red]Ошибка при инициализации:[/bold red] {e}")
        sys.exit(1)

    console.print(
        Panel.fit(
            f"Запросов к выполнению: [bold]{len(queries)}[/bold]\n"
            f"API: [cyan]{cfg.base_url}[/cyan]{cfg.search_path} / {cfg.status_path_template}",
            title="Гибридный поиск — бенчмарк",
        )
    )

    metrics_list: List[RequestMetrics] = []

    # Прогресс-бар по полным запросам
    with tqdm(total=len(queries), desc="Выполнение запросов", unit="req") as pbar:
        for idx, query in enumerate(queries, start=1):
            try:
                m = run_single_benchmark(idx, query, cfg, console)
                metrics_list.append(m)
            except BenchmarkError as e:
                console.print("\n[bold red]Ошибка при выполнении запроса.[/bold red]")
                console.print(f"[red]{e}[/red]")
                if metrics_list:
                    console.print("\n[yellow]Статистика по успешно выполненным запросам до ошибки:[/yellow]")
                    summary = compute_summary(metrics_list)
                    print_summary(summary, console)
                sys.exit(1)
            pbar.update(1)

    summary = compute_summary(metrics_list)
    console.print("\n[bold green]Сводная статистика по всем запросам[/bold green]")
    print_summary(summary, console)

    console.print("\n[bold cyan]Top N запросов по различным метрикам[/bold cyan]")
    # Топ-таблицы печатаем широким консолью (width=200)
    print_top_tables(metrics_list, cfg.sorted_metrics_top_n, cfg.status_list_limit, wide_console)

    now = datetime.datetime.now()
    if cfg.results_json_path:
        main_path = Path(cfg.results_json_path)
        if not main_path.is_absolute():
            main_path = Path.cwd() / main_path
    else:
        ts = now.strftime("%Y%m%d_%H%M%S")
        # Без префикса hybrid_search_benchmark_
        main_path = Path.cwd() / f"{ts}.json"

    if cfg.separate_results_to_file:
        if cfg.results_file_path:
            results_path = Path(cfg.results_file_path)
            if not results_path.is_absolute():
                results_path = Path.cwd() / results_path
        else:
            stem = main_path.stem
            suffix = main_path.suffix or ".json"
            results_path = main_path.with_name(f"{stem}-results{suffix}")
    else:
        results_path = None

    include_response_in_main = not cfg.separate_results_to_file

    top_data = build_top_lists(
        metrics_list,
        cfg.sorted_metrics_top_n,
        include_response=include_response_in_main,
    )
    save_results_json(
        main_path,
        cfg,
        metrics_list,
        summary,
        top_data,
        console,
        include_response=include_response_in_main,
    )

    if results_path is not None:
        save_results_bodies(results_path, metrics_list, console)


if __name__ == "__main__":
    main()
