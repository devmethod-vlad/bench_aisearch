# bench_v3.py
# -*- coding: utf-8 -*-
"""
Бенчмарк гибридного поиска:
- Читает тестовый XLSX (env TEST_DATA_PATH), поля COL_QUERY/COL_TARGET_ID/COL_SOURCE (+ опционально COL_QUERY_SOURCE, COL_ANSWER)
- Дёргает API: POST API_SEARCH_URL ({"query", "top_k"}), затем poll GET API_INFO_URL/{task_id} до status=done
- Разбирает:
    * результаты по RESULTS_PATH (по умолчанию info.results),
    * метрики по METRICS_PATH (по умолчанию info.metrics),
    * (мягко) промежуточные результаты по INTERMEDIATE_RESULTS_PATH (по умолчанию info.intermediate_results) для dense/lex/ce
- Считает:
    * попадания по позициям (1..5, >5, >10), not_found,
    * MRR@K / nDCG@K, margin@1,
    * тайминги total/embedding/dense/lex (avg/p50/p95 точностью до .4f),
    * ce_avg (среднее время кросс-энкодера, по CE_SEARCH_TIME_FIELD),
    * response_avg - среднее время «постановки и ожидания результата» (от POST /search до получения status=done)
    * те же агрегаты по источникам (кроме intermediate response-таймингов)
    * расширенные списки найденных (объекты с target_id/test_query/test_query_source/test_answer)
    * top-N самых медленных запросов (total/embed/dense/ce/response) - и в JSON, и таблицами на общем холсте
- Сохраняет:
    * JSON (OUTPUT_BASENAME_DDMMYYYYHHMMSS.json),
    * XLSX (лист overall + листы по источникам) для финальной выдачи,
    * JPG - один общий холст (все графики сверху вниз + таблицы внизу).
Все параметры - через env (подгружаются через dotenv).
"""

import os
import time
import json
import math
import re
from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd
import requests

from dotenv import load_dotenv

import matplotlib.pyplot as plt
import matplotlib as mpl

# Используем современный доступ к палитрам (без DeprecationWarning)
# Пример: mpl.colormaps.get_cmap('tab20')
get_cmap = mpl.colormaps.get_cmap

# tqdm - мягкий импорт, чтобы скрипт работал без установленного tqdm
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    class _DummyTQDM:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a, **k): pass
        def update(self, *a, **k): pass
        def set_postfix(self, *a, **k): pass
        def set_postfix_str(self, *a, **k): pass
        def set_description_str(self, *a, **k): pass
        def refresh(self): pass
        def clear(self): pass
    tqdm = _DummyTQDM  # type: ignore


# ----------------------------- Утилиты ---------------------------------------

def safe_get(row, colname, default=""):
    """Безопасно достаём значение из строки DataFrame; если колонки/значения нет - default."""
    return str(row[colname]) if (colname in row and pd.notna(row[colname])) else default


def sanitize_sheet_name(name: str) -> str:
    """Допустимое имя листа Excel (≤31 символа, без : \\ / ? * [ ])."""
    if not name:
        return "source_unknown"
    name = re.sub(r'[:\\/?*\[\]]', "_", str(name))
    name = name.strip()
    if len(name) > 31:
        name = name[:31]
    return name or "source_unknown"




def parse_bool_env(name: str, default: bool) -> bool:
    """Парсинг bool env: поддерживает 1/0, true/false, yes/no, y/n, on/off."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def parse_int_list(s: str, default_vals):
    """Парсинг списка чисел из строки env ('1,3,5,10'). Если не получилось - вернём дефолт."""
    try:
        vals = []
        for tok in re.split(r"[,\s;]+", s.strip()):
            if tok:
                vals.append(int(tok))
        return vals if vals else list(default_vals)
    except Exception:
        return list(default_vals)


def percentile_dict(values):
    """p50/p90/p95/p99 для массива значений (секунды)."""
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def calc_time_stats(values):
    """
    Среднее и перцентили p50/p95 для времени (сек).
    Округляем до 4 знаков после запятой - важны короткие этапы (~0.0032 c).
    """
    if not values:
        return None
    arr = np.array(values, dtype=float)
    avg = float(np.mean(arr))
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    return {"avg": round(avg, 4), "p50": round(p50, 4), "p95": round(p95, 4)}


def compute_rank_metrics(found_ranks, ks):
    """
    found_ranks: список рангов (int>=1) или None, длиной = числу запросов.
    ks: список k.
    Возвращает dict с recall@k (в %), mrr@k, ndcg@k.
    """
    n = len(found_ranks)
    recalls, mrrs, ndcgs = {}, {}, {}
    for k in ks:
        hits = 0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        for r in found_ranks:
            if r is not None and r <= k:
                hits += 1
                mrr_sum += 1.0 / r
                ndcg_sum += 1.0 / math.log2(r + 1.0)
        recalls[str(k)] = round(hits * 100.0 / n, 2) if n > 0 else 0.0
        mrrs[str(k)] = round(mrr_sum / n, 4) if n > 0 else 0.0
        ndcgs[str(k)] = round(ndcg_sum / n, 4) if n > 0 else 0.0
    return {"recall_at_k": recalls, "mrr_at_k": mrrs, "ndcg_at_k": ndcgs}


def compute_margin_stats(margins, top1_correct_flags):
    """
    margins: список float margin@1 (top1_score - top2_score), может содержать None -> игнорируем.
    top1_correct_flags: список 0/1, если top1_id == target_id.
    """
    m = [float(x) for x in margins if x is not None]
    if not m:
        return {
            "count": 0, "avg": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0,
            "corr_with_top1_correct": None
        }
    arr = np.array(m, dtype=float)
    avg = float(np.mean(arr))
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))
    if len(m) >= 2:
        y = np.array(top1_correct_flags[:len(m)], dtype=float)
        try:
            corr = float(np.corrcoef(arr, y)[0, 1])
        except Exception:
            corr = None
    else:
        corr = None
    return {
        "count": len(m),
        "avg": round(avg, 6),
        "p50": round(p50, 6),
        "p90": round(p90, 6),
        "p95": round(p95, 6),
        "corr_with_top1_correct": None if corr is None else round(corr, 4)
    }


def get_by_dotted_path(data, dotted_path):
    """Возвращает под-словарь по точечному пути ('a.b.c'). Нет ключа - None."""
    if dotted_path is None:
        return None
    cur = data
    for key in dotted_path.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _extract_api_field(info_obj, key):
    """
    Мягко достаём поле key из словаря info_obj.
    Смотрим сам объект и типовые вложенности info/config/params/settings.
    """
    if not isinstance(info_obj, dict) or not key:
        return None
    if key in info_obj:
        return info_obj.get(key)
    for subkey in ("info", "config", "params", "settings"):
        sub = info_obj.get(subkey)
        if isinstance(sub, dict) and key in sub:
            return sub.get(key)
    return None


def _boolish(v):
    """True/False/строка/число -> красивый ON/OFF/строковое."""
    if isinstance(v, bool):
        return "ON" if v else "OFF"
    if isinstance(v, (int, float)):
        return "ON" if v != 0 else "OFF"
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return "ON"
        if s in ("false", "0", "no", "n", "off"):
            return "OFF"
    return str(v)


def rank_sort_key(r):
    """None уводим в «худшие», чтобы сортировать проблемных выше."""
    return (10**9 if r is None else r)


# -------------------------- Рисовалки (общие) --------------------------------

def nice_bar(ax, categories, values, title, counts=None, ylim=(0, 100)):
    """Красочный bar-plot: палитра, подписи, сетка (без DeprecationWarning)."""
    palette = get_cmap('tab20')
    colors = [palette(i / max(1, len(categories) - 1)) for i in range(len(categories))]
    ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    if counts is None:
        counts = [None] * len(values)
    for i, (v, c) in enumerate(zip(values, counts)):
        label = f"{v:.1f}%"
        if c is not None:
            label += f"\n({c})"
        ax.text(i, min(ylim[1] - 1, v + (ylim[1] * 0.02)), label,
                ha='center', va='bottom', fontsize=9)


def draw_table(ax, title, rows, headers, max_chars=60):
    """
    Рисует таблицу (матрица строковых значений) в оси ax.
    Строк «rows» не очень много (мы ограничиваем top-N), текст query переносится по словам.
    """
    ax.set_title(title, fontsize=12, pad=6)
    ax.axis("off")

    def wrap_cell(s):
        if s is None:
            return ""
        s = str(s)
        if len(s) <= max_chars:
            return s
        return "\n".join(wrap(s, max_chars))

    table_data = []
    for r in rows:
        table_data.append([wrap_cell(x) for x in r])

    the_table = ax.table(cellText=table_data, colLabels=headers,
                         loc="upper center", cellLoc="left")
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)
    the_table.scale(1, 1.2)
    # делаем заголовки жирными
    for (row, col), cell in the_table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
def build_timing_overlay_text(*, total_avg, total_latency_dict,
                              emb_stats, vec_stats, lex_stats,
                              ce_avg=None, response_avg=None,
                              full_task_avg=None, avg_chars=None):
    """
    Блок текста для итоговых графиков:
    - total avg + p50/p95
    - embedding avg/p50/p95
    - dense avg/p50/p95
    - lex avg/p50/p95
    - ce avg (если есть)
    - response avg (если есть)
    - total_search_task avg (если есть)
    - avg query length
    Все времена с точностью .4f.
    """
    lines = []
    total_p50 = (total_latency_dict or {}).get("p50", 0.0)
    total_p95 = (total_latency_dict or {}).get("p95", 0.0)
    lines.append(
        "total avg {:.4f}s  p50/p95 {:.4f}/{:.4f}".format(
            total_avg, float(total_p50), float(total_p95)
        )
    )
    if emb_stats:
        lines.append("emb   avg {:.4f}s  p50/p95 {:.4f}/{:.4f}".format(
            emb_stats["avg"], emb_stats["p50"], emb_stats["p95"]
        ))
    if vec_stats:
        lines.append("dense avg {:.4f}s  p50/p95 {:.4f}/{:.4f}".format(
            vec_stats["avg"], vec_stats["p50"], vec_stats["p95"]
        ))
    if lex_stats:
        lines.append("lex   avg {:.4f}s  p50/p95 {:.4f}/{:.4f}".format(
            lex_stats["avg"], lex_stats["p50"], lex_stats["p95"]
        ))
    if ce_avg is not None:
        lines.append("ce    avg {:.4f}s".format(ce_avg))
    if response_avg is not None:
        lines.append("resp  avg {:.4f}s".format(response_avg))
    if full_task_avg is not None:
        lines.append("total_search_task avg {:.4f}s".format(full_task_avg))
    if avg_chars is not None:
        lines.append(f"avg query length: {avg_chars:.1f} chars")
    return "\n".join(lines)

def format_metrics_overlay(metrics, latency, margin_stats, ks_label):
    """
    Строки с MRR/nDCG/margin для отображения на графике.
    Возвращаем (mrr_line, ndcg_line, lat_line, m_line).
    """
    mrr_vals = [metrics.get("mrr_at_k", {}).get(k, 0.0) for k in metrics.get("mrr_at_k", {}).keys()]
    ndcg_vals = [metrics.get("ndcg_at_k", {}).get(k, 0.0) for k in metrics.get("ndcg_at_k", {}).keys()]
    mrr_line = "MRR@{}: {}".format(ks_label, "/".join(f"{v:.3f}" for v in mrr_vals)) if mrr_vals else "MRR: n/a"
    ndcg_line = "nDCG@{}: {}".format(ks_label, "/".join(f"{v:.3f}" for v in ndcg_vals)) if ndcg_vals else "nDCG: n/a"
    lat_line = "p50/p95: {}/{}s".format(
        f"{(latency or {}).get('p50', 0.0):.2f}",
        f"{(latency or {}).get('p95', 0.0):.2f}",
    )
    if margin_stats and margin_stats.get("count", 0) > 0:
        m_line = "margin@1 avg: {:.4f} (corr: {})".format(
            margin_stats.get("avg", 0.0),
            "n/a" if margin_stats.get("corr_with_top1_correct") is None
            else f"{margin_stats['corr_with_top1_correct']:.2f}"
        )
    else:
        m_line = "margin@1: n/a"
    return mrr_line, ndcg_line, lat_line, m_line

def main():
    # Загрузка переменных окружения
    dotenv_path = Path(os.getenv("DOTENV_PATH", ".env_quality"))
    try:
        load_dotenv(dotenv_path)
    except Exception as e:  # pragma: no cover
        print(f"Ошибка: не удалось загрузить файл окружения: {e}")
        return

    # Основные параметры тестов
    test_data_path = os.getenv("TEST_DATA_PATH")
    col_query = os.getenv("COL_QUERY", "test_query")
    col_target = os.getenv("COL_TARGET_ID", "target_id")
    col_source = os.getenv("COL_SOURCE", "source")

    # Пути в ответах API
    results_path = os.getenv("RESULTS_PATH", "info.results")
    metrics_path = os.getenv("METRICS_PATH", "info.metrics")
    intermediate_path = os.getenv("INTERMEDIATE_RESULTS_PATH", "info.intermediate_results")

    # Поля результата
    result_id_field = os.getenv("RESULT_ID_FIELD", "ext_id")
    result_question_field = os.getenv("RESULT_QUESTION_FIELD", "question")
    result_answer_field = os.getenv("RESULT_ANSWER_FIELD", "answer")
    additional_fields_str = os.getenv("ADDITIONAL_RESULT_FIELDS", "")

    # Тайминги из метрик
    duration_field = os.getenv("DURATION_FIELD", "total_time")
    embedding_time_field = os.getenv("EMBEDDING_TIME_FIELD", "embedding_time")
    vector_time_field = os.getenv("VECTOR_SEARCH_TIME_FIELD", "vector_search_time")
    lexical_time_field = os.getenv("LEXICAL_SEARCH_TIME_FIELD", "lexical_search_time")
    ce_time_field = os.getenv("CE_SEARCH_TIME_FIELD", "cross_encoder_time")
    total_task_search_time_field = os.getenv("TOTAL_TASK_SEARCH_TIME_FIELD", "full_search_task_time")

    # Параметры запроса/выгрузки
    top_k = int(os.getenv("TOP_K", "15"))
    results_limit = int(os.getenv("RESULTS_LIMIT", str(top_k)))
    output_base = os.getenv("OUTPUT_BASENAME", "bench_results")
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:5155")
    api_search_url = os.getenv("API_SEARCH_URL", "/hybrid-search/search")
    api_info_url = os.getenv("API_INFO_URL", "/hybrid-search/info/")
    API_INFO_INTERVAL = float(os.getenv("API_INFO_INTERVAL", "0.5"))
    API_INFO_MAX_TIME = float(os.getenv("API_INFO_MAX_TIME", "60"))
    search_use_cache = parse_bool_env("SEARCH_USE_CACHE", False)
    metrics_enable = parse_bool_env("METRICS_ENABLE", True)
    show_intermediate_results = parse_bool_env("SHOW_INTERMEDIATE_RESULTS", True)
    presearch_field = (os.getenv("PRESEARCH_FIELD", "") or "").strip()

    # Доп. тестовые поля
    col_query_source = os.getenv("COL_QUERY_SOURCE", "test_query_source")
    col_answer = os.getenv("COL_ANSWER", "test_answer")

    # Метрики/диагностика
    METRICS_KS = parse_int_list(os.getenv("METRICS_KS", "1,3,5,10"), [1, 3, 5, 10])
    ks_label = ",".join(map(str, METRICS_KS))
    MARGIN_SCORE_FIELD = os.getenv("MARGIN_SCORE_FIELD", "final_score")
    DIAG_TOPN = int(os.getenv("DIAG_TOPN", "50"))
    AVG_SORTED_CANDIDATES = int(os.getenv("AVG_SORTED_CANDIDATES", "5"))

    # Параметры прогона из API (мягкий сбор на первом же успешном ответе)
    HYBRID_W_CE_FIELD = os.getenv("HYBRID_W_CE_FIELD", "hybrid_w_ce")
    HYBRID_W_DENSE_FIELD = os.getenv("HYBRID_W_DENSE_FIELD") or os.getenv("HYBRID_W_DENSE", "hybrid_w_dense")
    HYBRID_W_LEX_FIELD = os.getenv("HYBRID_W_LEX_FIELD") or os.getenv("HYBRID_W_LEX", "hybrid_w_lex")

    ENCODER_MODEL_FIELD = os.getenv("ENCODER_MODEL_FIELD", "encoder_model")                    # str
    RERANKER_MODEL_FIELD = os.getenv("RERANKER_MODEL_FIELD", "reranker_model")                # str
    HYBRID_DENSE_TOP_K_FIELD = os.getenv("HYBRID_DENSE_TOP_K_FIELD", "hybrid_dense_top_k")   # str|number
    HYBRID_LEX_TOP_K_FIELD = os.getenv("HYBRID_LEX_TOP_K_FIELD", "hybrid_lex_top_k")         # str|number
    HYBRID_TOP_K_FIELD = os.getenv("HYBRID_TOP_K_FIELD", "hybrid_top_k")                     # str|number
    RERANKER_ENABLE_FLAG = os.getenv("RERANKER_ENABLE_FLAG", "reranker_enabled")             # bool
    BM25_ENABLE_FLAG = os.getenv("BM25_ENABLE_FLAG", "bm_25_enabled")                        # bool
    OPENSEARCH_ENABLE_FLAG = os.getenv("OPENSEARCH_ENABLE_FLAG", "open_search_enabled")      # bool

    # Проверки входных данных
    if not test_data_path:
        print("Ошибка: не указан путь к файлу с тестовыми запросами (TEST_DATA_PATH).")
        return
    try:
        df = pd.read_excel(test_data_path)
    except Exception as e:
        print(f"Ошибка: не удалось загрузить файл тестовых данных Excel: {e}")
        return
    missing_cols = [col for col in [col_query, col_target, col_source] if col not in df.columns]
    if missing_cols:
        print(f"Ошибка: в тестовом наборе данных отсутствуют необходимые столбцы: {missing_cols}")
        return
    if df.empty:
        print("Ошибка: тестовый набор данных пустой.")
        return

    # Доп. поля результатов
    additional_fields = []
    if additional_fields_str:
        fields = re.split(r"[,\s;]+", additional_fields_str)
        additional_fields = [f for f in fields if f]

    # ------------------- Аккумуляторы основной статистики --------------------
    query_results = []  # для JSON и XLSX
    # счётчики позиционных попаданий
    count_at_1 = count_at_2 = count_at_3 = count_at_4 = count_at_5 = 0
    count_above_5 = count_above_10 = 0
    count_not_found = 0

    # тайминги
    times_overall = []              # total_time
    embedding_times_overall = []    # embedding_time
    vector_times_overall = []       # vector_search_time (dense)
    lexical_times_overall = []      # lexical_search_time
    ce_times_overall = []           # cross_encoder_time
    response_times_overall = []     # POST /search -> status=done
    full_task_times_overall = []   # total search task time (sec)

    # длины запросов в символах
    query_lengths_overall = []

    # для списков найденных по конкретным позициям
    overall_found_items_by_rank = {1: [], 2: [], 3: [], 4: [], 5: [], "gt5": [], "gt10": []}
    overall_not_found_items = []
    # ранги для метрик
    found_ranks_overall, margins_overall, top1_correct_overall = [], [], []

    # По источникам
    stats_by_source = {}  # source -> агрегаты

    def ensure_source(src: str):
        if src not in stats_by_source:
            stats_by_source[src] = {
                "count": 0,
                # позиционные
                "found_at_1": 0, "found_at_2": 0, "found_at_3": 0, "found_at_4": 0, "found_at_5": 0,
                "found_above_5": 0, "found_above_10": 0, "not_found": 0,
                "found_items_by_rank": {1: [], 2: [], 3: [], 4: [], 5: [], "gt5": [], "gt10": []},
                "not_found_items": [],
                "found_ranks": [], "margins": [], "top1_correct": [],
                # тайминги списками
                "times": [], "embedding_times": [], "vector_search_times": [], "lexical_search_times": [],
                "ce_times": [], "response_times": [], "full_task_times": [],
                "duration_sum": 0.0,
                "query_lengths": []
            }

    # ---------- Аккумуляторы промежуточных результатов (dense/lex/ce) --------
    def make_stage_agg():
        return {
            "count": 0,
            "found_at_1": 0, "found_at_2": 0, "found_at_3": 0, "found_at_4": 0, "found_at_5": 0,
            "found_above_5": 0, "found_above_10": 0, "not_found": 0,
            "found_items_by_rank": {1: [], 2: [], 3: [], 4: [], 5: [], "gt5": [], "gt10": []},
            "not_found_items": [],
            "found_ranks": [], "margins": [], "top1_correct": []
        }

    stage_names = ["dense", "lex", "fusion", "ce"]
    intermediate_stats = {name: {"overall": make_stage_agg(), "by_source": {}} for name in stage_names}

    # ------------------- Параметры прогона (из API, один раз) ----------------
    api_run_info = {
        "encoder_model": None, "reranker_model": None,
        "reranker_enabled": None, "bm25_enabled": None, "open_search_enabled": None,
        "hybrid_w_ce": None, "hybrid_w_dense": None, "hybrid_w_lex": None,
        "hybrid_dense_top_k": None, "hybrid_lex_top_k": None, "hybrid_top_k": None,
        "hybrid_fusion_mode": None, "hybrid_rrf_k": None, "short_mode_applied": None,
        "hybrid_score_final_mode": None, "from_cache": None, "search_use_cache": None,
        "show_intermediate_results": None, "presearch_enabled": None, "presearch_field": None
    }

    def _maybe_set_run_param(norm_key, value):
        if value is None:
            return
        if api_run_info.get(norm_key) is None:
            api_run_info[norm_key] = value

    # --------------------------- Основной цикл --------------------------------
    mpl.rcParams["figure.dpi"] = 300  # высокое разрешение при сохранении
    with tqdm(total=len(df), desc="Benchmarking", unit="q", dynamic_ncols=True, mininterval=0.2, leave=True) as pbar:
        for idx, row in df.iterrows():
            user_query = str(row[col_query])
            expected_id = str(row[col_target])
            source_name = str(row[col_source]) if pd.notna(row[col_source]) else ""
            test_query_source_val = safe_get(row, col_query_source, "")
            test_answer_val = safe_get(row, col_answer, "")

            ensure_source(source_name)
            stats_by_source[source_name]["count"] += 1

            qlen = len(user_query)
            query_lengths_overall.append(qlen)
            stats_by_source[source_name]["query_lengths"].append(qlen)

            pbar.set_description_str(f"[{idx+1}/{len(df)}] src={source_name[:20]}")
            pbar.set_postfix({"status": "search"})
            pbar.refresh()

            # --- измеряем response_time: от перед POST /search до получения status=done
            response_start = time.time()

            # --- /hybrid-search/search ---
            search_url = f"{api_base_url.rstrip('/')}{api_search_url}"
            try:
                payload = {
                    "query": user_query,
                    "top_k": top_k,
                    "search_use_cache": search_use_cache,
                    "metrics_enable": metrics_enable,
                    "show_intermediate_results": show_intermediate_results,
                }
                if presearch_field:
                    payload["presearch"] = {"field": presearch_field}
                resp = requests.post(search_url, json=payload)
            except Exception as e:
                pbar.clear()
                print(f"Ошибка при выполнении /search для запроса '{user_query}': {e}")
                return

            if resp.status_code != 202:
                try:
                    error_info = resp.json()
                except Exception:
                    error_info = resp.text
                pbar.clear()
                print(f"Ошибка: /search вернул {resp.status_code} для '{user_query}'. Детали: {error_info}")
                return

            try:
                search_data = resp.json()
            except Exception as e:
                pbar.clear()
                print(f"Ошибка: не удалось разобрать JSON-ответ /search для '{user_query}': {e}")
                return

            ticket_id = search_data.get("task_id")
            if not ticket_id:
                pbar.clear()
                print(f"Ошибка: в ответе /search отсутствует 'task_id' для '{user_query}'. Ответ: {search_data}")
                return

            # --- /hybrid-search/info/{ticket_id} (polling) ---
            info_url = f"{api_base_url.rstrip('/')}{api_info_url}{ticket_id}"
            polls = 0
            while True:
                try:
                    info_resp = requests.get(info_url)
                except Exception as e:
                    pbar.clear()
                    print(f"Ошибка: сбой при обращении к /info для '{user_query}', ticket {ticket_id}: {e}")
                    return

                if info_resp.status_code != 200:
                    try:
                        error_info = info_resp.json()
                    except Exception:
                        error_info = info_resp.text
                    pbar.clear()
                    print(f"Ошибка: /info вернул {info_resp.status_code} для '{user_query}', ticket {ticket_id}. Детали: {error_info}")
                    return

                try:
                    info_data = info_resp.json()
                except Exception as e:
                    pbar.clear()
                    print(f"Ошибка: не удалось разобрать JSON-ответ /info для '{user_query}', ticket {ticket_id}: {e}")
                    return

                status = info_data.get("status")
                if status is None:
                    pbar.clear()
                    print(f"Ошибка: в /info нет поля 'status' для '{user_query}', ticket {ticket_id}. Ответ: {info_data}")
                    return

                status = status.lower()
                polls += 1
                elapsed = time.time() - response_start
                pbar.set_postfix({"status": status, "polls": polls, "elapsed": f"{elapsed:.1f}s"})
                pbar.refresh()

                if status == "done":
                    break
                if status == "failed":
                    fail_reason = info_data.get("answer") or info_data.get("info") or "неизвестная причина"
                    pbar.clear()
                    print(f"Ошибка: задача поиска не выполнена (failed) для '{user_query}', ticket {ticket_id}. Причина: {fail_reason}")
                    return
                if elapsed > API_INFO_MAX_TIME:
                    pbar.clear()
                    print(f"Ошибка: превышено ожидание результатов для '{user_query}', ticket {ticket_id}.")
                    return

                time.sleep(API_INFO_INTERVAL)

            response_time = time.time() - response_start
            response_times_overall.append(response_time)
            stats_by_source[source_name]["response_times"].append(response_time)

            # --- финальные результаты ---
            # достаём список по RESULTS_PATH
            results_data = info_data
            try:
                for key in results_path.split('.'):
                    results_data = results_data[key]
            except Exception:
                pbar.clear()
                print(f"Ошибка: не найден массив результатов по пути {results_path} для '{user_query}'.")
                return
            if not isinstance(results_data, list):
                pbar.clear()
                print(f"Ошибка: по пути {results_path} ожидался список, получено: {type(results_data)} для '{user_query}'.")
                return
            top_results = results_data[:results_limit]

            # --- метрики/конфиг ---
            metrics_data = get_by_dotted_path(info_data, metrics_path)
            if not isinstance(metrics_data, dict):
                metrics_data = {}

            def _to_float_safe(v):
                try: return float(v)
                except Exception: return None

            # тайминги (из metrics_path)
            duration_f = _to_float_safe(metrics_data.get(duration_field, info_data.get(duration_field))) or 0.0
            embedding_f = _to_float_safe(metrics_data.get(embedding_time_field))
            vector_f    = _to_float_safe(metrics_data.get(vector_time_field))
            lexical_f   = _to_float_safe(metrics_data.get(lexical_time_field))
            ce_f        = _to_float_safe(metrics_data.get(ce_time_field))
            full_task_ms = _to_float_safe(metrics_data.get(total_task_search_time_field))
            full_task_sec = (full_task_ms / 1000.0) if full_task_ms is not None else None

            # аккумуляторы времен
            times_overall.append(duration_f)
            stats_by_source[source_name]["times"].append(duration_f)
            stats_by_source[source_name]["duration_sum"] += duration_f
            if full_task_sec is not None:
                full_task_times_overall.append(round(full_task_sec, 4))
                stats_by_source[source_name]["full_task_times"].append(round(full_task_sec, 4))

            if embedding_f is not None:
                embedding_times_overall.append(embedding_f)
                stats_by_source[source_name]["embedding_times"].append(embedding_f)
            if vector_f is not None:
                vector_times_overall.append(vector_f)
                stats_by_source[source_name]["vector_search_times"].append(vector_f)
            if lexical_f is not None:
                lexical_times_overall.append(lexical_f)
                stats_by_source[source_name]["lexical_search_times"].append(lexical_f)
            if ce_f is not None:
                ce_times_overall.append(ce_f)
                stats_by_source[source_name]["ce_times"].append(ce_f)

            # запоминаем параметры прогона (один раз, мягко)
            _maybe_set_run_param("encoder_model",        _extract_api_field(metrics_data, ENCODER_MODEL_FIELD))
            _maybe_set_run_param("reranker_model",       _extract_api_field(metrics_data, RERANKER_MODEL_FIELD))
            _maybe_set_run_param("reranker_enabled",     _extract_api_field(metrics_data, RERANKER_ENABLE_FLAG))
            _maybe_set_run_param("bm25_enabled",         _extract_api_field(metrics_data, BM25_ENABLE_FLAG))
            _maybe_set_run_param("open_search_enabled",  _extract_api_field(metrics_data, OPENSEARCH_ENABLE_FLAG))
            _maybe_set_run_param("hybrid_w_ce",          _extract_api_field(metrics_data, HYBRID_W_CE_FIELD))
            _maybe_set_run_param("hybrid_w_dense",       _extract_api_field(metrics_data, HYBRID_W_DENSE_FIELD))
            _maybe_set_run_param("hybrid_w_lex",         _extract_api_field(metrics_data, HYBRID_W_LEX_FIELD))
            _maybe_set_run_param("hybrid_dense_top_k",   _extract_api_field(metrics_data, HYBRID_DENSE_TOP_K_FIELD))
            _maybe_set_run_param("hybrid_lex_top_k",     _extract_api_field(metrics_data, HYBRID_LEX_TOP_K_FIELD))
            _maybe_set_run_param("hybrid_top_k",         _extract_api_field(metrics_data, HYBRID_TOP_K_FIELD))
            _maybe_set_run_param("hybrid_fusion_mode",   _extract_api_field(metrics_data, "hybrid_fusion_mode"))
            _maybe_set_run_param("hybrid_rrf_k",         _extract_api_field(metrics_data, "hybrid_rrf_k"))
            _maybe_set_run_param("short_mode_applied",   _extract_api_field(metrics_data, "short_mode_applied"))
            _maybe_set_run_param("hybrid_score_final_mode", _extract_api_field(metrics_data, "hybrid_score_final_mode"))
            _maybe_set_run_param("from_cache",           _extract_api_field(metrics_data, "from_cache"))
            _maybe_set_run_param("search_use_cache",     _extract_api_field(metrics_data, "search_use_cache"))
            _maybe_set_run_param("show_intermediate_results", _extract_api_field(metrics_data, "show_intermediate_results"))
            _maybe_set_run_param("presearch_enabled",    _extract_api_field(metrics_data, "presearch_enabled"))
            _maybe_set_run_param("presearch_field",      _extract_api_field(metrics_data, "presearch_field"))

            # --- попадание ожидаемого ID (финальная выдача) ---
            found_rank = None
            for rank, res in enumerate(top_results, start=1):
                res_id = str(res.get(result_id_field, ""))
                if res_id == expected_id:
                    found_rank = rank
                    break

            top1_id = str(top_results[0].get(result_id_field, "")) if top_results else ""
            top1_q  = top_results[0].get(result_question_field, "") if top_results else ""
            top1_ans = top_results[0].get(result_answer_field, "") if top_results else ""

            margin_val = None
            try:
                if len(top_results) >= 2:
                    s1 = top_results[0].get(MARGIN_SCORE_FIELD, None)
                    s2 = top_results[1].get(MARGIN_SCORE_FIELD, None)
                    if s1 is not None and s2 is not None:
                        margin_val = float(s1) - float(s2)
            except Exception:
                margin_val = None

            # --- счётчики по позициям (финальная выдача) ---
            if found_rank == 1:
                count_at_1 += 1
                stats_by_source[source_name]["found_at_1"] += 1
                item = {"target_id": expected_id, "test_query": user_query,
                        "test_query_source": test_query_source_val, "test_answer": test_answer_val}
                overall_found_items_by_rank[1].append(item)
                stats_by_source[source_name]["found_items_by_rank"][1].append(item)
            elif found_rank == 2:
                count_at_2 += 1
                stats_by_source[source_name]["found_at_2"] += 1
                item = {"target_id": expected_id, "test_query": user_query,
                        "test_query_source": test_query_source_val, "test_answer": test_answer_val}
                overall_found_items_by_rank[2].append(item)
                stats_by_source[source_name]["found_items_by_rank"][2].append(item)
            elif found_rank == 3:
                count_at_3 += 1
                stats_by_source[source_name]["found_at_3"] += 1
                item = {"target_id": expected_id, "test_query": user_query,
                        "test_query_source": test_query_source_val, "test_answer": test_answer_val}
                overall_found_items_by_rank[3].append(item)
                stats_by_source[source_name]["found_items_by_rank"][3].append(item)
            elif found_rank == 4:
                count_at_4 += 1
                stats_by_source[source_name]["found_at_4"] += 1
                item = {"target_id": expected_id, "test_query": user_query,
                        "test_query_source": test_query_source_val, "test_answer": test_answer_val}
                overall_found_items_by_rank[4].append(item)
                stats_by_source[source_name]["found_items_by_rank"][4].append(item)
            elif found_rank == 5:
                count_at_5 += 1
                stats_by_source[source_name]["found_at_5"] += 1
                item = {"target_id": expected_id, "test_query": user_query,
                        "test_query_source": test_query_source_val, "test_answer": test_answer_val}
                overall_found_items_by_rank[5].append(item)
                stats_by_source[source_name]["found_items_by_rank"][5].append(item)

            if found_rank is not None and found_rank > 5:
                count_above_5 += 1
                stats_by_source[source_name]["found_above_5"] += 1
                item = {"target_id": expected_id, "test_query": user_query,
                        "test_query_source": test_query_source_val, "test_answer": test_answer_val}
                overall_found_items_by_rank["gt5"].append(item)
                stats_by_source[source_name]["found_items_by_rank"]["gt5"].append(item)
                if found_rank > 10:
                    count_above_10 += 1
                    stats_by_source[source_name]["found_above_10"] += 1
                    overall_found_items_by_rank["gt10"].append(item)
                    stats_by_source[source_name]["found_items_by_rank"]["gt10"].append(item)

            if found_rank is None:
                count_not_found += 1
                stats_by_source[source_name]["not_found"] += 1
                # совместимость со старой трактовкой: если лимиты большие, учитываем >5/>10
                if results_limit >= 5:
                    count_above_5 += 1
                    stats_by_source[source_name]["found_above_5"] += 1
                if results_limit >= 10:
                    count_above_10 += 1
                    stats_by_source[source_name]["found_above_10"] += 1
                nf_item = {"target_id": expected_id, "test_query": user_query,
                           "test_query_source": test_query_source_val, "test_answer": test_answer_val}
                overall_not_found_items.append(nf_item)
                stats_by_source[source_name]["not_found_items"].append(nf_item)

            found_ranks_overall.append(found_rank)
            margins_overall.append(margin_val)
            top1_correct_overall.append(1 if (top1_id == expected_id) else 0)
            stats_by_source[source_name]["found_ranks"].append(found_rank)
            stats_by_source[source_name]["margins"].append(margin_val)
            stats_by_source[source_name]["top1_correct"].append(1 if (top1_id == expected_id) else 0)

            # --- запись результата запроса для JSON/XLSX ---
            result_entry = {
                "query": user_query, "expected_id": expected_id, "source": source_name, "results": [],
                "search_time": float(duration_f),
                "response_time": float(response_time),
                "total_search_task_time": round(full_task_sec, 4) if full_task_sec is not None else None,
                "expected_found_rank": found_rank if found_rank is not None else None,
                "_top1": {"id": top1_id, "question": top1_q, "answer": top1_ans, "test_answer": test_answer_val},
                "_times": {  # сохраняем по-запросно для диагностики/сортировок
                    "embedding_time": embedding_f,
                    "vector_search_time": vector_f,
                    "lexical_search_time": lexical_f,
                    "cross_encoder_time": ce_f
                }
            }
            if margin_val is not None:
                result_entry["margin_at_1"] = margin_val

            for rank, res in enumerate(top_results, start=1):
                res_id = str(res.get(result_id_field, ""))
                res_q = res.get(result_question_field, "")
                result_item = {"rank": rank, result_id_field: res_id, result_question_field: res_q}
                for field in additional_fields:
                    if field in res:
                        result_item[field] = res[field]
                if result_answer_field in res:
                    result_item[result_answer_field] = res[result_answer_field]
                result_entry["results"].append(result_item)

            # --- промежуточные результаты: dense/lex/ce (мягко) ---
            intermediate = get_by_dotted_path(info_data, intermediate_path)
            if isinstance(intermediate, dict):
                for stage in stage_names:
                    stage_list = intermediate.get(stage, None)
                    if not isinstance(stage_list, list) or len(stage_list) == 0:
                        continue
                    top_stage = stage_list[:results_limit]
                    # ранк таргета
                    s_found_rank = None
                    for rnk, rr in enumerate(top_stage, start=1):
                        rid = str(rr.get(result_id_field, ""))
                        if rid == expected_id:
                            s_found_rank = rnk
                            break
                    # margin@1
                    s_margin = None
                    try:
                        if len(top_stage) >= 2:
                            s1 = top_stage[0].get(MARGIN_SCORE_FIELD, None)
                            s2 = top_stage[1].get(MARGIN_SCORE_FIELD, None)
                            if s1 is not None and s2 is not None:
                                s_margin = float(s1) - float(s2)
                    except Exception:
                        s_margin = None
                    s_top1_id = str(top_stage[0].get(result_id_field, "")) if top_stage else ""

                    # overall
                    agg_over = intermediate_stats[stage]["overall"]
                    agg_over["count"] += 1
                    agg_over["found_ranks"].append(s_found_rank)
                    agg_over["margins"].append(s_margin)
                    agg_over["top1_correct"].append(1 if (s_top1_id == expected_id) else 0)

                    # позиции
                    if s_found_rank == 1:
                        agg_over["found_at_1"] += 1
                        agg_over["found_items_by_rank"][1].append({
                            "target_id": expected_id, "test_query": user_query,
                            "test_query_source": test_query_source_val, "test_answer": test_answer_val
                        })
                    elif s_found_rank == 2:
                        agg_over["found_at_2"] += 1
                        agg_over["found_items_by_rank"][2].append({
                            "target_id": expected_id, "test_query": user_query,
                            "test_query_source": test_query_source_val, "test_answer": test_answer_val
                        })
                    elif s_found_rank == 3:
                        agg_over["found_at_3"] += 1
                        agg_over["found_items_by_rank"][3].append({
                            "target_id": expected_id, "test_query": user_query,
                            "test_query_source": test_query_source_val, "test_answer": test_answer_val
                        })
                    elif s_found_rank == 4:
                        agg_over["found_at_4"] += 1
                        agg_over["found_items_by_rank"][4].append({
                            "target_id": expected_id, "test_query": user_query,
                            "test_query_source": test_query_source_val, "test_answer": test_answer_val
                        })
                    elif s_found_rank == 5:
                        agg_over["found_at_5"] += 1
                        agg_over["found_items_by_rank"][5].append({
                            "target_id": expected_id, "test_query": user_query,
                            "test_query_source": test_query_source_val, "test_answer": test_answer_val
                        })
                    total_stage_shown = min(len(stage_list), results_limit)
                    if s_found_rank is None:
                        agg_over["not_found"] += 1
                        if total_stage_shown > 5:
                            agg_over["found_above_5"] += 1
                        if total_stage_shown > 10:
                            agg_over["found_above_10"] += 1
                        agg_over["not_found_items"].append({
                            "target_id": expected_id, "test_query": user_query,
                            "test_query_source": test_query_source_val, "test_answer": test_answer_val
                        })
                    elif s_found_rank > 5:
                        agg_over["found_above_5"] += 1
                        agg_over["found_items_by_rank"]["gt5"].append({
                            "target_id": expected_id, "test_query": user_query,
                            "test_query_source": test_query_source_val, "test_answer": test_answer_val
                        })
                        if s_found_rank > 10:
                            agg_over["found_above_10"] += 1
                            agg_over["found_items_by_rank"]["gt10"].append({
                                "target_id": expected_id, "test_query": user_query,
                                "test_query_source": test_query_source_val, "test_answer": test_answer_val
                            })

                    # by source
                    if source_name not in intermediate_stats[stage]["by_source"]:
                        intermediate_stats[stage]["by_source"][source_name] = make_stage_agg()
                    agg_src = intermediate_stats[stage]["by_source"][source_name]
                    agg_src["count"] += 1
                    agg_src["found_ranks"].append(s_found_rank)
                    agg_src["margins"].append(s_margin)
                    agg_src["top1_correct"].append(1 if (s_top1_id == expected_id) else 0)

                    if s_found_rank == 1:
                        agg_src["found_at_1"] += 1
                        agg_src["found_items_by_rank"][1].append({
                            "target_id": expected_id, "test_query": user_query,
                            "test_query_source": test_query_source_val, "test_answer": test_answer_val
                        })
                    elif s_found_rank == 2:
                        agg_src["found_at_2"] += 1
                        agg_src["found_items_by_rank"][2].append({
                            "target_id": expected_id, "test_query": user_query,
                            "test_query_source": test_query_source_val, "test_answer": test_answer_val
                        })
                    elif s_found_rank == 3:
                        agg_src["found_at_3"] += 1
                        agg_src["found_items_by_rank"][3].append({
                            "target_id": expected_id, "test_query": user_query,
                            "test_query_source": test_query_source_val, "test_answer": test_answer_val
                        })
                    elif s_found_rank == 4:
                        agg_src["found_at_4"] += 1
                        agg_src["found_items_by_rank"][4].append({
                            "target_id": expected_id, "test_query": user_query,
                            "test_query_source": test_query_source_val, "test_answer": test_answer_val
                        })
                    elif s_found_rank == 5:
                        agg_src["found_at_5"] += 1
                        agg_src["found_items_by_rank"][5].append({
                            "target_id": expected_id, "test_query": user_query,
                            "test_query_source": test_query_source_val, "test_answer": test_answer_val
                        })
                    if s_found_rank is None:
                        agg_src["not_found"] += 1
                        if total_stage_shown > 5:
                            agg_src["found_above_5"] += 1
                        if total_stage_shown > 10:
                            agg_src["found_above_10"] += 1
                        agg_src["not_found_items"].append({
                            "target_id": expected_id, "test_query": user_query,
                            "test_query_source": test_query_source_val, "test_answer": test_answer_val
                        })
                    elif s_found_rank > 5:
                        agg_src["found_above_5"] += 1
                        agg_src["found_items_by_rank"]["gt5"].append({
                            "target_id": expected_id, "test_query": user_query,
                            "test_query_source": test_query_source_val, "test_answer": test_answer_val
                        })
                        if s_found_rank > 10:
                            agg_src["found_above_10"] += 1
                            agg_src["found_items_by_rank"]["gt10"].append({
                                "target_id": expected_id, "test_query": user_query,
                                "test_query_source": test_query_source_val, "test_answer": test_answer_val
                            })

            # финальный - в общий массив запросов
            query_results.append(result_entry)

            pbar.set_postfix({"status": "done", "rank": found_rank if found_rank is not None else "-",
                              "time": f"{float(duration_f):.4f}s"})
            pbar.update(1)

    # ------------------- Сводная статистика по всему набору -------------------
    overall_stats = {}
    total_queries = len(found_ranks_overall)
    if total_queries > 0:
        overall_stats["total_queries"] = total_queries
        overall_stats["found_at_1_count"] = count_at_1
        overall_stats["found_at_1_percent"] = round(count_at_1 * 100.0 / total_queries, 2)
        overall_stats["found_at_2_count"] = count_at_2
        overall_stats["found_at_2_percent"] = round(count_at_2 * 100.0 / total_queries, 2)
        overall_stats["found_at_3_count"] = count_at_3
        overall_stats["found_at_3_percent"] = round(count_at_3 * 100.0 / total_queries, 2)
        overall_stats["found_at_4_count"] = count_at_4
        overall_stats["found_at_4_percent"] = round(count_at_4 * 100.0 / total_queries, 2)
        overall_stats["found_at_5_count"] = count_at_5
        overall_stats["found_at_5_percent"] = round(count_at_5 * 100.0 / total_queries, 2)
        overall_stats["found_above_5_count"] = count_above_5
        overall_stats["found_above_5_percent"] = round(count_above_5 * 100.0 / total_queries, 2)
        overall_stats["found_above_10_count"] = count_above_10
        overall_stats["found_above_10_percent"] = round(count_above_10 * 100.0 / total_queries, 2)
        overall_stats["not_found_count"] = count_not_found
        overall_stats["not_found_percent"] = round(count_not_found * 100.0 / total_queries, 2)

        # средняя total_time и средняя длина запросов
        overall_stats["avg_search_time"] = round(sum(times_overall) / total_queries, 4)
        overall_stats["avg_query_length_chars"] = round(float(np.mean(query_lengths_overall)), 1) if query_lengths_overall else 0.0

        # ce_avg и response_avg
        overall_stats["ce_avg"] = round(float(np.mean(ce_times_overall)), 4) if ce_times_overall else None
        overall_stats["response_avg"] = round(float(np.mean(response_times_overall)), 4) if response_times_overall else None
        overall_stats["total_search_task_avg"] = round(float(np.mean(full_task_times_overall)), 4) if full_task_times_overall else None

        # списки объектов для инспекции
        overall_stats["found_ids_at_1"] = overall_found_items_by_rank[1]
        overall_stats["found_ids_at_2"] = overall_found_items_by_rank[2]
        overall_stats["found_ids_at_3"] = overall_found_items_by_rank[3]
        overall_stats["found_ids_at_4"] = overall_found_items_by_rank[4]
        overall_stats["found_ids_at_5"] = overall_found_items_by_rank[5]
        overall_stats["found_ids_above_5"] = overall_found_items_by_rank["gt5"]
        overall_stats["found_ids_above_10"] = overall_found_items_by_rank["gt10"]
        overall_stats["not_found_ids"] = overall_not_found_items

        # аггрегаты по времени для overlay
        overall_stats["embedding_time_stats"] = calc_time_stats(embedding_times_overall)
        overall_stats["vector_search_time_stats"] = calc_time_stats(vector_times_overall)
        overall_stats["lexical_time_stats"] = calc_time_stats(lexical_times_overall)
        overall_stats["cross_encoder_time_stats"] = calc_time_stats(ce_times_overall)

    # метрики качества (финальная выдача)
    overall_metrics = compute_rank_metrics(found_ranks_overall, METRICS_KS)
    overall_latency = percentile_dict(times_overall)
    overall_margin = compute_margin_stats(margins_overall, top1_correct_overall)
    overall_stats["metrics"] = {
        "k_list": METRICS_KS,
        **overall_metrics,
        "latency": overall_latency,
        "margin_at_1": {"score_field": MARGIN_SCORE_FIELD, **overall_margin}
    }

    # -------------------- Статистика по каждому источнику ---------------------
    stats_by_source_output = {}
    for src, st in stats_by_source.items():
        n = st["count"]
        if n == 0:
            continue
        src_stats = {
            "total_queries": n,
            "found_at_1_count": st["found_at_1"],
            "found_at_1_percent": round(st["found_at_1"] * 100.0 / n, 2),
            "found_at_2_count": st["found_at_2"],
            "found_at_2_percent": round(st["found_at_2"] * 100.0 / n, 2),
            "found_at_3_count": st["found_at_3"],
            "found_at_3_percent": round(st["found_at_3"] * 100.0 / n, 2),
            "found_at_4_count": st["found_at_4"],
            "found_at_4_percent": round(st["found_at_4"] * 100.0 / n, 2),
            "found_at_5_count": st["found_at_5"],
            "found_at_5_percent": round(st["found_at_5"] * 100.0 / n, 2),
            "found_above_5_count": st["found_above_5"],
            "found_above_5_percent": round(st["found_above_5"] * 100.0 / n, 2),
            "found_above_10_count": st["found_above_10"],
            "found_above_10_percent": round(st["found_above_10"] * 100.0 / n, 2),
            "not_found_count": st["not_found"],
            "not_found_percent": round(st["not_found"] * 100.0 / n, 2),

            "avg_search_time": round(st["duration_sum"] / n, 4) if n > 0 else 0.0,
            "avg_query_length_chars": round(float(np.mean(st["query_lengths"])), 1) if st["query_lengths"] else 0.0,

            # средние по ce/response
            "ce_avg": round(float(np.mean(st["ce_times"])), 4) if st["ce_times"] else None,
            "response_avg": round(float(np.mean(st["response_times"])), 4) if st["response_times"] else None,
            "total_search_task_avg": round(float(np.mean(st["full_task_times"])), 4) if st["full_task_times"] else None,

            "found_ids_at_1": st["found_items_by_rank"][1],
            "found_ids_at_2": st["found_items_by_rank"][2],
            "found_ids_at_3": st["found_items_by_rank"][3],
            "found_ids_at_4": st["found_items_by_rank"][4],
            "found_ids_at_5": st["found_items_by_rank"][5],
            "found_ids_above_5": st["found_items_by_rank"]["gt5"],
            "found_ids_above_10": st["found_items_by_rank"]["gt10"],
            "not_found_ids": st["not_found_items"],

            "embedding_time_stats": calc_time_stats(st.get("embedding_times", [])),
            "vector_search_time_stats": calc_time_stats(st.get("vector_search_times", [])),
            "lexical_time_stats": calc_time_stats(st.get("lexical_search_times", [])),
            "cross_encoder_time_stats": calc_time_stats(st.get("ce_times", [])),
        }

        m = compute_rank_metrics(st["found_ranks"], METRICS_KS)
        lat = percentile_dict(st["times"])
        marg = compute_margin_stats(st["margins"], st["top1_correct"])
        src_stats["metrics"] = {
            "k_list": METRICS_KS,
            **m,
            "latency": lat,
            "margin_at_1": {"score_field": MARGIN_SCORE_FIELD, **marg}
        }
        stats_by_source_output[src] = src_stats

    # ------------------------- Диагностика и per-target -----------------------
    # Сложные по рангу (хуже - выше), и самые медленные по total
    diag_hard_by_rank = sorted(
        query_results,
        key=lambda q: (rank_sort_key(q.get("expected_found_rank", None)), -q.get("search_time", 0.0)),
    )
    diag_hard_by_rank = list(reversed(diag_hard_by_rank))[:DIAG_TOPN]
    diag_hard_by_rank = [
        {"rank": q.get("expected_found_rank"), "target_id": q.get("expected_id"),
         "test_query": q.get("query"), "source": q.get("source"),
         "search_time": q.get("search_time"),
         "top1_id": q.get("_top1", {}).get("id", ""), "top1_question": q.get("_top1", {}).get("question", "")}
        for q in diag_hard_by_rank
    ]

    diag_slowest_total = sorted(query_results, key=lambda q: q.get("search_time", 0.0), reverse=True)[:DIAG_TOPN]
    diag_slowest_total = [
        {"rank": q.get("expected_found_rank"), "target_id": q.get("expected_id"),
         "test_query": q.get("query"), "source": q.get("source"),
         "search_time": q.get("search_time")}
        for q in diag_slowest_total
    ]

    # per-target
    per_target = {}
    for q, r in zip(query_results, found_ranks_overall):
        tid = q.get("expected_id", "")
        per_target.setdefault(tid, {"count": 0, "ranks": []})
        per_target[tid]["count"] += 1
        per_target[tid]["ranks"].append(r)

    per_target_stats = {}
    for tid, obj in per_target.items():
        ranks = obj["ranks"]
        m = compute_rank_metrics(ranks, METRICS_KS)
        found_only = [x for x in ranks if x is not None]
        median_rank = float(np.median(found_only)) if found_only else None
        per_target_stats[tid] = {"total_queries": obj["count"], **m, "median_found_rank": median_rank}

    # --------- Top-N самых медленных: total/embed/dense/ce/response ----------
    def _sorted_metric_list(metric_key_in_entry, fallback=None):
        pairs = []
        for q in query_results:
            v = q
            for k in metric_key_in_entry.split('.'):
                v = v.get(k, None) if isinstance(v, dict) else None
                if v is None:
                    break
            if v is None:
                if fallback is None:
                    continue
                v = q.get(fallback)
            if v is None:
                continue
            try:
                val = float(v)
            except Exception:
                continue
            pairs.append((q, val))
        pairs.sort(key=lambda t: t[1], reverse=True)
        return pairs[:AVG_SORTED_CANDIDATES]

    slow_total = _sorted_metric_list("search_time")
    slow_embed = _sorted_metric_list("_times.embedding_time")
    slow_dense = _sorted_metric_list("_times.vector_search_time")
    slow_ce    = _sorted_metric_list("_times.cross_encoder_time")
    slow_resp  = _sorted_metric_list("response_time")
    slow_full_task = _sorted_metric_list("total_search_task_time")

    def _pack_rows(pairs, col_target, col_source, text_col_name):
        rows = []
        for q, val in pairs:
            rows.append([
                q.get("expected_id", ""),
                q.get("source", ""),
                q.get("query", ""),
                f"{val:.4f}"
            ])
        return rows

    slow_blocks = {
        "total_avg_sorted": _pack_rows(slow_total, col_target, col_source, col_query),
        "embed_avg_sorted": _pack_rows(slow_embed, col_target, col_source, col_query),
        "dense_avg_sorted": _pack_rows(slow_dense, col_target, col_source, col_query),
        "ce_avg_sorted":    _pack_rows(slow_ce,    col_target, col_source, col_query),
        "total_search_task_avg_sorted": _pack_rows(slow_full_task, col_target, col_source, col_query),
        "response_avg_sorted": _pack_rows(slow_resp, col_target, col_source, col_query),
    }

    # -------------------- Свертка промежуточной статистики --------------------
    def finalize_stage_block(agg_obj):
        # На вход - агрегатор (overall или source). Возвращаем подсчитанные проценты + метрики.
        n = agg_obj["count"]
        out = {
            "total_queries": n,
            "found_at_1_count": agg_obj["found_at_1"],
            "found_at_1_percent": round(agg_obj["found_at_1"] * 100.0 / n, 2) if n else 0.0,
            "found_at_2_count": agg_obj["found_at_2"],
            "found_at_2_percent": round(agg_obj["found_at_2"] * 100.0 / n, 2) if n else 0.0,
            "found_at_3_count": agg_obj["found_at_3"],
            "found_at_3_percent": round(agg_obj["found_at_3"] * 100.0 / n, 2) if n else 0.0,
            "found_at_4_count": agg_obj["found_at_4"],
            "found_at_4_percent": round(agg_obj["found_at_4"] * 100.0 / n, 2) if n else 0.0,
            "found_at_5_count": agg_obj["found_at_5"],
            "found_at_5_percent": round(agg_obj["found_at_5"] * 100.0 / n, 2) if n else 0.0,
            "found_above_5_count": agg_obj["found_above_5"],
            "found_above_5_percent": round(agg_obj["found_above_5"] * 100.0 / n, 2) if n else 0.0,
            "found_above_10_count": agg_obj["found_above_10"],
            "found_above_10_percent": round(agg_obj["found_above_10"] * 100.0 / n, 2) if n else 0.0,
            "not_found_count": agg_obj["not_found"],
            "not_found_percent": round(agg_obj["not_found"] * 100.0 / n, 2) if n else 0.0,

            "found_ids_at_1": agg_obj["found_items_by_rank"][1],
            "found_ids_at_2": agg_obj["found_items_by_rank"][2],
            "found_ids_at_3": agg_obj["found_items_by_rank"][3],
            "found_ids_at_4": agg_obj["found_items_by_rank"][4],
            "found_ids_at_5": agg_obj["found_items_by_rank"][5],
            "found_ids_above_5": agg_obj["found_items_by_rank"]["gt5"],
            "found_ids_above_10": agg_obj["found_items_by_rank"]["gt10"],
            "not_found_ids": agg_obj["not_found_items"],
        }
        # метрики качества (MRR/nDCG/margin)
        m = compute_rank_metrics(agg_obj["found_ranks"], METRICS_KS)
        marg = compute_margin_stats(agg_obj["margins"], agg_obj["top1_correct"])
        out["metrics"] = {"k_list": METRICS_KS, **m, "margin_at_1": {"score_field": MARGIN_SCORE_FIELD, **marg}}
        return out

    intermediate_output = {}
    for stage in stage_names:
        st_block = intermediate_stats[stage]
        if st_block["overall"]["count"] == 0:
            continue
        overall_stage_stats = finalize_stage_block(st_block["overall"])
        by_src_stage_stats = {}
        for src, agg_src in st_block["by_source"].items():
            if agg_src["count"] == 0:
                continue
            by_src_stage_stats[src] = finalize_stage_block(agg_src)
        intermediate_output[stage] = {"overall_stats": overall_stage_stats, "stats_by_source": by_src_stage_stats}

    # ------------------------ Сбор итогового JSON -----------------------------
    output_data = {
        "queries": query_results,
        "overall_stats": overall_stats,
        "stats_by_source": stats_by_source_output,
        "intermediate_stats": intermediate_output,  # dense/lex/ce блоки
        "diagnostics": {
            "topN": DIAG_TOPN,
            "hard_examples_by_rank": diag_hard_by_rank,
            "slow_examples_by_total": diag_slowest_total
        },
        "sorted_slowest": slow_blocks,  # новые top-N по avg-метрикам
        "api_run_info": {k: v for k, v in api_run_info.items() if v is not None},
        "config": {
            "metrics_k_list": METRICS_KS,
            "margin_score_field": MARGIN_SCORE_FIELD,
            "avg_sorted_candidates": AVG_SORTED_CANDIDATES
        }
    }

    # ------------------------ Генерация имён файлов ---------------------------
    timestamp = time.strftime("%d%m%Y%H%M%S")
    ts_human = time.strftime("%d.%m.%Y %H:%M")
    base_name = output_base
    base_dir = ""
    if os.path.sep in base_name:
        base_dir, base_name = os.path.split(base_name)
    if base_name.lower().endswith(".json"):
        base_name = base_name[:-5]
    json_filename = f"{base_name}_{timestamp}.json"
    xlsx_filename = f"{base_name}_{timestamp}.xlsx"
    jpg_filename  = f"{base_name}_{timestamp}.jpg"
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)
        json_filepath = os.path.join(base_dir, json_filename)
        xlsx_filepath = os.path.join(base_dir, xlsx_filename)
        jpg_filepath  = os.path.join(base_dir, jpg_filename)
    else:
        json_filepath = json_filename
        xlsx_filepath = xlsx_filename
        jpg_filepath  = jpg_filename

    # ------------------------------ Save JSON --------------------------------
    try:
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
    except Exception as e:  # pragma: no cover
        print(f"Ошибка: не удалось сохранить JSON-файл результатов {json_filepath}: {e}")
        return

    # ----------- Экспорт статистики по местам (финальная выдача) в XLSX -------
    try:
        rows_overall = []
        for q in query_results:
            rank = q.get("expected_found_rank")
            if rank is None:
                continue
            top1 = q.get("_top1", {})
            rows_overall.append({
                "rank": rank,
                "target_id": q.get("expected_id", ""),
                "test_query": q.get("query", ""),
                "test_answer": top1.get("test_answer", ""),
                "1 rank id": top1.get("id", ""),
                "1 rank question": top1.get("question", ""),
                "1 rank answer": top1.get("answer", "")
            })
        df_overall = pd.DataFrame(rows_overall).sort_values(by=["rank", "target_id"]).reset_index(drop=True)

        from collections import defaultdict
        grouped = defaultdict(list)
        for q in query_results:
            grouped[q.get("source", "")].append(q)

        dfs_by_source = {}
        for src, items in grouped.items():
            rows_src = []
            for q in items:
                rank = q.get("expected_found_rank")
                if rank is None:
                    continue
                top1 = q.get("_top1", {})
                rows_src.append({
                    "rank": rank,
                    "target_id": q.get("expected_id", ""),
                    "test_query": q.get("query", ""),
                    "test_answer": top1.get("test_answer", ""),
                    "1 rank id": top1.get("id", ""),
                    "1 rank question": top1.get("question", ""),
                    "1 rank answer": top1.get("answer", "")
                })
            df_src = pd.DataFrame(rows_src).sort_values(by=["rank", "target_id"]).reset_index(drop=True)
            dfs_by_source[src] = df_src

        with pd.ExcelWriter(xlsx_filepath) as writer:
            df_overall.to_excel(writer, index=False, sheet_name="overall")
            for src, df_src in dfs_by_source.items():
                sheet = sanitize_sheet_name(f"source_{src}" if src else "source_unknown")
                df_src.to_excel(writer, index=False, sheet_name=sheet)
    except Exception as e:  # pragma: no cover
        print(f"Предупреждение: не удалось сохранить XLSX статистику {xlsx_filepath}: {e}")

    # ------------------------------ Графики -----------------------------------
    # Планируем все графики на одном холсте (одна колонка, много строк)
    plot_entries = []  # последовательность блоков к отрисовке сверху вниз

    # 1) Финальная выдача: overall
    plot_entries.append(("final_overall", None))

    # 2) Финальная выдача: по источникам
    for src in stats_by_source_output.keys():
        plot_entries.append(("final_source", src))

    # 3) Промежуточные стадии (если есть): каждая - overall, затем по источникам
    stages_order = ["dense", "lex", "ce"]
    intermediate_output_ordered = []
    for stage in stages_order:
        if stage in intermediate_output:
            intermediate_output_ordered.append(stage)
    for stage in intermediate_output_ordered:
        plot_entries.append(("stage_overall", stage))
        for src in intermediate_output[stage].get("stats_by_source", {}).keys():
            plot_entries.append(("stage_source", (stage, src)))

    # 4) Таблицы top-N slowest (внизу)
    table_titles_and_rows = []
    def _maybe_add_table(title, key):
        rows = slow_blocks.get(key, [])
        if rows:
            table_titles_and_rows.append((title, rows))
    _maybe_add_table("total_avg sorted", "total_avg_sorted")
    _maybe_add_table("embed_avg sorted", "embed_avg_sorted")
    _maybe_add_table("dense_avg sorted", "dense_avg_sorted")
    _maybe_add_table("ce_avg sorted",    "ce_avg_sorted")
    _maybe_add_table("response_avg sorted", "response_avg_sorted")
    _maybe_add_table("total_search_task avg sorted", "total_search_task_avg_sorted")

    # --- создаём фигуру ---
    n_bar_plots = len(plot_entries)
    n_tables = len(table_titles_and_rows)
    rows = n_bar_plots + n_tables
    cols = 1

    height_per_bar = 5.2
    height_per_table = 2.6
    fig_w = 12
    fig_h = max(6.0, n_bar_plots * height_per_bar + n_tables * height_per_table)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=300)
    if rows == 1:
        axes = [axes]

    # Заголовок с параметрами прогона
    header_lines = []
    # 1) дата запуска
    header_lines.append(f"Benchmark run: {ts_human}")
    # 2) ENCODER MODEL NAME и RERANKER MODEL NAME
    line2 = []
    if api_run_info.get("encoder_model") is not None:
        line2.append(f"ENCODER MODEL NAME: {api_run_info['encoder_model']}")
    if api_run_info.get("reranker_model") is not None:
        line2.append(f"RERANKER MODEL NAME: {api_run_info['reranker_model']}")
    if line2:
        header_lines.append(" • ".join(line2))
    # 3) флаги
    line3 = []
    if api_run_info.get("reranker_enabled") is not None:
        line3.append(f"RERANKER ENABLE: {_boolish(api_run_info['reranker_enabled'])}")
    if api_run_info.get("bm25_enabled") is not None:
        line3.append(f"BM25 ENABLE: {_boolish(api_run_info['bm25_enabled'])}")
    if api_run_info.get("open_search_enabled") is not None:
        line3.append(f"OPENSEARCH ENABLE: {_boolish(api_run_info['open_search_enabled'])}")
    if line3:
        header_lines.append(" • ".join(line3))
    # 4) веса гибрида
    line4 = []
    if api_run_info.get("hybrid_w_ce") is not None:
        line4.append(f"CE: {api_run_info['hybrid_w_ce']}")
    if api_run_info.get("hybrid_w_dense") is not None:
        line4.append(f"Dense: {api_run_info['hybrid_w_dense']}")
    if api_run_info.get("hybrid_w_lex") is not None:
        line4.append(f"Lex: {api_run_info['hybrid_w_lex']}")
    if line4:
        header_lines.append(" • ".join(line4))
    # 5) топ-k
    line5 = []
    if api_run_info.get("hybrid_dense_top_k") is not None:
        line5.append(f"DENSE TOP K: {api_run_info['hybrid_dense_top_k']}")
    if api_run_info.get("hybrid_lex_top_k") is not None:
        line5.append(f"LEX TOP K: {api_run_info['hybrid_lex_top_k']}")
    if api_run_info.get("hybrid_top_k") is not None:
        line5.append(f"HYBRID TOP K: {api_run_info['hybrid_top_k']}")
    if line5:
        header_lines.append(" • ".join(line5))

    fig.suptitle("\n".join(header_lines), y=0.996, fontsize=12)

    # Вспомогательные категории
    categories = ["Found@1", "Found@2", "Found@3", "Found@4", "Found@5", ">5", ">10"]

    # Функции отрисовки блоков
    def draw_final_overall(ax):
        overall_percents = [
            overall_stats.get("found_at_1_percent", 0),
            overall_stats.get("found_at_2_percent", 0),
            overall_stats.get("found_at_3_percent", 0),
            overall_stats.get("found_at_4_percent", 0),
            overall_stats.get("found_at_5_percent", 0),
            overall_stats.get("found_above_5_percent", 0),
            overall_stats.get("found_above_10_percent", 0)
        ]
        overall_counts = [
            overall_stats.get("found_at_1_count", 0),
            overall_stats.get("found_at_2_count", 0),
            overall_stats.get("found_at_3_count", 0),
            overall_stats.get("found_at_4_count", 0),
            overall_stats.get("found_at_5_count", 0),
            overall_stats.get("found_above_5_count", 0),
            overall_stats.get("found_above_10_count", 0)
        ]
        nice_bar(ax, categories, overall_percents, "Overall (%)", counts=overall_counts)

        overall_metrics_for_plot = overall_stats.get("metrics", {})
        overall_latency_plot = overall_metrics_for_plot.get("latency", {})
        overall_margin_plot = overall_metrics_for_plot.get("margin_at_1", {})

        timing_block_text_overall = build_timing_overlay_text(
            total_avg=overall_stats.get("avg_search_time", 0.0),
            total_latency_dict=overall_latency_plot,
            emb_stats=overall_stats.get("embedding_time_stats"),
            vec_stats=overall_stats.get("vector_search_time_stats"),
            lex_stats=overall_stats.get("lexical_time_stats"),
            ce_avg=overall_stats.get("ce_avg"),
            response_avg=overall_stats.get("response_avg"),
            full_task_avg=overall_stats.get("total_search_task_avg"),
            avg_chars=overall_stats.get("avg_query_length_chars", 0.0)
        )
        ax.text(0.98, 0.95, timing_block_text_overall,
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, linewidth=0))

        mrr_line, ndcg_line, _lat_line_unused, m_line = format_metrics_overlay(
            overall_metrics_for_plot, overall_latency_plot, overall_margin_plot, ks_label
        )
        ax.text(0.98, 0.60, mrr_line, transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, linewidth=0))
        ax.text(0.98, 0.48, ndcg_line, transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, linewidth=0))
        ax.text(0.98, 0.10, m_line, transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, linewidth=0))

    def draw_final_source(ax, src, src_stats):
        percents = [
            src_stats.get("found_at_1_percent", 0),
            src_stats.get("found_at_2_percent", 0),
            src_stats.get("found_at_3_percent", 0),
            src_stats.get("found_at_4_percent", 0),
            src_stats.get("found_at_5_percent", 0),
            src_stats.get("found_above_5_percent", 0),
            src_stats.get("found_above_10_percent", 0)
        ]
        counts = [
            src_stats.get("found_at_1_count", 0),
            src_stats.get("found_at_2_count", 0),
            src_stats.get("found_at_3_count", 0),
            src_stats.get("found_at_4_count", 0),
            src_stats.get("found_at_5_count", 0),
            src_stats.get("found_above_5_count", 0),
            src_stats.get("found_above_10_count", 0)
        ]
        title = f"{src} (%)" if src else "Unknown Source (%)"
        nice_bar(ax, categories, percents, title, counts=counts)

        ax.text(0.98, 0.95,
                build_timing_overlay_text(
                    total_avg=src_stats.get("avg_search_time", 0.0),
                    total_latency_dict=src_stats.get("metrics", {}).get("latency", {}),
                    emb_stats=src_stats.get("embedding_time_stats"),
                    vec_stats=src_stats.get("vector_search_time_stats"),
                    lex_stats=src_stats.get("lexical_time_stats"),
                    ce_avg=src_stats.get("ce_avg"),
                    response_avg=src_stats.get("response_avg"),
                    full_task_avg=src_stats.get("total_search_task_avg"),
                    avg_chars=src_stats.get("avg_query_length_chars", 0.0)
                ),
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, linewidth=0))

        m_for_plot = src_stats.get("metrics", {})
        lat_src = m_for_plot.get("latency", {})
        marg_src = m_for_plot.get("margin_at_1", {})
        mrr_line, ndcg_line, _lat_line_unused2, m_line = format_metrics_overlay(
            m_for_plot, lat_src, marg_src, ks_label
        )
        ax.text(0.98, 0.60, mrr_line,
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, linewidth=0))
        ax.text(0.98, 0.48, ndcg_line,
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, linewidth=0))
        ax.text(0.98, 0.10, m_line,
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, linewidth=0))

    def draw_stage_overall(ax, stage_name, stage_overall_stats):
        stage_cats = ["Found@1", "Found@2", "Found@3", "Found@4", "Found@5", ">5", ">10"]
        n = stage_overall_stats.get("total_queries", 0) or 1
        vals = [
            round(stage_overall_stats.get("found_at_1_count", 0) * 100.0 / n, 2),
            round(stage_overall_stats.get("found_at_2_count", 0) * 100.0 / n, 2),
            round(stage_overall_stats.get("found_at_3_count", 0) * 100.0 / n, 2),
            round(stage_overall_stats.get("found_at_4_count", 0) * 100.0 / n, 2),
            round(stage_overall_stats.get("found_at_5_count", 0) * 100.0 / n, 2),
            round(stage_overall_stats.get("found_above_5_count", 0) * 100.0 / n, 2),
            round(stage_overall_stats.get("found_above_10_count", 0) * 100.0 / n, 2),
        ]
        counts = [
            stage_overall_stats.get("found_at_1_count", 0),
            stage_overall_stats.get("found_at_2_count", 0),
            stage_overall_stats.get("found_at_3_count", 0),
            stage_overall_stats.get("found_at_4_count", 0),
            stage_overall_stats.get("found_at_5_count", 0),
            stage_overall_stats.get("found_above_5_count", 0),
            stage_overall_stats.get("found_above_10_count", 0),
        ]
        nice_bar(ax, stage_cats, vals, f"{stage_name.upper()} overall (%)", counts=counts)

        m = stage_overall_stats.get("metrics", {})
        marg = m.get("margin_at_1", {})
        mrr_line, ndcg_line, _lat_line_unused3, m_line = format_metrics_overlay(m, {}, marg, ks_label)
        ax.text(0.98, 0.60, mrr_line, transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, linewidth=0))
        ax.text(0.98, 0.48, ndcg_line, transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, linewidth=0))
        ax.text(0.98, 0.10, m_line, transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, linewidth=0))

        # для промежуточных не выводим тайминги, только среднюю длину запросов (берём из overall)
        # но у нас нет по-стейджевской длины; просто ничего не пишем, чтобы не усложнять.

    def draw_stage_source(ax, stage_name, src_name, src_stats):
        stage_cats = ["Found@1", "Found@2", "Found@3", "Found@4", "Found@5", ">5", ">10"]
        n = src_stats.get("total_queries", 0) or 1
        vals = [
            round(src_stats.get("found_at_1_count", 0) * 100.0 / n, 2),
            round(src_stats.get("found_at_2_count", 0) * 100.0 / n, 2),
            round(src_stats.get("found_at_3_count", 0) * 100.0 / n, 2),
            round(src_stats.get("found_at_4_count", 0) * 100.0 / n, 2),
            round(src_stats.get("found_at_5_count", 0) * 100.0 / n, 2),
            round(src_stats.get("found_above_5_count", 0) * 100.0 / n, 2),
            round(src_stats.get("found_above_10_count", 0) * 100.0 / n, 2),
        ]
        counts = [
            src_stats.get("found_at_1_count", 0),
            src_stats.get("found_at_2_count", 0),
            src_stats.get("found_at_3_count", 0),
            src_stats.get("found_at_4_count", 0),
            src_stats.get("found_at_5_count", 0),
            src_stats.get("found_above_5_count", 0),
            src_stats.get("found_above_10_count", 0),
        ]
        nice_bar(ax, stage_cats, vals, f"{stage_name.upper()} - {src_name or 'Unknown'} (%)", counts=counts)
        m = src_stats.get("metrics", {})
        marg = m.get("margin_at_1", {})
        mrr_line, ndcg_line, _lat_line_unused4, m_line = format_metrics_overlay(m, {}, marg, ks_label)
        ax.text(0.98, 0.60, mrr_line, transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, linewidth=0))
        ax.text(0.98, 0.48, ndcg_line, transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, linewidth=0))
        ax.text(0.98, 0.10, m_line, transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, linewidth=0))

    # Рисуем все бар-графики
    ax_i = 0
    for kind, payload in plot_entries:
        ax = axes[ax_i]
        if kind == "final_overall":
            draw_final_overall(ax)
        elif kind == "final_source":
            s = payload
            draw_final_source(ax, s, stats_by_source_output.get(s, {}))
        elif kind == "stage_overall":
            st = payload  # 'dense'|'lex'|'ce'
            draw_stage_overall(ax, st, intermediate_output[st]["overall_stats"])
        elif kind == "stage_source":
            st, s = payload
            draw_stage_source(ax, st, s, intermediate_output[st]["stats_by_source"][s])
        ax_i += 1

    # Таблицы slow-N в самом низу
    table_headers = [col_target, col_source, col_query, "metric"]
    for title, rows in table_titles_and_rows:
        ax = axes[ax_i]
        draw_table(ax, title, rows, table_headers, max_chars=40)
        ax_i += 1

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    try:
        fig.savefig(jpg_filepath, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"Предупреждение: не удалось сохранить график {jpg_filepath}: {e}")

    print(f"Готово.\nJSON: {json_filepath}\nXLSX: {xlsx_filepath}\nJPG:  {jpg_filepath}")


if __name__ == "__main__":
    main()