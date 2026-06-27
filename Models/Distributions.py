# =============================================================================
#  FIRM-DEADLINE TOOL CONTROL (FTC) / CADTR
#  Product Signature: FTC
# ------------------------------------------------------------------------------
#  File: Models/Distributions.py
#  Purpose: Define interarrival and service-time distributions.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import math
from statistics import NormalDist

import numpy as np

from Core.Task import Mode


class Interarrival_Model(Protocol):
    def Sample(self, rng_generator: np.random.Generator) -> float:
        ...


class Service_Time_Model(Protocol):
    def Sample(self, mode_mode: Mode, rng_generator: np.random.Generator) -> float:
        ...

    def Expected(self, mode_mode: Mode) -> float:
        ...

    def Quantile(self, mode_mode: Mode, q_f64: float) -> float:
        ...

    def Sample_By_Tier(self, tier_index: int, rng_generator: np.random.Generator) -> float:
        """Sample service time for a given tier index. Default J=2 mapping."""
        mode = Mode.FAST if tier_index == 0 else Mode.SLOW
        return self.Sample(mode, rng_generator)

    def Expected_By_Tier(self, tier_index: int) -> float:
        """Expected service time for a given tier index. Default J=2 mapping."""
        mode = Mode.FAST if tier_index == 0 else Mode.SLOW
        return self.Expected(mode)


@dataclass(frozen=True)
class Exponential_Interarrival:

    lambda_rate_f64 : float

    def __post_init__(self) -> None:
        if self.lambda_rate_f64 <= 0:
            raise ValueError("lambda_rate must be > 0.")

    def Sample(self, rng_generator: np.random.Generator) -> float:
        return float(rng_generator.exponential(scale=1.0 / self.lambda_rate_f64))


@dataclass(frozen=True)
class Lognormal_Service_Times:

    slow_mu_f64          : float
    slow_sigma_f64       : float
    fast_mu_f64          : float
    fast_sigma_f64       : float
    min_service_time_f64 : float = 1e-6                  

    def __post_init__(self) -> None:
        if self.slow_sigma_f64 <= 0 or self.fast_sigma_f64 <= 0:
            raise ValueError("Lognormal sigmas must be > 0.")
        if self.min_service_time_f64 <= 0:
            raise ValueError("min_service_time must be > 0.")

    def Sample(self, mode_mode: Mode, rng_generator: np.random.Generator) -> float:
        if mode_mode == Mode.SLOW:
            s_f64 = float(rng_generator.lognormal(mean=self.slow_mu_f64, sigma=self.slow_sigma_f64))
        elif mode_mode == Mode.FAST:
            s_f64 = float(rng_generator.lognormal(mean=self.fast_mu_f64, sigma=self.fast_sigma_f64))
        else:
            raise ValueError(f"Unsupported mode: {mode_mode}")

        return max(s_f64, self.min_service_time_f64)

    def Expected(self, mode_mode: Mode) -> float:
        if mode_mode == Mode.SLOW:
            return float(np.exp(self.slow_mu_f64 + 0.5 * (self.slow_sigma_f64 ** 2)))
        if mode_mode == Mode.FAST:
            return float(np.exp(self.fast_mu_f64 + 0.5 * (self.fast_sigma_f64 ** 2)))
        raise ValueError(f"Unsupported mode: {mode_mode}")

    def Quantile(self, mode_mode: Mode, q_f64: float) -> float:
        if not (0.0 < q_f64 < 1.0):
            raise ValueError("q_f64 must be in (0, 1).")

        z_f64 = float(NormalDist().inv_cdf(q_f64))

        if mode_mode == Mode.SLOW:
            mu_f64 = float(self.slow_mu_f64)
            sigma_f64 = float(self.slow_sigma_f64)
        elif mode_mode == Mode.FAST:
            mu_f64 = float(self.fast_mu_f64)
            sigma_f64 = float(self.fast_sigma_f64)
        else:
            raise ValueError(f"Unsupported mode: {mode_mode}")

        val_f64 = math.exp(mu_f64 + sigma_f64 * z_f64)
        return max(float(val_f64), float(self.min_service_time_f64))

    def Sample_By_Tier(self, tier_index: int, rng_generator: np.random.Generator) -> float:
        mode = Mode.FAST if tier_index == 0 else Mode.SLOW
        return self.Sample(mode, rng_generator)

    def Expected_By_Tier(self, tier_index: int) -> float:
        mode = Mode.FAST if tier_index == 0 else Mode.SLOW
        return self.Expected(mode)


# Module-level cache of parsed traces, keyed by (path, columns, filters).
# Parsing the trace CSV is the dominant per-simulation cost; caching makes
# long serial sweeps (e.g. reproduce/run_sweep.py --serial) practical without
# changing any results (the parsed rows are read-only during sampling).
_TRACE_CACHE: Dict[tuple, tuple] = {}


@dataclass(frozen=True)
class Trace_Service_Times:

    csv_path_str : str
    fast_latency_column_str : str = "fast_total_latency_sec"
    slow_latency_column_str : str = "slow_total_latency_sec"
    drop_error_rows_bool : bool = True
    prompt_type_filter_opt : Optional[str] = None
    min_service_time_f64 : float = 1e-6

    _rows_list_dict: List[Dict[str, object]] = field(init=False, repr=False)
    _fast_samples_list_f64: List[float] = field(init=False, repr=False)
    _slow_samples_list_f64: List[float] = field(init=False, repr=False)
    _fast_sorted_list_f64: List[float] = field(init=False, repr=False)
    _slow_sorted_list_f64: List[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.min_service_time_f64 <= 0:
            raise ValueError("min_service_time must be > 0.")

        csv_path = self.csv_path_str
        if not os.path.isabs(csv_path):
            csv_path = os.path.normpath(csv_path)

        cache_key = (
            os.path.abspath(csv_path),
            self.fast_latency_column_str,
            self.slow_latency_column_str,
            self.drop_error_rows_bool,
            self.prompt_type_filter_opt,
        )
        cached = _TRACE_CACHE.get(cache_key)
        if cached is not None:
            rows_list, fast_samples, slow_samples, fast_sorted, slow_sorted = cached
            object.__setattr__(self, "_rows_list_dict", rows_list)
            object.__setattr__(self, "_fast_samples_list_f64", fast_samples)
            object.__setattr__(self, "_slow_samples_list_f64", slow_samples)
            object.__setattr__(self, "_fast_sorted_list_f64", fast_sorted)
            object.__setattr__(self, "_slow_sorted_list_f64", slow_sorted)
            return

        rows_list: List[Dict[str, object]] = []
        fast_samples: List[float] = []
        slow_samples: List[float] = []

        with open(csv_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"No header found in trace CSV: {csv_path}")
            required_cols = {self.fast_latency_column_str, self.slow_latency_column_str}
            missing_cols = required_cols.difference(set(reader.fieldnames))
            if missing_cols:
                missing_list = ", ".join(sorted(missing_cols))
                raise ValueError(f"Trace CSV missing required columns: {missing_list}")

            for row in reader:
                if self.drop_error_rows_bool:
                    if _Has_Error(row.get("fast_error")) or _Has_Error(row.get("slow_error")):
                        continue

                prompt_type = str(row.get("prompt_type") or "").strip()
                if self.prompt_type_filter_opt is not None and prompt_type != self.prompt_type_filter_opt:
                    continue

                parsed = _Parse_Trace_Row(row)
                fast_val = _Get_Float(parsed, self.fast_latency_column_str)
                slow_val = _Get_Float(parsed, self.slow_latency_column_str)
                if fast_val is None or slow_val is None:
                    continue
                if fast_val <= 0.0 or slow_val <= 0.0:
                    continue

                parsed["prompt_type"] = prompt_type
                rows_list.append(parsed)
                fast_samples.append(float(fast_val))
                slow_samples.append(float(slow_val))

        if not rows_list:
            raise ValueError(f"No usable rows found in trace CSV: {csv_path}")

        fast_sorted = sorted(fast_samples)
        slow_sorted = sorted(slow_samples)
        object.__setattr__(self, "_rows_list_dict", rows_list)
        object.__setattr__(self, "_fast_samples_list_f64", fast_samples)
        object.__setattr__(self, "_slow_samples_list_f64", slow_samples)
        object.__setattr__(self, "_fast_sorted_list_f64", fast_sorted)
        object.__setattr__(self, "_slow_sorted_list_f64", slow_sorted)
        _TRACE_CACHE[cache_key] = (rows_list, fast_samples, slow_samples,
                                   fast_sorted, slow_sorted)

    def _Row_For_Task(self, task_task: "Task", rng_generator: np.random.Generator) -> Dict[str, object]:
        meta = task_task.meta_dict_obj
        row_idx = meta.get("trace_row_index")
        if row_idx is None:
            row_idx = int(rng_generator.integers(0, len(self._rows_list_dict)))
            meta["trace_row_index"] = row_idx
            row = self._rows_list_dict[row_idx]
            fast_val = _Get_Float(row, self.fast_latency_column_str)
            slow_val = _Get_Float(row, self.slow_latency_column_str)
            if fast_val is not None:
                meta["trace_fast_sec"] = float(fast_val)
            if slow_val is not None:
                meta["trace_slow_sec"] = float(slow_val)
            meta["trace_prompt_type"] = str(row.get("prompt_type") or "")
        return self._rows_list_dict[int(row_idx)]

    def Sample_For_Task(self, task_task: "Task", mode_mode: Mode, rng_generator: np.random.Generator) -> float:
        row = self._Row_For_Task(task_task, rng_generator)
        if mode_mode == Mode.SLOW:
            val = _Get_Float(row, self.slow_latency_column_str)
        elif mode_mode == Mode.FAST:
            val = _Get_Float(row, self.fast_latency_column_str)
        else:
            raise ValueError(f"Unsupported mode: {mode_mode}")

        if val is None:
            raise ValueError("Trace row missing latency values.")

        return max(float(val), self.min_service_time_f64)

    def Sample(self, mode_mode: Mode, rng_generator: np.random.Generator) -> float:
        idx = int(rng_generator.integers(0, len(self._rows_list_dict)))
        row = self._rows_list_dict[idx]
        if mode_mode == Mode.SLOW:
            val = _Get_Float(row, self.slow_latency_column_str)
        elif mode_mode == Mode.FAST:
            val = _Get_Float(row, self.fast_latency_column_str)
        else:
            raise ValueError(f"Unsupported mode: {mode_mode}")

        if val is None:
            raise ValueError("Trace row missing latency values.")

        return max(float(val), self.min_service_time_f64)

    def Expected(self, mode_mode: Mode) -> float:
        if mode_mode == Mode.SLOW:
            return float(np.mean(self._slow_samples_list_f64))
        if mode_mode == Mode.FAST:
            return float(np.mean(self._fast_samples_list_f64))
        raise ValueError(f"Unsupported mode: {mode_mode}")

    def Quantile(self, mode_mode: Mode, q_f64: float) -> float:
        if not (0.0 < q_f64 < 1.0):
            raise ValueError("q_f64 must be in (0, 1).")
        if mode_mode == Mode.SLOW:
            samples = self._slow_sorted_list_f64
        elif mode_mode == Mode.FAST:
            samples = self._fast_sorted_list_f64
        else:
            raise ValueError(f"Unsupported mode: {mode_mode}")

        if not samples:
            return float("nan")

        pos = (len(samples) - 1) * q_f64
        lower = int(math.floor(pos))
        upper = int(math.ceil(pos))
        if lower == upper:
            return float(samples[lower])
        weight = pos - lower
        return float(samples[lower] * (1.0 - weight) + samples[upper] * weight)

    def Sample_By_Tier(self, tier_index: int, rng_generator: np.random.Generator) -> float:
        mode = Mode.FAST if tier_index == 0 else Mode.SLOW
        return self.Sample(mode, rng_generator)

    def Sample_By_Tier_For_Task(self, task_task: "Task", tier_index: int, rng_generator: np.random.Generator) -> float:
        mode = Mode.FAST if tier_index == 0 else Mode.SLOW
        return self.Sample_For_Task(task_task, mode, rng_generator)

    def Expected_By_Tier(self, tier_index: int) -> float:
        mode = Mode.FAST if tier_index == 0 else Mode.SLOW
        return self.Expected(mode)


@dataclass
class EWMA_Service_Time_Estimator:

    base_model: Service_Time_Model
    alpha_f64: float = 0.125
    warmup_count_i32: int = 20

    _ewma_dict_mode_to_f64: dict = field(init=False, default_factory=dict)
    _count_dict_mode_to_i32: dict = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha_f64 <= 1.0):
            raise ValueError("alpha_f64 must be in (0, 1].")
        if self.warmup_count_i32 < 0:
            raise ValueError("warmup_count_i32 must be >= 0.")

        self._ewma_dict_mode_to_f64 = {
            Mode.SLOW: float(self.base_model.Expected(Mode.SLOW)),
            Mode.FAST: float(self.base_model.Expected(Mode.FAST)),
        }
        self._count_dict_mode_to_i32 = {Mode.SLOW: 0, Mode.FAST: 0}

    def Sample(self, mode_mode: Mode, rng_generator: np.random.Generator) -> float:
        return float(self.base_model.Sample(mode_mode, rng_generator))

    def Sample_For_Task(self, task_task: "Task", mode_mode: Mode, rng_generator: np.random.Generator) -> float:
        sample_for_task = getattr(self.base_model, "Sample_For_Task", None)
        if callable(sample_for_task):
            return float(sample_for_task(task_task, mode_mode, rng_generator))
        return float(self.base_model.Sample(mode_mode, rng_generator))

    def Expected(self, mode_mode: Mode) -> float:
        if mode_mode not in self._count_dict_mode_to_i32:
            raise ValueError(f"Unsupported mode: {mode_mode}")

        count_i32 = int(self._count_dict_mode_to_i32[mode_mode])
        if count_i32 >= int(self.warmup_count_i32):
            return float(self._ewma_dict_mode_to_f64[mode_mode])

        return float(self.base_model.Expected(mode_mode))

    def Quantile(self, mode_mode: Mode, q_f64: float) -> float:
        base_quantile = getattr(self.base_model, "Quantile", None)
        if callable(base_quantile):
            return float(base_quantile(mode_mode, q_f64))
                                                         
        return float(self.base_model.Expected(mode_mode))

    def Observe(self, mode_mode: Mode, service_time_f64: float) -> None:
        if mode_mode not in self._count_dict_mode_to_i32:
            return
        if service_time_f64 <= 0.0:
            return

        prev_f64 = float(self._ewma_dict_mode_to_f64[mode_mode])
        alpha_f64 = float(self.alpha_f64)
        new_f64 = (1.0 - alpha_f64) * prev_f64 + alpha_f64 * float(service_time_f64)

        self._ewma_dict_mode_to_f64[mode_mode] = new_f64
        self._count_dict_mode_to_i32[mode_mode] = int(self._count_dict_mode_to_i32[mode_mode]) + 1

    def Snapshot(self) -> dict:
        return {
            "slow_ewma": float(self._ewma_dict_mode_to_f64[Mode.SLOW]),
            "fast_ewma": float(self._ewma_dict_mode_to_f64[Mode.FAST]),
            "slow_count": int(self._count_dict_mode_to_i32[Mode.SLOW]),
            "fast_count": int(self._count_dict_mode_to_i32[Mode.FAST]),
        }

    def Sample_By_Tier(self, tier_index: int, rng_generator: np.random.Generator) -> float:
        mode = Mode.FAST if tier_index == 0 else Mode.SLOW
        return self.Sample(mode, rng_generator)

    def Sample_By_Tier_For_Task(self, task_task: "Task", tier_index: int, rng_generator: np.random.Generator) -> float:
        mode = Mode.FAST if tier_index == 0 else Mode.SLOW
        return self.Sample_For_Task(task_task, mode, rng_generator)

    def Expected_By_Tier(self, tier_index: int) -> float:
        mode = Mode.FAST if tier_index == 0 else Mode.SLOW
        return self.Expected(mode)


def _Has_Error(val_opt: Optional[str]) -> bool:
    if val_opt is None:
        return False
    return str(val_opt).strip() not in {"", "None", "none", "NULL", "null"}


def _Parse_Trace_Row(row: Dict[str, str]) -> Dict[str, object]:
    numeric_fields = {
        "fast_total_latency_sec",
        "slow_total_latency_sec",
        "fast_generation_only_sec",
        "slow_generation_only_sec",
        "router_latency_sec",
        "fast_response_length_char",
        "slow_response_length_char",
    }

    parsed: Dict[str, object] = {}
    for key, val in row.items():
        if key in numeric_fields:
            parsed[key] = _Parse_Number(val)
        elif key in {"id", "run_iter", "prompt_index"}:
            parsed[key] = _Parse_Int(val)
        else:
            parsed[key] = val
    return parsed


def _Parse_Number(val_opt: Optional[str]) -> Optional[float]:
    if val_opt is None:
        return None
    text = str(val_opt).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _Parse_Int(val_opt: Optional[str]) -> Optional[int]:
    if val_opt is None:
        return None
    text = str(val_opt).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return None


def _Get_Float(row_dict: Dict[str, object], key: str) -> Optional[float]:
    val = row_dict.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
