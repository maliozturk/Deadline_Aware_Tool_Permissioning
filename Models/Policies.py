# =============================================================================
#  FIRM-DEADLINE TOOL CONTROL (FTC)
#  Product Signature: FTC
# ------------------------------------------------------------------------------
#  File: Models/Policies.py
#  Purpose: Implement scheduling policies including FTC and baselines.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np

from Configurations import Policy_Config, Utility_Config
from Core.Task import Mode, Task
from Models.Distributions import Service_Time_Model
from Models.Policy_Base import Scheduling_Policy, System_State


@dataclass(frozen=True)
class FTC_Decision_Trace:
    task_id_i32            : int
    arrival_time_f64       : float
    delta_k_f64            : float
    queue_length_i32       : int
    in_service_rem_f64     : float
    s_slow_f64             : float
    s_fast_f64             : float
    s_avg_f64              : float
    w_hat_f64              : float
    slack_f64              : float
    decision_mode          : Mode


@dataclass
class FTCPolicy(Scheduling_Policy):
    cfg: Policy_Config
    service_model: Service_Time_Model
    decision_trace_list: List[FTC_Decision_Trace] = field(default_factory=list)
    epsilon_trace_times_list_f64: List[float] = field(default_factory=list)
    epsilon_trace_values_list_f64: List[float] = field(default_factory=list)
    _epsilon_recent_misses_deque_i32: Deque[int] = field(init=False)
    _epsilon_error_integral_f64: float = field(init=False, default=0.0)
    _epsilon_f64: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.cfg.datp_wait_estimator not in {"conservative", "mix"}:
            raise ValueError("datp_wait_estimator must be one of {'conservative','mix'}.")
        p = float(self.cfg.datp_queue_slow_mix_p)
        if not (0.0 <= p <= 1.0):
            raise ValueError("datp_queue_slow_mix_p must be in [0, 1].")
        if float(self.cfg.datp_slack_factor) <= 0:
            raise ValueError("datp_slack_factor must be > 0.")
        eps = float(self.cfg.datp_epsilon_f64)
        if not (-1.0 < eps <= 1.0):
            raise ValueError("datp_epsilon_f64 must be in (-1, 1].")

        self._epsilon_f64 = float(self.cfg.datp_epsilon_f64)
        self._epsilon_recent_misses_deque_i32 = deque(maxlen=int(self.cfg.datp_adaptive_epsilon_window_i32))

        if self.cfg.datp_adaptive_epsilon_enabled_bool:
            if self.cfg.datp_adaptive_epsilon_window_i32 <= 0:
                raise ValueError("datp_adaptive_epsilon_window_i32 must be > 0.")
            if float(self.cfg.datp_adaptive_epsilon_kp_f64) < 0:
                raise ValueError("datp_adaptive_epsilon_kp_f64 must be >= 0.")
            if float(self.cfg.datp_adaptive_epsilon_ki_f64) < 0:
                raise ValueError("datp_adaptive_epsilon_ki_f64 must be >= 0.")
            eps_min = float(self.cfg.datp_adaptive_epsilon_min_f64)
            eps_max = float(self.cfg.datp_adaptive_epsilon_max_f64)
            if eps_min <= -1.0:
                raise ValueError("datp_adaptive_epsilon_min_f64 must be > -1.")
            if eps_max <= -1.0:
                raise ValueError("datp_adaptive_epsilon_max_f64 must be > -1.")
            if eps_min > eps_max:
                raise ValueError("datp_adaptive_epsilon_min_f64 must be <= datp_adaptive_epsilon_max_f64.")
            self._epsilon_f64 = self._Clamp_Epsilon(self._epsilon_f64)

    def Decide_Mode(self, task: Task, state: System_State) -> Mode:
        delta_k = task.deadline - state.now_f64
        if delta_k <= 0:
            return Mode.FAST

                     
        s_slow = float(self.service_model.Expected(Mode.SLOW))
        s_fast = float(self.service_model.Expected(Mode.FAST))

                               
        if self.cfg.datp_wait_estimator == "conservative":
            s_avg = s_slow
        else:
                                                         
            p = float(self.cfg.datp_queue_slow_mix_p)
            s_avg = p * s_slow + (1.0 - p) * s_fast

                                                             
        in_service = float(state.server_remaining_time) if (state.server_busy_bool and self.cfg.datp_include_in_service) else 0.0
        W_hat = in_service + float(state.queue_length_i32) * s_avg

        epsilon = self._Current_Epsilon()
        slack = (
            float(self.cfg.datp_slack_factor)
            * float(delta_k)
            * (1.0 + float(epsilon))
        )

        decision = Mode.SLOW if (W_hat + s_slow) <= slack else Mode.FAST

        if self.cfg.datp_trace_enabled_bool:
            self.decision_trace_list.append(
                FTC_Decision_Trace(
                    task_id_i32=int(task.task_id),
                    arrival_time_f64=float(task.arrival_time),
                    delta_k_f64=float(delta_k),
                    queue_length_i32=int(state.queue_length_i32),
                    in_service_rem_f64=float(in_service),
                    s_slow_f64=float(s_slow),
                    s_fast_f64=float(s_fast),
                    s_avg_f64=float(s_avg),
                    w_hat_f64=float(W_hat),
                    slack_f64=float(slack),
                    decision_mode=decision,
                )
            )

        return decision

    def Should_Switch_Mode(self, task: Task, state: System_State) -> bool:
                                                                                               
        return False

    def Observe_Task_Outcome(self, task: Task) -> None:
        if not self.cfg.datp_adaptive_epsilon_enabled_bool:
            return

        missed = 1
        if task.dropped_in_queue_bool:
            missed = 1
        elif task.completion_time_f64_opt is None:
            missed = 1
        else:
            missed = 1 if (task.completion_time_f64_opt > task.deadline) else 0

        self._epsilon_recent_misses_deque_i32.append(int(missed))
        if len(self._epsilon_recent_misses_deque_i32) == 0:
            return

        dmr_k = float(np.mean(self._epsilon_recent_misses_deque_i32))
        target = float(self.cfg.datp_epsilon_f64)
                                                                            
        error = target - dmr_k

        kp = float(self.cfg.datp_adaptive_epsilon_kp_f64)
        ki = float(self.cfg.datp_adaptive_epsilon_ki_f64)

        eps_min = float(self.cfg.datp_adaptive_epsilon_min_f64)
        eps_max = float(self.cfg.datp_adaptive_epsilon_max_f64)

        provisional_integral = self._epsilon_error_integral_f64 + float(error)
        candidate = self._epsilon_f64 + (kp * error) + (ki * provisional_integral)
        clamped = float(np.clip(float(candidate), eps_min, eps_max))

                                                                                              
        if clamped == candidate:
            self._epsilon_error_integral_f64 = provisional_integral
        elif (clamped >= eps_max and error < 0.0) or (clamped <= eps_min and error > 0.0):
            self._epsilon_error_integral_f64 = provisional_integral

        self._epsilon_f64 = clamped

        if task.completion_time_f64_opt is not None:
            t_f64 = float(task.completion_time_f64_opt)
        elif task.drop_time_f64_opt is not None:
            t_f64 = float(task.drop_time_f64_opt)
        else:
            t_f64 = float(task.arrival_time)

        self.epsilon_trace_times_list_f64.append(t_f64)
        self.epsilon_trace_values_list_f64.append(float(self._epsilon_f64))

    def FTC_Policy_Identifier(self) -> None:
        return

    def _Current_Epsilon(self) -> float:
        if self.cfg.datp_adaptive_epsilon_enabled_bool:
            return float(self._epsilon_f64)
        return float(self.cfg.datp_epsilon_f64)

    def _Clamp_Epsilon(self, eps_f64: float) -> float:
        if not self.cfg.datp_adaptive_epsilon_enabled_bool:
            return float(eps_f64)
        eps_min = float(self.cfg.datp_adaptive_epsilon_min_f64)
        eps_max = float(self.cfg.datp_adaptive_epsilon_max_f64)
        return float(np.clip(float(eps_f64), eps_min, eps_max))

    def TFG_Policy_Identifier(self) -> None:
        return self.FTC_Policy_Identifier()


DATP_Decision_Trace = FTC_Decision_Trace
DATPPolicy = FTCPolicy
TFG_Decision_Trace = FTC_Decision_Trace
TFGPolicy = FTCPolicy


@dataclass
@dataclass
class Baseline_Heuristic_Policy(Scheduling_Policy):

    cfg_policy_config                : Policy_Config
    service_model_service_time_model : Service_Time_Model

    def Decide_Mode(self, task: Task, state_system_state: System_State) -> Mode:
        ttl_rem_f64 = task.Ttl_Remaining(state_system_state.now_f64)
        if ttl_rem_f64 <= 0:
                                                                                       
            return Mode.FAST

        qlen_i32 = state_system_state.queue_length_i32
        expected_slow_f64 = self.service_model_service_time_model.Expected(Mode.SLOW)

                                                                                       
        if (qlen_i32 > self.cfg_policy_config.L_threshold_i32) or (
            expected_slow_f64 > self.cfg_policy_config.alpha_f64 * ttl_rem_f64
        ):
            return Mode.FAST

        return Mode.SLOW

    def Should_Switch_Mode(self, task: Task, state_system_state: System_State) -> bool:
        if not self.cfg_policy_config.enable_mode_switching_bool:
            return False

                                                         
        if task.chosen_mode_mode_opt != Mode.SLOW:
            return False

        ttl_rem_f64 = task.Ttl_Remaining(state_system_state.now_f64)
        if ttl_rem_f64 <= 0:
            return True

                                                                           
        expected_slow_total_f64 = self.service_model_service_time_model.Expected(Mode.SLOW)
        remaining_slow_est_f64 = max(expected_slow_total_f64 - task.service_consumed_f64, 0.0)

                                                                                           
                                                                 
        return remaining_slow_est_f64 > (self.cfg_policy_config.switch_alpha_f64 * ttl_rem_f64)


@dataclass
class Fcfs_Always_Slow_Policy(Scheduling_Policy):

    def Decide_Mode(self, task: Task, state_system_state: System_State) -> Mode:
        return Mode.SLOW

    def Should_Switch_Mode(self, task: Task, state_system_state: System_State) -> bool:
        return False


@dataclass
class Fcfs_Always_Fast_Policy(Scheduling_Policy):

    def Decide_Mode(self, task: Task, state_system_state: System_State) -> Mode:
        return Mode.FAST

    def Should_Switch_Mode(self, task: Task, state_system_state: System_State) -> bool:
        return False


@dataclass
class Static_Mix_Policy(Scheduling_Policy):
    cfg: Policy_Config
    rng: np.random.Generator

    def __post_init__(self) -> None:
        p = float(self.cfg.static_mix_p)
        if not (0.0 <= p <= 1.0):
            raise ValueError("static_mix_p must be in [0, 1].")

    def Decide_Mode(self, task: Task, state: System_State) -> Mode:
        u = float(self.rng.random())
        return Mode.SLOW if u < float(self.cfg.static_mix_p) else Mode.FAST

    def Should_Switch_Mode(self, task: Task, state: System_State) -> bool:
        return False


@dataclass
class Queue_Length_Bang_Bang_Policy(Scheduling_Policy):
    cfg: Policy_Config

    def Decide_Mode(self, task: Task, state: System_State) -> Mode:
        tau = int(self.cfg.queue_threshold_tau)
        return Mode.SLOW if state.queue_length_i32 < tau else Mode.FAST

    def Should_Switch_Mode(self, task: Task, state: System_State) -> bool:
        return False


@dataclass
class TTL_Feasibility_Bang_Bang_Policy(Scheduling_Policy):
    cfg: Policy_Config
    service_model: Service_Time_Model

    def Decide_Mode(self, task: Task, state: System_State) -> Mode:
        ttl_rem_f64 = task.Ttl_Remaining(state.now_f64)
        if ttl_rem_f64 <= 0.0:
            return Mode.FAST

        s_slow = float(self.service_model.Expected(Mode.SLOW))
        in_service = float(state.server_remaining_time) if state.server_busy_bool else 0.0
        w_tilde = in_service + float(state.queue_length_i32) * s_slow
        return Mode.SLOW if (w_tilde + s_slow) <= ttl_rem_f64 else Mode.FAST

    def Should_Switch_Mode(self, task: Task, state: System_State) -> bool:
        return False


@dataclass
class Drift_Penalty_Myopic_Policy(Scheduling_Policy):
    cfg: Policy_Config
    service_model: Service_Time_Model
    utility_cfg: Utility_Config

    def __post_init__(self) -> None:
        if float(self.cfg.drift_V) <= 0:
            raise ValueError("drift_V must be > 0.")

    def Decide_Mode(self, task: Task, state: System_State) -> Mode:
        Q = float(state.queue_length_i32)
        V = float(self.cfg.drift_V)

                            
        u_slow = float(self.utility_cfg.slow_success_utility_f64)
        u_fast = float(self.utility_cfg.fast_success_utility_f64)

                                
        s_slow = float(self.service_model.Expected(Mode.SLOW))
        s_fast = float(self.service_model.Expected(Mode.FAST))

        score_slow = V * u_slow - Q * s_slow
        score_fast = V * u_fast - Q * s_fast

                                                                                      
        if score_slow > score_fast:
            return Mode.SLOW
        return Mode.FAST

    def Should_Switch_Mode(self, task: Task, state: System_State) -> bool:
        return False


Queue_Threshold_Policy = Queue_Length_Bang_Bang_Policy


