# =============================================================================
#  FIRM-DEADLINE TOOL CONTROL (FTC) / CADTR
#  Product Signature: FTC
# ------------------------------------------------------------------------------
#  File: Core/Simulator.py
#  Purpose: Run the single-server discrete-event simulation.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

from Configurations import Simulation_Config
from Core.Events import Event, Event_Calendar, Event_Type
from Core.Task import Mode, Task, _tier_to_mode
from Metrics.Collector import MetricsCollector
from Models.Distributions import Interarrival_Model, Service_Time_Model
from Models.Policy_Base import Scheduling_Policy, System_State


@dataclass
class Server:
    busy_bool                   : bool           = False
    current_task_task_opt       : Optional[Task]  = None
    current_service_end_f64_opt : Optional[float] = None


class Simulator:

    def __init__(
        self,
        cfg_simulation_config           : Simulation_Config,
        interarrival_model_interarrival : Interarrival_Model,
        service_model_service_time      : Service_Time_Model,
        policy_scheduling_policy        : Scheduling_Policy,
        metrics_metrics_collector       : MetricsCollector,
    ) -> None:
        self.cfg_simulation_config           = cfg_simulation_config
        self.interarrival_model_interarrival = interarrival_model_interarrival
        self.service_model_service_time      = service_model_service_time
        self.policy_scheduling_policy        = policy_scheduling_policy
        self.metrics_metrics_collector       = metrics_metrics_collector

        self.policy_scheduling_policy.priority_first_enabled_bool = bool(
            cfg_simulation_config.priority_first_enabled_bool
        )

        self.rng_rng = np.random.default_rng(cfg_simulation_config.seed_i32)
        self.calendar_event_calendar = Event_Calendar()

        self.queue_deque_task: Deque[Task] = deque()
        self.server_server = Server()

        self.now_f64              : float = 0.0
        self._next_task_id_i32    : int   = 0
        self._arrivals_count_i32  : int   = 0

        self._in_system_list_task: List[Task] = []

        # J-mode: number of tiers (default 2 for backward compat)
        self._num_tiers_i32: int = int(getattr(policy_scheduling_policy, 'num_tiers', 2))
        # Propagate J into the metrics collector for tier_fractions sizing
        self.metrics_metrics_collector.num_tiers_i32 = self._num_tiers_i32

    def _Make_Task(self, arrival_time_f64: float) -> Task:
        self._next_task_id_i32 += 1
        ttl_cfg = self.cfg_simulation_config.ttl_ttl_config
        ttl_mean_f64 = float(ttl_cfg.ttl_seconds_f64)
        ttl_std_f64 = float(ttl_cfg.ttl_std_f64)
        ttl_min_f64 = float(ttl_cfg.ttl_min_f64)
        ttl_max_opt = ttl_cfg.ttl_max_f64

        if ttl_std_f64 > 0.0:
            ttl_sample_f64 = float(self.rng_rng.normal(loc=ttl_mean_f64, scale=ttl_std_f64))
        else:
            ttl_sample_f64 = ttl_mean_f64

        ttl_f64 = max(ttl_sample_f64, ttl_min_f64)
        if ttl_max_opt is not None:
            ttl_f64 = min(ttl_f64, float(ttl_max_opt))
        deadline_f64 = arrival_time_f64 + ttl_f64
        priority_rate_f64 = float(self.cfg_simulation_config.priority_task_rate_f64)
        is_high_priority = bool(self.rng_rng.random() < priority_rate_f64)
        return Task(
            task_id=self._next_task_id_i32,
            arrival_time=arrival_time_f64,
            deadline=deadline_f64,
            high_priority_bool=is_high_priority,
        )

    def _State(self) -> System_State:
        remaining = 0.0
        if self.server_server.busy_bool and self.server_server.current_task_task_opt is not None:
            remaining = max(self.server_server.current_service_end_f64_opt - self.now_f64, 0.0)

        return System_State(
            now_f64=self.now_f64,
            queue_length_i32=len(self.queue_deque_task),
            server_busy_bool=self.server_server.busy_bool,
            current_task_task_opt=self.server_server.current_task_task_opt,
            server_remaining_time=remaining,
            queued_tasks_tuple=tuple(self.queue_deque_task),
        )

    def Initialize(self) -> None:
        self.calendar_event_calendar.Schedule(
            time_f64=self.cfg_simulation_config.until_time_f64,
            event_type_event_type=Event_Type.STOP,
            payload_any=None,
        )

        first_inter_f64 = self.interarrival_model_interarrival.Sample(self.rng_rng)
        first_arrival_time_f64 = 0.0 + first_inter_f64

        self.calendar_event_calendar.Schedule(
            time_f64=first_arrival_time_f64,
            event_type_event_type=Event_Type.ARRIVAL,
            payload_any=None,
        )

    def Run(self) -> dict:
        self.Initialize()

        while True:
            ev_event_opt = self.calendar_event_calendar.Pop_Next()
            if ev_event_opt is None:
                break

            self.now_f64 = ev_event_opt.time

            if ev_event_opt.event_type == Event_Type.STOP:
                break

                        
            self.metrics_metrics_collector.Record_Queue_Length(self.now_f64, len(self.queue_deque_task))
            self.metrics_metrics_collector.Record_Server_Busy(self.now_f64, self.server_server.busy_bool)
            self.metrics_metrics_collector.Record_Queue_Nonempty(self.now_f64, len(self.queue_deque_task) > 0)

            if ev_event_opt.event_type == Event_Type.ARRIVAL:
                self._Handle_Arrival()
            elif ev_event_opt.event_type == Event_Type.SERVICE_COMPLETE:
                self._Handle_Service_Complete(ev_event_opt)
            elif ev_event_opt.event_type == Event_Type.SWITCH_CHECK:
                self._Handle_Switch_Check(ev_event_opt)
            else:
                raise ValueError(f"Unknown event type: {ev_event_opt.event_type}")

                                                            
            self._Try_Start_Service()

                                                             
        if hasattr(self.metrics_metrics_collector, "Set_End_Time"):
            self.metrics_metrics_collector.Set_End_Time(self.now_f64)

                                                                   
        pending_list_task: List[Task] = list(self.queue_deque_task)

        if (
            self.server_server.current_task_task_opt is not None
            and self.server_server.current_task_task_opt.completion_time_f64_opt is None
        ):
            pending_list_task.append(self.server_server.current_task_task_opt)

        self.metrics_metrics_collector.Finalize_Unfinished(pending_list_task)
        for tk in pending_list_task:
            if hasattr(self.policy_scheduling_policy, "Observe_Task_Outcome"):
                self.policy_scheduling_policy.Observe_Task_Outcome(tk)

        return self.metrics_metrics_collector.Aggregate()

    def _Schedule_Next_Arrival(self) -> None:
        max_tasks_i32_opt = self.cfg_simulation_config.arrival_config.max_tasks_i32_opt
        if max_tasks_i32_opt is not None and self._arrivals_count_i32 >= max_tasks_i32_opt:
            return

        inter_f64 = self.interarrival_model_interarrival.Sample(self.rng_rng)
        t_f64 = self.now_f64 + inter_f64

        if t_f64 < self.cfg_simulation_config.until_time_f64:
            self.calendar_event_calendar.Schedule(
                time_f64=t_f64,
                event_type_event_type=Event_Type.ARRIVAL,
                payload_any=None,
            )

    def _Handle_Arrival(self) -> None:
        self._arrivals_count_i32 += 1
        task_ = self._Make_Task(self.now_f64)
        self._in_system_list_task.append(task_)

        # All policies are defined as arrival-epoch mode selectors in the paper setup.
        state_at_arrival = self._State()
        tier_decision = self.policy_scheduling_policy.Decide_Mode(task_, state_at_arrival)
        # Store tier index and map to Mode for backward compat
        J = self._num_tiers_i32
        task_.chosen_tier_i32 = int(tier_decision)
        task_.chosen_mode_mode_opt = _tier_to_mode(int(tier_decision), J)

        self.queue_deque_task.append(task_)

                               
        self._Schedule_Next_Arrival()

    def _Try_Start_Service(self) -> None:
        if self.server_server.busy_bool:
            return

                                                                             
        if self.cfg_simulation_config.drop_expired_in_queue_bool and self.queue_deque_task:
            self._Drop_All_Expired_In_Queue()

                                                                   
        task_task_opt = self.policy_scheduling_policy.Select_Task(self.queue_deque_task, self.now_f64)
        if task_task_opt is None:
            return

                                                                         
        if task_task_opt.Is_Expired(self.now_f64):
            task_task_opt.Mark_Dropped(self.now_f64)
            self.metrics_metrics_collector.Record_Task_Final(task_task_opt)
            if hasattr(self.policy_scheduling_policy, "Observe_Task_Outcome"):
                self.policy_scheduling_policy.Observe_Task_Outcome(task_task_opt)
            return

        # Resolve mode: use arrival-epoch decision if available, else re-query policy.
        J = self._num_tiers_i32
        if task_task_opt.chosen_mode_mode_opt is not None:
            mode_mode = task_task_opt.chosen_mode_mode_opt
        else:
            tier_decision = self.policy_scheduling_policy.Decide_Mode(task_task_opt, self._State())
            task_task_opt.chosen_tier_i32 = int(tier_decision)
            mode_mode = _tier_to_mode(int(tier_decision), J)
            task_task_opt.chosen_mode_mode_opt = mode_mode
                                                                                             
        task_task_opt.Mark_Started(self.now_f64, mode_mode=mode_mode, num_tiers=J)

        # Sample service time.
        # For J>2, prefer Sample_By_Tier to avoid misrouting middle tiers
        # through the binary Mode mapping. For J=2, use the legacy Mode path.
        tier_idx = task_task_opt.chosen_tier_i32
        if J > 2 and tier_idx is not None:
            sample_by_tier_for_task = getattr(self.service_model_service_time, "Sample_By_Tier_For_Task", None)
            sample_by_tier = getattr(self.service_model_service_time, "Sample_By_Tier", None)
            if callable(sample_by_tier_for_task):
                service_time_f64 = float(sample_by_tier_for_task(task_task_opt, tier_idx, self.rng_rng))
            elif callable(sample_by_tier):
                service_time_f64 = float(sample_by_tier(tier_idx, self.rng_rng))
            else:
                # Fallback: Mode-based dispatch (will misroute middle tiers)
                sample_for_task = getattr(self.service_model_service_time, "Sample_For_Task", None)
                if callable(sample_for_task):
                    service_time_f64 = float(sample_for_task(task_task_opt, mode_mode, self.rng_rng))
                else:
                    service_time_f64 = float(self.service_model_service_time.Sample(mode_mode, self.rng_rng))
        else:
            # J=2: use existing Mode-based path (exact backward compat)
            sample_for_task = getattr(self.service_model_service_time, "Sample_For_Task", None)
            if callable(sample_for_task):
                service_time_f64 = float(sample_for_task(task_task_opt, mode_mode, self.rng_rng))
            else:
                service_time_f64 = float(self.service_model_service_time.Sample(mode_mode, self.rng_rng))
        task_task_opt.service_time_f64_opt = service_time_f64

        self.server_server.busy_bool = True
        self.server_server.current_task_task_opt = task_task_opt
        self.server_server.current_service_end_f64_opt = self.now_f64 + service_time_f64

        self.calendar_event_calendar.Schedule(
            time_f64=self.server_server.current_service_end_f64_opt,
            event_type_event_type=Event_Type.SERVICE_COMPLETE,
            payload_any=task_task_opt,
        )

                                                        
        if self.cfg_simulation_config.policy_config.enable_mode_switching_bool:
            quantum_f64 = max(float(self.cfg_simulation_config.policy_config.switch_check_quantum_f64), 1e-6)
            next_check_f64 = self.now_f64 + quantum_f64

            if (
                self.server_server.current_service_end_f64_opt is not None
                and next_check_f64 < self.server_server.current_service_end_f64_opt
            ):
                self.calendar_event_calendar.Schedule(
                    time_f64=next_check_f64,
                    event_type_event_type=Event_Type.SWITCH_CHECK,
                    payload_any=task_task_opt,
                )

    def _Handle_Service_Complete(self, ev_event: Event) -> None:
        task_: Task = ev_event.payload

                                                                                            
        if (
            self.server_server.current_task_task_opt is None
            or task_.task_id != self.server_server.current_task_task_opt.task_id
        ):
            return

        start_service_time_f64 = task_.start_service_time_f64_opt or self.now_f64
        task_.service_consumed_f64 = max(self.now_f64 - start_service_time_f64, 0.0)
        task_.Mark_Completed(self.now_f64)

        if task_.service_time_f64_opt is not None and task_.chosen_mode_mode_opt is not None:
            if hasattr(self.service_model_service_time, "Observe"):
                self.service_model_service_time.Observe(
                    task_.chosen_mode_mode_opt,
                    float(task_.service_time_f64_opt),
                )
            if hasattr(self.service_model_service_time, "Snapshot"):
                snapshot_dict_obj = self.service_model_service_time.Snapshot()
                self.metrics_metrics_collector.Record_EWMA_Estimates(
                    now_f64=self.now_f64,
                    slow_ewma_f64=float(snapshot_dict_obj.get("slow_ewma", float("nan"))),
                    fast_ewma_f64=float(snapshot_dict_obj.get("fast_ewma", float("nan"))),
                    slow_count_i32=int(snapshot_dict_obj.get("slow_count", 0)),
                    fast_count_i32=int(snapshot_dict_obj.get("fast_count", 0)),
                )

                        
        self.server_server.busy_bool = False
        self.server_server.current_task_task_opt = None
        self.server_server.current_service_end_f64_opt = None

        self.metrics_metrics_collector.Record_Task_Final(task_)
        if hasattr(self.policy_scheduling_policy, "Observe_Task_Outcome"):
            self.policy_scheduling_policy.Observe_Task_Outcome(task_)

    def _Handle_Switch_Check(self, ev_event: Event) -> None:
        if not self.cfg_simulation_config.policy_config.enable_mode_switching_bool:
            return

        task_: Task = ev_event.payload
        if (
            self.server_server.current_task_task_opt is None
            or task_.task_id != self.server_server.current_task_task_opt.task_id
        ):
            return

                                                 
        if task_.completion_time_f64_opt is not None:
            return

                                                                 
        if task_.start_service_time_f64_opt is not None:
            task_.service_consumed_f64 = max(self.now_f64 - task_.start_service_time_f64_opt, 0.0)

                                      
        if self.policy_scheduling_policy.Should_Switch_Mode(task_, self._State()):
                                                                          
            task_.mode_switches_i32 += 1
            # TODO(Prompt 4): For J>2, a policy may want to switch to an
            # intermediate tier (e.g. tier 2 → tier 1) rather than always
            # downgrading to tier 0.  Revisit when FTC-SE and other
            # mode-switching policies arrive.
            task_.chosen_mode_mode_opt = Mode.FAST
            task_.chosen_tier_i32 = 0  # tier 0 = fastest

                                                                  
            sample_for_task = getattr(self.service_model_service_time, "Sample_For_Task", None)
            if callable(sample_for_task):
                remaining_service_f64 = float(sample_for_task(task_, Mode.FAST, self.rng_rng))
            else:
                remaining_service_f64 = float(self.service_model_service_time.Sample(Mode.FAST, self.rng_rng))

                                                                                     
            task_.service_time_f64_opt = float(task_.service_consumed_f64 + remaining_service_f64)

            new_end_f64 = self.now_f64 + remaining_service_f64
            self.server_server.current_service_end_f64_opt = new_end_f64

            self.calendar_event_calendar.Schedule(
                time_f64=new_end_f64,
                event_type_event_type=Event_Type.SERVICE_COMPLETE,
                payload_any=task_,
            )

                                                                            
            return

                                                                              
        quantum_f64 = max(float(self.cfg_simulation_config.policy_config.switch_check_quantum_f64), 1e-6)
        next_check_f64 = self.now_f64 + quantum_f64

        if (
            self.server_server.current_service_end_f64_opt is not None
            and next_check_f64 < self.server_server.current_service_end_f64_opt
        ):
            self.calendar_event_calendar.Schedule(
                time_f64=next_check_f64,
                event_type_event_type=Event_Type.SWITCH_CHECK,
                payload_any=task_,
            )

    def _Drop_All_Expired_In_Queue(self) -> None:
        if not self.queue_deque_task:
            return

        survivors_deque_task: Deque[Task] = deque()

        while self.queue_deque_task:
            task_ = self.queue_deque_task.popleft()
            if task_.Is_Expired(self.now_f64):
                task_.Mark_Dropped(self.now_f64)
                self.metrics_metrics_collector.Record_Task_Final(task_)
                if hasattr(self.policy_scheduling_policy, "Observe_Task_Outcome"):
                    self.policy_scheduling_policy.Observe_Task_Outcome(task_)
            else:
                survivors_deque_task.append(task_)

        self.queue_deque_task = survivors_deque_task
