# =============================================================================
#  DEADLINE-AWARE TOOL PERMISSIONING (DATP)
#  Product Signature: DATP
# ------------------------------------------------------------------------------
#  File: Models/Policy_Base.py
#  Purpose: Define scheduling policy interfaces and system state snapshot.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Protocol

from Core.Task import Mode, Task


@dataclass(frozen=True)
class System_State:

    now_f64                : float
    queue_length_i32       : int
    server_busy_bool       : bool
    current_task_task_opt  : Optional[Task]

    server_remaining_time  : float


class Scheduling_Policy(Protocol):

    def Select_Task(self, queue_deque_task: Deque[Task], now_f64: float) -> Optional[Task]:
        while queue_deque_task:
            head_task = queue_deque_task[0]
            if head_task.Is_Expired(now_f64):
                head_task = queue_deque_task.popleft()
                head_task.Mark_Dropped(now_f64)
                continue
            break
        else:
            return None

        priority_first = bool(getattr(self, "priority_first_enabled_bool", False))
        if not priority_first:
            return queue_deque_task.popleft()

        selected_index = None
        for idx, task in enumerate(queue_deque_task):
            if task.Is_Expired(now_f64):
                continue
            if getattr(task, "high_priority_bool", False):
                selected_index = idx
                break

        if selected_index is None:
            return queue_deque_task.popleft()

        selected_task: Optional[Task] = None
        new_queue: Deque[Task] = deque()
        for idx, task in enumerate(queue_deque_task):
            if idx == selected_index:
                selected_task = task
                continue
            new_queue.append(task)

        queue_deque_task.clear()
        queue_deque_task.extend(new_queue)
        return selected_task

    def Decide_Mode(self, task: Task, state_system_state: System_State) -> Mode:
        ...

    def Should_Switch_Mode(self, task: Task, state_system_state: System_State) -> bool:
        ...
