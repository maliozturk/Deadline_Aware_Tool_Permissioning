# =============================================================================
#  FIRM-DEADLINE TOOL CONTROL (FTC) / CADTR
#  Product Signature: FTC
# ------------------------------------------------------------------------------
#  File: Core/Events.py
#  Purpose: Define discrete-event types and the event calendar.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

import heapq
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple


class Event_Type(str, Enum):
    ARRIVAL          = "arrival"
    SERVICE_COMPLETE = "service_complete"
    SWITCH_CHECK     = "switch_check"                                          
    STOP             = "stop"


@dataclass(frozen=True)
class Event:

    time       : float
    seq        : int
    event_type : Event_Type
    payload    : Any = None

    def Key(self) -> Tuple[float, int]:
        return (self.time, self.seq)


class Event_Calendar:

    def __init__(self) -> None:
        self._heap_list_tuple_f64_i32_event : List[Tuple[float, int, Event]] = []
        self._seq_i32                      : int = 0

    def Schedule(self, time_f64: float, event_type_event_type: Event_Type, payload_any: Any = None) -> None:
        if time_f64 < 0:
            raise ValueError("Event time must be non-negative.")

        self._seq_i32 += 1
        ev_event = Event(time=time_f64, seq=self._seq_i32, event_type=event_type_event_type, payload=payload_any)
        heapq.heappush(self._heap_list_tuple_f64_i32_event, (ev_event.time, ev_event.seq, ev_event))

    def Pop_Next(self) -> Optional[Event]:
        if not self._heap_list_tuple_f64_i32_event:
            return None
        return heapq.heappop(self._heap_list_tuple_f64_i32_event)[2]

    def Peek_Time(self) -> Optional[float]:
        if not self._heap_list_tuple_f64_i32_event:
            return None
        return self._heap_list_tuple_f64_i32_event[0][0]

    def __len__(self) -> int:
        return len(self._heap_list_tuple_f64_i32_event)
