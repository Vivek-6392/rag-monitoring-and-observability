"""
Lightweight tracing — no Jaeger needed on Streamlit Cloud.
Spans are stored in-memory per request and flushed to SQLite.
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    attributes: dict = field(default_factory=dict)

    def end(self, **attrs):
        self.end_time = time.perf_counter()
        self.attributes.update(attrs)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class Tracer:
    """Per-request tracer that collects spans."""

    def __init__(self):
        self.trace_id = uuid.uuid4().hex
        self.spans: list[Span] = []

    def start_span(self, name: str) -> Span:
        span = Span(name=name, trace_id=self.trace_id)
        self.spans.append(span)
        return span

    def summary(self) -> dict:
        result = {"trace_id": self.trace_id, "spans": []}
        for s in self.spans:
            result["spans"].append(
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 2),
                    **s.attributes,
                }
            )
        return result

    def get_span_duration(self, name: str) -> float:
        for s in self.spans:
            if s.name == name:
                return s.duration_ms
        return 0.0