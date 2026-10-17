"""Per-operation UTC replay clock; affects only the dedicated judge process."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime,timezone
class ReplayClock:
 def __init__(self):self.context=ContextVar('judge_reference_time',default=None)
 def now(self):return self.context.get() or datetime.now(timezone.utc)
 def stamp(self):return self.now().isoformat()
 @contextmanager
 def at(self,value):
  token=self.context.set(value)
  try:yield
  finally:self.context.reset(token)
