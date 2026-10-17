"""P5 gate with exact microsecond-to-nanosecond cutoff arithmetic.

P5's original derive method multiplies timestamp seconds by 1e9. The
float conversion can round a same-time observation out of the visible set.
This local datetime adapter preserves that method and all gate rules while
making its nanosecond scaling exact. No temporal tolerance is increased.
"""
from datetime import datetime,timezone
from patient_readiness.service import Readiness


def cutoff_ns(reference):
    epoch=datetime(1970,1,1,tzinfo=timezone.utc)
    delta=reference.astimezone(timezone.utc)-epoch
    return (delta.days*86400+delta.seconds)*1_000_000_000+delta.microseconds*1000


class _ExactEpochSeconds(float):
    def __new__(cls,nanoseconds):
        instance=super().__new__(cls,nanoseconds/1_000_000_000)
        instance.nanoseconds=nanoseconds
        return instance

    def __mul__(self,factor):
        if factor==1_000_000_000:return self.nanoseconds
        return super().__mul__(factor)


class _ExactReference(datetime):
    def timestamp(self):return _ExactEpochSeconds(cutoff_ns(self))


class InferenceReadiness(Readiness):
    def _derive(self,pid,reference):
        exact=_ExactReference.fromisoformat(reference.isoformat())
        return super()._derive(pid,exact)
