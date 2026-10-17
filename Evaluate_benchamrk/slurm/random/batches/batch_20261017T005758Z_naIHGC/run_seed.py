"""Run an unchanged benchmark seed with concurrency-safe summary aggregation."""
import fcntl
from pathlib import Path
import sys

batch = Path(__file__).resolve().parent
sys.path.insert(0, str(batch / 'code/Evaluate_benchamrk'))
import run

original_aggregate = run.aggregate


def serialized_aggregate(group, *args, **kwargs):
    # Different seed tasks can finish together. Queue their short summary
    # writes rather than let the benchmark's nonblocking lock reject a task.
    group = Path(group)
    group.mkdir(parents=True, exist_ok=True)
    with (group / '.aggregation_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return original_aggregate(group, *args, **kwargs)


run.aggregate = serialized_aggregate
if __name__ == '__main__':
    run.main()
