# Helper functions

# Memory tracking utilities for monitoring RAM usage during incremental training.

import os
import psutil
import pandas as pd


def get_memory_mb():
    """
    Current RAM used by this Python process, in megabytes.

    Uses RSS (Resident Set Size) -- the physical RAM actually held by the
    process, which is the number that matters for "will this fit in memory".

    Returns:
        float: Resident memory in MB
    """
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def get_system_memory_mb():
    """
    System-wide RAM snapshot, in megabytes.

    Returns:
        dict: total, available, and percent-used for the whole machine
    """
    vm = psutil.virtual_memory()
    return {
        'total_mb': vm.total / (1024 ** 2),
        'available_mb': vm.available / (1024 ** 2),
        'percent_used': vm.percent
    }


class MemoryTracker:
    """
    Records process memory at labelled checkpoints during training.

    The point of incremental learning is that memory stays flat no matter how
    much data streams through. This class collects the evidence for that claim:
    call log() at each batch, then summary() to see whether memory grew.

    Example:
        tracker = MemoryTracker()
        tracker.log('start')
        for i, (X_b, y_b) in enumerate(batches):
            model.train_on_batch(X_b, y_b)
            tracker.log('batch', batch=i)
        print(tracker.summary())
    """

    def __init__(self):
        self.records = []
        self.baseline_mb = get_memory_mb()

    def log(self, label, batch=None):
        """
        Record a memory reading.

        Args:
            label: What is happening right now (e.g. 'after_data_load')
            batch: Optional batch number, for readings inside the training loop

        Returns:
            float: The memory reading in MB
        """
        mb = get_memory_mb()
        self.records.append({
            'label': label,
            'batch': batch,
            'memory_mb': mb,
            'delta_from_start_mb': mb - self.baseline_mb
        })
        return mb

    def to_dataframe(self):
        """
        All recorded readings as a DataFrame.

        Returns:
            pd.DataFrame: One row per log() call
        """
        return pd.DataFrame(self.records)

    def summary(self):
        """
        Peak/start/end memory across every reading taken.

        Returns:
            dict: start_mb, end_mb, peak_mb, growth_mb, n_readings
        """
        if not self.records:
            return {}

        readings = [r['memory_mb'] for r in self.records]
        return {
            'start_mb': readings[0],
            'end_mb': readings[-1],
            'peak_mb': max(readings),
            'growth_mb': readings[-1] - readings[0],
            'n_readings': len(readings)
        }


if __name__ == "__main__":
    print(f"Process memory: {get_memory_mb():.1f} MB")

    sys_mem = get_system_memory_mb()
    print(f"System total:   {sys_mem['total_mb']:.0f} MB")
    print(f"System free:    {sys_mem['available_mb']:.0f} MB")
    print(f"System in use:  {sys_mem['percent_used']:.1f}%")

    # Show the tracker noticing an allocation
    tracker = MemoryTracker()
    tracker.log('before')
    blob = [0.0] * 5_000_000          # ~40 MB
    tracker.log('after_allocating')
    del blob

    print()
    print(tracker.to_dataframe().to_string(index=False))
