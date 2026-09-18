# Real-time training monitor
import time
import sys
import os
import psutil
import threading
from datetime import datetime
from pathlib import Path

class TrainingMonitor:
    """Real-time monitoring of training progress and system health."""

    def __init__(self, log_file='results/monitor_log.txt'):
        self.log_file = log_file
        self.start_time = datetime.now()
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = self.initial_memory
        self.batch_count = 0
        self.alerts = []

        Path('results').mkdir(exist_ok=True)
        self.log(f"Monitor started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Initial memory: {self.initial_memory:.1f} MB")

    def log(self, message):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')

    def check_batch(self, batch_num, loss_value):
        """Check batch for anomalies."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

        self.batch_count = batch_num

        # Check for NaN/Inf
        try:
            if str(loss_value) == 'nan':
                alert = f"ALERT: NaN loss detected at batch {batch_num}"
                self.alerts.append(alert)
                self.log(alert)
                return False

            if str(loss_value) == 'inf':
                alert = f"ALERT: Inf loss detected at batch {batch_num}"
                self.alerts.append(alert)
                self.log(alert)
                return False
        except:
            pass

        # Check for memory spike
        memory_growth = current_memory - self.initial_memory
        if memory_growth > 500:
            alert = f"WARNING: High memory growth {memory_growth:.1f} MB at batch {batch_num}"
            self.alerts.append(alert)
            self.log(alert)

        # Log progress every 500 batches
        if batch_num % 500 == 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = batch_num / elapsed if elapsed > 0 else 0
            self.log(f"Batch {batch_num:5d} | Loss: {loss_value:10.4f} | "
                    f"Memory: {current_memory:7.1f} MB (+{memory_growth:6.1f}) | "
                    f"Rate: {rate:6.1f} batches/sec")

        return True

    def check_convergence(self, recent_losses):
        """Check if model is converging."""
        if len(recent_losses) < 10:
            return True

        recent = recent_losses[-10:]
        # Check if all losses are valid numbers
        try:
            valid_losses = [l for l in recent if str(l) not in ['nan', 'inf', '-inf']]
            if len(valid_losses) < 5:
                self.log("WARNING: Too many invalid values in recent losses")
                return False
        except:
            pass

        return True

    def finalize(self, r2_train, r2_test, mse_train, mse_test, training_time):
        """Log final results."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.log("\n" + "="*70)
        self.log("TRAINING COMPLETE")
        self.log("="*70)
        self.log(f"Total batches: {self.batch_count}")
        self.log(f"Training time: {training_time:.2f} seconds")
        self.log(f"Wall clock time: {elapsed:.2f} seconds")
        self.log(f"Peak memory: {self.peak_memory:.1f} MB")
        self.log(f"\nResults:")
        self.log(f"  Train R2: {r2_train:.4f}")
        self.log(f"  Test R2:  {r2_test:.4f}")
        self.log(f"  Train MSE: {mse_train:.4f}")
        self.log(f"  Test MSE:  {mse_test:.4f}")

        if self.alerts:
            self.log(f"\nAlerts during training: {len(self.alerts)}")
            for alert in self.alerts:
                self.log(f"  - {alert}")
        else:
            self.log("\nNo alerts - clean training run!")

        self.log("="*70)

# Global monitor instance
_monitor = None

def get_monitor():
    """Get or create global monitor."""
    global _monitor
    if _monitor is None:
        _monitor = TrainingMonitor()
    return _monitor

def init_monitor():
    """Initialize monitoring."""
    return get_monitor()

def monitor_batch(batch_num, loss_value):
    """Monitor a training batch."""
    monitor = get_monitor()
    return monitor.check_batch(batch_num, loss_value)

def monitor_losses(recent_losses):
    """Check convergence on recent losses."""
    monitor = get_monitor()
    return monitor.check_convergence(recent_losses)

def finalize_monitor(r2_train, r2_test, mse_train, mse_test, training_time):
    """Finalize monitoring."""
    monitor = get_monitor()
    monitor.finalize(r2_train, r2_test, mse_train, mse_test, training_time)
