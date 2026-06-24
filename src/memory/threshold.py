import numpy as np
import torch

class DynamicThresholdTracker:
    """
    Sliding percentile tracker with circular buffer of size N=10000.
    Computes tau_d = P99(B_d) recomputed every 1000 new benign samples.
    Supports a Generic IoT fallback with tau_generic = P95.
    """
    def __init__(self, buffer_size: int = 10000, update_interval: int = 1000, 
                 percentile: float = 0.99, initial_threshold: float = 1.5):
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.percentile = percentile
        
        self.buffer = np.zeros(buffer_size)
        self.pointer = 0
        self.count = 0
        self.is_full = False
        self.samples_since_update = 0
        self.threshold = initial_threshold

    def update(self, scores: np.ndarray):
        """
        Appends new benign anomaly scores to the circular buffer.
        Triggers threshold recalculation if the update interval is reached.
        """
        # If scores is scalar, convert to array
        if np.isscalar(scores):
            scores = np.array([scores])
            
        for score in scores:
            self.buffer[self.pointer] = score
            self.pointer = (self.pointer + 1) % self.buffer_size
            self.count = min(self.count + 1, self.buffer_size)
            if self.pointer == 0:
                self.is_full = True
                
            self.samples_since_update += 1
            
        # Recompute threshold if update interval is reached
        if self.samples_since_update >= self.update_interval:
            self.recompute_threshold()
            self.samples_since_update = 0

    def recompute_threshold(self):
        """
        Computes the percentile of the current buffer.
        """
        if self.count == 0:
            return
            
        valid_scores = self.buffer[:self.count] if not self.is_full else self.buffer
        self.threshold = float(np.percentile(valid_scores, self.percentile * 100))

    def is_anomaly(self, score: float) -> bool:
        """
        Checks if a given anomaly score exceeds the calculated threshold.
        """
        return score > self.threshold


class DeviceThresholdManager:
    """
    Manages sliding thresholds for multiple devices and a fallback Generic IoT bank.
    """
    def __init__(self, buffer_size: int = 10000, update_interval: int = 1000,
                 benign_percentile: float = 0.99, generic_percentile: float = 0.95):
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.benign_percentile = benign_percentile
        self.generic_percentile = generic_percentile
        
        # Dictionary of device_id -> DynamicThresholdTracker
        self.trackers = {}
        
        # Fallback Generic IoT tracker
        self.generic_tracker = DynamicThresholdTracker(
            buffer_size=buffer_size,
            update_interval=update_interval,
            percentile=generic_percentile,
            initial_threshold=2.0 # conservative fallback
        )

    def get_tracker(self, device_id: int) -> DynamicThresholdTracker:
        if device_id not in self.trackers:
            self.trackers[device_id] = DynamicThresholdTracker(
                buffer_size=self.buffer_size,
                update_interval=self.update_interval,
                percentile=self.benign_percentile,
                initial_threshold=1.5
            )
        return self.trackers[device_id]

    def update_benign(self, device_id: int, score: float, is_ambiguous: bool = False):
        if is_ambiguous:
            self.generic_tracker.update(np.array([score]))
        else:
            tracker = self.get_tracker(device_id)
            tracker.update(np.array([score]))

    def get_threshold(self, device_id: int, is_ambiguous: bool = False) -> float:
        if is_ambiguous:
            return self.generic_tracker.threshold
        return self.get_tracker(device_id).threshold

    def verify_anomaly(self, device_id: int, score: float, is_ambiguous: bool = False) -> bool:
        threshold = self.get_threshold(device_id, is_ambiguous)
        return score > threshold
