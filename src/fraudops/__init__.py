"""FraudOps — real-time fraud detection with a closed-loop MLOps pipeline.

Replays the IEEE-CIS transaction dataset as a chronological stream, scores it
in real time, monitors data and performance drift, and retrains and promotes a
challenger model only when it beats the champion under a cost-based gate.
"""

__version__ = "0.1.0"
