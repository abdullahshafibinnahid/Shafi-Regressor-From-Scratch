# Shafi-Regressor-From-Scratch
# Shafi Regressor(SR) from Scratch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Traditional Random Forests predict the *average* value, which leaves you blind to uncertainty. This repository contains a **pure NumPy-optimized implementation of Shafi Regressor (SR)** built completely from scratch, allowing you to estimate exact prediction intervals and uncertainty bounds.



---

## ✨ Key Features
- **Pure NumPy Core:** High-performance vectorized calculation of quantiles.
- **Uncertainty Quantification:** Predicts any quantile ($\alpha = 0.1, 0.5, 0.9$, etc.) to build confidence intervals.
- **Scikit-Learn Compatible:** Seamlessly trains on top of standard tree architectures using optimized leaf addressing.
- **Built-in Serialization:** One-click `save()` and `load()` features for easy model deployment.

---

## 🚀 Quick Start & Usage

### 1. Training and Saving the Model
```python
from src import ShafiRegressor
import numpy as np

# Initialize and Fit
model = ShafiRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save for later
model.save("models/my_qrf_model.pkl")
