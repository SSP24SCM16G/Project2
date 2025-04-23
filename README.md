# Gradient Boosting Tree Classifier from Scratch

## What does the model you have implemented do and when should it be used?
This project implements a **Gradient Boosting Tree classifier** from first principles, based on sections 10.9–10.10 of *The Elements of Statistical Learning* (2nd Edition). It can be used for binary classification tasks, especially where decision tree ensembles are appropriate.

## How did you test your model to determine if it is working reasonably correctly?
I wrote tests using small, synthetic datasets with known patterns and verified that the model achieved at least 75% accuracy. All tests are placed in the `tests` folder and can be run using `pytest`.

## What parameters have you exposed to users of your implementation in order to tune performance?
- `n_estimators`: Number of boosting iterations.
- `learning_rate`: Shrinks contribution of each tree.

### Usage Example:
```python
from boosting.boosting import GradientBoostingClassifier
import numpy as np

X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)
clf.fit(X, y)
y_pred = clf.predict(X)

## Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?
The implementation currently supports only binary classification.
It may overfit small datasets if learning_rate and n_estimators are not tuned carefully.
There is no regularization or pruning in individual trees, which could make the model sensitive to noise.

### Given more time, I would:
Extend support for multi-class classification.
Add regularization and pruning for trees.
Improve performance and memory handling for large datasets.


## How to Run

1. Clone the repo and create a virtual environment:
```bash
python -m venv venv
# On Windows
venv\\Scripts\\activate
# On Linux/macOS
source venv/bin/activate
