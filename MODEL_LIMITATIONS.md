# Model Limitations

## Dataset Limitations

The model was trained on a dataset containing **10,000 records**.

### Class Distribution

- Placed: 8,356
- Not Placed: 1,644

The dataset is imbalanced, with significantly more placed students than
not-placed students. Therefore, accuracy alone may not fully represent
model performance.

## Training Feature Ranges

The model was trained using values approximately within the following ranges:

| Feature | Training Range |
|---|---|
| Study Hours | 1–11 hours |
| Attendance | 40–100% |
| Sleep Hours | 4–9 hours |
| Internet Usage | 1–11 hours |
| Assignments Completed | 0–20 |
| Previous Score | 35–95 |

Predictions for values significantly outside these ranges may be unreliable
because the model has not been trained on such examples.

The Flask application performs input validation to reduce invalid or
out-of-distribution inputs.

## Generalization Limitations

- The model's predictions depend heavily on the quality of the training data.
- The dataset may not accurately represent students from different colleges,
  regions, courses, or placement processes.
- Placement depends on many factors that are not included in the current
  feature set.
- The prediction should not be treated as a guarantee of actual placement.

## Future Improvements

- Train on a real-world placement dataset.
- Collect data from multiple institutions.
- Add additional predictive features.
- Improve handling of class imbalance.
- Evaluate precision, recall, F1-score, and ROC-AUC in addition to accuracy.
- Add SHAP-based model explainability.
