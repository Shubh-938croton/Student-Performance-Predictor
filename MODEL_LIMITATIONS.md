# Model Limitations

## Dataset

- Training data contains 10,000 rows.
- Placement distribution:
  - Placed: 8356
  - Not Placed: 1644

## Feature Ranges

- Study Hours: 1–11
- Attendance: 40–100
- Sleep Hours: 4–9
- Internet Usage: 1–11
- Assignments Completed: 0–20
- Previous Score: 35–95

## Observations

- Model should not be used outside these ranges.
- Input validation added to Flask application.
- Better dataset required for wider generalization.

## Future Improvements

- Use real-world placement dataset.
- Add more predictive features.
- Improve class balance.
- Add SHAP explainability.