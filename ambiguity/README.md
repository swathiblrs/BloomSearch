# Ambiguity Detection
## Using the ambiguity detection model
To use the model, use import statement `from model import detect_ambiguity`
Then, you can use the `detect_ambiguity` function, which takes a string as input and outputs ambiguity_score, needs_clarification, and clarification_question.
An example of how to use the function can be found in `test_detector.py`
## Current stats of the ambiguity detector:
Accuracy:                 0.6379
Precision:                0.6306
Recall:                   0.9206
F1-score:                 0.7485