# src/inference/pipeline.py

import joblib

def classify_ticket(query_result):
    """Classify the user input into a department using the pre-trained model."""
    model = joblib.load('modelsvm.pk1')
    result = model.predict([query_result])
    return result[0]
