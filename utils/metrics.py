from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics_fn(eval_pred):
    predictions, labels = eval_pred
    # Nehme den argmax, um die höchste Wahrscheinlichkeit als Vorhersage zu nehmen
    preds = np.argmax(predictions, axis=-1)
    
    # Überprüfen, ob es sich um mehrdimensionale Arrays handelt
    if preds.ndim > 1:
        accuracies = [accuracy_score(label, pred) for label, pred in zip(labels, preds)]
        accuracy = np.mean(accuracies)
    else:
        accuracy = accuracy_score(labels, preds)
    
    return {"accuracy": accuracy}