from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import numpy as np


def weighted_f1(y_true, y_pred, convert_to_int=False):
    ## if y_pred is float, convert it to int
    if convert_to_int:
        y_pred = y_pred.round().astype(int)
        print("y_pred is float, converting to int")

    return f1_score(y_true, y_pred, average="weighted")
