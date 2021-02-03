import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from InfraredSolarModules.utils.plot_confusion_matrix import plot_confusion_matrix


def generate_cm(label, pred):
    cm = confusion_matrix(label, pred.argmax(dim=1))
    return cm


