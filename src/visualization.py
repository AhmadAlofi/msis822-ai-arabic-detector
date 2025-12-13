import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def plot_confusion(cm, labels=("Human","AI"), title="Confusion matrix"):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))
    disp.plot(values_format="d")
    plt.title(title)
    plt.show()

def plot_roc(y_true, scores, title="ROC curve"):
    RocCurveDisplay.from_predictions(y_true, scores)
    plt.title(title)
    plt.show()
