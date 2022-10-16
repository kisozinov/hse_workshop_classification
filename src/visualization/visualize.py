import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def plot_roc(y_test, y_probs, auc):
    fpr1, tpr1, thresholds1 = roc_curve(y_test, y_probs)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr1, tpr1, label='ROC curve (area = %0.2f)' % auc, marker='.')
    plt.title("ROC curve")
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend(loc="lower right")
    plt.show()