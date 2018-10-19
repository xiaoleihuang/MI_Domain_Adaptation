# for multi-classes
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle
# save figure to pdf
from matplotlib.backends.backend_pdf import PdfPages

def plot_roc(code_test, y_pred, n_classes=3):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if n_classes == 3:

        classes_mapping = ['fn', 'pos code', 'neg code']

        # Compute ROC curve and ROC area for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(code_test[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        with PdfPages('roc_patients.pdf') as pdf:
            plt.figure(figsize=(10, 8))

            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                         ''.format(classes_mapping[i], roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=16)
            plt.ylabel('True Positive Rate', fontsize=16)
            plt.title('ROC curves of Patients\' Behaviors Prediction', fontsize=24)
            plt.legend(loc="lower right", fontsize=16)
            pdf.savefig()
            plt.close()

    elif n_classes == 2:
        fpr, tpr, thresholds_roc = roc_curve(code_test, y_pred)
        roc_auc = auc(fpr, tpr)

        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")

        plt.show()


# This script is to load t