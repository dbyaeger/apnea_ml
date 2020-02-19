from sklearn.preprocessing import normalize
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_confusions(cm, xlabel, filename):
    cm = normalize(cm, norm='l1', axis=1, copy=True)
    classes = preproc.classes
    matplotlib.rcParams['figure.figsize'] = (8, 7)
    plt.pcolor(np.flipud(cm), cmap="Blues")
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16) 
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks + .5, classes, rotation=90, fontsize=16)
    plt.yticks(tick_marks + .5, reversed(classes), fontsize=16)
    plt.clim(0, 1)
    plt.ylabel("Committee consensus label", fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename,
                dpi=400,
                format='pdf',
                bbox_inches='tight')