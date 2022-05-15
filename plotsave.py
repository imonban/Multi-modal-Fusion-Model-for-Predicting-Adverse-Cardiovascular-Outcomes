import roc_utils 
from matplotlib import pyplot as plt
from roc_utils import compute_roc,plot_roc
import matplotlib.colors as mcolors
 

def plotROC_outcomes(true_variables, pred_variables, outcomes, imagename):
    rocs_boost = []
    _, ax3 = plt.subplots(figsize=(15,10))
    names = list(mcolors.CSS4_COLORS)
    for i in range(len(outcomes)):
        rocs_boost.append(compute_roc(X=pred_variables[i] , y=true_variables[i], pos_label=1, objective=["minopt", "minoptsym","acc", "cohen"]))
        plot_roc(rocs_boost[i], label=outcomes[i],color=names[i+15], ax=ax3)
    ax3.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax3.set_title("ROC curves");
    plt.savefig(imagename, dpi = 300, bbox_inches = 'tight')
