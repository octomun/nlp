from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def class_bar_plot(class_count, save_path):
    class_acc, class_label_count = class_count
    keys = list(class_acc.keys())
    vals = [float(class_acc[k]/class_label_count[k]) for k in keys]
    ax = sns.barplot(x=keys, y=vals)
    xlbl = ax.get_xticklabels()
    for i, p in enumerate(ax.patches): #patches
        print(p)
        lagend = xlbl[i].get_text()
        print(lagend)
        height = p.get_height()
        ax.text(p.get_x() , height, '{}/{}\n{}%'.format(class_acc[lagend], class_label_count[lagend],np.round(height,2)*100))

    plt.savefig(save_path)