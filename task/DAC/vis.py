import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

font_scale = 1.2

names = ['R-ETIOL', 'R-PRCTN', 'R-MA', 'I-ETIOL', 'DIAG', 'R-PC', 'R-DR', 'I-MA',
         'R-EET', 'I-PC', 'I-PRCTN', 'I-EET', 'I-DR', 'R-SX', 'I-SX', 'OTHER']
save_path = 'task/DAC/BERT-DAC/ernie_predictions.npz'
confusion = np.load(save_path)['test_confusion']


sns.set_context('paper', font_scale=font_scale)
ax = sns.heatmap(confusion / confusion.sum(axis=0), center=1.0, square=True, linewidths=0.1, cmap='Blues',
                 cbar_kws={"shrink": 1.0}, xticklabels=names, yticklabels=names, annot=False)
ax.xaxis.tick_top()
plt.xticks(rotation=90)
plt.tight_layout()
# plt.show()
plt.savefig('da_ernie_confusion.png', dpi=1200)

