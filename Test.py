import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

matrix = 5 * np.eye(5)

sns.heatmap(matrix, annot=True, cmap="Blues", xticklabels=["number 1", "number 2", "number 3", "number 4", "number 5"], yticklabels=["number 1", "number 2", "number 3", "number 4", "number 5"])

plt.savefig("filename.pdf", dpi=300)

plt.show()
