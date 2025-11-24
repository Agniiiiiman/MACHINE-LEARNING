'''Gene expression refers to how actively a gene is producing its product (RNA or protein).
To measure this, techniques such as:DNA Microarrays,RNA-Seq are used to quantify 
how much each gene is expressed under different conditions.'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load gene expression dataset
df = pd.read_csv("gene_expression.csv", index_col=0)

# Log transformation
df_log = np.log2(df + 1)

# Compute distance matrix using correlation distance
# 1 - correlation
distance_matrix = 1 - np.corrcoef(df_log)

# Perform hierarchical clustering
Z = linkage(distance_matrix, method='average')

# Heatmap with clustering
sns.clustermap(
    df_log,
    method='average',
    metric='correlation',
    cmap='viridis',
    figsize=(12, 10)
)

plt.show()
