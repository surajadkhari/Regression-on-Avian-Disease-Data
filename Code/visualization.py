# Set up a 1x3 subplot to include all three EDA visualizations
import matplotlib.pyplot as plt  # Import pyplot for plotting functions
import seaborn as sns           # Import seaborn for statistical data visualization
import pandas as pd
import numpy as np
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 
# 1. Correlation Heatmap
corr_matrix = df[["Total", "Temperature", "Vaccination Rate (%)"]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0])
axes[0].set_title("Correlation Heatmap")
 
# 2. Scatter Plot with Regression Line
sns.regplot(data=df, x="Temperature", y="Total", scatter_kws={'alpha': 0.5}, ax=axes[1])
axes[1].set_title("Total Clinical Signs vs Temperature")
axes[1].set_xlabel("Temperature (°C)")
axes[1].set_ylabel("Total Clinical Signs")
 
# 3. Boxplot by Temperature Range
df['Temperature Range'] = pd.cut(
    df["Temperature"],
    bins=[-np.inf, 15, 25, np.inf],
    labels=["Low (<15°C)", "Moderate (15–25°C)", "High (>25°C)"]
)
sns.boxplot(data=df, x="Temperature Range", y="Total", ax=axes[2])
axes[2].set_title("Clinical Signs by Temperature Range")
axes[2].set_xlabel("Temperature Range")
axes[2].set_ylabel("Total Clinical Signs")
 
plt.tight_layout()
plt.show()
