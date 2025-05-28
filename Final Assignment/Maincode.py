from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
df= pd.read_csv("/content/drive/MyDrive/Coding/final_avian_dataset.csv")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load your final dataset


# Ensure 'High_Risk' column exists
df['High_Risk'] = (df['Total_Cases'] >= 10).astype(int)

# Set style
sns.set(style="whitegrid")

# 1. Total Cases by Region
plt.figure(figsize=(10, 5))
region_cases = df.groupby('Region')['Total_Cases'].sum().sort_values(ascending=False)
region_cases.plot(kind='bar')
plt.title('Total Cases by Region')
plt.ylabel('Total Cases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("total_cases_by_region.png")
plt.show()
plt.close()

# 2. Boxplot - Total Cases by Diagnosis
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Diagnosis', y='Total_Cases')
plt.title('Total Cases Distribution by Diagnosis')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("boxplot_total_cases_by_diagnosis.png")
plt.show()
plt.close()

# 3. Monthly Trend of Outbreaks
plt.figure(figsize=(8, 5))
monthly_trend = df.groupby('Month')['Total_Cases'].sum()
monthly_trend.plot(marker='o')
plt.title('Monthly Outbreak Trend')
plt.xlabel('Month')
plt.ylabel('Total Cases')
plt.grid(True)
plt.tight_layout()
plt.savefig("monthly_trend_outbreaks.png")
plt.show()
plt.close()
