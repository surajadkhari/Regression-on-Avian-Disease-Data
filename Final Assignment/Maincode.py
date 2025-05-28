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

# 4. Correlation Heatmap of All Numeric Features
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numeric_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of All Numeric Features')
plt.tight_layout()
plt.savefig("correlation_heatmap_all_features.png")
plt.show()
plt.close()

# 5. High-Risk vs Low-Risk Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='High_Risk')
plt.title('Distribution of High-Risk vs Low-Risk Outbreaks')
plt.xticks([0, 1], ['Low Risk', 'High Risk'])
plt.tight_layout()
plt.savefig("high_risk_distribution.png")
plt.show()
plt.close()

# === Apply KM# Select features
categorical = ['Region', 'Diagnosis']
numerical = [
    'Total_Cases', 'Month', 'Avg_Temperature', 'Humidity',
    'Migration_Level', 'Biosecurity_Score', 'Vaccine_Rate', 'Poultry_Population_000s'
]

df = df[categorical + numerical].dropna()

# Pipeline for preprocessing + PCA
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(drop='first'), categorical)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=2))
])

X_pca = pipeline.fit_transform(df)

# === Elbow Method ===
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertias, marker='o')
plt.axvline(x=4, linestyle='--', color='red', label='Suggested K = 4')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.legend()
plt.tight_layout()
plt.show()eans with optimal K ===
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init='auto')
clusters = kmeans_final.fit_predict(X_pca)

# Plot Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=60)
plt.title("Clusters of Avian Disease Profiles (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()


