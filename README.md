# 🐦 Avian Disease Outbreak Prediction using Machine Learning

## 📁 Project Overview
This project analyzes bird flu (avian influenza) outbreaks in the UK using an enriched, multi-source dataset. It combines epidemiological, environmental, and operational data to classify high-risk regions and months, using machine learning models like Naive Bayes and Random Forest.

---

## 🎯 Objectives

- Predict regions/months likely to experience severe avian flu outbreaks.
- Analyze key contributors to high-risk outbreaks (e.g., vaccination, poultry population).
- Evaluate model performance through EDA, clustering, and classification techniques.
- Build early-warning systems to aid health agencies and policymakers.

---

## 📊 Dataset

- **Records:** 1,680
- **Key Features:**
  - `Region`, `Diagnosis`, `Total_Cases`, `Month`, `Year`
  - `Poultry_Population_000s`, `Vaccine_Rate`, `Biosecurity_Score`
  - `Avg_Temperature`, `Humidity`, `Migration_Level`
- **Target:** `High_Risk` (1 if `Total_Cases` ≥ 10, else 0)
- **Source:** [APHA Tableau Dashboard](https://public.tableau.com/app/profile/siu.apha/viz/AvianDashboard/Overview), UK Met Office, DEFRA

---

## 🔍 Exploratory Data Analysis (EDA)

- **Top-Affected Region:** England North
- **Diagnosis Insight:** Avian Influenza shows rare but severe spikes
- **Seasonal Trends:** Peaks around June and October
- **Correlations:**
  - Poultry population is strongly correlated with migration and vaccination
  - Weather variables (temperature, humidity) show weak correlation with outbreaks

**EDA Visuals**:
- `total_cases_by_region.png`
- `boxplot_total_cases_by_diagnosis.png`
- `monthly_trend_outbreaks.png`
- `correlation_heatmap_all_features.png`
- `high_risk_distribution.png`

---

## 🧠 Clustering & Dimensionality Reduction

- Applied PCA to reduce features to 2D
- Used Elbow Method to determine `K=4` clusters
- Segmented disease profiles into 4 distinct regional patterns

---

## 🤖 Classification Models

### 🔹 Naive Bayes
- **Features Used:** Poultry Population, Vaccine Rate, Biosecurity Score, Month, Migration Level
- **Accuracy:** 54%
- **Recall (High Risk):** 73%
- **Best For:** Early warning (high recall, faster predictions)

### 🔸 Random Forest
- **Features Used:** Poultry Population, Vaccine Rate, Biosecurity Score, Humidity, Temperature
- **Accuracy:** 46%
- **Recall (High Risk):** 34%
- **Best For:** Feature insight & pattern recognition

**Model Visuals:**
- Confusion Matrices
- Decision Boundaries
- Feature Importance Charts

---

## 📌 Key Insights

- High poultry density and low biosecurity = higher outbreak risk
- Vaccination has a strong protective effect
- Naive Bayes is more suitable for early detection due to higher recall
- Random Forest provides better interpretability and feature relevance

---

## ✅ Recommendations

- Prioritize surveillance in regions with low vaccine rates & high poultry counts
- Improve on-farm biosecurity practices
- Use Naive Bayes for quick screening
- Expand dataset with more farm-level features (e.g., human contact, feed)
- Consider advanced models (e.g., XGBoost, SMOTE for balancing)

---
## 👥 Team Members

- **Rupesh Pun** (S378248) – [GitHub: Ruphello](https://github.com/Ruphello)
- **Shiva Raj Bhurtel** (S374134) – [GitHub: shiva709](https://github.com/shiva709)
- **Susmita Khadka** (S378032) – [GitHub: Susmitakhadka](https://github.com/Susmitakhadka)
- **Surat Adhikari** (S376778) – GitHub link pending

---

## 📁 Project Structure

📦 avian-outbreak-prediction/
├── final_avian_dataset.csv
├── avian_eda_classification_clustering.py
├── Final_Report.pdf
├── images/
│ ├── total_cases_by_region.png
│ ├── monthly_trend_outbreaks.png
│ └── ...
├── README.md


---

## 📚 References

- Marchesini & De Sanctis (2021), *The Virus Paradigm*, Cambridge University Press  
- Thompson et al. (2024), *Global Wildlife Disease Surveillance*, Frontiers in Veterinary Science  
- DEFRA Reports & GOV.UK Avian Flu Outbreak Notes  

---

*Made with 🧠 by Group G7 - Charles Darwin University (PRT564)*
