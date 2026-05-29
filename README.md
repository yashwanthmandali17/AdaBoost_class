# ❤️ Heart Disease Prediction — AdaBoost Classifier

An end-to-end machine learning project that predicts heart disease risk using an **AdaBoost Classifier** with Decision Tree stumps as base estimators.

Built following the same project structure as the **DTree-Regression** (House Price Prediction) project.

---

## 📁 Project Structure

```
adaboost_classifier/
│
├── data/
│   └── heart_disease.csv        # 2,000-row dataset (13 clinical features)
│
├── models/
│   ├── adaboost_classifier.pkl  # Trained AdaBoost model
│   └── scaler.pkl               # StandardScaler (saved during preprocessing)
│
├── notebooks/
│   └── eda.ipynb                # Exploratory Data Analysis notebook
│
├── preprocessing.py             # Data loading, splitting, scaling
├── train.py                     # Model training & evaluation script
├── app.py                       # Streamlit web application
└── requirements.txt
```

---

## 🧠 Model Details

| Parameter | Value |
|---|---|
| Algorithm | AdaBoost Classifier |
| Base Estimator | Decision Tree (max_depth=1, stumps) |
| n_estimators | 100 |
| learning_rate | 0.5 |
| Test Accuracy | ~88.5% |
| ROC-AUC | ~0.955 |

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `age` | Patient age (years) |
| `sex` | Sex (0=Female, 1=Male) |
| `chest_pain_type` | Chest pain type (0–3) |
| `resting_bp` | Resting blood pressure (mm Hg) |
| `cholesterol` | Serum cholesterol (mg/dl) |
| `fasting_blood_sugar` | Fasting blood sugar > 120 mg/dl (0/1) |
| `rest_ecg` | Resting ECG results (0–2) |
| `max_heart_rate` | Maximum heart rate achieved |
| `exercise_angina` | Exercise-induced angina (0/1) |
| `oldpeak` | ST depression (exercise vs rest) |
| `slope` | Slope of peak exercise ST segment (0–2) |
| `ca` | Major vessels coloured by fluoroscopy (0–4) |
| `thal` | Thalassemia type (0–2) |
| **`heart_disease`** | **Target: 1=Disease, 0=No Disease** |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 📈 App Pages

| Page | Description |
|---|---|
| 🏠 Home | Dataset overview, metrics, preview & summary stats |
| 📊 EDA | Target distribution, feature plots, heatmaps, correlations |
| 📈 Model Performance | Confusion matrix, ROC curve, feature importance, staged accuracy, learning rate sensitivity |
| 🔮 Prediction | Interactive patient input → real-time risk prediction with probability gauge |
