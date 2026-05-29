import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, ConfusionMatrixDisplay, classification_report,
)
from preprocessing import load_and_preprocess

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction — AdaBoost Classifier",
    page_icon="",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return pd.read_csv("data/heart_disease.csv")

@st.cache_resource
def get_model_and_splits():
    model = joblib.load("models/adaboost_classifier.pkl")
    X_train, X_test, y_train, y_test = load_and_preprocess()
    return model, X_train, X_test, y_train, y_test

df = get_data()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("❤️ Heart Disease Predictor")
st.sidebar.markdown("AdaBoost Classifier")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 EDA", "📈 Model Performance", "🔮 Prediction"],
)

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("❤️ Heart Disease Prediction")
    st.markdown(
        """
        This app uses an **AdaBoost Classifier** (ensemble of Decision Tree stumps)
        to predict whether a patient has heart disease based on 13 clinical features.

        | Feature | Description |
        |---|---|
        | `age` | Patient age in years |
        | `sex` | Sex (0 = Female, 1 = Male) |
        | `chest_pain_type` | Chest pain type (0–3) |
        | `resting_bp` | Resting blood pressure (mm Hg) |
        | `cholesterol` | Serum cholesterol (mg/dl) |
        | `fasting_blood_sugar` | Fasting blood sugar > 120 mg/dl (0/1) |
        | `rest_ecg` | Resting ECG results (0–2) |
        | `max_heart_rate` | Maximum heart rate achieved |
        | `exercise_angina` | Exercise-induced angina (0/1) |
        | `oldpeak` | ST depression induced by exercise |
        | `slope` | Slope of peak exercise ST segment (0–2) |
        | `ca` | Number of major vessels coloured by fluoroscopy (0–4) |
        | `thal` | Thalassemia type (0–2) |
        | **`heart_disease`** | **Target — 1 = Disease, 0 = No Disease** |
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",   f"{len(df):,}")
    col2.metric("Features",        "13")
    col3.metric("Disease Cases",   f"{df['heart_disease'].sum():,}")
    col4.metric("Class Balance",   f"{df['heart_disease'].mean()*100:.1f}% Positive")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe().T.style.background_gradient(cmap="Reds"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    # --- Target distribution
    st.subheader("Target Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df["heart_disease"].value_counts()
    axes[0].bar(["No Disease (0)", "Disease (1)"], counts.values,
                color=["#1E88E5", "#E53935"], edgecolor="black", alpha=0.85)
    axes[0].set_title("Heart Disease Class Distribution")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 10, str(v), ha="center", fontweight="bold")

    axes[1].pie(counts.values, labels=["No Disease", "Disease"],
                autopct="%1.1f%%", colors=["#1E88E5", "#E53935"],
                startangle=90, wedgeprops=dict(edgecolor="black"))
    axes[1].set_title("Class Proportion")
    plt.tight_layout()
    st.pyplot(fig)

    # --- Numerical feature distributions
    st.subheader("Numerical Feature Distributions")
    num_cols = ["age", "resting_bp", "cholesterol", "max_heart_rate", "oldpeak"]
    palette  = ["#7E57C2", "#26A69A", "#EF5350", "#FFA726", "#29B6F6"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, col, color in zip(axes, num_cols, palette):
        ax.hist(df[col], bins=30, color=color, edgecolor="black", alpha=0.85)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    plt.suptitle("Numerical Feature Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Feature by target (violin / box)
    st.subheader("Feature Distribution by Heart Disease Status")
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    for ax, col, color in zip(axes, num_cols, palette):
        df.boxplot(column=col, by="heart_disease", ax=ax, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7))
        ax.set_title(f"{col} by Disease")
        ax.set_xlabel("Heart Disease (0/1)")
        ax.set_ylabel(col)
        plt.sca(ax)
        plt.title(f"{col}")
    plt.suptitle("Feature by Target", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Categorical features
    st.subheader("Categorical Feature vs Heart Disease Rate")
    cat_cols = ["sex", "chest_pain_type", "fasting_blood_sugar",
                "rest_ecg", "exercise_angina", "slope", "ca", "thal"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, col in zip(axes.flat, cat_cols):
        rate = df.groupby(col)["heart_disease"].mean()
        ax.bar(rate.index.astype(str), rate.values,
               color="#E53935", edgecolor="black", alpha=0.8)
        ax.set_title(f"{col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Disease Rate")
        ax.set_ylim(0, 1)
        ax.axhline(df["heart_disease"].mean(), color="navy",
                   linestyle="--", linewidth=1, label="Overall avg")
    plt.suptitle("Disease Rate by Categorical Feature", fontsize=14, y=1.01)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig)

    # --- Correlation bar with target
    st.subheader("Feature Correlations with Heart Disease")
    corr_target = df.corr()["heart_disease"].drop("heart_disease").sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#E53935" if v > 0 else "#1E88E5" for v in corr_target]
    ax.bar(corr_target.index, corr_target.values, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Correlation with Heart Disease Target")
    ax.set_ylabel("Pearson r")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    model, X_train, X_test, y_train, y_test = get_model_and_splits()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    # --- Metrics
    st.subheader("Evaluation Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{acc:.4f}")
    c2.metric("Precision", f"{prec:.4f}")
    c3.metric("Recall",    f"{rec:.4f}")
    c4.metric("F1 Score",  f"{f1:.4f}")
    c5.metric("ROC-AUC",   f"{auc:.4f}")

    # --- Confusion matrix + ROC curve
    st.subheader("Confusion Matrix & ROC Curve")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Disease", "Disease"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color="#E53935", linewidth=2,
                 label=f"AdaBoost (AUC = {auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="#E53935")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    st.pyplot(fig)

    # --- Classification report
    st.subheader("Full Classification Report")
    report = classification_report(y_test, y_pred,
                                   target_names=["No Disease", "Disease"],
                                   output_dict=True)
    st.dataframe(pd.DataFrame(report).T.style.background_gradient(cmap="Greens"),
                 use_container_width=True)

    # --- Feature Importance (AdaBoost uses estimator_weights_ & feature_importances_)
    st.subheader("Feature Importances")
    importances = model.feature_importances_
    feat_names  = X_test.columns.tolist()
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(importances)), importances[idx],
           color="#E53935", edgecolor="black", alpha=0.85)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feat_names[i] for i in idx], rotation=45, ha="right")
    ax.set_title("AdaBoost Feature Importances (Averaged over Stumps)")
    ax.set_ylabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig)

    # --- Estimator error vs number of estimators
    st.subheader("Boosting Performance: Accuracy vs Number of Estimators")
    staged_acc_train = [accuracy_score(y_train, p) for p in model.staged_predict(X_train)]
    staged_acc_test  = [accuracy_score(y_test,  p) for p in model.staged_predict(X_test)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(staged_acc_train) + 1), staged_acc_train,
            color="#E53935", linewidth=1.5, label="Train Accuracy")
    ax.plot(range(1, len(staged_acc_test) + 1), staged_acc_test,
            color="#1E88E5", linewidth=1.5, label="Test Accuracy")
    best_n = int(np.argmax(staged_acc_test)) + 1
    ax.axvline(best_n, color="grey", linestyle="--",
                label=f"Best n_estimators = {best_n}")
    ax.set_xlabel("Number of Estimators")
    ax.set_ylabel("Accuracy")
    ax.set_title("Staged Accuracy — Bias-Variance Trade-off by Boosting Round")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # --- Learning rate sensitivity
    st.subheader("Accuracy vs Learning Rate")
    lr_vals = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    acc_train_lr, acc_test_lr = [], []
    for lr in lr_vals:
        m = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=100, learning_rate=lr, random_state=42,
        )
        m.fit(X_train, y_train)
        acc_train_lr.append(accuracy_score(y_train, m.predict(X_train)))
        acc_test_lr.append(accuracy_score(y_test,  m.predict(X_test)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lr_vals, acc_train_lr, "o-", color="#E53935", label="Train Accuracy")
    ax.plot(lr_vals, acc_test_lr,  "o-", color="#1E88E5", label="Test Accuracy")
    ax.axvline(0.5, color="grey", linestyle="--", label="Chosen lr=0.5")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Learning Rate (n_estimators=100)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.title("🔮 Predict Heart Disease Risk")
    st.markdown("Enter the patient's clinical data to get a risk prediction.")

    col1, col2 = st.columns(2)

    with col1:
        age              = st.slider("Age (years)", 30, 80, 55)
        sex              = st.selectbox("Sex", [0, 1],
                                        format_func=lambda x: "Female" if x == 0 else "Male")
        chest_pain_type  = st.selectbox("Chest Pain Type",
                                        [0, 1, 2, 3],
                                        format_func=lambda x:
                                            ["Typical Angina", "Atypical Angina",
                                             "Non-anginal Pain", "Asymptomatic"][x])
        resting_bp       = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 130)
        cholesterol      = st.slider("Serum Cholesterol (mg/dl)", 150, 400, 240)
        fasting_sugar    = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                                        format_func=lambda x: "Yes" if x else "No")
        rest_ecg         = st.selectbox("Resting ECG Result", [0, 1, 2],
                                        format_func=lambda x:
                                            ["Normal", "ST-T Wave Abnormality",
                                             "Left Ventricular Hypertrophy"][x])

    with col2:
        max_hr           = st.slider("Maximum Heart Rate Achieved", 70, 200, 150)
        exercise_angina  = st.selectbox("Exercise-Induced Angina", [0, 1],
                                        format_func=lambda x: "Yes" if x else "No")
        oldpeak          = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
        slope            = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2],
                                        format_func=lambda x:
                                            ["Upsloping", "Flat", "Downsloping"][x])
        ca               = st.selectbox("Major Vessels Coloured by Fluoroscopy", [0, 1, 2, 3, 4])
        thal             = st.selectbox("Thalassemia", [0, 1, 2],
                                        format_func=lambda x:
                                            ["Normal", "Fixed Defect", "Reversible Defect"][x])

    if st.button("❤️ Predict Risk", type="primary"):
        model  = joblib.load("models/adaboost_classifier.pkl")
        scaler = joblib.load("models/scaler.pkl")

        FEATURES = ["age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
                    "fasting_blood_sugar", "rest_ecg", "max_heart_rate",
                    "exercise_angina", "oldpeak", "slope", "ca", "thal"]

        sample = pd.DataFrame([[age, sex, chest_pain_type, resting_bp, cholesterol,
                                 fasting_sugar, rest_ecg, max_hr, exercise_angina,
                                 oldpeak, slope, ca, thal]], columns=FEATURES)

        sample_sc  = pd.DataFrame(scaler.transform(sample), columns=FEATURES)
        prediction = model.predict(sample_sc)[0]
        probability = model.predict_proba(sample_sc)[0][1]

        if prediction == 1:
            st.error(f"### ⚠️ High Risk: Heart Disease Detected")
        else:
            st.success(f"### ✅ Low Risk: No Heart Disease Detected")

        st.metric("Disease Probability", f"{probability * 100:.1f}%")

        # Probability gauge bar
        fig, ax = plt.subplots(figsize=(8, 1.5))
        ax.barh(["Risk"], [probability], color="#E53935" if probability > 0.5 else "#1E88E5",
                height=0.4, edgecolor="black")
        ax.barh(["Risk"], [1 - probability], left=[probability],
                color="#e0e0e0", height=0.4, edgecolor="black")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Predicted Probability of Heart Disease")
        ax.set_title(f"Risk Score: {probability:.3f}")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(
            f"""
            | Patient Summary | Value |
            |---|---|
            | Age | {age} years |
            | Sex | {'Male' if sex else 'Female'} |
            | Chest Pain Type | {['Typical Angina','Atypical Angina','Non-anginal Pain','Asymptomatic'][chest_pain_type]} |
            | Resting BP | {resting_bp} mm Hg |
            | Cholesterol | {cholesterol} mg/dl |
            | Fasting Blood Sugar > 120 | {'Yes' if fasting_sugar else 'No'} |
            | Max Heart Rate | {max_hr} bpm |
            | Exercise Angina | {'Yes' if exercise_angina else 'No'} |
            | Oldpeak | {oldpeak} |
            | Major Vessels (ca) | {ca} |
            | Thalassemia | {['Normal','Fixed Defect','Reversible Defect'][thal]} |
            """
        )
