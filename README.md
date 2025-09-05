🚀 Support Efficiency MLOps Using github issues data from Hugging Face

An end-to-end machine learning + MLOps pipeline built on top of 31M GitHub issues.
The goal: predict issue resolution class to improve developer support efficiency and provide deeper insights into what drives resolution times.

Classes:

🟢 Short: ≤ 1 day

🟡 Medium: 1–7 days

🔴 Long: > 7 days

All of this is wrapped in a production-grade CI/CD pipeline and deployed to the cloud with Azure Web App.

📊 Data Source

We use the BigCode GitHub Issues Dataset
 on Hugging Face.
From 31M+ issues, we extract, clean, and engineer structured features such as:

text_size (issue description length)

comment_count

participants_count

first_response_minutes (time until first maintainer response)

🔍 Findings (EDA)

📓 See the notebook: notebooks/EDA.ipynb

Highlights:

Resolution times are heavily skewed toward “Long”.

Repository activity and labels correlate strongly with long times.

First response speed emerges as the most critical factor.

Text features (title/description length) have weaker but non-trivial predictive power.

🤖 Modeling

We evaluated a suite of ML classifiers:

Logistic Regression, Linear SVC, kNN

Tree ensembles: Random Forest, XGBoost, CatBoost, LightGBM

Neural MLP

✅ Best Model

CatBoost and XGBoost consistently perform best on F1-macro.

Selected via grid search hyperparameter tuning.

Model artifacts are saved in artifacts/model.pkl.

📈 Explainability with SHAP

We applied SHAP (SHapley Additive exPlanations) to interpret predictions.

Global importance:

first_response_minutes dominates model decisions.

comment_count and participants_count have moderate influence.

text_size contributes the least.

Local explanations: SHAP force plots show why a single issue was predicted “Short”, “Medium”, or “Long”.

This ensures the model is interpretable and provides actionable insights.

🌳 Causal Inference

We also ran Causal Forests (econml) to measure the causal effect of fast responses (< 60 minutes) on resolution time.

Results:

Average Treatment Effect (ATE) ≈ -0.05% → Very small causal effect once confounders are controlled.

Heterogeneity:

Effect varies mostly with comment_count.

Tickets with many comments show more variation in benefit.

text_size and participants_count play minimal roles.

Why trees?

Capture non-linearities and interactions.

Handle skewed tabular data.

Provide heterogeneous treatment effects instead of just one number.

This shifts us from correlation (“fast responses often close tickets quicker”) to causal insight (“fast responses alone only marginally shorten resolution time”).

✨ Features

Prediction: Resolution class (Short / Medium / Long)

Explainability: SHAP global + local interpretability

Causal Inference: Tree-based causal methods for intervention analysis

MLOps: MLflow for experiment tracking

Deployment:

FastAPI app for production APIs

Flask app for prototyping / dashboards

Automation: CI/CD with GitHub Actions → build, test, deploy to Azure Web App

🛠️ Quickstart (Local Dev)
1. Clone
git clone https://github.com/OBINNADINNEYA/Support-Efficiency-MLOps-Project.git
cd Support-Efficiency-MLOps-Project

2. Environment
conda create -n support_efficiency python=3.12 -y
conda activate support_efficiency
pip install -r requirements.txt

3. Train
python -m src.pipelines.train_pipeline

4. Predict (batch)
python -m src.pipelines.predict_pipeline --in data/sample.jsonl --out predictions.csv

5. Serve API

FastAPI (production-ready):

uvicorn app.main:app --reload


Flask (prototype UI / dashboard):

python app_flask.py

📦 Artifacts

artifacts/model.pkl → Best trained model

artifacts/preprocessor.pkl → Feature scaling pipeline

artifacts/encoder.pkl → Label encoder for resolution class

artifacts/shap_global.png → SHAP feature importance plot

📈 Example Output
Best model: CatBoost
Test | Acc: 0.71  F1-macro: 0.70
Prec-macro: 0.72  Rec-macro: 0.68
Recall(Long): 0.81  Precision(Short): 0.66


SHAP global importance:

first_response_minutes   >>> strongest driver
comment_count            >> medium influence
participants_count       >> medium influence
text_size                > minor influence


Causal Forest results:

Average Treatment Effect: -0.05%
Causal feature importances: comment_count ≫ participants_count > text_size

📌 Takeaways

Predictive ML models can classify ticket resolution times with good accuracy.

First response time is the strongest predictor.

But causal analysis shows its direct impact is small—most signal comes from related issue dynamics.

SHAP + causal inference ensure the model is both accurate and interpretable.

Deployment via FastAPI + Flask ensures flexibility:

FastAPI = scalable production API

Flask = simple dashboards / internal tooling

CI/CD with GitHub Actions and Azure Web App gives a full MLOps pipeline from training → testing → deployment.