import gradio as gr
import torch
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "dataset/creditcard.csv"
df = pd.read_csv(file_path)

# Preprocess data
X = df.drop(columns=["Class"])
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

# Load Models
models = {}

# Load Scikit-learn models
for model_name in ["Logistic Regression", "Random Forest"]:
    with open(f"models/{model_name}_model.pkl", "rb") as f:
        models[model_name] = pickle.load(f)

# Load MLP Model
class MLPModel(torch.nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

mlp_model = MLPModel(input_size=X.shape[1])
mlp_model.load_state_dict(torch.load("models/mlp_fraud_detection_model.pth"))
mlp_model.eval()
models["MLP"] = mlp_model

# Load LSTM Model
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(x)

lstm_model = LSTMModel(input_size=X.shape[1])
lstm_model.load_state_dict(torch.load("models/lstm_model.pth"))
lstm_model.eval()
models["LSTM"] = lstm_model

# Function to randomly select a transaction (Fraud or Non-Fraud)
def random_transaction(fraud_type=None):
    if fraud_type == "fraud":
        sample = df[df["Class"] == 1].drop(columns=["Class"]).sample(n=1, random_state=np.random.randint(0, 10000))
    elif fraud_type == "non-fraud":
        sample = df[df["Class"] == 0].drop(columns=["Class"]).sample(n=1, random_state=np.random.randint(0, 10000))
    else:
        sample = df.drop(columns=["Class"]).sample(n=1, random_state=np.random.randint(0, 10000))

    return sample.values.tolist()

# Function to analyze & predict
def predict_fraud(model_name, input_data):
    try:
        input_data = np.array(input_data, dtype=np.float32)
        input_scaled = scaler.transform(input_data)

        if model_name in ["Logistic Regression", "Random Forest"]:
            y_pred = models[model_name].predict_proba(input_scaled)[:, 1][0]
        
        elif model_name == "XGBoost":
            dmatrix = xgb.DMatrix(input_scaled)
            y_pred = models["XGBoost"].predict(dmatrix)[0]

        elif model_name == "MLP":
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
            with torch.no_grad():
                y_pred = models["MLP"](input_tensor).item()

        elif model_name == "LSTM":
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).view(1, 1, -1)
            with torch.no_grad():
                y_pred = models["LSTM"](input_tensor).item()

        fraud_probability = round(y_pred, 4)
        result = "**HIGH RISK: Likely Fraudulent Transaction**" if fraud_probability > 0.5 else "**LOW RISK: Normal Transaction**"

        # Feature Importance Analysis
        feature_contributions = abs(input_scaled.flatten())  # Mock importance
        feature_importance_df = pd.DataFrame({
            "Feature": X.columns.tolist(),
            "Importance": feature_contributions
        }).sort_values(by="Importance", ascending=False)

        # **Generate Explanation Based on Top 5 Features**
        top_features = feature_importance_df.head(5)
        explanation = "**Top 5 Contributing Features:**\n\n"
        summary_analysis = "**Analysis Summary:**\n\n"

        for i, row in top_features.iterrows():
            feature_name = row["Feature"]
            score = row["Importance"]

            if score > 2.0:
                explanation += f"🔴 **{feature_name}**: Extremely high deviation from normal transactions.\n"
                summary_analysis += f"**{feature_name}** shows an extreme abnormality, which is highly linked to fraudulent transactions.\n"
            elif score > 1.5:
                explanation += f"🟠 **{feature_name}**: Moderately abnormal, requires caution.\n"
                summary_analysis += f"**{feature_name}** is significantly different from normal, contributing to risk assessment.\n"
            else:
                explanation += f"🟢 **{feature_name}**: Slightly unusual but not critical.\n"
                summary_analysis += f"**{feature_name}** is slightly off but does not strongly indicate fraud.\n"

        # 📊 Visualization
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))

        sns.barplot(y=feature_importance_df["Feature"][:10], x=feature_importance_df["Importance"][:10], ax=ax[0])
        ax[0].set_title("Top 10 Feature Importances")

        sns.heatmap(df.corr(), cmap="coolwarm", ax=ax[1])
        ax[1].set_title("Feature Correlation")

        fraud_count = df["Class"].value_counts()
        sns.barplot(x=fraud_count.index, y=fraud_count.values, ax=ax[2])
        ax[2].set_title("Fraud vs Non-Fraud Transactions")
        ax[2].set_xticklabels(["Non-Fraud", "Fraud"])

        return result, explanation, summary_analysis, fraud_probability, feature_importance_df, fig

    except Exception as e:
        return f"Error: {str(e)}", "", "", "", None, None

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Fraud Detection & Risk Analysis")
    gr.Markdown("### Select Model & Enter Transaction Details:")

    model_selector = gr.Dropdown(
        choices=list(models.keys()),
        value="Logistic Regression",
        label="Select Model"
    )

    input_data = gr.Dataframe(
        headers=X.columns.tolist(),
        row_count=1,
        col_count=(X.shape[1], X.shape[1]),
        type="numpy",
        value=random_transaction()
    )

    btn_fraud = gr.Button("Load Fraud Transaction from Dataset")
    btn_non_fraud = gr.Button("Load Non-Fraud Transaction from Dataset")
    btn_analyze = gr.Button("Analyze Transaction")

    result_text = gr.Markdown()
    explanation_text = gr.Markdown()
    summary_text = gr.Markdown()
    fraud_probability = gr.Textbox()
    feature_importance_table = gr.Dataframe()
    analysis_plot = gr.Plot()

    btn_analyze.click(predict_fraud, inputs=[model_selector, input_data], outputs=[result_text, explanation_text, summary_text, fraud_probability, feature_importance_table, analysis_plot])
    btn_fraud.click(lambda: random_transaction("fraud"), inputs=[], outputs=[input_data])
    btn_non_fraud.click(lambda: random_transaction("non-fraud"), inputs=[], outputs=[input_data])

# Run Gradio
demo.launch()
