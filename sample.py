import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import re

# Load Data
file_path = "response.json"
eligibilities_path = "eligibilities.txt"

with open(file_path, "r") as file:
    data = json.load(file)

eligibilities = pd.read_csv(eligibilities_path, delimiter="|")

# Extract relevant fields and merge
trials = []
for study in data.get("studies", []):
    protocol = study.get("protocolSection", {})
    outcomes = protocol.get("outcomesModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    
    nct_id = protocol.get("identificationModule", {}).get("nctId", "")
    study_title = protocol.get("identificationModule", {}).get("briefTitle", "")
    primary_outcomes = " ".join(outcome.get("measure", "") for outcome in outcomes.get("primaryOutcomes", []))
    secondary_outcomes = " ".join(outcome.get("measure", "") for outcome in outcomes.get("secondaryOutcomes", []))
    conditions = ", ".join(protocol.get("conditionsModule", {}).get("conditions", []))
    interventions = ", ".join(i.get("name", "") for i in protocol.get("armsInterventionsModule", {}).get("interventions", []))
    phases = ", ".join(protocol.get("designModule", {}).get("phases", []))
    criteria = eligibility.get("eligibilityCriteria", "")
    
    trials.append({
        "NCT ID": nct_id,
        "Study Title": study_title,
        "Primary Outcome Measures": primary_outcomes,
        "Secondary Outcome Measures": secondary_outcomes,
        "Conditions": conditions,
        "Interventions": interventions,
        "Phases": phases,
        "Criteria": criteria,
    })

df = pd.DataFrame(trials)
df = df.merge(eligibilities, how="left", left_on="NCT ID", right_on="nct_id")

# Clean and preprocess the data
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r"[^a-zA-Z0-9\s,]", "", text)
    return text.lower().strip()

columns_to_clean = [
    "Study Title", "Primary Outcome Measures", "Secondary Outcome Measures",
    "Conditions", "Interventions", "Phases", "Criteria"
]
for col in columns_to_clean:
    df[col] = df[col].apply(clean_text)

# Combine fields with weights
weights = {
    "Study Title": 1,
    "Primary Outcome Measures": 1.5,
    "Secondary Outcome Measures": 1,
    "Conditions": 2,
    "Interventions": 1.5,
    "Phases": 2,
    "Criteria": 2.5
}

df["combined"] = (
    df["Study Title"] + " " +
    df["Primary Outcome Measures"] + " " +
    df["Secondary Outcome Measures"] + " " +
    df["Conditions"] + " " +
    df["Interventions"] + " " +
    df["Phases"] + " " +
    df["Criteria"]
)

# Compute similarity
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined"])
similarity_matrix = cosine_similarity(tfidf_matrix)

# Function to get top similar trials
def get_similar_trials(nct_id, top_n=10):
    if nct_id not in df["NCT ID"].values:
        return []
    index = df[df["NCT ID"] == nct_id].index[0]
    scores = list(enumerate(similarity_matrix[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    similar_trials = [{"NCT ID": df.iloc[i]["NCT ID"], "Similarity": score} for i, score in scores[1:top_n+1]]
    return similar_trials

# Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# Navbar
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.I(className="bi bi-bar-chart-fill text-primary", style={"fontSize": "2rem"})),
            dbc.Col(dbc.NavbarBrand("Clinical Trials Dashboard", className="ms-2 text-primary", style={"fontSize": "1.5rem"})),
        ], align="center", className="g-0"),
    ]), color="light", dark=False, className="mb-4"
)

# Dashboard Layout
app.layout = html.Div([
    navbar,
    dbc.Container([
        # Dropdown for NCT ID selection
        html.H3("Top Similar Trials for Specific NCT IDs", className="text-center text-primary mb-4"),
        dcc.Dropdown(
            id="nct-id-dropdown",
            options=[{"label": nct, "value": nct} for nct in df["NCT ID"].unique()],
            value=df["NCT ID"].iloc[0],
            placeholder="Select a Clinical Trial NCT ID",
            className="mb-4"
        ),
        dcc.Graph(id="similar-trials-chart", style={"height": "500px"}),

        # Additional EDA Visualizations
        html.H3("Exploratory Data Analysis", className="text-center text-primary mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="phase-distribution", style={"height": "400px"}), width=6),
            dbc.Col(dcc.Graph(id="conditions-distribution", style={"height": "400px"}), width=6),
        ])
    ])
])

# Callback for Dashboard Visualizations
@app.callback(
    [
        Output("similar-trials-chart", "figure"),
        Output("phase-distribution", "figure"),
        Output("conditions-distribution", "figure")
    ],
    Input("nct-id-dropdown", "value")
)
def update_dashboard(selected_nct_id):
    # Similar Trials Chart
    similar_trials = get_similar_trials(selected_nct_id, top_n=10)
    if similar_trials:
        df_similar = pd.DataFrame(similar_trials)
        similar_fig = px.bar(
            df_similar, x="NCT ID", y="Similarity", text="NCT ID",
            title=f"Top 10 Similar Trials for {selected_nct_id}",
            labels={"Similarity": "Similarity Score", "NCT ID": "Trial ID"}
        )
        similar_fig.update_traces(textangle=0)
        similar_fig.update_layout(title_x=0.5, xaxis_tickangle=45)
    else:
        similar_fig = px.bar(title=f"No Similar Trials Found for {selected_nct_id}")

    # Phase Distribution
    phase_counts = df["Phases"].value_counts().head(10)
    phase_fig = px.bar(phase_counts, x=phase_counts.index, y=phase_counts.values,
                       labels={"x": "Phases", "y": "Count"}, title="Phase Distribution")
    phase_fig.update_layout(title_x=0.5)

    # Condition Distribution
    condition_counts = df["Conditions"].value_counts().head(10)
    condition_fig = px.bar(condition_counts, x=condition_counts.index, y=condition_counts.values,
                           labels={"x": "Conditions", "y": "Count"}, title="Condition Distribution")
    condition_fig.update_layout(title_x=0.5)

    return similar_fig, phase_fig, condition_fig

if __name__ == "__main__":
    app.run_server(debug=True)
