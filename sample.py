import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

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

# Combine fields for similarity computation
df['combined'] = (
    df['Study Title'].fillna('') + " " +
    df['Primary Outcome Measures'].fillna('') + " " +
    df['Secondary Outcome Measures'].fillna('') + " " +
    df['Conditions'].fillna('') + " " +
    df['Interventions'].fillna('') + " " +
    df['Phases'].fillna('') + " " +
    df['Criteria'].fillna('')
)

# Compute similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])
similarity_matrix = cosine_similarity(tfidf_matrix)

# Build Dash App
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Clinical Trials Dashboard", style={"textAlign": "center"}),

    # Dropdown for selecting trial
    html.Div([
        html.Label("Select Trial:"),
        dcc.Dropdown(
            id="trial-dropdown",
            options=[{"label": title, "value": nct_id} for nct_id, title in zip(df["NCT ID"], df["Study Title"])],
            value=df["NCT ID"].iloc[0],
            style={"width": "50%"}
        )
    ], style={"textAlign": "center"}),

    # Visualization containers
    dcc.Graph(id="field-distribution"),
    dcc.Graph(id="similarity-heatmap"),
    dcc.Graph(id="top-similar-trials"),
])

# Callbacks for Interactivity
@app.callback(
    [Output("field-distribution", "figure"),
     Output("similarity-heatmap", "figure"),
     Output("top-similar-trials", "figure")],
    [Input("trial-dropdown", "value")]
)
def update_visuals(selected_trial):
    # Field Distribution
    phase_dist = df['Phases'].value_counts().head(10)
    fig1 = px.bar(
        x=phase_dist.index,
        y=phase_dist.values,
        labels={"x": "Phases", "y": "Count"},
        title="Top 10 Phases Distribution"
    )

    # Similarity Heatmap
    trial_idx = df[df["NCT ID"] == selected_trial].index[0]
    subset = similarity_matrix[trial_idx, :10]  # Show similarity with first 10 trials
    fig2 = px.imshow([subset], labels={"color": "Similarity Score"}, title="Similarity Heatmap")

    # Top Similar Trials
    similar_indices = similarity_matrix[trial_idx].argsort()[-11:-1][::-1]
    similar_trials = df.iloc[similar_indices]
    fig3 = px.bar(
        x=similar_trials["Study Title"],
        y=similarity_matrix[trial_idx, similar_indices],
        labels={"x": "Trial", "y": "Similarity Score"},
        title=f"Top 10 Similar Trials for {selected_trial}"
    )

    return fig1, fig2, fig3

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
