import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

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
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])
similarity_matrix = cosine_similarity(tfidf_matrix)

# Generate Visualizations
# Phase Distribution
phase_dist = df['Phases'].value_counts().head(10)
fig1 = px.bar(
    x=phase_dist.index,
    y=phase_dist.values,
    labels={"x": "Phases", "y": "Count"},
    title="Top 10 Phases Distribution",
    color_discrete_sequence=px.colors.sequential.Blues
)
fig1.update_layout(title_x=0.5, template="simple_white")

# Similarity Heatmap
subset_size = min(len(df), 10)
similarity_subset = similarity_matrix[:subset_size, :subset_size]
fig2 = px.imshow(
    similarity_subset,
    labels={"color": "Similarity Score"},
    title="Similarity Heatmap",
    color_continuous_scale="Viridis"
)
fig2.update_layout(title_x=0.5, template="plotly")

# Top Similar Trials
similar_trials = df.head(10)
similar_trials['Short Title'] = similar_trials['Study Title'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
fig3 = px.bar(
    x=similar_trials["Short Title"],
    y=list(range(len(similar_trials), 0, -1)),  # Match length of x
    labels={"x": "Trial", "y": "Similarity Rank"},
    title="Top 10 Similar Trials",
    color=list(range(len(similar_trials), 0, -1)),  # Match length of x
    color_continuous_scale="Plasma"
)
fig3.update_layout(
    title_x=0.5,
    template="simple_white",
    xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
    xaxis_title="Shortened Trial Titles"
)

# Build Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# Navbar
navbar = dbc.NavbarSimple(
    brand="Clinical Trials Dashboard",
    brand_href="#",
    color="primary",
    dark=True
)

# Layout
app.layout = dbc.Container([
    navbar,
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Phase Distribution", className="text-center text-primary"),
                dcc.Graph(figure=fig1)
            ], className="p-4 bg-light rounded shadow-sm")
        ], width=6),
        dbc.Col([
            html.Div([
                html.H4("Similarity Heatmap", className="text-center text-primary"),
                dcc.Graph(figure=fig2)
            ], className="p-4 bg-light rounded shadow-sm")
        ], width=6)
    ], className="mt-4"),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Top Similar Trials", className="text-center text-primary"),
                dcc.Graph(figure=fig3)
            ], className="p-4 bg-light rounded shadow-sm")
        ], width=12)
    ], className="mt-4")
], fluid=True)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
