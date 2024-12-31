import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# File paths
file_path = "response.json"  # Replace with your file path
eligibilities_path = "eligibilities.txt"  # Replace with your file path

# Load response.json
with open(file_path, "r") as file:
    data = json.load(file)

# Load eligibilities.txt and split columns using "|"
eligibilities = pd.read_csv(eligibilities_path, delimiter="|")

# Debugging: Check eligibilities.txt structure
print("Eligibilities Columns:", eligibilities.columns)
print(eligibilities.head())

# Extract and merge relevant fields
trials = []
for study in data.get("studies", []):
    protocol = study.get("protocolSection", {})
    outcomes = protocol.get("outcomesModule", {})
    eligibility = protocol.get("eligibilityModule", {})
    
    # Extract fields
    nct_id = protocol.get("identificationModule", {}).get("nctId", "")
    study_title = protocol.get("identificationModule", {}).get("briefTitle", "")
    primary_outcomes = " ".join(outcome.get("measure", "") for outcome in outcomes.get("primaryOutcomes", []))
    secondary_outcomes = " ".join(outcome.get("measure", "") for outcome in outcomes.get("secondaryOutcomes", []))
    conditions = ", ".join(protocol.get("conditionsModule", {}).get("conditions", []))
    interventions = ", ".join(i.get("name", "") for i in protocol.get("armsInterventionsModule", {}).get("interventions", []))
    phases = ", ".join(protocol.get("designModule", {}).get("phases", []))
    criteria = eligibility.get("eligibilityCriteria", "")
    
    # Add to trials list
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

# Convert to DataFrame
df = pd.DataFrame(trials)

# Merge using the correct column name
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

# Find top 10 similar trials for specific trials
specific_trials = ["NCT05013879", "NCT00517179", "NCT03058679"]
results = {}

for nct_id in specific_trials:
    idx = df[df['NCT ID'] == nct_id].index
    if idx.empty:
        print(f"Trial with NCT ID {nct_id} not found in the dataset.")
        continue
    idx = idx[0]
    similar_indices = similarity_matrix[idx].argsort()[-11:-1][::-1]  # Top 10 excluding itself
    results[nct_id] = df.iloc[similar_indices][[
        "NCT ID", "Study Title", "Primary Outcome Measures", "Secondary Outcome Measures", "Criteria"
    ]].to_dict(orient='records')

# Save results
output_file = "similar_trials.json"
with open(output_file, "w") as out_file:
    json.dump(results, out_file, indent=4)

print(f"Results saved to {output_file}")

# Step 1: Distribution of Fields
def plot_field_distribution(df, field, title):
    plt.figure(figsize=(10, 6))
    df[field].value_counts().head(10).plot(kind='bar', color='skyblue')
    plt.title(f"Top 10 {title} Distribution")
    plt.xlabel(title)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.show()

# Plot distributions
plot_field_distribution(df, 'Phases', 'Phases')
plot_field_distribution(df, 'Conditions', 'Conditions')
plot_field_distribution(df, 'Interventions', 'Interventions')

# Step 2: Similarity Matrix Heatmap
def plot_similarity_matrix(similarity_matrix, df, n_trials=10):
    plt.figure(figsize=(12, 8))
    subset = similarity_matrix[:n_trials, :n_trials]  # Subset for readability
    sns.heatmap(subset, annot=True, cmap='coolwarm', xticklabels=df['NCT ID'][:n_trials], yticklabels=df['NCT ID'][:n_trials])
    plt.title("Similarity Matrix Heatmap (Subset)")
    plt.xlabel("Trials")
    plt.ylabel("Trials")
    plt.show()

plot_similarity_matrix(similarity_matrix, df)

# Step 3: Top Similarities for Specific Trials
def plot_top_similarities(nct_id, results, title):
    similar_trials = results.get(nct_id, [])
    if not similar_trials:
        print(f"No similar trials found for {nct_id}.")
        return
    
    top_titles = [trial["Study Title"] for trial in similar_trials]
    similarity_scores = [1 - idx/10 for idx in range(len(similar_trials))]  # Mock scores for demonstration

    plt.figure(figsize=(10, 6))
    sns.barplot(x=similarity_scores, y=top_titles, palette='viridis')
    plt.title(f"Top 10 Similar Trials for {title}")
    plt.xlabel("Similarity Score")
    plt.ylabel("Trial Title")
    plt.show()

for nct_id in specific_trials:
    plot_top_similarities(nct_id, results, nct_id)

# Step 4: Explainability using SHAP
def explain_similarity(tfidf_matrix, vectorizer):
    explainer = shap.Explainer(tfidf_matrix.toarray())
    shap_values = explainer(tfidf_matrix.toarray())

    # Feature importance visualization
    feature_names = vectorizer.get_feature_names_out()
    shap.summary_plot(shap_values, features=tfidf_matrix.toarray(), feature_names=feature_names)

explain_similarity(tfidf_matrix, vectorizer)
