import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Debugging: Check if NCT ID exists in both DataFrames
print("NCT IDs in trials DataFrame:", df['NCT ID'].head())
print("NCT IDs in eligibilities DataFrame:", eligibilities['nct_id'].head())

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
specific_trials = ["NCT00385736", "NCT00386607", "NCT03518073"]
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
