# pipeline.py
# This is the main orchestrator of our entire system
# Think of it as the manager that calls each worker in the right order:
# 1. preprocess.py  → loads and cleans the data
# 2. model.py       → detects anomalies
# 3. explain.py     → generates plain English explanations
# The dashboard will call this file to get everything in one shot

# ---- IMPORTS ----
# importing our own modules
from src.preprocess import load_domains, load_ips, scale_features
from src.model import train_model, predict_anomalies, save_metrics
from src.explain import explain_all_anomalies

# json: to save and load results
import json

# os: to check if files exist and manage paths
import os

# pandas: for working with dataframes
import pandas as pd


def run_pipeline(max_explanations=5):
    """
    Runs the complete SecureIntel pipeline from start to finish.
    
    This single function:
    1. Loads and preprocesses the data
    2. Trains the anomaly detection model
    3. Detects anomalies
    4. Generates AI explanations for top anomalies
    5. Saves metrics to results/metrics.json
    
    Parameters:
        max_explanations: int, how many anomalies to explain with AI
                         keep low to respect free API limits
    
    Returns:
        results: dict containing all pipeline outputs
    """
    
    print("=" * 50)
    print("  SecureIntel DNS Guardian — Running Pipeline")
    print("=" * 50)
    
    # ---- STEP 1: LOAD AND PREPROCESS DATA ----
    print("\n[1/4] Loading and preprocessing data...")
    
    # load domain data from CSV
    domains_df = load_domains()
    
    # scale features so all numbers are on the same range
    # scaler is saved so we can use it later on new data
    domains_scaled, scaler = scale_features(domains_df)
    
    print(f"      Loaded {len(domains_df)} domain records")
    
    # ---- STEP 2: TRAIN MODEL AND DETECT ANOMALIES ----
    print("\n[2/4] Training anomaly detection model...")
    
    # train our Isolation Forest model
    model = train_model(domains_scaled)
    
    print("\n[3/4] Detecting anomalies...")
    
    # run predictions — returns dataframe with anomaly scores
    results_df = predict_anomalies(model, domains_scaled, domains_df)
    
    # count how many anomalies were found
    anomaly_count = results_df['is_anomaly'].sum()
    print(f"      Found {anomaly_count} anomalies out of {len(results_df)} records")
    
    # ---- STEP 3: SAVE METRICS ----
    print("\n      Saving metrics...")
    
    # save metrics to results/metrics.json
    metrics = save_metrics(results_df)
    
    # ---- STEP 4: GENERATE AI EXPLANATIONS ----
    print(f"\n[4/4] Generating AI explanations for top {max_explanations} anomalies...")
    print("      (This may take a moment due to API rate limits)")
    
    # get AI explanations for top anomalies
    explained_df = explain_all_anomalies(results_df, max_explanations=max_explanations)
    
    print("\n" + "=" * 50)
    print("  Pipeline Complete!")
    print("=" * 50)
    
    # ---- RETURN ALL RESULTS ----
    # package everything into a single dictionary
    # the dashboard will use this to display results
    return {
        'all_results': results_df,        # all records with anomaly scores
        'explained': explained_df,         # anomalies with AI explanations
        'metrics': metrics,                # summary statistics
        'model': model,                    # trained model (for reuse)
        'scaler': scaler                   # scaler (for new data)
    }


def load_metrics():
    """
    Loads previously saved metrics from results/metrics.json
    Used by the dashboard to show stats without rerunning the pipeline
    
    Returns:
        metrics: dict of metrics, or None if file doesn't exist
    """
    
    # check if the metrics file exists
    # os.path.exists() returns True if the file is there
    if os.path.exists('results/metrics.json'):
        
        # open file in read mode ('r')
        with open('results/metrics.json', 'r') as f:
            # json.load() reads the JSON file into a Python dictionary
            return json.load(f)
    
    # return None if no metrics file found yet
    return None


# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    
    # run the full pipeline with explanations for top 3 anomalies
    # keeping it at 3 to be safe with API rate limits
    results = run_pipeline(max_explanations=3)
    
    # print a final summary
    print(f"\nSummary:")
    print(f"  Total records analysed: {results['metrics']['total_records']}")
    print(f"  Anomalies detected: {results['metrics']['anomalies_detected']}")
    print(f"  Anomaly percentage: {results['metrics']['anomaly_percentage']}%")
    print(f"\nTop anomalies with AI explanations:")
    
    # loop through explained anomalies and print them
    for i, (idx, row) in enumerate(results['explained'].iterrows()):
        print(f"\n  Anomaly {i+1}:")
        print(f"  Reputation={row['Reputation']}, "
              f"Malicious Votes={row['Malicious_Votes']}")
        # [:200] takes only first 200 characters of explanation
        # so it doesn't flood the terminal
        print(f"  AI: {row['explanation'][:200]}...")