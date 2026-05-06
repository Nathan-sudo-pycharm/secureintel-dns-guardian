# model.py
# This file trains our anomaly detection model and uses it to find suspicious records
# Isolation Forest is an ML algorithm that learns what "normal" looks like
# and then flags anything that doesn't fit as an anomaly

# ---- IMPORTS ----
# pandas: library for working with data tables (like Excel but in Python)
import pandas as pd

# numpy: library for fast mathematical operations on numbers/arrays
import numpy as np

# IsolationForest: our ML model — detects outliers by isolating unusual data points
from sklearn.ensemble import IsolationForest

# classification_report: generates a summary of model performance (precision, recall, F1)
from sklearn.metrics import classification_report

# json: built-in Python library to save/load data in JSON format
import json

# os: built-in Python library to interact with the operating system (files, folders)
import os

# importing our own functions from preprocess.py
from src.preprocess import load_domains, load_ips, scale_features


def train_model(df_scaled, contamination=0.15):
    """
    Trains an Isolation Forest model on scaled feature data.
    
    contamination=0.15 means we expect about 15% of data to be anomalous.
    Think of it like telling the security guard: 
    'roughly 15 out of every 100 visitors look suspicious'
    
    Parameters:
        df_scaled: the scaled (normalized) feature dataframe
        contamination: float, expected proportion of anomalies in the data
    
    Returns:
        model: trained IsolationForest model
    """
    
    model = IsolationForest(
        n_estimators=100,      # number of decision trees to build (more = more accurate)
        contamination=contamination,  # expected % of anomalies in data
        random_state=42        # fixed seed so results are same every run (reproducible)
    )
    
    # fit() = train the model on our data
    # the model learns what "normal" traffic looks like here
    model.fit(df_scaled)
    
    return model  # return the trained model so we can use it later


def predict_anomalies(model, df_scaled, df_original):
    """
    Uses the trained model to find anomalies in the data.
    
    Parameters:
        model: our trained IsolationForest model
        df_scaled: scaled feature data (what the model sees)
        df_original: original unscaled data (for human-readable output)
    
    Returns:
        df_result: original dataframe with two new columns added:
                   - anomaly_score: how anomalous the record is (lower = more suspicious)
                   - is_anomaly: True if flagged as anomaly, False if normal
    """
    
    # predict() returns 1 for normal, -1 for anomaly
    # think of it like a stamp: 1 = SAFE, -1 = SUSPICIOUS
    predictions = model.predict(df_scaled)
    
    # decision_function() gives a score for each record
    # negative score = more anomalous, positive = more normal
    scores = model.decision_function(df_scaled)
    
    # copy() creates a new dataframe so we don't modify the original
    df_result = df_original.copy()
    
    # add anomaly score column to our results
    df_result['anomaly_score'] = scores
    
    # add True/False column — True means it IS an anomaly
    # predictions == -1 converts the -1/1 values to True/False
    df_result['is_anomaly'] = predictions == -1
    
    return df_result  # return the full results table


def save_metrics(results_df, filename='results/metrics.json'):
    """
    Saves model performance metrics to a JSON file.
    This is used later to display results in the dashboard
    and is required for our GitHub documentation.
    
    Parameters:
        results_df: dataframe with anomaly predictions
        filename: path where the JSON file will be saved
    
    Returns:
        metrics: dictionary of calculated metrics
    """
    
    # len() counts total number of rows in the dataframe
    total = len(results_df)
    
    # .sum() counts how many True values are in the is_anomaly column
    anomalies = results_df['is_anomaly'].sum()
    
    # build a dictionary of our key metrics
    metrics = {
        'total_records': int(total),           # int() converts numpy int to regular Python int
        'anomalies_detected': int(anomalies),
        'normal_records': int(total - anomalies),
        'anomaly_percentage': float(round((anomalies / total) * 100, 2))  # round() to 2 decimal places
    }
    
    # open the file for writing ('w' = write mode, creates file if it doesn't exist)
    with open(filename, 'w') as f:
        # json.dump() saves the dictionary as a formatted JSON file
        # indent=4 makes it human-readable with nice spacing
        json.dump(metrics, f, indent=4)
    
    return metrics  # return metrics so we can print them in the terminal


# ---- MAIN EXECUTION ----
# this block only runs when we execute THIS file directly
# it won't run if this file is imported by another file
if __name__ == "__main__":
    
    print("Loading and scaling data...")
    # load our domain data from the CSV
    domains_df = load_domains()
    # scale the features so all numbers are on the same range
    # _ means we're ignoring the second return value (the scaler object)
    domains_scaled, _ = scale_features(domains_df)

    print("Training Isolation Forest model...")
    # train our model on the scaled data
    model = train_model(domains_scaled)

    print("Detecting anomalies...")
    # run predictions on the same data
    results = predict_anomalies(model, domains_scaled, domains_df)

    # print a summary of what we found
    print("\nResults:")
    print(f"Total records: {len(results)}")           # f-string: embeds variable inside string
    print(f"Anomalies detected: {results['is_anomaly'].sum()}")
    
    # show top 5 anomalies
    # results[results['is_anomaly'] == True] filters rows where is_anomaly is True
    # .head(5) takes only the first 5 rows
    print(f"\nTop 5 anomalies:")
    print(results[results['is_anomaly'] == True].head(5))

    print("\nSaving metrics...")
    metrics = save_metrics(results)
    # f-string to display the metrics dictionary
    print(f"Metrics: {metrics}")