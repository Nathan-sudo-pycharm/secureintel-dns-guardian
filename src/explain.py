# explain.py
# This file takes anomalies detected by our ML model
# and uses NVIDIA's free LLM API to explain them in plain English
# We use Llama 3.1 model hosted on NVIDIA's cloud — completely free!

# ---- IMPORTS ----
# OpenAI: we use OpenAI's library but pointed at NVIDIA's servers
# NVIDIA supports the same API format as OpenAI — so this just works!
from openai import OpenAI

# os: built-in Python library to read environment variables
import os

# dotenv: loads our secret keys from the .env file
from dotenv import load_dotenv

# pandas: for working with data tables
import pandas as pd

# time: for adding delays between API calls to avoid rate limits
import time

# ---- SETUP ----
# load our .env file so we can access NVIDIA_API_KEY
from app.config import get_nvidia_key
NVIDIA_API_KEY = get_nvidia_key()

# initialize the client pointed at NVIDIA's servers
# base_url tells the OpenAI library to talk to NVIDIA instead of OpenAI
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)


def explain_domain_anomaly(row):
    """
    Takes a single anomalous domain record and asks Llama 3.1
    to explain why it looks suspicious in plain English.
    
    Parameters:
        row: a single row from our anomaly results dataframe
    
    Returns:
        explanation: string with the AI's plain English explanation
    """
    
    # build our prompt with all the anomaly details
    prompt = f"""
    You are a cybersecurity expert helping non-technical people understand network threats.
    
    A machine learning model has flagged the following domain record as potentially suspicious.
    
    Here are the details:
    - Domain Length: {row['Domain_Length']} characters
    - Malicious Votes: {row['Malicious_Votes']} (security engines that flagged it as malicious)
    - Suspicious Votes: {row['Suspicious_Votes']} (security engines that flagged it as suspicious)
    - Harmless Votes: {row['Harmless_Votes']} (security engines that said it is safe)
    - Reputation Score: {row['Reputation']} (lower/negative = worse reputation)
    - Has Numbers in domain: {row['Has_Numbers']}
    - Has Hyphens in domain: {row['Has_Hyphen']}
    - Anomaly Score: {row['anomaly_score']:.4f} (more negative = more suspicious)
    
    In 2-3 simple sentences, explain:
    1. Why this domain looks suspicious
    2. What kind of threat it might represent
    3. What a normal person should do about it
    
    Use simple language that a non-technical person can understand.
    Do not use technical jargon.
    """
    
    try:
        # create a chat completion request to NVIDIA's API
        # this is the same format as OpenAI's API
        completion = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",  # free model on NVIDIA
            messages=[
                {
                    # "role": "user" means this is the human's message
                    "role": "user",
                    # "content" is the actual message text
                    "content": prompt
                }
            ],
            temperature=0.2,    # lower = more focused/consistent responses
            top_p=0.7,          # controls diversity of responses
            max_tokens=300,     # limit response length to save quota
            stream=False        # False = wait for full response before returning
        )
        
        # extract the text from the response
        # .choices[0] = first response option
        # .message.content = the actual text
        return completion.choices[0].message.content
    
    except Exception as e:
        # if something goes wrong return a friendly error message
        return f"Could not generate explanation: {str(e)}"


def explain_all_anomalies(results_df, max_explanations=5):
    """
    Loops through detected anomalies and generates
    a plain English explanation for each one.
    
    Parameters:
        results_df: dataframe with anomaly detection results
        max_explanations: maximum number of anomalies to explain
    
    Returns:
        explained_df: dataframe with added 'explanation' column
    """
    
    # filter to only anomalous records
    anomalies_df = results_df[results_df['is_anomaly'] == True].copy()
    
    # limit to max_explanations to be safe with API limits
    anomalies_df = anomalies_df.head(max_explanations)
    
    # empty list to collect explanations
    explanations = []
    
    # loop through each anomaly row
    # enumerate() gives us both the count (i) and the value
    for i, (idx, row) in enumerate(anomalies_df.iterrows()):
        
        print(f"Generating explanation {i+1}/{len(anomalies_df)}...")
        
        # get AI explanation for this anomaly
        explanation = explain_domain_anomaly(row)
        explanations.append(explanation)
        
        # wait 5 seconds between calls to respect rate limits
        if i < len(anomalies_df) - 1:
            time.sleep(5)
    
    # add explanations as a new column in our dataframe
    anomalies_df['explanation'] = explanations
    
    return anomalies_df


# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    
    from src.preprocess import load_domains, scale_features
    from src.model import train_model, predict_anomalies
    
    print("Loading data and running anomaly detection...")
    domains_df = load_domains()
    domains_scaled, _ = scale_features(domains_df)
    ml_model = train_model(domains_scaled)
    results = predict_anomalies(ml_model, domains_scaled, domains_df)
    
    print(f"\nFound {results['is_anomaly'].sum()} anomalies")
    print("Generating AI explanations for top 5 anomalies...\n")
    
    explained = explain_all_anomalies(results, max_explanations=5)
    
    # print each anomaly with its explanation
    for i, (idx, row) in enumerate(explained.iterrows()):
        print(f"--- Anomaly {i+1} ---")
        print(f"Features: Reputation={row['Reputation']}, "
              f"Malicious Votes={row['Malicious_Votes']}, "
              f"Domain Length={row['Domain_Length']}")
        print(f"AI Explanation:\n{row['explanation']}")
        print()