import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_domains():
    # read the full CSV including Domain column
    df = pd.read_csv('data/3_malicious_domains.csv')
    
    # save domain names separately for display
    domain_names = df['Domain'].copy()
    
    features = [
        'Domain_Length',
        'Malicious_Votes',
        'Suspicious_Votes',
        'Harmless_Votes',
        'Total_Engines',
        'Reputation'
    ]
    
    result = df[features].copy()
    result['Has_Numbers'] = df['Has_Numbers'].map({'Yes': 1, 'No': 0}).fillna(0)
    result['Has_Hyphen'] = df['Has_Hyphen'].map({'Yes': 1, 'No': 0}).fillna(0)
    
    # add domain name as last column for display purposes
    result['Domain'] = domain_names
    
    result = result.fillna(0)
    return result

def load_ips():
    df = pd.read_csv('data/4_malicious_ips.csv')
    features = [
        'Malicious_Votes',
        'Suspicious_Votes',
        'Harmless_Votes',
        'Total_Reports',
        'Reputation_Score',
        'Times_Submitted'
    ]
    df = df[features].copy()
    df['TOR_Node'] = pd.read_csv('data/4_malicious_ips.csv')['TOR_Node'].map({'Yes': 1, 'No': 0})
    df['TOR_Node'] = df['TOR_Node'].fillna(0)
    df = df.fillna(0)
    return df

def scale_features(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns), scaler

if __name__ == "__main__":
    print("Loading domains...")
    domains_df = load_domains()
    print(f"Domains shape: {domains_df.shape}")
    print(domains_df.head(3))

    print("\nLoading IPs...")
    ips_df = load_ips()
    print(f"IPs shape: {ips_df.shape}")
    print(ips_df.head(3))

    print("\nScaling features...")
    domains_scaled, _ = scale_features(domains_df)
    ips_scaled, _ = scale_features(ips_df)
    print("Done! Data is ready for the ML model.")