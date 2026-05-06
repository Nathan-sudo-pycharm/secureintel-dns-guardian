# dashboard.py
# SecureIntel DNS Guardian — Streamlit Web Dashboard
# Features: multi-domain checking, real dataset lookup, AI explanations

# ---- IMPORTS ----
import streamlit as st
import pandas as pd
import sys
import os
import warnings

# suppress sklearn warnings
warnings.filterwarnings('ignore')

# add project root to path so we can import src/ modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_pipeline, load_metrics
from src.preprocess import load_domains, scale_features
from src.model import train_model, predict_anomalies
from src.explain import explain_domain_anomaly
import json

# ---- PAGE CONFIG ----
# must be first streamlit command
st.set_page_config(
    page_title="SecureIntel DNS Guardian",
    page_icon="🛡️",
    layout="wide"
)

# ---- HELPER FUNCTION ----
def check_single_domain(domain_input, domains_df, model, scaler):
    """
    Checks a single domain against our ML model.
    First looks it up in our dataset for real vote data.
    Falls back to structural analysis if not found.
    
    Parameters:
        domain_input: string, the domain name to check
        domains_df: our full domain dataframe
        model: trained IsolationForest model
        scaler: fitted StandardScaler
    
    Returns:
        result: dict with all check results
    """
    
    # clean the domain input — remove spaces and lowercase
    domain_input = domain_input.strip().lower()
    
    # skip empty inputs
    if not domain_input:
        return None
    
    # check if domain exists in our real dataset
    existing = pd.read_csv('data/3_malicious_domains.csv')
    match = existing[existing['Domain'].str.lower() == domain_input]
    
    # compute structural features from domain name
    has_numbers = int(any(char.isdigit() for char in domain_input))
    has_hyphen = int('-' in domain_input)
    domain_length = len(domain_input)
    
    if not match.empty:
        # domain found in dataset — use real vote data
        real_data = match.iloc[0]
        malicious_votes = real_data['Malicious_Votes']
        suspicious_votes = real_data['Suspicious_Votes']
        harmless_votes = real_data['Harmless_Votes']
        total_engines = real_data['Total_Engines']
        reputation = real_data['Reputation']
        data_source = "real"  # flag to show "found in database" message
    else:
        # not in dataset — use neutral values
        malicious_votes = 0
        suspicious_votes = 0
        harmless_votes = 0
        total_engines = 0
        reputation = 0
        data_source = "structural"  # flag to show "structural only" message
    
    # build feature row for the model
    new_domain = pd.DataFrame([{
        'Domain_Length': domain_length,
        'Malicious_Votes': malicious_votes,
        'Suspicious_Votes': suspicious_votes,
        'Harmless_Votes': harmless_votes,
        'Total_Engines': total_engines,
        'Reputation': reputation,
        'Has_Numbers': has_numbers,
        'Has_Hyphen': has_hyphen
    }])
    
    # scale the features using our fitted scaler
    new_domain_scaled = scaler.transform(new_domain)
    
    # run prediction
    prediction = model.predict(new_domain_scaled)[0]
    score = model.decision_function(new_domain_scaled)[0]
    is_anomaly = prediction == -1
    
    # build explanation row for AI
    explanation_row = new_domain.iloc[0].copy()
    explanation_row['anomaly_score'] = score
    
    # get AI explanation
    explanation = explain_domain_anomaly(explanation_row)
    
    return {
        'domain': domain_input,
        'is_anomaly': is_anomaly,
        'score': score,
        'data_source': data_source,
        'malicious_votes': malicious_votes,
        'reputation': reputation,
        'explanation': explanation
    }


# ---- LOAD MODEL ONCE ----
# we use st.cache_resource so the model loads only once
# and is reused across all domain checks — much faster!
@st.cache_resource
def load_model():
    """
    Loads and trains the model once and caches it.
    st.cache_resource persists this across reruns — 
    so we don't retrain every time a user checks a domain.
    """
    domains_df = load_domains()
    # drop Domain column — not a feature for the model
    domains_features = domains_df.drop(columns=['Domain'])
    domains_scaled, scaler = scale_features(domains_features)
    model = train_model(domains_scaled)
    return domains_df, model, scaler


# ---- HEADER ----
st.title("🛡️ SecureIntel — DNS Guardian")
st.markdown(
    "An open-source network anomaly detection system that watches your "
    "traffic, spots suspicious behaviour, and explains it in plain English."
)
st.divider()

# ---- SIDEBAR ----
with st.sidebar:
    st.header("Controls")
    max_explanations = st.slider(
        "Anomalies to explain (pipeline)",
        min_value=1,
        max_value=10,
        value=3,
        help="Higher = more AI explanations but slower"
    )
    st.divider()
    run_button = st.button(
        "🔍 Run Full Detection",
        type="primary",
        use_container_width=True
    )
    st.divider()
    st.markdown("**About**")
    st.markdown("Built with Python, scikit-learn, and NVIDIA Llama 3.1")
    st.markdown("Dataset: Cybersecurity Threat Intel 2026")
    st.markdown("[View on GitHub](https://github.com)")

# ---- METRICS ROW ----
existing_metrics = load_metrics()
col1, col2, col3, col4 = st.columns(4)

if existing_metrics:
    with col1:
        st.metric("Total Records", existing_metrics['total_records'])
    with col2:
        st.metric("Anomalies Detected", existing_metrics['anomalies_detected'])
    with col3:
        st.metric("Normal Records", existing_metrics['normal_records'])
    with col4:
        st.metric("Anomaly Rate", f"{existing_metrics['anomaly_percentage']}%")
else:
    with col1:
        st.metric("Total Records", "—")
    with col2:
        st.metric("Anomalies Detected", "—")
    with col3:
        st.metric("Normal Records", "—")
    with col4:
        st.metric("Anomaly Rate", "—")

st.divider()

# ---- MULTI DOMAIN CHECKER ----
st.subheader("🔎 Check Domains")
st.markdown(
    "Enter one or more domain names — one per line — and click **Check All Domains**. "
    "Domains found in our threat database use real vote data. "
    "Unknown domains are analysed by structure only."
)

# large text area for multiple domains
domain_text = st.text_area(
    "Domain names (one per line)",
    placeholder="example.com\ncryptoloot.pro\nsuspicious-site123.net\ngoogle.com",
    height=150,
    label_visibility="collapsed"
)

# check button
check_button = st.button(
    "🔍 Check All Domains",
    type="primary",
    use_container_width=False
)

# run checks when button clicked
if check_button and domain_text:
    
    # split input into list of domains
    # splitlines() splits by newline
    # strip() removes extra spaces
    # filter() removes empty strings
    domains_list = list(filter(None, [
        d.strip() for d in domain_text.splitlines()
    ]))
    
    if not domains_list:
        st.warning("Please enter at least one domain name!")
    
    else:
        # load model once for all checks
        domains_df, model, scaler = load_model()
        
        st.markdown(f"**Checking {len(domains_list)} domain(s)...**")
        
        # progress bar — shows overall progress across all domains
        # st.progress() takes a value between 0.0 and 1.0
        progress_bar = st.progress(0)
        
        # store all results
        all_check_results = []
        
        # check each domain one by one
        for i, domain in enumerate(domains_list):
            
            # update progress bar
            # (i+1)/len gives fraction like 0.25, 0.5, 0.75, 1.0
            progress_bar.progress((i + 1) / len(domains_list))
            
            # run the check
            result = check_single_domain(domain, domains_df, model, scaler)
            
            if result:
                all_check_results.append(result)
        
        # hide progress bar when done
        progress_bar.empty()
        
        # ---- SUMMARY TABLE ----
        st.markdown("### Results Summary")
        
        # build a summary dataframe for quick overview
        summary_data = []
        for r in all_check_results:
            summary_data.append({
                'Domain': r['domain'],
                'Status': '⚠️ SUSPICIOUS' if r['is_anomaly'] else '✅ NORMAL',
                'Score': round(r['score'], 4),
                'Malicious Votes': r['malicious_votes'],
                'Reputation': r['reputation'],
                'Data Source': '📂 Database' if r['data_source'] == 'real' else '🔍 Structure only'
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # colour the status column
        # st.dataframe with column_config lets us style columns
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )
        
        # quick stats below the table
        total_checked = len(all_check_results)
        total_suspicious = sum(1 for r in all_check_results if r['is_anomaly'])
        total_normal = total_checked - total_suspicious
        
        # show mini metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Checked", total_checked)
        with m2:
            st.metric("Suspicious", total_suspicious)
        with m3:
            st.metric("Normal", total_normal)
        
        st.divider()
        
        # ---- DETAILED RESULTS ----
        st.markdown("### Detailed AI Analysis")
        
        for r in all_check_results:
            
            # pick icon and colour based on result
            if r['is_anomaly']:
                icon = "⚠️"
                label = "SUSPICIOUS"
            else:
                icon = "✅"
                label = "NORMAL"
            
            # expandable section for each domain
            with st.expander(
                f"{icon} {r['domain']} — {label} (Score: {r['score']:.4f})"
            ):
                # data source notice
                if r['data_source'] == 'real':
                    st.success("📂 Found in threat database — analysis uses real vote data")
                else:
                    st.warning("🔍 Not in database — analysis based on domain structure only")
                
                # two columns — left metrics, right explanation
                left, right = st.columns([1, 2])
                
                with left:
                    st.markdown("**Key Metrics:**")
                    metrics_data = {
                        'Metric': [
                            'Anomaly Score',
                            'Malicious Votes',
                            'Reputation',
                            'Has Numbers',
                            'Has Hyphens',
                            'Domain Length'
                        ],
                        'Value': [
                            round(r['score'], 4),
                            r['malicious_votes'],
                            r['reputation'],
                            'Yes' if any(
                                c.isdigit() for c in r['domain']
                            ) else 'No',
                            'Yes' if '-' in r['domain'] else 'No',
                            len(r['domain'])
                        ]
                    }
                    st.dataframe(
                        pd.DataFrame(metrics_data),
                        hide_index=True,
                        use_container_width=True
                    )
                
                with right:
                    st.markdown("**AI Explanation:**")
                    st.info(r['explanation'])
                    st.caption(
                        "Note: For unknown domains, analysis is structural only. "
                        "Run full detection pipeline for complete dataset analysis."
                    )

elif check_button and not domain_text:
    st.warning("Please enter at least one domain name!")

st.divider()

# ---- RUN FULL PIPELINE ----
if run_button:
    with st.spinner("Running full anomaly detection pipeline... please wait"):
        results = run_pipeline(max_explanations=max_explanations)
        st.session_state['results'] = results
    st.success(
        f"Pipeline complete! "
        f"Found {results['metrics']['anomalies_detected']} anomalies."
    )
    st.rerun()

# ---- SHOW PIPELINE RESULTS ----
if 'results' in st.session_state:
    
    results = st.session_state['results']
    all_results = results['all_results']
    explained = results['explained']
    
    st.subheader("📊 Full Dataset Analysis")
    
    tab1, tab2 = st.tabs(["Anomalies Only", "All Records"])
    
    with tab1:
        anomalies_only = all_results[
            all_results['is_anomaly'] == True
        ].copy()
        anomalies_only['anomaly_score'] = anomalies_only[
            'anomaly_score'
        ].round(4)
        # show Domain column first
        cols = ['Domain'] + [
            c for c in anomalies_only.columns if c != 'Domain'
        ]
        st.dataframe(
            anomalies_only[cols],
            use_container_width=True,
            hide_index=True
        )
    
    with tab2:
        all_display = all_results.copy()
        all_display['anomaly_score'] = all_display[
            'anomaly_score'
        ].round(4)
        cols = ['Domain'] + [
            c for c in all_display.columns if c != 'Domain'
        ]
        st.dataframe(
            all_display[cols],
            use_container_width=True,
            hide_index=True
        )
    
    st.divider()
    
    # ---- AI EXPLANATIONS FROM PIPELINE ----
    st.subheader("🤖 AI Explanations — Full Pipeline")
    
    for i, (idx, row) in enumerate(explained.iterrows()):
        with st.expander(
            f"⚠️ Anomaly {i+1} — "
            f"Domain: {row.get('Domain', 'N/A')} | "
            f"Score: {row['anomaly_score']:.4f}"
        ):
            left, right = st.columns([1, 2])
            
            with left:
                st.markdown("**Features:**")
                feature_data = {
                    'Feature': [
                        'Domain',
                        'Domain Length',
                        'Malicious Votes',
                        'Suspicious Votes',
                        'Harmless Votes',
                        'Reputation',
                        'Anomaly Score'
                    ],
                    'Value': [
                        row.get('Domain', 'N/A'),
                        row['Domain_Length'],
                        row['Malicious_Votes'],
                        row['Suspicious_Votes'],
                        row['Harmless_Votes'],
                        row['Reputation'],
                        round(row['anomaly_score'], 4)
                    ]
                }
                st.dataframe(
                    pd.DataFrame(feature_data),
                    hide_index=True,
                    use_container_width=True
                )
            
            with right:
                st.markdown("**AI Explanation:**")
                st.info(row['explanation'])

else:
    st.info(
        "👈 Click **Run Full Detection** in the sidebar to analyse "
        "the complete dataset and see results here."
    )