import streamlit as st

import pandas as pd

import numpy as np

import joblib

import plotly.express as px

import shap

import matplotlib.pyplot as plt

#Page configuration

st.set_page_config(

    page_title='Fraud Detection Dashboard',

    layout='wide'
)

st.title(
    'AI-Powered Fraud Operations Dashboard'
)

#Loading deployment files

model = joblib.load(
    'model.pkl'
)

scaler = joblib.load(
    'scaler.pkl'
)

feature_columns = joblib.load(
    'feature_columns.pkl'
)

dashboard_df = pd.read_csv(
    'dashboard_data.csv'
)

#Sidebar navigation

page = st.sidebar.radio(

    'Navigation',

    [

        'Overview',

        'Transaction Explorer',

        'SHAP Explainer'
    ]
)

#Overview page

if page == 'Overview':

    st.header(
        'Fraud Detection Overview'
    )
    #Calculating metrics

    total_transactions = len(
        dashboard_df
    )

    total_fraud = dashboard_df[
        'isFraud'
    ].sum()

    detection_rate = (
        total_fraud
        /
        total_transactions
    ) * 100

    avg_fraud_amount = dashboard_df[
        dashboard_df['isFraud'] == 1
    ]['TransactionAmt'].mean()

    #Creating metric columns

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        'Total Transactions',
        total_transactions
    )

    col2.metric(
        'Total Fraud Count',
        int(total_fraud)
    )

    col3.metric(
        'Detection Rate (%)',
        round(detection_rate,2)
    )

    col4.metric(
        'Average Fraud Amount',
        round(avg_fraud_amount,2)
    )

    #Fraud distribution chart

    #Creating readable fraud labels

    dashboard_df['FraudLabel'] = np.where(
    
        dashboard_df['isFraud'] == 1,
    
        'Fraudulent',
    
        'Legitimate'
    )
    
    #Fraud distribution chart
    
    fraud_chart = px.pie(
    
        dashboard_df,
    
        names='FraudLabel',
    
        title='Fraud Distribution'
    )

    st.plotly_chart(
        fraud_chart,
        use_container_width=True
    ) 
    
    #Risk tier distribution chart
    risk_counts = dashboard_df[
        'RiskTier'
    ].value_counts().reset_index()
    
    risk_counts.columns = [
        'RiskTier',
        'Count'
    ]
    
    risk_chart = px.bar(
    
        risk_counts,
    
        x='RiskTier',
    
        y='Count',
    
        title='Risk Tier Distribution'
    )
    
    st.plotly_chart(
        risk_chart,
        use_container_width=True
    )

#Transaction Explorer page

elif page == 'Transaction Explorer':

    st.header(
        'Transaction Explorer'
    )

    #Sidebar risk filter

    selected_risk = st.sidebar.multiselect(

        'Select Risk Tier',

        dashboard_df['RiskTier'].unique(),

        default=dashboard_df['RiskTier'].unique()
    )

    #Filtering transactions

    filtered_df = dashboard_df[

        dashboard_df['RiskTier'].isin(
            selected_risk
        )
    ]

    #Transaction amount filter

    amount_range = st.sidebar.slider(

        'Transaction Amount Range',

        float(filtered_df['TransactionAmt'].min()),

        float(filtered_df['TransactionAmt'].max()),

        (

            float(filtered_df['TransactionAmt'].min()),

            float(filtered_df['TransactionAmt'].max())
        )
    )

    #Applying amount filter

    filtered_df = filtered_df[

        (filtered_df['TransactionAmt']
         >= amount_range[0])

        &

        (filtered_df['TransactionAmt']
         <= amount_range[1])
    ]

    #TransactionID search

    transaction_search = st.text_input(

        'Enter TransactionID'
    )

    #Searching transaction

    if transaction_search:

        result = filtered_df[

            filtered_df['TransactionID'].astype(str)
            ==
            transaction_search
        ]

        if len(result) > 0:

            st.subheader(
                'Transaction Details'
            )

            st.dataframe(result)

            #Displaying live risk metrics

            col1, col2 = st.columns(2)
            
            col1.metric(
            
                'Fraud Probability',
            
                f"{result['FraudProbability'].values[0]:.4f}"
            )
            
            col2.metric(
            
                'Risk Tier',
            
                result['RiskTier'].values[0]
            )

        else:

            st.error(
                'TransactionID not found'
            )

    #Displaying filtered transactions

    st.subheader(
        'Filtered Transactions'
    )

    st.dataframe(

        filtered_df.head(100)
    )
#SHAP Explainer page

elif page == 'SHAP Explainer':

    st.header(
        'SHAP Transaction Explainer'
    )

    #TransactionID input

    shap_transaction = st.text_input(

        'Enter TransactionID for SHAP Analysis'
    )

    #Checking transaction

    if shap_transaction:

        shap_row = dashboard_df[

            dashboard_df['TransactionID'].astype(str)
            ==
            shap_transaction
        ]

        #Transaction exists

        if len(shap_row) > 0:

            st.subheader(
                'Transaction Details'
            )

            st.dataframe(shap_row)

            #Displaying fraud probability

            st.metric(

                'Fraud Probability',

                f"{shap_row['FraudProbability'].values[0]:.4f}"
            )

            #Preparing features for SHAP

            shap_features = shap_row[
                feature_columns
            ]

            #Creating SHAP explainer

            explainer = shap.TreeExplainer(
                model
            )

            #Generating SHAP values

            shap_values = explainer.shap_values(
                shap_features
            )

            #SHAP waterfall plot

            st.subheader(
                'SHAP Waterfall Plot'
            )

            fig, ax = plt.subplots(
                figsize=(10,5)
            )

            shap.plots._waterfall.waterfall_legacy(

                explainer.expected_value,

                shap_values[0],

                shap_features.iloc[0],

                show=False
            )

            st.pyplot(fig)

            #Plain-English explanation

            st.subheader(
                'Plain-English Explanation'
            )
            
            #Top SHAP features
            
            top_features = pd.DataFrame({
            
                'Feature': feature_columns,
            
                'SHAP Value':
                np.abs(shap_values[0])
            
            })
            
            top_features = top_features.sort_values(
            
                by='SHAP Value',
            
                ascending=False
            )
            
            #Selecting top 3 influential features
            
            top_3 = top_features.head(3)
            
            top_feature_names = top_3[
                'Feature'
            ].tolist()
            
            #Getting fraud probability

            fraud_prob = shap_row[
                'FraudProbability'
            ].values[0]
            
            #Generating explanation text
            
            if fraud_prob >= 0.75:
            
                explanation = (
            
                    f"The transaction was classified as "
                    f"Critical Risk primarily due to unusual "
                    f"patterns in "
                    f"{top_feature_names[0]}, "
                    f"{top_feature_names[1]}, and "
                    f"{top_feature_names[2]}."
                )
            
            elif fraud_prob >= 0.40:
            
                explanation = (
            
                    f"The transaction was classified as "
                    f"Suspicious due to contributing patterns in "
                    f"{top_feature_names[0]}, "
                    f"{top_feature_names[1]}, and "
                    f"{top_feature_names[2]}."
                )
            
            else:
            
                explanation = (
            
                    f"The transaction was classified as "
                    f"low fraud risk. Although features like "
                    f"{top_feature_names[0]}, "
                    f"{top_feature_names[1]}, and "
                    f"{top_feature_names[2]} showed some influence, "
                    f"their combined impact was not strong enough "
                    f"to indicate fraudulent behavior."
                )
            
            st.write(explanation)
            
else:
    st.error('TransactionID not found')
