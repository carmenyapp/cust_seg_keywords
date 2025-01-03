import streamlit as st
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import joblib

kproto = joblib.load('kproto_model.pkl')

cluster_descriptions = {
    0: {
        "Target Audience": ["senior", "Long-term Customer", "Responsive Customer"],
        "Action/CTA": ["explore our website now!"],
        "Emotion/Benefit": ["Engaged", "satisfied"],
    },
    1: {
        "Target Audience": ["health-conscious individuals", "busy professionals", "families", "fitness enthusiasts"],
        "Action/CTA": ["sign up for subscription"],
        "Emotion/Benefit": ["Energised", "excited"],
    },
    2: {
       "Target Audience": ["non-responder", "active customer"],
        "Action/CTA": ["shop now", "explore our collection", "get your daily essentials"],
        "Emotion/Benefit": ["Energised", "excited"],
    },
}

def preprocess_data(data):
    categorical_cols = ['Education','Marital_Status','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain','Response']
    numeric_cols = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

    # Handle missing values
    data.fillna(0, inplace=True)

    # Encode categorical features
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col]) 
    categorical_indices = [data.columns.get_loc(col) for col in categorical_cols]

    # Scale numerical features (if necessary)
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    return data, categorical_indices

def display_cluster_info(cluster):
    if cluster in cluster_descriptions:
        cluster_info = cluster_descriptions[cluster]
        st.write("**Target Audience:**")
        for audience in cluster_info["Target Audience"]:
            st.write(f"- {audience}")
        st.write("**Action/CTA:**")
        for action in cluster_info["Action/CTA"]:
            st.write(f"- {action}")
        st.write("**Emotion/Benefit:**")
        for benefit in cluster_info["Emotion/Benefit"]:
            st.write(f"- {benefit}")
            
def main():
    st.title("Customer Segmentation App")
    st.header("Customer Information")
    
    year_birth = st.number_input("Year of Birth", min_value=1900, max_value=2024)
    education = st.selectbox("Education", ["Graduation", "PhD", "Master", "2n Cycle", "Basic"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Together", "Divorced", "Widow"])
    income = st.number_input("Income", min_value=0)
    kidhome = st.number_input("Number of Kids at Home", min_value=0, max_value=5)
    teenhome = st.number_input("Number of Teens at Home", min_value=0, max_value=5)
    recency = st.number_input("Recency", min_value=0)
    mntwines = st.number_input("Amount Spent on Wine", min_value=0)
    mntfruits = st.number_input("Amount Spent on Fruits", min_value=0)
    mntmeatproducts = st.number_input("Amount Spent on Meat Products", min_value=0)
    mntfishproducts = st.number_input("Amount Spent on Fish Products", min_value=0)
    mntsweetproducts = st.number_input("Amount Spent on Sweet Products", min_value=0)
    mntgoldprods = st.number_input("Amount Spent on Gold Products", min_value=0)
    numdealspurchases = st.number_input("Number of Deals Purchases", min_value=0)
    numwebpurchases = st.number_input("Number of Web Purchases", min_value=0)
    numcatalogpurchases = st.number_input("Number of Catalog Purchases", min_value=0)
    numstorepurchases = st.number_input("Number of Store Purchases", min_value=0)
    numwebvisitsmonth = st.number_input("Number of Web Visits Month", min_value=0)
    acceptedcmp3 = st.selectbox("Accepted Campaign 3", ["No", "Yes"])
    acceptedcmp4 = st.selectbox("Accepted Campaign 4", ["No", "Yes"])
    acceptedcmp5 = st.selectbox("Accepted Campaign 5", ["No", "Yes"])
    acceptedcmp1 = st.selectbox("Accepted Campaign 1", ["No", "Yes"])
    acceptedcmp2 = st.selectbox("Accepted Campaign 2", ["No", "Yes"])
    complain = st.selectbox("Complain", ["No", "Yes"])
    response = st.selectbox("Response", ["No", "Yes"])
    selected_date = st.date_input("Customer Acquisition Date")
    current_date = datetime.today().date()
    if selected_date is not None:
        customer_tenure = (current_date - selected_date).days
    else:
        customer_tenure = 0
    
    user_data = pd.DataFrame({
        "Year_Birth": [year_birth],
        "Education": [education],
        "Marital_Status": [marital_status],
        "Income": [income],
        "Kidhome": [kidhome],
        "Teenhome": [teenhome],
        "Recency": [recency],
        "MntWines": [mntwines],
        "MntFruits": [mntfruits],
        "MntMeatProducts": [mntmeatproducts],
        "MntFishProducts": [mntfishproducts],
        "MntSweetProducts": [mntsweetproducts],
        "MntGoldProds": [mntgoldprods],
        "NumDealsPurchases": [numdealspurchases],
        "NumWebPurchases": [numwebpurchases],
        "NumCatalogPurchases": [numcatalogpurchases],
        "NumStorePurchases": [numstorepurchases],
        "NumWebVisitsMonth": [numwebvisitsmonth],
        "AcceptedCmp3": [acceptedcmp3],
        "AcceptedCmp4": [acceptedcmp4],
        "AcceptedCmp5": [acceptedcmp5],
        "AcceptedCmp1": [acceptedcmp1],
        "AcceptedCmp2": [acceptedcmp2],
        "Complain": [complain],
        "Response": [response],
        "Customer_Tenure": [customer_tenure]
    })

    if st.button("Segment"):
        user_data_preprocessed, categorical_cols = preprocess_data(user_data)
        cluster = kproto.predict(user_data_preprocessed, categorical=categorical_cols)
    
        st.write("Customer Segment:")
        display_cluster_info(cluster[0])

if __name__ == "__main__":
    main()
