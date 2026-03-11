import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import plotly.graph_objects as go
import re

# Load the saved T5 model and tokenizer
saved_model_path = r"./Model"
saved_tokenizer_path = r"./Model"

model = T5ForConditionalGeneration.from_pretrained(saved_model_path)
tokenizer = T5Tokenizer.from_pretrained(saved_tokenizer_path)


# Sample transaction data (replace this with your actual dataset as needed)
@st.cache_data
def load_transaction_data():
    return pd.DataFrame({
        'transaction_id': range(1, 6),
        'type': ['TRANSFER', 'CASH_OUT', 'CASH_IN', 'PAYMENT', 'TRANSFER'],
        'amount': [420330.71, 50000.00, 10000.00, 25000.00, 100000.00],
        'nameOrig': ['C1868228472', 'C123456789', 'C987654321', 'C555555555', 'C777777777'],
        'nameDest': ['M123456789', 'C888888888', 'C999999999', 'M444444444', 'C666666666'],
        'oldbalanceOrg': [420330.71, 50000.00, 0.00, 25000.00, 100000.00],
        'newbalanceOrig': [0.00, 0.00, 10000.00, 0.0, 0.0],
        'isFraud': [1, 0, 0, 0, 1]
    })

def generate_predictions(model, tokenizer, input_texts, max_length=128):
    device = next(model.parameters()).device
    inputs = tokenizer.batch_encode_plus(input_texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def analyze_transaction(query, df):
    # Define regex patterns for each transaction type
    patterns = {
        "TRANSFER": r"transfer of \$(.*?) from account (C.*?) might be fraudulent\. It started with a balance of \$(.*?) and ended with \$(.*?)\.",
        "CASH_OUT": r"cash_out of \$(.*?) from account (C.*?) might be fraudulent\. It started with a balance of \$(.*?) and ended with \$(.*?)\.",
        "CASH_IN": r"cash_in of \$(.*?) from account (C.*?) might be fraudulent\. It started with a balance of \$(.*?) and ended with \$(.*?)\.",
        "PAYMENT": r"payment of \$(.*?) from account (C.*?) might be fraudulent\. It started with a balance of \$(.*?) and ended with \$(.*?)\."
    }
    
    match = None
    trans_type = None
    for t, pattern in patterns.items():
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            match = m
            trans_type = t
            break

    if match:
        # Extract the captured groups: amount, account, oldbalance, newbalance
        amount = float(match.group(1))
        account = match.group(2)
        oldbalance = float(match.group(3))
        newbalance = float(match.group(4))
        
        # Find the matching transaction based on account, amount, and transaction type
        result = df[(df['nameOrig'] == account) & (df['amount'] == amount) & (df['type'] == trans_type)]
        return result
    
    return pd.DataFrame()

def visualize_transaction(transaction_data):
    if transaction_data.empty:
        return None
    
    labels = list(set(transaction_data['nameOrig'].tolist() + transaction_data['nameDest'].tolist()))
    fig = go.Figure(data=[
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color="blue"
            ),
            link=dict(
                source=[labels.index(x) for x in transaction_data['nameOrig']],
                target=[labels.index(x) for x in transaction_data['nameDest']],
                value=transaction_data['amount'],
                color=["red" if x == 1 else "black" for x in transaction_data['isFraud']]
            )
        )
    ])
    
    fig.update_layout(
        title_text="Transaction Flow Visualization",
        font_size=12,
        height=600
    )
    
    return fig

# Page configuration
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title('Fraud Detection System')

# Inject custom CSS for animations, hover effects, and theme styling
base_css = """
<style>
/* Transition effects for interactive elements */
button[kind], div[data-testid="stMetric"], textarea {
    transition: all 0.3s ease-in-out;
}

/* Hover effect for metric cards */
div[data-testid="stMetric"] {
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

/* Button hover effect */
button[kind] {
    border: none;
    border-radius: 5px;
    padding: 0.5em 1em;
}
button[kind]:hover {
    background-color: #5a5a5a;
    color: white;
}

/* Sidebar item hover effect */
[data-testid="stSidebar"] .sidebar-content {
    border-right: 2px solid #ccc;
}
[data-testid="stSidebar"] .css-1d391kg:hover {
    background-color: #e0e0e0;
}

/* Textarea focus effect */
textarea {
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 0.5em;
}
textarea:focus {
    border-color: #0073e6;
    box-shadow: 0 0 5px rgba(0,115,230,0.5);
}

/* Chart container styling */
.reportview-container .element-container {
    padding: 1rem;
    border: 1px solid transparent;
    border-radius: 8px;
}
.reportview-container .element-container:hover {
    border-color: #ccc;
}
</style>
"""
st.markdown(base_css, unsafe_allow_html=True)

# Sidebar for theme selection
theme = st.sidebar.radio("Select Theme", options=["Light", "Dark"])

if theme == "Dark":
    dark_theme_css = """
    <style>
    .reportview-container, .main, .block-container {
        background-color: #303030;
        color: #e0e0e0;
    }
    .sidebar .sidebar-content {
        background-color: #424242;
    }
    /* Override text and widget colors for dark theme */
    .stMarkdown, .stText, .stButton, .stMetric, .stSelectbox, .stRadio, textarea {
        color: #e0e0e0;
    }
    /* Adjust button hover for dark mode */
    button[kind]:hover {
        background-color: #616161;
    }
    </style>
    """
    st.markdown(dark_theme_css, unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.title('User Input')
user_query = st.sidebar.text_area(
    'Enter your query in natural language', 
    placeholder='Examples:\n'
                'For CASH_IN: "Please check whether the cash_in of $10000.00 from account C987654321 might be fraudulent. It started with a balance of $0.00 and ended with $10000.00."\n'
                'For PAYMENT: "Please check whether the payment of $25000.00 from account C555555555 might be fraudulent. It started with a balance of $25000.00 and ended with $0.0."\n'
                'For TRANSFER: "Please check whether the transfer of $420330.71 from account C1868228472 might be fraudulent. It started with a balance of $420330.71 and ended with $0.0."\n'
                'Other queries not matching our dataset will display a warning.'
)

if st.sidebar.button('Analyze Transaction'):
    if user_query:
        try:
            # Load transaction data
            df = load_transaction_data()
            
            # Generate prediction using the T5 model
            predictions = generate_predictions(model, tokenizer, [user_query])
            st.write("Model Prediction:", predictions[0])
            
            # Analyze transaction based on the query
            result = analyze_transaction(user_query, df)
            
            if not result.empty:
                st.subheader("Transaction Details:")
                for _, transaction in result.iterrows():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Transaction Type:", transaction['type'])
                        st.write("Amount: $", transaction['amount'])
                        st.write("Sender Account:", transaction['nameOrig'])
                    with col2:
                        st.write("Recipient Account:", transaction['nameDest'])
                        st.write("Fraud Status:", "Fraudulent" if transaction['isFraud'] == 1 else "Not Fraudulent")
                        st.write("New Balance: $", transaction['newbalanceOrig'])
                
                # Visualize transaction flow if available
                fig = visualize_transaction(result)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Additional visualizations
                st.subheader("Transaction Analytics")
                chart_col1, chart_col2, chart_col3 = st.columns(3)
                
                with chart_col1:
                    type_counts = df['type'].value_counts()
                    fig_types = go.Figure(data=[
                        go.Pie(labels=type_counts.index, values=type_counts.values, hole=0.3)
                    ])
                    fig_types.update_layout(title="Transaction Types Distribution", height=400)
                    st.plotly_chart(fig_types, use_container_width=True)
                
                with chart_col2:
                    fig_fraud = go.Figure(data=[
                        go.Bar(
                            x=['Non-Fraudulent', 'Fraudulent'],
                            y=[len(df[df['isFraud'] == 0]), len(df[df['isFraud'] == 1])],
                            marker_color=['blue', 'red']
                        )
                    ])
                    fig_fraud.update_layout(title="Fraud vs Non-Fraud Transactions", height=400, showlegend=False)
                    st.plotly_chart(fig_fraud, use_container_width=True)
                
                with chart_col3:
                    fig_amount = go.Figure(data=[
                        go.Box(y=df['amount'], name='Transaction Amounts', marker_color='lightseagreen')
                    ])
                    fig_amount.update_layout(title="Transaction Amount Distribution", height=400)
                    st.plotly_chart(fig_amount, use_container_width=True)
                
                # Summary statistics
                st.subheader("Transaction Summary")
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.metric("Total Transactions", len(df))
                    st.metric("Average Transaction Amount", f"${df['amount'].mean():.2f}")
                
                with stats_col2:
                    st.metric("Total Fraudulent Transactions", len(df[df['isFraud'] == 1]))
                    st.metric("Fraud Rate", f"{(len(df[df['isFraud'] == 1])/len(df)*100):.1f}%")
            else:
                st.warning("No matching transaction found in the dataset.")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
