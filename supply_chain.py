import hashlib
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import uuid
import os
import pickle
import time
import base64

# Configurations and constants
THEME_COLORS = {
    "Modern Blue": {
        "primary": "#1E88E5",
        "secondary": "#0D47A1",
        "background": "#12499C",
        "text": "#333333",
        "success": "#4CAF50",
        "warning": "#FF9800",
        "danger": "#F44336",
        "card": "#FFFFFF"
    },
    "Corporate Gray": {
        "primary": "#546E7A",
        "secondary": "#263238",
        "background": "#ECEFF1",
        "text": "#37474F",
        "success": "#2E7D32",
        "warning": "#EF6C00",
        "danger": "#C62828",
        "card": "#FFFFFF"
    },
    "Tech Green": {
        "primary": "#00897B",
        "secondary": "#004D40",
        "background": "#E0F2F1",
        "text": "#004D40",
        "success": "#2E7D32",
        "warning": "#FF8F00",
        "danger": "#D32F2F"
    },
    "Dark Mode": {
        "primary": "#BB86FC",
        "secondary": "#03DAC6",
        "background": "#121212",
        "text": "#E1E1E1",
        "success": "#4CAF50",
        "warning": "#FB8C00",
        "danger": "#CF6679"
    },
    "Supply Chain Blue": {
        "primary": "#2A5CAA",
        "secondary": "#1A3A6F",
        "background": "#F8FAFC",
        "text": "#2D3748",
        "success": "#38A169",
        "warning": "#DD6B20",
        "danger": "#E53E3E",
        "accent": "#4299E1",
        "card": "#FFFFFF"
    },
    "Corporate Dark": {
        "primary": "#4C6FFF",
        "secondary": "#3B4DAA",
        "background": "#1A202C",
        "text": "#E2E8F0",
        "success": "#48BB78",
        "warning": "#ED8936",
        "danger": "#F56565",
        "accent": "#667EEA",
        "card": "#2D3748"
    },
    "Logistics Green": {
        "primary": "#2F855A",
        "secondary": "#276749",
        "background": "#F0FFF4",
        "text": "#1A202C",
        "success": "#38A169",
        "warning": "#DD6B20",
        "danger": "#E53E3E",
        "accent": "#68D391",
        "card": "#FFFFFF"
    },
    "Tech Purple": {
        "primary": "#6B46C1",
        "secondary": "#553C9A",
        "background": "#FAF5FF",
        "text": "#2D3748",
        "success": "#38A169",
        "warning": "#DD6B20",
        "danger": "#E53E3E",
        "accent": "#9F7AEA",
        "card": "#FFFFFF"
    }
}

# App state initialization
def initialize_session_state():
    if "blockchain" not in st.session_state:
        st.session_state.blockchain = Blockchain()
    if "theme" not in st.session_state:
        st.session_state.theme = "Modern Blue"
    if "font_size" not in st.session_state:
        st.session_state.font_size = 16
    if "ml_models" not in st.session_state:
        st.session_state.ml_models = {}
    if "transaction_history" not in st.session_state:
        st.session_state.transaction_history = []
    if "authentication" not in st.session_state:
        st.session_state.authentication = {"username": "admin", "password": hashlib.sha256("admin123".encode()).hexdigest()}
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    if "notifications" not in st.session_state:
        st.session_state.notifications = []

# Blockchain Class with enhanced security and features
class Blockchain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        # Genesis block
        self.create_block(proof=1, previous_hash='0', data={"message": "Genesis Block", "timestamp": str(datetime.now())})
    
    def create_block(self, proof, previous_hash, data):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(datetime.now()),
            'proof': proof,
            'previous_hash': previous_hash,
            'data': data,
            'transactions': self.pending_transactions,
            'block_hash': None  # Will be set after creation
        }
        
        # Clear pending transactions as they're now in a block
        self.pending_transactions = []
        
        # Set the block hash
        block['block_hash'] = self.hash(block)
        
        self.chain.append(block)
        # Add to transaction history
        if "transaction_history" in st.session_state:
            st.session_state.transaction_history.append({
                "action": "Block Created", 
                "timestamp": str(datetime.now()),
                "details": f"Block #{block['index']} added with {len(block['transactions'])} transactions"
            })
        return block

    def add_transaction(self, sender, receiver, amount, additional_data=None):
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'sender': sender,
            'receiver': receiver,
            'amount': amount,
            'timestamp': str(datetime.now()),
            'additional_data': additional_data or {}
        }
        self.pending_transactions.append(transaction)
        return self.get_previous_block()['index'] + 1

    def get_previous_block(self):
        return self.chain[-1] if self.chain else None

    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False
        difficulty = 4  # Adjust difficulty as needed
        
        start_time = time.time()
        while not check_proof:
            hash_operation = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:difficulty] == '0' * difficulty:
                check_proof = True
            else:
                new_proof += 1
                
            # Safety mechanism to prevent infinite loops
            if time.time() - start_time > 10:  # 10 second timeout
                difficulty = max(1, difficulty - 1)  # Reduce difficulty if taking too long
                start_time = time.time()
                
        return new_proof

    def hash(self, block):
        # Remove block_hash from the block before hashing to avoid recursion
        block_copy = block.copy()
        if 'block_hash' in block_copy:
            del block_copy['block_hash']
            
        encoded_block = json.dumps(block_copy, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def is_chain_valid(self):
        previous_block = self.chain[0]
        block_index = 1
        
        while block_index < len(self.chain):
            block = self.chain[block_index]
            
            # Check if the previous hash matches
            if block['previous_hash'] != self.hash(previous_block):
                return False
            
            # Check if the proof of work is valid
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(str(proof**2 - previous_proof**2).encode()).hexdigest()
            
            if hash_operation[:4] != '0000':
                return False
                
            previous_block = block
            block_index += 1
            
        return True
    
    def get_block_by_hash(self, block_hash):
        return next((block for block in self.chain if block.get('block_hash') == block_hash), None)
    
    def get_transaction_by_id(self, transaction_id):
        # Search in all blocks
        for block in self.chain:
            for transaction in block.get('transactions', []):
                if transaction.get('transaction_id') == transaction_id:
                    return transaction, block.get('block_hash')
        return None, None

# Machine Learning Models for Anomaly Detection
class AnomalyDetector:
    def __init__(self, algorithm="isolation_forest", contamination=0.05):
        self.algorithm = algorithm
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, data, columns=None):
        if columns:
            data = data[columns].dropna()
        else:
            data = data.select_dtypes(include=['number']).dropna()
            
        if len(data) == 0:
            return False
            
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Train model based on selected algorithm
        if self.algorithm == "isolation_forest":
            self.model = IsolationForest(n_estimators=100, contamination=self.contamination, random_state=42)
        else:
            # Default to Isolation Forest
            self.model = IsolationForest(n_estimators=100, contamination=self.contamination, random_state=42)
        
        self.model.fit(scaled_data)
        self.is_trained = True
        self.feature_names = columns if columns else list(data.columns)
        return True
        
    def predict(self, data_point):
        if not self.is_trained:
            return None
            
        # Ensure data_point has the same features as training data
        if not isinstance(data_point, pd.DataFrame):
            if isinstance(data_point, (list, np.ndarray)):
                if len(data_point) != len(self.feature_names):
                    raise ValueError(f"Expected {len(self.feature_names)} features, got {len(data_point)}")
                data_point = pd.DataFrame([data_point], columns=self.feature_names)
            else:
                raise ValueError("data_point must be a DataFrame, list, or NumPy array")
        else:
            data_point = data_point[self.feature_names]
            # Reorder columns to match the order used during training
            
            scaled_point = self.scaler.transform(data_point)
            
        # Predict
        prediction = self.model.predict(scaled_point)
        score = self.model.decision_function(scaled_point)
        
        return {
            "is_anomaly": prediction[0] == -1,
            "anomaly_score": score[0],
            "prediction": prediction[0]
        }
        
    def save_model(self, model_name):
        if not self.is_trained:
            return False
            
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "algorithm": self.algorithm,
            "contamination": self.contamination,
            "is_trained": self.is_trained
        }
        
        if not os.path.exists("models"):
            os.makedirs("models")
        
        with open(f"models/{model_name}.pkl", "wb") as f:
            pickle.dump(model_data, f)
            
        return True
        
    def load_model(self, model_name):
        try:
            with open(f"models/{model_name}.pkl", "rb") as f:
                model_data = pickle.load(f)
                
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_names = model_data["feature_names"]
            self.algorithm = model_data["algorithm"]
            self.contamination = model_data["contamination"]
            self.is_trained = model_data["is_trained"]
            
            return True
        except:
            return False

# Data Processing and Visualization Functions
def preprocess_data(df):
    """Preprocesses dataframe for analysis"""
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    return df

def generate_data_summary(df):
    """Generate summary statistics for the dataset"""
    summary = {}
    
    # Basic info
    summary["shape"] = df.shape
    summary["columns"] = df.columns.tolist()
    summary["dtypes"] = df.dtypes.to_dict()
    
    # Missing values
    missing_values = df.isnull().sum().to_dict()
    summary["missing_values"] = {k: v for k, v in missing_values.items() if v > 0}
    
    # Statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    summary["numeric_stats"] = df[numeric_cols].describe().to_dict()
    
    return summary

def create_transaction_visualization(blockchain):
    """Create visualization of blockchain transactions"""
    if not blockchain.chain or len(blockchain.chain) <= 1:
        return None
    
    # Extract transactions data
    block_sizes = []
    timestamps = []
    
    for block in blockchain.chain[1:]:  # Skip genesis block
        block_sizes.append(len(block['transactions']))
        timestamps.append(datetime.strptime(block['timestamp'].split('.')[0], '%Y-%m-%d %H:%M:%S'))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, 
        y=block_sizes,
        mode='lines+markers',
        name='Transactions per Block',
        marker=dict(size=10, color=THEME_COLORS[st.session_state.theme]["primary"])
    ))
    
    fig.update_layout(
        title='Transactions per Block Over Time',
        xaxis_title='Block Timestamp',
        yaxis_title='Number of Transactions',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_anomaly_visualization(data, predictions, columns):
    """Create visualization of anomaly detection results"""
    if len(columns) <= 1:
        # For single feature
        fig = px.scatter(
            data, x=data.index, y=columns[0],
            color=predictions,
            color_discrete_map={1: 'blue', -1: 'red'},
            labels={1: 'Normal', -1: 'Anomaly'}
        )
        return fig
    
    # For multiple features, use PCA to visualize in 2D
    pca = PCA(n_components=2)
    components = pca.fit_transform(data[columns])
    
    df_pca = pd.DataFrame({
        'PC1': components[:, 0],
        'PC2': components[:, 1],
        'Prediction': predictions
    })
    
    fig = px.scatter(
        df_pca, x='PC1', y='PC2',
        color='Prediction',
        color_discrete_map={1: 'blue', -1: 'red'},
        labels={1: 'Normal', -1: 'Anomaly'},
        title='Anomaly Detection Results (PCA Visualization)'
    )
    
    return fig

# CSS for dynamic theming
def get_css_with_theme(theme_name, font_size):
    theme = THEME_COLORS[theme_name]
    return f"""
    <style>
    .main {{
        background-color: {theme["background"]};
        color: {theme["text"]};
        padding: 2rem;
        border-radius: 10px;
    }}
    .stButton>button {{
        background-color: {theme["primary"]};
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        border: none;
        font-size: {font_size}px;
        transition: all 0.3s ease;
        box-shadow: 0 0 8px {theme["primary"]}99;
    }}
    .stButton>button:hover {{
        background-color: {theme["secondary"]};
        box-shadow: 0 0 12px {theme["primary"]}cc;
    }}
    .stTextInput>div>div>input {{
        border-radius: 5px;
        padding: 10px;
        border-color: {theme["primary"]};
    }}
    .stTextArea>div>div>textarea {{
        border-radius: 5px;
        padding: 10px;
        border-color: {theme["primary"]};
    }}
    .stSelectbox>div>div>div {{
        border-radius: 5px;
        padding: 2px;
        border-color: {theme["primary"]};
    }}
    .stDataFrame {{
        border-radius: 8px;
    }}
    h1, h2, h3 {{
        color: {theme["primary"]};
    }}
    .card {{
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid {theme["primary"]};
    }}
    .success-card {{
        border-left: 5px solid {theme["success"]};
    }}
    .warning-card {{
        border-left: 5px solid {theme["warning"]};
    }}
    .danger-card {{
        border-left: 5px solid {theme["danger"]};
    }}
    .dashboard-metrics {{
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
    }}
    .metric-card {{
        background: linear-gradient(135deg, {theme["primary"]}22, {theme["primary"]}44);
        border-radius: 15px;
        padding: 1rem;
        flex: 1;
        min-width: 150px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }}
    .metric-value {{
        font-size: 1.8rem;
        font-weight: bold;
        color: {theme["primary"]};
    }}
    .metric-label {{
        color: {theme["text"]};
        font-size: 0.9rem;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
    }}
    th, td {{
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }}
    th {{
        background-color: {theme["primary"]}22;
        color: {theme["text"]};
    }}
    tr:hover {{
        background-color: {theme["primary"]}11;
    }}
    .header-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .badge {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }}
    .badge-success {{
        background-color: {theme["success"]};
        color: white;
    }}
    .badge-warning {{
        background-color: {theme["warning"]};
        color: white;
    }}
    .badge-danger {{
        background-color: {theme["danger"]};
        color: white;
    }}
    .notification {{
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .notification-info {{
        background-color: {theme["primary"]}22;
        border-left: 4px solid {theme["primary"]};
    }}
    .notification-success {{
        background-color: {theme["success"]}22;
        border-left: 4px solid {theme["success"]};
    }}
    .notification-warning {{
        background-color: {theme["warning"]}22;
        border-left: 4px solid {theme["warning"]};
    }}
    .notification-danger {{
        background-color: {theme["danger"]}22;
        border-left: 4px solid {theme["danger"]};
    }}
    </style>
    """

# Authentication functions
def login():
    st.markdown("""
    <div style='text-align: center; padding: 20px;
                border-radius: 20px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);'>
        <h2>📊 Secure Supply Chain Management using Blockchain & Anomaly Detection</h2>
    </div>
""", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'> </p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please login to access the system</p>", unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if username == st.session_state.authentication["username"] and hashlib.sha256(password.encode()).hexdigest() == st.session_state.authentication["password"]:
                st.session_state.is_authenticated = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

# Dashboard header with metrics
def render_dashboard_header(blockchain, dataset=None):
    theme = THEME_COLORS[st.session_state.theme]

    st.markdown("""
<div style='text-align: center; padding: 20px; 
            border-radius: 10px; box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3);'>
    <h2>📈 Secure Supply Chain Management using Blockchain & Anomaly Detection</h2>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div class='dashboard-metrics'>", unsafe_allow_html=True)
# ... (Your metrics/content here)
    st.markdown("</div>", unsafe_allow_html=True)  # Close the div
    
    # Blockchain metrics
    total_blocks = len(blockchain.chain)
    total_transactions = sum(len(block.get('transactions', [])) for block in blockchain.chain)
    pending_txs = len(blockchain.pending_transactions)
    
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{total_blocks}</div>
        <div class='metric-label'>Total Blocks</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value'>{total_transactions}</div>
        <div class='metric-label'>Total Transactions</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value'>{pending_txs}</div>
        <div class='metric-label'>Pending Transactions</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset metrics if available
    if dataset is not None:
        rows = dataset.shape[0]
        anomalies = "N/A"  # This would be calculated after anomaly detection
        
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{rows}</div>
            <div class='metric-label'>Dataset Rows</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    # Main application function
def main():
    # Initialize session state
    initialize_session_state()
    
    # Authentication check
    if not st.session_state.is_authenticated:
        login()
        return
    
    # Apply theme
    theme_name = st.session_state.theme
    font_size = st.session_state.font_size
    st.markdown(get_css_with_theme(theme_name, font_size), unsafe_allow_html=True)
    
    # Configure sidebar
    with st.sidebar:
        st.image("https://etimg.etb2bimg.com/photo/103597308.cms", width=100)
        st.header("⚙️ Configuration")
        
        # User account section
        st.subheader("👨🏻‍💻 User Account")
        st.write(f"Logged in as: **{st.session_state.authentication['username']}**")
        if st.button("Logout"):
            st.session_state.is_authenticated = False
            st.rerun()
        
        # Theme settings
        st.subheader("🎨 Theme Settings")
        selected_theme = st.selectbox("Select Theme", list(THEME_COLORS.keys()), index=list(THEME_COLORS.keys()).index(st.session_state.theme))
        if selected_theme != st.session_state.theme:
            st.session_state.theme = selected_theme
            st.rerun()
        
        font_size = st.slider("Font Size", min_value=12, max_value=24, value=st.session_state.font_size)
        if font_size != st.session_state.font_size:
            st.session_state.font_size = font_size
            st.rerun()
        
        # Dataset upload
        st.subheader("📂 Dataset Management")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="Upload a CSV file to analyze supply chain data")
        
        dataset = None
        if uploaded_file is not None:
            try:
                dataset = pd.read_csv(uploaded_file)
                dataset = preprocess_data(dataset)
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                st.write(f"Rows: {dataset.shape[0]}, Columns: {dataset.shape[1]}")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    # Main content
    blockchain = st.session_state.blockchain
    
    # Header metrics
    render_dashboard_header(blockchain, dataset)
    
    # Show notifications
    if st.session_state.notifications:
        st.subheader("📢 Notifications")
        for i, notification in enumerate(st.session_state.notifications):
            notification_type = notification.get("type", "info")
            message = notification.get("message", "")
            
            notification_html = f"""
            <div class='notification notification-{notification_type}'>
                <span>{message}</span>
                <button onclick="this.parentElement.style.display='none'">×</button>
            </div>
            """
            st.markdown(notification_html, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔗 Blockchain Operations", "⚠️ Anomaly Detection", "⚙️ System Settings"])
    
    with tab1:
       st.markdown("""
    <div style='text-align: center; padding: 15px;  
                border-radius: 8px; box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.2);'>
        <h3 style='margin: 0; font-size: 1.5em;'>📈 Supply Chain Dashboard</h3>
    </div>""", unsafe_allow_html=True)    
        
        # Transaction Activity
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("ℹ️ Transaction Activity")
       

        
        # Transaction visualization
    fig = create_transaction_visualization(blockchain)
    if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
            st.info("Not enough blockchain data for visualization. Add some transactions first.")
        
    st.markdown("</div>", unsafe_allow_html=True)
        
        # Recent Activity
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📜 Recent Activity")
        
    if st.session_state.transaction_history:
            history_df = pd.DataFrame(st.session_state.transaction_history[-10:])  # Show last 10 activities
            st.dataframe(history_df, use_container_width=True)
    else:
            st.info("No recent activity to display")
        
    st.markdown("</div>", unsafe_allow_html=True)
        
        # Dataset Overview (if uploaded)
    if dataset is not None:
            # st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Dataset Overview")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Sample Data:**")
                st.dataframe(dataset.head(5))
            
            with col2:
                st.write("**Statistics:**")
                
                # Show basic stats
                summary = generate_data_summary(dataset)
                st.write(f"Total Records: {summary['shape'][0]}")
                st.write(f"Total Features: {summary['shape'][1]}")
                
                # Display missing values if any
                if summary["missing_values"]:
                    st.write("**Missing Values:**")
                    for col, count in summary["missing_values"].items():
                        st.write(f"- {col}: {count} ({count/summary['shape'][0]*100:.1f}%)")
            
            # Data visualization
            st.write("**Data Distribution:**")
            
            numeric_cols = dataset.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column to visualize", numeric_cols)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(dataset[selected_col].dropna(), kde=True, ax=ax)
                st.pyplot(fig)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
    <div style='text-align: center; padding: 15px; background-color: #f0f0f0; 
                border-radius: 8px; box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.2);'>
        <h3 style='margin: 0; font-size: 1.5em;'>🔗 Blockchain Operations</h3>
    </div>
""", unsafe_allow_html=True)
        
        # Add Transaction
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("✚ Add New Transaction")
        
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                sender = st.text_input("Sender", placeholder="e.g., Supplier A")
                amount = st.number_input("💰Amount", min_value=0.0, step=0.01)
            
            with col2:
                receiver = st.text_input("Receiver", placeholder="e.g., Manufacturer B")
                
            additional_data = st.text_area("Additional Data (JSON format)", placeholder='{"productId": "123", "quantity": 50, "status": "shipped"}')
            submit_tx = st.form_submit_button("✅ Submit Transaction")
            
            if submit_tx:
                try:
                    # Parse additional data if provided
                    additional_json = {}
                    if additional_data:
                        try:
                            additional_json = json.loads(additional_data)
                        except json.JSONDecodeError:
                            st.error("Invalid JSON format in Additional Data")
                            additional_json = {"raw_data": additional_data}
                    
                    # Add transaction to blockchain
                    if sender and receiver:
                        blockchain.add_transaction(sender, receiver, amount, additional_json)
                        
                        # Add to transaction history
                        st.session_state.transaction_history.append({
                            "action": "Transaction Added", 
                            "timestamp": str(datetime.now()),
                            "details": f"Transaction from {sender} to {receiver} for {amount}"
                        })
                        
                        st.success("Transaction added successfully!")
                        
                        # Add notification
                        st.session_state.notifications.append({
                            "type": "success",
                            "message": f"New transaction added: {sender} → {receiver}"
                        })
                    else:
                        st.error("Sender and Receiver cannot be empty")
                except Exception as e:
                    st.error(f"Error adding transaction: {str(e)}")
                    
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Mine Blocks
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Mine New Block")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            block_data = st.text_area("Block Data (optional)", placeholder="Additional block information")
        
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("⛏️ Mine New Block"):
                if blockchain.pending_transactions:
                    with st.spinner("Mining block... This may take a moment"):
                        try:
                            # Prepare block data
                            data = {}
                            if block_data:
                                try:
                                    data = json.loads(block_data)
                                except json.JSONDecodeError:
                                    data = {"message": block_data}
                            
                            # Mine the block
                            last_block = blockchain.get_previous_block()
                            proof = blockchain.proof_of_work(last_block['proof'])
                            previous_hash = blockchain.hash(last_block)
                            block = blockchain.create_block(proof, previous_hash, data)
                            
                            # Add notification
                            st.session_state.notifications.append({
                                "type": "success",
                                "message": f"New block mined: #{block['index']} with {len(block['transactions'])} transactions"
                            })
                            
                            st.success(f"Block #{block['index']} mined successfully!")
                        except Exception as e:
                            st.error(f"Error mining block: {str(e)}")
                else:
                    st.warning("No pending transactions to include in the block.")
                    
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # View Blockchain
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔍Blockchain Explorer")
        
        # Add tabs for different views
        explorer_tab1, explorer_tab2, explorer_tab3 = st.tabs(["Blocks", "Transactions", "Search by Hash"])
        
        with explorer_tab1:
            if blockchain.chain:
                # Create a simplified view of the blockchain
                simple_chain = []
                for block in blockchain.chain:
                    simple_block = {
                        "index": block["index"],
                        "timestamp": block["timestamp"].split(".")[0],
                        "transactions": len(block["transactions"]),
                        "block_hash": block.get("block_hash", "N/A")[:16] + "...",
                        "previous_hash": block["previous_hash"][:16] + "..."
                    }
                    simple_chain.append(simple_block)
                
                # Display blocks
                df_chain = pd.DataFrame(simple_chain)
                st.dataframe(df_chain, use_container_width=True)
                
                # Block details
                selected_block = st.selectbox("Select Block for Details", options=[f"Block #{b['index']}" for b in blockchain.chain])
                if selected_block:
                    block_idx = int(selected_block.split("#")[1])
                    block = next((b for b in blockchain.chain if b["index"] == block_idx), None)
                    
                    if block:
                        st.json(block)
            else:
                st.info("Blockchain is empty. Mine some blocks first.")
        
        with explorer_tab2:
            # Extract all transactions from all blocks
            all_transactions = []
            for block in blockchain.chain:
                for tx in block.get("transactions", []):
                    tx_with_block = tx.copy()
                    tx_with_block["block_index"] = block["index"]
                    tx_with_block["block_hash"] = block.get("block_hash", "N/A")[:16] + "..."
                    all_transactions.append(tx_with_block)
            
            if all_transactions:
                # Simplify transactions for display
                simple_txs = [{
                    "tx_id": tx.get("transaction_id", "N/A")[:8] + "...",
                    "sender": tx.get("sender", "N/A"),
                    "receiver": tx.get("receiver", "N/A"),
                    "amount": tx.get("amount", 0),
                    "timestamp": tx.get("timestamp", "N/A").split(".")[0],
                    "block_index": tx.get("block_index", "N/A")
                } for tx in all_transactions]
                
                df_txs = pd.DataFrame(simple_txs)
                st.dataframe(df_txs, use_container_width=True)
                
                # Transaction details
                tx_ids = [f"{tx['sender']} → {tx['receiver']} ({tx['tx_id']})" for tx in simple_txs]
                selected_tx = st.selectbox("Select Transaction for Details", options=tx_ids)
                
                if selected_tx:
                    tx_id = selected_tx.split("(")[1].split(")")[0]
                    tx = next((t for t in all_transactions if t.get("transaction_id", "")[:8] + "..." == tx_id), None)
                    
                    if tx:
                        st.json(tx)
            else:
                st.info("No transactions found in blockchain.")
        
        with explorer_tab3:
            st.subheader("Search Block by Hash")
            hash_input = st.text_input("Enter Block Hash", placeholder="e.g., 0000a1b2c3d4...")
    
            if hash_input:
                block = blockchain.get_block_by_hash(hash_input)
                if block:
                    st.success(f"Block found for hash: {hash_input}")
                    st.json(block)  # Display the full block data
                else:
                    st.error(f"No block found with hash: {hash_input}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Blockchain Verification
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Blockchain Verification")
        
        if st.button("🔍 Verify Blockchain Integrity"):
            with st.spinner("Verifying blockchain..."):
                is_valid = blockchain.is_chain_valid()
                if is_valid:
                    st.success("✅ Blockchain integrity verified! All blocks are valid.")
                else:
                    st.error("❌ Blockchain integrity compromised! Invalid chain detected.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    with tab3:
        st.markdown("""
    <div style='text-align: center; padding: 15px; background-color: #f0f0f0; 
                border-radius: 8px; box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.2);'>
        <h3 style='margin: 0; font-size: 1.5em;'>🔗 Blockchain Operations</h3>
    </div>
""", unsafe_allow_html=True)
        
        if dataset is not None:
            # st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Configure and Train Detection Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                algorithm = st.selectbox(
                    "Select Algorithm",
                    ["Isolation Forest"],
                    help="Isolation Forest works well for detecting outliers in supply chain data"
                )
                
                contamination = st.slider(
                    "Contamination (expected anomaly %)",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.05,
                    step=0.01,
                    help="Expected percentage of anomalies in the dataset"
                )
            
            with col2:
                numeric_cols = dataset.select_dtypes(include=['number']).columns.tolist()
                selected_cols = st.multiselect(
                    "Select Features for Anomaly Detection",
                    options=numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))],
                    help="Select numeric columns to use for anomaly detection"
                )
                
                model_name = st.text_input("Model Name", value="supply_chain_model")
            
            train_model = st.button("🧠 Train Model")
            
            if train_model and selected_cols:
                with st.spinner("Training anomaly detection model..."):
                    # Create and train model
                    detector = AnomalyDetector(algorithm="isolation_forest", contamination=contamination)
                    success = detector.train(dataset, selected_cols)
                    
                    if success:
                        # Save model in session state
                        st.session_state.ml_models[model_name] = detector
                        
                        # Add to history
                        st.session_state.transaction_history.append({
                            "action": "Model Trained", 
                            "timestamp": str(datetime.now()),
                            "details": f"Trained {algorithm} model using {', '.join(selected_cols)}"
                        })
                        
                        st.success(f"✅ Model '{model_name}' trained successfully!")
                    else:
                        st.error("❌ Failed to train model. Please check your data and selected features.")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Detect Anomalies Section
            # st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Detect Anomalies")
            
            # List available models
            available_models = list(st.session_state.ml_models.keys())
            
            if available_models:
                selected_model = st.selectbox("Select Trained Model", available_models)
                
                if st.button("🔍 Detect Anomalies in Dataset"):
                    with st.spinner("Detecting anomalies..."):
                        detector = st.session_state.ml_models[selected_model]
                        
                        # Get features used in the model
                        features = detector.feature_names
                        
                        # Prepare data
                        detection_data = dataset[features].dropna()
                        
                        # Make predictions
                        predictions = []
                        anomaly_scores = []
                        
                        for i, row in detection_data.iterrows():
                            result = detector.predict(row.to_frame().T)
                            predictions.append(result["prediction"])
                            anomaly_scores.append(result["anomaly_score"])
                        
                        # Add predictions to the dataset
                        detection_results = detection_data.copy()
                        detection_results["anomaly"] = predictions
                        detection_results["anomaly_score"] = anomaly_scores
                        
                        # Count anomalies
                        anomaly_count = sum(1 for p in predictions if p == -1)
                        anomaly_percentage = anomaly_count / len(predictions) * 100
                        
                        # Display results
                        st.write(f"**Results:** Found {anomaly_count} anomalies ({anomaly_percentage:.2f}% of data)")
                        
                        # Visualize results
                        fig = create_anomaly_visualization(detection_data, predictions, features)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display anomalies
                        st.write("**Anomalies:**")
                        anomalies = detection_results[detection_results["anomaly"] == -1].sort_values(by="anomaly_score")
                        st.dataframe(anomalies)
                        
                        # Add to blockchain as transaction
                        add_to_blockchain = st.button("📝 Record Anomaly Detection Results to Blockchain")
                        if add_to_blockchain:
                            # Prepare data for blockchain
                            blockchain_data = {
                                "analysis_type": "anomaly_detection",
                                "model": selected_model,
                                "features": features,
                                "anomaly_count": anomaly_count,
                                "anomaly_percentage": anomaly_percentage,
                                "timestamp": str(datetime.now())
                            }
                            
                            # Add as transaction
                            blockchain.add_transaction(
                                sender="AnomalyDetectionSystem",
                                receiver="BlockchainLedger",
                                amount=0,
                                additional_data=blockchain_data
                            )
                            
                            st.success("✅ Anomaly detection results recorded to blockchain!")
            else:
                st.info("No trained models available. Please train a model first.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Please upload a dataset to use anomaly detection features.")

    with tab4:
        st.markdown("""
    <div style='text-align: center; padding: 15px; background-color: #f0f0f0; 
                border-radius: 8px; box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.2);'>
        <h3 style='margin: 0; font-size: 1.5em;'>⚙️ System Settings</h3>
    </div>
""", unsafe_allow_html=True)
        
        # Export/Import Blockchain
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Export/Import Blockchain")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Export Blockchain**")
            if st.button("📤 Export Blockchain to JSON"):
                blockchain_json = json.dumps(blockchain.chain, indent=4)
                b64 = base64.b64encode(blockchain_json.encode()).decode()
                href = f'<a href="data:application/json;base64,{b64}" download="blockchain_export.json">Download JSON File</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            st.write("**Import Blockchain**")
            uploaded_blockchain = st.file_uploader("Upload Blockchain JSON", type=["json"])
            
            if uploaded_blockchain is not None:
                try:
                    imported_chain = json.loads(uploaded_blockchain.getvalue().decode())
                    if st.button("📥 Import Blockchain"):
                        # Basic validation
                        if isinstance(imported_chain, list) and all('index' in block for block in imported_chain):
                            # Create new blockchain with imported chain
                            new_blockchain = Blockchain()
                            new_blockchain.chain = imported_chain
                            
                            # Check if valid
                            if new_blockchain.is_chain_valid():
                                st.session_state.blockchain = new_blockchain
                                st.success("✅ Blockchain imported successfully!")
                                
                                # Add notification
                                st.session_state.notifications.append({
                                    "type": "success",
                                    "message": f"Blockchain imported with {len(imported_chain)} blocks"
                                })
                            else:
                                st.error("❌ Imported blockchain is invalid!")
                        else:
                            st.error("❌ Invalid blockchain format!")
                except Exception as e:
                    st.error(f"Error importing blockchain: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Change Authentication
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Change Password")
        
        with st.form("change_password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            submit_password = st.form_submit_button("Change Password")
            
            if submit_password:
                if hashlib.sha256(current_password.encode()).hexdigest() == st.session_state.authentication["password"]:
                    if new_password == confirm_password:
                        if len(new_password) >= 6:
                            st.session_state.authentication["password"] = hashlib.sha256(new_password.encode()).hexdigest()
                            st.success("✅ Password changed successfully!")
                        else:
                            st.error("❌ New password must be at least 6 characters long!")
                    else:
                        st.error("❌ New passwords do not match!")
                else:
                    st.error("❌ Current password is incorrect!")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # System About & Information
        # st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("About System")
        
        st.write("""
        **Supply Chain Blockchain & Anomaly Detection System v2.0**
        
        This advanced system combines blockchain technology with machine learning to provide:
        
        - Secure, immutable supply chain record-keeping
        - Real-time transaction monitoring
        - Anomaly detection for supply chain data
        - Data visualization and analytics
        - Multiple user authentication
        
        Designed for enterprises seeking to enhance supply chain transparency,
        reduce fraud, and improve operational efficiency.
        """)
        
        # System metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Blockchain Size", f"{len(blockchain.chain)} blocks")
        with col2:
            st.metric("Transactions", sum(len(block.get('transactions', [])) for block in blockchain.chain))
        with col3:
            st.metric("Models", len(st.session_state.ml_models))
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()