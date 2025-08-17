# Importing Libraries
import base64
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, \
    classification_report
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

# --- Background Styling (fixed with local image) ---
def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }}

    /* Dark overlay for contrast */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.6);
        z-index: 0;
    }}

    /* Ensure content sits above overlay */
    .block-container {{
        position: relative;
        z-index: 1;
        padding: 3rem 5rem;
    }}

    /* Text styling for readability */
h1, h2, h3, h4, h5, h6, p {{
    color: white !important;
    text-shadow: 1px 1px 5px rgba(0,0,0,0.8);
}}

/* Global Button Styling */
div.stButton > button {{
    background-color: #333333 !important;   /* dark grey / almost black */
    color: white !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    font-size: 1.1rem !important;
    border: none;
    transition: all 0.3s ease;
    cursor: pointer;
}}

div.stButton > button:hover {{
    background-color: #555555 !important;   /* lighter grey on hover */
    transform: scale(1.05);
}}
</style>
"""

    st.markdown(page_bg_img, unsafe_allow_html=True)

# Use your downloaded licensed image here
set_bg("istockphoto-1711144461-612x612.jpg")   # <-- put your iStock image in the same folder

# --- Hero Section ---
st.markdown("""
# Credit Card Fraud Detection
### Detect fraudulent transactions in real-time.
Leverage AI-powered insights to keep your financial transactions secure.
""")


# File uploader
data_file = st.file_uploader("Upload CSV", type=["csv"])

# Global Variables for storing processed data and models
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'target_encoder' not in st.session_state:
    st.session_state.target_encoder = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None


def preprocess_datetime_features(df):
    """Extract features from datetime column"""
    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day'] = df['trans_date_trans_time'].dt.day
        df['month'] = df['trans_date_trans_time'].dt.month
        df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
        df = df.drop(['trans_date_trans_time'], axis=1)

    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'])
        df['age'] = 2024 - df['dob'].dt.year
        df = df.drop(['dob'], axis=1)

    return df


def page1():
    st.title("Credit Card Fraud Detection - Data Overview")

    df = None

    if data_file is not None:
        # File details
        if st.button("View File Details"):
            file_details = {
                "File Name": data_file.name,
                "File Type": data_file.type,
                "File Size (KB)": round(data_file.size / 1024, 2)
            }
            st.write(file_details)

        try:
            # Read CSV and store in session state immediately
            df = pd.read_csv(data_file)
            st.session_state.data = df.copy()  # Store original data in session state

            st.success(
                f"File uploaded successfully! Dataset contains {df.shape[0]:,} rows and {df.shape[1]:,} columns.")

            # Preview
            st.subheader("Data Preview")
            if st.checkbox("Show First 100 Records"):
                st.dataframe(df.head(100))

            # Summary statistics
            st.subheader("Summary Statistics (Numeric Columns)")
            if st.checkbox("Display Summary Statistics"):
                st.write(df.describe())

            # Data information
            st.subheader("Dataset Structure")
            if st.checkbox("Show Column Details"):
                col_info = pd.DataFrame({
                    'Column Name': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Missing Values': df.isnull().sum()
                })
                st.dataframe(col_info)

            # Fraud distribution
            if 'is_fraud' in df.columns:
                st.subheader("Fraud vs. Non-Fraud Transactions")
                fraud_counts = df['is_fraud'].value_counts()
                fig_pie = px.pie(
                    values=fraud_counts.values,
                    names=['Non-Fraud' if i == 0 else 'Fraud' for i in fraud_counts.index],
                    title="Fraudulent Transaction Proportion"
                )
                st.plotly_chart(fig_pie)

                fig_bar = px.bar(
                    x=['Non-Fraud' if i == 0 else 'Fraud' for i in fraud_counts.index],
                    y=fraud_counts.values,
                    title="Transaction Count by Type",
                    labels={'x': 'Transaction Type', 'y': 'Number of Transactions'}
                )
                st.plotly_chart(fig_bar)
            else:
                st.warning("The column 'is_fraud' is missing from the dataset. Fraud analysis will be skipped.")

            # Histograms for numeric features
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numerical_cols:
                st.subheader("Distribution of Numeric Features")
                selected_num_col = st.selectbox("Choose a numeric column to visualize", numerical_cols)
                fig_hist = px.histogram(
                    df,
                    x=selected_num_col,
                    title=f"Distribution of {selected_num_col}"
                )
                st.plotly_chart(fig_hist)

            # Correlation matrix
            if len(numerical_cols) > 1:
                st.subheader("Correlation Analysis")
                if st.checkbox("Display Correlation Heatmap"):
                    corr_matrix = df[numerical_cols].corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred while reading the CSV file: {e}")
            st.info("Please upload a valid CSV file containing tabular data.")
    else:
        st.info("Upload a CSV file to explore the dataset.")

    # Display current session state info
    if st.session_state.data is not None:
        st.sidebar.success("Data loaded successfully")
        st.sidebar.info(f"Dataset shape: {st.session_state.data.shape}")
    else:
        st.sidebar.warning("No data loaded")


def page2(target_col=None):
    st.header('Data Preprocessing')

    # Check if data exists in session state
    if st.session_state.data is None:
        st.warning("Please upload a CSV file in the 'Dataset Overview' section first.")
        return

    # Use data from session state
    df = st.session_state.data.copy()

    st.subheader('1. Initial Data Processing')
    st.write(f"Original dataset shape: {df.shape}")

    # Handle datetime features
    st.subheader('2. Feature Engineering')
    if 'trans_date_trans_time' in df.columns or 'dob' in df.columns:
        if st.checkbox("Extract datetime features"):
            df = preprocess_datetime_features(df)
            st.success("Datetime features extracted!")
            st.write("New features created: hour, day, month, dayofweek, age")

    # Remove unnecessary columns
    st.subheader('3. Remove Unnecessary Columns')
    cols_to_drop = []
    # Identify columns to potentially drop
    high_cardinality_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() > 100:
            high_cardinality_cols.append(col)

    if high_cardinality_cols:
        st.write("High cardinality categorical columns found:")
        st.write(f"Found columns: {high_cardinality_cols}")

        
        # Create default selection based on common high-cardinality columns that actually exist
        common_drop_cols = ['cc_num', 'trans_num', 'first', 'last', 'street', 'merchant', 'trans_id', 'user_id']
        default_drops = [col for col in common_drop_cols if col in high_cardinality_cols]

        selected_drops = st.multiselect(
            "Select columns to drop (recommended for high cardinality text columns):",
            high_cardinality_cols,
            default=default_drops
        )
        cols_to_drop.extend(selected_drops)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        st.success(f"Dropped columns: {cols_to_drop}")

    st.subheader('4. Checking Missing Values')
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.write("Missing values found:")
        st.write(missing_values[missing_values > 0])

        # Handle missing values
        strategy = st.selectbox("Select strategy for numerical features",
                                ["mean", "median", "most_frequent"])

        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        if len(numerical_cols) > 0:
            imputer_num = SimpleImputer(strategy=strategy)
            df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

        if len(categorical_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

        st.success("Missing values handled!")
    else:
        st.success("No missing values found!")

    st.subheader("5. Encoding Categorical Variables")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Remove target column from categorical encoding if it exists
    target_col = None
    for col in ['is_fraud', 'fraud', 'Fraud']:
        if col in df.columns:
            target_col = col
            break

    if target_col and target_col in categorical_cols:
        categorical_cols.remove(target_col)

    if categorical_cols:
        st.write(f"Categorical columns found: {categorical_cols}")

        # Encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        st.success("Categorical variables encoded!")
        # Store encoders for later use
        st.session_state.label_encoders = label_encoders

    # Handle target variable
    if target_col and target_col in df.columns:
        if df[target_col].dtype == 'object':
            le_target = LabelEncoder()
            df[target_col] = le_target.fit_transform(df[target_col])
            st.info(f"Target variable '{target_col}' encoded")
            # Store target encoder for later use
            st.session_state.target_encoder = le_target

    st.subheader("6. Feature Scaling")
    scaling_method = st.selectbox("Select scaling method",
                                  ["StandardScaler", "MinMaxScaler", "None"])

    if scaling_method != "None":
        # Get feature columns (exclude target)
        feature_cols = [col for col in df.columns if col not in ['is_fraud', 'fraud', 'Fraud']]

        if scaling_method == "StandardScaler":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

        st.session_state.scaler = scaler
        st.success(f"Features scaled using {scaling_method}!")

        # Show comparison
        if st.checkbox("Compare Raw vs Scaled Data"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("Raw Data")
                st.write(df[feature_cols].describe())
            with col2:
                st.write("Scaled Data")
                st.write(df_scaled[feature_cols].describe())

        df = df_scaled

    st.subheader("7. Final Processed Dataset")
    if st.checkbox("Show Processed Data"):
        st.dataframe(df.head())

    # Save processed data and feature columns to session state
    st.session_state.processed_data = df
    st.session_state.feature_columns = [col for col in df.columns if col not in ['is_fraud', 'fraud', 'Fraud']]
    st.success("Data preprocessing completed! Processed data saved.")

    # Show final data info
    st.info(f"Final dataset shape: {df.shape}")
    st.info(f"Number of features: {len(st.session_state.feature_columns)}")

    # Update sidebar status
    st.sidebar.success("Data preprocessed")
    st.sidebar.info(f"Features: {len(st.session_state.feature_columns)}")


def page3():
    st.subheader('Model Training - XGBoost & Random Forest')

    # Check if processed data exists
    if st.session_state.processed_data is None:
        st.warning("Please complete data preprocessing first.")
        return

    df = st.session_state.processed_data

    # Find target column
    target_col = None
    for col in ['is_fraud', 'fraud', 'Fraud']:
        if col in df.columns:
            target_col = col
            break

    if not target_col:
        st.error("Target column not found. Expected 'is_fraud', 'fraud', or 'Fraud'.")
        return

    # Debug: Show target column info
    st.write("*Target Column Analysis:*")
    target_series = df[target_col]
    st.write(f"Target column: {target_col}")
    st.write(f"Data type: {target_series.dtype}")
    st.write(f"Unique values: {sorted(target_series.unique())}")
    st.write(f"Value counts:\n{target_series.value_counts().sort_index()}")

    # Convert continuous target to binary classes
    if target_series.dtype in ['float64', 'float32'] or len(target_series.unique()) > 2:
        st.warning("Converting continuous/multi-class target to binary classification...")

        # Option 1: Threshold-based conversion (recommended for fraud detection)
        threshold = st.slider("Fraud threshold (values >= threshold = Fraud)",
                              min_value=float(target_series.min()),
                              max_value=float(target_series.max()),
                              value=0.5,
                              step=0.01,
                              format="%.3f")

        # Convert to binary
        y = (target_series >= threshold).astype(int)

        st.write(f"After conversion with threshold {threshold}:")
        st.write(f"Class distribution: {pd.Series(y).value_counts().sort_index()}")

        # Option 2: Alternative - Use only exact matches for known fraud cases
        # Uncomment if you want to try this approach instead
        # st.write("Alternative: Exact value conversion")
        # y_exact = target_series.copy()
        # y_exact = (y_exact == 1.0).astype(int)  # Only exact 1.0 values are fraud
        # st.write(f"Exact conversion (only 1.0 = fraud): {pd.Series(y_exact).value_counts().sort_index()}")

    else:
        # Target is already binary
        y = target_series.astype(int)

    # Prepare features
    X = df.drop(target_col, axis=1)

    # Check for any remaining issues
    if X.isnull().any().any():
        st.warning("Found missing values in features. Filling with mean/mode...")
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0)

    st.subheader("1. Train-Test Split")
    test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, min_value=0)

    # Check if we have both classes for stratification
    if len(y.unique()) < 2:
        st.error("Only one class found in target variable. Cannot proceed with classification.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    st.success(f"Data split completed! Training: {X_train.shape[0]}, Testing: {X_test.shape[0]}")

    # Display class distribution
    st.write("Class distribution in training set:")
    train_dist = pd.Series(y_train).value_counts().sort_index()

    # Safe index renaming
    if len(train_dist) == 2:
        class_labels = []
        for idx in train_dist.index:
            if idx == 0:
                class_labels.append('Not Fraud')
            elif idx == 1:
                class_labels.append('Fraud')
            else:
                class_labels.append(f'Class {idx}')
        train_dist.index = class_labels

    st.bar_chart(train_dist)

    st.header("2. Model Training")

    # Model configuration
    st.subheader("Random Forest Configuration")
    rf_n_estimators = st.slider("Number of estimators", 50, 300, 100)
    rf_max_depth = st.slider("Max depth", 5, 30, 15)

    st.subheader("XGBoost Configuration")
    xgb_n_estimators = st.slider("XGB Number of estimators", 50, 300, 100, key="xgb_n_est")
    xgb_max_depth = st.slider("XGB Max depth", 3, 15, 6, key="xgb_depth")
    xgb_learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1)

    # Initialize models with proper handling
    try:
        # Calculate class weights for imbalanced data
        class_counts = pd.Series(y_train).value_counts().sort_index()
        if len(class_counts) == 2:
            pos_weight = class_counts[0] / class_counts[1]
        else:
            pos_weight = 1

        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                random_state=random_state,
                class_weight='balanced'  # Handle class imbalance
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=xgb_learning_rate,
                random_state=random_state,
                eval_metric='logloss',
                scale_pos_weight=pos_weight,  # Handle class imbalance
                use_label_encoder=False  # Suppress warning
            )
        }
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        return

    # Train models
    if st.button("Train Models", type="primary"):
        trained_models = {}

        with st.spinner("Training models..."):
            for name, model in models.items():
                try:
                    # Ensure data types are correct
                    X_train_clean = X_train.copy()
                    y_train_clean = y_train.copy().astype(int)

                    # Train model
                    model.fit(X_train_clean, y_train_clean)
                    trained_models[name] = model
                    st.success(f"{name} trained successfully!")

                    # Show basic metrics
                    train_score = model.score(X_train_clean, y_train_clean)
                    st.write(f"Training accuracy for {name}: {train_score:.4f}")

                except Exception as e:
                    st.error(f"Error training {name}: {e}")
                    st.write(f"Debug info - X_train shape: {X_train.shape}, y_train unique: {y_train.unique()}")

        if trained_models:
            # Save everything to session state
            st.session_state.models = trained_models
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test.astype(int)
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train.astype(int)
            st.session_state.target_col = target_col

            st.success("All models trained and saved!")

    # Update sidebar status
    if hasattr(st.session_state, 'models') and st.session_state.models:
        st.sidebar.success("Models trained")
        st.sidebar.info(f"Models: {list(st.session_state.models.keys())}")
    else:
        st.sidebar.warning("Models not trained")


def page4():
    st.title("Model Evaluation")

    if not st.session_state.models:
        st.warning("Please train models first.")
        return

    models = st.session_state.models
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train

    st.subheader("1. Leave-One-Out Cross Validation")

    # Note: LOO CV is computationally expensive, so we'll use a subset for demonstration
    sample_size = st.slider("Sample size for LOO CV (full dataset may take very long)", 100, 1000, 500)

    if st.button("Run Leave-One-Out CV", type="primary"):
        with st.spinner("Running Leave-One-Out Cross Validation..."):
            # Sample data for LOO CV
            X_sample = X_train.sample(n=min(sample_size, len(X_train)), random_state=42)
            y_sample = y_train.loc[X_sample.index]

            loo = LeaveOneOut()
            loo_results = {}

            for name, model in models.items():
                try:
                    scores = cross_val_score(model, X_sample, y_sample, cv=loo, scoring='accuracy', n_jobs=-1)
                    loo_results[name] = {
                        'mean_accuracy': scores.mean(),
                        'std_accuracy': scores.std()
                    }
                    st.success(f"{name} LOO CV completed!")
                except Exception as e:
                    st.error(f"Error in LOO CV for {name}: {e}")

            # Display LOO results
            if loo_results:
                st.write("Leave-One-Out Cross Validation Results:")
                loo_df = pd.DataFrame(loo_results).T
                st.dataframe(loo_df.round(4))

    st.subheader("2. Test Set Classification Metrics")

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='binary'),
            'Recall': recall_score(y_test, y_pred, average='binary'),
            'F1-Score': f1_score(y_test, y_pred, average='binary'),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

    # Display metrics table
    metrics_df = pd.DataFrame({
        name: [results[name]['Accuracy'], results[name]['Precision'],
               results[name]['Recall'], results[name]['F1-Score']]
        for name in results.keys()
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])

    st.dataframe(metrics_df.round(4))

    # Best model
    best_model_name = metrics_df.loc['F1-Score'].idxmax()  # Use F1-Score for imbalanced dataset
    st.success(f"Best performing model (by F1-Score): **{best_model_name}**")

    st.subheader("3. Confusion Matrix")
    selected_model = st.selectbox("Select model for detailed analysis", list(models.keys()))

    if selected_model:
        y_pred = results[selected_model]['y_pred']
        cm = confusion_matrix(y_test, y_pred)

        # Create confusion matrix with labels
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Not Fraud', 'Fraud'],
                    yticklabels=['Not Fraud', 'Fraud'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {selected_model}')
        st.pyplot(fig)

    st.subheader("4. ROC Curve")
    if selected_model and results[selected_model]['y_pred_proba'] is not None:
        y_pred_proba = results[selected_model]['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 name=f'{selected_model} (AUC = {roc_auc:.3f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 name='Random Classifier', line=dict(dash='dash')))
        fig.update_layout(
            title=f'ROC Curve - {selected_model}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700, height=500
        )
        st.plotly_chart(fig)

        st.info(f"AUC Score: {roc_auc:.3f}")

    st.subheader("5. Model Performance Comparison")

    # Bar chart comparing all models
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=list(results.keys()),
               y=[results[name]['Accuracy'] for name in results.keys()]),
        go.Bar(name='Precision', x=list(results.keys()),
               y=[results[name]['Precision'] for name in results.keys()]),
        go.Bar(name='Recall', x=list(results.keys()),
               y=[results[name]['Recall'] for name in results.keys()]),
        go.Bar(name='F1-Score', x=list(results.keys()),
               y=[results[name]['F1-Score'] for name in results.keys()])
    ])
    fig.update_layout(title='Model Performance Comparison', barmode='group')
    st.plotly_chart(fig)

    # Detailed classification report
    st.subheader("6. Detailed Classification Report")
    if selected_model:
        y_pred = results[selected_model]['y_pred']
        report = classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud'], output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df)

    # Update sidebar status
    st.sidebar.success("Models evaluated")


def page5():
    st.header("Fraud Prediction")

    if not st.session_state.models:
        st.warning("Please train models first.")
        return

    if st.session_state.processed_data is None:
        st.warning("Please complete data preprocessing first.")
        return

    feature_cols = st.session_state.feature_columns
    if not feature_cols:
        st.error("Feature columns not found.")
        return

    df = st.session_state.processed_data

    st.subheader("1. Enter Transaction Information")

    # Create input form with more intuitive layout
    input_data = {}

    # Key transaction features
    st.write("*Primary Transaction Details:*")
    col1, col2 = st.columns(2)

    with col1:
        if 'amt' in feature_cols:
            input_data['amt'] = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                value=50.0,  # Changed from 50 to 50.0 (float)
                step=0.01
            )

    with col2:
        if 'city_pop' in feature_cols:
            # Fix: Safe max value calculation with fallback
            try:
                if 'city_pop' in df.columns:
                    max_pop = int(df['city_pop'].max())
                    # Ensure max_pop is reasonable (at least 100k)
                    max_pop = max(max_pop, 100000)
                else:
                    max_pop = 1000000  # Default fallback
            except:
                max_pop = 1000000  # Safe fallback

            # Debug info to see what's happening
            st.write(f"Debug: max_pop = {max_pop}")

            input_data['city_pop'] = st.number_input(
                "City Population",
                min_value=0,
                max_value=max_pop,
                value=min(50000, max_pop),  # Ensure value doesn't exceed max
                step=1
            )

    # Time-based features
    if any(col in feature_cols for col in ['hour', 'day', 'month', 'dayofweek']):
        st.write("*Time-based Features:*")
        time_cols = st.columns(4)

        if 'hour' in feature_cols:
            with time_cols[0]:
                input_data['hour'] = st.selectbox("Hour of Day", range(24), index=12)
        if 'day' in feature_cols:
            with time_cols[1]:
                input_data['day'] = st.selectbox("Day of Month", range(1, 32), index=15)
        if 'month' in feature_cols:
            with time_cols[2]:
                input_data['month'] = st.selectbox("Month", range(1, 13), index=6)
        if 'dayofweek' in feature_cols:
            with time_cols[3]:
                input_data['dayofweek'] = st.selectbox("Day of Week",
                                                       options=range(7),
                                                       format_func=lambda x:
                                                       ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x],
                                                       index=0)

    # Remaining features
    st.write("*Other Features:*")
    remaining_features = [col for col in feature_cols if col not in input_data.keys()]

    # Create columns for remaining features
    num_cols = 3
    for i in range(0, len(remaining_features), num_cols):
        cols = st.columns(num_cols)
        for j, col_name in enumerate(remaining_features[i:i + num_cols]):
            with cols[j]:
                if col_name in df.columns:
                    # Fix: Ensure consistent data types
                    try:
                        min_val = float(df[col_name].min())
                        max_val = float(df[col_name].max())
                        mean_val = float(df[col_name].mean())

                        input_data[col_name] = st.number_input(
                            f"{col_name}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=0.01,  # Use consistent float step
                            key=col_name,
                            format="%.2f"
                        )
                    except Exception as e:
                        # Fallback for problematic columns
                        input_data[col_name] = st.number_input(
                            f"{col_name}",
                            value=0.0,
                            step=0.01,
                            key=col_name
                        )

    st.subheader("2. Model Selection")
    selected_model_name = st.selectbox(
        "Select model for prediction",
        list(st.session_state.models.keys())
    )

    # Make prediction
    if st.button("Analyze Transaction", type="primary"):
        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])

            # Ensure all feature columns are present
            for col in feature_cols:
                if col not in input_df.columns:
                    input_df[col] = float(df[col].mean())  # Ensure float type

            # Reorder columns to match training data
            input_df = input_df[feature_cols]

            # Apply same scaling if used during training
            if hasattr(st.session_state, 'scaler') and st.session_state.scaler is not None:
                input_scaled = st.session_state.scaler.transform(input_df)
                input_df = pd.DataFrame(input_scaled, columns=feature_cols)

            # Make prediction
            model = st.session_state.models[selected_model_name]
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]

            # Display results
            st.header("3. Fraud Analysis Results")

            col1, col2 = st.columns(2)

            with col1:
                if prediction == 1:
                    st.error("*FRAUD DETECTED*")
                    st.error("This transaction is likely fraudulent!")
                    risk_level = "HIGH RISK"
                    risk_color = "red"
                else:
                    st.success("*LEGITIMATE TRANSACTION*")
                    st.success("This transaction appears to be legitimate!")
                    risk_level = "LOW RISK"
                    risk_color = "green"

            with col2:
                fraud_probability = prediction_proba[1]
                confidence = max(prediction_proba)

                st.metric("Fraud Probability", f"{fraud_probability:.1%}")
                st.metric("Confidence", f"{confidence:.1%}")

            # Risk meter visualization
            st.subheader("Risk Assessment")
            risk_score = fraud_probability * 100

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fraud Risk Score"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if risk_score > 50 else "darkgreen"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))

            fig.update_layout(height=400)
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Debug info:")
            st.write(f"Input data shape: {input_df.shape if 'input_df' in locals() else 'Not created'}")
            st.write(f"Feature columns: {len(feature_cols)}")

    # Update sidebar status
    st.sidebar.success("Ready for prediction")


def page6():
    st.header("Interpretation and Conclusions")

    st.subheader("1. Project Summary")
    st.write("""
    This Credit Card Fraud Detection application demonstrates a complete machine learning pipeline
    for identifying fraudulent transactions in credit card data. The application includes:

    - **Data Import and Overview**: Loading and exploring credit card transaction data
    - **Data Preprocessing**: Feature engineering, handling missing values, encoding, and scaling
    - **Model Training**: Implementing XGBoost and Random Forest algorithms optimized for fraud detection
    - **Model Evaluation**: Comprehensive performance analysis using Leave-One-Out CV and standard metrics
    - **Prediction Interface**: Real-time fraud detection for new transactions
    """)

    if st.session_state.models and st.session_state.X_test is not None:
        st.subheader("2. Model Performance Analysis")

        models = st.session_state.models
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        # Recalculate results for analysis
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = auc(fpr, tpr)

            results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'AUC': auc_score
            }

        # Performance comparison
        st.subheader("Model Performance Comparison")
        comparison_df = pd.DataFrame(results).T.round(4)
        st.dataframe(comparison_df)

        # Best model analysis
        best_f1_model = comparison_df['F1-Score'].idxmax()
        best_auc_model = comparison_df['AUC'].idxmax()

        st.success(f"**Best F1-Score**: {best_f1_model} ({comparison_df.loc[best_f1_model, 'F1-Score']:.4f})")
        st.success(f"**Best AUC**: {best_auc_model} ({comparison_df.loc[best_auc_model, 'AUC']:.4f})")

        # Feature importance analysis
        st.subheader("3. Feature Importance Analysis")

        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                st.write(f"**{name} - Most Important Features:**")
                feature_names = st.session_state.feature_columns
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)

                # Top 10 features
                top_features = importance_df.head(10)

                fig = px.bar(top_features, x='Importance', y='Feature',
                             orientation='h',
                             title=f'Top 10 Features - {name}',
                             height=500)
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig)

                # Show top features
                st.write("Top 5 Most Predictive Features:")
                for idx, row in top_features.head().iterrows():
                    st.write(f"- **{row['Feature']}**: {row['Importance']:.4f}")
    else:
        st.info("Train models first to see detailed performance analysis.")

    st.subheader("4. Key Insights and Business Impact")

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        ### Key Findings:
        - **Transaction Amount**: Often a critical factor in fraud detection
        - **Time-based Features**: Hour and day patterns reveal suspicious timing
        - **Location Data**: Geographic inconsistencies indicate fraud
        - **Account Age**: Newer accounts may have higher fraud risk
        """)

    with col2:
        st.write("""
        ### Business Impact:
        - **Cost Savings**: Early fraud detection saves millions in losses
        - **Customer Trust**: Reduced false positives improve customer experience  
        - **Real-time Protection**: Instant transaction analysis prevents fraud
        - **Compliance**: Helps meet regulatory requirements
        """)

    st.subheader("5. Model Trade-offs and Recommendations")

    # Model comparison insights
    if st.session_state.models:
        st.write("""
        ### Model Performance Trade-offs:

        **Random Forest:**
        - Excellent interpretability with feature importance
        - Robust to overfitting
        - Handles mixed data types well
        - May be slower on very large datasets

        **XGBoost:**
        - Often superior predictive performance
        - Efficient memory usage and speed
        - Built-in regularization prevents overfitting
        - More complex hyperparameter tuning
        """)

    st.subheader("6. Implementation Recommendations")

    st.write("""
    ### Deployment Strategy:
    1. **Model Selection**: Deploy the best-performing model (typically XGBoost for fraud)
    2. **Threshold Tuning**: Optimize decision threshold based on business costs
    3. **Real-time Integration**: Implement API for live transaction scoring
    4. **Monitoring**: Set up automated model performance monitoring
    5. **Retraining**: Schedule regular model updates with new fraud patterns

    ### Risk Management:
    - **False Positives**: Balance fraud detection with customer experience
    - **False Negatives**: Minimize missed fraud while avoiding over-blocking
    - **Model Drift**: Monitor for changes in fraud patterns over time
    - **Explainability**: Maintain interpretable features for regulatory compliance
    """)

    st.subheader("7. Future Enhancements")

    st.write("""
    ### Advanced Techniques:
    - **Deep Learning**: Neural networks for complex pattern recognition
    - **Ensemble Methods**: Combine multiple models for better performance
    - **Real-time Features**: Incorporate streaming transaction history
    - **Graph Analytics**: Detect fraud networks and connections
    - **Anomaly Detection**: Unsupervised learning for unknown fraud types

    ### Business Intelligence:
    - **Fraud Trend Analysis**: Track fraud patterns over time
    - **Geographic Risk Mapping**: Identify high-risk locations
    - **Merchant Risk Scoring**: Assess merchant-level fraud risks
    - **Customer Behavior Profiling**: Build individual risk profiles
    """)

    st.success("""
    **Conclusion**: This fraud detection system provides a robust foundation for protecting 
    against credit card fraud while maintaining excellent customer experience. The combination 
    of XGBoost and Random Forest models offers both high performance and interpretability, 
    crucial for financial applications.
    """)

    # Display session state summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Session Summary")
    if st.session_state.data is not None:
        st.sidebar.success(f"Original Data: {st.session_state.data.shape}")
    if st.session_state.processed_data is not None:
        st.sidebar.success(f"Processed Data: {st.session_state.processed_data.shape}")
    if st.session_state.models:
        st.sidebar.success(f"Trained Models: {len(st.session_state.models)}")
    if st.session_state.X_test is not None:
        st.sidebar.success(f"Test Set: {st.session_state.X_test.shape}")


# Navigation
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Sidebar navigation
st.sidebar.title("Fraud Detection System")
st.sidebar.write("Navigate through the fraud detection pipeline:")

pages = {
    'Dataset Overview': page1,
    'Data Preprocessing': page2,
    'Model Training': page3,
    'Model Evaluation': page4,
    'Fraud Prediction': page5,
    'Conclusions': page6
}

# Page selection
selected_page = st.sidebar.selectbox("Select a page:", list(pages.keys()))

# Add some information in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About This App
This application uses machine learning to detect credit card fraud in real-time.

**Models Used:**
- Random Forest
- XGBoost

**Evaluation Method:**
- Leave-One-Out Cross Validation
- Standard Classification Metrics

**Key Features:**
- Real-time fraud scoring
- Feature importance analysis  
- Comprehensive model evaluation
""")

# Display current session status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Pipeline Status")

# Check each stage completion
stages = [
    ("Data Upload", st.session_state.data is not None),
    ("Data Preprocessing", st.session_state.processed_data is not None),
    ("Model Training", len(st.session_state.models) > 0),
    ("Model Evaluation", st.session_state.X_test is not None),
]

for stage, completed in stages:
    if completed:
        st.sidebar.success(f"{stage}")
    else:
        st.sidebar.error(f" {stage}")

# Display selected page
pages[selected_page]()