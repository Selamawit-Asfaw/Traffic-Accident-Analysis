import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

# Import ML libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Traffic Accident Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚦 Traffic Accident Prediction & Analysis System")
st.markdown("**EDA ➜ Machine Learning ➜ Explainability ➜ Deployment**")

# --------------------------------------------------
# DATA LOADING FUNCTIONS
# --------------------------------------------------
@st.cache_data
def load_data():
    """Load the complete processed data for analysis."""
    try:
        # Try to load processed data first (FULL DATA)
        processed_data_path = project_root / "data" / "processed" / "traffic_model_ready.csv"
        
        if processed_data_path.exists():
            st.info("📊 Loading complete processed dataset...")
            with st.spinner("Loading full dataset..."):
                df = pd.read_csv(processed_data_path)  # Load ALL data
            target = "most_severe_injury"
            st.success(f"✅ Loaded complete dataset: {len(df):,} records")
            return df, target, True
        
        # Try to load from splits (FULL DATA)
        data_dir = project_root / "data" / "processed"
        if (data_dir / "X_train.csv").exists():
            st.info("📊 Loading complete training dataset...")
            with st.spinner("Loading full training data..."):
                X_train = pd.read_csv(data_dir / "X_train.csv")  # Load ALL training data
                y_train = pd.read_csv(data_dir / "y_train.csv")  # Load ALL training labels
            
            # Get target column name
            target_col = y_train.columns[0] if len(y_train.columns) == 1 else "most_severe_injury"
            
            # Combine for analysis
            df = X_train.copy()
            df[target_col] = y_train[target_col]
            
            st.success(f"✅ Loaded complete training dataset: {len(df):,} records")
            return df, target_col, True
        
        # Try raw data as fallback (FULL DATA)
        raw_data_path = project_root / "data" / "raw" / "traffic_accidents.csv"
        if raw_data_path.exists():
            st.info("📊 Loading complete raw dataset...")
            with st.spinner("Loading full raw data..."):
                df = pd.read_csv(raw_data_path)  # Load ALL raw data
            target = "most_severe_injury"
            st.success(f"✅ Loaded complete raw dataset: {len(df):,} records")
            return df, target, True
        
        st.error("❌ No data files found. Please ensure data exists in data/processed/ or data/raw/")
        return None, None, False
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.error("💡 If the file is too large, try using the processed splits instead")
        return None, None, False

@st.cache_data
def load_full_splits():
    """Load the complete train/test splits for model training."""
    try:
        data_dir = project_root / "data" / "processed"
        
        st.info("📊 Loading complete train/test splits...")
        with st.spinner("Loading full training and test data..."):
            # Load ALL data without row limits
            X_train = pd.read_csv(data_dir / "X_train.csv")
            X_test = pd.read_csv(data_dir / "X_test.csv")
            y_train = pd.read_csv(data_dir / "y_train.csv")
            y_test = pd.read_csv(data_dir / "y_test.csv")
        
        target_col = y_train.columns[0] if len(y_train.columns) == 1 else "most_severe_injury"
        
        st.success(f"✅ Loaded complete splits - Train: {len(X_train):,}, Test: {len(X_test):,}")
        return X_train, X_test, y_train[target_col], y_test[target_col], target_col
        
    except Exception as e:
        st.error(f"Error loading splits: {str(e)}")
        return None, None, None, None, None

def load_available_models():
    """Load information about available models."""
    models_dir = project_root / "models"
    available_models = {}
    
    if not models_dir.exists():
        return available_models
    
    for model_file in models_dir.glob("*.pkl"):
        model_name = model_file.stem
        metadata_file = models_dir / f"{model_name}_metadata.json"
        
        model_info = {
            "file": str(model_file),
            "name": model_name
        }
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                model_info.update(metadata)
            except:
                pass
        
        available_models[model_name] = model_info
    
    return available_models

# --------------------------------------------------
# ANALYSIS FUNCTIONS
# --------------------------------------------------

def dataset_overview(df, target):
    """Display dataset overview."""
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", df.shape[1] - 1)
    with col3:
        st.metric("Target Classes", df[target].nunique())
    
    st.success("📝 Displaying complete dataset analysis")
    
    # Display first few rows
    st.subheader("Data Sample")
    st.dataframe(df.head(10))
    
    # Target distribution
    st.subheader("Target Distribution")
    target_counts = df[target].value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=target_counts.index,
            y=target_counts.values,
            labels={'x': target, 'y': 'Count'},
            title=f"Distribution of {target}"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig_pie = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            title="Target Distribution (%)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Basic statistics
    st.subheader("Dataset Statistics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe())

def univariate_analysis(df, target):
    """Fixed univariate analysis using actual data columns."""
    st.subheader("📈 Univariate Analysis")
    
    # Get all features except target
    features = [col for col in df.columns if col != target]
    
    # Feature selection
    selected_feature = st.selectbox("Select Feature to Analyze", features)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Analyzing: {selected_feature}**")
        
        # Check if feature is numeric or categorical
        if df[selected_feature].dtype in ['object', 'category'] or df[selected_feature].nunique() < 20:
            # Categorical or low-cardinality
            value_counts = df[selected_feature].value_counts().head(15)
            fig = px.bar(
                x=value_counts.values,
                y=value_counts.index,
                orientation='h',
                title=f"Distribution of {selected_feature}",
                labels={'x': 'Count', 'y': selected_feature}
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
        else:
            # Continuous
            fig = px.histogram(
                df,
                x=selected_feature,
                nbins=30,
                title=f"Distribution of {selected_feature}"
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Summary Statistics:**")
        
        if df[selected_feature].dtype in ['int64', 'float64']:
            # Numeric feature
            stats = df[selected_feature].describe()
            for stat, value in stats.items():
                if isinstance(value, (int, float)):
                    st.write(f"- **{stat}**: {value:.2f}")
                else:
                    st.write(f"- **{stat}**: {value}")
        else:
            # Categorical feature
            st.write(f"- **Unique values**: {df[selected_feature].nunique()}")
            st.write(f"- **Most frequent**: {df[selected_feature].mode().iloc[0]}")
            st.write(f"- **Missing values**: {df[selected_feature].isnull().sum()}")
            
            # Show top categories
            st.write("**Top categories:**")
            top_cats = df[selected_feature].value_counts().head(5)
            for cat, count in top_cats.items():
                pct = (count / len(df)) * 100
                st.write(f"  - {cat}: {count} ({pct:.1f}%)")

def bivariate_analysis(df, target):
    """Fixed bivariate analysis using actual data columns."""
    st.subheader("🔗 Bivariate Analysis")
    
    features = [col for col in df.columns if col != target]
    selected_feature = st.selectbox("Select Feature vs Target", features, key="bivariate_select")
    
    st.write(f"**Analyzing: {selected_feature} vs {target}**")
    
    # Create visualization based on feature type
    if df[selected_feature].dtype in ['object', 'category'] or df[selected_feature].nunique() < 20:
        # Categorical feature
        st.subheader("Categorical vs Target Analysis")
        
        # Cross-tabulation
        crosstab = pd.crosstab(df[selected_feature], df[target])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Stacked bar chart
            fig = px.bar(
                crosstab.reset_index().melt(id_vars=selected_feature),
                x=selected_feature,
                y='value',
                color='variable',
                title=f"{selected_feature} vs {target}",
                barmode='stack',
                labels={'value': 'Count', 'variable': target}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Percentage breakdown
            crosstab_pct = pd.crosstab(df[selected_feature], df[target], normalize='index') * 100
            fig_pct = px.bar(
                crosstab_pct.reset_index().melt(id_vars=selected_feature),
                x=selected_feature,
                y='value',
                color='variable',
                title=f"{selected_feature} vs {target} (%)",
                barmode='stack',
                labels={'value': 'Percentage', 'variable': target}
            )
            fig_pct.update_xaxes(tickangle=45)
            st.plotly_chart(fig_pct, use_container_width=True)
        
        # Show crosstab table
        st.write("**Cross-tabulation:**")
        st.dataframe(crosstab)
        
        # Show percentage table
        st.write("**Percentage breakdown:**")
        st.dataframe(crosstab_pct.round(1))
        
    else:
        # Continuous feature
        st.subheader("Continuous vs Target Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot
            fig = px.box(
                df,
                x=target,
                y=selected_feature,
                title=f"{selected_feature} by {target}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Violin plot for more detail
            fig_violin = px.violin(
                df,
                x=target,
                y=selected_feature,
                title=f"{selected_feature} Distribution by {target}"
            )
            st.plotly_chart(fig_violin, use_container_width=True)
        
        # Statistics by target
        st.write("**Statistics by Target:**")
        stats_by_target = df.groupby(target)[selected_feature].describe().round(2)
        st.dataframe(stats_by_target)

def model_training_section(df, target):
    """Fixed model training using actual data."""
    st.subheader("🤖 Model Training")
    
    st.info("🔄 Training models on the complete dataset.")
    
    # Option to use pre-split data or create new splits
    use_presplit = st.checkbox("Use pre-processed train/test splits (recommended)", value=True)
    
    if use_presplit:
        # Try to load pre-split data
        splits_data = load_full_splits()
        if splits_data[0] is not None:
            X_train, X_test, y_train, y_test, target_col = splits_data
            st.success(f"✅ Using pre-processed splits - Train: {len(X_train):,}, Test: {len(X_test):,}")
            
            # Model selection
            model_options = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=15),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1)
            }
            
            selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
            
            if st.button("Train Model on Full Data", type="primary"):
                try:
                    with st.spinner(f"Training {selected_model_name} on {len(X_train):,} samples..."):
                        # Train model
                        model = model_options[selected_model_name]
                        model.fit(X_train, y_train)
                        
                        # Evaluate
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Display results
                        st.success(f"✅ Model trained successfully on full dataset!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Test Accuracy", f"{accuracy:.4f}")
                        with col2:
                            st.metric("Training Samples", f"{len(X_train):,}")
                        with col3:
                            st.metric("Test Samples", f"{len(X_test):,}")
                        
                        # Save model
                        models_dir = project_root / "models"
                        models_dir.mkdir(exist_ok=True)
                        
                        model_filename = f"{selected_model_name.replace(' ', '_')}_full_data.pkl"
                        model_path = models_dir / model_filename
                        
                        joblib.dump(model, model_path)
                        st.success(f"💾 Model saved as: {model_filename}")
                        
                        # Show confusion matrix
                        cm = confusion_matrix(y_test, y_pred)
                        
                        fig = px.imshow(
                            cm,
                            text_auto=True,
                            title="Confusion Matrix",
                            labels=dict(x="Predicted", y="Actual"),
                            color_continuous_scale="Blues"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Classification report
                        st.subheader("📊 Classification Report")
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(3))
                        
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
            return
    
    # Fallback to using the loaded dataframe
    if df is None or len(df) == 0:
        st.error("❌ No data available for training")
        return
    
    st.warning("⚠️ Using loaded dataframe for training (may be slower than pre-processed splits)")
    
    # Prepare data
    X = df.drop(columns=[target])
    y = df[target]
    
    # Handle categorical variables by encoding them
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.write(f"**Encoding {len(categorical_cols)} categorical features...**")
        for col in categorical_cols:
            X[col] = pd.Categorical(X[col]).codes
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        st.write("**Handling missing values...**")
        X = X.fillna(X.mean())
    
    # Model selection
    model_options = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    }
    
    selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    
    with col2:
        random_state = st.number_input("Random State", value=42, min_value=0)
    
    if st.button("Train Model", type="primary"):
        try:
            with st.spinner("Training model..."):
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                
                # Train model
                model = model_options[selected_model_name]
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Display results
                st.success(f"✅ Model trained successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test Accuracy", f"{accuracy:.4f}")
                with col2:
                    st.metric("Training Samples", f"{len(X_train):,}")
                with col3:
                    st.metric("Test Samples", f"{len(X_test):,}")
                
                # Save model
                models_dir = project_root / "models"
                models_dir.mkdir(exist_ok=True)
                
                model_filename = f"{selected_model_name.replace(' ', '_')}_demo.pkl"
                model_path = models_dir / model_filename
                
                joblib.dump(model, model_path)
                st.success(f"💾 Model saved as: {model_filename}")
                
                # Show confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    title="Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual"),
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Classification report
                st.subheader("📊 Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(3))
                
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")

def model_evaluation_section(df, target):
    """Fixed model evaluation using available models."""
    st.subheader("📊 Model Evaluation")
    
    available_models = load_available_models()
    
    if not available_models:
        st.warning("⚠️ No models found. Please train a model first or run the model pipeline.")
        return
    
    # Model selection
    model_names = list(available_models.keys())
    selected_model = st.selectbox("Select Model", model_names)
    
    model_info = available_models[selected_model]
    
    # Display model metadata
    if "performance_metrics" in model_info:
        st.subheader("📈 Model Performance Metrics")
        metrics = model_info["performance_metrics"]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics.get('Accuracy', 0):.4f}")
        with col2:
            st.metric("Precision", f"{metrics.get('Precision', 0):.4f}")
        with col3:
            st.metric("Recall", f"{metrics.get('Recall', 0):.4f}")
        with col4:
            st.metric("F1-Score", f"{metrics.get('F1_Score', 0):.4f}")
        
        # Additional metrics if available
        if "Matthews_Corr" in metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Matthews Correlation", f"{metrics['Matthews_Corr']:.4f}")
            with col2:
                st.metric("Cohen Kappa", f"{metrics.get('Cohen_Kappa', 0):.4f}")
    
    # Load and evaluate model
    try:
        model = joblib.load(model_info["file"])
        
        st.subheader("🔍 Model Analysis")
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_') and df is not None:
            st.subheader("🔍 Feature Importance")
            
            feature_names = df.drop(columns=[target]).columns
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Most Important Features',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Model parameters
        if hasattr(model, 'get_params'):
            st.subheader("⚙️ Model Parameters")
            params = model.get_params()
            
            # Display key parameters
            key_params = {}
            for param, value in params.items():
                if not callable(value) and str(value) != 'None':
                    key_params[param] = value
            
            if key_params:
                st.json(key_params)
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")

def prediction_interface(df, target):
    """Fixed real-time prediction using actual data."""
    st.subheader("🚨 Real-Time Prediction")
    
    available_models = load_available_models()
    
    if not available_models:
        st.warning("⚠️ No models found. Please train a model first.")
        return
    
    # Model selection
    model_names = list(available_models.keys())
    selected_model = st.selectbox("Choose Model for Prediction", model_names, key="pred_model_select")
    
    try:
        model = joblib.load(available_models[selected_model]["file"])
        
        st.subheader("🔧 Input Features")
        st.write("**Enter values for prediction:**")
        
        # Get feature columns from the data
        feature_columns = [col for col in df.columns if col != target]
        input_data = {}
        
        # Create input form with actual data ranges
        num_cols = 3
        cols = st.columns(num_cols)
        
        for i, feature in enumerate(feature_columns):
            col_idx = i % num_cols
            
            with cols[col_idx]:
                if feature in df.columns:
                    if df[feature].dtype in ['object', 'category'] or df[feature].nunique() < 20:
                        # Categorical feature
                        unique_vals = sorted(df[feature].dropna().unique())
                        if len(unique_vals) > 0:
                            input_data[feature] = st.selectbox(
                                f"**{feature}**", 
                                unique_vals,
                                key=f"pred_{feature}"
                            )
                        else:
                            input_data[feature] = 0
                    else:
                        # Numerical feature
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        mean_val = float(df[feature].mean())
                        
                        input_data[feature] = st.number_input(
                            f"**{feature}**",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            key=f"pred_{feature}",
                            format="%.2f"
                        )
                else:
                    # Default value for missing features
                    input_data[feature] = st.number_input(
                        f"**{feature}**",
                        value=0.0,
                        key=f"pred_{feature}"
                    )
        
        # Prediction button
        if st.button("🔮 Make Prediction", type="primary"):
            try:
                # Create input dataframe
                input_df = pd.DataFrame([input_data])
                
                # Handle categorical encoding (same as training)
                categorical_cols = input_df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if col in df.columns:
                        # Use the same encoding as in the original data
                        unique_vals = df[col].dropna().unique()
                        if input_df[col].iloc[0] in unique_vals:
                            input_df[col] = pd.Categorical(input_df[col], categories=unique_vals).codes
                        else:
                            input_df[col] = 0
                
                # Handle missing values
                input_df = input_df.fillna(0)
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # Display result
                st.success(f"🚦 **Predicted Accident Severity: {prediction}**")
                
                # Show probabilities if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_df)[0]
                    classes = model.classes_ if hasattr(model, 'classes_') else range(len(probabilities))
                    
                    prob_df = pd.DataFrame({
                        'Severity Level': classes,
                        'Probability': probabilities
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x='Severity Level',
                        y='Probability',
                        title='Prediction Confidence',
                        color='Probability',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show confidence level
                    max_prob = probabilities.max()
                    confidence = "High" if max_prob > 0.7 else "Medium" if max_prob > 0.5 else "Low"
                    st.info(f"**Prediction Confidence**: {confidence} ({max_prob:.1%})")
                    
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.write("**Debug info:**")
                st.write(f"Input data shape: {input_df.shape}")
                st.write(f"Input data types: {input_df.dtypes.to_dict()}")
                
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
def main():
    # Load data
    df, target, data_loaded = load_data()
    
    if not data_loaded:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    sections = [
        "📊 Dataset Overview",
        "📈 Univariate Analysis", 
        "🔗 Bivariate Analysis",
        "🤖 Model Training",
        "📊 Model Evaluation",
        "🚨 Real-Time Prediction"
    ]
    
    selected_section = st.sidebar.radio("Select Section", sections)
    
    # Display selected section
    if selected_section == "📊 Dataset Overview":
        dataset_overview(df, target)
        
    elif selected_section == "📈 Univariate Analysis":
        univariate_analysis(df, target)
        
    elif selected_section == "🔗 Bivariate Analysis":
        bivariate_analysis(df, target)
        
    elif selected_section == "🤖 Model Training":
        model_training_section(df, target)
        
    elif selected_section == "📊 Model Evaluation":
        model_evaluation_section(df, target)
        
    elif selected_section == "🚨 Real-Time Prediction":
        prediction_interface(df, target)
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Data Info")
    if df is not None:
        st.sidebar.info(
            f"**Dataset**: {len(df):,} records\n\n"
            f"**Features**: {df.shape[1]-1}\n\n"
            f"**Target**: {target}\n\n"
            f"**Classes**: {df[target].nunique()}"
        )
    
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.info(
        "• **Dataset Overview**: Explore your data structure\n\n"
        "• **Univariate**: Analyze individual features\n\n"
        "• **Bivariate**: Explore feature relationships\n\n"
        "• **Model Training**: Train ML models\n\n"
        "• **Model Evaluation**: Assess model performance\n\n"
        "• **Real-Time Prediction**: Make predictions"
    )

if __name__ == "__main__":
    main()