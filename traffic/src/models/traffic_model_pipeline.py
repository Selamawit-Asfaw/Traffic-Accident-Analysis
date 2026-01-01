import os
import json
import time
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    log_loss
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')


class TrafficModelPipeline:
    """
    Enhanced Modeling Pipeline for Traffic Accident Analysis
    
    Features:
    - Multiple ML algorithms with hyperparameter tuning
    - Cross-validation and advanced metrics
    - Comprehensive model comparison and visualization
    - Feature importance analysis
    - Model interpretability tools
    - Automated model selection and saving
    """

    def __init__(self, data_path: str, target: str = "most_severe_injury"):
        self.data_path = data_path
        self.target = target
        self.df = None
        self.models = {}
        self.tuned_models = {}
        self.results = []
        self.cv_results = []
        self.feature_importance_data = {}
        self.training_log = []

    # --------------------------------------------------
    # 1. Enhanced Data Loading with Validation
    # --------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """Load processed data with validation"""
        try:
            self.df = pd.read_csv(self.data_path)
            
            print("✅ Processed data loaded successfully.")
            print(f"📊 Dataset shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            
            # Validate target column
            if self.target not in self.df.columns:
                raise ValueError(f"Target column '{self.target}' not found in dataset")
            
            # Check for missing values
            missing_values = self.df.isnull().sum().sum()
            if missing_values > 0:
                print(f"⚠️ Warning: {missing_values:,} missing values detected")
            
            # Display target distribution
            target_dist = self.df[self.target].value_counts().sort_index()
            print(f"\n🎯 Target Distribution ({self.target}):")
            for class_val, count in target_dist.items():
                pct = (count / len(self.df)) * 100
                print(f"   Class {class_val}: {count:,} ({pct:.1f}%)")
            
            return self.df
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None

    # --------------------------------------------------
    # 2. Enhanced Train-Test Split
    # --------------------------------------------------
    def split_data(self, test_size: float = 0.2, validation_size: float = 0.1):
        """Enhanced train-test-validation split with stratification"""
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        # Initial train-test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create validation set if requested
        if validation_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=validation_size, random_state=42, stratify=y_temp
            )
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            return X_temp, X_test, y_temp, y_test

    # --------------------------------------------------
    # 2b. Enhanced Load Train/Test Splits from Prepared CSV Files
    # --------------------------------------------------
    def load_splits_from_dir(self, splits_dir: str, include_validation: bool = True):
        """
        Load pre-saved train/test/validation splits with enhanced validation
        """
        print(f"\n🔄 Loading splits from: {splits_dir}")
        
        # Required files
        required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
        optional_files = ["X_val.csv", "y_val.csv"] if include_validation else []
        
        # Check required files
        for filename in required_files:
            filepath = os.path.join(splits_dir, filename)
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"Required split file not found: {filepath}")
        
        # Load required splits
        X_train = pd.read_csv(os.path.join(splits_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(splits_dir, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(splits_dir, "y_train.csv"))[self.target]
        y_test = pd.read_csv(os.path.join(splits_dir, "y_test.csv"))[self.target]
        
        # Load validation splits if available
        X_val, y_val = None, None
        if include_validation:
            val_x_path = os.path.join(splits_dir, "X_val.csv")
            val_y_path = os.path.join(splits_dir, "y_val.csv")
            
            if os.path.isfile(val_x_path) and os.path.isfile(val_y_path):
                X_val = pd.read_csv(val_x_path)
                y_val = pd.read_csv(val_y_path)[self.target]
                print(f"✅ Validation set loaded: {X_val.shape[0]:,} samples")
        
        # Display split information
        print(f"📊 Split Information:")
        print(f"   Training: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
        print(f"   Testing: {X_test.shape[0]:,} samples × {X_test.shape[1]} features")
        if X_val is not None:
            print(f"   Validation: {X_val.shape[0]:,} samples × {X_val.shape[1]} features")
        
        # Validate feature consistency
        if not X_train.columns.equals(X_test.columns):
            raise ValueError("Feature columns mismatch between train and test sets")
        
        if X_val is not None:
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            return X_train, X_test, y_train, y_test

    # --------------------------------------------------
    # 3. Enhanced Model Definition with Multiple Algorithms
    # --------------------------------------------------
    def define_models(self, include_advanced: bool = True):
        """
        Define core set of 3 ML models for comparison:
        - Logistic Regression: Linear probabilistic classifier
        - Decision Tree: Rule-based interpretable model
        - Random Forest: Ensemble of decision trees
        
        Parameters:
        include_advanced: Kept for compatibility but not used (only core models included)
        """
        print("\n🤖 Defining core machine learning models...")
        
        # Core 3 models only
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42,
                class_weight='balanced',
                max_depth=10
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        }
        
        print(f"✅ Defined {len(self.models)} core models:")
        for name in self.models.keys():
            print(f"   • {name}")
        
        return self.models

    # --------------------------------------------------
    # 4. Enhanced Cross-Validation Analysis
    # --------------------------------------------------
    def perform_cross_validation(self, X_train, y_train, cv_folds: int = 5):
        """
        Perform comprehensive cross-validation analysis
        """
        print(f"\n🔄 Performing {cv_folds}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"   Evaluating {name}...")
            
            # Multiple scoring metrics
            scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            cv_scores = {}
            
            for metric in scoring_metrics:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric, n_jobs=-1)
                cv_scores[metric] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
            
            self.cv_results.append({
                'Model': name,
                'CV_Accuracy_Mean': cv_scores['accuracy']['mean'],
                'CV_Accuracy_Std': cv_scores['accuracy']['std'],
                'CV_Precision_Mean': cv_scores['precision_weighted']['mean'],
                'CV_Precision_Std': cv_scores['precision_weighted']['std'],
                'CV_Recall_Mean': cv_scores['recall_weighted']['mean'],
                'CV_Recall_Std': cv_scores['recall_weighted']['std'],
                'CV_F1_Mean': cv_scores['f1_weighted']['mean'],
                'CV_F1_Std': cv_scores['f1_weighted']['std']
            })
        
        cv_df = pd.DataFrame(self.cv_results)
        
        # Visualize CV results
        self._plot_cv_results(cv_df)
        
        return cv_df
    
    def _plot_cv_results(self, cv_df):
        """Plot cross-validation results"""
        
        # CV scores with error bars
        metrics = ['CV_Accuracy_Mean', 'CV_Precision_Mean', 'CV_Recall_Mean', 'CV_F1_Mean']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metric_names,
            vertical_spacing=0.12
        )
        
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            row, col = positions[i]
            std_metric = metric.replace('_Mean', '_Std')
            
            fig.add_trace(
                go.Bar(
                    x=cv_df['Model'],
                    y=cv_df[metric],
                    error_y=dict(type='data', array=cv_df[std_metric]),
                    name=name,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Cross-Validation Results (Mean ± Std)",
            height=600
        )
        fig.show()

    # --------------------------------------------------
    # 5. Enhanced Training & Evaluation with Advanced Metrics
    # --------------------------------------------------
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, X_val=None, y_val=None):
        """
        Enhanced training and evaluation with comprehensive metrics and visualizations
        """
        print(f"\n🚀 Training and evaluating {len(self.models)} models...")
        print("="*60)
        
        # Get unique classes for multi-class analysis
        classes = sorted(y_train.unique())
        n_classes = len(classes)
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            start_time = time.time()
            
            # Train model
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            
            # Validation predictions if available
            val_metrics = {}
            if X_val is not None and y_val is not None:
                y_val_pred = model.predict(X_val)
                val_metrics = {
                    "Val_Accuracy": accuracy_score(y_val, y_val_pred),
                    "Val_Precision": precision_score(y_val, y_val_pred, average="weighted", zero_division=0),
                    "Val_Recall": recall_score(y_val, y_val_pred, average="weighted", zero_division=0),
                    "Val_F1": f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
                }
            
            # Comprehensive metrics
            metrics = {
                "Model": name,
                "Training_Time": training_time,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Balanced_Accuracy": balanced_accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "F1_Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "Matthews_Corr": matthews_corrcoef(y_test, y_pred),
                "Cohen_Kappa": cohen_kappa_score(y_test, y_pred)
            }
            
            # Add validation metrics
            metrics.update(val_metrics)
            
            # Add log loss if probabilities available
            if y_proba is not None:
                try:
                    metrics["Log_Loss"] = log_loss(y_test, y_proba)
                except:
                    metrics["Log_Loss"] = np.nan
            
            self.results.append(metrics)
            
            # Log training details
            self.training_log.append({
                'model': name,
                'timestamp': datetime.now().isoformat(),
                'training_time': training_time,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': X_train.shape[1],
                'classes': n_classes
            })
            
            print(f"   ✅ Completed in {training_time:.2f}s")
            print(f"   📊 Test Accuracy: {metrics['Accuracy']:.4f}")
            print(f"   📊 Test F1-Score: {metrics['F1_Score']:.4f}")
            
            # Generate visualizations
            self._generate_model_visualizations(name, model, X_train, X_test, y_test, y_pred, y_proba, classes)
        
        results_df = pd.DataFrame(self.results)
        return results_df
    
    def _generate_model_visualizations(self, name, model, X_train, X_test, y_test, y_pred, y_proba, classes):
        """Generate comprehensive visualizations for each model"""
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        
        # Calculate percentages for better interpretation
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations combining counts and percentages
        annotations = []
        for i in range(len(classes)):
            for j in range(len(classes)):
                annotations.append(f"{cm[i,j]}<br>({cm_percent[i,j]:.1f}%)")
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f"Predicted {c}" for c in classes],
            y=[f"Actual {c}" for c in classes],
            text=np.array(annotations).reshape(cm.shape),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale='Blues',
            showscale=True
        ))
        
        fig_cm.update_layout(
            title=f"Confusion Matrix - {name}",
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            width=600,
            height=500
        )
        fig_cm.show()
        
        # 2. ROC Curves (Multi-class)
        if y_proba is not None:
            self._plot_roc_curves(name, y_test, y_proba, classes)
        
        # 3. Feature Importance (for applicable models)
        if hasattr(model, "feature_importances_"):
            self._plot_feature_importance(name, model, X_train.columns)
        elif hasattr(model, "coef_") and len(model.coef_.shape) == 2:
            self._plot_feature_coefficients(name, model, X_train.columns, classes)
    
    def _plot_roc_curves(self, name, y_test, y_proba, classes):
        """Plot ROC curves for multi-class classification"""
        
        n_classes = len(classes)
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(width=2)
            ))
            
        else:
            # Multi-class classification
            y_test_bin = label_binarize(y_test, classes=classes)
            
            fig = go.Figure()
            
            # Compute ROC curve for each class
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'Class {cls} (AUC = {roc_auc:.3f})',
                    line=dict(width=2)
                ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray', width=1)
        ))
        
        fig.update_layout(
            title=f'ROC Curves - {name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700,
            height=500,
            hovermode='closest'
        )
        fig.show()
    
    def _plot_feature_importance(self, name, model, feature_names):
        """Plot feature importance for tree-based models"""
        
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(20)
        
        # Store for later analysis
        self.feature_importance_data[name] = importance_df
        
        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top 20 Feature Importance - {name}",
            color="Importance",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
        fig.show()
    
    def _plot_feature_coefficients(self, name, model, feature_names, classes):
        """Plot feature coefficients for linear models"""
        
        if len(classes) == 2:
            # Binary classification
            coef_df = pd.DataFrame({
                "Feature": feature_names,
                "Coefficient": model.coef_[0]
            }).sort_values("Coefficient", key=abs, ascending=False).head(20)
            
            fig = px.bar(
                coef_df,
                x="Coefficient",
                y="Feature",
                orientation="h",
                title=f"Top 20 Feature Coefficients - {name}",
                color="Coefficient",
                color_continuous_scale="RdBu"
            )
            
        else:
            # Multi-class classification - show coefficients for each class
            fig = make_subplots(
                rows=1, cols=min(len(classes), 3),
                subplot_titles=[f"Class {cls}" for cls in classes[:3]]
            )
            
            for i, cls in enumerate(classes[:3]):
                coef_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Coefficient": model.coef_[i]
                }).sort_values("Coefficient", key=abs, ascending=False).head(15)
                
                fig.add_trace(
                    go.Bar(
                        x=coef_df["Coefficient"],
                        y=coef_df["Feature"],
                        orientation="h",
                        name=f"Class {cls}",
                        showlegend=False
                    ),
                    row=1, col=i+1
                )
            
            fig.update_layout(title=f"Feature Coefficients by Class - {name}", height=500)
        
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        fig.show()

    # --------------------------------------------------
    # 6. Enhanced Model Comparison with Advanced Visualizations
    # --------------------------------------------------
    def compare_models(self, results_df):
        """
        Comprehensive model comparison with multiple visualization types
        """
        print(f"\n📊 Comparing {len(results_df)} models...")
        
        # 1. Performance metrics bar chart
        metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1_Score", "Balanced_Accuracy"]
        available_metrics = [m for m in metrics_to_plot if m in results_df.columns]
        
        fig_bar = px.bar(
            results_df,
            x="Model",
            y=available_metrics,
            barmode="group",
            title="Model Performance Comparison - All Metrics",
            labels={"value": "Score", "variable": "Metric"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_bar.update_layout(height=500, xaxis_tickangle=-45)
        fig_bar.show()

        # 2. Radar chart for comprehensive comparison
        fig_radar = go.Figure()
        
        for idx, row in results_df.iterrows():
            values = [row[m] for m in available_metrics if m in row]
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=available_metrics,
                fill='toself',
                name=row["Model"],
                opacity=0.7
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart",
            height=600
        )
        fig_radar.show()

        # 3. Performance heatmap
        metrics_df = results_df.set_index("Model")[available_metrics]
        
        fig_heat = px.imshow(
            metrics_df.T,  # Transpose for better visualization
            text_auto=".3f",
            title="Model Performance Heatmap",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            labels={"x": "Model", "y": "Metric", "color": "Score"}
        )
        fig_heat.update_layout(height=400)
        fig_heat.show()

        # 4. Training time vs Performance scatter plot
        if "Training_Time" in results_df.columns:
            fig_scatter = px.scatter(
                results_df,
                x="Training_Time",
                y="F1_Score",
                size="Accuracy",
                color="Model",
                title="Training Time vs F1-Score (bubble size = Accuracy)",
                labels={"Training_Time": "Training Time (seconds)"},
                hover_data=["Accuracy", "Precision", "Recall"]
            )
            fig_scatter.update_layout(height=500)
            fig_scatter.show()

        # 5. Model ranking analysis
        self._create_model_ranking(results_df, available_metrics)
        
        return results_df
    
    def _create_model_ranking(self, results_df, metrics):
        """Create comprehensive model ranking analysis"""
        
        # Calculate ranks for each metric (higher is better)
        ranking_df = results_df.copy()
        
        for metric in metrics:
            if metric in ranking_df.columns:
                ranking_df[f"{metric}_Rank"] = ranking_df[metric].rank(ascending=False)
        
        # Calculate average rank
        rank_columns = [col for col in ranking_df.columns if col.endswith('_Rank')]
        ranking_df['Average_Rank'] = ranking_df[rank_columns].mean(axis=1)
        ranking_df['Overall_Rank'] = ranking_df['Average_Rank'].rank()
        
        # Sort by overall rank
        ranking_df = ranking_df.sort_values('Overall_Rank')
        
        # Display ranking table
        print(f"\n🏆 Model Ranking Analysis:")
        print("="*60)
        
        display_cols = ['Model', 'Overall_Rank', 'Average_Rank'] + metrics[:4]
        ranking_display = ranking_df[display_cols].round(4)
        
        for idx, row in ranking_display.iterrows():
            rank = int(row['Overall_Rank'])
            model = row['Model']
            avg_rank = row['Average_Rank']
            
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            print(f"{medal} {model} (Avg Rank: {avg_rank:.2f})")
        
        # Visualize ranking
        fig_rank = px.bar(
            ranking_df,
            x="Model",
            y="Average_Rank",
            title="Model Ranking (Lower is Better)",
            color="Average_Rank",
            color_continuous_scale="RdYlGn_r"
        )
        fig_rank.update_layout(height=400, xaxis_tickangle=-45)
        fig_rank.show()
        
        return ranking_df

    # --------------------------------------------------
    # 9. Enhanced Model Selection and Saving
    # --------------------------------------------------
    def get_best_model_name(self, results_df: pd.DataFrame, metric: str = "F1_Score") -> str:
        """
        Return the name of the best-performing model according to `metric`.
        """
        if metric not in results_df.columns:
            available_metrics = [col for col in results_df.columns if col not in ['Model', 'Training_Time']]
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {available_metrics}")

        best_idx = results_df[metric].idxmax()
        best_name = results_df.loc[best_idx, "Model"]
        return best_name

    def save_best_model(
        self,
        results_df: pd.DataFrame,
        model_dir: str = "models",
        metric: str = "F1_Score",
        include_tuned: bool = True
    ) -> str:
        """
        Enhanced model saving with metadata and model comparison
        """
        print(f"\n💾 Saving best model based on {metric}...")
        
        # Determine which models to consider
        models_to_consider = {}
        
        # Add original models
        for name, model in self.models.items():
            if any(results_df['Model'] == name):
                models_to_consider[name] = model
        
        # Add tuned models if available and requested
        if include_tuned and self.tuned_models:
            for name, model in self.tuned_models.items():
                tuned_name = f"{name} (Tuned)"
                if any(results_df['Model'] == tuned_name):
                    models_to_consider[tuned_name] = model
        
        # Find best model
        best_name = self.get_best_model_name(results_df, metric=metric)
        
        if best_name not in models_to_consider:
            # Try to find the base model name
            base_name = best_name.replace(" (Tuned)", "")
            if base_name in self.tuned_models:
                best_model = self.tuned_models[base_name]
            elif base_name in self.models:
                best_model = self.models[base_name]
            else:
                raise ValueError(f"Best model '{best_name}' not found in available models.")
        else:
            best_model = models_to_consider[best_name]

        # Create model directory
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        filename = best_name.replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
        filepath = os.path.join(model_dir, filename)
        joblib.dump(best_model, filepath)

        # Save model metadata
        best_row = results_df[results_df['Model'] == best_name].iloc[0]
        
        metadata = {
            'model_name': best_name,
            'selection_metric': metric,
            'selection_score': float(best_row[metric]),
            'model_type': type(best_model).__name__,
            'training_timestamp': datetime.now().isoformat(),
            'model_file': filename,
            'performance_metrics': {
                col: float(best_row[col]) for col in best_row.index 
                if col not in ['Model'] and pd.notna(best_row[col]) and isinstance(best_row[col], (int, float))
            },
            'feature_count': len(best_model.feature_names_in_) if hasattr(best_model, 'feature_names_in_') else 'unknown',
            'model_parameters': best_model.get_params() if hasattr(best_model, 'get_params') else {}
        }
        
        metadata_path = os.path.join(model_dir, filename.replace('.pkl', '_metadata.json'))
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Best model '{best_name}' saved to: {filepath}")
        print(f"📄 Metadata saved to: {metadata_path}")
        print(f"🎯 {metric}: {best_row[metric]:.4f}")

        return filepath

    # --------------------------------------------------
    # 10. Comprehensive Analysis Report
    # --------------------------------------------------
    def generate_analysis_report(self, results_df, cv_df=None, tuned_df=None):
        """
        Generate comprehensive analysis report
        """
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE MODEL ANALYSIS REPORT")
        print("="*80)
        
        # 1. Dataset Summary
        if self.df is not None:
            print(f"\n📊 Dataset Summary:")
            print(f"   Total samples: {len(self.df):,}")
            print(f"   Features: {self.df.shape[1] - 1}")  # Exclude target
            print(f"   Target classes: {self.df[self.target].nunique()}")
            
            class_dist = self.df[self.target].value_counts().sort_index()
            print(f"   Class distribution:")
            for class_val, count in class_dist.items():
                pct = (count / len(self.df)) * 100
                print(f"     Class {class_val}: {count:,} ({pct:.1f}%)")
        
        # 2. Model Performance Summary
        print(f"\n🤖 Model Performance Summary:")
        print(f"   Models evaluated: {len(results_df)}")
        
        # Best performers by metric
        key_metrics = ['Accuracy', 'F1_Score', 'Precision', 'Recall']
        available_metrics = [m for m in key_metrics if m in results_df.columns]
        
        for metric in available_metrics:
            best_model = results_df.loc[results_df[metric].idxmax(), 'Model']
            best_score = results_df[metric].max()
            print(f"   Best {metric}: {best_model} ({best_score:.4f})")
        
        # 3. Cross-Validation Results
        if cv_df is not None:
            print(f"\n🔄 Cross-Validation Summary:")
            cv_metric = 'CV_F1_Mean'
            if cv_metric in cv_df.columns:
                best_cv_model = cv_df.loc[cv_df[cv_metric].idxmax(), 'Model']
                best_cv_score = cv_df[cv_metric].max()
                print(f"   Best CV F1-Score: {best_cv_model} ({best_cv_score:.4f})")
        
        # 4. Hyperparameter Tuning Results
        if tuned_df is not None:
            print(f"\n🔧 Hyperparameter Tuning Summary:")
            print(f"   Models tuned: {len(tuned_df)}")
            if len(tuned_df) > 0:
                best_tuned = tuned_df.loc[tuned_df['Best_Score'].idxmax()]
                print(f"   Best tuned model: {best_tuned['Model']} ({best_tuned['Best_Score']:.4f})")
        
        # 5. Training Performance
        if self.training_log:
            print(f"\n⏱️ Training Performance:")
            total_time = sum(log['training_time'] for log in self.training_log)
            avg_time = total_time / len(self.training_log)
            print(f"   Total training time: {total_time:.2f}s")
            print(f"   Average training time: {avg_time:.2f}s")
            
            fastest_model = min(self.training_log, key=lambda x: x['training_time'])
            slowest_model = max(self.training_log, key=lambda x: x['training_time'])
            print(f"   Fastest model: {fastest_model['model']} ({fastest_model['training_time']:.2f}s)")
            print(f"   Slowest model: {slowest_model['model']} ({slowest_model['training_time']:.2f}s)")
        
        # 6. Feature Importance Summary
        if self.feature_importance_data:
            print(f"\n🔍 Feature Importance Summary:")
            print(f"   Models with feature importance: {len(self.feature_importance_data)}")
            
            # Find most important features across models
            all_features = {}
            for model_name, importance_df in self.feature_importance_data.items():
                for _, row in importance_df.head(10).iterrows():
                    feature = row['Feature']
                    importance = row['Importance']
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append(importance)
            
            # Calculate average importance
            avg_importance = {
                feature: np.mean(importances) 
                for feature, importances in all_features.items()
            }
            
            # Top 5 most important features
            top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   Top 5 features (average importance):")
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"     {i}. {feature}: {importance:.4f}")
        
        # 7. Recommendations
        print(f"\n💡 Recommendations:")
        
        # Performance-based recommendations
        if 'F1_Score' in results_df.columns:
            f1_scores = results_df['F1_Score']
            if f1_scores.max() - f1_scores.min() < 0.05:
                print("   • Model performances are similar - consider ensemble methods")
            
            if f1_scores.max() < 0.8:
                print("   • Consider feature engineering or data augmentation")
                print("   • Try advanced algorithms (XGBoost, LightGBM)")
            
            if 'Training_Time' in results_df.columns:
                # Find models with good performance/time trade-off
                results_df['Efficiency'] = results_df['F1_Score'] / results_df['Training_Time']
                efficient_model = results_df.loc[results_df['Efficiency'].idxmax(), 'Model']
                print(f"   • Most efficient model (performance/time): {efficient_model}")
        
        # Cross-validation recommendations
        if cv_df is not None and 'CV_F1_Std' in cv_df.columns:
            high_variance_models = cv_df[cv_df['CV_F1_Std'] > 0.05]['Model'].tolist()
            if high_variance_models:
                print(f"   • High variance models (consider regularization): {', '.join(high_variance_models)}")
        
        print(f"\n✅ Analysis complete! Use the best model for deployment.")
        
        return {
            'best_models': {metric: results_df.loc[results_df[metric].idxmax(), 'Model'] 
                          for metric in available_metrics},
            'performance_summary': results_df.describe(),
            'training_summary': self.training_log,
            'feature_importance': self.feature_importance_data
        }
