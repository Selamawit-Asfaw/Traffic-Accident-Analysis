import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class TrafficDataPrep:
    """
    Enhanced Data Cleaning & Preparation Pipeline
    for Traffic Accident Analysis Project
    """

    def __init__(self, data_path, target="most_severe_injury"):
        self.data_path = data_path
        self.target = target
        self.df = None
        self.original_df = None  # Keep original for comparison
        self.scaler = StandardScaler()
        self.encoders = {}
        self.preparation_log = []  # Track all transformations
        self.data_quality_metrics = {}  # Store quality metrics

    # --------------------------------------------------
    # 1. Enhanced Data Loading with Validation
    # --------------------------------------------------
    def load_data(self):
        """Load data with comprehensive validation and quality assessment"""
        try:
            self.df = pd.read_csv(self.data_path)
            self.original_df = self.df.copy()  # Keep original for comparison
            
            print("✅ Data loaded successfully.")
            print(f"📊 Dataset shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            
            # Initial data quality assessment
            self._assess_data_quality()
            
            # Log the loading step
            self.preparation_log.append({
                'step': 'data_loading',
                'action': 'loaded_data',
                'shape_before': None,
                'shape_after': self.df.shape,
                'details': f'Loaded {self.df.shape[0]:,} records'
            })
            
            return self.df
            
        except FileNotFoundError:
            print(f"❌ Error: File not found at {self.data_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None
    
    def _assess_data_quality(self):
        """Assess initial data quality metrics"""
        self.data_quality_metrics = {
            'total_records': len(self.df),
            'total_features': self.df.shape[1],
            'missing_values': self.df.isnull().sum().sum(),
            'missing_percentage': (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100,
            'duplicate_rows': self.df.duplicated().sum(),
            'numeric_features': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(self.df.select_dtypes(include=['object']).columns)
        }
        
        print(f"📈 Data Quality Summary:")
        print(f"   Missing values: {self.data_quality_metrics['missing_values']:,} ({self.data_quality_metrics['missing_percentage']:.1f}%)")
        print(f"   Duplicate rows: {self.data_quality_metrics['duplicate_rows']:,}")
        print(f"   Numeric features: {self.data_quality_metrics['numeric_features']}")
        print(f"   Categorical features: {self.data_quality_metrics['categorical_features']}")

    # --------------------------------------------------
    # 2. Enhanced Missing Value Handling
    # --------------------------------------------------
    def handle_missing_values(self, strategy='auto'):
        """
        Enhanced missing value handling with multiple strategies and validation
        
        Parameters:
        strategy: 'auto', 'mode_median', 'drop', 'forward_fill'
        """
        print("\n" + "="*60)
        print("🔍 MISSING VALUE HANDLING")
        print("="*60)
        
        shape_before = self.df.shape
        
        # Analyze missing patterns
        missing_before = self.df.isnull().sum().reset_index()
        missing_before.columns = ["Feature", "Missing_Count"]
        missing_before["Missing_Percent"] = (missing_before["Missing_Count"] / len(self.df)) * 100
        missing_before = missing_before[missing_before["Missing_Count"] > 0].sort_values("Missing_Count", ascending=False)

        if missing_before.empty:
            print("✅ No missing values found!")
            return self.df

        print(f"📊 Found missing values in {len(missing_before)} features:")
        for _, row in missing_before.head(10).iterrows():
            print(f"   • {row['Feature']}: {row['Missing_Count']:,} ({row['Missing_Percent']:.1f}%)")

        # Visualize missing values
        if len(missing_before) > 0:
            fig = px.bar(
                missing_before.head(15),
                x="Missing_Count",
                y="Feature",
                orientation="h",
                title="Missing Values by Feature (Before Cleaning)",
                color="Missing_Percent",
                color_continuous_scale="Reds"
            )
            fig.update_layout(height=max(400, len(missing_before.head(15)) * 25))
            fig.show()

        # Apply imputation strategy
        imputation_summary = []
        
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                missing_count = self.df[col].isnull().sum()
                
                if strategy == 'auto':
                    # Smart imputation based on data type and missing percentage
                    missing_pct = (missing_count / len(self.df)) * 100
                    
                    if missing_pct > 50:
                        # Too many missing values - consider dropping
                        print(f"⚠️  {col}: {missing_pct:.1f}% missing - consider dropping this feature")
                        continue
                    
                    if self.df[col].dtype == "object":
                        # Categorical: use mode
                        if not self.df[col].mode().empty:
                            fill_value = self.df[col].mode().iloc[0]
                            self.df[col].fillna(fill_value, inplace=True)
                            imputation_summary.append({
                                'feature': col,
                                'method': 'mode',
                                'fill_value': fill_value,
                                'missing_count': missing_count
                            })
                    else:
                        # Numeric: use median (more robust than mean)
                        fill_value = self.df[col].median()
                        self.df[col].fillna(fill_value, inplace=True)
                        imputation_summary.append({
                            'feature': col,
                            'method': 'median',
                            'fill_value': fill_value,
                            'missing_count': missing_count
                        })

        # Verify missing values after imputation
        missing_after = self.df.isnull().sum().sum()
        
        print(f"\n📈 Imputation Results:")
        print(f"   Features imputed: {len(imputation_summary)}")
        print(f"   Missing values before: {missing_before['Missing_Count'].sum():,}")
        print(f"   Missing values after: {missing_after:,}")
        
        if imputation_summary:
            print(f"\n🔧 Imputation Methods Used:")
            for item in imputation_summary[:10]:  # Show first 10
                print(f"   • {item['feature']}: {item['method']} = {item['fill_value']}")

        # Log the transformation
        self.preparation_log.append({
            'step': 'missing_values',
            'action': 'imputation',
            'shape_before': shape_before,
            'shape_after': self.df.shape,
            'details': f'Imputed {len(imputation_summary)} features, {missing_before["Missing_Count"].sum():,} → {missing_after:,} missing values'
        })

        print("✅ Missing value handling completed.")
        return self.df

    # --------------------------------------------------
    # 3. Enhanced Outlier Treatment
    # --------------------------------------------------
    def outlier_treatment(self, method='iqr', threshold=1.5):
        """
        Enhanced outlier treatment with multiple methods and impact analysis
        
        Parameters:
        method: 'iqr', 'zscore', 'robust', 'none'
        threshold: threshold for outlier detection
        """
        print("\n" + "="*60)
        print("📦 OUTLIER TREATMENT")
        print("="*60)
        
        shape_before = self.df.shape
        
        # Key numeric columns for outlier treatment
        numeric_cols = [
            "injuries_total", "injuries_incapacitating", 
            "injuries_non_incapacitating", "num_units"
        ]
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if not available_cols:
            print("❌ No numeric columns found for outlier treatment.")
            return self.df

        print(f"🔍 Analyzing outliers in {len(available_cols)} features using {method.upper()} method")
        
        outlier_summary = []
        
        for col in available_cols:
            print(f"\n📊 Processing '{col}':")
            
            original_values = self.df[col].copy()
            
            if method == 'iqr':
                # IQR method
                Q1 = original_values.quantile(0.25)
                Q3 = original_values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (original_values < lower_bound) | (original_values > upper_bound)
                
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs(stats.zscore(original_values))
                outlier_mask = z_scores > threshold
                lower_bound = original_values.mean() - threshold * original_values.std()
                upper_bound = original_values.mean() + threshold * original_values.std()
                
            elif method == 'robust':
                # Robust scaling method
                median = original_values.median()
                mad = np.median(np.abs(original_values - median))
                modified_z_scores = 0.6745 * (original_values - median) / mad
                outlier_mask = np.abs(modified_z_scores) > threshold
                lower_bound = median - threshold * mad / 0.6745
                upper_bound = median + threshold * mad / 0.6745
            
            outliers_count = outlier_mask.sum()
            outliers_pct = (outliers_count / len(self.df)) * 100
            
            print(f"   Original range: {original_values.min():.2f} to {original_values.max():.2f}")
            print(f"   Outliers detected: {outliers_count:,} ({outliers_pct:.1f}%)")
            
            if outliers_count > 0:
                # Apply capping instead of removal to preserve data
                self.df[col] = np.clip(original_values, lower_bound, upper_bound)
                
                capped_values = self.df[col]
                print(f"   After capping: {capped_values.min():.2f} to {capped_values.max():.2f}")
                
                outlier_summary.append({
                    'feature': col,
                    'outliers_detected': outliers_count,
                    'outliers_percent': outliers_pct,
                    'method': method,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                })

        # Create comprehensive visualization
        if len(available_cols) > 1:
            fig = make_subplots(
                rows=2, cols=len(available_cols),
                subplot_titles=[f"{col} - Before" for col in available_cols] + 
                              [f"{col} - After" for col in available_cols],
                vertical_spacing=0.15
            )
            
            for i, col in enumerate(available_cols):
                # Before treatment
                fig.add_trace(
                    go.Box(y=self.original_df[col], name=f"{col}_before", showlegend=False),
                    row=1, col=i+1
                )
                
                # After treatment
                fig.add_trace(
                    go.Box(y=self.df[col], name=f"{col}_after", showlegend=False),
                    row=2, col=i+1
                )
            
            fig.update_layout(
                title=f"Outlier Treatment Results ({method.upper()} method)",
                height=600
            )
            fig.show()

        # Summary
        if outlier_summary:
            total_outliers = sum(item['outliers_detected'] for item in outlier_summary)
            print(f"\n📈 Outlier Treatment Summary:")
            print(f"   Total outliers treated: {total_outliers:,}")
            print(f"   Features processed: {len(outlier_summary)}")
            print(f"   Method used: {method.upper()}")
            
            # Show feature-wise summary
            for item in outlier_summary:
                print(f"   • {item['feature']}: {item['outliers_detected']:,} outliers ({item['outliers_percent']:.1f}%)")

        # Log the transformation
        self.preparation_log.append({
            'step': 'outlier_treatment',
            'action': f'{method}_capping',
            'shape_before': shape_before,
            'shape_after': self.df.shape,
            'details': f'Treated {sum(item["outliers_detected"] for item in outlier_summary):,} outliers in {len(outlier_summary)} features'
        })

        print("✅ Outlier treatment completed.")
        return self.df

    # --------------------------------------------------
    # 4. Enhanced Data Transformation
    # --------------------------------------------------
    def data_transformation(self, auto_detect_skew=True, skew_threshold=1.0):
        """
        Enhanced data transformation with automatic skewness detection
        
        Parameters:
        auto_detect_skew: automatically detect skewed features
        skew_threshold: threshold for considering a feature skewed
        """
        print("\n" + "="*60)
        print("🔄 DATA TRANSFORMATION")
        print("="*60)
        
        shape_before = self.df.shape
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target in numeric_cols:
            numeric_cols.remove(self.target)  # Don't transform target
        
        if not numeric_cols:
            print("❌ No numeric columns found for transformation.")
            return self.df
        
        # Analyze skewness
        skewness_analysis = []
        for col in numeric_cols:
            if self.df[col].min() >= 0:  # Only for non-negative values (log transform requirement)
                skew_value = self.df[col].skew()
                skewness_analysis.append({
                    'feature': col,
                    'skewness': skew_value,
                    'needs_transform': abs(skew_value) > skew_threshold
                })
        
        if auto_detect_skew:
            transform_cols = [item['feature'] for item in skewness_analysis if item['needs_transform']]
        else:
            # Default columns if not auto-detecting
            transform_cols = [col for col in ["injuries_total", "num_units"] if col in numeric_cols]
        
        if not transform_cols:
            print("✅ No features require transformation (skewness within acceptable range).")
            return self.df
        
        print(f"🔍 Skewness Analysis:")
        for item in skewness_analysis:
            status = "🔄 TRANSFORM" if item['needs_transform'] else "✅ OK"
            print(f"   • {item['feature']}: {item['skewness']:.2f} {status}")
        
        print(f"\n🔄 Applying log transformation to {len(transform_cols)} features:")
        
        transformation_summary = []
        
        for col in transform_cols:
            skew_before = self.df[col].skew()
            
            # Apply log1p transformation (handles zeros)
            self.df[col] = np.log1p(self.df[col])
            skew_after = self.df[col].skew()
            
            transformation_summary.append({
                'feature': col,
                'skew_before': skew_before,
                'skew_after': skew_after,
                'improvement': abs(skew_before) - abs(skew_after)
            })
            
            print(f"   • {col}: {skew_before:.2f} → {skew_after:.2f} (improvement: {abs(skew_before) - abs(skew_after):.2f})")

        # Visualize transformations
        if len(transform_cols) > 0:
            # Create before/after comparison
            fig = make_subplots(
                rows=2, cols=min(len(transform_cols), 3),
                subplot_titles=[f"{col} - Original" for col in transform_cols[:3]] + 
                              [f"{col} - Transformed" for col in transform_cols[:3]],
                vertical_spacing=0.15
            )
            
            for i, col in enumerate(transform_cols[:3]):  # Show max 3 features
                # Original distribution (from backup)
                if col in self.original_df.columns:
                    fig.add_trace(
                        go.Histogram(x=self.original_df[col], name=f"{col}_original", showlegend=False, nbinsx=30),
                        row=1, col=i+1
                    )
                
                # Transformed distribution
                fig.add_trace(
                    go.Histogram(x=self.df[col], name=f"{col}_transformed", showlegend=False, nbinsx=30),
                    row=2, col=i+1
                )
            
            fig.update_layout(
                title="Data Transformation Results: Original vs Transformed Distributions",
                height=500
            )
            fig.show()

        # Log the transformation
        self.preparation_log.append({
            'step': 'data_transformation',
            'action': 'log_transform',
            'shape_before': shape_before,
            'shape_after': self.df.shape,
            'details': f'Log-transformed {len(transform_cols)} features: {", ".join(transform_cols)}'
        })

        print("✅ Data transformation completed.")
        return self.df

    def transform_datetime(self):
        """
        Convert crash_date to datetime features
        """
        if "crash_date" in self.df.columns:
            self.df["crash_date"] = pd.to_datetime(
                self.df["crash_date"],
                errors="coerce"
            )

            self.df["crash_year"] = self.df["crash_date"].dt.year
            self.df["crash_month"] = self.df["crash_date"].dt.month
            self.df["crash_day"] = self.df["crash_date"].dt.day
            self.df["crash_hour_from_date"] = self.df["crash_date"].dt.hour

            # Drop raw datetime column
            self.df.drop(columns=["crash_date"], inplace=True)

            print("Datetime features extracted and raw crash_date removed.")


    # --------------------------------------------------
    # 5. Enhanced Feature Engineering
    # --------------------------------------------------
    def feature_engineering(self, create_interactions=True):
        """
        Enhanced feature engineering with domain knowledge and interaction features
        
        Parameters:
        create_interactions: whether to create interaction features
        """
        print("\n" + "="*60)
        print("🔧 FEATURE ENGINEERING")
        print("="*60)
        
        shape_before = self.df.shape
        new_features = []
        
        # 1. Time-based features
        if "crash_hour" in self.df.columns:
            print("🕐 Creating time-based features:")
            
            # Night indicator (more specific than original)
            self.df["is_night"] = ((self.df["crash_hour"] >= 22) | (self.df["crash_hour"] <= 6)).astype(int)
            new_features.append("is_night")
            
            # Rush hour indicators
            self.df["is_morning_rush"] = ((self.df["crash_hour"] >= 7) & (self.df["crash_hour"] <= 9)).astype(int)
            self.df["is_evening_rush"] = ((self.df["crash_hour"] >= 17) & (self.df["crash_hour"] <= 19)).astype(int)
            new_features.extend(["is_morning_rush", "is_evening_rush"])
            
            # Time period categories
            def get_time_period(hour):
                if 6 <= hour <= 11:
                    return 0  # Morning
                elif 12 <= hour <= 17:
                    return 1  # Afternoon
                elif 18 <= hour <= 21:
                    return 2  # Evening
                else:
                    return 3  # Night
            
            self.df["time_period"] = self.df["crash_hour"].apply(get_time_period)
            new_features.append("time_period")
            
            print(f"   • Night accidents: {self.df['is_night'].sum():,} ({self.df['is_night'].mean()*100:.1f}%)")
            print(f"   • Morning rush: {self.df['is_morning_rush'].sum():,}")
            print(f"   • Evening rush: {self.df['is_evening_rush'].sum():,}")

        # 2. Weekend indicator
        if "crash_day_of_week" in self.df.columns:
            print("\n📅 Creating day-based features:")
            self.df["is_weekend"] = (self.df["crash_day_of_week"].isin([6, 7])).astype(int)
            new_features.append("is_weekend")
            print(f"   • Weekend accidents: {self.df['is_weekend'].sum():,} ({self.df['is_weekend'].mean()*100:.1f}%)")

        # 3. Injury-based features
        injury_cols = [col for col in ["injuries_fatal", "injuries_incapacitating", 
                      "injuries_non_incapacitating", "injuries_reported_not_evident"] 
                      if col in self.df.columns]
        
        if len(injury_cols) > 1 and "injuries_total" in self.df.columns:
            print("\n🏥 Creating injury-based features:")
            
            # Severe injury indicator
            severe_cols = [col for col in ["injuries_fatal", "injuries_incapacitating"] if col in self.df.columns]
            if severe_cols:
                self.df["has_severe_injury"] = (self.df[severe_cols].sum(axis=1) > 0).astype(int)
                new_features.append("has_severe_injury")
                print(f"   • Severe injury cases: {self.df['has_severe_injury'].sum():,}")
            
            # Injury rate (avoid division by zero)
            self.df["injury_rate"] = self.df[injury_cols].sum(axis=1) / (self.df["injuries_total"] + 1)
            new_features.append("injury_rate")

        # 4. Risk score (composite feature)
        risk_components = []
        if "is_night" in self.df.columns:
            risk_components.append("is_night")
        if "is_weekend" in self.df.columns:
            risk_components.append("is_weekend")
        
        # Add weather risk if available (assuming encoded weather)
        if "weather_condition" in self.df.columns:
            # Assume clear weather is the most common (encoded as mode)
            clear_weather_code = self.df["weather_condition"].mode().iloc[0] if not self.df["weather_condition"].mode().empty else 0
            self.df["bad_weather"] = (self.df["weather_condition"] != clear_weather_code).astype(int)
            risk_components.append("bad_weather")
            new_features.append("bad_weather")
        
        if risk_components:
            self.df["risk_score"] = self.df[risk_components].sum(axis=1)
            new_features.append("risk_score")
            print(f"\n🚨 Risk score distribution:")
            risk_dist = self.df["risk_score"].value_counts().sort_index()
            for score, count in risk_dist.items():
                print(f"   • Risk level {score}: {count:,} accidents ({count/len(self.df)*100:.1f}%)")

        # 5. Interaction features (if requested)
        if create_interactions and len(new_features) > 1:
            print(f"\n🔗 Creating interaction features:")
            
            # Time-weather interaction
            if "is_night" in new_features and "bad_weather" in new_features:
                self.df["night_bad_weather"] = self.df["is_night"] * self.df["bad_weather"]
                new_features.append("night_bad_weather")
                print(f"   • Night + bad weather: {self.df['night_bad_weather'].sum():,} cases")
            
            # Weekend-night interaction
            if "is_weekend" in new_features and "is_night" in new_features:
                self.df["weekend_night"] = self.df["is_weekend"] * self.df["is_night"]
                new_features.append("weekend_night")
                print(f"   • Weekend nights: {self.df['weekend_night'].sum():,} cases")

        # Visualize key new features
        if new_features:
            # Select top features for visualization
            viz_features = [f for f in ["is_night", "is_weekend", "risk_score", "has_severe_injury"] 
                           if f in new_features][:4]
            
            if viz_features:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=viz_features
                )
                
                positions = [(1,1), (1,2), (2,1), (2,2)]
                
                for i, feature in enumerate(viz_features):
                    row, col = positions[i]
                    
                    if self.df[feature].nunique() <= 10:  # Categorical-like
                        value_counts = self.df[feature].value_counts().sort_index()
                        fig.add_trace(
                            go.Bar(x=value_counts.index, y=value_counts.values, 
                                  name=feature, showlegend=False),
                            row=row, col=col
                        )
                    else:  # Continuous
                        fig.add_trace(
                            go.Histogram(x=self.df[feature], name=feature, showlegend=False),
                            row=row, col=col
                        )
                
                fig.update_layout(
                    title="New Engineered Features Distribution",
                    height=500
                )
                fig.show()

        # Log the transformation
        self.preparation_log.append({
            'step': 'feature_engineering',
            'action': 'create_features',
            'shape_before': shape_before,
            'shape_after': self.df.shape,
            'details': f'Created {len(new_features)} new features: {", ".join(new_features)}'
        })

        print(f"\n✅ Feature engineering completed. Created {len(new_features)} new features.")
        return self.df

    # --------------------------------------------------
    # 6. Enhanced Feature Encoding
    # --------------------------------------------------
    def encode_features(self, encoding_strategy='auto'):
        """
        Enhanced categorical feature encoding with multiple strategies
        
        Parameters:
        encoding_strategy: 'auto', 'label', 'onehot', 'target'
        """
        print("\n" + "="*60)
        print("🔤 FEATURE ENCODING")
        print("="*60)
        
        shape_before = self.df.shape
        
        categorical_cols = self.df.select_dtypes(include="object").columns.tolist()
        
        if not categorical_cols:
            print("✅ No categorical features found to encode.")
            return self.df
        
        print(f"🔍 Found {len(categorical_cols)} categorical features to encode:")
        
        encoding_summary = []
        
        for col in categorical_cols:
            unique_values = self.df[col].nunique()
            most_frequent = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else "Unknown"
            
            print(f"   • {col}: {unique_values} unique values (most frequent: '{most_frequent}')")
            
            # Decide encoding strategy
            if encoding_strategy == 'auto':
                if unique_values <= 10:
                    # Few categories - use label encoding
                    strategy = 'label'
                elif unique_values <= 50:
                    # Medium categories - use label encoding (one-hot would create too many features)
                    strategy = 'label'
                else:
                    # Many categories - use label encoding and warn
                    strategy = 'label'
                    print(f"     ⚠️ High cardinality ({unique_values} values) - consider feature selection")
            else:
                strategy = encoding_strategy
            
            # Apply encoding
            if strategy == 'label':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.encoders[col] = le
                
                encoding_summary.append({
                    'feature': col,
                    'method': 'label_encoding',
                    'unique_before': unique_values,
                    'unique_after': self.df[col].nunique()
                })
        
        # Verify encoding results
        print(f"\n📈 Encoding Results:")
        for item in encoding_summary:
            print(f"   • {item['feature']}: {item['method']} ({item['unique_before']} → {item['unique_after']} values)")
        
        # Show encoding distribution for a few features
        if len(categorical_cols) > 0:
            viz_cols = categorical_cols[:3]  # Show first 3
            
            fig = make_subplots(
                rows=1, cols=len(viz_cols),
                subplot_titles=[f"{col} (Encoded)" for col in viz_cols]
            )
            
            for i, col in enumerate(viz_cols, 1):
                value_counts = self.df[col].value_counts().sort_index()
                fig.add_trace(
                    go.Bar(x=value_counts.index, y=value_counts.values, 
                          name=col, showlegend=False),
                    row=1, col=i
                )
            
            fig.update_layout(
                title="Encoded Categorical Features Distribution",
                height=400
            )
            fig.show()

        # Log the transformation
        self.preparation_log.append({
            'step': 'feature_encoding',
            'action': 'label_encoding',
            'shape_before': shape_before,
            'shape_after': self.df.shape,
            'details': f'Encoded {len(categorical_cols)} categorical features'
        })

        print("✅ Feature encoding completed.")
        return self.df


    # --------------------------------------------------
    # 7. Enhanced Feature Scaling
    # --------------------------------------------------
    def scale_features(self, scaling_method='standard', exclude_binary=True):
        """
        Enhanced feature scaling with multiple methods and smart feature selection
        
        Parameters:
        scaling_method: 'standard', 'robust', 'minmax'
        exclude_binary: whether to exclude binary features from scaling
        """
        print("\n" + "="*60)
        print("⚖️ FEATURE SCALING")
        print("="*60)
        
        shape_before = self.df.shape
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target if it exists
        if self.target in numeric_cols:
            numeric_cols.remove(self.target)
        
        # Exclude binary features if requested
        if exclude_binary:
            binary_cols = []
            for col in numeric_cols:
                unique_vals = self.df[col].nunique()
                if unique_vals == 2 and set(self.df[col].unique()).issubset({0, 1}):
                    binary_cols.append(col)
            
            numeric_cols = [col for col in numeric_cols if col not in binary_cols]
            
            if binary_cols:
                print(f"🔍 Excluding {len(binary_cols)} binary features from scaling: {binary_cols}")

        if not numeric_cols:
            print("✅ No numeric features require scaling.")
            return self.df

        print(f"⚖️ Scaling {len(numeric_cols)} features using {scaling_method} method")
        
        # Store original values for comparison
        original_stats = {}
        for col in numeric_cols:
            original_stats[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max()
            }

        # Choose scaler
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        elif scaling_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
            print(f"⚠️ Unknown scaling method '{scaling_method}', using standard scaling")

        # Apply scaling
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        self.scaler = scaler  # Store for later use

        # Calculate new statistics
        scaled_stats = {}
        for col in numeric_cols:
            scaled_stats[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max()
            }

        # Show scaling results
        print(f"\n📊 Scaling Results (first 5 features):")
        for col in numeric_cols[:5]:
            orig = original_stats[col]
            scaled = scaled_stats[col]
            print(f"   • {col}:")
            print(f"     Before: mean={orig['mean']:.2f}, std={orig['std']:.2f}, range=[{orig['min']:.2f}, {orig['max']:.2f}]")
            print(f"     After:  mean={scaled['mean']:.2f}, std={scaled['std']:.2f}, range=[{scaled['min']:.2f}, {scaled['max']:.2f}]")

        # Create visualization comparing before/after scaling
        if len(numeric_cols) > 0:
            # Select features for visualization
            viz_cols = numeric_cols[:4]  # Show first 4 features
            
            fig = make_subplots(
                rows=2, cols=len(viz_cols),
                subplot_titles=[f"{col} - Original" for col in viz_cols] + 
                              [f"{col} - Scaled" for col in viz_cols],
                vertical_spacing=0.15
            )
            
            for i, col in enumerate(viz_cols, 1):
                # Original distribution (from backup)
                if col in self.original_df.columns:
                    fig.add_trace(
                        go.Histogram(x=self.original_df[col], name=f"{col}_original", 
                                   showlegend=False, nbinsx=30),
                        row=1, col=i
                    )
                
                # Scaled distribution
                fig.add_trace(
                    go.Histogram(x=self.df[col], name=f"{col}_scaled", 
                               showlegend=False, nbinsx=30),
                    row=2, col=i
                )
            
            fig.update_layout(
                title=f"Feature Scaling Results ({scaling_method.title()} Scaling)",
                height=500
            )
            fig.show()

        # Log the transformation
        self.preparation_log.append({
            'step': 'feature_scaling',
            'action': f'{scaling_method}_scaling',
            'shape_before': shape_before,
            'shape_after': self.df.shape,
            'details': f'Scaled {len(numeric_cols)} features using {scaling_method} method'
        })

        print(f"✅ Feature scaling completed using {scaling_method} method.")
        return self.df


    # --------------------------------------------------
    # 8. Enhanced Imbalance Handling
    # --------------------------------------------------
    def handle_imbalance(self, method='smote', sampling_strategy='auto'):
        """
        Enhanced class imbalance handling with multiple methods
        
        Parameters:
        method: 'smote', 'adasyn', 'borderline', 'none'
        sampling_strategy: 'auto', 'minority', 'all', or dict
        """
        print("\n" + "="*60)
        print("⚖️ CLASS IMBALANCE HANDLING")
        print("="*60)
        
        shape_before = self.df.shape
        
        # Analyze current class distribution
        target_counts_before = self.df[self.target].value_counts().sort_index()
        
        print("📊 Current class distribution:")
        for class_val, count in target_counts_before.items():
            pct = (count / len(self.df)) * 100
            print(f"   • {class_val}: {count:,} ({pct:.1f}%)")
        
        # Calculate imbalance ratio
        max_class = target_counts_before.max()
        min_class = target_counts_before.min()
        imbalance_ratio = max_class / min_class
        
        print(f"\n📈 Imbalance Analysis:")
        print(f"   • Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio < 2:
            print("   ✅ Classes are relatively balanced - no resampling needed")
            if method != 'none':
                print("   Skipping resampling...")
            return self.df
        elif imbalance_ratio < 5:
            print("   ⚠️ Moderate imbalance detected")
        else:
            print("   🚨 Severe imbalance detected - resampling recommended")

        if method == 'none':
            print("   Skipping resampling as requested.")
            return self.df

        # Visualize before
        fig_before = px.bar(
            x=target_counts_before.index.astype(str),
            y=target_counts_before.values,
            title="Class Distribution BEFORE Balancing",
            labels={'x': self.target, 'y': 'Count'},
            color=target_counts_before.values,
            color_continuous_scale='Reds'
        )
        fig_before.show()

        # Prepare data for resampling
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        # Choose resampling method
        try:
            if method == 'smote':
                from imblearn.over_sampling import SMOTE
                resampler = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
            elif method == 'adasyn':
                from imblearn.over_sampling import ADASYN
                resampler = ADASYN(random_state=42, sampling_strategy=sampling_strategy)
            elif method == 'borderline':
                from imblearn.over_sampling import BorderlineSMOTE
                resampler = BorderlineSMOTE(random_state=42, sampling_strategy=sampling_strategy)
            else:
                print(f"⚠️ Unknown method '{method}', using SMOTE")
                resampler = SMOTE(random_state=42, sampling_strategy=sampling_strategy)

            print(f"🔄 Applying {method.upper()} resampling...")
            
            # Apply resampling
            X_resampled, y_resampled = resampler.fit_resample(X, y)
            
            # Reconstruct dataframe
            self.df = pd.concat([X_resampled, y_resampled], axis=1)
            
            # Analyze results
            target_counts_after = self.df[self.target].value_counts().sort_index()
            
            print(f"\n📊 Class distribution after {method.upper()}:")
            for class_val, count in target_counts_after.items():
                pct = (count / len(self.df)) * 100
                before_count = target_counts_before.get(class_val, 0)
                change = count - before_count
                print(f"   • {class_val}: {count:,} ({pct:.1f}%) [+{change:,}]")
            
            # Visualize after
            fig_after = px.bar(
                x=target_counts_after.index.astype(str),
                y=target_counts_after.values,
                title=f"Class Distribution AFTER {method.upper()} Balancing",
                labels={'x': self.target, 'y': 'Count'},
                color=target_counts_after.values,
                color_continuous_scale='Blues'
            )
            fig_after.show()
            
            # Show improvement
            new_imbalance_ratio = target_counts_after.max() / target_counts_after.min()
            print(f"\n📈 Balancing Results:")
            print(f"   • Imbalance ratio: {imbalance_ratio:.1f}:1 → {new_imbalance_ratio:.1f}:1")
            print(f"   • Dataset size: {shape_before[0]:,} → {self.df.shape[0]:,} (+{self.df.shape[0] - shape_before[0]:,})")
            
        except Exception as e:
            print(f"❌ Error during resampling: {str(e)}")
            print("   Continuing without resampling...")
            return self.df

        # Log the transformation
        self.preparation_log.append({
            'step': 'imbalance_handling',
            'action': f'{method}_resampling',
            'shape_before': shape_before,
            'shape_after': self.df.shape,
            'details': f'Applied {method.upper()}: {shape_before[0]:,} → {self.df.shape[0]:,} samples'
        })

        print(f"✅ Class imbalance handling completed using {method.upper()}.")
        return self.df


    # --------------------------------------------------
    # 9. Data Quality Validation
    # --------------------------------------------------
    def validate_data_quality(self):
        """
        Comprehensive data quality validation after preprocessing
        """
        print("\n" + "="*60)
        print("✅ DATA QUALITY VALIDATION")
        print("="*60)
        
        validation_results = []
        
        # 1. Check for remaining missing values
        missing_values = self.df.isnull().sum().sum()
        validation_results.append({
            'check': 'Missing Values',
            'status': 'PASS' if missing_values == 0 else 'FAIL',
            'details': f'{missing_values:,} missing values found'
        })
        
        # 2. Check for infinite values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        inf_values = 0
        for col in numeric_cols:
            inf_count = np.isinf(self.df[col]).sum()
            inf_values += inf_count
        
        validation_results.append({
            'check': 'Infinite Values',
            'status': 'PASS' if inf_values == 0 else 'FAIL',
            'details': f'{inf_values:,} infinite values found'
        })
        
        # 3. Check data types consistency
        expected_numeric = len(numeric_cols)
        actual_numeric = len(self.df.select_dtypes(include=[np.number]).columns)
        
        validation_results.append({
            'check': 'Data Types',
            'status': 'PASS' if expected_numeric == actual_numeric else 'WARN',
            'details': f'{actual_numeric} numeric columns'
        })
        
        # 4. Check target variable
        if self.target in self.df.columns:
            target_unique = self.df[self.target].nunique()
            validation_results.append({
                'check': 'Target Variable',
                'status': 'PASS' if target_unique > 1 else 'FAIL',
                'details': f'{target_unique} unique classes'
            })
        
        # 5. Check for duplicate rows
        duplicates = self.df.duplicated().sum()
        validation_results.append({
            'check': 'Duplicate Rows',
            'status': 'PASS' if duplicates == 0 else 'WARN',
            'details': f'{duplicates:,} duplicate rows'
        })
        
        # Display results
        print("🔍 Validation Results:")
        for result in validation_results:
            status_icon = "✅" if result['status'] == 'PASS' else "⚠️" if result['status'] == 'WARN' else "❌"
            print(f"   {status_icon} {result['check']}: {result['details']}")
        
        # Overall assessment
        failed_checks = [r for r in validation_results if r['status'] == 'FAIL']
        warning_checks = [r for r in validation_results if r['status'] == 'WARN']
        
        if not failed_checks and not warning_checks:
            print("\n🎉 All data quality checks passed! Data is ready for modeling.")
        elif failed_checks:
            print(f"\n❌ {len(failed_checks)} critical issues found - review before modeling.")
        else:
            print(f"\n⚠️ {len(warning_checks)} warnings found - consider reviewing.")
        
        return validation_results

    # --------------------------------------------------
    # 10. Preparation Summary
    # --------------------------------------------------
    def generate_preparation_summary(self):
        """
        Generate comprehensive summary of all preparation steps
        """
        print("\n" + "="*60)
        print("📋 DATA PREPARATION SUMMARY")
        print("="*60)
        
        if not self.preparation_log:
            print("No preparation steps recorded.")
            return
        
        # Overall transformation
        initial_shape = self.preparation_log[0]['shape_after'] if self.preparation_log else (0, 0)
        final_shape = self.df.shape
        
        print(f"📊 Overall Transformation:")
        print(f"   Initial dataset: {initial_shape[0]:,} rows × {initial_shape[1]} columns")
        print(f"   Final dataset: {final_shape[0]:,} rows × {final_shape[1]} columns")
        print(f"   Change: {final_shape[0] - initial_shape[0]:+,} rows, {final_shape[1] - initial_shape[1]:+} columns")
        
        # Step-by-step summary
        print(f"\n🔄 Preparation Steps Completed:")
        for i, step in enumerate(self.preparation_log, 1):
            print(f"   {i}. {step['step'].replace('_', ' ').title()}: {step['details']}")
        
        # Data quality metrics comparison
        if hasattr(self, 'data_quality_metrics'):
            print(f"\n📈 Data Quality Improvement:")
            original_missing = self.data_quality_metrics.get('missing_values', 0)
            current_missing = self.df.isnull().sum().sum()
            print(f"   Missing values: {original_missing:,} → {current_missing:,}")
            
            original_duplicates = self.data_quality_metrics.get('duplicate_rows', 0)
            current_duplicates = self.df.duplicated().sum()
            print(f"   Duplicate rows: {original_duplicates:,} → {current_duplicates:,}")
        
        # Feature summary
        numeric_features = len(self.df.select_dtypes(include=[np.number]).columns)
        categorical_features = len(self.df.select_dtypes(include=['object']).columns)
        
        print(f"\n🔧 Final Feature Composition:")
        print(f"   Numeric features: {numeric_features}")
        print(f"   Categorical features: {categorical_features}")
        print(f"   Total features: {self.df.shape[1]}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        recommendations = [
            "Validate model performance with cross-validation",
            "Monitor for data drift in production",
            "Consider feature selection if model complexity is high",
            "Document preprocessing steps for reproducibility"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        print(f"\n✅ Data preparation pipeline completed successfully!")
        
        return {
            'initial_shape': initial_shape,
            'final_shape': final_shape,
            'steps_completed': len(self.preparation_log),
            'preparation_log': self.preparation_log
        }

    # --------------------------------------------------
    # 11. Enhanced Data Saving
    # --------------------------------------------------
    def save_model_ready_data(self, path):
        """
        Save the fully processed dataset with metadata
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save main dataset
            self.df.to_csv(path, index=False)
            
            # Save metadata
            metadata_path = path.replace('.csv', '_metadata.json')
            metadata = {
                'preparation_timestamp': pd.Timestamp.now().isoformat(),
                'original_shape': self.original_df.shape if self.original_df is not None else None,
                'final_shape': self.df.shape,
                'target_variable': self.target,
                'preparation_steps': self.preparation_log,
                'encoders_info': {col: len(encoder.classes_) for col, encoder in self.encoders.items()},
                'feature_names': self.df.columns.tolist()
            }
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Model-ready dataset saved to: {path}")
            print(f"📄 Metadata saved to: {metadata_path}")
            
        except Exception as e:
            print(f"❌ Error saving data: {str(e)}")

    # --------------------------------------------------
    # 12. Enhanced Train-Test Split
    # --------------------------------------------------
    def save_train_test_splits(self, output_dir: str, test_size: float = 0.2, 
                              validation_size: float = 0.1):
        """
        Enhanced train-test split with optional validation set
        
        Parameters:
        output_dir: directory to save splits
        test_size: proportion for test set
        validation_size: proportion for validation set (from training data)
        """
        print("\n" + "="*60)
        print("🔄 CREATING TRAIN-TEST SPLITS")
        print("="*60)
        
        if self.df is None:
            raise ValueError("Dataframe is empty. Run preparation steps first.")

        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataframe.")

        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        print(f"📊 Split Configuration:")
        print(f"   Test size: {test_size*100:.1f}%")
        print(f"   Validation size: {validation_size*100:.1f}% (from training)")
        print(f"   Final training size: ~{(1-test_size)*(1-validation_size)*100:.1f}%")

        # Initial train-test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Create validation set from training data
        if validation_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=validation_size, random_state=42, stratify=y_temp
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save splits
        splits_info = {}
        
        # Training set
        X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
        y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
        splits_info['train'] = {'samples': len(X_train), 'features': X_train.shape[1]}
        
        # Test set
        X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
        y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
        splits_info['test'] = {'samples': len(X_test), 'features': X_test.shape[1]}
        
        # Validation set (if created)
        if X_val is not None:
            X_val.to_csv(os.path.join(output_dir, "X_val.csv"), index=False)
            y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)
            splits_info['validation'] = {'samples': len(X_val), 'features': X_val.shape[1]}

        # Display split information
        print(f"\n📈 Split Results:")
        for split_name, info in splits_info.items():
            pct = (info['samples'] / len(self.df)) * 100
            print(f"   {split_name.title()}: {info['samples']:,} samples ({pct:.1f}%)")
        
        # Check class distribution in splits
        print(f"\n🎯 Class Distribution Check:")
        for split_name, y_split in [('Train', y_train), ('Test', y_test)] + ([('Validation', y_val)] if y_val is not None else []):
            class_dist = y_split.value_counts(normalize=True).sort_index()
            dist_str = ", ".join([f"{cls}: {pct:.1%}" for cls, pct in class_dist.items()])
            print(f"   {split_name}: {dist_str}")

        # Save split metadata
        split_metadata = {
            'split_timestamp': pd.Timestamp.now().isoformat(),
            'test_size': test_size,
            'validation_size': validation_size,
            'random_state': 42,
            'stratified': True,
            'splits_info': splits_info,
            'target_variable': self.target,
            'feature_names': X.columns.tolist()
        }
        
        metadata_path = os.path.join(output_dir, "split_metadata.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(split_metadata, f, indent=2)

        print(f"\n✅ Train-test splits saved in: {output_dir}")
        print(f"📄 Split metadata saved to: {metadata_path}")
        
        # Log the split
        self.preparation_log.append({
            'step': 'train_test_split',
            'action': 'create_splits',
            'shape_before': self.df.shape,
            'shape_after': self.df.shape,
            'details': f'Created train ({len(X_train):,}), test ({len(X_test):,})' + 
                      (f', validation ({len(X_val):,})' if X_val is not None else '') + ' splits'
        })

        return splits_info

