import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly, fall back to matplotlib if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    # Safe renderer (Windows + Python 3.13)
    pio.renderers.default = "notebook_connected"
    PLOTLY_AVAILABLE = True
except ImportError:
    print("⚠️ Plotly not available. Using matplotlib for visualizations.")
    PLOTLY_AVAILABLE = False


class TrafficEDA:
    """
    Enhanced Exploratory Data Analysis Pipeline
    for Traffic Accident Analysis Project
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.insights = []  # Store key insights for summary

    # --------------------------------------------------
    # 1. Load Data with Enhanced Validation
    # --------------------------------------------------
    def load_data(self):
        """Load data with basic validation and info"""
        try:
            self.df = pd.read_csv(self.data_path)
            print("✅ Data loaded successfully.")
            print(f"📊 Dataset shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            
            # Basic data quality check
            memory_usage = self.df.memory_usage(deep=True).sum() / 1024**2
            print(f"💾 Memory usage: {memory_usage:.1f} MB")
            
            return self.df
        except FileNotFoundError:
            print(f"❌ Error: File not found at {self.data_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None

    # --------------------------------------------------
    # 2. Enhanced Initial Data Inspection
    # --------------------------------------------------
    def initial_exploration(self):
        """Enhanced data overview with key insights"""
        print("\n" + "="*60)
        print("📋 INITIAL DATA EXPLORATION")
        print("="*60)
        
        # Basic info
        print(f"📊 Dataset Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
        # Data types summary
        dtype_summary = self.df.dtypes.value_counts()
        print(f"\n📈 Data Types Summary:")
        for dtype, count in dtype_summary.items():
            print(f"   {dtype}: {count} columns")
        
        # Missing data overview
        missing_total = self.df.isnull().sum().sum()
        missing_pct = (missing_total / (self.df.shape[0] * self.df.shape[1])) * 100
        print(f"\n🔍 Missing Data: {missing_total:,} values ({missing_pct:.1f}% of total)")
        
        # Duplicate rows
        duplicates = self.df.duplicated().sum()
        print(f"🔄 Duplicate Rows: {duplicates:,} ({duplicates/len(self.df)*100:.1f}%)")
        
        # Sample data
        print(f"\n📝 Sample Records (first 5 rows):")
        display_df = self.df.head()
        print(display_df.to_string())
        
        # Quick stats for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n📊 Numeric Features Summary:")
            print(self.df[numeric_cols].describe().round(2))
        
        # Store insight
        self.insights.append(f"Dataset contains {self.df.shape[0]:,} accident records with {missing_pct:.1f}% missing data")
        
        return self.df

    # --------------------------------------------------
    # 3. Enhanced Target Distribution Analysis
    # --------------------------------------------------
    def target_distribution(self, target: str = "most_severe_injury"):
        """
        Enhanced target variable analysis with insights
        """
        print("\n" + "="*60)
        print("🎯 TARGET VARIABLE ANALYSIS")
        print("="*60)

        if target not in self.df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe.")

        counts = self.df[target].value_counts().reset_index()
        counts.columns = [target, "count"]
        counts["percent"] = counts["count"] / len(self.df) * 100

        print(f"📊 Distribution of {target}:")
        for _, row in counts.iterrows():
            print(f"   {row[target]}: {row['count']:,} ({row['percent']:.1f}%)")

        # Class imbalance analysis
        max_class_pct = counts['percent'].max()
        min_class_pct = counts['percent'].min()
        imbalance_ratio = max_class_pct / min_class_pct
        
        print(f"\n⚖️ Class Balance Analysis:")
        print(f"   Most common class: {max_class_pct:.1f}%")
        print(f"   Least common class: {min_class_pct:.1f}%")
        print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 5:
            print("   ⚠️ Significant class imbalance detected - consider balancing techniques")
        
        # Visualization
        if PLOTLY_AVAILABLE:
            # Enhanced plotly visualization
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "bar"}, {"type": "pie"}]],
                subplot_titles=["Count Distribution", "Percentage Distribution"]
            )
            
            # Bar chart
            fig.add_trace(
                go.Bar(x=counts[target], y=counts["count"], 
                       text=counts["percent"].round(1).astype(str) + "%",
                       textposition="outside", name="Count"),
                row=1, col=1
            )
            
            # Pie chart
            fig.add_trace(
                go.Pie(labels=counts[target], values=counts["count"], 
                       textinfo="label+percent", name="Distribution"),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f"Target Variable Distribution: {target}",
                showlegend=False,
                height=400
            )
            fig.show()
        else:
            # Matplotlib fallback
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 2, 1)
            counts.set_index(target)['count'].plot(kind='bar', color='skyblue')
            plt.title('Count Distribution')
            plt.xlabel('Category')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            plt.pie(counts['count'], labels=counts[target], autopct='%1.1f%%')
            plt.title('Percentage Distribution')
            
            plt.tight_layout()
            plt.show()
        
        # Store insight
        most_common = counts.iloc[0][target]
        self.insights.append(f"Most accidents result in '{most_common}' ({max_class_pct:.1f}%)")
        
        return counts

    # --------------------------------------------------
    # 4. Enhanced Missing Value Analysis
    # --------------------------------------------------
    def missing_value_analysis(self):
        """Comprehensive missing value analysis with patterns"""
        print("\n" + "="*60)
        print("🔍 MISSING VALUE ANALYSIS")
        print("="*60)
        
        missing = self.df.isnull().sum().reset_index()
        missing.columns = ["Feature", "Missing_Count"]
        missing["Missing_Percent"] = (missing["Missing_Count"] / len(self.df)) * 100
        missing = missing[missing["Missing_Count"] > 0].sort_values("Missing_Count", ascending=False)
        
        if missing.empty:
            print("✅ No missing values found in the dataset!")
            return missing
        
        print(f"📊 Features with missing values: {len(missing)}")
        print("\nTop missing value features:")
        for _, row in missing.head(10).iterrows():
            print(f"   {row['Feature']}: {row['Missing_Count']:,} ({row['Missing_Percent']:.1f}%)")
        
        # Visualization
        if len(missing) > 0:
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    missing.head(15),
                    x="Missing_Percent",
                    y="Feature",
                    orientation="h",
                    title="Missing Values by Feature (Top 15)",
                    color="Missing_Percent",
                    color_continuous_scale="Reds"
                )
                fig.update_layout(height=max(400, len(missing.head(15)) * 25))
                fig.show()
            else:
                plt.figure(figsize=(10, 6))
                missing_pct = missing.set_index('Feature')['Missing_Percent']
                missing_pct.head(15).plot(kind='barh', color='coral')
                plt.title('Missing Values by Feature (%)')
                plt.xlabel('Missing Percentage')
                plt.tight_layout()
                plt.show()
        
        # Missing value patterns
        if len(missing) > 1:
            print(f"\n🔗 Missing Value Patterns:")
            # Check for correlated missing values
            missing_matrix = self.df[missing['Feature'].head(5)].isnull()
            if len(missing_matrix.columns) > 1:
                missing_corr = missing_matrix.corr()
                high_corr_pairs = []
                for i in range(len(missing_corr.columns)):
                    for j in range(i+1, len(missing_corr.columns)):
                        corr_val = missing_corr.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            high_corr_pairs.append((missing_corr.columns[i], missing_corr.columns[j], corr_val))
                
                if high_corr_pairs:
                    print("   Correlated missing patterns found:")
                    for col1, col2, corr in high_corr_pairs:
                        print(f"   {col1} ↔ {col2}: {corr:.2f}")
                else:
                    print("   No strong missing value correlations found")
        
        # Store insight
        if not missing.empty:
            worst_feature = missing.iloc[0]
            self.insights.append(f"'{worst_feature['Feature']}' has the most missing values ({worst_feature['Missing_Percent']:.1f}%)")
        
        return missing

    # --------------------------------------------------
    # 5. Enhanced Univariate Analysis
    # --------------------------------------------------
    def univariate_analysis(self):
        """
        Enhanced single-variable analysis with insights
        """
        print("\n" + "="*60)
        print("📊 UNIVARIATE ANALYSIS")
        print("="*60)

        # 1) Temporal Analysis - Crash Hour
        if "crash_hour" in self.df.columns:
            print("\n🕐 Accident Distribution by Hour:")
            
            hourly_counts = self.df["crash_hour"].value_counts().sort_index()
            peak_hour = hourly_counts.idxmax()
            peak_count = hourly_counts.max()
            
            print(f"   Peak hour: {peak_hour}:00 ({peak_count:,} accidents)")
            
            # Identify rush hours
            morning_rush = hourly_counts[7:10].sum()
            evening_rush = hourly_counts[17:20].sum()
            night_time = hourly_counts[(hourly_counts.index >= 22) | (hourly_counts.index <= 6)].sum()
            
            print(f"   Morning rush (7-9 AM): {morning_rush:,} accidents")
            print(f"   Evening rush (5-7 PM): {evening_rush:,} accidents")
            print(f"   Night time (10 PM-6 AM): {night_time:,} accidents")
            
            if PLOTLY_AVAILABLE:
                fig1 = px.bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    title="Accident Distribution by Hour of Day",
                    labels={"x": "Hour of Day", "y": "Number of Accidents"},
                    color=hourly_counts.values,
                    color_continuous_scale="Reds"
                )
                
                # Add rush hour annotations
                fig1.add_vrect(x0=7, x1=9, fillcolor="yellow", opacity=0.2, 
                              annotation_text="Morning Rush", annotation_position="top left")
                fig1.add_vrect(x0=17, x1=19, fillcolor="orange", opacity=0.2,
                              annotation_text="Evening Rush", annotation_position="top right")
                
                fig1.show()
            else:
                plt.figure(figsize=(12, 4))
                hourly_counts.plot(kind='bar', color='lightcoral')
                plt.title('Accidents by Hour of Day')
                plt.xlabel('Hour')
                plt.ylabel('Number of Accidents')
                plt.axvspan(7, 9, alpha=0.2, color='yellow', label='Morning Rush')
                plt.axvspan(17, 19, alpha=0.2, color='orange', label='Evening Rush')
                plt.legend()
                plt.xticks(rotation=0)
                plt.tight_layout()
                plt.show()
            
            self.insights.append(f"Peak accident hour is {peak_hour}:00 with {peak_count:,} accidents")

        # 2) Day of Week Analysis
        if "crash_day_of_week" in self.df.columns:
            print("\n📅 Accident Distribution by Day of Week:")
            
            day_names = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 
                        5: "Friday", 6: "Saturday", 7: "Sunday"}
            
            daily_counts = self.df["crash_day_of_week"].value_counts().sort_index()
            daily_counts.index = [day_names[i] for i in daily_counts.index]
            
            peak_day = daily_counts.idxmax()
            weekend_total = daily_counts[["Saturday", "Sunday"]].sum()
            weekday_total = daily_counts[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]].sum()
            
            print(f"   Peak day: {peak_day} ({daily_counts[peak_day]:,} accidents)")
            print(f"   Weekend total: {weekend_total:,} accidents")
            print(f"   Weekday total: {weekday_total:,} accidents")
            
            if PLOTLY_AVAILABLE:
                fig2 = px.bar(
                    x=daily_counts.index,
                    y=daily_counts.values,
                    title="Accident Distribution by Day of Week",
                    labels={"x": "Day of Week", "y": "Number of Accidents"},
                    color=daily_counts.values,
                    color_continuous_scale="Blues"
                )
                fig2.show()
            else:
                plt.figure(figsize=(10, 4))
                daily_counts.plot(kind='bar', color='lightblue')
                plt.title('Accidents by Day of Week')
                plt.xlabel('Day')
                plt.ylabel('Number of Accidents')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

        # 3) Monthly Analysis
        if "crash_month" in self.df.columns:
            print("\n📆 Accident Distribution by Month:")
            
            month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                          7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
            
            monthly_counts = self.df["crash_month"].value_counts().sort_index()
            peak_month = monthly_counts.idxmax()
            
            print(f"   Peak month: {month_names[peak_month]} ({monthly_counts[peak_month]:,} accidents)")
            
            if PLOTLY_AVAILABLE:
                fig3 = px.line(
                    x=[month_names[i] for i in monthly_counts.index],
                    y=monthly_counts.values,
                    title="Accident Trends by Month",
                    labels={"x": "Month", "y": "Number of Accidents"},
                    markers=True
                )
                fig3.show()
            else:
                plt.figure(figsize=(10, 4))
                month_labels = [month_names[i] for i in monthly_counts.index]
                plt.plot(month_labels, monthly_counts.values, marker='o', color='green')
                plt.title('Accident Trends by Month')
                plt.xlabel('Month')
                plt.ylabel('Number of Accidents')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

        # 4) Weather Conditions (if categorical)
        if "weather_condition" in self.df.columns:
            print("\n🌤️ Weather Conditions Analysis:")
            
            weather_counts = self.df["weather_condition"].value_counts().head(10)
            most_common_weather = weather_counts.index[0]
            
            print(f"   Most common condition: {most_common_weather} ({weather_counts.iloc[0]:,} accidents)")
            print(f"   Total weather categories: {self.df['weather_condition'].nunique()}")
            
            if PLOTLY_AVAILABLE:
                fig4 = px.bar(
                    x=weather_counts.values,
                    y=weather_counts.index,
                    orientation="h",
                    title="Top 10 Weather Conditions",
                    labels={"x": "Number of Accidents", "y": "Weather Condition"},
                    color=weather_counts.values,
                    color_continuous_scale="Greens"
                )
                fig4.show()
            else:
                plt.figure(figsize=(10, 6))
                weather_counts.plot(kind='barh', color='lightgreen')
                plt.title('Top 10 Weather Conditions')
                plt.xlabel('Number of Accidents')
                plt.ylabel('Weather Condition')
                plt.tight_layout()
                plt.show()
            
            self.insights.append(f"Most accidents occur in '{most_common_weather}' conditions")

  # --------------------------------------------------
    # 6. Enhanced Bivariate Analysis (Simplified)
    # --------------------------------------------------
    def bivariate_analysis(self, target="most_severe_injury"):
        """Two-variable analysis focused on visualization and trends"""
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Check for Plotly
        try:
            import plotly.express as px
            PLOTLY_AVAILABLE = True
        except ImportError:
            PLOTLY_AVAILABLE = False

        print("\n" + "="*60)
        print("🔗 BIVARIATE ANALYSIS")
        print("="*60)

        # 1) Severity by Hour
        if "crash_hour" in self.df.columns:
            print("\n🕐 Accident Severity by Hour:")
            
            # Visualization
            if PLOTLY_AVAILABLE:
                fig = px.histogram(
                    self.df,
                    x="crash_hour",
                    color=target,
                    title="Accident Severity Distribution by Hour of Day",
                    barmode="stack",
                    labels={"crash_hour": "Hour of Day", "count": "Number of Accidents"}
                )
                fig.show()
            else:
                plt.figure(figsize=(12, 6))
                for severity in self.df[target].unique():
                    subset = self.df[self.df[target] == severity]
                    plt.hist(subset["crash_hour"], alpha=0.7, label=severity, bins=24)
                plt.title("Accident Severity Distribution by Hour of Day")
                plt.xlabel("Hour of Day")
                plt.ylabel("Number of Accidents")
                plt.legend()
                plt.tight_layout()
                plt.show()
            
            # Find peak severity hours
            severity_by_hour = self.df.groupby("crash_hour")[target].apply(
                lambda x: (x != x.mode().iloc[0]).sum() / len(x) * 100 if len(x) > 0 else 0
            ).round(1)
            peak_severity_hour = severity_by_hour.idxmax()
            print(f"   Hour with highest severity rate: {peak_severity_hour}:00 ({severity_by_hour[peak_severity_hour]:.1f}%)")

        # 2) Severity by Day of Week
        if "crash_day_of_week" in self.df.columns:
            print("\n📅 Accident Severity by Day of Week:")
            
            day_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
            df_day = self.df.copy()
            df_day["day_name"] = df_day["crash_day_of_week"].map(day_names)
            
            if PLOTLY_AVAILABLE:
                fig = px.histogram(
                    df_day,
                    x="day_name",
                    color=target,
                    title="Accident Severity by Day of Week",
                    barmode="group",
                    category_orders={"day_name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]},
                    labels={"day_name": "Day of Week", "count": "Number of Accidents"}
                )
                fig.show()
            else:
                plt.figure(figsize=(12, 6))
                day_severity_pct = pd.crosstab(df_day["day_name"], df_day[target], normalize='index') * 100
                day_severity_pct.plot(kind='bar', stacked=True)
                plt.title("Accident Severity by Day of Week (%)")
                plt.xlabel("Day of Week")
                plt.ylabel("Percentage")
                plt.xticks(rotation=45)
                plt.legend(title="Severity")
                plt.tight_layout()
                plt.show()

        # 3) Weather vs Severity Analysis
        if "weather_condition" in self.df.columns:
            print("\n🌤️ Weather Conditions vs Severity:")
            
            weather_severity = pd.crosstab(self.df["weather_condition"], self.df[target])
            top_weather = weather_severity.sum(axis=1).nlargest(8).index
            weather_subset = weather_severity.loc[top_weather]
            
            if PLOTLY_AVAILABLE:
                fig = px.imshow(
                    weather_subset.T,
                    title="Weather Conditions vs Accident Severity (Top 8 Conditions)",
                    labels=dict(x="Weather Condition", y="Severity Level", color="Count"),
                    aspect="auto",
                    color_continuous_scale="Reds"
                )
                fig.update_xaxes(tickangle=45)
                fig.show()
            else:
                import seaborn as sns
                plt.figure(figsize=(12, 6))
                sns.heatmap(weather_subset.T, annot=True, fmt='d', cmap='Reds')
                plt.title("Weather Conditions vs Accident Severity (Top 8 Conditions)")
                plt.xlabel("Weather Condition")
                plt.ylabel("Severity Level")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

        # 4) Injuries Analysis
        if "injuries_total" in self.df.columns:
            print("\n🏥 Injury Analysis by Severity:")
            
            # Note: Changed 'stats' to 'injury_summary' to avoid library name conflicts
            injury_summary = self.df.groupby(target)["injuries_total"].agg(['mean', 'median', 'std']).round(2)
            print("   Average injuries by severity level:")
            for severity, row in injury_summary.iterrows():
                print(f"   {severity}: Mean={row['mean']}, Median={row['median']}, Std={row['std']}")
            
            if PLOTLY_AVAILABLE:
                fig = px.box(
                    self.df,
                    x=target,
                    y="injuries_total",
                    title="Total Injuries Distribution by Severity Level",
                    labels={"injuries_total": "Total Injuries", target: "Severity Level"}
                )
                fig.show()
            else:
                plt.figure(figsize=(10, 6))
                self.df.boxplot(column="injuries_total", by=target, ax=plt.gca())
                plt.title("Total Injuries Distribution by Severity Level")
                plt.xlabel("Severity Level")
                plt.ylabel("Total Injuries")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

        # Store insights
        if "crash_hour" in self.df.columns:
            self.insights.append(f"Hour {peak_severity_hour}:00 has the highest severity rate")

            
    # --------------------------------------------------
    # 7. Focused Multivariate Analysis
    # --------------------------------------------------
    def multivariate_analysis(self, target="most_severe_injury"):
        """Focused multivariate analysis with actionable insights"""
        print("\n" + "="*60)
        print("🔍 MULTIVARIATE ANALYSIS")
        print("="*60)

        # 1) Time-based Risk Analysis
        if all(col in self.df.columns for col in ["crash_hour", "crash_day_of_week"]):
            print("\n⏰ Time-based Risk Patterns:")
            
            # Create time-based risk heatmap
            heatmap_data = pd.crosstab(self.df["crash_day_of_week"], self.df["crash_hour"])
            
            # Identify high-risk time slots
            risk_threshold = heatmap_data.values.mean() + heatmap_data.values.std()
            high_risk_slots = []
            
            for day in heatmap_data.index:
                for hour in heatmap_data.columns:
                    if heatmap_data.loc[day, hour] > risk_threshold:
                        day_name = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}[day]
                        high_risk_slots.append(f"{day_name} {hour}:00")
            
            print(f"   High-risk time slots ({len(high_risk_slots)} identified):")
            for slot in high_risk_slots[:10]:  # Show top 10
                print(f"   • {slot}")
            
            if PLOTLY_AVAILABLE:
                fig = px.imshow(
                    heatmap_data,
                    title="Accident Risk Heatmap: Day of Week × Hour of Day",
                    labels=dict(x="Hour of Day", y="Day of Week (1=Mon, 7=Sun)", color="Accident Count"),
                    aspect="auto",
                    color_continuous_scale="Reds"
                )
                fig.show()
            else:
                plt.figure(figsize=(12, 8))
                sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Reds')
                plt.title("Accident Risk Heatmap: Day of Week × Hour of Day")
                plt.xlabel("Hour of Day")
                plt.ylabel("Day of Week (1=Mon, 7=Sun)")
                plt.tight_layout()
                plt.show()

        # 2) Weather-Time Interaction
        if all(col in self.df.columns for col in ["weather_condition", "crash_hour"]):
            print("\n🌤️ Weather-Time Interaction Analysis:")
            
            # Focus on top weather conditions and peak hours
            top_weather = self.df["weather_condition"].value_counts().head(5).index
            peak_hours = self.df["crash_hour"].value_counts().head(8).index
            
            weather_time_subset = self.df[
                (self.df["weather_condition"].isin(top_weather)) & 
                (self.df["crash_hour"].isin(peak_hours))
            ]
            
            if not weather_time_subset.empty:
                weather_time_crosstab = pd.crosstab(
                    weather_time_subset["weather_condition"], 
                    weather_time_subset["crash_hour"]
                )
                
                if PLOTLY_AVAILABLE:
                    fig = px.imshow(
                        weather_time_crosstab,
                        title="Weather × Hour Interaction (Top Conditions & Peak Hours)",
                        labels=dict(x="Hour of Day", y="Weather Condition", color="Accident Count"),
                        aspect="auto",
                        color_continuous_scale="Blues"
                    )
                    fig.show()
                else:
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(weather_time_crosstab, annot=True, fmt='d', cmap='Blues')
                    plt.title("Weather × Hour Interaction (Top Conditions & Peak Hours)")
                    plt.xlabel("Hour of Day")
                    plt.ylabel("Weather Condition")
                    plt.tight_layout()
                    plt.show()

        # 3) Severity Risk Factors
        if "injuries_total" in self.df.columns:
            print("\n🚨 Severity Risk Factor Analysis:")
            
            # Create risk categories
            self.df["risk_category"] = "Low Risk"
            
            # Define high-risk conditions
            if "crash_hour" in self.df.columns:
                night_mask = (self.df["crash_hour"] >= 22) | (self.df["crash_hour"] <= 6)
                self.df.loc[night_mask, "risk_category"] = "High Risk (Night)"
            
            if "crash_day_of_week" in self.df.columns:
                weekend_mask = self.df["crash_day_of_week"].isin([6, 7])
                self.df.loc[weekend_mask, "risk_category"] = "Medium Risk (Weekend)"
            
            # Analyze severity by risk category
            risk_severity = pd.crosstab(self.df["risk_category"], self.df[target], normalize="index") * 100
            
            print("   Severity distribution by risk category (%):")
            for risk_cat in risk_severity.index:
                severe_pct = risk_severity.loc[risk_cat].iloc[-1]  # Assuming last category is most severe
                print(f"   {risk_cat}: {severe_pct:.1f}% severe accidents")
            
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    risk_severity.reset_index(),
                    x="risk_category",
                    y=risk_severity.columns.tolist(),
                    title="Accident Severity by Risk Category (%)",
                    barmode="stack",
                    labels={"value": "Percentage", "risk_category": "Risk Category"}
                )
                fig.show()
            else:
                plt.figure(figsize=(10, 6))
                risk_severity.plot(kind='bar', stacked=True)
                plt.title("Accident Severity by Risk Category (%)")
                plt.xlabel("Risk Category")
                plt.ylabel("Percentage")
                plt.xticks(rotation=45)
                plt.legend(title="Severity")
                plt.tight_layout()
                plt.show()

        # 4) Feature Interaction Summary
        print("\n📊 Key Feature Interactions:")
        
        interactions = []
        
        # Time-based interactions
        if all(col in self.df.columns for col in ["crash_hour", "crash_day_of_week"]):
            weekend_night = self.df[
                (self.df["crash_day_of_week"].isin([6, 7])) & 
                ((self.df["crash_hour"] >= 22) | (self.df["crash_hour"] <= 6))
            ]
            interactions.append(f"Weekend nights: {len(weekend_night):,} accidents ({len(weekend_night)/len(self.df)*100:.1f}%)")
        
        # Weather interactions
        if "weather_condition" in self.df.columns and "crash_hour" in self.df.columns:
            bad_weather = self.df[self.df["weather_condition"] != self.df["weather_condition"].mode().iloc[0]]
            rush_hour_bad_weather = bad_weather[
                ((bad_weather["crash_hour"] >= 7) & (bad_weather["crash_hour"] <= 9)) |
                ((bad_weather["crash_hour"] >= 17) & (bad_weather["crash_hour"] <= 19))
            ]
            interactions.append(f"Rush hour + bad weather: {len(rush_hour_bad_weather):,} accidents")
        
        for interaction in interactions:
            print(f"   • {interaction}")
        
        # Store insights
        if high_risk_slots:
            self.insights.append(f"Identified {len(high_risk_slots)} high-risk time slots")
        
        return self.df

    # --------------------------------------------------
    # 8. Enhanced Correlation Analysis
    # --------------------------------------------------
    def correlation_analysis(self, target: str = "most_severe_injury"):
        """
        Enhanced correlation analysis with insights and feature relationships
        """
        print("\n" + "="*60)
        print("🔗 CORRELATION ANALYSIS")
        print("="*60)

        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print("❌ No numeric columns found for correlation analysis.")
            return

        print(f"📊 Analyzing correlations for {len(numeric_df.columns)} numeric features")
        
        corr = numeric_df.corr()

        # Enhanced correlation heatmap with better styling
        if PLOTLY_AVAILABLE:
            fig = px.imshow(
                corr,
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu_r",
                aspect="auto",
                zmin=-1, zmax=1
            )
            
            # Add correlation values as text
            fig.update_traces(
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 8}
            )
            
            fig.update_layout(
                width=max(600, len(corr.columns) * 40),
                height=max(600, len(corr.columns) * 40)
            )
            fig.show()
        else:
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, 
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.show()

        # Find strong correlations
        print("\n🔍 Strong Correlations Analysis:")
        
        strong_correlations = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_val = corr.iloc[i, j]
                if abs(corr_val) > 0.5:  # Strong correlation threshold
                    strong_correlations.append({
                        'Feature 1': corr.columns[i],
                        'Feature 2': corr.columns[j],
                        'Correlation': corr_val,
                        'Strength': 'Very Strong' if abs(corr_val) > 0.8 else 'Strong'
                    })
        
        if strong_correlations:
            strong_df = pd.DataFrame(strong_correlations)
            strong_df = strong_df.sort_values('Correlation', key=abs, ascending=False)
            
            print(f"   Found {len(strong_correlations)} strong correlations:")
            for _, row in strong_df.head(10).iterrows():
                print(f"   • {row['Feature 1']} ↔ {row['Feature 2']}: {row['Correlation']:.3f} ({row['Strength']})")
            
            # Visualize top correlations
            if len(strong_df) > 0:
                if PLOTLY_AVAILABLE:
                    fig_corr = px.bar(
                        strong_df.head(10),
                        x='Correlation',
                        y=[f"{row['Feature 1']} ↔ {row['Feature 2']}" for _, row in strong_df.head(10).iterrows()],
                        orientation='h',
                        title="Top 10 Strongest Feature Correlations",
                        color='Correlation',
                        color_continuous_scale='RdBu_r'
                    )
                    fig_corr.show()
                else:
                    plt.figure(figsize=(10, 6))
                    correlations = strong_df.head(10)['Correlation']
                    labels = [f"{row['Feature 1']} ↔ {row['Feature 2']}" for _, row in strong_df.head(10).iterrows()]
                    plt.barh(labels, correlations, color=['red' if x < 0 else 'blue' for x in correlations])
                    plt.title("Top 10 Strongest Feature Correlations")
                    plt.xlabel("Correlation")
                    plt.tight_layout()
                    plt.show()
        else:
            print("   No strong correlations (>0.5) found between features")

        # Target correlation analysis (if target is numeric or can be encoded)
        if target in numeric_df.columns:
            print(f"\n🎯 Correlations with Target Variable '{target}':")
            
            target_corr = corr[target].drop(labels=[target]).sort_values(key=abs, ascending=False)
            
            print("   Top positive correlations:")
            positive_corr = target_corr[target_corr > 0].head(5)
            for feature, corr_val in positive_corr.items():
                print(f"   • {feature}: {corr_val:.3f}")
            
            print("   Top negative correlations:")
            negative_corr = target_corr[target_corr < 0].head(5)
            for feature, corr_val in negative_corr.items():
                print(f"   • {feature}: {corr_val:.3f}")
            
            # Visualize target correlations
            top_target_corr = pd.concat([positive_corr.head(5), negative_corr.head(5)])
            if not top_target_corr.empty:
                if PLOTLY_AVAILABLE:
                    fig_target = px.bar(
                        x=top_target_corr.values,
                        y=top_target_corr.index,
                        orientation='h',
                        title=f"Top Correlations with {target}",
                        color=top_target_corr.values,
                        color_continuous_scale='RdBu_r'
                    )
                    fig_target.show()
                else:
                    plt.figure(figsize=(10, 6))
                    colors = ['red' if x < 0 else 'blue' for x in top_target_corr.values]
                    plt.barh(top_target_corr.index, top_target_corr.values, color=colors)
                    plt.title(f"Top Correlations with {target}")
                    plt.xlabel("Correlation")
                    plt.tight_layout()
                    plt.show()
        
        # Multicollinearity detection
        print(f"\n⚠️ Multicollinearity Check:")
        high_corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.9:
                    high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        
        if high_corr_pairs:
            print(f"   Found {len(high_corr_pairs)} potential multicollinearity issues (>0.9):")
            for feat1, feat2, corr_val in high_corr_pairs:
                print(f"   • {feat1} ↔ {feat2}: {corr_val:.3f}")
            print("   Consider removing one feature from each highly correlated pair")
        else:
            print("   No severe multicollinearity detected")
        
        # Store insights
        if strong_correlations:
            strongest = strong_df.iloc[0]
            self.insights.append(f"Strongest correlation: {strongest['Feature 1']} ↔ {strongest['Feature 2']} ({strongest['Correlation']:.3f})")
        
        return corr

    # --------------------------------------------------
    # 9. Enhanced Outlier Detection
    # --------------------------------------------------
    def outlier_detection(self):
        """Enhanced outlier detection with actionable insights"""
        print("\n" + "="*60)
        print("📦 OUTLIER DETECTION ANALYSIS")
        print("="*60)

        # Focus on key numeric columns
        key_numeric_cols = [
            "injuries_total", "injuries_incapacitating", 
            "injuries_non_incapacitating", "num_units"
        ]
        
        available_cols = [col for col in key_numeric_cols if col in self.df.columns]
        
        if not available_cols:
            print("❌ No key numeric columns found for outlier analysis.")
            return

        print(f"🔍 Analyzing outliers in {len(available_cols)} key features")
        
        outlier_summary = []
        outlier_details = {}

        for col in available_cols:
            print(f"\n📊 Analyzing '{col}':")
            
            # Calculate IQR bounds
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outliers_count = outlier_mask.sum()
            outlier_pct = (outliers_count / len(self.df)) * 100
            
            # Get outlier values
            outlier_values = self.df[outlier_mask][col]
            
            print(f"   Range: {self.df[col].min():.1f} to {self.df[col].max():.1f}")
            print(f"   IQR bounds: {lower_bound:.1f} to {upper_bound:.1f}")
            print(f"   Outliers: {outliers_count:,} ({outlier_pct:.1f}%)")
            
            if outliers_count > 0:
                print(f"   Outlier range: {outlier_values.min():.1f} to {outlier_values.max():.1f}")
                print(f"   Most extreme: {outlier_values.max():.1f}")

            outlier_summary.append({
                "Feature": col,
                "Outliers": outliers_count,
                "Percent": outlier_pct,
                "Lower_Bound": lower_bound,
                "Upper_Bound": upper_bound,
                "Max_Outlier": outlier_values.max() if outliers_count > 0 else None
            })
            
            outlier_details[col] = {
                'mask': outlier_mask,
                'values': outlier_values,
                'bounds': (lower_bound, upper_bound)
            }

        # Create comprehensive visualization
        if len(available_cols) > 1:
            if PLOTLY_AVAILABLE:
                fig = make_subplots(
                    rows=2, cols=len(available_cols),
                    subplot_titles=[f"{col} - Box Plot" for col in available_cols] + 
                                  [f"{col} - Distribution" for col in available_cols],
                    vertical_spacing=0.12
                )
                
                for i, col in enumerate(available_cols):
                    # Box plot
                    fig.add_trace(
                        go.Box(y=self.df[col], name=col, showlegend=False),
                        row=1, col=i+1
                    )
                    
                    # Histogram with bounds
                    fig.add_trace(
                        go.Histogram(x=self.df[col], name=col, showlegend=False, nbinsx=30),
                        row=2, col=i+1
                    )
                    
                    # Add bound lines
                    bounds = outlier_details[col]['bounds']
                    fig.add_vline(x=bounds[0], line_dash="dash", line_color="red", 
                                 annotation_text="Lower", row=2, col=i+1)
                    fig.add_vline(x=bounds[1], line_dash="dash", line_color="red", 
                                 annotation_text="Upper", row=2, col=i+1)
                
                fig.update_layout(
                    title="Outlier Analysis: Box Plots and Distributions",
                    height=600,
                    showlegend=False
                )
                fig.show()
            else:
                fig, axes = plt.subplots(2, len(available_cols), figsize=(15, 8))
                if len(available_cols) == 1:
                    axes = axes.reshape(-1, 1)
                
                for i, col in enumerate(available_cols):
                    # Box plot
                    axes[0, i].boxplot(self.df[col])
                    axes[0, i].set_title(f"{col} - Box Plot")
                    axes[0, i].set_ylabel("Values")
                    
                    # Histogram with bounds
                    axes[1, i].hist(self.df[col], bins=30, alpha=0.7, color='skyblue')
                    bounds = outlier_details[col]['bounds']
                    axes[1, i].axvline(bounds[0], color='red', linestyle='--', label='Lower Bound')
                    axes[1, i].axvline(bounds[1], color='red', linestyle='--', label='Upper Bound')
                    axes[1, i].set_title(f"{col} - Distribution")
                    axes[1, i].set_xlabel("Values")
                    axes[1, i].set_ylabel("Frequency")
                    axes[1, i].legend()
                
                plt.tight_layout()
                plt.show()

        # Outlier summary visualization
        if outlier_summary:
            summary_df = pd.DataFrame(outlier_summary)
            
            if PLOTLY_AVAILABLE:
                fig_summary = px.bar(
                    summary_df,
                    x="Feature",
                    y="Percent",
                    title="Outlier Percentage by Feature",
                    text="Outliers",
                    color="Percent",
                    color_continuous_scale="Reds"
                )
                fig_summary.update_traces(texttemplate='%{text}', textposition='outside')
                fig_summary.show()
            else:
                plt.figure(figsize=(10, 6))
                plt.bar(summary_df["Feature"], summary_df["Percent"], color='coral')
                plt.title("Outlier Percentage by Feature")
                plt.xlabel("Feature")
                plt.ylabel("Outlier Percentage")
                plt.xticks(rotation=45)
                for i, v in enumerate(summary_df["Outliers"]):
                    plt.text(i, summary_df["Percent"].iloc[i] + 0.1, str(v), ha='center')
                plt.tight_layout()
                plt.show()

        # Outlier impact analysis
        print(f"\n🎯 Outlier Impact Analysis:")
        
        total_outliers = sum(row["Outliers"] for row in outlier_summary)
        print(f"   Total outlier instances: {total_outliers:,}")
        
        # Find records with multiple outliers
        if len(available_cols) > 1:
            multiple_outlier_mask = pd.Series(False, index=self.df.index)
            for col in available_cols:
                multiple_outlier_mask |= outlier_details[col]['mask']
            
            multiple_outliers = multiple_outlier_mask.sum()
            print(f"   Records with any outliers: {multiple_outliers:,} ({multiple_outliers/len(self.df)*100:.1f}%)")
            
            # Check for records with outliers in multiple features
            outlier_counts_per_record = pd.Series(0, index=self.df.index)
            for col in available_cols:
                outlier_counts_per_record += outlier_details[col]['mask']
            
            multi_feature_outliers = (outlier_counts_per_record > 1).sum()
            if multi_feature_outliers > 0:
                print(f"   Records with multiple feature outliers: {multi_feature_outliers:,}")

        # Recommendations
        print(f"\n💡 Recommendations:")
        
        high_outlier_features = [row for row in outlier_summary if row["Percent"] > 5]
        if high_outlier_features:
            print("   Features with high outlier rates (>5%):")
            for feature in high_outlier_features:
                print(f"   • {feature['Feature']}: Consider capping or transformation")
        
        extreme_outliers = [row for row in outlier_summary if row["Max_Outlier"] and row["Max_Outlier"] > row["Upper_Bound"] * 3]
        if extreme_outliers:
            print("   Features with extreme outliers:")
            for feature in extreme_outliers:
                print(f"   • {feature['Feature']}: Investigate data quality")
        
        if not high_outlier_features and not extreme_outliers:
            print("   ✅ Outlier levels appear reasonable for this dataset")

        # Store insights
        if outlier_summary:
            worst_outlier_feature = max(outlier_summary, key=lambda x: x["Percent"])
            self.insights.append(f"'{worst_outlier_feature['Feature']}' has the most outliers ({worst_outlier_feature['Percent']:.1f}%)")
        
        return outlier_summary

    # --------------------------------------------------
    # 10. Enhanced Domain Validation
    # --------------------------------------------------
    def domain_validation(self):
        """Enhanced domain-specific data validation"""
        print("\n" + "="*60)
        print("✅ DOMAIN VALIDATION CHECKS")
        print("="*60)

        validation_results = []
        
        # Define validation rules
        validation_rules = {
            "crash_hour": {
                "range": (0, 23),
                "description": "Hour of day (0-23)"
            },
            "crash_day_of_week": {
                "range": (1, 7),
                "description": "Day of week (1=Monday, 7=Sunday)"
            },
            "crash_month": {
                "range": (1, 12),
                "description": "Month (1-12)"
            },
            "injuries_total": {
                "range": (0, float('inf')),
                "description": "Total injuries (non-negative)"
            },
            "injuries_fatal": {
                "range": (0, float('inf')),
                "description": "Fatal injuries (non-negative)"
            }
        }

        print("🔍 Checking domain-specific validation rules:")
        
        total_invalid = 0
        
        for col, rules in validation_rules.items():
            if col in self.df.columns:
                min_val, max_val = rules["range"]
                
                # Check range violations
                if max_val == float('inf'):
                    invalid_mask = self.df[col] < min_val
                else:
                    invalid_mask = (self.df[col] < min_val) | (self.df[col] > max_val)
                
                invalid_count = invalid_mask.sum()
                total_invalid += invalid_count
                
                status = "✅ PASS" if invalid_count == 0 else "❌ FAIL"
                print(f"   {col}: {status}")
                print(f"      Rule: {rules['description']}")
                print(f"      Invalid values: {invalid_count:,}")
                
                if invalid_count > 0:
                    invalid_values = self.df[invalid_mask][col].unique()
                    print(f"      Invalid examples: {invalid_values[:5]}")
                
                validation_results.append({
                    "Feature": col,
                    "Rule": rules["description"],
                    "Invalid_Count": invalid_count,
                    "Status": "PASS" if invalid_count == 0 else "FAIL"
                })

        # Additional logical validations
        print(f"\n🧠 Logical Consistency Checks:")
        
        # Check if total injuries >= sum of injury categories
        injury_cols = [col for col in ["injuries_fatal", "injuries_incapacitating", 
                      "injuries_non_incapacitating", "injuries_reported_not_evident"] 
                      if col in self.df.columns]
        
        if "injuries_total" in self.df.columns and len(injury_cols) > 1:
            injury_sum = self.df[injury_cols].sum(axis=1)
            inconsistent_injuries = (self.df["injuries_total"] < injury_sum).sum()
            
            status = "✅ PASS" if inconsistent_injuries == 0 else "❌ FAIL"
            print(f"   Injury totals consistency: {status}")
            print(f"      Inconsistent records: {inconsistent_injuries:,}")
            
            validation_results.append({
                "Feature": "injuries_total_consistency",
                "Rule": "Total >= sum of categories",
                "Invalid_Count": inconsistent_injuries,
                "Status": "PASS" if inconsistent_injuries == 0 else "FAIL"
            })

        # Summary
        print(f"\n📋 Validation Summary:")
        print(f"   Total validation rules: {len(validation_results)}")
        failed_rules = [r for r in validation_results if r["Status"] == "FAIL"]
        print(f"   Failed rules: {len(failed_rules)}")
        print(f"   Total invalid records: {total_invalid:,}")
        
        if len(failed_rules) == 0:
            print("   🎉 All validation checks passed!")
        else:
            print("   ⚠️ Some validation issues found - review data quality")
        
        # Store insight
        if total_invalid > 0:
            self.insights.append(f"Found {total_invalid:,} domain validation issues")
        else:
            self.insights.append("All domain validation checks passed")
        
        return validation_results

    # --------------------------------------------------
    # 11. Analysis Summary
    # --------------------------------------------------
    def generate_summary(self):
        """Generate a comprehensive analysis summary"""
        print("\n" + "="*60)
        print("📋 ANALYSIS SUMMARY & KEY INSIGHTS")
        print("="*60)
        
        if not self.insights:
            print("No insights generated. Run analysis methods first.")
            return
        
        print("🔍 Key Findings:")
        for i, insight in enumerate(self.insights, 1):
            print(f"   {i}. {insight}")
        
        # Data quality summary
        print(f"\n📊 Dataset Overview:")
        print(f"   • Total records: {self.df.shape[0]:,}")
        print(f"   • Features: {self.df.shape[1]}")
        print(f"   • Numeric features: {len(self.df.select_dtypes(include=[np.number]).columns)}")
        print(f"   • Categorical features: {len(self.df.select_dtypes(include=['object']).columns)}")
        
        # Missing data summary
        missing_pct = (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        print(f"   • Missing data: {missing_pct:.1f}%")
        
        # Recommendations
        print(f"\n💡 Recommendations for Next Steps:")
        
        recommendations = [
            "Proceed with data preprocessing based on identified patterns",
            "Address any domain validation issues before modeling",
            "Consider feature engineering based on temporal patterns",
            "Plan for class imbalance handling if detected",
            "Use correlation insights for feature selection"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        print(f"\n✅ EDA Analysis Complete!")
        
        return {
            'insights': self.insights,
            'dataset_shape': self.df.shape,
            'missing_data_pct': missing_pct
        }

    # --------------------------------------------------
    # 12. Complete Analysis Runner
    # --------------------------------------------------
    def run_complete_analysis(self, target="most_severe_injury"):
        """
        Run the complete EDA pipeline with all analyses
        """
        print("🚀 Starting Complete Traffic Accident EDA Analysis...")
        print("="*80)
        
        # Load and explore data
        if self.load_data() is None:
            print("❌ Failed to load data. Stopping analysis.")
            return None
        
        try:
            # Run all analysis steps
            self.initial_exploration()
            self.target_distribution(target)
            self.missing_value_analysis()
            self.univariate_analysis()
            self.bivariate_analysis(target)
            self.multivariate_analysis(target)
            self.correlation_analysis(target)
            self.outlier_detection()
            self.domain_validation()
            
            # Generate final summary
            summary = self.generate_summary()
            
            print("\n" + "="*80)
            print("✅ COMPLETE EDA ANALYSIS FINISHED SUCCESSFULLY!")
            print("="*80)
            
            return summary
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            print("Partial results may be available in self.insights")
            return None
