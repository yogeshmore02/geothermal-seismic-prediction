import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from predictor_3day import predict_warning  # your inference function

# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Balmatt Seismic Warning Dashboard",
    layout="wide",
)

st.title("🌋 Balmatt Seismic Warning Dashboard")
st.caption("Hybrid 3-Day Window Model (48-hour warning)")

st.markdown("---")

# ============================================================
# CONFIG - DEFAULT DATA PATH
# ============================================================
DEFAULT_OPERATIONS_CSV = "operational_metrics.csv"

# ============================================================
# HELPERS TO LOAD MODEL + DATA
# ============================================================

@st.cache_resource
def load_model():
    """Load trained hybrid 3-day model from disk."""
    with open("trained_models/hybrid_3day.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data


@st.cache_data
def load_raw_data():
    """
    Load operational + seismic data and build daily features + labels,
    using the same logic as in train_hybrid_3day.py.
    """
    try:
        seismic_df = pd.read_csv("seismic_events_1.csv")
        operations_df = pd.read_csv(DEFAULT_OPERATIONS_CSV)
    except FileNotFoundError:
        return None, None, None, None, None

    # Parse dates - operational uses DD/MM/YYYY format, seismic uses ISO format
    operations_df["recorded_at"] = pd.to_datetime(operations_df["recorded_at"], format='mixed', dayfirst=True)
    seismic_df["occurred_at"] = pd.to_datetime(seismic_df["occurred_at"])  # ISO format, no dayfirst needed
    operations_df = operations_df.sort_values("recorded_at").reset_index(drop=True)
    operations_df = operations_df.ffill().bfill()

    # ----- remove shutdown periods (same as training) -----
    shutdown_periods = [
        ("2019-07-01", "2021-04-30"),
        ("2022-12-01", "2023-03-31"),
        ("2023-10-01", "2023-11-30"),
        ("2025-07-01", "2025-09-30"),
    ]

    for start, end in shutdown_periods:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        operations_df = operations_df[
            ~((operations_df["recorded_at"] >= start) &
              (operations_df["recorded_at"] <= end))
        ]

        seismic_df = seismic_df[
            ~((seismic_df["occurred_at"] >= start) &
              (seismic_df["occurred_at"] <= end))
        ]

    operations_df = operations_df.reset_index(drop=True)
    seismic_df = seismic_df.reset_index(drop=True)

    # ----- aggregate to daily features (same as training) -----
    operations_df["date"] = operations_df["recorded_at"].dt.date

    daily = operations_df.groupby("date").agg({
        "inj_flow": ["mean", "max", "std"],
        "inj_whp": ["mean", "max", "std"],
        "prod_flow": ["mean", "max", "std"],
        "prod_whp": ["mean", "max", "std"],
        "inj_temp": ["mean", "max"],
        "prod_temp": ["mean", "max"],
        "is_producing": "mean",
    }).reset_index()

    daily.columns = ["date"] + [f"{a}_{b}" for a, b in daily.columns[1:]]
    daily["date"] = pd.to_datetime(daily["date"])

    # derived features
    daily["pressure_diff"] = daily["inj_whp_mean"] - daily["prod_whp_mean"]
    daily["flow_imbalance"] = daily["inj_flow_mean"] - daily["prod_flow_mean"]
    daily["temp_diff"] = daily["prod_temp_mean"] - daily["inj_temp_mean"]

    # rolling 7-day stats
    for col in ["inj_whp_max", "flow_imbalance", "pressure_diff"]:
        daily[f"{col}_7d_max"] = daily[col].rolling(7, min_periods=1).max()
        daily[f"{col}_7d_mean"] = daily[col].rolling(7, min_periods=1).mean()

    # rate of change
    daily["inj_whp_change"] = daily["inj_whp_mean"].diff()
    daily["inj_flow_change"] = daily["inj_flow_mean"].diff()

    daily = daily.fillna(0)

    # ----- labels: event in next 48h (same idea as training) -----
    seismic_df["date"] = seismic_df["occurred_at"].dt.date
    significant = seismic_df[seismic_df["magnitude"] > 0.5]

    daily["has_event"] = 0
    for _, ev in significant.iterrows():
        ev_date = pd.Timestamp(ev["date"])
        for d in range(0, 3):  # event day, day-1, day-2
            target = ev_date - pd.Timedelta(days=d)
            daily.loc[daily["date"] == target, "has_event"] = 1

    # ----- build 3-day windows -----
    label_col = "has_event"
    feature_cols = [c for c in daily.columns if c not in ["date", label_col]]
    X, y = [], []

    for i in range(2, len(daily)):
        window = []
        for j in range(i - 2, i + 1):  # i-2, i-1, i
            window.extend(daily.iloc[j][feature_cols].values)
        X.append(window)
        y.append(daily.iloc[i][label_col])

    X = np.array(X)
    y = np.array(y)

    return operations_df, seismic_df, daily, (X, y), feature_cols


def parse_dates_flexible(df, col='recorded_at'):
    """Parse dates flexibly, handling multiple formats."""
    df[col] = pd.to_datetime(df[col], format='mixed', dayfirst=True)
    return df


def get_available_dates_from_csv(csv_source):
    """Extract available dates from a CSV for the date picker.
    
    Note: No caching because this handles file objects which can't be cached.
    """
    try:
        if isinstance(csv_source, str):
            df = pd.read_csv(csv_source)
        else:
            # For uploaded files, read and reset pointer
            df = pd.read_csv(csv_source)
            csv_source.seek(0)
        
        df = parse_dates_flexible(df, 'recorded_at')
        df['date'] = df['recorded_at'].dt.date
        dates = sorted(df['date'].unique())
        
        # Need at least 3 days for prediction
        if len(dates) >= 3:
            return dates[2:]  # Can only predict from day 3 onwards
        return []
    except Exception as e:
        st.error(f"Error reading dates: {e}")
        return []


def compute_metrics(model_data, X, y):
    """Compute Accuracy / Precision / Recall / F1 using existing model."""
    if X is None or len(X) == 0:
        return None

    model = model_data["model"]
    scaler = model_data["scaler"]

    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (proba >= 0.3).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    return acc, prec, rec, f1


def aggregated_feature_importance(model_data, base_feature_cols, n_days=3):
    """
    Sum importances across the 3 days for each base feature.
    model.feature_importances_ has length len(base_feature_cols) * 3.
    """
    model = model_data["model"]
    raw_importance = model.feature_importances_

    per_feature = np.zeros(len(base_feature_cols))
    for day in range(n_days):
        start = day * len(base_feature_cols)
        end = (day + 1) * len(base_feature_cols)
        per_feature += raw_importance[start:end]

    fi_df = pd.DataFrame({"feature": base_feature_cols, "importance": per_feature})
    fi_df = fi_df.sort_values("importance", ascending=False)
    return fi_df


# ============================================================
# LOAD EVERYTHING
# ============================================================
model_data = load_model()
ops_df, seis_df, daily_df, XY_tuple, feature_cols_daily = load_raw_data()
X_all, y_all = XY_tuple if XY_tuple is not None else (None, None)

# ============================================================
# LAYOUT: TABS
# ============================================================
tab_overview, tab_metrics, tab_importance, tab_yearly, tab_dynamic, tab_predict = st.tabs(
    [
        "🏠 Overview",
        "📊 Model Metrics",
        "🔥 Feature Importance",
        "📅 Yearly Seismicity",
        "📈 Dynamic Plots",
        "🔮 Predict From CSV",
    ]
)

# ------------------------------------------------------------
# OVERVIEW TAB
# ------------------------------------------------------------
with tab_overview:
    st.subheader("System Summary")

    st.write(
        """
        This dashboard visualizes the **Hybrid 3-Day Seismic Warning Model**:
        - Uses sliding 3-day windows of operational data to predict whether a
          significant event (M > 0.5) will occur within the next 48 hours.
        - Model: Random Forest classifier trained offline.
        - This UI **only loads the trained model** and runs analytics & predictions;
          it does **not retrain** anything.
        """
    )

    st.markdown("**Model info:**")
    st.write(f"- Training timestamp: `{model_data.get('training_date', 'N/A')}`")
    st.write(f"- Number of base features per day: `{len(feature_cols_daily)}`")
    st.write(f"- Total features per 3-day window: `{model_data['model'].n_features_in_}`")

# ------------------------------------------------------------
# METRICS TAB
# ------------------------------------------------------------
with tab_metrics:
    st.subheader("📊 Model Performance")

    if X_all is None:
        st.error("Could not load data to compute metrics. Check CSV paths.")
    else:
        # Cross-validation metrics (from training - these are the real metrics)
        st.markdown("### Cross-Validation Metrics (5-Fold)")
        st.caption("These are the real performance metrics from training, tested on held-out data.")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Precision", "0.585", help="Of all warnings raised, how many were correct")
        col2.metric("Recall", "0.731", help="Of all actual events, how many were predicted")
        col3.metric("F1 Score", "0.650", help="Harmonic mean of precision and recall")
        
        # Show per-fold breakdown
        with st.expander("View per-fold results"):
            fold_data = pd.DataFrame({
                'Fold': [1, 2, 3, 4, 5],
                'Precision': [0.500, 0.500, 0.462, 0.727, 0.769],
                'Recall': [0.727, 0.545, 0.600, 0.800, 1.000],
                'F1 Score': [0.593, 0.522, 0.522, 0.762, 0.870]
            })
            st.dataframe(fold_data, use_container_width=True, hide_index=True)
            
            fig_folds = px.bar(
                fold_data.melt(id_vars='Fold', var_name='Metric', value_name='Score'),
                x='Fold',
                y='Score',
                color='Metric',
                barmode='group',
                title='Performance by Fold'
            )
            fig_folds.update_layout(height=350)
            st.plotly_chart(fig_folds, use_container_width=True)

        st.markdown("---")
        
        # Model configuration
        st.markdown("### Model Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Details**")
            st.write(f"- Model type: Random Forest Classifier")
            st.write(f"- Number of trees: {model_data['model'].n_estimators}")
            st.write(f"- Max depth: {model_data['model'].max_depth}")
            st.write(f"- Training date: {model_data.get('training_date', 'N/A')}")
        
        with col2:
            st.markdown("**Feature Engineering**")
            st.write(f"- Base features per day: {len(feature_cols_daily)}")
            st.write(f"- Window size: 3 days")
            st.write(f"- Total features: {model_data['model'].n_features_in_}")
            st.write(f"- Warning threshold: 0.3 probability")

        st.markdown("---")
        
        # Dataset info
        st.markdown("### Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        total_samples = len(y_all)
        positive_samples = int(y_all.sum())
        negative_samples = total_samples - positive_samples
        
        col1.metric("Total Samples", f"{total_samples:,}")
        col2.metric("Event Days", f"{positive_samples:,}", help="Days labeled as having significant seismic activity within 48h")
        col3.metric("Non-Event Days", f"{negative_samples:,}")
        col4.metric("Event Rate", f"{positive_samples/total_samples*100:.1f}%")
        
        # Class balance bar
        st.markdown("**Class Distribution**")
        class_df = pd.DataFrame({
            'Class': ['No Event (0)', 'Event (1)'],
            'Count': [negative_samples, positive_samples],
            'Percentage': [negative_samples/total_samples*100, positive_samples/total_samples*100]
        })
        
        fig_class = px.bar(
            class_df, 
            x='Class', 
            y='Count',
            color='Class',
            text=class_df['Percentage'].apply(lambda x: f'{x:.1f}%'),
            color_discrete_map={'No Event (0)': '#4CAF50', 'Event (1)': '#f44336'}
        )
        fig_class.update_traces(textposition='outside')
        fig_class.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_class, use_container_width=True)
        
        st.markdown("---")
        
        # Resubstitution metrics (for reference)
        with st.expander("View Resubstitution Metrics (Training Data)"):
            st.warning("⚠️ These metrics are computed on the same data used for training and are overly optimistic. Use cross-validation metrics above for real performance.")
            
            metrics = compute_metrics(model_data, X_all, y_all)
            if metrics:
                acc, prec, rec, f1 = metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{acc:.3f}")
                col2.metric("Precision", f"{prec:.3f}")
                col3.metric("Recall", f"{rec:.3f}")
                col4.metric("F1 Score", f"{f1:.3f}")
            
            # Confusion matrix
            st.markdown("**Confusion Matrix (Resubstitution)**")
            
            X_scaled = model_data["scaler"].transform(X_all)
            proba = model_data["model"].predict_proba(X_scaled)[:, 1]
            y_pred = (proba >= 0.3).astype(int)
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_all, y_pred)
            
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['No Event', 'Event'],
                y=['No Event', 'Event'],
                color_continuous_scale='Blues',
                text_auto=True
            )
            fig_cm.update_layout(height=300)
            st.plotly_chart(fig_cm, use_container_width=True)

# ------------------------------------------------------------
# FEATURE IMPORTANCE TAB
# ------------------------------------------------------------
with tab_importance:
    st.subheader("🔥 Feature Importance (3-Day Aggregated)")

    if feature_cols_daily is None:
        st.error("Feature information not available.")
    else:
        fi_df = aggregated_feature_importance(model_data, feature_cols_daily, n_days=3)

        st.dataframe(fi_df.head(20), use_container_width=True)

        fig = px.bar(
            fi_df.head(20),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 20 Aggregated Feature Importances",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# YEARLY SEISMICITY TAB
# ------------------------------------------------------------
with tab_yearly:
    st.subheader("📅 Yearly Seismic Event Counts")

    if seis_df is None:
        st.error("Could not load seismic_events_1.csv")
    else:
        seis = seis_df.copy()
        seis["year"] = seis["occurred_at"].dt.year
        yearly = seis.groupby("year").size().reset_index(name="event_count")

        st.dataframe(yearly, use_container_width=True)

        fig = px.bar(
            yearly,
            x="year",
            y="event_count",
            title="Number of Seismic Events per Year",
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# DYNAMIC PLOTS TAB
# ------------------------------------------------------------
with tab_dynamic:
    st.subheader("📈 Dynamic Operational Key Plots")

    if ops_df is None:
        st.error("Could not load operational_metrics.csv")
    else:
        # choose numeric columns only
        numeric_cols = [
            c for c in ops_df.columns
            if np.issubdtype(ops_df[c].dtype, np.number)
        ]

        time_col = "recorded_at"
        if time_col not in ops_df.columns:
            st.error("Column 'recorded_at' not found in operational_metrics.csv")
        else:
            selected_key = st.selectbox(
                "Select variable to visualise:",
                numeric_cols,
                index=0 if "inj_flow" in numeric_cols else 0,
            )

            fig = px.line(
                ops_df,
                x=time_col,
                y=selected_key,
                title=f"{selected_key} over time",
                render_mode="svg"
            )
            st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# PREDICTION TAB
# ------------------------------------------------------------
with tab_predict:
    st.subheader("🔮 Predict Seismic Risk")

    st.write(
        """
        Upload a **new operational_metrics-style CSV** (same columns as training),
        or use the **default operational_metrics.csv** file.
        Then select the **date you want the prediction for** — this determines
        which 3-day window the model evaluates.
        """
    )

    # File uploader (optional)
    uploaded_csv = st.file_uploader(
        "Upload operational CSV for prediction (optional - leave empty to use default)",
        type=["csv"],
        key="prediction_csv",
    )

    # Determine which data source to use
    if uploaded_csv is not None:
        data_source = uploaded_csv
        data_source_name = uploaded_csv.name
        st.success(f"📁 Using uploaded file: **{data_source_name}**")
    else:
        data_source = DEFAULT_OPERATIONS_CSV
        data_source_name = "operational_metrics.csv (default)"
        st.info(f"📁 Using default file: **{data_source_name}**")

    # Get available dates from the selected data source
    available_dates = get_available_dates_from_csv(data_source)
    
    # Reset file pointer if it's an uploaded file
    if uploaded_csv is not None:
        uploaded_csv.seek(0)

    if not available_dates:
        st.error("Could not extract dates from the data source, or not enough days (need at least 3).")
    else:
        st.write(f"**Available date range:** {available_dates[0]} to {available_dates[-1]}")
        
        # Date selector with available dates
        selected_date = st.date_input(
            "Select the date you want the prediction for (end of 3-day window):",
            value=available_dates[-1],  # Default to most recent date
            min_value=available_dates[0],
            max_value=available_dates[-1],
        )

        if st.button("Run Prediction", type="primary"):
            with st.spinner("Running prediction..."):
                # Reset file pointer again before prediction
                if uploaded_csv is not None:
                    uploaded_csv.seek(0)
                
                # Run the model
                result = predict_warning(data_source, target_date=selected_date)

                # ------------------------
                # HANDLE ERRORS CLEANLY
                # ------------------------
                if "error" in result and result["error"] is not None:
                    st.error(f"❌ Prediction failed: {result['error']}")
                    st.stop()

                # ------------------------
                # SAFE DATE HANDLING
                # ------------------------
                pred_date = result["date"]
                if hasattr(pred_date, "date"):
                    pred_date = pred_date.date()
                else:
                    pred_date = pd.to_datetime(pred_date).date()

                # ------------------------
                # DISPLAY RESULTS
                # ------------------------
                st.markdown("---")
                st.markdown(f"### Results for **{pred_date}**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Event Probability (next 48h)", 
                        f"{result['event_probability']:.3f}",
                        help="Probability of a significant seismic event (M > 0.5) in the next 48 hours"
                    )

                with col2:
                    if result["warning_flag"] == 2:
                        st.error("⚠️ **WARNING**: Severe Elevated seismic risk detected!")
                    elif result["warning_flag"] == 1:
                        st.error("⚠️ **WARNING**: Seismic risk detected")
                    else:
                        st.success("✅ **LOW RISK**: No warning flag raised.")

                # Show raw result
                with st.expander("View raw prediction result"):
                    st.json({
                        "date": str(pred_date),
                        "event_probability": result["event_probability"],
                        "warning_flag": result["warning_flag"]
                    })
