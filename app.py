"""
Streamlit application for the Energy‑to‑Tonnage Optimiser demo.

This app demonstrates how machine learning can be used to predict
grinding circuit performance (throughput and specific energy) from
process variables. It uses a simulated dataset generated to mimic
typical grinding circuit behaviour and provides an end‑to‑end workflow:

1. **Data exploration** – inspect the dataset, summarise statistics and explore
   correlations between variables.
2. **Model training and evaluation** – choose a regression model,
   train it on the data, and assess performance using standard metrics
   and plots.
3. **Interactive prediction** – input hypothetical operating conditions
   and obtain predicted throughput and specific energy consumption.
4. **Operating window analysis** – compute recommended parameter ranges
   from the top‑performing rows in the dataset and visualise them.

The app is designed to be self‑contained and easy to extend. It
demonstrates the potential of AI to guide process optimisation in
mineral processing plants, supporting energy‑efficient and stable
operations.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load the synthetic dataset from a CSV file."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def compute_model_metrics(y_true, y_pred) -> dict:
    """Compute regression performance metrics and return a dictionary."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"R²": r2, "RMSE": rmse, "MAE": mae}


def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot a correlation heatmap for numerical features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)


def plot_pred_vs_actual(y_true, y_pred, metric_name: str = "Throughput"):
    """Plot predicted vs. actual values for a regression problem."""
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='red')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Measured vs Predicted ({metric_name})")
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Energy‑to‑Tonnage Optimiser", layout="wide")
    st.title("Energy‑to‑Tonnage Optimiser")
    st.markdown(
        "This demonstration shows how machine learning can model and optimise a grinding circuit.\n"
        "You can explore the data, train predictive models, input new operating conditions, and see recommended operating windows."
    )

    # Sidebar navigation
    section = st.sidebar.radio(
        "Navigation",
        [
            "Data Exploration",
            "Model Training & Evaluation",
            "Interactive Prediction",
            "Operating Window Analysis",
        ],
    )

    # Load dataset
    df = load_data("synthetic_data.csv")

    # Define feature and target lists
    feature_cols = [
        "mill_power_kW",
        "feed_rate_tph",
        "pct_solids",
        "sump_level_percent",
        "cyclone_pressure_kPa",
        "pump_speed_rpm",
        "density_kg_m3",
    ]
    target_cols = ["throughput_tph", "specific_energy_kWh_per_t"]

    # Data Exploration Section
    if section == "Data Exploration":
        st.header("📊 Data Exploration")
        st.write(
            "Below is a preview of the synthetic dataset used for this demo. Each row represents"
            " a snapshot of a grinding circuit with recorded process variables and calculated outputs."
        )

        # Show data preview
        st.dataframe(df.head(15))

        # Summary statistics
        if st.checkbox("Show summary statistics"):
            st.subheader("Summary Statistics")
            st.write(df.describe())

        # Correlation heatmap
        if st.checkbox("Show correlation heatmap"):
            plot_correlation_heatmap(df)

    # Model Training & Evaluation Section
    elif section == "Model Training & Evaluation":
        st.header("🤖 Model Training & Evaluation")
        st.write(
            "Select a target variable and model type below. The app will train the model on"
            " a split of the dataset and display performance metrics and a parity plot."
        )

        # Choose target variable
        target = st.selectbox("Target to Predict", target_cols)

        # Choose model type
        model_name = st.selectbox("Model", ["Random Forest", "Gradient Boosting"])

        # Train/test split proportion
        test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0

        # Button to train model
        if st.button("Train Model"):
            X = df[feature_cols]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Build pipeline
            if model_name == "Random Forest":
                model = RandomForestRegressor(n_estimators=200, random_state=42)
            else:
                model = GradientBoostingRegressor(random_state=42)

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model),
            ])

            # Fit model
            pipeline.fit(X_train, y_train)

            # Predict on test set
            y_pred = pipeline.predict(X_test)

            # Compute metrics
            metrics = compute_model_metrics(y_test, y_pred)

            st.subheader("Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("R²", f"{metrics['R²']:.3f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col3.metric("MAE", f"{metrics['MAE']:.2f}")

            # Plot predicted vs actual
            st.subheader("Parity Plot")
            plot_pred_vs_actual(y_test, y_pred, metric_name=target.replace("_", " "))

            # Feature importance (RandomForest and GradientBoosting provide feature importances)
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": importances,
                }).sort_values(by="Importance", ascending=False)
                st.subheader("Feature Importance")
                st.bar_chart(importance_df.set_index("Feature"))
            else:
                st.info("Feature importances are not available for the selected model.")

    # Interactive Prediction Section
    elif section == "Interactive Prediction":
        st.header("🔮 Interactive Prediction")
        st.write(
            "Use the sliders and inputs below to define hypothetical operating conditions."
            " The model will predict throughput (tph) and specific energy (kWh/t)."
        )

        # Load default model (RandomForest) trained on throughput and specific energy simultaneously
        # We'll train two separate models for throughput and specific energy to simplify the interface
        X = df[feature_cols]
        y_throughput = df["throughput_tph"]
        y_energy = df["specific_energy_kWh_per_t"]

        # Split data (80/20) for default models
        X_train, X_test, y_train_t, y_test_t = train_test_split(X, y_throughput, test_size=0.2, random_state=42)
        X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X, y_energy, test_size=0.2, random_state=42)

        # Train models (RandomForest for throughput, GradientBoosting for energy consumption)
        model_throughput = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
        ])
        model_energy = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(random_state=42)),
        ])

        model_throughput.fit(X_train, y_train_t)
        model_energy.fit(X_train_e, y_train_e)

        # Create slider inputs
        st.subheader("Input Operating Conditions")
        col1, col2, col3 = st.columns(3)
        with col1:
            mill_power_in = st.slider("Mill Power (kW)", float(df["mill_power_kW"].min()), float(df["mill_power_kW"].max()), float(df["mill_power_kW"].mean()))
            feed_rate_in = st.slider("Feed Rate (tph)", float(df["feed_rate_tph"].min()), float(df["feed_rate_tph"].max()), float(df["feed_rate_tph"].mean()))
            pct_solids_in = st.slider("% Solids", float(df["pct_solids"].min()), float(df["pct_solids"].max()), float(df["pct_solids"].mean()))
        with col2:
            sump_level_in = st.slider("Sump Level (%)", float(df["sump_level_percent"].min()), float(df["sump_level_percent"].max()), float(df["sump_level_percent"].mean()))
            cyclone_pressure_in = st.slider("Cyclone Pressure (kPa)", float(df["cyclone_pressure_kPa"].min()), float(df["cyclone_pressure_kPa"].max()), float(df["cyclone_pressure_kPa"].mean()))
            pump_speed_in = st.slider("Pump Speed (rpm)", float(df["pump_speed_rpm"].min()), float(df["pump_speed_rpm"].max()), float(df["pump_speed_rpm"].mean()))
        with col3:
            density_in = st.slider("Density (kg/m³)", float(df["density_kg_m3"].min()), float(df["density_kg_m3"].max()), float(df["density_kg_m3"].mean()))

        # Prepare input data
        input_df = pd.DataFrame({
            "mill_power_kW": [mill_power_in],
            "feed_rate_tph": [feed_rate_in],
            "pct_solids": [pct_solids_in],
            "sump_level_percent": [sump_level_in],
            "cyclone_pressure_kPa": [cyclone_pressure_in],
            "pump_speed_rpm": [pump_speed_in],
            "density_kg_m3": [density_in],
        })

        # Predict
        throughput_pred = model_throughput.predict(input_df)[0]
        energy_pred = model_energy.predict(input_df)[0]

        st.subheader("Predicted Performance")
        colA, colB = st.columns(2)
        colA.metric("Predicted Throughput (tph)", f"{throughput_pred:.2f}")
        colB.metric("Predicted Specific Energy (kWh/t)", f"{energy_pred:.2f}")

    # Operating Window Analysis Section
    else:
        st.header("📈 Operating Window Analysis")
        st.write(
            "This section identifies a high‑performing subset of the data and summarises"
            " recommended ranges for each process variable. Rows with the highest performance"
            " (throughput relative to energy consumption) form the basis for these recommendations."
        )

        # Compute performance score (throughput divided by specific energy)
        df["performance_score"] = df["throughput_tph"] / df["specific_energy_kWh_per_t"]
        top_fraction = st.slider("Select top x% high‑performing rows", min_value=5, max_value=30, value=10, step=5)
        n_top = int(len(df) * top_fraction / 100)
        top_df = df.nlargest(n_top, "performance_score")

        # Compute recommended ranges and means
        summary = []
        for col in feature_cols:
            col_min = top_df[col].min()
            col_max = top_df[col].max()
            col_mean = top_df[col].mean()
            summary.append({"Variable": col, "Min": col_min, "Max": col_max, "Mean": col_mean})

        summary_df = pd.DataFrame(summary).set_index("Variable")

        st.subheader("Recommended Operating Ranges")
        st.write(
            "These ranges are calculated from the top-performing rows. Operating within"
            " these bounds is likely to yield high throughput and low specific energy."
        )
        # Format the table nicely
        st.dataframe(summary_df.style.format({"Min": "{:.2f}", "Max": "{:.2f}", "Mean": "{:.2f}"}))

        # Visualise distributions for a selected variable
        selected_var = st.selectbox(
            "Select variable for distribution comparison",
            feature_cols,
        )
        fig, ax = plt.subplots()
        sns.kdeplot(df[selected_var], label="All data", fill=True, alpha=0.3, color="skyblue", ax=ax)
        sns.kdeplot(top_df[selected_var], label="High performance", fill=True, alpha=0.5, color="orange", ax=ax)
        ax.set_title(f"Distribution of {selected_var}")
        ax.legend()
        st.pyplot(fig)


if __name__ == "__main__":
    main()
