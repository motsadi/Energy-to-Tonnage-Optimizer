"""
Streamlit application for the Energy‑to‑Tonnage Optimiser demo.

This app demonstrates how machine learning can be used to predict
grinding circuit performance (throughput and specific energy) from
process variables. It uses a synthetic dataset generated to mimic
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
5. **Inverse design and solution surfaces** – work backwards from desired targets
   to find suitable process inputs, and visualise 3D response surfaces.

The app is designed to be self‑contained and easy to extend. It
demonstrates the potential of AI to guide process optimisation in
mineral processing plants, supporting energy‑efficient and stable
operations.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


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
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', color='red')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Measured vs Predicted ({metric_name})")
    st.pyplot(fig)


def inverse_design(
    model_t: Pipeline,
    model_e: Pipeline,
    target_t: float,
    target_e: float,
    bounds: list,
    init: np.ndarray,
    lr: float = 0.1,
    max_iters: int = 100,
    tol: float = 1e-3,
) -> tuple:
    """Simple gradient descent to approximate inverse design.

    Minimises squared error between predicted throughput/energy and targets.
    """
    x = init.copy()
    for i in range(max_iters):
        pred_t = model_t.predict(pd.DataFrame([x], columns=[c for c, _ in bounds]))[0]
        pred_e = model_e.predict(pd.DataFrame([x], columns=[c for c, _ in bounds]))[0]
        loss = (pred_t - target_t) ** 2 + (pred_e - target_e) ** 2
        grad = np.zeros_like(x)
        for j in range(len(x)):
            eps = 1e-4 * (bounds[j][1] - bounds[j][0])
            dx = x.copy()
            dx[j] += eps
            p_t = model_t.predict(pd.DataFrame([dx], columns=[c for c, _ in bounds]))[0]
            p_e = model_e.predict(pd.DataFrame([dx], columns=[c for c, _ in bounds]))[0]
            loss_dx = (p_t - target_t) ** 2 + (p_e - target_e) ** 2
            grad[j] = (loss_dx - loss) / eps
        if np.linalg.norm(grad) < tol:
            break
        x -= lr * grad
        for j in range(len(x)):
            x[j] = np.clip(x[j], bounds[j][0], bounds[j][1])
    final_t = model_t.predict(pd.DataFrame([x], columns=[c for c, _ in bounds]))[0]
    final_e = model_e.predict(pd.DataFrame([x], columns=[c for c, _ in bounds]))[0]
    return x, final_t, final_e


def main():
    st.set_page_config(page_title="Energy‑to‑Tonnage Optimiser", layout="wide")
    st.title("Energy‑to‑Tonnage Optimiser")
    st.markdown(
        "This demonstration shows how machine learning can model and optimise a grinding circuit.\n"
        "You can explore the data, train predictive models, input new operating conditions, "
        "see recommended operating windows, work backwards from desired targets, and view 3D solution surfaces."
    )

    section = st.sidebar.radio(
        "Navigation",
        [
            "Data Exploration",
            "Model Training & Evaluation",
            "Interactive Prediction",
            "Operating Window Analysis",
            "Inverse Design & Solution Surfaces",
        ],
    )

    df = load_data("synthetic_data.csv")
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

    if section == "Data Exploration":
        st.header("📊 Data Exploration")
        st.write(
            "Below is a preview of the synthetic dataset used for this demo. "
            "Each row represents a snapshot of a grinding circuit with recorded process variables and calculated outputs."
        )
        st.dataframe(df.head(15))
        if st.checkbox("Show summary statistics"):
            st.subheader("Summary Statistics")
            st.write(df.describe())
        if st.checkbox("Show correlation heatmap"):
            plot_correlation_heatmap(df)

    elif section == "Model Training & Evaluation":
        st.header("🤖 Model Training & Evaluation")
        st.write(
            "Select a target variable and model type below. The app will train the model on "
            "a split of the dataset and display performance metrics and a parity plot."
        )

        target = st.selectbox("Target to Predict", target_cols)
        model_name = st.selectbox("Model", ["Random Forest", "Gradient Boosting"])
        test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0

        if st.button("Train Model"):
            X = df[feature_cols]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            if model_name == "Random Forest":
                model = RandomForestRegressor(n_estimators=200, random_state=42)
            else:
                model = GradientBoostingRegressor(random_state=42)
            pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            metrics = compute_model_metrics(y_test, y_pred)
            st.subheader("Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("R²", f"{metrics['R²']:.3f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col3.metric("MAE", f"{metrics['MAE']:.2f}")
            st.subheader("Parity Plot")
            plot_pred_vs_actual(y_test, y_pred, metric_name=target.replace("_", " "))
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                importance_df = pd.DataFrame(
                    {"Feature": feature_cols, "Importance": importances}
                ).sort_values(by="Importance", ascending=False)
                st.subheader("Feature Importance")
                st.bar_chart(importance_df.set_index("Feature"))
            else:
                st.info("Feature importances are not available for the selected model.")

    elif section == "Interactive Prediction":
        st.header("🔮 Interactive Prediction")
        st.write(
            "Use the sliders and inputs below to define hypothetical operating conditions. "
            "The model will predict throughput (tph) and specific energy (kWh/t)."
        )

        X = df[feature_cols]
        y_throughput = df["throughput_tph"]
        y_energy = df["specific_energy_kWh_per_t"]
        X_train, X_test, y_train_t, y_test_t = train_test_split(X, y_throughput, test_size=0.2, random_state=42)
        X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X, y_energy, test_size=0.2, random_state=42)
        model_throughput = Pipeline([("scaler", StandardScaler()),
                                     ("model", RandomForestRegressor(n_estimators=200, random_state=42))])
        model_energy = Pipeline([("scaler", StandardScaler()),
                                 ("model", GradientBoostingRegressor(random_state=42))])
        model_throughput.fit(X_train, y_train_t)
        model_energy.fit(X_train_e, y_train_e)

        st.subheader("Input Operating Conditions")
        col1, col2, col3 = st.columns(3)
        with col1:
            mill_power_in = st.slider(
                "Mill Power (kW)", float(df["mill_power_kW"].min()),
                float(df["mill_power_kW"].max()), float(df["mill_power_kW"].mean())
            )
            feed_rate_in = st.slider(
                "Feed Rate (tph)", float(df["feed_rate_tph"].min()),
                float(df["feed_rate_tph"].max()), float(df["feed_rate_tph"].mean())
            )
            pct_solids_in = st.slider(
                "% Solids", float(df["pct_solids"].min()), float(df["pct_solids"].max()),
                float(df["pct_solids"].mean())
            )
        with col2:
            sump_level_in = st.slider(
                "Sump Level (%)", float(df["sump_level_percent"].min()),
                float(df["sump_level_percent"].max()), float(df["sump_level_percent"].mean())
            )
            cyclone_pressure_in = st.slider(
                "Cyclone Pressure (kPa)", float(df["cyclone_pressure_kPa"].min()),
                float(df["cyclone_pressure_kPa"].max()), float(df["cyclone_pressure_kPa"].mean())
            )
            pump_speed_in = st.slider(
                "Pump Speed (rpm)", float(df["pump_speed_rpm"].min()),
                float(df["pump_speed_rpm"].max()), float(df["pump_speed_rpm"].mean())
            )
        with col3:
            density_in = st.slider(
                "Density (kg/m³)", float(df["density_kg_m3"].min()),
                float(df["density_kg_m3"].max()), float(df["density_kg_m3"].mean())
            )

        input_df = pd.DataFrame({
            "mill_power_kW": [mill_power_in],
            "feed_rate_tph": [feed_rate_in],
            "pct_solids": [pct_solids_in],
            "sump_level_percent": [sump_level_in],
            "cyclone_pressure_kPa": [cyclone_pressure_in],
            "pump_speed_rpm": [pump_speed_in],
            "density_kg_m3": [density_in],
        })
        throughput_pred = model_throughput.predict(input_df)[0]
        energy_pred = model_energy.predict(input_df)[0]
        st.subheader("Predicted Performance")
        colA, colB = st.columns(2)
        colA.metric("Predicted Throughput (tph)", f"{throughput_pred:.2f}")
        colB.metric("Predicted Specific Energy (kWh/t)", f"{energy_pred:.2f}")

    elif section == "Operating Window Analysis":
        st.header("📈 Operating Window Analysis")
        st.write(
            "This section identifies a high‑performing subset of the data and summarises "
            "recommended ranges for each process variable. Rows with the highest performance "
            "(throughput relative to energy consumption) form the basis for these recommendations."
        )
        df["performance_score"] = df["throughput_tph"] / df["specific_energy_kWh_per_t"]
        top_fraction = st.slider(
            "Select top x% high‑performing rows", min_value=5, max_value=30, value=10, step=5
        )
        n_top = int(len(df) * top_fraction / 100)
        top_df = df.nlargest(n_top, "performance_score")
        summary = []
        for col in feature_cols:
            summary.append({
                "Variable": col,
                "Min": top_df[col].min(),
                "Max": top_df[col].max(),
                "Mean": top_df[col].mean()
            })
        summary_df = pd.DataFrame(summary).set_index("Variable")
        st.subheader("Recommended Operating Ranges")
        st.write(
            "These ranges are calculated from the top-performing rows. "
            "Operating within these bounds is likely to yield high throughput and low specific energy."
        )
        st.dataframe(summary_df.style.format({"Min": "{:.2f}", "Max": "{:.2f}", "Mean": "{:.2f}"}))
        selected_var = st.selectbox("Select variable for distribution comparison", feature_cols)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.kdeplot(df[selected_var], label="All data", fill=True, alpha=0.3, color="skyblue", ax=ax)
        sns.kdeplot(top_df[selected_var], label="High performance", fill=True, alpha=0.5, color="orange", ax=ax)
        ax.set_title(f"Distribution of {selected_var}")
        ax.legend()
        st.pyplot(fig)

    else:  # Inverse Design & Solution Surfaces
        st.header("🔁 Inverse Design & Solution Surfaces")
        st.write(
            "This section lets you work backwards from a target output or optimise a performance metric. "
            "You can set desired throughput and specific energy goals and let the model suggest operating conditions. "
            "Additionally, explore 3D solution surfaces by varying two parameters and observing the predicted outputs."
        )

        # Train default models
        X = df[feature_cols]
        y_t = df["throughput_tph"]
        y_e = df["specific_energy_kWh_per_t"]
        Xt, Xv, yt, yv = train_test_split(X, y_t, test_size=0.2, random_state=42)
        Xt_e, Xv_e, yt_e, yv_e = train_test_split(X, y_e, test_size=0.2, random_state=42)
        model_t = Pipeline([("scaler", StandardScaler()),
                            ("model", RandomForestRegressor(n_estimators=200, random_state=42))])
        model_e = Pipeline([("scaler", StandardScaler()),
                            ("model", GradientBoostingRegressor(random_state=42))])
        model_t.fit(Xt, yt)
        model_e.fit(Xt_e, yt_e)

        st.subheader("Inverse Design (Goal Seeking)")
        st.write(
            "Provide desired throughput and specific energy values, and the optimiser will search "
            "for operating conditions that achieve these targets within the data range."
        )
        col_goal1, col_goal2 = st.columns(2)
        with col_goal1:
            target_t = st.number_input(
                "Desired Throughput (tph)",
                min_value=float(df["throughput_tph"].min()),
                max_value=float(df["throughput_tph"].max()),
                value=float(df["throughput_tph"].mean())
            )
        with col_goal2:
            target_e = st.number_input(
                "Desired Specific Energy (kWh/t)",
                min_value=float(df["specific_energy_kWh_per_t"].min()),
                max_value=float(df["specific_energy_kWh_per_t"].max()),
                value=float(df["specific_energy_kWh_per_t"].mean())
            )
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            lr = st.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        with col_opt2:
            iters = st.slider("Max Iterations", min_value=10, max_value=300, value=100, step=10)

        if st.button("Run Inverse Design"):
            bounds = [(df[col].min(), df[col].max()) for col in feature_cols]
            init_guess = np.array([df[col].mean() for col in feature_cols])
            opt_x, opt_t, opt_e = inverse_design(
                model_t, model_e, target_t, target_e, bounds, init_guess, lr=lr, max_iters=iters
            )
            result_df = pd.DataFrame({"Variable": feature_cols, "Suggested Value": opt_x}).set_index("Variable")
            st.success("Optimisation complete! Suggested operating conditions:")
            st.dataframe(result_df.style.format({"Suggested Value": "{:.2f}"}))
            st.write(
                f"Predicted Throughput: **{opt_t:.2f} tph**\n"
                f"Predicted Specific Energy: **{opt_e:.2f} kWh/t**"
            )

        st.markdown("---")

        st.subheader("3D Solution Surfaces")
        st.write(
            "Select two parameters to vary and visualise the predicted output values across "
            "the range of those parameters. All other variables are fixed at their mean values."
        )
        param_x = st.selectbox("Parameter for X‑axis", feature_cols, index=1)
        param_y = st.selectbox("Parameter for Y‑axis", feature_cols, index=2)
        metric = st.selectbox("Output metric to plot", ["Throughput", "Specific Energy", "Performance Score"])
        n_points = st.slider("Resolution (grid size)", min_value=10, max_value=50, value=25, step=5)
        if st.button("Generate Surface"):
            x_vals = np.linspace(df[param_x].min(), df[param_x].max(), n_points)
            y_vals = np.linspace(df[param_y].min(), df[param_y].max(), n_points)
            X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)
            base_input = [df[col].mean() for col in feature_cols]
            idx_x = feature_cols.index(param_x)
            idx_y = feature_cols.index(param_y)
            Z = np.zeros_like(X_mesh)
            for i in range(n_points):
                for j in range(n_points):
                    inp = base_input.copy()
                    inp[idx_x] = X_mesh[i, j]
                    inp[idx_y] = Y_mesh[i, j]
                    pred_t = model_t.predict(pd.DataFrame([inp], columns=feature_cols))[0]
                    pred_e = model_e.predict(pd.DataFrame([inp], columns=feature_cols))[0]
                    if metric == "Throughput":
                        Z[i, j] = pred_t
                    elif metric == "Specific Energy":
                        Z[i, j] = pred_e
                    else:
                        Z[i, j] = pred_t / pred_e
            surface = go.Surface(x=X_mesh, y=Y_mesh, z=Z, colorscale="Viridis", showscale=True)
            fig = go.Figure(data=[surface])
            fig.update_layout(
                title=f"{metric} Surface: {param_x} vs {param_y}",
                scene=dict(
                    xaxis_title=param_x,
                    yaxis_title=param_y,
                    zaxis_title=metric,
                ),
                autosize=True,
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
