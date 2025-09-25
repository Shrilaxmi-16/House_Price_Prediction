import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


st.set_page_config(page_title="🏠 Housing Dashboard", layout="wide")

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Housing.csv")
    return df

df = load_data()

st.title("🏠 Housing Data Dashboard - Analysis & Prediction")

# -------------------------
# Sidebar Navigation
# -------------------------
menu = st.sidebar.radio(
    "📌 Navigate",
    ["EDA", "Visualization", "Model Training", "Prediction"]
)

# -------------------------
# Exploratory Data Analysis
# -------------------------
if menu == "EDA":
    st.header("📊 Exploratory Data Analysis")

    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    st.pyplot(plt)

# -------------------------
# Visualization Section
# -------------------------
elif menu == "Visualization":
    st.header("📈 Data Visualizations")

    # Numeric column selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Histogram
    st.subheader("Histogram")
    selected_num = st.selectbox("Choose a numeric column", numeric_cols)
    plt.figure(figsize=(8, 5))
    sns.histplot(df[selected_num], kde=True, bins=30)
    st.pyplot(plt)

    # Boxplot
    st.subheader("Boxplot")
    if cat_cols:
        selected_cat = st.selectbox("Choose a categorical column (for grouping)", cat_cols)
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=selected_cat, y=selected_num, data=df)
        st.pyplot(plt)
    else:
        st.info("No categorical columns available for boxplot.")

    # Scatterplot
    st.subheader("Scatterplot")
    x_axis = st.selectbox("X-axis", numeric_cols, index=0)
    y_axis = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[x_axis], y=df[y_axis])
    st.pyplot(plt)

# -------------------------
# Model Training
# -------------------------
elif menu == "Model Training":
    st.header("🤖 Train Models for House Price Prediction")

    if "price" in df.columns:
        df_encoded = pd.get_dummies(df, drop_first=True)

        X = df_encoded.drop("price", axis=1)
        y = df_encoded["price"]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        results = {}
        trained_models = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results[name] = {
                "R² Score": r2_score(y_test, preds),
                "RMSE": np.sqrt(mean_squared_error(y_test, preds))
            }
            trained_models[name] = model

        st.subheader("📊 Model Performance Comparison")
        result_df = pd.DataFrame(results).T
        st.dataframe(result_df)

        # Bar chart
        st.subheader("Model Performance Visualization")
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        result_df["R² Score"].plot(kind="bar", ax=ax[0], title="R² Score")
        result_df["RMSE"].plot(kind="bar", ax=ax[1], title="RMSE", color="orange")
        st.pyplot(fig)

        # Feature Importance (only for tree-based models)
        st.subheader("🔑 Feature Importance (Random Forest)")
        rf_model = trained_models["Random Forest"]
        importances = pd.Series(rf_model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False).head(10)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=importances.values, y=importances.index)
        plt.title("Top 10 Important Features")
        st.pyplot(plt)

        st.success("✅ Best model: " + result_df["R² Score"].idxmax())
    else:
        st.error("Dataset must contain a 'price' column for prediction.")

# -------------------------
# Prediction
# -------------------------
elif menu == "Prediction":
    st.header("🔮 Predict House Price")

    df_encoded = pd.get_dummies(df, drop_first=True)

    if "price" in df_encoded.columns:
        X = df_encoded.drop("price", axis=1)
        y = df_encoded["price"]

        # Train a default model (best one can be chosen)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        input_data = {}
        st.write("### Enter House Features")

        for col in X.columns:
            if col in df.columns and df[col].dtype == "object":
                input_data[col] = st.selectbox(f"{col}", df[col].unique())
            else:
                input_data[col] = st.number_input(
                    f"{col}",
                    float(df_encoded[col].min()),
                    float(df_encoded[col].max()),
                    float(df_encoded[col].mean())
                )

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)

        # Align columns
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        if st.button("Predict Price"):
            prediction = model.predict(input_df)[0]
            st.success(f"🏡 Predicted House Price: {prediction:,.2f}")
    else:
        st.error("Dataset must contain a 'price' column for prediction.")

