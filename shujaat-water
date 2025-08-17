import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Water Quality Prediction shujaat waqar",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">💧 Water Quality Prediction shujaat waqar</h1>', unsafe_allow_html=True)
st.markdown("""
This application predicts water quality based on various chemical and physical parameters.
Upload your data or use the sample dataset to train models and make predictions.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page",
                            ["Data Upload & Overview", "Model Training", "Prediction", "Data Visualization"])


# Function to generate sample water quality data
@st.cache_data
def generate_sample_data(n_samples=1000):
    """Generate synthetic water quality data"""
    np.random.seed(42)

    # Generate features
    ph = np.random.normal(7.2, 1.5, n_samples)
    hardness = np.random.exponential(150, n_samples)
    solids = np.random.normal(20000, 5000, n_samples)
    chloramines = np.random.normal(7, 2, n_samples)
    sulfate = np.random.normal(250, 100, n_samples)
    conductivity = np.random.normal(400, 100, n_samples)
    organic_carbon = np.random.normal(14, 3, n_samples)
    trihalomethanes = np.random.normal(70, 20, n_samples)
    turbidity = np.random.exponential(4, n_samples)

    # Create quality labels based on realistic thresholds
    quality = []
    for i in range(n_samples):
        score = 0
        # pH should be between 6.5-8.5
        if 6.5 <= ph[i] <= 8.5:
            score += 1
        # Hardness should be reasonable
        if hardness[i] < 300:
            score += 1
        # Chloramines should be moderate
        if chloramines[i] < 4:
            score += 1
        # Sulfate should be within limits
        if sulfate[i] < 400:
            score += 1
        # Turbidity should be low
        if turbidity[i] < 5:
            score += 1
        # Organic carbon should be moderate
        if organic_carbon[i] < 20:
            score += 1

        # Classify based on score (with some randomness)
        if score >= 4:
            quality.append(1 if np.random.random() > 0.2 else 0)
        else:
            quality.append(0 if np.random.random() > 0.3 else 1)

    # Create DataFrame
    data = pd.DataFrame({
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity,
        'Potability': quality
    })

    # Ensure positive values where appropriate
    data['Hardness'] = np.abs(data['Hardness'])
    data['Solids'] = np.abs(data['Solids'])
    data['Chloramines'] = np.abs(data['Chloramines'])
    data['Sulfate'] = np.abs(data['Sulfate'])
    data['Conductivity'] = np.abs(data['Conductivity'])
    data['Organic_carbon'] = np.abs(data['Organic_carbon'])
    data['Trihalomethanes'] = np.abs(data['Trihalomethanes'])
    data['Turbidity'] = np.abs(data['Turbidity'])

    return data


# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# Page 1: Data Upload & Overview
if page == "Data Upload & Overview":
    st.header("📊 Data Upload & Overview")

    # Option to use sample data or upload
    data_option = st.radio("Choose data source:", ["Use Sample Data", "Upload Your Data"])

    if data_option == "Use Sample Data":
        if st.button("Generate Sample Water Quality Data"):
            st.session_state.data = generate_sample_data()
            st.success("Sample data generated successfully!")

    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # Display data overview if data exists
    if st.session_state.data is not None:
        data = st.session_state.data

        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(data))
        with col2:
            st.metric("Features", len(data.columns) - 1)
        with col3:
            if 'Potability' in data.columns:
                safe_percentage = (data['Potability'].sum() / len(data)) * 100
                st.metric("Safe Water %", f"{safe_percentage:.1f}%")

        # Data preview
        st.subheader("Data Preview")
        st.dataframe(data.head(10))

        # Data statistics
        st.subheader("Statistical Summary")
        st.dataframe(data.describe())

        # Missing values
        st.subheader("Missing Values")
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            fig = px.bar(x=missing_data.index, y=missing_data.values,
                         title="Missing Values by Feature")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found!")

# Page 2: Model Training
elif page == "Model Training":
    st.header("🤖 Model Training")

    if st.session_state.data is None:
        st.warning("Please upload data first!")
        st.stop()

    data = st.session_state.data

    # Check if target column exists
    if 'Potability' not in data.columns:
        st.error("Target column 'Potability' not found in the dataset!")
        st.stop()

    # Prepare data
    X = data.drop('Potability', axis=1)
    y = data['Potability']

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Model selection
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox("Select Model Type",
                                  ["Random Forest", "Gradient Boosting", "Logistic Regression"])
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

    with col2:
        scale_features = st.checkbox("Scale Features", value=True)
        random_state = st.number_input("Random State", value=42)

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_imputed, y, test_size=test_size, random_state=int(random_state)
            )

            # Scale features if requested
            if scale_features:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                st.session_state.scaler = scaler
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
                st.session_state.scaler = None

            # Train model
            if model_type == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=int(random_state))
            elif model_type == "Gradient Boosting":
                model = GradientBoostingClassifier(n_estimators=100, random_state=int(random_state))
            else:
                model = LogisticRegression(random_state=int(random_state))

            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            # Store in session state
            st.session_state.model = model
            st.session_state.feature_names = X.columns.tolist()

            # Display results
            st.success(f"Model trained successfully! Accuracy: {accuracy:.3f}")

            # Model performance metrics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

            with col2:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, text_auto=True, aspect="auto",
                                title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)

            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)

                fig = px.bar(importance_df, x='Importance', y='Feature',
                             orientation='h', title="Feature Importance")
                st.plotly_chart(fig, use_container_width=True)

# Page 3: Prediction
elif page == "Prediction":
    st.header("🔮 Water Quality Prediction")

    if st.session_state.model is None:
        st.warning("Please train a model first!")
        st.stop()

    st.subheader("Enter Water Quality Parameters")

    # Create input fields
    col1, col2 = st.columns(2)

    with col1:
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=200.0, step=1.0)
        solids = st.number_input("Total Dissolved Solids (ppm)", min_value=0.0, value=20000.0, step=100.0)
        chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.0, step=0.1)
        sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=250.0, step=1.0)

    with col2:
        conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=400.0, step=1.0)
        organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=14.0, step=0.1)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, value=70.0, step=1.0)
        turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=4.0, step=0.1)

    if st.button("Predict Water Quality"):
        # Prepare input data
        input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
                                conductivity, organic_carbon, trihalomethanes, turbidity]])

        # Scale if necessary
        if st.session_state.scaler is not None:
            input_data = st.session_state.scaler.transform(input_data)

        # Make prediction
        prediction = st.session_state.model.predict(input_data)[0]
        prediction_proba = st.session_state.model.predict_proba(input_data)[0]

        # Display result
        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.success("✅ Water is SAFE for consumption!")
                st.balloons()
            else:
                st.error("⚠️ Water is NOT SAFE for consumption!")

        with col2:
            st.subheader("Prediction Confidence")
            confidence_df = pd.DataFrame({
                'Class': ['Not Safe', 'Safe'],
                'Probability': prediction_proba
            })
            fig = px.bar(confidence_df, x='Class', y='Probability',
                         title="Prediction Probabilities")
            st.plotly_chart(fig, use_container_width=True)

        # Display input summary
        st.subheader("Input Parameters Summary")
        input_df = pd.DataFrame({
            'Parameter': ['pH', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                          'Conductivity', 'Organic Carbon', 'Trihalomethanes', 'Turbidity'],
            'Value': [ph, hardness, solids, chloramines, sulfate,
                      conductivity, organic_carbon, trihalomethanes, turbidity]
        })
        st.dataframe(input_df)

# Page 4: Data Visualization
elif page == "Data Visualization":
    st.header("📈 Data Visualization")

    if st.session_state.data is None:
        st.warning("Please upload data first!")
        st.stop()

    data = st.session_state.data

    # Visualization options
    viz_type = st.selectbox("Select Visualization Type",
                            ["Distribution Plots", "Correlation Matrix", "Quality Comparison", "Feature Relationships"])

    if viz_type == "Distribution Plots":
        st.subheader("Feature Distributions")
        feature = st.selectbox("Select Feature", data.columns[:-1])

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f'{feature} Distribution', f'{feature} by Water Quality'])

        # Histogram
        fig.add_trace(go.Histogram(x=data[feature], name=feature, nbinsx=30), row=1, col=1)

        # Box plot by quality
        if 'Potability' in data.columns:
            safe_data = data[data['Potability'] == 1][feature]
            unsafe_data = data[data['Potability'] == 0][feature]

            fig.add_trace(go.Box(y=safe_data, name='Safe', boxmean=True), row=1, col=2)
            fig.add_trace(go.Box(y=unsafe_data, name='Unsafe', boxmean=True), row=1, col=2)

        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Correlation Matrix":
        st.subheader("Feature Correlation Matrix")
        numeric_data = data.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()

        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Correlation Matrix of Water Quality Features")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Quality Comparison":
        if 'Potability' in data.columns:
            st.subheader("Water Quality Distribution")

            col1, col2 = st.columns(2)

            with col1:
                quality_counts = data['Potability'].value_counts()
                fig = px.pie(values=quality_counts.values,
                             names=['Unsafe', 'Safe'],
                             title="Water Quality Distribution")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Average feature values by quality
                avg_by_quality = data.groupby('Potability').mean()
                fig = go.Figure()

                for col in avg_by_quality.columns:
                    fig.add_trace(go.Bar(name=col, x=['Unsafe', 'Safe'],
                                         y=[avg_by_quality.loc[0, col], avg_by_quality.loc[1, col]]))

                fig.update_layout(title="Average Feature Values by Water Quality",
                                  barmode='group', height=500)
                st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Feature Relationships":
        st.subheader("Feature Relationships")

        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis Feature", data.columns[:-1])
        with col2:
            y_feature = st.selectbox("Y-axis Feature", data.columns[:-1])

        if 'Potability' in data.columns:
            fig = px.scatter(data, x=x_feature, y=y_feature,
                             color='Potability',
                             title=f'{x_feature} vs {y_feature}',
                             color_discrete_map={0: 'red', 1: 'blue'},
                             labels={'0': 'Unsafe', '1': 'Safe'})
        else:
            fig = px.scatter(data, x=x_feature, y=y_feature,
                             title=f'{x_feature} vs {y_feature}')

        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Water Quality Prediction shujaat waqar | Built with Streamlit & Scikit-learn</p>
    <p>💧 Ensuring safe water for everyone 💧</p>
</div>
""", unsafe_allow_html=True)
