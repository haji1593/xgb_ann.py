# xgb_ann.py
This  is machine learning project for predicting early detection of diabetes 
# 🩺 Professional Diabetes Prediction ML Platform
# Strategic Implementation by ML Engineering Team - Group 4
# Multi-page Interactive Application with Advanced Analytics

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report, precision_recall_curve)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import warnings
import io
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# STREAMLIT CONFIGURATION AND STYLING
# =============================================================================
st.set_page_config(
    page_title="Diabetes Prediction ML Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.8rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .sub-header {
        font-size: 2rem;
        color: #A23B72;
        border-bottom: 3px solid #F18F01;
        padding-bottom: 10px;
        margin: 2rem 0 1rem 0;
    }

    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 10px 0;
    }

    .info-box {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .warning-box {
        background: linear-gradient(90deg, #FF9800, #FF5722);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .success-box {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2E86AB 0%, #A23B72 100%);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #F18F01, #C73E1D);
        color: white;
        border-radius: 10px;
    }

    /* Progress bar */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# =============================================================================
@st.cache_data
def load_real_data():
    """Load the real diabetes CSV data with comprehensive error handling"""
    try:
        # Read the uploaded CSV file
        df = pd.read_csv('diabetes.csv')

        # Data validation
        required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

        if not all(col in df.columns for col in required_columns):
            st.error("❌ Dataset missing required columns")
            return None

        # Replace zeros with NaN for specific columns (medical impossibilities)
        zero_to_nan_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_to_nan_cols:
            df[col] = df[col].replace(0, np.nan)

        return df

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None


@st.cache_data
def comprehensive_preprocessing(df, strategy='advanced'):
    """Advanced data preprocessing with multiple strategies"""
    df_processed = df.copy()

    # Handle missing values with advanced imputation
    if strategy == 'advanced':
        # Use median for robust imputation
        imputer = SimpleImputer(strategy='median')
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.drop('Outcome')
        df_processed[numeric_columns] = imputer.fit_transform(df_processed[numeric_columns])

    # Advanced outlier handling using IQR method
    for column in numeric_columns:
        Q1 = df_processed[column].quantile(0.25)
        Q3 = df_processed[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap outliers instead of removing them
        df_processed[column] = df_processed[column].clip(lower_bound, upper_bound)

    # Feature engineering
    df_processed['BMI_Category'] = pd.cut(df_processed['BMI'],
                                          bins=[0, 18.5, 25, 30, float('inf')],
                                          labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    df_processed['Age_Group'] = pd.cut(df_processed['Age'],
                                       bins=[0, 30, 40, 50, float('inf')],
                                       labels=['Young', 'Adult', 'Middle-aged', 'Senior'])

    # Risk scores
    df_processed['Glucose_Risk'] = (df_processed['Glucose'] > 140).astype(int)
    df_processed['BP_Risk'] = (df_processed['BloodPressure'] > 90).astype(int)
    df_processed['BMI_Risk'] = (df_processed['BMI'] > 30).astype(int)

    return df_processed


# =============================================================================
# ADVANCED VISUALIZATION FUNCTIONS
# =============================================================================
def create_advanced_overview_dashboard(df):
    """Create an advanced overview dashboard with interactive elements"""
    st.markdown('<div class="sub-header">📊 Comprehensive Dataset Overview</div>', unsafe_allow_html=True)

    # Key metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>{len(df)}</h3>
            <p>Total Patients</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        diabetes_rate = df['Outcome'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>{diabetes_rate:.1%}</h3>
            <p>Diabetes Rate</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_age = df['Age'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>{avg_age:.1f}</h3>
            <p>Average Age</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_glucose = df['Glucose'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>{avg_glucose:.1f}</h3>
            <p>Avg Glucose</p>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        avg_bmi = df['BMI'].mean()
        st.markdown(f"""
        <div class="metric-container">
            <h3>{avg_bmi:.1f}</h3>
            <p>Average BMI</p>
        </div>
        """, unsafe_allow_html=True)

    # Interactive visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Enhanced pie chart
        outcome_counts = df['Outcome'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['No Diabetes', 'Diabetes'],
            values=outcome_counts.values,
            hole=0.4,
            marker_colors=['#3498db', '#e74c3c'],
            textinfo='label+percent+value',
            textfont_size=14
        )])
        fig.update_layout(
            title={
                'text': 'Diabetes Distribution',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2E86AB'}
            },
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Age distribution with outcome
        fig = px.histogram(df, x='Age', color='Outcome',
                           title='Age Distribution by Diabetes Status',
                           color_discrete_map={0: '#3498db', 1: '#e74c3c'},
                           marginal='box')
        fig.update_layout(
            title_x=0.5,
            title_font_size=18,
            title_font_color='#2E86AB'
        )
        st.plotly_chart(fig, use_container_width=True)


def create_correlation_heatmap(df):
    """Create an interactive correlation heatmap"""
    st.markdown('<div class="sub-header">🔗 Feature Correlation Analysis</div>', unsafe_allow_html=True)

    # Calculate correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    # Create interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={'size': 10},
        hoverongaps=False
    ))

    fig.update_layout(
        title={
            'text': 'Feature Correlation Matrix',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2E86AB'}
        },
        width=800,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # Feature importance based on correlation with target
    target_corr = abs(corr_matrix['Outcome'].drop('Outcome')).sort_values(ascending=False)

    fig_bar = px.bar(
        x=target_corr.values,
        y=target_corr.index,
        orientation='h',
        title='Feature Importance (Correlation with Diabetes)',
        color=target_corr.values,
        color_continuous_scale='viridis'
    )
    fig_bar.update_layout(
        title_x=0.5,
        title_font_size=18,
        title_font_color='#2E86AB',
        height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)


def create_distribution_analysis(df):
    """Create comprehensive distribution analysis"""
    st.markdown('<div class="sub-header">📈 Feature Distribution Analysis</div>', unsafe_allow_html=True)

    numeric_features = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin', 'Pregnancies']

    # Create subplots for distributions
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=numeric_features,
        specs=[[{"secondary_y": True}] * 3] * 2
    )

    colors = ['#3498db', '#e74c3c']

    for i, feature in enumerate(numeric_features):
        row = i // 3 + 1
        col = i % 3 + 1

        for outcome in [0, 1]:
            data = df[df['Outcome'] == outcome][feature].dropna()
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=f'Outcome {outcome}',
                    opacity=0.7,
                    marker_color=colors[outcome],
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )

    fig.update_layout(
        height=600,
        title_text="Feature Distributions by Diabetes Outcome",
        title_x=0.5,
        title_font_size=20,
        title_font_color='#2E86AB'
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# ADVANCED MACHINE LEARNING FUNCTIONS
# =============================================================================
def advanced_model_training(X_train, X_test, y_train, y_test):
    """Advanced model training with hyperparameter considerations"""

    # Initialize models with optimized parameters
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            max_iter=2000,
            alpha=0.001,
            random_state=42,
            early_stopping=True
        )
    }

    results = {}
    trained_models = {}
    scalers = {}

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (name, model) in enumerate(models.items()):
        status_text.text(f'Training {name}...')

        # Scale features for Neural Network
        if name == 'Neural Network':
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            scalers[name] = scaler

            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                        cv=StratifiedKFold(n_splits=5),
                                        scoring='f1')

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            scalers[name] = None

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=StratifiedKFold(n_splits=5),
                                        scoring='f1')

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate comprehensive metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        trained_models[name] = model
        progress_bar.progress((i + 1) / len(models))

    status_text.text('✅ Model training completed!')

    return results, trained_models, scalers


def create_model_evaluation_dashboard(results, y_test):
    """Create comprehensive model evaluation dashboard"""
    st.markdown('<div class="sub-header">🎯 Model Performance Evaluation</div>', unsafe_allow_html=True)

    # Performance metrics table
    with st.expander("📊 Detailed Performance Metrics", expanded=True):
        metrics_data = []
        for model_name, metrics in results.items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'CV Mean': f"{metrics['cv_mean']:.4f}",
                'CV Std': f"{metrics['cv_std']:.4f}"
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Performance comparison bar chart
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        fig = go.Figure()

        for metric in metrics:
            values = [results[model][metric] for model in results.keys()]
            fig.add_trace(go.Bar(
                name=metric.title(),
                x=list(results.keys()),
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ))

        fig.update_layout(
            title='Model Performance Comparison',
            title_x=0.5,
            barmode='group',
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ROC Curve comparison
        fig_roc = go.Figure()

        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            auc_score = auc(fpr, tpr)

            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{name} (AUC = {auc_score:.3f})',
                line=dict(width=3)
            ))

        # Add diagonal line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))

        fig_roc.update_layout(
            title='ROC Curve Comparison',
            title_x=0.5,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Confusion matrices
    with st.expander("🎭 Confusion Matrices", expanded=False):
        fig_cm = make_subplots(
            rows=1, cols=3,
            subplot_titles=list(results.keys()),
            specs=[[{"type": "heatmap"}] * 3]
        )

        for i, (name, result) in enumerate(results.items()):
            cm = confusion_matrix(y_test, result['y_pred'])

            fig_cm.add_trace(
                go.Heatmap(
                    z=cm,
                    x=['No Diabetes', 'Diabetes'],
                    y=['No Diabetes', 'Diabetes'],
                    colorscale='Blues',
                    showscale=(i == 2),
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 16}
                ),
                row=1, col=i + 1
            )

        fig_cm.update_layout(height=400, title_text="Confusion Matrices")
        st.plotly_chart(fig_cm, use_container_width=True)


def create_prediction_interface(trained_models, scalers, feature_names):
    """Advanced prediction interface with risk assessment"""
    st.markdown('<div class="sub-header">🔮 Diabetes Risk Assessment Tool</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h4>🩺 Patient Information Input</h4>
        <p>Enter the patient's medical information below for comprehensive diabetes risk assessment</p>
    </div>
    """, unsafe_allow_html=True)

    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👩‍⚕️ Basic Information**")
            pregnancies = st.number_input("Number of Pregnancies", 0, 15, 1, help="Total number of pregnancies")
            age = st.slider("Age", 18, 80, 30, help="Patient's age in years")

        with col2:
            st.markdown("**🩸 Vital Signs**")
            glucose = st.number_input("Glucose Level (mg/dL)", 50, 300, 120, help="Plasma glucose concentration")
            blood_pressure = st.number_input("Blood Pressure (mmHg)", 40, 150, 80, help="Diastolic blood pressure")

        with col3:
            st.markdown("**📏 Physical Measurements**")
            bmi = st.number_input("BMI", 10.0, 60.0, 25.0, 0.1, help="Body Mass Index")
            skin_thickness = st.number_input("Skin Thickness (mm)", 0, 60, 25, help="Triceps skinfold thickness")

        col4, col5 = st.columns(2)

        with col4:
            insulin = st.number_input("Insulin Level (μU/mL)", 0, 500, 100, help="2-Hour serum insulin")

        with col5:
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, 0.01,
                                                help="Diabetes pedigree function score")

        submitted = st.form_submit_button("🔍 Analyze Diabetes Risk", type="primary")

    if submitted:
        # Create input array
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, diabetes_pedigree, age]])

        st.markdown("---")
        st.markdown("### 📋 Risk Assessment Results")

        # Risk factor analysis
        risk_factors = []
        if glucose > 140:
            risk_factors.append("High Glucose Level")
        if bmi > 30:
            risk_factors.append("Obesity")
        if blood_pressure > 90:
            risk_factors.append("High Blood Pressure")
        if age > 45:
            risk_factors.append("Advanced Age")

        col1, col2 = st.columns([1, 2])

        with col1:
            if risk_factors:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠️ Risk Factors Detected</h4>
                    <ul>
                    {"".join([f"<li>{factor}</li>" for factor in risk_factors])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                    <h4>✅ No Major Risk Factors</h4>
                    <p>Patient shows good metabolic indicators</p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            # Model predictions
            predictions_df = []

            for name, model in trained_models.items():
                scaler = scalers[name]

                if scaler:  # Neural Network
                    input_scaled = scaler.transform(input_data)
                    prediction = model.predict(input_scaled)[0]
                    probability = model.predict_proba(input_scaled)[0]
                else:
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0]

                predictions_df.append({
                    'Model': name,
                    'Prediction': 'Diabetes' if prediction == 1 else 'No Diabetes',
                    'Confidence': f"{probability[1]:.1%}",
                    'Risk Level': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.3 else 'Low'
                })

            predictions_table = pd.DataFrame(predictions_df)
            st.dataframe(predictions_table, use_container_width=True)

            # Average risk assessment
            avg_risk = np.mean(
                [model.predict_proba(scalers[name].transform(input_data) if scalers[name] else input_data)[0][1]
                 for name, model in trained_models.items()])

            # Risk gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=avg_risk * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Average Diabetes Risk (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)


# =============================================================================
# PAGE NAVIGATION SYSTEM
# =============================================================================
def main():
    """Main application with page navigation"""

    # Sidebar navigation
    st.sidebar.title("🧭 Navigation Menu")
    st.sidebar.markdown("---")

    # Page selection
    pages = {
        "🏠 Home": "home",
        "📊 Data Overview": "overview",
        "🔍 Exploratory Analysis": "eda",
        "🤖 Model Training": "training",
        "📈 Model Evaluation": "evaluation",
        "🔮 Risk Prediction": "prediction",
        "📋 Final Report": "report"
    }

    selected_page = st.sidebar.selectbox("Choose a section:", list(pages.keys()))
    current_page = pages[selected_page]

    # Display team information
    st.sidebar.markdown("---")
    st.sidebar.markdown("**👨‍💼 Project Team**")
    team_members = [
        "Lamidi Okyere",
        "Eugene Nortey",
        "Mireku Raphael Debrah",
        "Ofori Samuel",
        "Hajara Abdul Mumin"
    ]

    with st.sidebar.expander("👥 Team Members"):
        for member in team_members:
            st.write(f"• {member}")

    # Load data once
    with st.spinner("🔄 Loading diabetes dataset..."):
        df = load_real_data()
        if df is None:
            st.error("❌ Failed to load data. Please check the CSV file.")
            return
