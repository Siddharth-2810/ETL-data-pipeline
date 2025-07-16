import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from data_pipeline import DataPipeline
from ml_models import MLModelTrainer
from utils import load_titanic_data, create_download_link

# Configure page
st.set_page_config(
    page_title="ETL Data Pipeline - Titanic Dataset",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = None

def main():
    st.title("🚢 Automated ETL Data Pipeline")
    st.subheader("Titanic Dataset Analysis & Machine Learning")
    
    # Sidebar navigation
    st.sidebar.title("Pipeline Navigation")
    pipeline_steps = [
        "📊 Data Loading & Exploration",
        "🔧 Data Preprocessing",
        "🤖 Machine Learning",
        "📈 Model Evaluation",
        "💾 Export Results"
    ]
    
    selected_step = st.sidebar.radio("Select Pipeline Step:", pipeline_steps)
    
    # Data Loading Section
    if selected_step == "📊 Data Loading & Exploration":
        data_loading_section()
    
    # Data Preprocessing Section
    elif selected_step == "🔧 Data Preprocessing":
        data_preprocessing_section()
    
    # Machine Learning Section
    elif selected_step == "🤖 Machine Learning":
        machine_learning_section()
    
    # Model Evaluation Section
    elif selected_step == "📈 Model Evaluation":
        model_evaluation_section()
    
    # Export Results Section
    elif selected_step == "💾 Export Results":
        export_results_section()

def data_loading_section():
    st.header("📊 Data Loading & Exploration")
    
    # Load data button
    if st.button("Load Titanic Dataset", type="primary"):
        with st.spinner("Loading dataset..."):
            try:
                df = load_titanic_data()
                st.session_state.original_data = df
                st.session_state.pipeline = DataPipeline(df)
                st.success("Dataset loaded successfully!")
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
                return
    
    if st.session_state.original_data is not None:
        df = st.session_state.original_data
        
        # Dataset overview
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Survival Rate", f"{df['Survived'].mean():.2%}")
        
        # Display raw data
        st.subheader("Raw Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Basic statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Data types and missing values
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.subheader("Missing Values Visualization")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig = px.bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation='h',
                    title="Missing Values by Column",
                    labels={'x': 'Count', 'y': 'Column'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No missing values found in the dataset!")
        
        # Data distribution visualizations
        st.subheader("Data Distributions")
        
        # Survival distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=df['Survived'].value_counts().values,
                names=['Did not survive', 'Survived'],
                title="Survival Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                df, x='Age', color='Survived',
                title="Age Distribution by Survival",
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                df, x='Survived', y='Fare',
                title="Fare Distribution by Survival"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            survival_by_class = df.groupby('Pclass')['Survived'].mean()
            fig = px.bar(
                x=survival_by_class.index,
                y=survival_by_class.values,
                title="Survival Rate by Passenger Class",
                labels={'x': 'Passenger Class', 'y': 'Survival Rate'}
            )
            st.plotly_chart(fig, use_container_width=True)

def data_preprocessing_section():
    st.header("🔧 Data Preprocessing")
    
    if st.session_state.pipeline is None:
        st.warning("Please load the dataset first from the Data Loading section.")
        return
    
    pipeline = st.session_state.pipeline
    
    # Preprocessing options
    st.subheader("Preprocessing Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Value Imputation**")
        age_strategy = st.selectbox(
            "Age imputation strategy:",
            ["median", "mean", "mode", "drop"]
        )
        embarked_strategy = st.selectbox(
            "Embarked imputation strategy:",
            ["mode", "drop", "forward_fill"]
        )
        
    with col2:
        st.write("**Feature Engineering**")
        create_family_size = st.checkbox("Create Family Size feature", value=True)
        create_title = st.checkbox("Extract Title from Name", value=True)
        create_age_groups = st.checkbox("Create Age Groups", value=True)
    
    st.write("**Encoding & Scaling**")
    col1, col2 = st.columns(2)
    
    with col1:
        encoding_method = st.selectbox(
            "Categorical encoding method:",
            ["one_hot", "label_encoding"]
        )
        
    with col2:
        scaling_method = st.selectbox(
            "Numerical scaling method:",
            ["standard", "minmax", "robust", "none"]
        )
    
    # Process data button
    if st.button("Apply Preprocessing", type="primary"):
        with st.spinner("Processing data..."):
            try:
                # Configure pipeline
                config = {
                    'age_strategy': age_strategy,
                    'embarked_strategy': embarked_strategy,
                    'create_family_size': create_family_size,
                    'create_title': create_title,
                    'create_age_groups': create_age_groups,
                    'encoding_method': encoding_method,
                    'scaling_method': scaling_method
                }
                
                # Apply preprocessing
                processed_df = pipeline.process_data(config)
                st.session_state.processed_data = processed_df
                st.success("Data preprocessing completed successfully!")
                
                # Show processing summary
                st.subheader("Preprocessing Summary")
                summary = pipeline.get_preprocessing_summary()
                for step, details in summary.items():
                    st.write(f"**{step}:** {details}")
                
            except Exception as e:
                st.error(f"Error during preprocessing: {str(e)}")
                return
    
    # Display processed data if available
    if st.session_state.processed_data is not None:
        processed_df = st.session_state.processed_data
        
        st.subheader("Processed Data")
        st.dataframe(processed_df.head(10), use_container_width=True)
        
        # Before/After comparison
        st.subheader("Before vs After Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Data**")
            original_info = {
                'Shape': st.session_state.original_data.shape,
                'Missing Values': st.session_state.original_data.isnull().sum().sum(),
                'Columns': list(st.session_state.original_data.columns)
            }
            st.json(original_info)
        
        with col2:
            st.write("**Processed Data**")
            processed_info = {
                'Shape': processed_df.shape,
                'Missing Values': processed_df.isnull().sum().sum(),
                'Columns': list(processed_df.columns)
            }
            st.json(processed_info)
        
        # Data quality checks
        st.subheader("Data Quality Validation")
        quality_checks = pipeline.validate_data_quality(processed_df)
        
        for check, result in quality_checks.items():
            if result['status'] == 'PASS':
                st.success(f"✅ {check}: {result['message']}")
            else:
                st.error(f"❌ {check}: {result['message']}")

def machine_learning_section():
    st.header("🤖 Machine Learning")
    
    if st.session_state.processed_data is None:
        st.warning("Please complete data preprocessing first.")
        return
    
    processed_df = st.session_state.processed_data
    
    # Model selection
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Model Type:",
            ["Random Forest", "Logistic Regression", "SVM", "Gradient Boosting"]
        )
        
        test_size = st.slider(
            "Test Size Ratio:",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05
        )
    
    with col2:
        random_state = st.number_input(
            "Random State:",
            min_value=0,
            max_value=1000,
            value=42
        )
        
        cross_validation = st.checkbox("Enable Cross Validation", value=True)
        cv_folds = st.slider("CV Folds:", 3, 10, 5) if cross_validation else 5
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Identify target and feature columns
    target_col = 'Survived'
    feature_cols = [col for col in processed_df.columns if col != target_col]
    
    selected_features = st.multiselect(
        "Select Features for Training:",
        feature_cols,
        default=feature_cols
    )
    
    if len(selected_features) == 0:
        st.warning("Please select at least one feature for training.")
        return
    
    # Train model button
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                # Initialize model trainer
                trainer = MLModelTrainer(processed_df, target_col, selected_features)
                st.session_state.model_trainer = trainer
                
                # Train model
                results = trainer.train_model(
                    model_type=model_type,
                    test_size=test_size,
                    random_state=random_state,
                    cross_validation=cross_validation,
                    cv_folds=cv_folds
                )
                
                st.success("Model training completed successfully!")
                
                # Display training results
                st.subheader("Training Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Training Accuracy", f"{results['train_accuracy']:.4f}")
                with col2:
                    st.metric("Test Accuracy", f"{results['test_accuracy']:.4f}")
                with col3:
                    if cross_validation:
                        st.metric("CV Score (Mean)", f"{results['cv_score']:.4f}")
                
                # Feature importance
                if 'feature_importance' in results:
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': results['feature_importance']
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Feature Importance"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")

def model_evaluation_section():
    st.header("📈 Model Evaluation")
    
    if st.session_state.model_trainer is None:
        st.warning("Please train a model first.")
        return
    
    trainer = st.session_state.model_trainer
    
    # Get evaluation metrics
    try:
        evaluation_results = trainer.evaluate_model()
        
        # Display classification report
        st.subheader("Classification Report")
        st.text(evaluation_results['classification_report'])
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig = px.imshow(
            evaluation_results['confusion_matrix'],
            labels=dict(x="Predicted", y="Actual"),
            x=['Did not survive', 'Survived'],
            y=['Did not survive', 'Survived'],
            title="Confusion Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve
        st.subheader("ROC Curve")
        fig = px.line(
            x=evaluation_results['fpr'],
            y=evaluation_results['tpr'],
            title=f"ROC Curve (AUC = {evaluation_results['auc_score']:.4f})",
            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model Performance Metrics
        st.subheader("Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{evaluation_results['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{evaluation_results['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{evaluation_results['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{evaluation_results['f1_score']:.4f}")
        
        # Prediction Examples
        st.subheader("Sample Predictions")
        predictions_df = trainer.get_sample_predictions(n_samples=10)
        st.dataframe(predictions_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error during model evaluation: {str(e)}")

def export_results_section():
    st.header("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Download Processed Dataset")
        if st.session_state.processed_data is not None:
            csv_data = st.session_state.processed_data.to_csv(index=False)
            st.download_button(
                label="Download Cleaned Dataset (CSV)",
                data=csv_data,
                file_name="titanic_processed.csv",
                mime="text/csv"
            )
        else:
            st.info("No processed data available for download.")
    
    with col2:
        st.subheader("Download Model Results")
        if st.session_state.model_trainer is not None:
            # Create model summary
            try:
                model_summary = st.session_state.model_trainer.get_model_summary()
                st.download_button(
                    label="Download Model Summary (JSON)",
                    data=model_summary,
                    file_name="model_summary.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error creating model summary: {str(e)}")
        else:
            st.info("No trained model available for download.")
    
    # Pipeline Summary
    st.subheader("Pipeline Summary")
    
    if st.session_state.pipeline is not None:
        summary = {
            "Dataset": "Titanic Dataset",
            "Original Shape": str(st.session_state.original_data.shape) if st.session_state.original_data is not None else "Not loaded",
            "Processed Shape": str(st.session_state.processed_data.shape) if st.session_state.processed_data is not None else "Not processed",
            "Model Trained": st.session_state.model_trainer is not None,
            "Processing Steps": st.session_state.pipeline.get_preprocessing_summary() if hasattr(st.session_state.pipeline, 'get_preprocessing_summary') else {}
        }
        
        st.json(summary)
    else:
        st.info("No pipeline data available.")

if __name__ == "__main__":
    main()
