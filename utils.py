import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import base64
import io

def load_titanic_data():
    """
    Load the Titanic dataset
    Uses seaborn's built-in dataset or creates a sample if unavailable
    """
    try:
        # Try to load from seaborn
        df = sns.load_dataset('titanic')
        
        # Ensure consistent column names
        column_mapping = {
            'survived': 'Survived',
            'pclass': 'Pclass',
            'sex': 'Sex',
            'age': 'Age',
            'sibsp': 'SibSp',
            'parch': 'Parch',
            'fare': 'Fare',
            'embarked': 'Embarked',
            'class': 'Class',
            'who': 'Who',
            'adult_male': 'Adult_male',
            'deck': 'Deck',
            'embark_town': 'Embark_town',
            'alive': 'Alive',
            'alone': 'Alone'
        }
        
        # Rename columns to match expected format
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Add missing columns that are typically in Titanic dataset
        if 'PassengerId' not in df.columns:
            df['PassengerId'] = range(1, len(df) + 1)
        
        if 'Name' not in df.columns:
            # Generate sample names
            titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.', 'Rev.']
            first_names = ['John', 'Mary', 'James', 'Patricia', 'Robert', 'Jennifer', 
                          'Michael', 'Linda', 'William', 'Elizabeth']
            last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia',
                         'Miller', 'Davis', 'Rodriguez', 'Martinez']
            
            names = []
            for i in range(len(df)):
                title = np.random.choice(titles)
                first = np.random.choice(first_names)
                last = np.random.choice(last_names)
                names.append(f"{last}, {first} {title}")
            
            df['Name'] = names
        
        if 'Ticket' not in df.columns:
            # Generate sample ticket numbers
            df['Ticket'] = [f"TICKET{i:05d}" for i in range(1, len(df) + 1)]
        
        if 'Cabin' not in df.columns:
            # Generate sample cabin numbers (with many missing values)
            cabins = []
            for i in range(len(df)):
                if np.random.random() > 0.7:  # 70% missing rate
                    cabins.append(f"{np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'])}{np.random.randint(1, 100)}")
                else:
                    cabins.append(np.nan)
            df['Cabin'] = cabins
        
        # Ensure target variable is properly formatted
        if 'Survived' in df.columns:
            df['Survived'] = df['Survived'].astype(int)
        
        # Select and reorder key columns
        key_columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 
                      'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        
        # Keep only columns that exist
        available_columns = [col for col in key_columns if col in df.columns]
        df = df[available_columns]
        
        return df
        
    except Exception as e:
        st.error(f"Error loading Titanic dataset: {str(e)}")
        
        # Create a minimal sample dataset as fallback
        np.random.seed(42)
        n_samples = 100
        
        sample_data = {
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
            'Name': [f"Sample, Person {i}" for i in range(1, n_samples + 1)],
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'Age': np.random.normal(30, 15, n_samples),
            'SibSp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.07, 0.03]),
            'Parch': np.random.choice([0, 1, 2], n_samples, p=[0.8, 0.15, 0.05]),
            'Ticket': [f"TICKET{i:05d}" for i in range(1, n_samples + 1)],
            'Fare': np.random.exponential(15, n_samples),
            'Cabin': [None] * n_samples,  # All missing
            'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1])
        }
        
        # Add some missing values to Age
        age_missing_indices = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
        for idx in age_missing_indices:
            sample_data['Age'][idx] = np.nan
        
        # Add some missing values to Embarked
        embarked_missing_indices = np.random.choice(n_samples, size=2, replace=False)
        for idx in embarked_missing_indices:
            sample_data['Embarked'][idx] = None
        
        df = pd.DataFrame(sample_data)
        
        st.warning("Using sample data due to loading issues. In a real application, you would load the actual Titanic dataset.")
        
        return df

def create_download_link(df, filename, text="Download"):
    """
    Create a download link for a pandas DataFrame
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def format_data_summary(df):
    """
    Create a formatted summary of the dataset
    """
    summary = {
        'Shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
        'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
        'Missing Values': df.isnull().sum().sum(),
        'Duplicate Rows': df.duplicated().sum(),
        'Numerical Columns': len(df.select_dtypes(include=[np.number]).columns),
        'Categorical Columns': len(df.select_dtypes(include=['object', 'category']).columns)
    }
    
    return summary

def validate_data_preprocessing(original_df, processed_df):
    """
    Validate data preprocessing results
    """
    validation_results = {}
    
    # Check if processed data has fewer or equal missing values
    original_missing = original_df.isnull().sum().sum()
    processed_missing = processed_df.isnull().sum().sum()
    
    validation_results['missing_values'] = {
        'original': original_missing,
        'processed': processed_missing,
        'improved': processed_missing <= original_missing
    }
    
    # Check if processed data is all numeric (except target)
    non_numeric_cols = []
    for col in processed_df.columns:
        if col != 'Survived' and not pd.api.types.is_numeric_dtype(processed_df[col]):
            non_numeric_cols.append(col)
    
    validation_results['data_types'] = {
        'all_numeric': len(non_numeric_cols) == 0,
        'non_numeric_columns': non_numeric_cols
    }
    
    # Check for infinite values
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(processed_df[numeric_cols]).sum().sum()
    
    validation_results['infinite_values'] = {
        'count': inf_count,
        'clean': inf_count == 0
    }
    
    return validation_results

def get_feature_statistics(df, feature_cols):
    """
    Get statistics for feature columns
    """
    stats = {}
    
    for col in feature_cols:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col] = {
                    'type': 'numeric',
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'missing': df[col].isnull().sum()
                }
            else:
                stats[col] = {
                    'type': 'categorical',
                    'unique_values': df[col].nunique(),
                    'most_common': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                    'missing': df[col].isnull().sum()
                }
    
    return stats

def create_correlation_matrix(df, target_col='Survived'):
    """
    Create correlation matrix for numeric columns
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None
    
    correlation_matrix = df[numeric_cols].corr()
    
    # Get correlations with target variable
    if target_col in correlation_matrix.columns:
        target_correlations = correlation_matrix[target_col].abs().sort_values(ascending=False)
        return correlation_matrix, target_correlations
    
    return correlation_matrix, None

def calculate_data_quality_score(df):
    """
    Calculate an overall data quality score
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    
    # Missing data penalty
    missing_penalty = (missing_cells / total_cells) * 100
    
    # Duplicate rows penalty
    duplicate_penalty = (df.duplicated().sum() / df.shape[0]) * 100
    
    # Data type consistency bonus
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    type_bonus = (len(numeric_cols) / df.shape[1]) * 10
    
    # Calculate final score (0-100)
    quality_score = max(0, 100 - missing_penalty - duplicate_penalty + type_bonus)
    quality_score = min(100, quality_score)
    
    return {
        'overall_score': quality_score,
        'missing_penalty': missing_penalty,
        'duplicate_penalty': duplicate_penalty,
        'type_bonus': type_bonus,
        'grade': 'A' if quality_score >= 90 else 'B' if quality_score >= 80 else 'C' if quality_score >= 70 else 'D'
    }
