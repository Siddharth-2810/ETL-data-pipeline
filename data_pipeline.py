import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import re

class DataPipeline:
    def __init__(self, data):
        self.original_data = data.copy()
        self.processed_data = None
        self.preprocessing_steps = []
        self.encoders = {}
        self.scalers = {}
        
    def process_data(self, config):
        """
        Apply complete data preprocessing pipeline
        """
        df = self.original_data.copy()
        self.preprocessing_steps = []
        
        # 1. Handle missing values
        df = self._handle_missing_values(df, config)
        
        # 2. Feature engineering
        df = self._feature_engineering(df, config)
        
        # 3. Encode categorical variables
        df = self._encode_categorical_variables(df, config)
        
        # 4. Scale numerical features
        df = self._scale_numerical_features(df, config)
        
        # 5. Final data validation
        df = self._final_validation(df)
        
        self.processed_data = df
        return df
    
    def _handle_missing_values(self, df, config):
        """
        Handle missing values based on configuration
        """
        initial_missing = df.isnull().sum().sum()
        
        # Handle Age missing values
        if 'Age' in df.columns and df['Age'].isnull().sum() > 0:
            age_strategy = config.get('age_strategy', 'median')
            
            if age_strategy == 'drop':
                df = df.dropna(subset=['Age'])
                self.preprocessing_steps.append(f"Dropped rows with missing Age values")
            else:
                if age_strategy == 'median':
                    fill_value = df['Age'].median()
                elif age_strategy == 'mean':
                    fill_value = df['Age'].mean()
                elif age_strategy == 'mode':
                    fill_value = df['Age'].mode()[0]
                
                df['Age'].fillna(fill_value, inplace=True)
                self.preprocessing_steps.append(f"Filled Age missing values with {age_strategy}: {fill_value:.2f}")
        
        # Handle Embarked missing values
        if 'Embarked' in df.columns and df['Embarked'].isnull().sum() > 0:
            embarked_strategy = config.get('embarked_strategy', 'mode')
            
            if embarked_strategy == 'drop':
                df = df.dropna(subset=['Embarked'])
                self.preprocessing_steps.append("Dropped rows with missing Embarked values")
            elif embarked_strategy == 'mode':
                fill_value = df['Embarked'].mode()[0]
                df['Embarked'].fillna(fill_value, inplace=True)
                self.preprocessing_steps.append(f"Filled Embarked missing values with mode: {fill_value}")
            elif embarked_strategy == 'forward_fill':
                df['Embarked'].fillna(method='ffill', inplace=True)
                self.preprocessing_steps.append("Forward filled Embarked missing values")
        
        # Handle Cabin missing values (typically drop due to high missing percentage)
        if 'Cabin' in df.columns:
            cabin_missing_pct = df['Cabin'].isnull().sum() / len(df) * 100
            if cabin_missing_pct > 50:  # If more than 50% missing, create binary feature
                df['Has_Cabin'] = df['Cabin'].notna().astype(int)
                df = df.drop('Cabin', axis=1)
                self.preprocessing_steps.append(f"Created Has_Cabin binary feature (Cabin was {cabin_missing_pct:.1f}% missing)")
        
        final_missing = df.isnull().sum().sum()
        self.preprocessing_steps.append(f"Missing values reduced from {initial_missing} to {final_missing}")
        
        return df
    
    def _feature_engineering(self, df, config):
        """
        Create new features based on existing data
        """
        # Create Family Size feature
        if config.get('create_family_size', True):
            if 'SibSp' in df.columns and 'Parch' in df.columns:
                df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
                df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
                self.preprocessing_steps.append("Created FamilySize and IsAlone features")
        
        # Extract Title from Name
        if config.get('create_title', True) and 'Name' in df.columns:
            df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            
            # Group rare titles
            title_counts = df['Title'].value_counts()
            rare_titles = title_counts[title_counts < 10].index
            df['Title'] = df['Title'].replace(rare_titles, 'Rare')
            
            # Standardize titles
            title_mapping = {
                'Mr': 'Mr',
                'Miss': 'Miss',
                'Mrs': 'Mrs',
                'Master': 'Master',
                'Dr': 'Officer',
                'Rev': 'Officer',
                'Col': 'Officer',
                'Major': 'Officer',
                'Mlle': 'Miss',
                'Countess': 'Royalty',
                'Ms': 'Miss',
                'Lady': 'Royalty',
                'Jonkheer': 'Royalty',
                'Don': 'Royalty',
                'Dona': 'Royalty',
                'Mme': 'Mrs',
                'Capt': 'Officer',
                'Sir': 'Royalty',
                'Rare': 'Rare'
            }
            df['Title'] = df['Title'].map(title_mapping)
            self.preprocessing_steps.append("Extracted and standardized Title from Name")
        
        # Create Age Groups
        if config.get('create_age_groups', True) and 'Age' in df.columns:
            df['AgeGroup'] = pd.cut(df['Age'], 
                                  bins=[0, 12, 18, 35, 60, 100], 
                                  labels=['Child', 'Teen', 'Adult', 'MiddleAge', 'Senior'])
            self.preprocessing_steps.append("Created AgeGroup categorical feature")
        
        # Create Fare Bins
        if 'Fare' in df.columns:
            df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
            self.preprocessing_steps.append("Created FareBin categorical feature")
        
        return df
    
    def _encode_categorical_variables(self, df, config):
        """
        Encode categorical variables
        """
        encoding_method = config.get('encoding_method', 'one_hot')
        
        # Identify categorical columns (excluding target)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'Survived' in categorical_cols:
            categorical_cols.remove('Survived')
        
        # Remove high cardinality columns that should be dropped
        high_cardinality_cols = ['Name', 'Ticket', 'PassengerId']
        categorical_cols = [col for col in categorical_cols if col not in high_cardinality_cols]
        
        # Drop high cardinality columns
        df = df.drop([col for col in high_cardinality_cols if col in df.columns], axis=1)
        
        if encoding_method == 'one_hot':
            # One-hot encoding for categorical variables
            for col in categorical_cols:
                if col in df.columns:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(col, axis=1)
            
            self.preprocessing_steps.append(f"Applied one-hot encoding to {len(categorical_cols)} categorical columns")
        
        elif encoding_method == 'label_encoding':
            # Label encoding for categorical variables
            for col in categorical_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
            
            self.preprocessing_steps.append(f"Applied label encoding to {len(categorical_cols)} categorical columns")
        
        return df
    
    def _scale_numerical_features(self, df, config):
        """
        Scale numerical features
        """
        scaling_method = config.get('scaling_method', 'standard')
        
        if scaling_method == 'none':
            self.preprocessing_steps.append("No scaling applied to numerical features")
            return df
        
        # Identify numerical columns (excluding target)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Survived' in numerical_cols:
            numerical_cols.remove('Survived')
        
        if len(numerical_cols) == 0:
            self.preprocessing_steps.append("No numerical columns found for scaling")
            return df
        
        # Choose scaler
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        
        # Apply scaling
        X_scaled = scaler.fit_transform(df[numerical_cols])
        df[numerical_cols] = X_scaled
        
        self.scalers['numerical'] = scaler
        self.preprocessing_steps.append(f"Applied {scaling_method} scaling to {len(numerical_cols)} numerical columns")
        
        return df
    
    def _final_validation(self, df):
        """
        Final data validation and cleanup
        """
        # Check for any remaining missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            # Fill any remaining missing values with 0 (should be rare at this point)
            df = df.fillna(0)
            self.preprocessing_steps.append(f"Filled {missing_values} remaining missing values with 0")
        
        # Ensure all columns are numeric (except target)
        for col in df.columns:
            if col != 'Survived' and df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                    self.preprocessing_steps.append(f"Force-encoded non-numeric column: {col}")
        
        # Remove any infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        self.preprocessing_steps.append("Final data validation completed")
        return df
    
    def get_preprocessing_summary(self):
        """
        Get summary of preprocessing steps
        """
        return {
            'steps': self.preprocessing_steps,
            'original_shape': str(self.original_data.shape),
            'processed_shape': str(self.processed_data.shape) if self.processed_data is not None else 'Not processed',
            'features_created': len(self.processed_data.columns) - len(self.original_data.columns) if self.processed_data is not None else 0
        }
    
    def validate_data_quality(self, df):
        """
        Perform data quality checks
        """
        checks = {}
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        checks['missing_values'] = {
            'status': 'PASS' if missing_count == 0 else 'FAIL',
            'message': f"Missing values: {missing_count}"
        }
        
        # Check for infinite values
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        checks['infinite_values'] = {
            'status': 'PASS' if inf_count == 0 else 'FAIL',
            'message': f"Infinite values: {inf_count}"
        }
        
        # Check data types
        non_numeric_cols = []
        for col in df.columns:
            if col != 'Survived' and not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
        
        checks['data_types'] = {
            'status': 'PASS' if len(non_numeric_cols) == 0 else 'FAIL',
            'message': f"Non-numeric columns (excluding target): {non_numeric_cols}"
        }
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        checks['duplicates'] = {
            'status': 'PASS' if duplicate_count == 0 else 'WARNING',
            'message': f"Duplicate rows: {duplicate_count}"
        }
        
        # Check target variable
        if 'Survived' in df.columns:
            target_unique = df['Survived'].nunique()
            checks['target_variable'] = {
                'status': 'PASS' if target_unique == 2 else 'FAIL',
                'message': f"Target variable has {target_unique} unique values (expected: 2)"
            }
        
        return checks
