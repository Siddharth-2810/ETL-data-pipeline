import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import json

class MLModelTrainer:
    def __init__(self, data, target_column, feature_columns):
        self.data = data
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def train_model(self, model_type="Random Forest", test_size=0.2, random_state=42, 
                   cross_validation=True, cv_folds=5):
        """
        Train machine learning model
        """
        # Prepare features and target
        X = self.data[self.feature_columns]
        y = self.data[self.target_column]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Initialize model based on type
        if model_type == "Random Forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
        elif model_type == "Logistic Regression":
            self.model = LogisticRegression(
                random_state=random_state,
                max_iter=1000
            )
        elif model_type == "SVM":
            self.model = SVC(
                kernel='rbf',
                random_state=random_state,
                probability=True
            )
        elif model_type == "Gradient Boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=random_state,
                max_depth=6,
                learning_rate=0.1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        if hasattr(self.model, 'predict_proba'):
            self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        train_accuracy = self.model.score(self.X_train, self.y_train)
        test_accuracy = accuracy_score(self.y_test, self.y_pred)
        
        results = {
            'model_type': model_type,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_size': test_size,
            'random_state': random_state
        }
        
        # Cross validation
        if cross_validation:
            cv_scores = cross_val_score(self.model, X, y, cv=cv_folds, scoring='accuracy')
            results['cv_score'] = cv_scores.mean()
            results['cv_std'] = cv_scores.std()
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            results['feature_importance'] = self.model.feature_importances_.tolist()
        elif hasattr(self.model, 'coef_'):
            results['feature_importance'] = abs(self.model.coef_[0]).tolist()
        
        return results
    
    def evaluate_model(self):
        """
        Comprehensive model evaluation
        """
        if self.model is None or self.y_pred is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Basic metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        
        # Classification report
        class_report = classification_report(self.y_test, self.y_pred)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        
        # ROC curve and AUC (if probabilities available)
        if self.y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
            auc_score = auc(fpr, tpr)
        else:
            fpr, tpr, auc_score = None, None, None
        
        evaluation_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'auc_score': auc_score,
            'fpr': fpr.tolist() if fpr is not None else None,
            'tpr': tpr.tolist() if tpr is not None else None
        }
        
        return evaluation_results
    
    def get_sample_predictions(self, n_samples=10):
        """
        Get sample predictions with actual values
        """
        if self.model is None or self.y_pred is None:
            raise ValueError("Model must be trained before getting predictions")
        
        # Select random samples
        indices = np.random.choice(len(self.X_test), min(n_samples, len(self.X_test)), replace=False)
        
        sample_data = []
        for idx in indices:
            sample = {
                'Index': self.X_test.index[idx],
                'Actual': int(self.y_test.iloc[idx]),
                'Predicted': int(self.y_pred[idx]),
                'Correct': self.y_test.iloc[idx] == self.y_pred[idx]
            }
            
            # Add feature values
            for col in self.feature_columns[:5]:  # Show first 5 features
                if col in self.X_test.columns:
                    sample[col] = self.X_test.iloc[idx][col]
            
            # Add probability if available
            if self.y_pred_proba is not None:
                sample['Probability'] = f"{self.y_pred_proba[idx]:.3f}"
            
            sample_data.append(sample)
        
        return pd.DataFrame(sample_data)
    
    def get_model_summary(self):
        """
        Get comprehensive model summary as JSON
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting summary")
        
        # Basic model info
        summary = {
            'model_type': type(self.model).__name__,
            'features_used': self.feature_columns,
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'target_column': self.target_column
        }
        
        # Model parameters
        summary['model_parameters'] = self.model.get_params()
        
        # Performance metrics
        if self.y_pred is not None:
            evaluation = self.evaluate_model()
            summary['performance'] = {
                'accuracy': evaluation['accuracy'],
                'precision': evaluation['precision'],
                'recall': evaluation['recall'],
                'f1_score': evaluation['f1_score'],
                'auc_score': evaluation['auc_score']
            }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            summary['feature_importance'] = dict(sorted_features)
        elif hasattr(self.model, 'coef_'):
            feature_importance = dict(zip(self.feature_columns, abs(self.model.coef_[0])))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            summary['feature_importance'] = dict(sorted_features)
        
        return json.dumps(summary, indent=2)
    
    def predict_new_data(self, new_data):
        """
        Make predictions on new data
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure new data has the same features
        missing_features = set(self.feature_columns) - set(new_data.columns)
        if missing_features:
            raise ValueError(f"Missing features in new data: {missing_features}")
        
        # Make predictions
        predictions = self.model.predict(new_data[self.feature_columns])
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(new_data[self.feature_columns])[:, 1]
            return predictions, probabilities
        
        return predictions, None
