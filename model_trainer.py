import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import logging
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, features_path: str, output_path: str, 
                 test_size: float = 0.2, random_state: int = 42):
        self.features_path = features_path
        self.output_path = output_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Model parameters
        self.rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        self.svm_params = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }

    def load_features(self, feature_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads features from a .npz file."""
        try:
            data = np.load(os.path.join(self.features_path, f"{feature_type}_features.npz"))
            return data['patient_ids'], data['features'], data['labels']
        except Exception as e:
            logging.error(f"Error loading {feature_type} features: {e}")
            raise

    def preprocess_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits and scales the data."""
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=self.test_size, random_state=self.random_state, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Trains a model using GridSearchCV."""
        if model_type == 'rf':
            model = RandomForestClassifier(random_state=self.random_state)
            params = self.rf_params
        elif model_type == 'svm':
            model = SVC(probability=True, random_state=self.random_state)
            params = self.svm_params
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        grid_search = GridSearchCV(
            model, params, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        logging.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluates model performance."""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        return metrics

    def plot_confusion_matrix(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                            feature_type: str, model_type: str) -> None:
        """Plots and saves confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {feature_type} - {model_type}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.output_path, f'confusion_matrix_{feature_type}_{model_type}.png'))
        plt.close()

    def save_model(self, model: Any, feature_type: str, model_type: str) -> None:
        """Saves the trained model."""
        model_path = os.path.join(self.output_path, f'model_{feature_type}_{model_type}.joblib')
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")

    def run_training(self, feature_types: list = ['glcm', 'hog', 'lbp'], 
                    model_types: list = ['rf', 'svm']) -> None:
        """Runs the complete training pipeline."""
        results = []
        
        for feature_type in feature_types:
            try:
                # Load features
                patient_ids, features, labels = self.load_features(feature_type)
                
                # Preprocess data
                X_train, X_test, y_train, y_test = self.preprocess_data(features, labels)
                
                for model_type in model_types:
                    logging.info(f"Training {model_type} model with {feature_type} features...")
                    
                    # Train model
                    model = self.train_model(model_type, X_train, y_train)
                    
                    # Evaluate model
                    metrics = self.evaluate_model(model, X_test, y_test)
                    
                    # Plot confusion matrix
                    self.plot_confusion_matrix(model, X_test, y_test, feature_type, model_type)
                    
                    # Save model
                    self.save_model(model, feature_type, model_type)
                    
                    # Store results
                    results.append({
                        'feature_type': feature_type,
                        'model_type': model_type,
                        **metrics
                    })
                    
            except Exception as e:
                logging.error(f"Error processing {feature_type}: {e}")
                continue
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_path, 'training_results.csv'), index=False)
        logging.info("Training completed! Results saved to training_results.csv")

if __name__ == "__main__":
    # Example usage
    FEATURES_PATH = "./output/features"
    OUTPUT_PATH = "./output/models"
    
    trainer = ModelTrainer(FEATURES_PATH, OUTPUT_PATH)
    trainer.run_training() 