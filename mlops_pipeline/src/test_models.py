import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

# mlops_pipeline/src/test_model_training.py


from model_training import (
    HeuristicModel,
    _evaluate_candidate,
    summarize_classification,
    build_model,
)


class TestDataLeakage:
    """Tests to detect data leakage that causes perfect scores."""
    
    def test_evaluate_candidate_no_data_leakage_simple(self):
        """Test that preprocessing doesn't leak test data into training.
        
        This test creates data where perfect separation is impossible
        unless there's leakage.
        """
        np.random.seed(42)
        n_samples = 200
        
        # Create overlapping distributions - impossible to get perfect scores
        X_train = pd.DataFrame({
            'capital_prestado': np.random.normal(5000, 1000, n_samples),
            'salario_cliente': np.random.normal(3000, 500, n_samples),
            'edad_cliente': np.random.randint(25, 60, n_samples),
            'puntaje_datacredito': np.random.normal(0, 1, n_samples),
            'tipo_credito': np.random.choice([4, 9], n_samples),
            'tipo_laboral': np.random.choice(['Empleado', 'Independiente'], n_samples),
            'saldo_mora': np.random.exponential(100, n_samples),
            'plazo_meses': np.random.randint(6, 36, n_samples),
        })
        
        # Random labels - no pattern
        y_train = pd.Series(np.random.randint(0, 2, n_samples))
        
        X_test = pd.DataFrame({
            'capital_prestado': np.random.normal(5000, 1000, 50),
            'salario_cliente': np.random.normal(3000, 500, 50),
            'edad_cliente': np.random.randint(25, 60, 50),
            'puntaje_datacredito': np.random.normal(0, 1, 50),
            'tipo_credito': np.random.choice([4, 9], 50),
            'tipo_laboral': np.random.choice(['Empleado', 'Independiente'], 50),
            'saldo_mora': np.random.exponential(100, 50),
            'plazo_meses': np.random.randint(6, 36, 50),
        })
        
        y_test = pd.Series(np.random.randint(0, 2, 50))
        
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        
        result = _evaluate_candidate(
            model_name="logistic_regression",
            estimator=LogisticRegression(random_state=42),
            x_train_raw=X_train,
            y_train=y_train,
            x_test_raw=X_test,
            y_test=y_test,
            cv=cv,
        )
        
        # With random labels, we should NOT get perfect scores
        # If we do, there's data leakage
        assert result['test_accuracy'] < 0.9, \
            f"Suspiciously high accuracy {result['test_accuracy']:.3f} on random data suggests leakage"
        assert result['test_f1'] < 0.9, \
            f"Suspiciously high F1 {result['test_f1']:.3f} on random data suggests leakage"
    
    def test_preprocessing_independence(self):
        """Test that preprocessing is fitted only on training data."""
        np.random.seed(42)
        
        # Create train data with NaN values
        X_train = pd.DataFrame({
            'capital_prestado': [1000, 2000, np.nan, 4000, 5000],
            'salario_cliente': [1000, 2000, 3000, 4000, 5000],
            'edad_cliente': [25, 30, 35, np.nan, 45],
            'puntaje_datacredito': [0.5, -0.5, 0.0, 1.0, -1.0],
            'tipo_credito': [4, 4, 9, 4, 9],
            'tipo_laboral': ['Empleado', 'Empleado', 'Independiente', 'Empleado', 'Independiente'],
            'saldo_mora': [0, 0, 100, 0, 50],
            'plazo_meses': [12, 24, 12, 36, 24],
        })
        y_train = pd.Series([1, 1, 0, 1, 0])
        
        # Create test data with VERY different values
        X_test = pd.DataFrame({
            'capital_prestado': [10000, 20000],  # Much larger than train
            'salario_cliente': [10000, 20000],   # Much larger than train
            'edad_cliente': [25, 30],
            'puntaje_datacredito': [5.0, -5.0],  # Much larger magnitude than train
            'tipo_credito': [4, 9],
            'tipo_laboral': ['Empleado', 'Independiente'],
            'saldo_mora': [0, 0],
            'plazo_meses': [12, 24],
        })
        y_test = pd.Series([1, 0])
        
        # Build and fit pipeline
        model = build_model(LogisticRegression(random_state=42))
        model.fit(X_train, y_train)
        
        # Transform test data
        X_test_transformed = model.named_steps['features'].transform(X_test)
        
        # Check that test data transformation doesn't use test statistics
        # If imputation used test data, the imputed values would be influenced by test set
        # This is hard to test directly, but we can check that the pipeline is properly isolated
        
        # The key issue: check if StandardScaler used test data statistics
        if hasattr(model.named_steps['features'], 'named_steps'):
            pipeline_steps = model.named_steps['features'].named_steps
            if 'preprocessor' in pipeline_steps:
                preprocessor = pipeline_steps['preprocessor']
                # Check that scaler statistics come from training data only
                if hasattr(preprocessor, 'transformers_'):
                    for name, transformer, cols in preprocessor.transformers_:
                        if hasattr(transformer, 'named_steps') and 'scaler' in transformer.named_steps:
                            scaler = transformer.named_steps['scaler']
                            # Mean should be close to training data mean, not influenced by test
                            assert scaler.mean_[0] < 7000, \
                                f"Scaler mean {scaler.mean_[0]} suggests test data leakage"
    
    def test_cross_validation_isolation(self):
        """Test that each CV fold is properly isolated."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'capital_prestado': np.random.normal(5000, 1000, n_samples),
            'salario_cliente': np.random.normal(3000, 500, n_samples),
            'edad_cliente': np.random.randint(25, 60, n_samples),
            'puntaje_datacredito': np.random.normal(0, 1, n_samples),
            'tipo_credito': np.random.choice([4, 9], n_samples),
            'tipo_laboral': np.random.choice(['Empleado', 'Independiente'], n_samples),
            'saldo_mora': np.random.exponential(100, n_samples),
            'plazo_meses': np.random.randint(6, 36, n_samples),
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        
        result = _evaluate_candidate(
            model_name="logistic_regression",
            estimator=LogisticRegression(random_state=42),
            x_train_raw=X,
            y_train=y,
            x_test_raw=X.iloc[:20],  # Use subset for test
            y_test=y.iloc[:20],
            cv=cv,
        )
        
        # CV std should be > 0 if folds are different
        assert result['cv_f1_std'] > 0.001, \
            f"CV std = {result['cv_f1_std']:.6f} suggests all folds identical (possible leakage)"


class TestEvaluateCandidate:
    """Tests for _evaluate_candidate function."""
    
    def test_evaluate_candidate_returns_all_metrics(self):
        """Test that all expected metrics are returned."""
        np.random.seed(42)
        n_samples = 100
        
        X_train = pd.DataFrame({
            'capital_prestado': np.random.normal(5000, 1000, n_samples),
            'salario_cliente': np.random.normal(3000, 500, n_samples),
            'edad_cliente': np.random.randint(25, 60, n_samples),
            'puntaje_datacredito': np.random.normal(0, 1, n_samples),
            'tipo_credito': np.random.choice([4, 9], n_samples),
            'tipo_laboral': np.random.choice(['Empleado', 'Independiente'], n_samples),
            'saldo_mora': np.random.exponential(100, n_samples),
            'plazo_meses': np.random.randint(6, 36, n_samples),
        })
        y_train = pd.Series(np.random.randint(0, 2, n_samples))
        
        X_test = X_train.iloc[:20].copy()
        y_test = y_train.iloc[:20].copy()
        
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        
        result = _evaluate_candidate(
            model_name="heuristic",
            estimator=HeuristicModel(),
            x_train_raw=X_train,
            y_train=y_train,
            x_test_raw=X_test,
            y_test=y_test,
            cv=cv,
        )
        
        expected_keys = {
            'model', 'cv_accuracy_mean', 'cv_precision_mean', 'cv_recall_mean',
            'cv_f1_mean', 'cv_f1_std', 'cv_fit_time_mean', 'test_accuracy',
            'test_precision', 'test_recall', 'test_f1', 'test_balanced_accuracy',
            'test_roc_auc', 'selection_score'
        }
        
        assert set(result.keys()) == expected_keys
        assert result['model'] == 'heuristic'
        
        # Check all metrics are valid numbers
        for key, value in result.items():
            if key != 'model':
                assert isinstance(value, (int, float))
                assert not np.isnan(value) or key == 'test_roc_auc'  # ROC AUC can be NaN for HeuristicModel
    
    def test_evaluate_candidate_realistic_scores(self):
        """Test that scores are in realistic ranges (not suspiciously perfect)."""
        np.random.seed(42)
        n_samples = 200
        
        # Create data with some signal but not perfect
        X_train = pd.DataFrame({
            'capital_prestado': np.random.normal(5000, 2000, n_samples),
            'salario_cliente': np.random.normal(3000, 1000, n_samples),
            'edad_cliente': np.random.randint(20, 70, n_samples),
            'puntaje_datacredito': np.random.normal(0, 1.5, n_samples),
            'tipo_credito': np.random.choice([4, 9, 10, 68], n_samples),
            'tipo_laboral': np.random.choice(['Empleado', 'Independiente'], n_samples),
            'saldo_mora': np.random.exponential(200, n_samples),
            'plazo_meses': np.random.randint(3, 60, n_samples),
        })
        
        # Create target with some relationship to features but with noise
        y_train = pd.Series(
            ((X_train['puntaje_datacredito'] > 0) & 
             (X_train['saldo_mora'] < 100)).astype(int)
        )
        # Add noise
        flip_idx = np.random.choice(n_samples, size=int(0.2 * n_samples), replace=False)
        y_train.iloc[flip_idx] = 1 - y_train.iloc[flip_idx]
        
        X_test = X_train.iloc[:50].copy()
        y_test = y_train.iloc[:50].copy()
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        result = _evaluate_candidate(
            model_name="logistic_regression",
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            x_train_raw=X_train,
            y_train=y_train,
            x_test_raw=X_test,
            y_test=y_test,
            cv=cv,
        )
        
        # Scores should be reasonable, not perfect
        assert 0.5 <= result['test_accuracy'] <= 0.95, \
            f"Accuracy {result['test_accuracy']:.3f} outside realistic range"
        assert 0.0 <= result['test_f1'] <= 0.95, \
            f"F1 {result['test_f1']:.3f} outside realistic range"
        assert result['cv_f1_std'] > 0.001, \
            "CV std too low, suggests overfitting or data leakage"


class TestDiagnostics:
    """Diagnostic tests to understand the perfect scores issue."""
    
    def test_imbalanced_dataset_handling(self):
        """Test behavior with highly imbalanced dataset (like your 95% positive class)."""
        np.random.seed(42)
        n_samples = 200
        
        X_train = pd.DataFrame({
            'capital_prestado': np.random.normal(5000, 1000, n_samples),
            'salario_cliente': np.random.normal(3000, 500, n_samples),
            'edad_cliente': np.random.randint(25, 60, n_samples),
            'puntaje_datacredito': np.random.normal(0, 1, n_samples),
            'tipo_credito': np.random.choice([4, 9], n_samples),
            'tipo_laboral': np.random.choice(['Empleado', 'Independiente'], n_samples),
            'saldo_mora': np.random.exponential(100, n_samples),
            'plazo_meses': np.random.randint(6, 36, n_samples),
        })
        
        # Create highly imbalanced dataset: 95% class 1, 5% class 0
        y_train = pd.Series(np.ones(n_samples, dtype=int))
        minority_idx = np.random.choice(n_samples, size=10, replace=False)
        y_train.iloc[minority_idx] = 0
        
        X_test = X_train.iloc[:50].copy()
        y_test = pd.Series(np.ones(50, dtype=int))
        y_test.iloc[:3] = 0  # 94% class 1
        
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        
        result = _evaluate_candidate(
            model_name="logistic_regression",
            estimator=LogisticRegression(class_weight='balanced', random_state=42),
            x_train_raw=X_train,
            y_train=y_train,
            x_test_raw=X_test,
            y_test=y_test,
            cv=cv,
        )
        
        # With imbalanced data, balanced_accuracy should differ from accuracy
        assert abs(result['test_accuracy'] - result['test_balanced_accuracy']) > 0.01, \
            "Accuracy and balanced_accuracy too similar for imbalanced data"
        
        print(f"\nImbalanced dataset results:")
        print(f"  Accuracy: {result['test_accuracy']:.3f}")
        print(f"  Balanced Accuracy: {result['test_balanced_accuracy']:.3f}")
        print(f"  F1: {result['test_f1']:.3f}")
        print(f"  Precision: {result['test_precision']:.3f}")
        print(f"  Recall: {result['test_recall']:.3f}")
    
    def test_print_diagnostic_info(self, capsys):
        """Print diagnostic information about the evaluation process."""
        np.random.seed(42)
        n_samples = 100
        
        X_train = pd.DataFrame({
            'capital_prestado': np.random.normal(5000, 1000, n_samples),
            'salario_cliente': np.random.normal(3000, 500, n_samples),
            'edad_cliente': np.random.randint(25, 60, n_samples),
            'puntaje_datacredito': np.random.normal(0, 1, n_samples),
            'tipo_credito': np.random.choice([4, 9], n_samples),
            'tipo_laboral': np.random.choice(['Empleado', 'Independiente'], n_samples),
            'saldo_mora': np.random.exponential(100, n_samples),
            'plazo_meses': np.random.randint(6, 36, n_samples),
        })
        y_train = pd.Series(np.random.randint(0, 2, n_samples))
        
        print("\n" + "="*60)
        print("DIAGNOSTIC INFORMATION")
        print("="*60)
        print(f"\nTraining set shape: {X_train.shape}")
        print(f"Class distribution: {y_train.value_counts().to_dict()}")
        print(f"Class imbalance ratio: {y_train.value_counts()[1] / y_train.value_counts()[0]:.2f}")
        
        # Check for duplicate rows
        duplicates = X_train.duplicated().sum()
        print(f"Duplicate rows in training: {duplicates}")
        
        # Check feature statistics
        print(f"\nFeature statistics:")
        print(X_train.describe())