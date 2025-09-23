"""
CTR Prediction Model - Criteo Dataset Optimized
Focus: Feature Engineering, Model Training, Evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
import hashlib


class CTRFeatureEngineering:
    """Feature engineering pipeline for Criteo CTR dataset"""

    def __init__(self, hash_bucket_size: int = 1000000):
        """
        Initialize feature engineering
        Criteo dataset: 13 numerical + 26 categorical features
        """
        self.hash_bucket_size = hash_bucket_size
        self.numerical_cols = [f'num_{i}' for i in range(1, 14)]
        self.categorical_cols = [f'cat_{i}' for i in range(1, 27)]
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def hash_trick(self, value: str, bucket_size: int = None) -> int:
        """
        Hash trick for high-cardinality categoricals
        Real use: Criteo has billions of unique values
        """
        if bucket_size is None:
            bucket_size = self.hash_bucket_size

        if pd.isna(value):
            return 0

        hash_val = int(hashlib.md5(str(value).encode()).hexdigest(), 16)
        return hash_val % bucket_size

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        Time: O(n), Space: O(n)
        """
        df = df.copy()

        # 1. Handle missing values in numerical features
        for col in self.numerical_cols:
            if col in df.columns:
                # Fill with median or use -1 as missing indicator
                df[col] = df[col].fillna(-1)

                # Log transform for skewed features
                if df[col].min() >= 0:
                    df[f'{col}_log'] = np.log1p(df[col])

                # Binning for non-linear patterns
                df[f'{col}_bin'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')

        # 2. Hash trick for categorical features
        for col in self.categorical_cols:
            if col in df.columns:
                df[f'{col}_hash'] = df[col].apply(lambda x: self.hash_trick(x))

                # Count encoding (frequency)
                freq = df[col].value_counts().to_dict()
                df[f'{col}_freq'] = df[col].map(freq).fillna(0)

        # 3. Feature interactions (top performing)
        if 'cat_1' in df.columns and 'cat_2' in df.columns:
            df['cat_1_2_interaction'] = df['cat_1'].astype(str) + '_' + df['cat_2'].astype(str)
            df['cat_1_2_hash'] = df['cat_1_2_interaction'].apply(lambda x: self.hash_trick(x))

        # 4. Numerical interactions
        if 'num_1' in df.columns and 'num_2' in df.columns:
            df['num_1_2_product'] = df['num_1'] * df['num_2']
            df['num_1_2_ratio'] = df['num_1'] / (df['num_2'] + 1e-8)

        # 5. Statistical features
        numerical_features = [col for col in df.columns if col.startswith('num_')]
        if numerical_features:
            df['num_mean'] = df[numerical_features].mean(axis=1)
            df['num_std'] = df[numerical_features].std(axis=1)
            df['num_max'] = df[numerical_features].max(axis=1)
            df['num_min'] = df[numerical_features].min(axis=1)

        return df

    def create_crossing_features(self, df: pd.DataFrame,
                                crosses: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create feature crosses (inspired by Wide & Deep)
        """
        for col1, col2 in crosses:
            if col1 in df.columns and col2 in df.columns:
                cross_name = f'{col1}_X_{col2}'
                df[cross_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
                df[f'{cross_name}_hash'] = df[cross_name].apply(lambda x: self.hash_trick(x, 100000))

        return df


class CTRModel:
    """CTR prediction model with multiple algorithms"""

    def __init__(self, model_type: str = 'logistic'):
        """
        Initialize model
        Options: logistic, gbm, deep
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = {}

    def build_logistic_model(self):
        """
        Logistic Regression baseline
        Fast, interpretable, good for real-time
        """
        from sklearn.linear_model import SGDClassifier

        self.model = SGDClassifier(
            loss='log',
            penalty='l2',
            alpha=1e-4,
            max_iter=1000,
            learning_rate='optimal',
            n_jobs=-1
        )

    def build_gbm_model(self):
        """
        LightGBM for better accuracy
        Handles categoricals natively
        """
        import lightgbm as lgb

        self.model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=255,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=0,
            n_jobs=-1,
            max_depth=8,
            lambda_l1=0.1,
            lambda_l2=0.1
        )

    def build_deep_model(self, input_dim: int):
        """
        Deep learning model (simplified DeepFM)
        """
        import tensorflow as tf
        from tensorflow import keras

        # Wide part (linear)
        wide_input = keras.Input(shape=(input_dim,))
        wide_output = keras.layers.Dense(1, activation='linear')(wide_input)

        # Deep part
        deep_input = keras.Input(shape=(input_dim,))
        deep_hidden1 = keras.layers.Dense(400, activation='relu')(deep_input)
        deep_dropout1 = keras.layers.Dropout(0.3)(deep_hidden1)
        deep_hidden2 = keras.layers.Dense(400, activation='relu')(deep_dropout1)
        deep_dropout2 = keras.layers.Dropout(0.3)(deep_hidden2)
        deep_hidden3 = keras.layers.Dense(400, activation='relu')(deep_dropout2)
        deep_output = keras.layers.Dense(1, activation='linear')(deep_hidden3)

        # Combine Wide & Deep
        combined = keras.layers.add([wide_output, deep_output])
        output = keras.layers.Activation('sigmoid')(combined)

        self.model = keras.Model(inputs=[wide_input, deep_input], outputs=output)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['AUC']
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        if self.model_type == 'logistic':
            self.build_logistic_model()
            self.model.fit(X_train, y_train)

        elif self.model_type == 'gbm':
            self.build_gbm_model()
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=50,
                verbose=False
            )
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(
                    range(len(self.model.feature_importances_)),
                    self.model.feature_importances_
                ))

        elif self.model_type == 'deep':
            self.build_deep_model(X_train.shape[1])
            validation_data = (([X_val, X_val], y_val)) if X_val is not None else None
            self.model.fit(
                [X_train, X_train], y_train,
                validation_data=validation_data,
                epochs=10,
                batch_size=256,
                verbose=0
            )

    def predict_proba(self, X):
        """Predict CTR probabilities"""
        if self.model_type == 'deep':
            return self.model.predict([X, X])[:, 0]
        elif hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            # For SGDClassifier with log loss
            return 1 / (1 + np.exp(-self.model.decision_function(X)))


class CTREvaluation:
    """Comprehensive evaluation metrics for CTR models"""

    @staticmethod
    def calculate_metrics(y_true, y_pred_proba) -> Dict:
        """
        Calculate all relevant metrics
        Criteo benchmark: LogLoss < 0.44, AUC > 0.80
        """
        metrics = {}

        # LogLoss (primary metric for Criteo)
        metrics['logloss'] = log_loss(y_true, y_pred_proba)

        # AUC-ROC
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)

        # Calibration metrics
        metrics['mean_pred'] = np.mean(y_pred_proba)
        metrics['mean_true'] = np.mean(y_true)
        metrics['calibration_ratio'] = metrics['mean_pred'] / (metrics['mean_true'] + 1e-10)

        # Brier score (calibration)
        metrics['brier_score'] = np.mean((y_pred_proba - y_true) ** 2)

        # Percentile analysis
        metrics['p90_score'] = np.percentile(y_pred_proba, 90)
        metrics['p99_score'] = np.percentile(y_pred_proba, 99)

        return metrics

    @staticmethod
    def plot_calibration_curve(y_true, y_pred_proba, n_bins=10):
        """
        Calibration plot (reliability diagram)
        Important for bidding decisions
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        empirical_probs = []
        counts = []

        for i in range(n_bins):
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                empirical_probs.append(y_true[mask].mean())
                counts.append(mask.sum())

        return bin_centers, empirical_probs, counts

    @staticmethod
    def calculate_lift(y_true, y_pred_proba, percentile=10):
        """
        Calculate lift at top percentile
        Real use: Target top 10% most likely clickers
        """
        threshold = np.percentile(y_pred_proba, 100 - percentile)
        top_mask = y_pred_proba >= threshold

        baseline_ctr = y_true.mean()
        top_ctr = y_true[top_mask].mean()

        lift = top_ctr / (baseline_ctr + 1e-10)
        return lift


def create_sample_pipeline():
    """
    Complete pipeline example for interview
    """
    print("=== CTR Prediction Pipeline ===\n")

    # 1. Generate sample data (simulating Criteo dataset)
    np.random.seed(42)
    n_samples = 10000

    data = pd.DataFrame()

    # Numerical features
    for i in range(1, 14):
        data[f'num_{i}'] = np.random.lognormal(0, 2, n_samples)

    # Categorical features
    for i in range(1, 27):
        n_categories = np.random.randint(10, 1000)
        data[f'cat_{i}'] = np.random.choice([f'c{j}' for j in range(n_categories)], n_samples)

    # Target (click or not)
    data['click'] = np.random.binomial(1, 0.05, n_samples)  # 5% CTR

    print(f"Dataset shape: {data.shape}")
    print(f"CTR: {data['click'].mean():.2%}\n")

    # 2. Feature engineering
    print("Performing feature engineering...")
    fe = CTRFeatureEngineering()
    features = fe.create_features(data)

    # Select features
    feature_cols = [col for col in features.columns if col != 'click' and not col.startswith('cat_')]
    X = features[feature_cols].fillna(0).values
    y = data['click'].values

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}\n")

    # 4. Train models
    results = {}

    for model_type in ['logistic', 'gbm']:
        print(f"Training {model_type} model...")

        model = CTRModel(model_type)
        model.train(X_train, y_train, X_test, y_test)

        # Predictions
        y_pred = model.predict_proba(X_test)

        # Evaluation
        evaluator = CTREvaluation()
        metrics = evaluator.calculate_metrics(y_test, y_pred)
        lift = evaluator.calculate_lift(y_test, y_pred, percentile=10)

        results[model_type] = metrics
        results[model_type]['lift_top10'] = lift

        print(f"  LogLoss: {metrics['logloss']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Calibration ratio: {metrics['calibration_ratio']:.2f}")
        print(f"  Top 10% lift: {lift:.2f}x\n")

    # 5. Compare models
    print("=== Model Comparison ===")
    comparison_df = pd.DataFrame(results).T
    print(comparison_df[['logloss', 'auc', 'calibration_ratio', 'lift_top10']])

    return results


if __name__ == "__main__":
    # Run the pipeline
    results = create_sample_pipeline()

    print("\n✅ CTR module ready!")
    print("Key achievements:")
    print("- Feature engineering with hash trick")
    print("- Multiple model architectures")
    print("- Comprehensive evaluation metrics")
    print("- Calibration analysis for bidding")