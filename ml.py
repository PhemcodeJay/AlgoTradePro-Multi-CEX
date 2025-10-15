import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
import os
import sys
import argparse
from db import DatabaseManager, FeedbackModel
from numpy.typing import NDArray
from logging_config import get_trading_logger

logger = get_trading_logger(__name__)

# Load exchange and mode
EXCHANGE = os.getenv("EXCHANGE", "binance").lower()
TRADING_MODE = os.getenv("TRADING_MODE", "virtual").lower()

class MLFilter:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'rsi', 'stoch_rsi_k', 'stoch_rsi_d', 'macd', 'macd_signal',
            'macd_histogram', 'bb_upper', 'bb_middle', 'bb_lower',
            'sma_200', 'ema_9', 'volume_ratio', 'price_change_1h',
            'price_change_4h', 'price_change_24h'
        ]
        self.model_path = f"models/ml_model_{EXCHANGE}.joblib"
        self.scaler_path = f"models/ml_scaler_{EXCHANGE}.joblib"
        
        # Load existing model if available
        self.load_model()

    def prepare_features(self, indicators: Dict[str, Any]) -> Optional[NDArray[np.float64]]:
        """Prepare features from indicators for ML prediction"""
        try:
            # Extract indicators matching those from indicators.py
            features = []
            for key in self.feature_columns:
                value = indicators.get(key, 0.0)
                if value is None:
                    logger.warning(f"Missing or None value for {key} in indicators")
                    value = 0.0
                features.append(float(value))
            
            features_array = np.array(features).reshape(1, -1)
            return features_array
            
        except Exception as e:
            logger.error(f"Error preparing features for {EXCHANGE}: {e}")
            return None

    def train_model(self, training_data: List[Dict]) -> bool:
        """Train the ML model with historical trade data"""
        try:
            if len(training_data) < 100:
                logger.warning(f"Insufficient training data for {EXCHANGE}: {len(training_data)} samples")
                return False
            
            X = []
            y = []
            
            for data in training_data:
                features = self.prepare_features(data['indicators'])
                if features is not None and features.shape[1] == len(self.feature_columns):
                    X.append(features.flatten())
                    y.append(1 if data.get('pnl', 0) > 0 else 0)
            
            if len(X) < 50:
                logger.error(f"Not enough valid training samples for {EXCHANGE}: {len(X)}")
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model accuracy for {EXCHANGE}: {accuracy:.2f}")
            logger.debug(f"Classification report:\n{classification_report(y_test, y_pred)}")
            
            # Save model
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {EXCHANGE}: {e}")
            return False

    def filter_signals(self, signals: List[Dict[str, Any]], threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Filter signals using ML model"""
        try:
            if self.model is None:
                logger.warning(f"No model loaded for {EXCHANGE}, skipping ML filter")
                return signals
            
            filtered = []
            for signal in signals:
                features = self.prepare_features(signal.get('indicators', {}))
                if features is None or features.shape[1] != len(self.feature_columns):
                    logger.warning(f"Invalid features for signal {signal.get('symbol', 'UNKNOWN')}")
                    continue
                
                features_scaled = self.scaler.transform(features)
                prob = self.model.predict_proba(features_scaled)[0][1]  # Probability of positive outcome
                
                signal['ml_score'] = float(prob)
                if prob >= threshold:
                    filtered.append(signal)
            
            logger.info(f"Filtered {len(filtered)}/{len(signals)} signals for {EXCHANGE}")
            return filtered
            
        except Exception as e:
            logger.error(f"Error filtering signals for {EXCHANGE}: {e}")
            return signals

    def save_model(self) -> bool:
        """Save trained model and scaler"""
        try:
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Model and scaler saved for {EXCHANGE} at {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model for {EXCHANGE}: {e}")
            return False

    def load_model(self) -> bool:
        """Load saved model and scaler"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"ML model and scaler loaded for {EXCHANGE} from {self.model_path}")
                return True
            logger.warning(f"No model or scaler found at {self.model_path} or {self.scaler_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading model for {EXCHANGE}: {e}")
            return False

    def update_model_with_feedback(self, signal: Dict, outcome: bool):
        """Update model with feedback from trade outcome"""
        try:
            logger.info(f"Feedback received for {signal.get('symbol')} on {EXCHANGE}: {outcome}")
            
            db_manager = DatabaseManager()
            with db_manager.get_session() as session:
                feedback_model = FeedbackModel(
                    symbol=signal.get('symbol', 'UNKNOWN'),
                    exchange=EXCHANGE,
                    outcome=outcome,
                    pnl=signal.get('pnl', 0),
                    indicators=signal.get('indicators', {})
                )
                db_manager.add_feedback(feedback_model)
                
                # Retrain if enough new feedback
                feedback = db_manager.get_feedback(limit=100, exchange=EXCHANGE)
                if len(feedback) >= 50:
                    training_data = [
                        {'indicators': f.indicators, 'pnl': f.pnl}
                        for f in feedback if f.indicators is not None and f.pnl is not None
                    ]
                    self.train_model(training_data)
            
        except Exception as e:
            logger.error(f"Error updating model with feedback for {EXCHANGE}: {e}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if self.model is None:
                logger.warning(f"No trained model available for {EXCHANGE}")
                return {}
            
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_columns, importance))
            
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error getting feature importance for {EXCHANGE}: {e}")
            return {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f"ML Signal Filter for {EXCHANGE.capitalize()}")
    parser.add_argument('--train', action='store_true', help="Train the model with trade data from database")
    parser.add_argument('--threshold', type=float, default=0.6, help="ML score threshold for filtering")
    args = parser.parse_args()

    db_manager = DatabaseManager()
    ml_filter = MLFilter()

    try:
        if args.train:
            # Fetch trades from database for training, filter by exchange and include both modes
            with db_manager.get_session() as session:
                trades = db_manager.get_trades(limit=1000, exchange=EXCHANGE)
                if not trades:
                    logger.error(f"No trades found in database for training on {EXCHANGE}")
                    print(json.dumps({'success': False, 'error': 'No trades found'}))
                    sys.exit(1)
                
                training_data = [
                    {
                        'indicators': trade.indicators,
                        'pnl': trade.pnl
                    }
                    for trade in trades
                    if trade.indicators is not None and trade.pnl is not None
                ]
                
                success = ml_filter.train_model(training_data)
                print(json.dumps({'success': success}))
        else:
            # Fetch signals from database
            with db_manager.get_session() as session:
                signals = [signal.to_dict() for signal in db_manager.get_signals(limit=100, exchange=EXCHANGE)]
                if not signals:
                    logger.error(f"No signals found in database for {EXCHANGE}")
                    print(json.dumps([]))
                    sys.exit(0)
                
                # Filter signals
                filtered_signals = ml_filter.filter_signals(signals, threshold=args.threshold)
                print(json.dumps(filtered_signals, indent=2))
            
    except Exception as e:
        logger.error(f"Error in main execution for {EXCHANGE}: {e}")
        print(json.dumps({'error': str(e)}))
        sys.exit(1)