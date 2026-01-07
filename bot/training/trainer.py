"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Trainer
نظام التدريب المتكامل
═══════════════════════════════════════════════════════════════
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
from loguru import logger

# إضافة المسار الرئيسي
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tft_model import TemporalFusionTransformer
from models.lstm_attention import LSTMWithAttention
from models.market_regime import MarketRegimeClassifier
from models.ensemble import EnsembleModel


class Trainer:
    """
    نظام التدريب المتكامل
    
    يدير تدريب جميع النماذج
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        output_dir: str = None,
        device: str = None
    ):
        """
        تهيئة المدرب
        
        Args:
            config: إعدادات التدريب
            output_dir: مجلد المخرجات
            device: الجهاز (cpu/cuda)
        """
        self.config = config or self._default_config()
        self.output_dir = Path(output_dir or '/tmp/legendary_agent/models')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # تحديد الجهاز
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"🏋️ Trainer initialized on {self.device}")
        
        # النماذج
        self.models: Dict[str, nn.Module] = {}
        self.training_history: Dict[str, List] = {}
    
    def _default_config(self) -> Dict[str, Any]:
        """الإعدادات الافتراضية"""
        return {
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'early_stopping_patience': 10,
            'validation_split': 0.2,
            'sequence_length': 60,
            'num_features': 50,
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.2
        }
    
    # ═══════════════════════════════════════════════════════════════
    # DATA PREPARATION
    # ═══════════════════════════════════════════════════════════════
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'target',
        sequence_length: int = None
    ) -> Tuple[DataLoader, DataLoader]:
        """
        تحضير البيانات للتدريب
        
        Args:
            df: البيانات
            target_col: عمود الهدف
            sequence_length: طول التسلسل
            
        Returns:
            DataLoader للتدريب والتحقق
        """
        seq_len = sequence_length or self.config['sequence_length']
        
        # فصل الميزات والهدف
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].values
        y = df[target_col].values if target_col in df.columns else np.zeros(len(df))
        
        # إنشاء التسلسلات
        X_seq, y_seq = self._create_sequences(X, y, seq_len)
        
        # تقسيم البيانات
        split_idx = int(len(X_seq) * (1 - self.config['validation_split']))
        
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # تحويل إلى Tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        logger.info(f"📊 Data prepared: {len(train_dataset)} train, {len(val_dataset)} val")
        
        return train_loader, val_loader
    
    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """إنشاء التسلسلات"""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    # ═══════════════════════════════════════════════════════════════
    # MODEL TRAINING
    # ═══════════════════════════════════════════════════════════════
    
    def train_tft(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_features: int = None
    ) -> TemporalFusionTransformer:
        """
        تدريب نموذج TFT
        
        Args:
            train_loader: بيانات التدريب
            val_loader: بيانات التحقق
            num_features: عدد الميزات
            
        Returns:
            النموذج المدرب
        """
        logger.info("🔄 Training Temporal Fusion Transformer...")
        
        # تحديد عدد الميزات
        sample_batch = next(iter(train_loader))
        n_features = num_features or sample_batch[0].shape[-1]
        seq_len = sample_batch[0].shape[1]
        
        # إنشاء النموذج
        model = TemporalFusionTransformer(
            num_features=n_features,
            hidden_dim=self.config['hidden_dim'],
            num_heads=4,
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # التدريب
        history = self._train_model(
            model, train_loader, val_loader,
            model_name='tft'
        )
        
        self.models['tft'] = model
        self.training_history['tft'] = history
        
        # حفظ النموذج
        self._save_model(model, 'tft_model.pt')
        
        return model
    
    def train_lstm(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_features: int = None
    ) -> LSTMWithAttention:
        """
        تدريب نموذج LSTM
        
        Args:
            train_loader: بيانات التدريب
            val_loader: بيانات التحقق
            num_features: عدد الميزات
            
        Returns:
            النموذج المدرب
        """
        logger.info("🔄 Training LSTM with Attention...")
        
        # تحديد عدد الميزات
        sample_batch = next(iter(train_loader))
        n_features = num_features or sample_batch[0].shape[-1]
        
        # إنشاء النموذج
        model = LSTMWithAttention(
            input_dim=n_features,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # التدريب
        history = self._train_model(
            model, train_loader, val_loader,
            model_name='lstm'
        )
        
        self.models['lstm'] = model
        self.training_history['lstm'] = history
        
        # حفظ النموذج
        self._save_model(model, 'lstm_model.pt')
        
        return model
    
    def train_regime_classifier(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_features: int = None
    ) -> MarketRegimeClassifier:
        """
        تدريب مصنف حالة السوق
        
        Args:
            train_loader: بيانات التدريب
            val_loader: بيانات التحقق
            num_features: عدد الميزات
            
        Returns:
            النموذج المدرب
        """
        logger.info("🔄 Training Market Regime Classifier...")
        
        # تحديد عدد الميزات
        sample_batch = next(iter(train_loader))
        n_features = num_features or sample_batch[0].shape[-1]
        
        # إنشاء النموذج
        model = MarketRegimeClassifier(
            input_dim=n_features,
            hidden_dim=self.config['hidden_dim'],
            num_regimes=7,
            dropout=self.config['dropout']
        ).to(self.device)
        
        # التدريب
        history = self._train_model(
            model, train_loader, val_loader,
            model_name='regime',
            is_classification=True
        )
        
        self.models['regime'] = model
        self.training_history['regime'] = history
        
        # حفظ النموذج
        self._save_model(model, 'regime_model.pt')
        
        return model
    
    def _train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str,
        is_classification: bool = False
    ) -> List[Dict]:
        """تدريب نموذج"""
        # تحديد دالة الخسارة
        if is_classification:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        history = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # التدريب
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                if is_classification:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y.long())
                else:
                    outputs = model(batch_X)
                    if outputs.dim() > 1:
                        outputs = outputs.squeeze(-1)
                    loss = criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # التحقق
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    if is_classification:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y.long())
                    else:
                        outputs = model(batch_X)
                        if outputs.dim() > 1:
                            outputs = outputs.squeeze(-1)
                        loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # تحديث المجدول
            scheduler.step(val_loss)
            
            # حفظ التاريخ
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # طباعة التقدم
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{self.config['epochs']} - "
                    f"Train: {train_loss:.6f}, Val: {val_loss:.6f}"
                )
            
            # التوقف المبكر
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # حفظ أفضل نموذج
                self._save_model(model, f'{model_name}_best.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                    break
        
        logger.info(f"✅ {model_name} training complete. Best val loss: {best_val_loss:.6f}")
        
        return history
    
    # ═══════════════════════════════════════════════════════════════
    # ENSEMBLE TRAINING
    # ═══════════════════════════════════════════════════════════════
    
    def train_ensemble(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> EnsembleModel:
        """
        تدريب نموذج الدمج
        
        Args:
            train_loader: بيانات التدريب
            val_loader: بيانات التحقق
            
        Returns:
            نموذج الدمج
        """
        logger.info("🔄 Training Ensemble Model...")
        
        # التأكد من وجود النماذج الفرعية
        if not self.models:
            raise ValueError("No models trained yet. Train individual models first.")
        
        # تحديد عدد الميزات
        sample_batch = next(iter(train_loader))
        n_features = sample_batch[0].shape[-1]
        seq_len = sample_batch[0].shape[1]
        
        # إنشاء نموذج الدمج
        ensemble = EnsembleModel(
            num_features=n_features,
            sequence_length=seq_len,
            hidden_dim=self.config['hidden_dim']
        ).to(self.device)
        
        # نسخ النماذج المدربة
        if 'tft' in self.models:
            ensemble.tft = self.models['tft']
        if 'lstm' in self.models:
            ensemble.lstm = self.models['lstm']
        if 'regime' in self.models:
            ensemble.regime_classifier = self.models['regime']
        
        # تدريب طبقة الدمج فقط
        for param in ensemble.tft.parameters():
            param.requires_grad = False
        for param in ensemble.lstm.parameters():
            param.requires_grad = False
        for param in ensemble.regime_classifier.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.Adam(
            ensemble.fusion_layer.parameters(),
            lr=self.config['learning_rate']
        )
        criterion = nn.MSELoss()
        
        for epoch in range(20):  # تدريب قصير للدمج
            ensemble.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = ensemble(batch_X)
                loss = criterion(outputs['final_prediction'].squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"  Ensemble Epoch {epoch+1}/20 - Loss: {train_loss/len(train_loader):.6f}")
        
        self.models['ensemble'] = ensemble
        self._save_model(ensemble, 'ensemble_model.pt')
        
        logger.info("✅ Ensemble training complete")
        
        return ensemble
    
    # ═══════════════════════════════════════════════════════════════
    # FULL TRAINING PIPELINE
    # ═══════════════════════════════════════════════════════════════
    
    def train_all(
        self,
        df: pd.DataFrame,
        target_col: str = 'target'
    ) -> Dict[str, nn.Module]:
        """
        تدريب جميع النماذج
        
        Args:
            df: البيانات
            target_col: عمود الهدف
            
        Returns:
            جميع النماذج المدربة
        """
        logger.info("🚀 Starting full training pipeline...")
        start_time = datetime.now()
        
        # تحضير البيانات
        train_loader, val_loader = self.prepare_data(df, target_col)
        
        # تدريب النماذج الفردية
        self.train_tft(train_loader, val_loader)
        self.train_lstm(train_loader, val_loader)
        
        # تحضير بيانات التصنيف
        # (يمكن استخدام نفس البيانات مع تحويل الهدف)
        self.train_regime_classifier(train_loader, val_loader)
        
        # تدريب الدمج
        self.train_ensemble(train_loader, val_loader)
        
        duration = datetime.now() - start_time
        logger.info(f"🎉 Full training complete in {duration}")
        
        return self.models
    
    # ═══════════════════════════════════════════════════════════════
    # SAVE/LOAD
    # ═══════════════════════════════════════════════════════════════
    
    def _save_model(self, model: nn.Module, filename: str) -> None:
        """حفظ نموذج"""
        path = self.output_dir / filename
        torch.save(model.state_dict(), path)
        logger.info(f"💾 Model saved: {path}")
    
    def load_model(self, model_class: type, filename: str, **kwargs) -> nn.Module:
        """تحميل نموذج"""
        path = self.output_dir / filename
        model = model_class(**kwargs)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def get_training_summary(self) -> Dict[str, Any]:
        """ملخص التدريب"""
        summary = {
            'models_trained': list(self.models.keys()),
            'device': str(self.device),
            'config': self.config
        }
        
        for name, history in self.training_history.items():
            if history:
                summary[f'{name}_final_train_loss'] = history[-1]['train_loss']
                summary[f'{name}_final_val_loss'] = history[-1]['val_loss']
                summary[f'{name}_best_val_loss'] = min(h['val_loss'] for h in history)
        
        return summary


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار المدرب
    trainer = Trainer()
    
    # إنشاء بيانات وهمية للاختبار
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    data = np.random.randn(n_samples, n_features)
    target = np.random.randn(n_samples)
    
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = target
    
    print("📊 Sample data shape:", df.shape)
    
    # تحضير البيانات
    train_loader, val_loader = trainer.prepare_data(df)
    
    # تدريب نموذج واحد للاختبار
    print("\n🔄 Training TFT model...")
    tft = trainer.train_tft(train_loader, val_loader)
    
    print("\n📈 Training Summary:")
    print(trainer.get_training_summary())
