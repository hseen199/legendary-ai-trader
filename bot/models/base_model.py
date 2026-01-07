"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Base Model
الكلاس الأساسي لجميع النماذج
═══════════════════════════════════════════════════════════════
"""

import os
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
from loguru import logger


class BaseModel(ABC, nn.Module):
    """
    الكلاس الأساسي لجميع نماذج التعلم الآلي
    يوفر واجهة موحدة للتدريب والتنبؤ والحفظ
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        تهيئة النموذج الأساسي
        
        Args:
            name: اسم النموذج
            config: إعدادات النموذج
        """
        super().__init__()
        self.name = name
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trained = False
        self.training_history: List[Dict] = []
        self.best_loss = float('inf')
        
        logger.info(f"🧠 {name} initialized on {self.device}")
    
    # ═══════════════════════════════════════════════════════════════
    # ABSTRACT METHODS - يجب تنفيذها في الكلاسات الفرعية
    # ═══════════════════════════════════════════════════════════════
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """التمرير الأمامي"""
        pass
    
    @abstractmethod
    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        """التنبؤ"""
        pass
    
    @abstractmethod
    def get_input_shape(self) -> Tuple[int, ...]:
        """الحصول على شكل المدخلات"""
        pass
    
    # ═══════════════════════════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════════════════════════
    
    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        save_best: bool = True,
        save_dir: str = "checkpoints"
    ) -> Dict[str, List[float]]:
        """
        تدريب النموذج
        
        Args:
            train_loader: محمّل بيانات التدريب
            val_loader: محمّل بيانات التحقق
            epochs: عدد الحقب
            learning_rate: معدل التعلم
            early_stopping_patience: صبر التوقف المبكر
            save_best: حفظ أفضل نموذج
            save_dir: مجلد الحفظ
            
        Returns:
            تاريخ التدريب
        """
        self.to(self.device)
        self.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = self._get_loss_function()
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"🚀 Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader:
                val_loss = self._validate_epoch(val_loader, criterion)
                history['val_loss'].append(val_loss)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_best:
                        self._save_checkpoint(save_dir, "best")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"⏹️ Early stopping at epoch {epoch+1}")
                        break
                
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f}"
                )
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f}")
        
        self.is_trained = True
        self.training_history = history
        self.best_loss = best_val_loss if val_loader else train_loss
        
        logger.info(f"✅ Training completed. Best loss: {self.best_loss:.6f}")
        return history
    
    def _train_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """تدريب حقبة واحدة"""
        self.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.forward(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> float:
        """التحقق من حقبة واحدة"""
        self.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.forward(batch_x)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _get_loss_function(self) -> nn.Module:
        """الحصول على دالة الخسارة"""
        return nn.MSELoss()
    
    # ═══════════════════════════════════════════════════════════════
    # SAVE & LOAD
    # ═══════════════════════════════════════════════════════════════
    
    def _save_checkpoint(self, save_dir: str, tag: str = "latest") -> str:
        """حفظ نقطة تفتيش"""
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{self.name}_{tag}.pt")
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filepath)
        logger.debug(f"💾 Checkpoint saved: {filepath}")
        return filepath
    
    def save(self, filepath: str) -> None:
        """حفظ النموذج"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'is_trained': self.is_trained,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"💾 Model saved: {filepath}")
    
    def load(self, filepath: str) -> None:
        """تحميل النموذج"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = checkpoint.get('is_trained', True)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"📂 Model loaded: {filepath}")
    
    # ═══════════════════════════════════════════════════════════════
    # ONNX EXPORT
    # ═══════════════════════════════════════════════════════════════
    
    def export_onnx(
        self,
        filepath: str,
        input_shape: Optional[Tuple[int, ...]] = None,
        opset_version: int = 14
    ) -> str:
        """
        تصدير النموذج إلى ONNX
        
        Args:
            filepath: مسار الملف
            input_shape: شكل المدخلات
            opset_version: إصدار ONNX
            
        Returns:
            مسار الملف المُصدّر
        """
        self.eval()
        self.to("cpu")
        
        if input_shape is None:
            input_shape = self.get_input_shape()
        
        # إنشاء مدخلات وهمية
        dummy_input = torch.randn(1, *input_shape)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.onnx.export(
            self,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info(f"📦 Model exported to ONNX: {filepath}")
        return filepath
    
    # ═══════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════
    
    def count_parameters(self) -> int:
        """عد المعاملات"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """الحصول على معلومات النموذج"""
        return {
            'name': self.name,
            'parameters': self.count_parameters(),
            'is_trained': self.is_trained,
            'best_loss': self.best_loss,
            'device': str(self.device),
            'config': self.config
        }
    
    def to_device(self, device: str = None) -> 'BaseModel':
        """نقل النموذج إلى جهاز"""
        if device:
            self.device = torch.device(device)
        return self.to(self.device)


class TradingDataset(torch.utils.data.Dataset):
    """
    مجموعة بيانات للتداول
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        تهيئة مجموعة البيانات
        
        Args:
            X: المدخلات
            y: المخرجات
            transform: تحويلات اختيارية
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y.dtype != np.int64 else torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


def create_data_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 64,
    shuffle: bool = True
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    إنشاء محمّلات البيانات
    
    Args:
        X_train: بيانات التدريب
        y_train: تصنيفات التدريب
        X_val: بيانات التحقق
        y_val: تصنيفات التحقق
        batch_size: حجم الدفعة
        shuffle: خلط البيانات
        
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = TradingDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TradingDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    return train_loader, val_loader
