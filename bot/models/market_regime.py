"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Market Regime Classifier
مصنف حالة السوق
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from enum import Enum

from .base_model import BaseModel


class MarketRegime(Enum):
    """حالات السوق"""
    STRONG_BULLISH = 0
    BULLISH = 1
    NEUTRAL = 2
    BEARISH = 3
    STRONG_BEARISH = 4
    HIGH_VOLATILITY = 5
    LOW_VOLATILITY = 6


class ConvBlock(nn.Module):
    """كتلة التفاف"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class MarketRegimeClassifier(BaseModel):
    """
    مصنف حالة السوق
    
    يحدد حالة السوق الحالية:
    - صعود قوي / صعود / محايد / هبوط / هبوط قوي
    - تقلب عالي / تقلب منخفض
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_regimes: int = 7,
        sequence_length: int = 60,
        dropout: float = 0.2,
        config: Optional[Dict] = None
    ):
        """
        تهيئة المصنف
        
        Args:
            input_dim: عدد الميزات
            hidden_dim: حجم الطبقة المخفية
            num_regimes: عدد حالات السوق
            sequence_length: طول التسلسل
            dropout: نسبة الإسقاط
            config: إعدادات إضافية
        """
        config = config or {}
        config.update({
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_regimes': num_regimes,
            'sequence_length': sequence_length,
            'dropout': dropout
        })
        
        super().__init__("MarketRegimeClassifier", config)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_regimes = num_regimes
        self.sequence_length = sequence_length
        
        # طبقات الالتفاف
        self.conv_layers = nn.Sequential(
            ConvBlock(input_dim, hidden_dim, 7, dropout),
            ConvBlock(hidden_dim, hidden_dim * 2, 5, dropout),
            ConvBlock(hidden_dim * 2, hidden_dim * 2, 3, dropout),
        )
        
        # تجميع عالمي
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # LSTM للتبعيات الزمنية
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # طبقات التصنيف
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2 + hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, num_regimes)
        )
        
        # طبقة الثقة
        self.confidence_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2 + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"🧠 Market Regime Classifier: {self.count_parameters():,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        التمرير الأمامي
        
        Args:
            x: المدخلات [batch, sequence, features]
            
        Returns:
            المخرجات [batch, num_regimes]
        """
        # تحويل لـ Conv1d: [batch, features, sequence]
        x = x.transpose(1, 2)
        
        # طبقات الالتفاف
        conv_out = self.conv_layers(x)  # [batch, hidden*2, seq]
        
        # التجميع العالمي
        avg_pool = self.global_avg_pool(conv_out).squeeze(-1)  # [batch, hidden*2]
        max_pool = self.global_max_pool(conv_out).squeeze(-1)  # [batch, hidden*2]
        
        # LSTM
        lstm_input = conv_out.transpose(1, 2)  # [batch, seq, hidden*2]
        lstm_out, (h_n, _) = self.lstm(lstm_input)
        lstm_final = lstm_out[:, -1, :]  # [batch, hidden*2]
        
        # دمج الميزات
        combined = torch.cat([avg_pool, max_pool, lstm_final], dim=1)
        
        # التصنيف
        output = self.classifier(combined)
        
        return output
    
    def forward_with_confidence(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        التمرير الأمامي مع الثقة
        
        Args:
            x: المدخلات
            
        Returns:
            (logits, confidence)
        """
        x = x.transpose(1, 2)
        conv_out = self.conv_layers(x)
        
        avg_pool = self.global_avg_pool(conv_out).squeeze(-1)
        max_pool = self.global_max_pool(conv_out).squeeze(-1)
        
        lstm_input = conv_out.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_final = lstm_out[:, -1, :]
        
        combined = torch.cat([avg_pool, max_pool, lstm_final], dim=1)
        
        logits = self.classifier(combined)
        confidence = self.confidence_layer(combined)
        
        return logits, confidence
    
    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        """
        التنبؤ بحالة السوق
        
        Args:
            x: المدخلات
            
        Returns:
            قاموس بالتنبؤات
        """
        self.eval()
        
        if x.ndim == 2:
            x = x[np.newaxis, :, :]
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            logits, confidence = self.forward_with_confidence(x_tensor)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        
        probs_np = probs.cpu().numpy()
        preds_np = predictions.cpu().numpy()
        conf_np = confidence.cpu().numpy()
        
        regime_names = [r.name for r in MarketRegime]
        
        results = []
        for i in range(len(preds_np)):
            regime_probs = {
                regime_names[j]: float(probs_np[i, j])
                for j in range(self.num_regimes)
            }
            
            results.append({
                'regime': regime_names[preds_np[i]],
                'regime_id': int(preds_np[i]),
                'confidence': float(conf_np[i, 0]),
                'probabilities': regime_probs,
                'is_bullish': preds_np[i] in [0, 1],
                'is_bearish': preds_np[i] in [3, 4],
                'is_volatile': preds_np[i] == 5,
                'trading_recommendation': self._get_trading_recommendation(
                    preds_np[i], conf_np[i, 0]
                )
            })
        
        return results[0] if len(results) == 1 else results
    
    def _get_trading_recommendation(
        self, 
        regime_id: int, 
        confidence: float
    ) -> Dict[str, Any]:
        """
        الحصول على توصية التداول بناءً على حالة السوق
        
        Args:
            regime_id: معرف الحالة
            confidence: مستوى الثقة
            
        Returns:
            توصية التداول
        """
        recommendations = {
            0: {  # STRONG_BULLISH
                'action': 'AGGRESSIVE_BUY',
                'position_size_multiplier': 1.5,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.5,
                'description': 'سوق صاعد قوي - فرصة شراء ممتازة'
            },
            1: {  # BULLISH
                'action': 'BUY',
                'position_size_multiplier': 1.2,
                'stop_loss_multiplier': 1.0,
                'take_profit_multiplier': 1.2,
                'description': 'سوق صاعد - فرصة شراء جيدة'
            },
            2: {  # NEUTRAL
                'action': 'HOLD',
                'position_size_multiplier': 0.8,
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 1.0,
                'description': 'سوق محايد - انتظر إشارة واضحة'
            },
            3: {  # BEARISH
                'action': 'REDUCE',
                'position_size_multiplier': 0.5,
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 0.8,
                'description': 'سوق هابط - قلل المراكز'
            },
            4: {  # STRONG_BEARISH
                'action': 'EXIT',
                'position_size_multiplier': 0.0,
                'stop_loss_multiplier': 2.0,
                'take_profit_multiplier': 0.5,
                'description': 'سوق هابط قوي - اخرج من المراكز'
            },
            5: {  # HIGH_VOLATILITY
                'action': 'CAUTION',
                'position_size_multiplier': 0.5,
                'stop_loss_multiplier': 2.0,
                'take_profit_multiplier': 2.0,
                'description': 'تقلب عالي - تداول بحذر'
            },
            6: {  # LOW_VOLATILITY
                'action': 'WAIT',
                'position_size_multiplier': 0.7,
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 0.8,
                'description': 'تقلب منخفض - انتظر اختراق'
            }
        }
        
        rec = recommendations.get(regime_id, recommendations[2])
        
        # تعديل بناءً على الثقة
        if confidence < 0.5:
            rec['position_size_multiplier'] *= 0.5
            rec['description'] += ' (ثقة منخفضة)'
        
        return rec
    
    def get_input_shape(self) -> Tuple[int, int]:
        """الحصول على شكل المدخلات"""
        return (self.sequence_length, self.input_dim)
    
    def _get_loss_function(self) -> nn.Module:
        """الحصول على دالة الخسارة"""
        return nn.CrossEntropyLoss()
    
    def detect_regime_change(
        self,
        current_regime: int,
        new_regime: int,
        confidence: float
    ) -> Dict[str, Any]:
        """
        كشف تغير حالة السوق
        
        Args:
            current_regime: الحالة الحالية
            new_regime: الحالة الجديدة
            confidence: مستوى الثقة
            
        Returns:
            معلومات التغير
        """
        if current_regime == new_regime:
            return {
                'changed': False,
                'significance': 'NONE',
                'action_required': False
            }
        
        # حساب أهمية التغير
        regime_order = [4, 3, 2, 1, 0]  # من الأكثر هبوطاً للأكثر صعوداً
        
        try:
            current_idx = regime_order.index(current_regime)
            new_idx = regime_order.index(new_regime)
            change_magnitude = abs(new_idx - current_idx)
        except ValueError:
            change_magnitude = 1
        
        if change_magnitude >= 3:
            significance = 'CRITICAL'
        elif change_magnitude >= 2:
            significance = 'HIGH'
        elif change_magnitude >= 1:
            significance = 'MEDIUM'
        else:
            significance = 'LOW'
        
        return {
            'changed': True,
            'from_regime': MarketRegime(current_regime).name,
            'to_regime': MarketRegime(new_regime).name,
            'significance': significance,
            'confidence': confidence,
            'action_required': significance in ['CRITICAL', 'HIGH'] and confidence > 0.6,
            'direction': 'BULLISH' if new_idx > current_idx else 'BEARISH'
        }


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار النموذج
    model = MarketRegimeClassifier(
        input_dim=50,
        hidden_dim=64,
        num_regimes=7,
        sequence_length=60
    )
    
    # بيانات تجريبية
    x = np.random.randn(4, 60, 50).astype(np.float32)
    
    # تنبؤ
    result = model.predict(x)
    print(f"Predictions: {result}")
    
    # كشف تغير الحالة
    change = model.detect_regime_change(2, 0, 0.85)
    print(f"\nRegime change: {change}")
    
    # معلومات النموذج
    print(f"\nModel info: {model.get_model_info()}")
