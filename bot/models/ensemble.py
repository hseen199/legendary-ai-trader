"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Ensemble Model
نموذج الدمج الذكي لجميع النماذج
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from loguru import logger

from .base_model import BaseModel
from .tft_model import TFTModel
from .lstm_attention import LSTMAttentionModel
from .market_regime import MarketRegimeClassifier


class MetaLearner(nn.Module):
    """
    المتعلم الفوقي لدمج تنبؤات النماذج
    """
    
    def __init__(
        self,
        num_models: int,
        num_classes: int = 3,
        hidden_dim: int = 32
    ):
        """
        تهيئة المتعلم الفوقي
        
        Args:
            num_models: عدد النماذج
            num_classes: عدد الفئات
            hidden_dim: حجم الطبقة المخفية
        """
        super().__init__()
        
        # المدخلات: احتمالات كل نموذج + ثقة + حالة السوق
        input_dim = num_models * num_classes + num_models + 7  # 7 لحالات السوق
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # طبقة الثقة
        self.confidence_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        model_probs: torch.Tensor,
        model_confidences: torch.Tensor,
        regime_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        التمرير الأمامي
        
        Args:
            model_probs: احتمالات النماذج [batch, num_models * num_classes]
            model_confidences: ثقة النماذج [batch, num_models]
            regime_probs: احتمالات حالة السوق [batch, 7]
            
        Returns:
            (logits, confidence)
        """
        # دمج المدخلات
        combined = torch.cat([model_probs, model_confidences, regime_probs], dim=1)
        
        logits = self.network(combined)
        confidence = self.confidence_net(combined)
        
        return logits, confidence


class EnsembleModel(BaseModel):
    """
    نموذج الدمج الذكي
    
    يجمع بين:
    - TFT للتنبؤ الزمني
    - LSTM+Attention للأنماط
    - Market Regime للسياق
    - Meta Learner للدمج الذكي
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        sequence_length: int = 60,
        output_dim: int = 3,
        config: Optional[Dict] = None
    ):
        """
        تهيئة نموذج الدمج
        
        Args:
            input_dim: عدد الميزات
            hidden_dim: حجم الطبقة المخفية
            sequence_length: طول التسلسل
            output_dim: عدد الفئات
            config: إعدادات إضافية
        """
        config = config or {}
        config.update({
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'sequence_length': sequence_length,
            'output_dim': output_dim
        })
        
        super().__init__("Ensemble", config)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        
        # النماذج الفرعية
        self.tft = TFTModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_encoder_layers=2,
            sequence_length=sequence_length,
            output_dim=output_dim
        )
        
        self.lstm = LSTMAttentionModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            sequence_length=sequence_length,
            output_dim=output_dim
        )
        
        self.regime_classifier = MarketRegimeClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2,
            num_regimes=7,
            sequence_length=sequence_length
        )
        
        # المتعلم الفوقي
        self.meta_learner = MetaLearner(
            num_models=2,  # TFT + LSTM
            num_classes=output_dim,
            hidden_dim=64
        )
        
        # أوزان النماذج (قابلة للتعلم)
        self.model_weights = nn.Parameter(torch.ones(2) / 2)
        
        # تاريخ الأداء
        self.performance_history: Dict[str, List[float]] = {
            'tft': [],
            'lstm': [],
            'ensemble': []
        }
        
        logger.info(f"🧠 Ensemble Model initialized with {self._count_total_params():,} total parameters")
    
    def _count_total_params(self) -> int:
        """عد جميع المعاملات"""
        total = 0
        total += sum(p.numel() for p in self.tft.parameters())
        total += sum(p.numel() for p in self.lstm.parameters())
        total += sum(p.numel() for p in self.regime_classifier.parameters())
        total += sum(p.numel() for p in self.meta_learner.parameters())
        total += self.model_weights.numel()
        return total
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        التمرير الأمامي
        
        Args:
            x: المدخلات [batch, sequence, features]
            
        Returns:
            المخرجات [batch, output_dim]
        """
        # تنبؤات النماذج الفرعية
        tft_logits = self.tft(x)
        lstm_logits = self.lstm(x)
        regime_logits, regime_conf = self.regime_classifier.forward_with_confidence(x)
        
        # تحويل إلى احتمالات
        tft_probs = F.softmax(tft_logits, dim=-1)
        lstm_probs = F.softmax(lstm_logits, dim=-1)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # حساب الثقة
        tft_conf = tft_probs.max(dim=-1)[0].unsqueeze(-1)
        lstm_conf = lstm_probs.max(dim=-1)[0].unsqueeze(-1)
        
        # دمج الاحتمالات
        model_probs = torch.cat([tft_probs, lstm_probs], dim=-1)
        model_confs = torch.cat([tft_conf, lstm_conf], dim=-1)
        
        # المتعلم الفوقي
        final_logits, _ = self.meta_learner(model_probs, model_confs, regime_probs)
        
        return final_logits
    
    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        """
        التنبؤ الشامل
        
        Args:
            x: المدخلات
            
        Returns:
            قاموس شامل بالتنبؤات
        """
        self.eval()
        
        if x.ndim == 2:
            x = x[np.newaxis, :, :]
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            
            # تنبؤات فردية
            tft_pred = self.tft.predict(x)
            lstm_pred = self.lstm.predict(x)
            regime_pred = self.regime_classifier.predict(x)
            
            # التنبؤ المدمج
            final_logits = self.forward(x_tensor)
            final_probs = F.softmax(final_logits, dim=-1)
            final_preds = torch.argmax(final_probs, dim=-1)
        
        final_probs_np = final_probs.cpu().numpy()
        final_preds_np = final_preds.cpu().numpy()
        
        action_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
        
        results = []
        for i in range(len(final_preds_np)):
            # حساب الاتفاق بين النماذج
            models_agree = (
                tft_pred['action'] == lstm_pred['action'] == action_map[final_preds_np[i]]
            )
            
            # حساب الثقة المركبة
            composite_confidence = (
                tft_pred['confidence'] * 0.35 +
                lstm_pred['confidence'] * 0.35 +
                float(final_probs_np[i].max()) * 0.30
            )
            
            # تعديل الثقة بناءً على حالة السوق
            regime_adjustment = self._get_regime_adjustment(regime_pred)
            adjusted_confidence = composite_confidence * regime_adjustment
            
            results.append({
                'action': action_map[final_preds_np[i]],
                'confidence': float(adjusted_confidence),
                'probabilities': {
                    'BUY': float(final_probs_np[i, 0]),
                    'SELL': float(final_probs_np[i, 1]),
                    'HOLD': float(final_probs_np[i, 2])
                },
                'models_agree': models_agree,
                'individual_predictions': {
                    'tft': tft_pred,
                    'lstm': lstm_pred
                },
                'market_regime': regime_pred,
                'reasoning': self._generate_reasoning(
                    action_map[final_preds_np[i]],
                    tft_pred,
                    lstm_pred,
                    regime_pred,
                    models_agree
                ),
                'risk_assessment': self._assess_risk(
                    action_map[final_preds_np[i]],
                    adjusted_confidence,
                    regime_pred
                )
            })
        
        return results[0] if len(results) == 1 else results
    
    def _get_regime_adjustment(self, regime_pred: Dict) -> float:
        """
        الحصول على معامل تعديل حسب حالة السوق
        
        Args:
            regime_pred: تنبؤ حالة السوق
            
        Returns:
            معامل التعديل
        """
        regime = regime_pred.get('regime', 'NEUTRAL')
        
        adjustments = {
            'STRONG_BULLISH': 1.1,
            'BULLISH': 1.05,
            'NEUTRAL': 1.0,
            'BEARISH': 0.9,
            'STRONG_BEARISH': 0.8,
            'HIGH_VOLATILITY': 0.85,
            'LOW_VOLATILITY': 0.95
        }
        
        return adjustments.get(regime, 1.0)
    
    def _generate_reasoning(
        self,
        action: str,
        tft_pred: Dict,
        lstm_pred: Dict,
        regime_pred: Dict,
        models_agree: bool
    ) -> str:
        """
        توليد التبرير المنطقي للقرار
        
        Args:
            action: الإجراء المقترح
            tft_pred: تنبؤ TFT
            lstm_pred: تنبؤ LSTM
            regime_pred: حالة السوق
            models_agree: هل النماذج متفقة
            
        Returns:
            نص التبرير
        """
        reasoning_parts = []
        
        # حالة السوق
        regime = regime_pred.get('regime', 'NEUTRAL')
        reasoning_parts.append(f"حالة السوق: {regime}")
        
        # اتفاق النماذج
        if models_agree:
            reasoning_parts.append("جميع النماذج متفقة على هذا القرار")
        else:
            reasoning_parts.append(
                f"TFT يقترح {tft_pred['action']} ({tft_pred['confidence']:.1%}), "
                f"LSTM يقترح {lstm_pred['action']} ({lstm_pred['confidence']:.1%})"
            )
        
        # توصية حالة السوق
        if 'trading_recommendation' in regime_pred:
            rec = regime_pred['trading_recommendation']
            reasoning_parts.append(f"توصية السوق: {rec.get('description', '')}")
        
        # التبرير النهائي
        if action == 'BUY':
            reasoning_parts.append("إشارات إيجابية تدعم الشراء")
        elif action == 'SELL':
            reasoning_parts.append("إشارات سلبية تدعم البيع")
        else:
            reasoning_parts.append("لا توجد إشارة واضحة - الانتظار أفضل")
        
        return " | ".join(reasoning_parts)
    
    def _assess_risk(
        self,
        action: str,
        confidence: float,
        regime_pred: Dict
    ) -> Dict[str, Any]:
        """
        تقييم المخاطر
        
        Args:
            action: الإجراء
            confidence: الثقة
            regime_pred: حالة السوق
            
        Returns:
            تقييم المخاطر
        """
        regime = regime_pred.get('regime', 'NEUTRAL')
        
        # حساب درجة المخاطر
        base_risk = 0.5
        
        # تعديل حسب الثقة
        if confidence > 0.8:
            base_risk -= 0.2
        elif confidence < 0.5:
            base_risk += 0.2
        
        # تعديل حسب حالة السوق
        regime_risks = {
            'STRONG_BULLISH': -0.1,
            'BULLISH': -0.05,
            'NEUTRAL': 0,
            'BEARISH': 0.1,
            'STRONG_BEARISH': 0.2,
            'HIGH_VOLATILITY': 0.25,
            'LOW_VOLATILITY': -0.05
        }
        base_risk += regime_risks.get(regime, 0)
        
        # تعديل حسب الإجراء
        if action == 'HOLD':
            base_risk -= 0.1
        
        risk_score = max(0, min(1, base_risk))
        
        # تحديد مستوى المخاطر
        if risk_score < 0.3:
            risk_level = 'LOW'
        elif risk_score < 0.5:
            risk_level = 'MEDIUM'
        elif risk_score < 0.7:
            risk_level = 'HIGH'
        else:
            risk_level = 'EXTREME'
        
        return {
            'score': risk_score,
            'level': risk_level,
            'factors': {
                'confidence': confidence,
                'market_regime': regime,
                'action': action
            },
            'recommendation': self._get_risk_recommendation(risk_level, action)
        }
    
    def _get_risk_recommendation(self, risk_level: str, action: str) -> str:
        """الحصول على توصية المخاطر"""
        if risk_level == 'EXTREME':
            return "مخاطر عالية جداً - تجنب التداول أو استخدم حجم صغير جداً"
        elif risk_level == 'HIGH':
            return "مخاطر عالية - قلل حجم الصفقة واستخدم وقف خسارة ضيق"
        elif risk_level == 'MEDIUM':
            return "مخاطر متوسطة - التزم بإدارة المخاطر المعتادة"
        else:
            return "مخاطر منخفضة - يمكن زيادة حجم الصفقة قليلاً"
    
    def get_input_shape(self) -> Tuple[int, int]:
        """الحصول على شكل المدخلات"""
        return (self.sequence_length, self.input_dim)
    
    def _get_loss_function(self) -> nn.Module:
        """الحصول على دالة الخسارة"""
        return nn.CrossEntropyLoss()
    
    def update_performance(self, model_name: str, accuracy: float) -> None:
        """
        تحديث تاريخ الأداء
        
        Args:
            model_name: اسم النموذج
            accuracy: الدقة
        """
        if model_name in self.performance_history:
            self.performance_history[model_name].append(accuracy)
            # الاحتفاظ بآخر 100 قيمة فقط
            if len(self.performance_history[model_name]) > 100:
                self.performance_history[model_name] = self.performance_history[model_name][-100:]
    
    def get_model_weights(self) -> Dict[str, float]:
        """الحصول على أوزان النماذج الحالية"""
        weights = F.softmax(self.model_weights, dim=0)
        return {
            'tft': float(weights[0]),
            'lstm': float(weights[1])
        }
    
    def save_all(self, save_dir: str) -> None:
        """حفظ جميع النماذج"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        self.tft.save(os.path.join(save_dir, "tft.pt"))
        self.lstm.save(os.path.join(save_dir, "lstm.pt"))
        self.regime_classifier.save(os.path.join(save_dir, "regime.pt"))
        self.save(os.path.join(save_dir, "ensemble.pt"))
        
        logger.info(f"✅ All models saved to {save_dir}")
    
    def load_all(self, save_dir: str) -> None:
        """تحميل جميع النماذج"""
        import os
        
        self.tft.load(os.path.join(save_dir, "tft.pt"))
        self.lstm.load(os.path.join(save_dir, "lstm.pt"))
        self.regime_classifier.load(os.path.join(save_dir, "regime.pt"))
        self.load(os.path.join(save_dir, "ensemble.pt"))
        
        logger.info(f"✅ All models loaded from {save_dir}")


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار النموذج
    model = EnsembleModel(
        input_dim=50,
        hidden_dim=64,
        sequence_length=60,
        output_dim=3
    )
    
    # بيانات تجريبية
    x = np.random.randn(2, 60, 50).astype(np.float32)
    
    # تنبؤ
    result = model.predict(x)
    print(f"Ensemble Prediction:")
    print(f"  Action: {result['action']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Models Agree: {result['models_agree']}")
    print(f"  Risk Level: {result['risk_assessment']['level']}")
    print(f"\nReasoning: {result['reasoning']}")
    
    # أوزان النماذج
    print(f"\nModel weights: {model.get_model_weights()}")
