"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - LSTM with Attention
نموذج LSTM مع آلية الانتباه
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from .base_model import BaseModel


class Attention(nn.Module):
    """آلية الانتباه"""
    
    def __init__(self, hidden_dim: int, attention_dim: int):
        """
        تهيئة آلية الانتباه
        
        Args:
            hidden_dim: حجم الطبقة المخفية
            attention_dim: حجم طبقة الانتباه
        """
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        حساب الانتباه
        
        Args:
            lstm_output: مخرجات LSTM [batch, seq, hidden]
            
        Returns:
            (context_vector, attention_weights)
        """
        # حساب درجات الانتباه
        attention_scores = self.attention(lstm_output)  # [batch, seq, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq, 1]
        
        # حساب متجه السياق
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # [batch, hidden]
        
        return context_vector, attention_weights.squeeze(-1)


class MultiHeadSelfAttention(nn.Module):
    """انتباه ذاتي متعدد الرؤوس"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        تهيئة الانتباه متعدد الرؤوس
        
        Args:
            hidden_dim: حجم الطبقة المخفية
            num_heads: عدد الرؤوس
            dropout: نسبة الإسقاط
        """
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        التمرير الأمامي
        
        Args:
            x: المدخلات [batch, seq, hidden]
            
        Returns:
            (output, attention_weights)
        """
        batch_size, seq_len, _ = x.size()
        
        # التحويلات الخطية
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # حساب الانتباه
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # تطبيق الانتباه
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        output = self.out(context)
        
        # متوسط أوزان الانتباه
        avg_weights = attention_weights.mean(dim=1)
        
        return output, avg_weights


class LSTMAttentionModel(BaseModel):
    """
    نموذج LSTM مع آلية الانتباه
    
    يجمع بين:
    - LSTM ثنائي الاتجاه لالتقاط التبعيات الزمنية
    - انتباه ذاتي متعدد الرؤوس
    - انتباه تسلسلي للتركيز على الأجزاء المهمة
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        bidirectional: bool = True,
        sequence_length: int = 60,
        output_dim: int = 3,
        config: Optional[Dict] = None
    ):
        """
        تهيئة النموذج
        
        Args:
            input_dim: عدد الميزات
            hidden_dim: حجم الطبقة المخفية
            num_layers: عدد طبقات LSTM
            num_heads: عدد رؤوس الانتباه
            dropout: نسبة الإسقاط
            bidirectional: ثنائي الاتجاه
            sequence_length: طول التسلسل
            output_dim: عدد الفئات
            config: إعدادات إضافية
        """
        config = config or {}
        config.update({
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'dropout': dropout,
            'bidirectional': bidirectional,
            'sequence_length': sequence_length,
            'output_dim': output_dim
        })
        
        super().__init__("LSTM_Attention", config)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        
        # حجم الإخراج الفعلي من LSTM
        self.lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # طبقة الإدخال
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # تطبيع بعد LSTM
        self.lstm_norm = nn.LayerNorm(self.lstm_output_dim)
        
        # انتباه ذاتي متعدد الرؤوس
        self.self_attention = MultiHeadSelfAttention(
            self.lstm_output_dim, num_heads, dropout
        )
        self.attention_norm = nn.LayerNorm(self.lstm_output_dim)
        
        # انتباه تسلسلي
        self.sequence_attention = Attention(self.lstm_output_dim, hidden_dim)
        
        # طبقات الإخراج
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # تهيئة الأوزان
        self._init_weights()
        
        logger.info(f"🧠 LSTM Attention Model: {self.count_parameters():,} parameters")
    
    def _init_weights(self):
        """تهيئة الأوزان"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        # تعيين forget gate bias إلى 1
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        التمرير الأمامي
        
        Args:
            x: المدخلات [batch, sequence, features]
            
        Returns:
            المخرجات [batch, output_dim]
        """
        # إسقاط الإدخال
        x = self.input_projection(x)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        
        # انتباه ذاتي
        self_attn_out, _ = self.self_attention(lstm_out)
        lstm_out = self.attention_norm(lstm_out + self_attn_out)
        
        # انتباه تسلسلي
        context, attention_weights = self.sequence_attention(lstm_out)
        
        # التصنيف
        output = self.classifier(context)
        
        return output
    
    def forward_with_attention(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        التمرير الأمامي مع إرجاع أوزان الانتباه
        
        Args:
            x: المدخلات
            
        Returns:
            (output, self_attention_weights, sequence_attention_weights)
        """
        # إسقاط الإدخال
        x = self.input_projection(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        
        # انتباه ذاتي
        self_attn_out, self_attn_weights = self.self_attention(lstm_out)
        lstm_out = self.attention_norm(lstm_out + self_attn_out)
        
        # انتباه تسلسلي
        context, seq_attn_weights = self.sequence_attention(lstm_out)
        
        # التصنيف
        output = self.classifier(context)
        
        return output, self_attn_weights, seq_attn_weights
    
    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        """
        التنبؤ
        
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
            logits, self_attn, seq_attn = self.forward_with_attention(x_tensor)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        
        probs_np = probs.cpu().numpy()
        preds_np = predictions.cpu().numpy()
        seq_attn_np = seq_attn.cpu().numpy()
        
        action_map = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
        
        results = []
        for i in range(len(preds_np)):
            results.append({
                'action': action_map[preds_np[i]],
                'confidence': float(probs_np[i].max()),
                'probabilities': {
                    'BUY': float(probs_np[i, 0]),
                    'SELL': float(probs_np[i, 1]),
                    'HOLD': float(probs_np[i, 2])
                },
                'attention_weights': seq_attn_np[i].tolist()
            })
        
        return results[0] if len(results) == 1 else results
    
    def get_input_shape(self) -> Tuple[int, int]:
        """الحصول على شكل المدخلات"""
        return (self.sequence_length, self.input_dim)
    
    def _get_loss_function(self) -> nn.Module:
        """الحصول على دالة الخسارة"""
        return nn.CrossEntropyLoss()
    
    def get_feature_importance(self, x: np.ndarray) -> np.ndarray:
        """
        الحصول على أهمية الميزات باستخدام الانتباه
        
        Args:
            x: المدخلات
            
        Returns:
            أهمية الميزات
        """
        self.eval()
        
        if x.ndim == 2:
            x = x[np.newaxis, :, :]
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            _, _, seq_attn = self.forward_with_attention(x_tensor)
        
        # حساب أهمية الميزات
        attention_weights = seq_attn.cpu().numpy()  # [batch, seq]
        
        # الجمع الموزون للميزات
        importance = np.zeros((x.shape[0], x.shape[2]))
        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                importance[i, j] = np.sum(attention_weights[i] * x[i, :, j])
        
        # تطبيع
        importance = np.abs(importance)
        importance = importance / (importance.sum(axis=1, keepdims=True) + 1e-10)
        
        return importance


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار النموذج
    model = LSTMAttentionModel(
        input_dim=50,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        sequence_length=60,
        output_dim=3
    )
    
    # بيانات تجريبية
    x = np.random.randn(4, 60, 50).astype(np.float32)
    
    # تنبؤ
    result = model.predict(x)
    print(f"Predictions: {result}")
    
    # أهمية الميزات
    importance = model.get_feature_importance(x)
    print(f"\nFeature importance shape: {importance.shape}")
    
    # معلومات النموذج
    print(f"\nModel info: {model.get_model_info()}")
