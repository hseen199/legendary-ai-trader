"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Temporal Fusion Transformer (TFT)
نموذج المحول الزمني للتنبؤ بالأسعار
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

from .base_model import BaseModel


class GatedLinearUnit(nn.Module):
    """وحدة خطية مبوبة (GLU)"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x[:, :, :self.output_dim] * torch.sigmoid(x[:, :, self.output_dim:])


class GatedResidualNetwork(nn.Module):
    """شبكة البقايا المبوبة (GRN)"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        
        # الطبقات
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # طبقة السياق (اختيارية)
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        
        # طبقة التخطي
        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # التخطي
        if self.skip is not None:
            residual = self.skip(x)
        else:
            residual = x
        
        # المعالجة
        x = self.fc1(x)
        
        if context is not None and self.context_dim is not None:
            context = self.context_fc(context)
            x = x + context.unsqueeze(1) if context.dim() == 2 else x + context
        
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)
        
        # الدمج مع البقايا
        return self.layer_norm(x + residual)


class VariableSelectionNetwork(nn.Module):
    """شبكة اختيار المتغيرات"""
    
    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_inputs = num_inputs
        
        # GRN لكل متغير
        self.grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_inputs)
        ])
        
        # GRN للأوزان
        self.weight_grn = GatedResidualNetwork(
            hidden_dim * num_inputs,
            hidden_dim,
            num_inputs,
            dropout,
            context_dim
        )
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(
        self, 
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # معالجة كل متغير
        processed = []
        for i, grn in enumerate(self.grns):
            processed.append(grn(x[:, :, i:i+1].expand(-1, -1, self.hidden_dim)))
        
        processed = torch.stack(processed, dim=-1)  # [batch, seq, hidden, num_inputs]
        
        # حساب الأوزان
        flat = processed.view(x.size(0), x.size(1), -1)
        weights = self.weight_grn(flat, context)
        weights = self.softmax(weights)
        
        # الجمع الموزون
        output = (processed * weights.unsqueeze(2)).sum(dim=-1)
        
        return output, weights


class InterpretableMultiHeadAttention(nn.Module):
    """انتباه متعدد الرؤوس قابل للتفسير"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # التحويلات الخطية
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
        
        # إعادة التشكيل للرؤوس المتعددة
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # حساب الانتباه
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # تطبيق الانتباه
        context = torch.matmul(attention_weights, v)
        
        # إعادة التشكيل
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_linear(context)
        
        # متوسط أوزان الانتباه عبر الرؤوس
        avg_attention = attention_weights.mean(dim=1)
        
        return output, avg_attention


class TFTModel(BaseModel):
    """
    نموذج Temporal Fusion Transformer
    
    مصمم للتنبؤ بالسلاسل الزمنية مع:
    - اختيار المتغيرات التلقائي
    - انتباه قابل للتفسير
    - معالجة متغيرات ثابتة ومتغيرة
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 3,
        dropout: float = 0.1,
        sequence_length: int = 60,
        output_dim: int = 3,  # BUY, SELL, HOLD
        config: Optional[Dict] = None
    ):
        """
        تهيئة النموذج
        
        Args:
            input_dim: عدد الميزات
            hidden_dim: حجم الطبقة المخفية
            num_heads: عدد رؤوس الانتباه
            num_encoder_layers: عدد طبقات المشفر
            dropout: نسبة الإسقاط
            sequence_length: طول التسلسل
            output_dim: عدد الفئات
            config: إعدادات إضافية
        """
        config = config or {}
        config.update({
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_heads': num_heads,
            'num_encoder_layers': num_encoder_layers,
            'dropout': dropout,
            'sequence_length': sequence_length,
            'output_dim': output_dim
        })
        
        super().__init__("TFT", config)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        
        # طبقة الإدخال
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # اختيار المتغيرات
        self.variable_selection = VariableSelectionNetwork(
            input_dim=1,
            num_inputs=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # LSTM للتشفير
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if num_encoder_layers > 1 else 0,
            bidirectional=False
        )
        
        # طبقات الانتباه
        self.attention_layers = nn.ModuleList([
            InterpretableMultiHeadAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # GRN بعد الانتباه
        self.post_attention_grn = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # تطبيع الطبقات
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_encoder_layers)
        ])
        
        # طبقة الإخراج
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # تهيئة الأوزان
        self._init_weights()
        
        logger.info(f"🧠 TFT Model: {self.count_parameters():,} parameters")
    
    def _init_weights(self):
        """تهيئة الأوزان"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        التمرير الأمامي
        
        Args:
            x: المدخلات [batch, sequence, features]
            
        Returns:
            المخرجات [batch, output_dim]
        """
        batch_size = x.size(0)
        
        # اختيار المتغيرات
        selected, var_weights = self.variable_selection(x)
        
        # تشفير LSTM
        lstm_out, (h_n, c_n) = self.lstm_encoder(selected)
        
        # طبقات الانتباه
        attention_output = lstm_out
        attention_weights_list = []
        
        for i in range(self.num_encoder_layers):
            # Self-attention
            attn_out, attn_weights = self.attention_layers[i](
                attention_output, attention_output, attention_output
            )
            attention_weights_list.append(attn_weights)
            
            # Add & Norm
            attention_output = self.layer_norms[i](attention_output + attn_out)
            
            # GRN
            attention_output = self.post_attention_grn[i](attention_output)
        
        # أخذ آخر خطوة زمنية
        final_output = attention_output[:, -1, :]
        
        # طبقة الإخراج
        output = self.output_layer(final_output)
        
        return output
    
    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        """
        التنبؤ
        
        Args:
            x: المدخلات [batch, sequence, features] أو [sequence, features]
            
        Returns:
            قاموس بالتنبؤات
        """
        self.eval()
        
        # التأكد من الأبعاد
        if x.ndim == 2:
            x = x[np.newaxis, :, :]
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            logits = self.forward(x_tensor)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        
        # تحويل إلى numpy
        probs_np = probs.cpu().numpy()
        preds_np = predictions.cpu().numpy()
        
        # تحديد الإجراء
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
                }
            })
        
        return results[0] if len(results) == 1 else results
    
    def get_input_shape(self) -> Tuple[int, int]:
        """الحصول على شكل المدخلات"""
        return (self.sequence_length, self.input_dim)
    
    def _get_loss_function(self) -> nn.Module:
        """الحصول على دالة الخسارة"""
        return nn.CrossEntropyLoss()
    
    def get_attention_weights(self, x: np.ndarray) -> np.ndarray:
        """
        الحصول على أوزان الانتباه للتفسير
        
        Args:
            x: المدخلات
            
        Returns:
            أوزان الانتباه
        """
        self.eval()
        
        if x.ndim == 2:
            x = x[np.newaxis, :, :]
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            
            # اختيار المتغيرات
            selected, var_weights = self.variable_selection(x_tensor)
            
            # تشفير LSTM
            lstm_out, _ = self.lstm_encoder(selected)
            
            # الحصول على أوزان الانتباه
            attention_output = lstm_out
            all_weights = []
            
            for i in range(self.num_encoder_layers):
                _, attn_weights = self.attention_layers[i](
                    attention_output, attention_output, attention_output
                )
                all_weights.append(attn_weights.cpu().numpy())
                attention_output = self.layer_norms[i](attention_output)
                attention_output = self.post_attention_grn[i](attention_output)
        
        return np.stack(all_weights, axis=0)


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار النموذج
    model = TFTModel(
        input_dim=50,
        hidden_dim=64,
        num_heads=4,
        num_encoder_layers=2,
        sequence_length=60,
        output_dim=3
    )
    
    # بيانات تجريبية
    x = np.random.randn(4, 60, 50).astype(np.float32)
    
    # تنبؤ
    result = model.predict(x)
    print(f"Predictions: {result}")
    
    # معلومات النموذج
    print(f"\nModel info: {model.get_model_info()}")
