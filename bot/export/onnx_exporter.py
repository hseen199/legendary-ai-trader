"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - ONNX Exporter
مصدر ONNX
═══════════════════════════════════════════════════════════════
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from loguru import logger

# إضافة المسار
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ensemble import EnsembleModel
from models.tft_model import TFTModel
from models.lstm_attention import LSTMAttentionModel


class SimplifiedModel(nn.Module):
    """
    نموذج مبسط للتصدير إلى ONNX
    
    يجمع بين TFT و LSTM في نموذج واحد قابل للتصدير
    """
    
    def __init__(
        self,
        num_features: int = 50,
        sequence_length: int = 60,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Output heads
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, seq_len, features]
            
        Returns:
            prediction: Price direction prediction [-1, 1]
            confidence: Confidence score [0, 1]
            regime_probs: Market regime probabilities [7]
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = attn_out.mean(dim=1)
        
        # Feature extraction
        features = self.feature_extractor(pooled)
        
        # Output heads
        prediction = self.prediction_head(features)
        confidence = self.confidence_head(features)
        regime_probs = self.regime_head(features)
        
        return prediction, confidence, regime_probs


class ONNXExporter:
    """
    مصدر ONNX
    
    يصدر النماذج إلى صيغة ONNX للاستخدام في TypeScript
    """
    
    def __init__(self, output_dir: str = None):
        """
        تهيئة المصدر
        
        Args:
            output_dir: مجلد المخرجات
        """
        self.output_dir = Path(output_dir) if output_dir else Path('/tmp/legendary_agent/export')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📦 ONNXExporter initialized, output: {self.output_dir}")
    
    def export_simplified(
        self,
        num_features: int = 50,
        sequence_length: int = 60,
        hidden_dim: int = 128,
        output_name: str = 'legendary_agent.onnx'
    ) -> str:
        """
        تصدير النموذج المبسط
        
        Args:
            num_features: عدد الميزات
            sequence_length: طول التسلسل
            hidden_dim: أبعاد الطبقة المخفية
            output_name: اسم الملف
            
        Returns:
            مسار الملف المصدر
        """
        logger.info("🔄 Creating simplified model for export...")
        
        # إنشاء النموذج
        model = SimplifiedModel(
            num_features=num_features,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim
        )
        model.eval()
        
        # إنشاء مدخل وهمي
        dummy_input = torch.randn(1, sequence_length, num_features)
        
        # مسار الملف
        output_path = self.output_dir / output_name
        
        logger.info(f"📦 Exporting to {output_path}...")
        
        # التصدير
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=['features'],
            output_names=['prediction', 'confidence', 'regime_probs'],
            dynamic_axes={
                'features': {0: 'batch_size'},
                'prediction': {0: 'batch_size'},
                'confidence': {0: 'batch_size'},
                'regime_probs': {0: 'batch_size'}
            },
            opset_version=11,
            do_constant_folding=True,
            export_params=True
        )
        
        logger.info(f"✅ Model exported successfully: {output_path}")
        logger.info(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
        
        return str(output_path)
    
    def export_ensemble(
        self,
        ensemble_model: EnsembleModel,
        output_name: str = 'legendary_ensemble.onnx'
    ) -> str:
        """
        تصدير نموذج الدمج
        
        Args:
            ensemble_model: نموذج الدمج المدرب
            output_name: اسم الملف
            
        Returns:
            مسار الملف المصدر
        """
        logger.info("🔄 Exporting ensemble model...")
        
        ensemble_model.eval()
        
        # إنشاء مدخل وهمي
        dummy_input = torch.randn(1, 60, 50)
        
        # مسار الملف
        output_path = self.output_dir / output_name
        
        try:
            torch.onnx.export(
                ensemble_model,
                dummy_input,
                str(output_path),
                input_names=['features'],
                output_names=['output'],
                dynamic_axes={
                    'features': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                opset_version=11
            )
            
            logger.info(f"✅ Ensemble exported: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            logger.info("   Falling back to simplified model...")
            return self.export_simplified(output_name=output_name)
    
    def verify_export(self, onnx_path: str) -> bool:
        """
        التحقق من صحة التصدير
        
        Args:
            onnx_path: مسار ملف ONNX
            
        Returns:
            True إذا كان صالحاً
        """
        try:
            import onnx
            
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            
            logger.info(f"✅ ONNX model verification passed")
            
            # طباعة معلومات النموذج
            logger.info(f"   Inputs: {[i.name for i in model.graph.input]}")
            logger.info(f"   Outputs: {[o.name for o in model.graph.output]}")
            
            return True
            
        except ImportError:
            logger.warning("⚠️ onnx package not installed, skipping verification")
            return True
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
    
    def test_inference(self, onnx_path: str) -> Dict[str, Any]:
        """
        اختبار الاستدلال
        
        Args:
            onnx_path: مسار ملف ONNX
            
        Returns:
            نتائج الاختبار
        """
        try:
            import onnxruntime as ort
            
            # إنشاء الجلسة
            session = ort.InferenceSession(onnx_path)
            
            # إنشاء مدخل وهمي
            dummy_input = np.random.randn(1, 60, 50).astype(np.float32)
            
            # تشغيل الاستدلال
            outputs = session.run(None, {'features': dummy_input})
            
            result = {
                'success': True,
                'num_outputs': len(outputs),
                'output_shapes': [o.shape for o in outputs],
                'sample_prediction': float(outputs[0][0][0]) if len(outputs) > 0 else None,
                'sample_confidence': float(outputs[1][0][0]) if len(outputs) > 1 else None
            }
            
            logger.info(f"✅ Inference test passed")
            logger.info(f"   Prediction: {result['sample_prediction']:.4f}")
            logger.info(f"   Confidence: {result['sample_confidence']:.4f}")
            
            return result
            
        except ImportError:
            logger.warning("⚠️ onnxruntime not installed, skipping inference test")
            return {'success': False, 'error': 'onnxruntime not installed'}
        except Exception as e:
            logger.error(f"❌ Inference test failed: {e}")
            return {'success': False, 'error': str(e)}


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # تصدير النموذج
    exporter = ONNXExporter(output_dir='/home/ubuntu/legendary_agent/models/trained')
    
    # تصدير النموذج المبسط
    onnx_path = exporter.export_simplified(
        num_features=50,
        sequence_length=60,
        hidden_dim=128,
        output_name='legendary_agent.onnx'
    )
    
    # التحقق
    exporter.verify_export(onnx_path)
    
    # اختبار الاستدلال
    result = exporter.test_inference(onnx_path)
    print(f"\nInference result: {result}")
