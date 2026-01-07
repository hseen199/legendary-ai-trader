"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Main Agent
الوكيل الأسطوري الرئيسي
═══════════════════════════════════════════════════════════════

وكيل ذكاء اصطناعي متكامل للتداول في سوق الكريبتو
يفكر، يبتكر، يتعلم، ويتطور بشكل مستقل

Author: Legendary Agent Team
Version: 1.0.0
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from loguru import logger

# إعداد المسار
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# استيراد المكونات
from types import MarketState, TradingSignal, AgentDecision, AgentConfig
from layers.perception import PerceptionLayer
from layers.understanding import UnderstandingLayer
from layers.planning import PlanningLayer
from layers.decision import DecisionLayer
from layers.protection import ProtectionLayer
from layers.evolution import EvolutionLayer
from mind.creative_mind import CreativeMind
from memory.memory_system import MemorySystem, MemoryType
from protection.circuit_breaker import CircuitBreaker
from protection.anomaly_detector import AnomalyDetector
from models.ensemble import EnsembleModel


@dataclass
class AgentState:
    """حالة الوكيل"""
    is_active: bool = True
    mode: str = "balanced"  # aggressive, balanced, conservative
    total_decisions: int = 0
    successful_decisions: int = 0
    current_positions: Dict[str, Any] = field(default_factory=dict)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)


class LegendaryAgent:
    """
    الوكيل الأسطوري
    
    وكيل ذكاء اصطناعي متكامل يجمع بين:
    - 6 طبقات معالجة (إدراك، فهم، تخطيط، قرار، حماية، تطور)
    - عقل مبدع (تفكير، ابتكار، حوار داخلي)
    - نظام ذاكرة متعدد الطبقات
    - أنظمة حماية متقدمة
    - نماذج تعلم آلي متعددة
    """
    
    def __init__(
        self,
        config: Union[Dict, AgentConfig] = None,
        model_path: str = None,
        data_dir: str = None
    ):
        """
        تهيئة الوكيل الأسطوري
        
        Args:
            config: إعدادات الوكيل
            model_path: مسار النماذج المدربة
            data_dir: مجلد البيانات
        """
        # تحويل الإعدادات
        if isinstance(config, dict):
            self.config = AgentConfig(**config) if config else AgentConfig()
        else:
            self.config = config or AgentConfig()
        
        self.model_path = Path(model_path) if model_path else ROOT_DIR / 'models' / 'trained'
        self.data_dir = Path(data_dir) if data_dir else ROOT_DIR / 'data'
        
        # تحديد الجهاز
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # تهيئة الحالة
        self.state = AgentState(mode=self.config.mode)
        
        # تهيئة المكونات
        self._initialize_components()
        
        logger.info(f"🦁 LegendaryAgent initialized in {self.config.mode} mode on {self.device}")
    
    def _initialize_components(self) -> None:
        """تهيئة جميع المكونات"""
        config_dict = {
            'stop_loss': self.config.stop_loss,
            'take_profit': self.config.take_profit,
            'max_position_size': self.config.max_position_size,
            'min_position_size': self.config.min_position_size,
            'max_daily_loss': self.config.max_daily_loss,
            'max_weekly_loss': self.config.max_weekly_loss,
            'portfolio_heat_limit': self.config.portfolio_heat_limit
        }
        
        # الطبقات الست
        logger.info("  Initializing 6 layers...")
        self.perception = PerceptionLayer(config_dict)
        self.understanding = UnderstandingLayer(config_dict)
        self.planning = PlanningLayer(config_dict)
        self.decision = DecisionLayer(config_dict)
        self.protection = ProtectionLayer(config_dict)
        self.evolution = EvolutionLayer(config_dict, str(self.data_dir / 'evolution'))
        
        # العقل المبدع
        logger.info("  Initializing Creative Mind...")
        self.mind = CreativeMind(config_dict)
        
        # نظام الذاكرة
        logger.info("  Initializing Memory System...")
        self.memory = MemorySystem(config_dict, str(self.data_dir / 'memory'))
        
        # أنظمة الحماية
        logger.info("  Initializing Protection Systems...")
        self.circuit_breaker = CircuitBreaker(config_dict)
        self.anomaly_detector = AnomalyDetector(config_dict)
        
        # النموذج
        logger.info("  Loading ML Model...")
        self.model = self._load_model()
    
    def _load_model(self) -> Optional[EnsembleModel]:
        """تحميل النموذج المدرب"""
        try:
            model_file = self.model_path / 'ensemble_model.pt'
            if model_file.exists():
                model = EnsembleModel(
                    num_features=self.config.num_features,
                    sequence_length=self.config.sequence_length,
                    hidden_dim=128
                )
                model.load_state_dict(torch.load(model_file, map_location=self.device))
                model.to(self.device)
                model.eval()
                logger.info(f"  ✅ Model loaded from {model_file}")
                return model
            else:
                logger.warning(f"  ⚠️ Model file not found: {model_file}")
                return None
        except Exception as e:
            logger.error(f"  ❌ Failed to load model: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN DECISION PIPELINE
    # ═══════════════════════════════════════════════════════════════
    
    def decide(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        portfolio: Dict[str, Any] = None
    ) -> AgentDecision:
        """
        اتخاذ قرار تداول
        
        هذه هي الدالة الرئيسية التي تستدعيها من نظامك
        
        Args:
            symbol: رمز العملة (مثل BTCUSDT)
            market_data: بيانات السوق (OHLCV, features, etc.)
            portfolio: حالة المحفظة
            
        Returns:
            قرار التداول مع التفاصيل
        """
        start_time = datetime.now()
        self.state.total_decisions += 1
        
        try:
            # 1. فحص قاطع الدائرة
            breaker_status = self.circuit_breaker.check(
                daily_pnl=self.state.daily_pnl,
                weekly_pnl=self.state.weekly_pnl,
                consecutive_losses=self._count_consecutive_losses(),
                price_change_5m=market_data.get('price_change_5m', 0),
                volatility=market_data.get('volatility', 0)
            )
            
            if not breaker_status.is_trading_allowed:
                return self._create_hold_decision(
                    symbol,
                    f"Circuit breaker active: {breaker_status.last_trip_reason}",
                    confidence=0.0
                )
            
            # 2. كشف الشذوذ
            anomaly_result = self.anomaly_detector.detect(
                symbol=symbol,
                current_price=market_data.get('close', 0),
                features=market_data.get('features', {}),
                orderbook=market_data.get('orderbook')
            )
            
            if anomaly_result.risk_score > 0.7:
                return self._create_hold_decision(
                    symbol,
                    f"High anomaly risk: {anomaly_result.recommendations[0] if anomaly_result.recommendations else 'Unknown'}",
                    confidence=0.0
                )
            
            # 3. طبقة الإدراك
            perception_state = self.perception.perceive(
                symbol=symbol,
                ohlcv=market_data.get('ohlcv', []),
                features=market_data.get('features', {}),
                orderbook=market_data.get('orderbook')
            )
            
            # 4. طبقة الفهم
            understanding_state = self.understanding.understand(perception_state)
            
            # 5. طبقة التخطيط
            planning_state = self.planning.plan(
                understanding=understanding_state,
                portfolio=portfolio or {},
                open_positions=self.state.current_positions
            )
            
            # 6. العقل المبدع - التفكير
            creative_output = self.mind.think(
                symbol=symbol,
                market_state=understanding_state,
                memory_context=self.memory.build_context(symbol)
            )
            
            # 7. طبقة الحماية
            protection_status = self.protection.check(
                symbol=symbol,
                proposed_action=creative_output.get('suggested_action', 'HOLD'),
                position_size=planning_state.suggested_position_size,
                portfolio=portfolio or {},
                market_state=understanding_state
            )
            
            if not protection_status.is_safe:
                return self._create_hold_decision(
                    symbol,
                    f"Protection blocked: {protection_status.alerts[0].message if protection_status.alerts else 'Risk too high'}",
                    confidence=0.3
                )
            
            # 8. تنبؤ النموذج (إذا متوفر)
            model_prediction = None
            if self.model and 'features_tensor' in market_data:
                model_prediction = self._get_model_prediction(market_data['features_tensor'])
            
            # 9. طبقة القرار النهائي
            decision = self.decision.decide(
                symbol=symbol,
                understanding=understanding_state,
                planning=planning_state,
                creative_output=creative_output,
                model_prediction=model_prediction,
                protection_status=protection_status
            )
            
            # 10. تحديث الذاكرة
            self.memory.remember(
                {
                    'symbol': symbol,
                    'decision': decision.action,
                    'confidence': decision.confidence,
                    'regime': understanding_state.regime.value if hasattr(understanding_state, 'regime') else 'UNKNOWN',
                    'timestamp': datetime.now().isoformat()
                },
                memory_type=MemoryType.SHORT_TERM,
                importance=decision.confidence,
                tags=[symbol, decision.action]
            )
            
            # 11. إنشاء القرار النهائي
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return AgentDecision(
                symbol=symbol,
                action=decision.action,
                confidence=decision.confidence,
                position_size_percent=decision.position_size,
                entry_price=market_data.get('close', 0),
                stop_loss=self._calculate_stop_loss(market_data.get('close', 0), decision.action),
                take_profit_levels=self._calculate_take_profits(market_data.get('close', 0), decision.action),
                reasoning=decision.reasoning,
                risk_score=protection_status.risk_score,
                market_regime=understanding_state.regime.value if hasattr(understanding_state, 'regime') else 'UNKNOWN',
                creative_insight=creative_output.get('insight', ''),
                processing_time_ms=processing_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Decision error for {symbol}: {e}")
            return self._create_hold_decision(symbol, f"Error: {str(e)}", confidence=0.0)
    
    def _get_model_prediction(self, features_tensor: torch.Tensor) -> Dict[str, Any]:
        """الحصول على تنبؤ النموذج"""
        try:
            with torch.no_grad():
                features_tensor = features_tensor.to(self.device)
                if features_tensor.dim() == 2:
                    features_tensor = features_tensor.unsqueeze(0)
                
                output = self.model(features_tensor)
                
                return {
                    'prediction': output['final_prediction'].cpu().numpy()[0],
                    'regime_probs': output['regime_probs'].cpu().numpy()[0] if 'regime_probs' in output else None,
                    'confidence': output.get('confidence', 0.5)
                }
        except Exception as e:
            logger.error(f"Model prediction error: {e}")
            return None
    
    def _create_hold_decision(
        self,
        symbol: str,
        reason: str,
        confidence: float
    ) -> AgentDecision:
        """إنشاء قرار انتظار"""
        return AgentDecision(
            symbol=symbol,
            action='HOLD',
            confidence=confidence,
            position_size_percent=0,
            entry_price=0,
            stop_loss=0,
            take_profit_levels=[],
            reasoning=reason,
            risk_score=1.0,
            market_regime='UNKNOWN',
            creative_insight='',
            processing_time_ms=0,
            timestamp=datetime.now()
        )
    
    def _calculate_stop_loss(self, price: float, action: str) -> float:
        """حساب وقف الخسارة"""
        if action == 'BUY':
            return price * (1 - self.config.stop_loss / 100)
        elif action == 'SELL':
            return price * (1 + self.config.stop_loss / 100)
        return 0
    
    def _calculate_take_profits(self, price: float, action: str) -> List[float]:
        """حساب مستويات جني الأرباح"""
        if action not in ['BUY', 'SELL']:
            return []
        
        levels = []
        for tp in self.config.take_profit:
            if action == 'BUY':
                levels.append(price * (1 + tp / 100))
            else:
                levels.append(price * (1 - tp / 100))
        
        return levels
    
    def _count_consecutive_losses(self) -> int:
        """حساب الخسائر المتتالية"""
        # يمكن تحسين هذا بالرجوع للذاكرة
        return 0
    
    # ═══════════════════════════════════════════════════════════════
    # LEARNING & EVOLUTION
    # ═══════════════════════════════════════════════════════════════
    
    def learn_from_trade(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        exit_price: float,
        holding_time: int,
        market_regime: str,
        features_at_entry: Dict[str, float],
        features_at_exit: Dict[str, float],
        decision_confidence: float
    ) -> None:
        """
        التعلم من صفقة
        
        استدعِ هذه الدالة بعد إغلاق كل صفقة
        
        Args:
            symbol: رمز العملة
            action: الإجراء
            entry_price: سعر الدخول
            exit_price: سعر الخروج
            holding_time: وقت الاحتفاظ بالدقائق
            market_regime: نظام السوق
            features_at_entry: الميزات عند الدخول
            features_at_exit: الميزات عند الخروج
            decision_confidence: ثقة القرار
        """
        # التعلم في طبقة التطور
        lesson = self.evolution.learn_from_trade(
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            exit_price=exit_price,
            holding_time=holding_time,
            market_regime=market_regime,
            features_at_entry=features_at_entry,
            features_at_exit=features_at_exit,
            decision_confidence=decision_confidence
        )
        
        # تسجيل في الذاكرة
        pnl = (exit_price - entry_price) / entry_price * 100 if action == 'BUY' else (entry_price - exit_price) / entry_price * 100
        outcome = 'WIN' if pnl > 0 else 'LOSS'
        
        self.memory.record_trade(
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            context={'regime': market_regime, 'confidence': decision_confidence},
            outcome=outcome
        )
        
        # تحديث الإحصائيات
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl
        
        if outcome == 'WIN':
            self.state.successful_decisions += 1
        
        # تثبيت الذكريات المهمة
        self.memory.consolidate()
        
        logger.info(f"📚 Learned from {symbol} trade: {outcome} ({pnl:+.2f}%)")
    
    # ═══════════════════════════════════════════════════════════════
    # STATUS & MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على حالة الوكيل"""
        return {
            'is_active': self.state.is_active,
            'mode': self.state.mode,
            'total_decisions': self.state.total_decisions,
            'successful_decisions': self.state.successful_decisions,
            'success_rate': (
                self.state.successful_decisions / self.state.total_decisions
                if self.state.total_decisions > 0 else 0
            ),
            'daily_pnl': f"{self.state.daily_pnl:+.2f}%",
            'weekly_pnl': f"{self.state.weekly_pnl:+.2f}%",
            'open_positions': len(self.state.current_positions),
            'circuit_breaker': self.circuit_breaker.get_status().state.value,
            'memory_status': self.memory.get_status(),
            'evolution_status': self.evolution.get_status(),
            'last_update': self.state.last_update.isoformat()
        }
    
    def set_mode(self, mode: str) -> None:
        """تغيير وضع التداول"""
        if mode in ['aggressive', 'balanced', 'conservative']:
            self.state.mode = mode
            self.config.mode = mode
            logger.info(f"🔄 Mode changed to: {mode}")
    
    def pause(self) -> None:
        """إيقاف مؤقت"""
        self.state.is_active = False
        logger.info("⏸️ Agent paused")
    
    def resume(self) -> None:
        """استئناف"""
        self.state.is_active = True
        logger.info("▶️ Agent resumed")
    
    def reset_daily_stats(self) -> None:
        """إعادة تعيين الإحصائيات اليومية"""
        self.state.daily_pnl = 0
        logger.info("📊 Daily stats reset")
    
    def reset_weekly_stats(self) -> None:
        """إعادة تعيين الإحصائيات الأسبوعية"""
        self.state.weekly_pnl = 0
        logger.info("📊 Weekly stats reset")


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار الوكيل
    print("🦁 Testing Legendary Agent...")
    
    # إنشاء الوكيل
    agent = LegendaryAgent()
    
    # بيانات وهمية للاختبار
    market_data = {
        'close': 50000,
        'high': 50500,
        'low': 49500,
        'volume': 1000000,
        'price_change_5m': 0.5,
        'volatility': 2.0,
        'ohlcv': [[50000, 50500, 49500, 50200, 1000000]],
        'features': {
            'rsi_14': 55,
            'macd': 100,
            'macd_signal': 80,
            'bb_upper': 51000,
            'bb_lower': 49000,
            'atr_14': 500,
            'volume_sma_20': 900000
        }
    }
    
    portfolio = {
        'balance': 10000,
        'available': 8000,
        'positions': {}
    }
    
    # اتخاذ قرار
    decision = agent.decide('BTCUSDT', market_data, portfolio)
    
    print(f"\n📋 Decision:")
    print(f"   Symbol: {decision.symbol}")
    print(f"   Action: {decision.action}")
    print(f"   Confidence: {decision.confidence:.1%}")
    print(f"   Position Size: {decision.position_size_percent:.1f}%")
    print(f"   Stop Loss: ${decision.stop_loss:,.2f}")
    print(f"   Take Profits: {[f'${tp:,.2f}' for tp in decision.take_profit_levels]}")
    print(f"   Reasoning: {decision.reasoning}")
    print(f"   Risk Score: {decision.risk_score:.2f}")
    print(f"   Processing Time: {decision.processing_time_ms:.1f}ms")
    
    print(f"\n📊 Agent Status:")
    status = agent.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
