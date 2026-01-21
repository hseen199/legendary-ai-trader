"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Decision Layer
طبقة القرار - اتخاذ القرارات النهائية
═══════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


class DecisionType(Enum):
    """أنواع القرارات"""
    STRONG_BUY = "شراء قوي"
    BUY = "شراء"
    WEAK_BUY = "شراء ضعيف"
    HOLD = "انتظار"
    WEAK_SELL = "بيع ضعيف"
    SELL = "بيع"
    STRONG_SELL = "بيع قوي"
    EMERGENCY_EXIT = "خروج طارئ"


class DecisionSource(Enum):
    """مصادر القرار"""
    MODEL_PREDICTION = "تنبؤ النموذج"
    INNER_VOICE = "الصوت الداخلي"
    RISK_MANAGEMENT = "إدارة المخاطر"
    STRATEGY = "الاستراتيجية"
    PROTECTION = "الحماية"
    MANUAL_OVERRIDE = "تجاوز يدوي"


@dataclass
class TradingDecision:
    """قرار تداول"""
    symbol: str
    decision_type: DecisionType
    action: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: Optional[float] = None
    position_size_percent: float = 0.0
    stop_loss: Optional[float] = None
    take_profit_levels: List[float] = field(default_factory=list)
    trailing_stop: Optional[float] = None
    reasoning: str = ""
    sources: List[DecisionSource] = field(default_factory=list)
    risk_score: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionContext:
    """سياق القرار"""
    model_prediction: Dict[str, Any]
    inner_voice: Dict[str, Any]
    market_context: Dict[str, Any]
    trade_plan: Optional[Dict[str, Any]]
    protection_status: Dict[str, Any]
    portfolio_state: Dict[str, Any]


class DecisionLayer:
    """
    طبقة القرار
    
    مسؤولة عن:
    - دمج جميع المدخلات
    - اتخاذ القرار النهائي
    - تحديد معاملات الصفقة
    - التحقق من القرار
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        تهيئة طبقة القرار
        
        Args:
            config: إعدادات الطبقة
        """
        self.config = config or {}
        
        # أوزان المصادر
        self.source_weights = {
            DecisionSource.MODEL_PREDICTION: 0.35,
            DecisionSource.INNER_VOICE: 0.25,
            DecisionSource.STRATEGY: 0.20,
            DecisionSource.RISK_MANAGEMENT: 0.15,
            DecisionSource.PROTECTION: 0.05
        }
        
        # عتبات القرار
        self.thresholds = {
            'strong_buy': 0.80,
            'buy': 0.65,
            'weak_buy': 0.55,
            'hold_upper': 0.55,
            'hold_lower': 0.45,
            'weak_sell': 0.45,
            'sell': 0.35,
            'strong_sell': 0.20,
            'min_confidence': 0.50
        }
        
        # تاريخ القرارات
        self.decision_history: List[TradingDecision] = []
        self.max_history = 1000
        
        # إحصائيات
        self.stats = {
            'total_decisions': 0,
            'buy_decisions': 0,
            'sell_decisions': 0,
            'hold_decisions': 0,
            'overridden_decisions': 0
        }
        
        logger.info("⚖️ DecisionLayer initialized")
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN DECISION MAKING
    # ═══════════════════════════════════════════════════════════════
    
    def decide(
        self,
        symbol: str,
        context: DecisionContext
    ) -> TradingDecision:
        """
        اتخاذ قرار
        
        Args:
            symbol: رمز العملة
            context: سياق القرار
            
        Returns:
            قرار التداول
        """
        # جمع الإشارات
        signals = self._collect_signals(context)
        
        # حساب الدرجة المرجحة
        weighted_score = self._calculate_weighted_score(signals)
        
        # تحديد نوع القرار
        decision_type = self._determine_decision_type(weighted_score)
        
        # تحديد الإجراء
        action = self._determine_action(decision_type)
        
        # حساب الثقة
        confidence = self._calculate_confidence(signals, weighted_score)
        
        # التحقق من الحماية
        protection_override = self._check_protection(context.protection_status)
        if protection_override:
            decision_type = protection_override
            action = 'HOLD' if protection_override == DecisionType.HOLD else 'SELL'
            confidence = 0.95
        
        # تحديد معاملات الصفقة
        trade_params = self._determine_trade_params(
            symbol, action, context, confidence
        )
        
        # بناء التبرير
        reasoning = self._build_reasoning(signals, decision_type, context)
        
        # تحديد المصادر
        sources = self._identify_sources(signals)
        
        # حساب درجة المخاطرة
        risk_score = self._calculate_risk_score(context)
        
        decision = TradingDecision(
            symbol=symbol,
            decision_type=decision_type,
            action=action,
            confidence=confidence,
            entry_price=trade_params.get('entry_price'),
            position_size_percent=trade_params.get('position_size', 0),
            stop_loss=trade_params.get('stop_loss'),
            take_profit_levels=trade_params.get('take_profits', []),
            trailing_stop=trade_params.get('trailing_stop'),
            reasoning=reasoning,
            sources=sources,
            risk_score=risk_score,
            metadata={
                'signals': signals,
                'weighted_score': weighted_score
            }
        )
        
        # حفظ القرار
        self._save_decision(decision)
        
        return decision
    
    # ═══════════════════════════════════════════════════════════════
    # SIGNAL COLLECTION
    # ═══════════════════════════════════════════════════════════════
    
    def _collect_signals(self, context: DecisionContext) -> Dict[str, float]:
        """جمع الإشارات"""
        signals = {}
        
        # إشارة النموذج
        model = context.model_prediction
        if model:
            model_action = model.get('action', 'HOLD')
            model_confidence = model.get('confidence', 0.5)
            
            if model_action == 'BUY':
                signals['model'] = 0.5 + model_confidence * 0.5
            elif model_action == 'SELL':
                signals['model'] = 0.5 - model_confidence * 0.5
            else:
                signals['model'] = 0.5
        
        # إشارة الصوت الداخلي
        inner = context.inner_voice
        if inner:
            inner_decision = inner.get('decision', 'HOLD')
            inner_confidence = inner.get('confidence', 0.5)
            
            if inner_decision == 'BUY':
                signals['inner_voice'] = 0.5 + inner_confidence * 0.5
            elif inner_decision == 'SELL':
                signals['inner_voice'] = 0.5 - inner_confidence * 0.5
            else:
                signals['inner_voice'] = 0.5
        
        # إشارة السوق
        market = context.market_context
        if market:
            regime = market.get('regime', 'RANGING')
            
            regime_scores = {
                'STRONG_BULL': 0.85,
                'BULL': 0.70,
                'WEAK_BULL': 0.60,
                'RANGING': 0.50,
                'WEAK_BEAR': 0.40,
                'BEAR': 0.30,
                'STRONG_BEAR': 0.15,
                'CRASH': 0.05,
                'EUPHORIA': 0.50  # محايد لأنه خطر
            }
            signals['market'] = regime_scores.get(regime, 0.5)
        
        # إشارة الاستراتيجية
        plan = context.trade_plan
        if plan:
            plan_action = plan.get('action', 'HOLD')
            plan_confidence = plan.get('confidence', 0.5)
            
            if plan_action == 'BUY':
                signals['strategy'] = 0.5 + plan_confidence * 0.5
            elif plan_action == 'SELL':
                signals['strategy'] = 0.5 - plan_confidence * 0.5
            else:
                signals['strategy'] = 0.5
        
        # إشارة إدارة المخاطر
        portfolio = context.portfolio_state
        if portfolio:
            heat = portfolio.get('portfolio_heat', 0)
            daily_pnl = portfolio.get('daily_pnl', 0)
            
            # إذا كانت المحفظة ساخنة، نميل للحذر
            if heat > 70:
                signals['risk'] = 0.3
            elif heat > 50:
                signals['risk'] = 0.4
            elif daily_pnl < -3:
                signals['risk'] = 0.3
            else:
                signals['risk'] = 0.5
        
        return signals
    
    def _calculate_weighted_score(self, signals: Dict[str, float]) -> float:
        """حساب الدرجة المرجحة"""
        total_weight = 0
        weighted_sum = 0
        
        signal_to_source = {
            'model': DecisionSource.MODEL_PREDICTION,
            'inner_voice': DecisionSource.INNER_VOICE,
            'strategy': DecisionSource.STRATEGY,
            'risk': DecisionSource.RISK_MANAGEMENT,
            'market': DecisionSource.MODEL_PREDICTION  # نستخدم وزن النموذج
        }
        
        for signal_name, signal_value in signals.items():
            source = signal_to_source.get(signal_name, DecisionSource.MODEL_PREDICTION)
            weight = self.source_weights.get(source, 0.1)
            
            weighted_sum += signal_value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    # ═══════════════════════════════════════════════════════════════
    # DECISION DETERMINATION
    # ═══════════════════════════════════════════════════════════════
    
    def _determine_decision_type(self, score: float) -> DecisionType:
        """تحديد نوع القرار"""
        if score >= self.thresholds['strong_buy']:
            return DecisionType.STRONG_BUY
        elif score >= self.thresholds['buy']:
            return DecisionType.BUY
        elif score >= self.thresholds['weak_buy']:
            return DecisionType.WEAK_BUY
        elif score >= self.thresholds['hold_lower']:
            return DecisionType.HOLD
        elif score >= self.thresholds['sell']:
            return DecisionType.WEAK_SELL
        elif score >= self.thresholds['strong_sell']:
            return DecisionType.SELL
        else:
            return DecisionType.STRONG_SELL
    
    def _determine_action(self, decision_type: DecisionType) -> str:
        """تحديد الإجراء"""
        buy_types = [DecisionType.STRONG_BUY, DecisionType.BUY, DecisionType.WEAK_BUY]
        sell_types = [DecisionType.STRONG_SELL, DecisionType.SELL, DecisionType.WEAK_SELL, DecisionType.EMERGENCY_EXIT]
        
        if decision_type in buy_types:
            return 'BUY'
        elif decision_type in sell_types:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_confidence(
        self,
        signals: Dict[str, float],
        weighted_score: float
    ) -> float:
        """حساب الثقة"""
        # الثقة تعتمد على:
        # 1. مدى اتفاق الإشارات
        # 2. قوة الإشارة
        
        if not signals:
            return 0.5
        
        # حساب التباين
        values = list(signals.values())
        variance = np.var(values)
        
        # اتفاق عالي = تباين منخفض
        agreement_score = 1 - min(1, variance * 4)
        
        # قوة الإشارة
        strength = abs(weighted_score - 0.5) * 2
        
        # الثقة النهائية
        confidence = agreement_score * 0.4 + strength * 0.6
        
        return max(0.1, min(0.95, confidence))
    
    # ═══════════════════════════════════════════════════════════════
    # PROTECTION CHECK
    # ═══════════════════════════════════════════════════════════════
    
    def _check_protection(
        self,
        protection_status: Dict[str, Any]
    ) -> Optional[DecisionType]:
        """التحقق من الحماية"""
        if not protection_status:
            return None
        
        # فحص الانهيار
        if protection_status.get('flash_crash_detected'):
            logger.warning("⚠️ Flash crash detected - forcing HOLD")
            return DecisionType.HOLD
        
        # فحص التلاعب
        if protection_status.get('manipulation_detected'):
            logger.warning("⚠️ Manipulation detected - forcing HOLD")
            return DecisionType.HOLD
        
        # فحص حد الخسارة
        if protection_status.get('daily_loss_limit_reached'):
            logger.warning("⚠️ Daily loss limit reached - forcing HOLD")
            return DecisionType.HOLD
        
        # فحص الخروج الطارئ
        if protection_status.get('emergency_exit_required'):
            logger.warning("🚨 Emergency exit required!")
            return DecisionType.EMERGENCY_EXIT
        
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # TRADE PARAMETERS
    # ═══════════════════════════════════════════════════════════════
    
    def _determine_trade_params(
        self,
        symbol: str,
        action: str,
        context: DecisionContext,
        confidence: float
    ) -> Dict[str, Any]:
        """تحديد معاملات الصفقة"""
        if action == 'HOLD':
            return {}
        
        # الحصول على المعاملات من الخطة
        plan = context.trade_plan or {}
        
        entry_price = plan.get('entry_price', context.market_context.get('current_price', 0))
        
        # حجم المركز يعتمد على الثقة
        base_size = 7.5
        max_size = 15.0
        position_size = base_size + (max_size - base_size) * confidence
        
        # وقف الخسارة
        stop_loss_percent = 2.0
        if action == 'BUY':
            stop_loss = entry_price * (1 - stop_loss_percent / 100)
        else:
            stop_loss = entry_price * (1 + stop_loss_percent / 100)
        
        # مستويات جني الأرباح
        tp_percents = [1.5, 3.5, 6.0]
        take_profits = []
        for tp in tp_percents:
            if action == 'BUY':
                take_profits.append(entry_price * (1 + tp / 100))
            else:
                take_profits.append(entry_price * (1 - tp / 100))
        
        # وقف متحرك
        trailing_stop = 2.0
        
        return {
            'entry_price': entry_price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profits': take_profits,
            'trailing_stop': trailing_stop
        }
    
    # ═══════════════════════════════════════════════════════════════
    # REASONING & SOURCES
    # ═══════════════════════════════════════════════════════════════
    
    def _build_reasoning(
        self,
        signals: Dict[str, float],
        decision_type: DecisionType,
        context: DecisionContext
    ) -> str:
        """بناء التبرير"""
        parts = []
        
        # القرار الرئيسي
        parts.append(f"القرار: {decision_type.value}")
        
        # تحليل الإشارات
        if 'model' in signals:
            model_signal = signals['model']
            if model_signal > 0.6:
                parts.append("النموذج يشير للشراء")
            elif model_signal < 0.4:
                parts.append("النموذج يشير للبيع")
        
        if 'inner_voice' in signals:
            inner_signal = signals['inner_voice']
            if inner_signal > 0.6:
                parts.append("الصوت الداخلي إيجابي")
            elif inner_signal < 0.4:
                parts.append("الصوت الداخلي سلبي")
        
        if 'market' in signals:
            market_signal = signals['market']
            if market_signal > 0.6:
                parts.append("السوق صاعد")
            elif market_signal < 0.4:
                parts.append("السوق هابط")
        
        # إضافة السياق
        inner_voice = context.inner_voice
        if inner_voice and inner_voice.get('debate_conclusion'):
            parts.append(f"النقاش الداخلي: {inner_voice['debate_conclusion']}")
        
        return " | ".join(parts)
    
    def _identify_sources(self, signals: Dict[str, float]) -> List[DecisionSource]:
        """تحديد المصادر"""
        sources = []
        
        signal_to_source = {
            'model': DecisionSource.MODEL_PREDICTION,
            'inner_voice': DecisionSource.INNER_VOICE,
            'strategy': DecisionSource.STRATEGY,
            'risk': DecisionSource.RISK_MANAGEMENT
        }
        
        for signal_name in signals.keys():
            source = signal_to_source.get(signal_name)
            if source and source not in sources:
                sources.append(source)
        
        return sources
    
    def _calculate_risk_score(self, context: DecisionContext) -> float:
        """حساب درجة المخاطرة"""
        risk_factors = 0
        
        # تقلب السوق
        market = context.market_context
        if market:
            volatility = market.get('volatility', 'MEDIUM')
            if volatility == 'EXTREME':
                risk_factors += 0.3
            elif volatility == 'HIGH':
                risk_factors += 0.2
        
        # حرارة المحفظة
        portfolio = context.portfolio_state
        if portfolio:
            heat = portfolio.get('portfolio_heat', 0)
            risk_factors += heat / 200  # 0-0.5
        
        # حالة الحماية
        protection = context.protection_status
        if protection:
            if protection.get('any_warning'):
                risk_factors += 0.2
        
        return min(1.0, risk_factors)
    
    # ═══════════════════════════════════════════════════════════════
    # DECISION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    
    def _save_decision(self, decision: TradingDecision) -> None:
        """حفظ القرار"""
        self.decision_history.append(decision)
        
        # تحديث الإحصائيات
        self.stats['total_decisions'] += 1
        if decision.action == 'BUY':
            self.stats['buy_decisions'] += 1
        elif decision.action == 'SELL':
            self.stats['sell_decisions'] += 1
        else:
            self.stats['hold_decisions'] += 1
        
        # الحفاظ على حجم التاريخ
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history:]
    
    def get_recent_decisions(
        self,
        symbol: Optional[str] = None,
        count: int = 10
    ) -> List[TradingDecision]:
        """الحصول على القرارات الأخيرة"""
        decisions = self.decision_history
        
        if symbol:
            decisions = [d for d in decisions if d.symbol == symbol]
        
        return decisions[-count:]
    
    def override_decision(
        self,
        original: TradingDecision,
        new_action: str,
        reason: str
    ) -> TradingDecision:
        """تجاوز قرار"""
        self.stats['overridden_decisions'] += 1
        
        return TradingDecision(
            symbol=original.symbol,
            decision_type=DecisionType.HOLD if new_action == 'HOLD' else original.decision_type,
            action=new_action,
            confidence=0.95,
            reasoning=f"تجاوز يدوي: {reason}",
            sources=[DecisionSource.MANUAL_OVERRIDE],
            risk_score=original.risk_score
        )
    
    # ═══════════════════════════════════════════════════════════════
    # STATUS
    # ═══════════════════════════════════════════════════════════════
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على حالة الطبقة"""
        return {
            'stats': self.stats,
            'history_size': len(self.decision_history),
            'thresholds': self.thresholds,
            'source_weights': {k.value: v for k, v in self.source_weights.items()}
        }


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار طبقة القرار
    decision_layer = DecisionLayer()
    
    context = DecisionContext(
        model_prediction={
            'action': 'BUY',
            'confidence': 0.75
        },
        inner_voice={
            'decision': 'BUY',
            'confidence': 0.65,
            'debate_conclusion': 'الأغلبية تميل للشراء'
        },
        market_context={
            'regime': 'BULL',
            'volatility': 'MEDIUM',
            'current_price': 50000
        },
        trade_plan={
            'action': 'BUY',
            'confidence': 0.70,
            'entry_price': 50000
        },
        protection_status={
            'flash_crash_detected': False,
            'manipulation_detected': False
        },
        portfolio_state={
            'portfolio_heat': 30,
            'daily_pnl': 1.5
        }
    )
    
    decision = decision_layer.decide('BTCUSDT', context)
    
    print("⚖️ Trading Decision:")
    print(f"Symbol: {decision.symbol}")
    print(f"Type: {decision.decision_type.value}")
    print(f"Action: {decision.action}")
    print(f"Confidence: {decision.confidence:.2%}")
    print(f"Entry Price: ${decision.entry_price:,.2f}")
    print(f"Position Size: {decision.position_size_percent:.1f}%")
    print(f"Stop Loss: ${decision.stop_loss:,.2f}")
    print(f"Take Profits: {decision.take_profit_levels}")
    print(f"Risk Score: {decision.risk_score:.2f}")
    print(f"\nReasoning: {decision.reasoning}")
    print(f"Sources: {[s.value for s in decision.sources]}")
