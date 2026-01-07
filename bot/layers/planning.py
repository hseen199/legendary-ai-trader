"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Planning Layer
طبقة التخطيط - التخطيط الاستراتيجي وإدارة المحفظة
═══════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from loguru import logger


class PlanType(Enum):
    """أنواع الخطط"""
    AGGRESSIVE = "عنيف"
    BALANCED = "متوازن"
    CONSERVATIVE = "محافظ"
    DEFENSIVE = "دفاعي"


class PositionStrategy(Enum):
    """استراتيجيات المراكز"""
    FULL_ENTRY = "دخول كامل"
    SCALED_ENTRY = "دخول متدرج"
    DCA = "متوسط التكلفة"
    PYRAMIDING = "الهرم"


@dataclass
class TradePlan:
    """خطة صفقة"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    entry_price: float
    position_size_percent: float
    stop_loss: float
    take_profit_levels: List[Tuple[float, float]]  # (price, percent_to_close)
    trailing_stop_percent: Optional[float] = None
    max_holding_time: Optional[int] = None  # بالساعات
    entry_strategy: PositionStrategy = PositionStrategy.FULL_ENTRY
    priority: int = 5  # 1-10
    reasoning: str = ""
    confidence: float = 0.5


@dataclass
class PortfolioPlan:
    """خطة المحفظة"""
    timestamp: datetime
    total_exposure_target: float  # النسبة المستهدفة من المحفظة في السوق
    max_positions: int
    position_allocation: Dict[str, float]  # symbol -> percent
    rebalance_needed: bool = False
    cash_reserve_percent: float = 20.0
    risk_budget: float = 5.0  # الحد الأقصى للخسارة اليومية


@dataclass
class PlanningState:
    """حالة التخطيط"""
    timestamp: datetime
    plan_type: PlanType
    trade_plans: List[TradePlan]
    portfolio_plan: PortfolioPlan
    active_opportunities: List[str]
    blocked_symbols: List[str]
    daily_trades_remaining: int
    risk_utilized: float  # نسبة المخاطر المستخدمة


class PlanningLayer:
    """
    طبقة التخطيط
    
    مسؤولة عن:
    - التخطيط الاستراتيجي
    - إدارة المحفظة
    - تخصيص المراكز
    - إدارة المخاطر على مستوى المحفظة
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        تهيئة طبقة التخطيط
        
        Args:
            config: إعدادات الطبقة
        """
        self.config = config or {}
        
        # إعدادات التداول
        self.trading_config = {
            'stop_loss_percent': self.config.get('stop_loss', 2.0),
            'take_profit_levels': self.config.get('take_profit', [1.5, 3.5, 6.0]),
            'trailing_stop_activation': self.config.get('trailing_activation', 4.0),
            'trailing_stop_percent': self.config.get('trailing_percent', 2.0),
            'min_position_size': self.config.get('min_position', 7.5),
            'max_position_size': self.config.get('max_position', 15.0),
            'max_daily_trades': self.config.get('max_daily_trades', 10),
            'max_positions': self.config.get('max_positions', 5),
            'max_daily_loss': self.config.get('max_daily_loss', 5.0),
            'max_portfolio_heat': self.config.get('max_heat', 80.0)
        }
        
        # حالة المحفظة
        self.portfolio_state = {
            'open_positions': {},
            'daily_pnl': 0.0,
            'daily_trades': 0,
            'blocked_symbols': [],
            'portfolio_heat': 0.0
        }
        
        # تاريخ الخطط
        self.plan_history: List[PlanningState] = []
        
        logger.info("📋 PlanningLayer initialized")
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN PLANNING
    # ═══════════════════════════════════════════════════════════════
    
    def plan(
        self,
        opportunities: List[Dict[str, Any]],
        market_context: Dict[str, Any],
        portfolio_balance: float
    ) -> PlanningState:
        """
        إنشاء خطة
        
        Args:
            opportunities: الفرص المتاحة
            market_context: سياق السوق
            portfolio_balance: رصيد المحفظة
            
        Returns:
            حالة التخطيط
        """
        # تحديد نوع الخطة
        plan_type = self._determine_plan_type(market_context)
        
        # تصفية الفرص
        filtered_opportunities = self._filter_opportunities(opportunities)
        
        # ترتيب الفرص
        ranked_opportunities = self._rank_opportunities(filtered_opportunities)
        
        # إنشاء خطط الصفقات
        trade_plans = self._create_trade_plans(
            ranked_opportunities,
            plan_type,
            portfolio_balance
        )
        
        # إنشاء خطة المحفظة
        portfolio_plan = self._create_portfolio_plan(
            trade_plans,
            portfolio_balance,
            plan_type
        )
        
        # حساب المخاطر المستخدمة
        risk_utilized = self._calculate_risk_utilized()
        
        state = PlanningState(
            timestamp=datetime.now(),
            plan_type=plan_type,
            trade_plans=trade_plans,
            portfolio_plan=portfolio_plan,
            active_opportunities=[o['symbol'] for o in ranked_opportunities[:5]],
            blocked_symbols=self.portfolio_state['blocked_symbols'],
            daily_trades_remaining=self.trading_config['max_daily_trades'] - self.portfolio_state['daily_trades'],
            risk_utilized=risk_utilized
        )
        
        self.plan_history.append(state)
        
        return state
    
    # ═══════════════════════════════════════════════════════════════
    # PLAN TYPE DETERMINATION
    # ═══════════════════════════════════════════════════════════════
    
    def _determine_plan_type(self, market_context: Dict) -> PlanType:
        """تحديد نوع الخطة"""
        regime = market_context.get('regime', 'RANGING')
        volatility = market_context.get('volatility', 'MEDIUM')
        daily_pnl = self.portfolio_state['daily_pnl']
        
        # إذا كانت الخسائر كبيرة، نتحول للدفاع
        if daily_pnl < -self.trading_config['max_daily_loss'] * 0.7:
            return PlanType.DEFENSIVE
        
        # في التقلب العالي، نكون محافظين
        if volatility in ['HIGH', 'EXTREME']:
            return PlanType.CONSERVATIVE
        
        # في الاتجاه القوي، نكون عنيفين
        if regime in ['STRONG_BULL', 'STRONG_BEAR']:
            return PlanType.AGGRESSIVE
        
        # في التذبذب، نكون محافظين
        if regime == 'RANGING':
            return PlanType.CONSERVATIVE
        
        return PlanType.BALANCED
    
    # ═══════════════════════════════════════════════════════════════
    # OPPORTUNITY FILTERING & RANKING
    # ═══════════════════════════════════════════════════════════════
    
    def _filter_opportunities(
        self,
        opportunities: List[Dict]
    ) -> List[Dict]:
        """تصفية الفرص"""
        filtered = []
        
        for opp in opportunities:
            symbol = opp.get('symbol', '')
            
            # تجاهل العملات المحظورة
            if symbol in self.portfolio_state['blocked_symbols']:
                continue
            
            # تجاهل إذا كان لدينا مركز مفتوح
            if symbol in self.portfolio_state['open_positions']:
                continue
            
            # تجاهل إذا كانت الثقة منخفضة
            if opp.get('confidence', 0) < 0.5:
                continue
            
            filtered.append(opp)
        
        return filtered
    
    def _rank_opportunities(
        self,
        opportunities: List[Dict]
    ) -> List[Dict]:
        """ترتيب الفرص"""
        def score_opportunity(opp: Dict) -> float:
            score = 0.0
            
            # الثقة (40%)
            score += opp.get('confidence', 0.5) * 0.4
            
            # نسبة المخاطرة للعائد (30%)
            risk = opp.get('risk', 2)
            reward = opp.get('potential_reward', 3)
            rr_ratio = reward / risk if risk > 0 else 0
            score += min(1.0, rr_ratio / 3) * 0.3
            
            # قوة الإشارة (20%)
            score += opp.get('signal_strength', 0.5) * 0.2
            
            # الحجم (10%)
            volume_score = opp.get('volume_score', 0.5)
            score += volume_score * 0.1
            
            return score
        
        ranked = sorted(opportunities, key=score_opportunity, reverse=True)
        return ranked
    
    # ═══════════════════════════════════════════════════════════════
    # TRADE PLAN CREATION
    # ═══════════════════════════════════════════════════════════════
    
    def _create_trade_plans(
        self,
        opportunities: List[Dict],
        plan_type: PlanType,
        portfolio_balance: float
    ) -> List[TradePlan]:
        """إنشاء خطط الصفقات"""
        plans = []
        
        # تحديد عدد المراكز المتاحة
        current_positions = len(self.portfolio_state['open_positions'])
        available_slots = self.trading_config['max_positions'] - current_positions
        
        # تحديد الحجم حسب نوع الخطة
        size_multiplier = {
            PlanType.AGGRESSIVE: 1.2,
            PlanType.BALANCED: 1.0,
            PlanType.CONSERVATIVE: 0.8,
            PlanType.DEFENSIVE: 0.5
        }.get(plan_type, 1.0)
        
        for i, opp in enumerate(opportunities[:available_slots]):
            plan = self._create_single_trade_plan(
                opp,
                plan_type,
                size_multiplier,
                priority=10 - i
            )
            plans.append(plan)
        
        return plans
    
    def _create_single_trade_plan(
        self,
        opportunity: Dict,
        plan_type: PlanType,
        size_multiplier: float,
        priority: int
    ) -> TradePlan:
        """إنشاء خطة صفقة واحدة"""
        symbol = opportunity.get('symbol', '')
        entry_price = opportunity.get('entry_price', 0)
        confidence = opportunity.get('confidence', 0.5)
        action = opportunity.get('action', 'BUY')
        
        # حساب حجم المركز
        base_size = self.trading_config['min_position_size']
        max_size = self.trading_config['max_position_size']
        
        # الحجم يعتمد على الثقة
        position_size = base_size + (max_size - base_size) * confidence
        position_size *= size_multiplier
        position_size = max(base_size, min(max_size, position_size))
        
        # حساب وقف الخسارة
        stop_loss_percent = self.trading_config['stop_loss_percent']
        if action == 'BUY':
            stop_loss = entry_price * (1 - stop_loss_percent / 100)
        else:
            stop_loss = entry_price * (1 + stop_loss_percent / 100)
        
        # حساب مستويات جني الأرباح
        tp_levels = self.trading_config['take_profit_levels']
        take_profit_levels = []
        
        for i, tp_percent in enumerate(tp_levels):
            if action == 'BUY':
                tp_price = entry_price * (1 + tp_percent / 100)
            else:
                tp_price = entry_price * (1 - tp_percent / 100)
            
            # توزيع الكمية على المستويات
            if i == 0:
                close_percent = 0.4  # 40% في المستوى الأول
            elif i == 1:
                close_percent = 0.35  # 35% في الثاني
            else:
                close_percent = 0.25  # 25% في الثالث
            
            take_profit_levels.append((tp_price, close_percent))
        
        # تحديد استراتيجية الدخول
        if plan_type == PlanType.AGGRESSIVE:
            entry_strategy = PositionStrategy.FULL_ENTRY
        elif plan_type == PlanType.CONSERVATIVE:
            entry_strategy = PositionStrategy.SCALED_ENTRY
        else:
            entry_strategy = PositionStrategy.FULL_ENTRY
        
        # بناء التبرير
        reasoning = self._build_trade_reasoning(opportunity)
        
        return TradePlan(
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            position_size_percent=position_size,
            stop_loss=stop_loss,
            take_profit_levels=take_profit_levels,
            trailing_stop_percent=self.trading_config['trailing_stop_percent'],
            max_holding_time=48,  # 48 ساعة كحد أقصى
            entry_strategy=entry_strategy,
            priority=priority,
            reasoning=reasoning,
            confidence=confidence
        )
    
    def _build_trade_reasoning(self, opportunity: Dict) -> str:
        """بناء تبرير الصفقة"""
        parts = []
        
        if opportunity.get('trend_aligned'):
            parts.append("الاتجاه متوافق")
        
        if opportunity.get('momentum_positive'):
            parts.append("زخم إيجابي")
        
        if opportunity.get('volume_confirmed'):
            parts.append("حجم مؤكد")
        
        if opportunity.get('near_support'):
            parts.append("قرب الدعم")
        
        return " | ".join(parts) if parts else "فرصة تداول"
    
    # ═══════════════════════════════════════════════════════════════
    # PORTFOLIO PLAN
    # ═══════════════════════════════════════════════════════════════
    
    def _create_portfolio_plan(
        self,
        trade_plans: List[TradePlan],
        portfolio_balance: float,
        plan_type: PlanType
    ) -> PortfolioPlan:
        """إنشاء خطة المحفظة"""
        # تحديد التعرض المستهدف
        exposure_targets = {
            PlanType.AGGRESSIVE: 80.0,
            PlanType.BALANCED: 60.0,
            PlanType.CONSERVATIVE: 40.0,
            PlanType.DEFENSIVE: 20.0
        }
        target_exposure = exposure_targets.get(plan_type, 60.0)
        
        # حساب التخصيص
        position_allocation = {}
        total_allocation = 0
        
        for plan in trade_plans:
            if total_allocation + plan.position_size_percent <= target_exposure:
                position_allocation[plan.symbol] = plan.position_size_percent
                total_allocation += plan.position_size_percent
        
        # احتياطي نقدي
        cash_reserve = 100 - target_exposure
        
        # ميزانية المخاطر
        risk_budget = self.trading_config['max_daily_loss'] - abs(self.portfolio_state['daily_pnl'])
        
        return PortfolioPlan(
            timestamp=datetime.now(),
            total_exposure_target=target_exposure,
            max_positions=self.trading_config['max_positions'],
            position_allocation=position_allocation,
            rebalance_needed=False,
            cash_reserve_percent=cash_reserve,
            risk_budget=max(0, risk_budget)
        )
    
    # ═══════════════════════════════════════════════════════════════
    # RISK MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    
    def _calculate_risk_utilized(self) -> float:
        """حساب المخاطر المستخدمة"""
        max_loss = self.trading_config['max_daily_loss']
        current_loss = abs(min(0, self.portfolio_state['daily_pnl']))
        
        return (current_loss / max_loss) * 100 if max_loss > 0 else 0
    
    def update_portfolio_state(
        self,
        open_positions: Dict[str, Dict],
        daily_pnl: float,
        daily_trades: int
    ) -> None:
        """تحديث حالة المحفظة"""
        self.portfolio_state['open_positions'] = open_positions
        self.portfolio_state['daily_pnl'] = daily_pnl
        self.portfolio_state['daily_trades'] = daily_trades
        
        # حساب حرارة المحفظة
        total_exposure = sum(
            pos.get('size_percent', 0)
            for pos in open_positions.values()
        )
        self.portfolio_state['portfolio_heat'] = total_exposure
    
    def block_symbol(self, symbol: str, reason: str = "") -> None:
        """حظر عملة"""
        if symbol not in self.portfolio_state['blocked_symbols']:
            self.portfolio_state['blocked_symbols'].append(symbol)
            logger.warning(f"🚫 Blocked {symbol}: {reason}")
    
    def unblock_symbol(self, symbol: str) -> None:
        """رفع الحظر عن عملة"""
        if symbol in self.portfolio_state['blocked_symbols']:
            self.portfolio_state['blocked_symbols'].remove(symbol)
            logger.info(f"✅ Unblocked {symbol}")
    
    def reset_daily_stats(self) -> None:
        """إعادة تعيين الإحصائيات اليومية"""
        self.portfolio_state['daily_pnl'] = 0.0
        self.portfolio_state['daily_trades'] = 0
        self.portfolio_state['blocked_symbols'] = []
        logger.info("📊 Daily stats reset")
    
    # ═══════════════════════════════════════════════════════════════
    # POSITION SIZING
    # ═══════════════════════════════════════════════════════════════
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        portfolio_balance: float,
        confidence: float = 0.5
    ) -> Dict[str, float]:
        """
        حساب حجم المركز
        
        Args:
            symbol: رمز العملة
            entry_price: سعر الدخول
            stop_loss: وقف الخسارة
            portfolio_balance: رصيد المحفظة
            confidence: مستوى الثقة
            
        Returns:
            تفاصيل الحجم
        """
        # حساب المخاطرة لكل صفقة
        risk_per_trade = self.trading_config['max_daily_loss'] / self.trading_config['max_positions']
        
        # حساب المسافة لوقف الخسارة
        stop_distance = abs(entry_price - stop_loss) / entry_price
        
        # حساب الحجم بناءً على المخاطرة
        if stop_distance > 0:
            risk_based_size = (risk_per_trade / stop_distance) / 100
        else:
            risk_based_size = self.trading_config['min_position_size']
        
        # تعديل بناءً على الثقة
        confidence_adjusted = risk_based_size * (0.5 + confidence * 0.5)
        
        # التأكد من الحدود
        final_size = max(
            self.trading_config['min_position_size'],
            min(self.trading_config['max_position_size'], confidence_adjusted)
        )
        
        # التأكد من عدم تجاوز حرارة المحفظة
        available_heat = self.trading_config['max_portfolio_heat'] - self.portfolio_state['portfolio_heat']
        final_size = min(final_size, available_heat)
        
        return {
            'size_percent': final_size,
            'size_value': portfolio_balance * final_size / 100,
            'risk_percent': stop_distance * final_size,
            'max_loss': portfolio_balance * stop_distance * final_size / 100
        }
    
    # ═══════════════════════════════════════════════════════════════
    # STATUS
    # ═══════════════════════════════════════════════════════════════
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على حالة الطبقة"""
        return {
            'portfolio_state': self.portfolio_state,
            'trading_config': self.trading_config,
            'plans_count': len(self.plan_history),
            'last_plan_type': (
                self.plan_history[-1].plan_type.value
                if self.plan_history else None
            )
        }


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار طبقة التخطيط
    planning = PlanningLayer()
    
    opportunities = [
        {
            'symbol': 'BTCUSDT',
            'action': 'BUY',
            'entry_price': 50000,
            'confidence': 0.75,
            'risk': 2,
            'potential_reward': 4,
            'signal_strength': 0.8,
            'volume_score': 0.7,
            'trend_aligned': True,
            'momentum_positive': True
        },
        {
            'symbol': 'ETHUSDT',
            'action': 'BUY',
            'entry_price': 3000,
            'confidence': 0.65,
            'risk': 2,
            'potential_reward': 3,
            'signal_strength': 0.6,
            'volume_score': 0.6
        }
    ]
    
    market_context = {
        'regime': 'BULL',
        'volatility': 'MEDIUM'
    }
    
    state = planning.plan(opportunities, market_context, 10000)
    
    print("📋 Planning State:")
    print(f"Plan Type: {state.plan_type.value}")
    print(f"Risk Utilized: {state.risk_utilized:.1f}%")
    print(f"Daily Trades Remaining: {state.daily_trades_remaining}")
    
    print("\n📊 Trade Plans:")
    for plan in state.trade_plans:
        print(f"\n{plan.symbol}:")
        print(f"  Action: {plan.action}")
        print(f"  Entry: ${plan.entry_price:,.2f}")
        print(f"  Size: {plan.position_size_percent:.1f}%")
        print(f"  Stop Loss: ${plan.stop_loss:,.2f}")
        print(f"  Take Profits: {plan.take_profit_levels}")
        print(f"  Confidence: {plan.confidence:.2%}")
    
    print(f"\n💼 Portfolio Plan:")
    print(f"  Target Exposure: {state.portfolio_plan.total_exposure_target}%")
    print(f"  Cash Reserve: {state.portfolio_plan.cash_reserve_percent}%")
    print(f"  Risk Budget: {state.portfolio_plan.risk_budget}%")
