"""
Trading System Integration Service
خدمة دمج نظام التداول مع المنصة
مدمج من نسخة المستخدم (crowdfund/integration.py)
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models import Trade, AgentState, NAVHistory
from app.services.binance_subaccount_service import get_binance_service
from app.services.nav_service import nav_service

logger = logging.getLogger(__name__)


class TradingIntegrationService:
    """
    خدمة دمج نظام التداول مع منصة الاستثمار
    
    المسؤوليات:
    - تسجيل الصفقات من البوت
    - تحديث حالات الوكلاء
    - تسجيل لقطات NAV
    - جلب إحصائيات التداول
    """
    
    def __init__(self):
        self._initialized = False
        
    async def initialize(self):
        """تهيئة الخدمة"""
        if self._initialized:
            return
        logger.info("🔗 تهيئة خدمة تكامل التداول...")
        self._initialized = True
        logger.info("✅ تم تهيئة خدمة تكامل التداول")
        
    async def record_trade(
        self,
        db: AsyncSession,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        agent_decision: str = None,
        risk_score: float = None,
        confidence: float = None
    ) -> Trade:
        """
        تسجيل صفقة جديدة
        
        Args:
            db: جلسة قاعدة البيانات
            symbol: رمز العملة
            side: اتجاه الصفقة (BUY/SELL)
            entry_price: سعر الدخول
            quantity: الكمية
            agent_decision: قرار الوكيل
            risk_score: درجة المخاطرة
            confidence: درجة الثقة
            
        Returns:
            Trade: الصفقة المسجلة
        """
        trade = Trade(
            symbol=symbol,
            side=side.upper(),
            entry_price=Decimal(str(entry_price)),
            quantity=Decimal(str(quantity)),
            status='open',
            agent_decision=agent_decision,
            risk_score=risk_score,
            confidence=confidence,
            created_at=datetime.utcnow()
        )
        
        db.add(trade)
        await db.commit()
        await db.refresh(trade)
        
        logger.info(f"📈 Trade recorded: {side} {quantity} {symbol} @ {entry_price}")
        return trade
        
    async def close_trade(
        self,
        db: AsyncSession,
        trade_id: int,
        exit_price: float
    ) -> Optional[Trade]:
        """
        إغلاق صفقة
        
        Args:
            db: جلسة قاعدة البيانات
            trade_id: معرف الصفقة
            exit_price: سعر الخروج
            
        Returns:
            Trade: الصفقة المغلقة أو None
        """
        result = await db.execute(
            select(Trade).where(Trade.id == trade_id)
        )
        trade = result.scalar_one_or_none()
        
        if not trade:
            logger.error(f"Trade {trade_id} not found")
            return None
            
        trade.exit_price = Decimal(str(exit_price))
        trade.closed_at = datetime.utcnow()
        trade.status = 'closed'
        
        # حساب الربح/الخسارة
        entry = float(trade.entry_price)
        exit_p = float(exit_price)
        qty = float(trade.quantity)
        
        if trade.side.upper() == 'BUY':
            pnl = (exit_p - entry) * qty
        else:
            pnl = (entry - exit_p) * qty
            
        trade.pnl = Decimal(str(pnl))
        
        if entry > 0:
            trade.pnl_percent = Decimal(str((pnl / (entry * qty)) * 100))
            
        await db.commit()
        await db.refresh(trade)
        
        logger.info(f"📉 Trade {trade_id} closed @ {exit_price}, PnL: {pnl:.2f}")
        return trade
        
    async def update_agent_state(
        self,
        db: AsyncSession,
        agent_name: str,
        agent_type: str,
        status: str,
        signal: str = None,
        signal_strength: float = None,
        analysis: str = None
    ) -> AgentState:
        """
        تحديث حالة وكيل
        
        Args:
            db: جلسة قاعدة البيانات
            agent_name: اسم الوكيل
            agent_type: نوع الوكيل
            status: الحالة
            signal: الإشارة
            signal_strength: قوة الإشارة
            analysis: ملخص التحليل
            
        Returns:
            AgentState: حالة الوكيل
        """
        result = await db.execute(
            select(AgentState).where(AgentState.agent_name == agent_name)
        )
        agent = result.scalar_one_or_none()
        
        if agent:
            agent.status = status
            agent.last_signal = signal
            agent.signal_strength = signal_strength
            agent.analysis_summary = analysis
            agent.last_update = datetime.utcnow()
        else:
            agent = AgentState(
                agent_name=agent_name,
                agent_type=agent_type,
                status=status,
                last_signal=signal,
                signal_strength=signal_strength,
                analysis_summary=analysis
            )
            db.add(agent)
            
        await db.commit()
        await db.refresh(agent)
        
        logger.debug(f"🤖 Agent {agent_name} updated: {signal} ({signal_strength})")
        return agent
        
    async def record_nav_snapshot(
        self,
        db: AsyncSession,
        btc_price: float = 0
    ) -> Optional[NAVHistory]:
        """
        تسجيل لقطة NAV
        
        Args:
            db: جلسة قاعدة البيانات
            btc_price: سعر البيتكوين
            
        Returns:
            NAVHistory: سجل NAV
        """
        try:
            binance_service = get_binance_service()
            
            # جلب قيمة المحفظة
            portfolio_value = await binance_service.get_total_portfolio_value()
            
            # جلب إجمالي الوحدات
            current_nav = await nav_service.get_current_nav(db)
            total_units = await nav_service.get_total_units(db)
            
            # جلب عدد المستثمرين
            from app.models import Investor
            result = await db.execute(
                select(func.count(Investor.id)).where(Investor.status == 'active')
            )
            total_investors = result.scalar() or 0
            
            nav_record = NAVHistory(
                timestamp=datetime.utcnow(),
                total_assets=Decimal(str(portfolio_value)),
                total_units=Decimal(str(total_units)),
                nav_per_unit=Decimal(str(current_nav)),
                btc_price=Decimal(str(btc_price)),
                total_investors=total_investors
            )
            
            db.add(nav_record)
            await db.commit()
            await db.refresh(nav_record)
            
            logger.info(f"📊 NAV snapshot: {current_nav:.4f} (Assets: {portfolio_value:.2f})")
            return nav_record
            
        except Exception as e:
            logger.error(f"Failed to record NAV snapshot: {e}")
            return None
            
    async def get_open_trades(self, db: AsyncSession) -> List[Trade]:
        """جلب الصفقات المفتوحة"""
        result = await db.execute(
            select(Trade).where(Trade.status == 'open')
        )
        return result.scalars().all()
        
    async def get_recent_trades(
        self,
        db: AsyncSession,
        limit: int = 50
    ) -> List[Trade]:
        """جلب أحدث الصفقات"""
        result = await db.execute(
            select(Trade)
            .order_by(Trade.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
        
    async def get_agent_states(self, db: AsyncSession) -> List[AgentState]:
        """جلب حالات الوكلاء"""
        result = await db.execute(select(AgentState))
        return result.scalars().all()
        
    async def get_trading_stats(self, db: AsyncSession) -> Dict[str, Any]:
        """
        إحصائيات التداول
        
        Returns:
            Dict: إحصائيات شاملة
        """
        # إجمالي الصفقات
        result = await db.execute(select(func.count(Trade.id)))
        total_trades = result.scalar() or 0
        
        # الصفقات الرابحة
        result = await db.execute(
            select(func.count(Trade.id))
            .where(Trade.status == 'closed', Trade.pnl > 0)
        )
        winning_trades = result.scalar() or 0
        
        # الصفقات الخاسرة
        result = await db.execute(
            select(func.count(Trade.id))
            .where(Trade.status == 'closed', Trade.pnl < 0)
        )
        losing_trades = result.scalar() or 0
        
        # إجمالي الربح/الخسارة
        result = await db.execute(
            select(func.sum(Trade.pnl))
            .where(Trade.status == 'closed')
        )
        total_pnl = float(result.scalar() or 0)
        
        # نسبة الفوز
        closed_trades = winning_trades + losing_trades
        win_rate = (winning_trades / closed_trades * 100) if closed_trades > 0 else 0
        
        # أفضل صفقة
        result = await db.execute(
            select(Trade)
            .where(Trade.status == 'closed')
            .order_by(Trade.pnl.desc())
            .limit(1)
        )
        best_trade = result.scalar_one_or_none()
        
        # أسوأ صفقة
        result = await db.execute(
            select(Trade)
            .where(Trade.status == 'closed')
            .order_by(Trade.pnl.asc())
            .limit(1)
        )
        worst_trade = result.scalar_one_or_none()
        
        return {
            'total_trades': total_trades,
            'open_trades': total_trades - closed_trades,
            'closed_trades': closed_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'best_trade': {
                'symbol': best_trade.symbol if best_trade else None,
                'pnl': float(best_trade.pnl) if best_trade else 0
            },
            'worst_trade': {
                'symbol': worst_trade.symbol if worst_trade else None,
                'pnl': float(worst_trade.pnl) if worst_trade else 0
            }
        }
        
    async def initialize_sample_agents(self, db: AsyncSession):
        """
        إنشاء حالات وكلاء تجريبية للعرض
        """
        agents_data = [
            {
                'agent_name': 'Technical Analyst',
                'agent_type': 'analyst',
                'status': 'active',
                'last_signal': 'BULLISH',
                'signal_strength': 0.75,
                'analysis_summary': 'RSI oversold, MACD bullish crossover detected'
            },
            {
                'agent_name': 'Sentiment Analyst',
                'agent_type': 'analyst',
                'status': 'active',
                'last_signal': 'NEUTRAL',
                'signal_strength': 0.5,
                'analysis_summary': 'Mixed sentiment on social media'
            },
            {
                'agent_name': 'On-Chain Analyst',
                'agent_type': 'analyst',
                'status': 'active',
                'last_signal': 'BULLISH',
                'signal_strength': 0.65,
                'analysis_summary': 'Whale accumulation detected, exchange outflows increasing'
            },
            {
                'agent_name': 'Macro Analyst',
                'agent_type': 'analyst',
                'status': 'active',
                'last_signal': 'NEUTRAL',
                'signal_strength': 0.55,
                'analysis_summary': 'Fed policy uncertain, DXY stable'
            },
            {
                'agent_name': 'Risk Manager',
                'agent_type': 'risk',
                'status': 'active',
                'last_signal': 'LOW_RISK',
                'signal_strength': 0.8,
                'analysis_summary': 'Portfolio within risk limits, VaR acceptable'
            },
            {
                'agent_name': 'Portfolio Manager',
                'agent_type': 'manager',
                'status': 'active',
                'last_signal': 'REBALANCE',
                'signal_strength': 0.6,
                'analysis_summary': 'Suggesting 60% BTC, 30% ETH, 10% USDC'
            },
            {
                'agent_name': 'DRL Agent (PPO)',
                'agent_type': 'trader',
                'status': 'active',
                'last_signal': 'BUY',
                'signal_strength': 0.7,
                'analysis_summary': 'PPO model suggests long position on BTC'
            },
            {
                'agent_name': 'DRL Agent (A2C)',
                'agent_type': 'trader',
                'status': 'active',
                'last_signal': 'HOLD',
                'signal_strength': 0.45,
                'analysis_summary': 'A2C model suggests holding current positions'
            },
            {
                'agent_name': 'Creative Mind',
                'agent_type': 'creative',
                'status': 'active',
                'last_signal': 'OPPORTUNITY',
                'signal_strength': 0.85,
                'analysis_summary': 'Detected potential breakout pattern forming'
            }
        ]
        
        for data in agents_data:
            result = await db.execute(
                select(AgentState).where(AgentState.agent_name == data['agent_name'])
            )
            existing = result.scalar_one_or_none()
            
            if not existing:
                agent = AgentState(**data)
                db.add(agent)
                
        await db.commit()
        logger.info("✅ Sample agent states initialized")


# Singleton instance
trading_integration = TradingIntegrationService()


def get_trading_integration() -> TradingIntegrationService:
    """الحصول على نسخة الخدمة"""
    return trading_integration
