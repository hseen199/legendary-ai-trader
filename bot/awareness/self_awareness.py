"""
Legendary Trading System - Self Awareness System
نظام التداول الخارق - نظام الوعي الذاتي

نظام لمراقبة أداء الوكيل واكتشاف نقاط الضعف وتقييم الثقة.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging
import statistics
import json


class PerformanceState(Enum):
    """حالات الأداء"""
    EXCELLENT = "excellent"     # ممتاز
    GOOD = "good"               # جيد
    NORMAL = "normal"           # عادي
    POOR = "poor"               # ضعيف
    CRITICAL = "critical"       # حرج


class ConfidenceLevel(Enum):
    """مستويات الثقة"""
    VERY_HIGH = 0.9
    HIGH = 0.75
    MEDIUM = 0.5
    LOW = 0.25
    VERY_LOW = 0.1


@dataclass
class PerformanceMetrics:
    """مقاييس الأداء"""
    timestamp: datetime
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration: float
    total_trades: int
    consecutive_losses: int
    consecutive_wins: int
    daily_pnl: float
    weekly_pnl: float


@dataclass
class WeaknessReport:
    """تقرير نقاط الضعف"""
    id: str
    category: str
    description: str
    severity: float  # 0-1
    detected_at: datetime
    evidence: List[str]
    suggested_fixes: List[str]
    status: str = "active"  # active, addressed, resolved


@dataclass
class ConfidenceAssessment:
    """تقييم الثقة"""
    timestamp: datetime
    overall_confidence: float
    components: Dict[str, float]
    factors: Dict[str, Any]
    recommendation: str


class SelfAwarenessSystem:
    """
    نظام الوعي الذاتي.
    
    يوفر:
    - مراقبة أداء الوكيل لنفسه
    - اكتشاف نقاط الضعف تلقائياً
    - تقييم مستوى الثقة في كل قرار
    - معرفة متى يجب التوقف عن التداول
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger("SelfAwarenessSystem")
        self.config = config or {}
        
        # تاريخ الأداء
        self.performance_history: deque = deque(maxlen=1000)
        
        # نقاط الضعف المكتشفة
        self.weaknesses: Dict[str, WeaknessReport] = {}
        
        # تقييمات الثقة
        self.confidence_history: deque = deque(maxlen=500)
        
        # حالة التداول
        self.trading_state = {
            "is_active": True,
            "pause_reason": None,
            "pause_until": None,
            "performance_state": PerformanceState.NORMAL
        }
        
        # عتبات الأداء
        self.thresholds = {
            "min_win_rate": 0.4,
            "min_profit_factor": 1.0,
            "max_drawdown": 0.15,
            "max_consecutive_losses": 5,
            "min_sharpe_ratio": 0.5,
            "confidence_threshold": 0.3
        }
        
        # إحصائيات
        self.stats = {
            "total_assessments": 0,
            "weaknesses_detected": 0,
            "trading_pauses": 0,
            "confidence_drops": 0
        }
    
    async def monitor_performance(self, 
                                 trade_result: Dict[str, Any]) -> PerformanceMetrics:
        """
        مراقبة الأداء بعد كل صفقة.
        
        Args:
            trade_result: نتيجة الصفقة
            
        Returns:
            مقاييس الأداء المحدثة
        """
        # حساب المقاييس
        metrics = await self._calculate_metrics(trade_result)
        
        # حفظ في التاريخ
        self.performance_history.append(metrics)
        
        # تحديث حالة الأداء
        self.trading_state["performance_state"] = self._assess_performance_state(metrics)
        
        # فحص نقاط الضعف
        await self._check_for_weaknesses(metrics)
        
        # تحديد إذا يجب إيقاف التداول
        should_pause = await self._should_pause_trading(metrics)
        if should_pause:
            await self._pause_trading(should_pause)
        
        self.logger.debug(f"أداء: {self.trading_state['performance_state'].value}")
        
        return metrics
    
    async def _calculate_metrics(self, 
                                trade_result: Dict[str, Any]) -> PerformanceMetrics:
        """حساب مقاييس الأداء."""
        # جمع البيانات من التاريخ
        recent_trades = list(self.performance_history)[-100:] if self.performance_history else []
        
        # حساب معدل الربح
        if recent_trades:
            wins = sum(1 for t in recent_trades if getattr(t, 'daily_pnl', 0) > 0)
            win_rate = wins / len(recent_trades)
        else:
            win_rate = 0.5
        
        # حساب عامل الربح
        gross_profit = sum(max(0, getattr(t, 'daily_pnl', 0)) for t in recent_trades)
        gross_loss = abs(sum(min(0, getattr(t, 'daily_pnl', 0)) for t in recent_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 2.0
        
        # حساب السحب الأقصى
        if recent_trades:
            pnls = [getattr(t, 'daily_pnl', 0) for t in recent_trades]
            cumulative = []
            total = 0
            for pnl in pnls:
                total += pnl
                cumulative.append(total)
            
            peak = cumulative[0]
            max_drawdown = 0
            for value in cumulative:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0
        
        # حساب الخسائر/الأرباح المتتالية
        consecutive_losses = 0
        consecutive_wins = 0
        
        for t in reversed(recent_trades):
            pnl = getattr(t, 'daily_pnl', 0)
            if pnl < 0:
                if consecutive_wins == 0:
                    consecutive_losses += 1
                else:
                    break
            elif pnl > 0:
                if consecutive_losses == 0:
                    consecutive_wins += 1
                else:
                    break
        
        # حساب Sharpe Ratio
        if recent_trades and len(recent_trades) > 1:
            returns = [getattr(t, 'daily_pnl', 0) for t in recent_trades]
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 1
            sharpe_ratio = (avg_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        return PerformanceMetrics(
            timestamp=datetime.utcnow(),
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_trade_duration=trade_result.get("duration", 0),
            total_trades=len(recent_trades) + 1,
            consecutive_losses=consecutive_losses,
            consecutive_wins=consecutive_wins,
            daily_pnl=trade_result.get("pnl", 0),
            weekly_pnl=sum(getattr(t, 'daily_pnl', 0) for t in recent_trades[-7:])
        )
    
    def _assess_performance_state(self, metrics: PerformanceMetrics) -> PerformanceState:
        """تقييم حالة الأداء."""
        score = 0
        
        # معدل الربح
        if metrics.win_rate >= 0.6:
            score += 2
        elif metrics.win_rate >= 0.5:
            score += 1
        elif metrics.win_rate < 0.4:
            score -= 2
        
        # عامل الربح
        if metrics.profit_factor >= 2:
            score += 2
        elif metrics.profit_factor >= 1.5:
            score += 1
        elif metrics.profit_factor < 1:
            score -= 2
        
        # السحب
        if metrics.max_drawdown < 0.05:
            score += 2
        elif metrics.max_drawdown < 0.1:
            score += 1
        elif metrics.max_drawdown > 0.15:
            score -= 2
        
        # الخسائر المتتالية
        if metrics.consecutive_losses >= 5:
            score -= 3
        elif metrics.consecutive_losses >= 3:
            score -= 1
        
        # Sharpe Ratio
        if metrics.sharpe_ratio >= 2:
            score += 2
        elif metrics.sharpe_ratio >= 1:
            score += 1
        elif metrics.sharpe_ratio < 0:
            score -= 2
        
        # تحديد الحالة
        if score >= 6:
            return PerformanceState.EXCELLENT
        elif score >= 3:
            return PerformanceState.GOOD
        elif score >= 0:
            return PerformanceState.NORMAL
        elif score >= -3:
            return PerformanceState.POOR
        else:
            return PerformanceState.CRITICAL
    
    async def _check_for_weaknesses(self, metrics: PerformanceMetrics):
        """فحص نقاط الضعف."""
        weaknesses_found = []
        
        # ضعف معدل الربح
        if metrics.win_rate < self.thresholds["min_win_rate"]:
            weaknesses_found.append({
                "category": "win_rate",
                "description": f"معدل الربح منخفض ({metrics.win_rate:.1%})",
                "severity": 1 - metrics.win_rate,
                "suggested_fixes": [
                    "مراجعة شروط الدخول",
                    "تحسين توقيت الصفقات",
                    "تقليل عدد الصفقات"
                ]
            })
        
        # سحب مرتفع
        if metrics.max_drawdown > self.thresholds["max_drawdown"]:
            weaknesses_found.append({
                "category": "drawdown",
                "description": f"السحب مرتفع ({metrics.max_drawdown:.1%})",
                "severity": min(1, metrics.max_drawdown / 0.3),
                "suggested_fixes": [
                    "تقليل حجم الصفقات",
                    "تحسين وقف الخسارة",
                    "تنويع المحفظة"
                ]
            })
        
        # خسائر متتالية
        if metrics.consecutive_losses >= self.thresholds["max_consecutive_losses"]:
            weaknesses_found.append({
                "category": "consecutive_losses",
                "description": f"خسائر متتالية ({metrics.consecutive_losses})",
                "severity": min(1, metrics.consecutive_losses / 10),
                "suggested_fixes": [
                    "إيقاف التداول مؤقتاً",
                    "مراجعة الاستراتيجية",
                    "تقليل المخاطرة"
                ]
            })
        
        # عامل ربح ضعيف
        if metrics.profit_factor < self.thresholds["min_profit_factor"]:
            weaknesses_found.append({
                "category": "profit_factor",
                "description": f"عامل الربح ضعيف ({metrics.profit_factor:.2f})",
                "severity": 1 - min(1, metrics.profit_factor),
                "suggested_fixes": [
                    "تحسين نسبة المخاطرة/العائد",
                    "تحسين شروط الخروج",
                    "زيادة أهداف الربح"
                ]
            })
        
        # Sharpe Ratio منخفض
        if metrics.sharpe_ratio < self.thresholds["min_sharpe_ratio"]:
            weaknesses_found.append({
                "category": "sharpe_ratio",
                "description": f"Sharpe Ratio منخفض ({metrics.sharpe_ratio:.2f})",
                "severity": max(0, 1 - metrics.sharpe_ratio),
                "suggested_fixes": [
                    "تقليل التقلب",
                    "تحسين الاتساق",
                    "تنويع الاستراتيجيات"
                ]
            })
        
        # تسجيل نقاط الضعف
        for weakness in weaknesses_found:
            weakness_id = f"{weakness['category']}_{datetime.utcnow().timestamp()}"
            self.weaknesses[weakness_id] = WeaknessReport(
                id=weakness_id,
                category=weakness["category"],
                description=weakness["description"],
                severity=weakness["severity"],
                detected_at=datetime.utcnow(),
                evidence=[f"metrics: {metrics}"],
                suggested_fixes=weakness["suggested_fixes"]
            )
            self.stats["weaknesses_detected"] += 1
            
            self.logger.warning(f"نقطة ضعف: {weakness['description']}")
    
    async def _should_pause_trading(self, metrics: PerformanceMetrics) -> Optional[str]:
        """تحديد إذا يجب إيقاف التداول."""
        # حالة حرجة
        if self.trading_state["performance_state"] == PerformanceState.CRITICAL:
            return "أداء حرج - يجب المراجعة"
        
        # خسائر متتالية كثيرة
        if metrics.consecutive_losses >= self.thresholds["max_consecutive_losses"]:
            return f"خسائر متتالية ({metrics.consecutive_losses})"
        
        # سحب مرتفع جداً
        if metrics.max_drawdown > self.thresholds["max_drawdown"] * 1.5:
            return f"سحب مرتفع ({metrics.max_drawdown:.1%})"
        
        # معدل ربح منخفض جداً
        if metrics.win_rate < 0.3 and metrics.total_trades > 20:
            return f"معدل ربح منخفض جداً ({metrics.win_rate:.1%})"
        
        return None
    
    async def _pause_trading(self, reason: str):
        """إيقاف التداول مؤقتاً."""
        self.trading_state["is_active"] = False
        self.trading_state["pause_reason"] = reason
        self.trading_state["pause_until"] = datetime.utcnow() + timedelta(hours=4)
        
        self.stats["trading_pauses"] += 1
        
        self.logger.warning(f"تم إيقاف التداول: {reason}")
    
    async def assess_confidence(self,
                               decision: str,
                               context: Dict[str, Any]) -> ConfidenceAssessment:
        """
        تقييم مستوى الثقة في قرار.
        
        Args:
            decision: القرار
            context: السياق
            
        Returns:
            تقييم الثقة
        """
        components = {}
        
        # ثقة الأداء التاريخي
        if self.performance_history:
            recent = list(self.performance_history)[-20:]
            win_rate = sum(1 for m in recent if m.daily_pnl > 0) / len(recent)
            components["historical_performance"] = win_rate
        else:
            components["historical_performance"] = 0.5
        
        # ثقة حالة السوق
        market_confidence = self._assess_market_confidence(context)
        components["market_conditions"] = market_confidence
        
        # ثقة الإشارة
        signal_strength = context.get("signal_strength", 0.5)
        components["signal_strength"] = signal_strength
        
        # ثقة التوافق
        consensus = context.get("consensus", 0.5)
        components["consensus"] = consensus
        
        # ثقة إدارة المخاطر
        risk_ok = context.get("risk_approved", True)
        components["risk_management"] = 1.0 if risk_ok else 0.3
        
        # حساب الثقة الإجمالية
        weights = {
            "historical_performance": 0.25,
            "market_conditions": 0.2,
            "signal_strength": 0.25,
            "consensus": 0.15,
            "risk_management": 0.15
        }
        
        overall = sum(
            components[k] * weights[k] 
            for k in components
        )
        
        # تحديد التوصية
        if overall >= 0.7:
            recommendation = "تنفيذ بثقة عالية"
        elif overall >= 0.5:
            recommendation = "تنفيذ بحذر"
        elif overall >= 0.3:
            recommendation = "تقليل الحجم أو الانتظار"
        else:
            recommendation = "عدم التنفيذ"
        
        assessment = ConfidenceAssessment(
            timestamp=datetime.utcnow(),
            overall_confidence=overall,
            components=components,
            factors=context,
            recommendation=recommendation
        )
        
        self.confidence_history.append(assessment)
        self.stats["total_assessments"] += 1
        
        if overall < self.thresholds["confidence_threshold"]:
            self.stats["confidence_drops"] += 1
        
        return assessment
    
    def _assess_market_confidence(self, context: Dict[str, Any]) -> float:
        """تقييم ثقة ظروف السوق."""
        confidence = 0.5
        
        # تقلب السوق
        volatility = context.get("volatility", 0.02)
        if volatility < 0.01:
            confidence += 0.1
        elif volatility > 0.05:
            confidence -= 0.2
        
        # وضوح الاتجاه
        trend_strength = abs(context.get("trend_strength", 0))
        confidence += trend_strength * 0.2
        
        # حجم التداول
        volume_ratio = context.get("volume_ratio", 1)
        if volume_ratio > 1.5:
            confidence += 0.1
        elif volume_ratio < 0.5:
            confidence -= 0.1
        
        return max(0, min(1, confidence))
    
    async def should_trade(self) -> Tuple[bool, str]:
        """
        تحديد إذا يجب التداول الآن.
        
        Returns:
            (يجب التداول؟, السبب)
        """
        # فحص إذا التداول متوقف
        if not self.trading_state["is_active"]:
            if self.trading_state["pause_until"]:
                if datetime.utcnow() < self.trading_state["pause_until"]:
                    return False, f"التداول متوقف: {self.trading_state['pause_reason']}"
                else:
                    # انتهت فترة الإيقاف
                    self.trading_state["is_active"] = True
                    self.trading_state["pause_reason"] = None
                    self.trading_state["pause_until"] = None
        
        # فحص حالة الأداء
        if self.trading_state["performance_state"] == PerformanceState.CRITICAL:
            return False, "الأداء في حالة حرجة"
        
        # فحص نقاط الضعف النشطة الخطيرة
        severe_weaknesses = [
            w for w in self.weaknesses.values()
            if w.status == "active" and w.severity > 0.7
        ]
        if len(severe_weaknesses) >= 3:
            return False, f"نقاط ضعف خطيرة متعددة ({len(severe_weaknesses)})"
        
        return True, "جاهز للتداول"
    
    async def get_self_report(self) -> Dict[str, Any]:
        """
        الحصول على تقرير ذاتي شامل.
        
        Returns:
            التقرير الذاتي
        """
        # آخر مقاييس
        latest_metrics = self.performance_history[-1] if self.performance_history else None
        
        # نقاط الضعف النشطة
        active_weaknesses = [
            w for w in self.weaknesses.values()
            if w.status == "active"
        ]
        
        # متوسط الثقة الأخيرة
        recent_confidence = list(self.confidence_history)[-20:]
        avg_confidence = (
            statistics.mean(c.overall_confidence for c in recent_confidence)
            if recent_confidence else 0.5
        )
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "trading_state": {
                "is_active": self.trading_state["is_active"],
                "performance_state": self.trading_state["performance_state"].value,
                "pause_reason": self.trading_state["pause_reason"]
            },
            "performance": {
                "win_rate": latest_metrics.win_rate if latest_metrics else None,
                "profit_factor": latest_metrics.profit_factor if latest_metrics else None,
                "sharpe_ratio": latest_metrics.sharpe_ratio if latest_metrics else None,
                "max_drawdown": latest_metrics.max_drawdown if latest_metrics else None,
                "consecutive_losses": latest_metrics.consecutive_losses if latest_metrics else 0
            },
            "weaknesses": {
                "total": len(self.weaknesses),
                "active": len(active_weaknesses),
                "severe": len([w for w in active_weaknesses if w.severity > 0.7]),
                "details": [
                    {
                        "category": w.category,
                        "description": w.description,
                        "severity": w.severity
                    }
                    for w in active_weaknesses[:5]
                ]
            },
            "confidence": {
                "average": avg_confidence,
                "trend": self._calculate_confidence_trend()
            },
            "recommendations": self._generate_recommendations(active_weaknesses),
            "stats": self.stats
        }
        
        return report
    
    def _calculate_confidence_trend(self) -> str:
        """حساب اتجاه الثقة."""
        if len(self.confidence_history) < 10:
            return "غير كافٍ"
        
        recent = list(self.confidence_history)[-10:]
        older = list(self.confidence_history)[-20:-10]
        
        if not older:
            return "غير كافٍ"
        
        recent_avg = statistics.mean(c.overall_confidence for c in recent)
        older_avg = statistics.mean(c.overall_confidence for c in older)
        
        if recent_avg > older_avg * 1.1:
            return "تحسن"
        elif recent_avg < older_avg * 0.9:
            return "تراجع"
        else:
            return "مستقر"
    
    def _generate_recommendations(self, 
                                 weaknesses: List[WeaknessReport]) -> List[str]:
        """توليد توصيات."""
        recommendations = []
        
        # توصيات بناءً على نقاط الضعف
        for weakness in weaknesses[:3]:
            recommendations.extend(weakness.suggested_fixes[:2])
        
        # توصيات عامة
        if self.trading_state["performance_state"] in [PerformanceState.POOR, PerformanceState.CRITICAL]:
            recommendations.append("مراجعة شاملة للاستراتيجية")
        
        if len(weaknesses) > 5:
            recommendations.append("تقليل نشاط التداول")
        
        return list(set(recommendations))[:5]
    
    def mark_weakness_addressed(self, weakness_id: str):
        """تحديد نقطة ضعف كمعالجة."""
        if weakness_id in self.weaknesses:
            self.weaknesses[weakness_id].status = "addressed"
    
    def mark_weakness_resolved(self, weakness_id: str):
        """تحديد نقطة ضعف كمحلولة."""
        if weakness_id in self.weaknesses:
            self.weaknesses[weakness_id].status = "resolved"
    
    def resume_trading(self):
        """استئناف التداول."""
        self.trading_state["is_active"] = True
        self.trading_state["pause_reason"] = None
        self.trading_state["pause_until"] = None
        self.logger.info("تم استئناف التداول")
