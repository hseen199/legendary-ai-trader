"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Evolution Layer
طبقة التطور - التعلم المستمر والتحسين الذاتي
═══════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import json
import os
from loguru import logger


class LearningType(Enum):
    """أنواع التعلم"""
    TRADE_OUTCOME = "نتيجة صفقة"
    STRATEGY_PERFORMANCE = "أداء استراتيجية"
    MARKET_PATTERN = "نمط سوق"
    ERROR_CORRECTION = "تصحيح خطأ"
    PARAMETER_TUNING = "ضبط معاملات"


@dataclass
class TradeLesson:
    """درس من صفقة"""
    symbol: str
    action: str
    entry_price: float
    exit_price: float
    pnl_percent: float
    holding_time: int  # بالدقائق
    market_regime: str
    features_at_entry: Dict[str, float]
    features_at_exit: Dict[str, float]
    decision_confidence: float
    actual_outcome: str  # WIN, LOSS, BREAKEVEN
    lesson: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StrategyLesson:
    """درس من استراتيجية"""
    strategy_name: str
    market_regime: str
    win_rate: float
    avg_pnl: float
    sample_size: int
    lesson: str
    adjustment: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvolutionState:
    """حالة التطور"""
    timestamp: datetime
    total_lessons: int
    win_rate: float
    avg_pnl: float
    best_performing_regime: str
    worst_performing_regime: str
    recent_improvements: List[str]
    pending_adjustments: List[str]


class EvolutionLayer:
    """
    طبقة التطور
    
    مسؤولة عن:
    - التعلم من الصفقات
    - تحسين الاستراتيجيات
    - ضبط المعاملات
    - التطور المستمر
    """
    
    def __init__(self, config: Dict[str, Any] = None, data_dir: str = None):
        """
        تهيئة طبقة التطور
        
        Args:
            config: إعدادات الطبقة
            data_dir: مجلد البيانات
        """
        self.config = config or {}
        self.data_dir = data_dir or '/tmp/legendary_agent/evolution'
        
        # إنشاء المجلد
        os.makedirs(self.data_dir, exist_ok=True)
        
        # دروس الصفقات
        self.trade_lessons: List[TradeLesson] = []
        self.max_lessons = 10000
        
        # دروس الاستراتيجيات
        self.strategy_lessons: List[StrategyLesson] = []
        
        # إحصائيات الأداء
        self.performance_stats = {
            'by_regime': defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0}),
            'by_symbol': defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0}),
            'by_hour': defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0}),
            'by_confidence': defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})
        }
        
        # تعديلات المعاملات
        self.parameter_adjustments = {
            'stop_loss': [],
            'take_profit': [],
            'position_size': [],
            'confidence_threshold': []
        }
        
        # الأنماط المكتشفة
        self.discovered_patterns: List[Dict] = []
        
        # تحميل البيانات المحفوظة
        self._load_state()
        
        logger.info("🧬 EvolutionLayer initialized")
    
    # ═══════════════════════════════════════════════════════════════
    # LEARNING FROM TRADES
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
    ) -> TradeLesson:
        """
        التعلم من صفقة
        
        Args:
            symbol: رمز العملة
            action: الإجراء (BUY/SELL)
            entry_price: سعر الدخول
            exit_price: سعر الخروج
            holding_time: وقت الاحتفاظ
            market_regime: نظام السوق
            features_at_entry: الميزات عند الدخول
            features_at_exit: الميزات عند الخروج
            decision_confidence: ثقة القرار
            
        Returns:
            الدرس المستفاد
        """
        # حساب الربح/الخسارة
        if action == 'BUY':
            pnl_percent = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_percent = (entry_price - exit_price) / entry_price * 100
        
        # تحديد النتيجة
        if pnl_percent > 0.5:
            outcome = 'WIN'
        elif pnl_percent < -0.5:
            outcome = 'LOSS'
        else:
            outcome = 'BREAKEVEN'
        
        # استخراج الدرس
        lesson_text = self._extract_lesson(
            outcome, pnl_percent, market_regime,
            features_at_entry, decision_confidence
        )
        
        lesson = TradeLesson(
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_percent=pnl_percent,
            holding_time=holding_time,
            market_regime=market_regime,
            features_at_entry=features_at_entry,
            features_at_exit=features_at_exit,
            decision_confidence=decision_confidence,
            actual_outcome=outcome,
            lesson=lesson_text
        )
        
        # حفظ الدرس
        self.trade_lessons.append(lesson)
        if len(self.trade_lessons) > self.max_lessons:
            self.trade_lessons = self.trade_lessons[-self.max_lessons:]
        
        # تحديث الإحصائيات
        self._update_stats(lesson)
        
        # البحث عن أنماط
        self._discover_patterns()
        
        # حفظ الحالة
        self._save_state()
        
        logger.info(f"📚 Learned from trade: {symbol} {outcome} {pnl_percent:.2f}%")
        
        return lesson
    
    def _extract_lesson(
        self,
        outcome: str,
        pnl: float,
        regime: str,
        features: Dict,
        confidence: float
    ) -> str:
        """استخراج الدرس"""
        lessons = []
        
        if outcome == 'WIN':
            if confidence > 0.7:
                lessons.append("الثقة العالية كانت مبررة")
            if regime in ['BULL', 'STRONG_BULL']:
                lessons.append("التداول مع الاتجاه ناجح")
        
        elif outcome == 'LOSS':
            if confidence > 0.7:
                lessons.append("⚠️ الثقة العالية كانت خاطئة - مراجعة المعايير")
            if regime == 'RANGING':
                lessons.append("⚠️ التذبذب خطر - تجنب أو تقليل الحجم")
            
            rsi = features.get('rsi_14', 50)
            if rsi > 70:
                lessons.append("⚠️ الدخول في منطقة تشبع شرائي")
            elif rsi < 30:
                lessons.append("⚠️ الدخول في منطقة تشبع بيعي")
        
        return " | ".join(lessons) if lessons else "صفقة عادية"
    
    def _update_stats(self, lesson: TradeLesson) -> None:
        """تحديث الإحصائيات"""
        # حسب النظام
        regime_stats = self.performance_stats['by_regime'][lesson.market_regime]
        if lesson.actual_outcome == 'WIN':
            regime_stats['wins'] += 1
        elif lesson.actual_outcome == 'LOSS':
            regime_stats['losses'] += 1
        regime_stats['total_pnl'] += lesson.pnl_percent
        
        # حسب العملة
        symbol_stats = self.performance_stats['by_symbol'][lesson.symbol]
        if lesson.actual_outcome == 'WIN':
            symbol_stats['wins'] += 1
        elif lesson.actual_outcome == 'LOSS':
            symbol_stats['losses'] += 1
        symbol_stats['total_pnl'] += lesson.pnl_percent
        
        # حسب الساعة
        hour = lesson.timestamp.hour
        hour_stats = self.performance_stats['by_hour'][hour]
        if lesson.actual_outcome == 'WIN':
            hour_stats['wins'] += 1
        elif lesson.actual_outcome == 'LOSS':
            hour_stats['losses'] += 1
        hour_stats['total_pnl'] += lesson.pnl_percent
        
        # حسب الثقة
        confidence_bucket = int(lesson.decision_confidence * 10) / 10
        conf_stats = self.performance_stats['by_confidence'][confidence_bucket]
        if lesson.actual_outcome == 'WIN':
            conf_stats['wins'] += 1
        elif lesson.actual_outcome == 'LOSS':
            conf_stats['losses'] += 1
        conf_stats['total_pnl'] += lesson.pnl_percent
    
    # ═══════════════════════════════════════════════════════════════
    # PATTERN DISCOVERY
    # ═══════════════════════════════════════════════════════════════
    
    def _discover_patterns(self) -> None:
        """اكتشاف الأنماط"""
        if len(self.trade_lessons) < 50:
            return
        
        recent = self.trade_lessons[-100:]
        
        # نمط: أفضل نظام
        best_regime = self._find_best_regime()
        if best_regime:
            self._add_pattern({
                'type': 'best_regime',
                'regime': best_regime['regime'],
                'win_rate': best_regime['win_rate'],
                'recommendation': f"التركيز على التداول في {best_regime['regime']}"
            })
        
        # نمط: أسوأ نظام
        worst_regime = self._find_worst_regime()
        if worst_regime:
            self._add_pattern({
                'type': 'worst_regime',
                'regime': worst_regime['regime'],
                'win_rate': worst_regime['win_rate'],
                'recommendation': f"تجنب التداول في {worst_regime['regime']}"
            })
        
        # نمط: أفضل ساعة
        best_hour = self._find_best_hour()
        if best_hour:
            self._add_pattern({
                'type': 'best_hour',
                'hour': best_hour['hour'],
                'win_rate': best_hour['win_rate'],
                'recommendation': f"التداول في الساعة {best_hour['hour']}"
            })
    
    def _find_best_regime(self) -> Optional[Dict]:
        """إيجاد أفضل نظام"""
        best = None
        best_rate = 0
        
        for regime, stats in self.performance_stats['by_regime'].items():
            total = stats['wins'] + stats['losses']
            if total >= 10:
                rate = stats['wins'] / total
                if rate > best_rate:
                    best_rate = rate
                    best = {'regime': regime, 'win_rate': rate}
        
        return best
    
    def _find_worst_regime(self) -> Optional[Dict]:
        """إيجاد أسوأ نظام"""
        worst = None
        worst_rate = 1
        
        for regime, stats in self.performance_stats['by_regime'].items():
            total = stats['wins'] + stats['losses']
            if total >= 10:
                rate = stats['wins'] / total
                if rate < worst_rate:
                    worst_rate = rate
                    worst = {'regime': regime, 'win_rate': rate}
        
        return worst
    
    def _find_best_hour(self) -> Optional[Dict]:
        """إيجاد أفضل ساعة"""
        best = None
        best_rate = 0
        
        for hour, stats in self.performance_stats['by_hour'].items():
            total = stats['wins'] + stats['losses']
            if total >= 5:
                rate = stats['wins'] / total
                if rate > best_rate:
                    best_rate = rate
                    best = {'hour': hour, 'win_rate': rate}
        
        return best
    
    def _add_pattern(self, pattern: Dict) -> None:
        """إضافة نمط"""
        pattern['discovered_at'] = datetime.now().isoformat()
        
        # تجنب التكرار
        for existing in self.discovered_patterns:
            if existing.get('type') == pattern.get('type'):
                existing.update(pattern)
                return
        
        self.discovered_patterns.append(pattern)
    
    # ═══════════════════════════════════════════════════════════════
    # PARAMETER OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════
    
    def suggest_parameter_adjustments(self) -> Dict[str, Any]:
        """اقتراح تعديلات المعاملات"""
        suggestions = {}
        
        if len(self.trade_lessons) < 50:
            return suggestions
        
        recent = self.trade_lessons[-200:]
        
        # تحليل وقف الخسارة
        sl_analysis = self._analyze_stop_loss(recent)
        if sl_analysis:
            suggestions['stop_loss'] = sl_analysis
        
        # تحليل جني الأرباح
        tp_analysis = self._analyze_take_profit(recent)
        if tp_analysis:
            suggestions['take_profit'] = tp_analysis
        
        # تحليل حجم المركز
        size_analysis = self._analyze_position_size(recent)
        if size_analysis:
            suggestions['position_size'] = size_analysis
        
        # تحليل عتبة الثقة
        conf_analysis = self._analyze_confidence_threshold(recent)
        if conf_analysis:
            suggestions['confidence_threshold'] = conf_analysis
        
        return suggestions
    
    def _analyze_stop_loss(self, lessons: List[TradeLesson]) -> Optional[Dict]:
        """تحليل وقف الخسارة"""
        losses = [l for l in lessons if l.actual_outcome == 'LOSS']
        
        if len(losses) < 10:
            return None
        
        avg_loss = np.mean([abs(l.pnl_percent) for l in losses])
        
        if avg_loss > 2.5:
            return {
                'current': 2.0,
                'suggested': 1.5,
                'reason': f"متوسط الخسارة ({avg_loss:.1f}%) أعلى من المتوقع"
            }
        elif avg_loss < 1.5:
            return {
                'current': 2.0,
                'suggested': 2.5,
                'reason': "وقف الخسارة ضيق جداً - يمكن توسيعه"
            }
        
        return None
    
    def _analyze_take_profit(self, lessons: List[TradeLesson]) -> Optional[Dict]:
        """تحليل جني الأرباح"""
        wins = [l for l in lessons if l.actual_outcome == 'WIN']
        
        if len(wins) < 10:
            return None
        
        avg_win = np.mean([l.pnl_percent for l in wins])
        
        if avg_win < 1.5:
            return {
                'current': [1.5, 3.5, 6.0],
                'suggested': [1.0, 2.5, 4.0],
                'reason': "جني الأرباح مبكراً لتأمين المكاسب"
            }
        
        return None
    
    def _analyze_position_size(self, lessons: List[TradeLesson]) -> Optional[Dict]:
        """تحليل حجم المركز"""
        # حساب نسبة الفوز حسب الثقة
        high_conf = [l for l in lessons if l.decision_confidence > 0.7]
        low_conf = [l for l in lessons if l.decision_confidence < 0.5]
        
        if len(high_conf) >= 10 and len(low_conf) >= 10:
            high_win_rate = len([l for l in high_conf if l.actual_outcome == 'WIN']) / len(high_conf)
            low_win_rate = len([l for l in low_conf if l.actual_outcome == 'WIN']) / len(low_conf)
            
            if high_win_rate > low_win_rate + 0.15:
                return {
                    'suggestion': 'زيادة الحجم في الصفقات عالية الثقة',
                    'high_conf_win_rate': high_win_rate,
                    'low_conf_win_rate': low_win_rate
                }
        
        return None
    
    def _analyze_confidence_threshold(self, lessons: List[TradeLesson]) -> Optional[Dict]:
        """تحليل عتبة الثقة"""
        by_confidence = defaultdict(list)
        
        for lesson in lessons:
            bucket = round(lesson.decision_confidence, 1)
            by_confidence[bucket].append(lesson)
        
        # إيجاد أفضل عتبة
        best_threshold = 0.5
        best_performance = 0
        
        for threshold in np.arange(0.4, 0.8, 0.1):
            above = [l for l in lessons if l.decision_confidence >= threshold]
            if len(above) >= 20:
                wins = len([l for l in above if l.actual_outcome == 'WIN'])
                performance = wins / len(above)
                if performance > best_performance:
                    best_performance = performance
                    best_threshold = threshold
        
        if best_threshold != 0.5:
            return {
                'current': 0.5,
                'suggested': best_threshold,
                'expected_win_rate': best_performance,
                'reason': f"تحسين نسبة الفوز إلى {best_performance:.1%}"
            }
        
        return None
    
    # ═══════════════════════════════════════════════════════════════
    # EVOLUTION STATE
    # ═══════════════════════════════════════════════════════════════
    
    def get_evolution_state(self) -> EvolutionState:
        """الحصول على حالة التطور"""
        # حساب الإحصائيات
        total_trades = len(self.trade_lessons)
        wins = len([l for l in self.trade_lessons if l.actual_outcome == 'WIN'])
        win_rate = wins / total_trades if total_trades > 0 else 0
        avg_pnl = np.mean([l.pnl_percent for l in self.trade_lessons]) if self.trade_lessons else 0
        
        # أفضل وأسوأ نظام
        best_regime = self._find_best_regime()
        worst_regime = self._find_worst_regime()
        
        # التحسينات الأخيرة
        recent_improvements = []
        for pattern in self.discovered_patterns[-5:]:
            recent_improvements.append(pattern.get('recommendation', ''))
        
        # التعديلات المعلقة
        pending = self.suggest_parameter_adjustments()
        pending_adjustments = [f"{k}: {v.get('reason', '')}" for k, v in pending.items()]
        
        return EvolutionState(
            timestamp=datetime.now(),
            total_lessons=total_trades,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            best_performing_regime=best_regime['regime'] if best_regime else 'N/A',
            worst_performing_regime=worst_regime['regime'] if worst_regime else 'N/A',
            recent_improvements=recent_improvements,
            pending_adjustments=pending_adjustments
        )
    
    # ═══════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ═══════════════════════════════════════════════════════════════
    
    def _save_state(self) -> None:
        """حفظ الحالة"""
        try:
            state = {
                'trade_lessons': [
                    {
                        'symbol': l.symbol,
                        'action': l.action,
                        'entry_price': l.entry_price,
                        'exit_price': l.exit_price,
                        'pnl_percent': l.pnl_percent,
                        'holding_time': l.holding_time,
                        'market_regime': l.market_regime,
                        'decision_confidence': l.decision_confidence,
                        'actual_outcome': l.actual_outcome,
                        'lesson': l.lesson,
                        'timestamp': l.timestamp.isoformat()
                    }
                    for l in self.trade_lessons[-1000:]  # آخر 1000 فقط
                ],
                'performance_stats': {
                    k: dict(v) for k, v in self.performance_stats.items()
                },
                'discovered_patterns': self.discovered_patterns
            }
            
            path = os.path.join(self.data_dir, 'evolution_state.json')
            with open(path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save evolution state: {e}")
    
    def _load_state(self) -> None:
        """تحميل الحالة"""
        try:
            path = os.path.join(self.data_dir, 'evolution_state.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    state = json.load(f)
                
                # تحميل الأنماط
                self.discovered_patterns = state.get('discovered_patterns', [])
                
                # تحميل الإحصائيات
                for key, value in state.get('performance_stats', {}).items():
                    if key in self.performance_stats:
                        for k, v in value.items():
                            self.performance_stats[key][k] = v
                
                logger.info(f"📂 Loaded evolution state with {len(self.discovered_patterns)} patterns")
                
        except Exception as e:
            logger.warning(f"Could not load evolution state: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # STATUS
    # ═══════════════════════════════════════════════════════════════
    
    def get_status(self) -> Dict[str, Any]:
        """الحصول على حالة الطبقة"""
        state = self.get_evolution_state()
        return {
            'total_lessons': state.total_lessons,
            'win_rate': f"{state.win_rate:.1%}",
            'avg_pnl': f"{state.avg_pnl:.2f}%",
            'best_regime': state.best_performing_regime,
            'worst_regime': state.worst_performing_regime,
            'patterns_discovered': len(self.discovered_patterns),
            'pending_adjustments': len(state.pending_adjustments)
        }


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار طبقة التطور
    evolution = EvolutionLayer()
    
    # محاكاة صفقات
    import random
    
    regimes = ['BULL', 'BEAR', 'RANGING', 'STRONG_BULL']
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    for i in range(100):
        regime = random.choice(regimes)
        symbol = random.choice(symbols)
        
        # صفقات في BULL أفضل
        if regime in ['BULL', 'STRONG_BULL']:
            pnl_base = 1.5
        else:
            pnl_base = -0.5
        
        pnl = pnl_base + random.uniform(-2, 2)
        
        lesson = evolution.learn_from_trade(
            symbol=symbol,
            action='BUY',
            entry_price=50000,
            exit_price=50000 * (1 + pnl/100),
            holding_time=random.randint(30, 240),
            market_regime=regime,
            features_at_entry={'rsi_14': random.uniform(30, 70)},
            features_at_exit={'rsi_14': random.uniform(30, 70)},
            decision_confidence=random.uniform(0.4, 0.9)
        )
    
    # عرض حالة التطور
    state = evolution.get_evolution_state()
    
    print("🧬 Evolution State:")
    print(f"Total Lessons: {state.total_lessons}")
    print(f"Win Rate: {state.win_rate:.1%}")
    print(f"Avg PnL: {state.avg_pnl:.2f}%")
    print(f"Best Regime: {state.best_performing_regime}")
    print(f"Worst Regime: {state.worst_performing_regime}")
    
    print("\n📈 Recent Improvements:")
    for imp in state.recent_improvements:
        print(f"  - {imp}")
    
    print("\n⚙️ Pending Adjustments:")
    for adj in state.pending_adjustments:
        print(f"  - {adj}")
    
    print("\n🔍 Discovered Patterns:")
    for pattern in evolution.discovered_patterns:
        print(f"  - {pattern.get('type')}: {pattern.get('recommendation')}")
