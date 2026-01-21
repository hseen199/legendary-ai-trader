"""
Legendary Trading System - Mistake Learning System
نظام التداول الخارق - نظام التعلم من الأخطاء

نظام متقدم لتحليل الأخطاء والتعلم منها.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import logging
import json
import hashlib


class MistakeCategory(Enum):
    """فئات الأخطاء"""
    ENTRY_TIMING = "entry_timing"           # توقيت الدخول
    EXIT_TIMING = "exit_timing"             # توقيت الخروج
    POSITION_SIZE = "position_size"         # حجم الصفقة
    STOP_LOSS = "stop_loss"                 # وقف الخسارة
    TAKE_PROFIT = "take_profit"             # جني الأرباح
    TREND_ANALYSIS = "trend_analysis"       # تحليل الاتجاه
    RISK_MANAGEMENT = "risk_management"     # إدارة المخاطر
    MARKET_CONDITIONS = "market_conditions" # ظروف السوق
    EMOTIONAL = "emotional"                 # عاطفي
    TECHNICAL = "technical"                 # فني


class MistakeSeverity(Enum):
    """شدة الخطأ"""
    MINOR = 1       # بسيط
    MODERATE = 2    # متوسط
    MAJOR = 3       # كبير
    CRITICAL = 4    # حرج


@dataclass
class TradeMistake:
    """خطأ في صفقة"""
    id: str
    trade_id: str
    category: MistakeCategory
    severity: MistakeSeverity
    description: str
    timestamp: datetime
    
    # تفاصيل الصفقة
    trade_details: Dict[str, Any] = field(default_factory=dict)
    
    # التحليل
    root_cause: str = ""
    contributing_factors: List[str] = field(default_factory=list)
    
    # الدروس
    lessons: List[str] = field(default_factory=list)
    prevention_rules: List[str] = field(default_factory=list)
    
    # الحالة
    addressed: bool = False
    recurrence_count: int = 0


@dataclass
class MistakePattern:
    """نمط خطأ متكرر"""
    id: str
    category: MistakeCategory
    pattern_description: str
    occurrences: int
    first_seen: datetime
    last_seen: datetime
    common_conditions: Dict[str, Any]
    prevention_strategy: str
    effectiveness: float = 0.0  # فعالية الوقاية


@dataclass
class LessonLearned:
    """درس مستفاد"""
    id: str
    title: str
    description: str
    source_mistakes: List[str]
    category: MistakeCategory
    importance: float
    created_at: datetime
    applied_count: int = 0
    success_rate: float = 0.0


class MistakeLearningSystem:
    """
    نظام التعلم من الأخطاء.
    
    يوفر:
    - تحليل كل صفقة خاسرة
    - استخراج الأنماط المشتركة في الخسائر
    - تجنب تكرار نفس الأخطاء
    - بناء قاعدة معرفة من الدروس
    """
    
    def __init__(self, memory_system=None):
        self.logger = logging.getLogger("MistakeLearningSystem")
        self.memory_system = memory_system
        
        # قاعدة بيانات الأخطاء
        self.mistakes: Dict[str, TradeMistake] = {}
        
        # أنماط الأخطاء
        self.patterns: Dict[str, MistakePattern] = {}
        
        # الدروس المستفادة
        self.lessons: Dict[str, LessonLearned] = {}
        
        # قواعد الوقاية
        self.prevention_rules: Dict[str, Dict[str, Any]] = {}
        
        # إحصائيات
        self.stats = {
            "total_mistakes": 0,
            "patterns_identified": 0,
            "lessons_learned": 0,
            "mistakes_prevented": 0,
            "by_category": defaultdict(int)
        }
        
        # تهيئة المحللات
        self._init_analyzers()
    
    def _init_analyzers(self):
        """تهيئة محللات الأخطاء."""
        self.analyzers = {
            MistakeCategory.ENTRY_TIMING: self._analyze_entry_timing,
            MistakeCategory.EXIT_TIMING: self._analyze_exit_timing,
            MistakeCategory.POSITION_SIZE: self._analyze_position_size,
            MistakeCategory.STOP_LOSS: self._analyze_stop_loss,
            MistakeCategory.TREND_ANALYSIS: self._analyze_trend,
            MistakeCategory.RISK_MANAGEMENT: self._analyze_risk,
            MistakeCategory.MARKET_CONDITIONS: self._analyze_market_conditions
        }
    
    async def analyze_losing_trade(self, 
                                  trade: Dict[str, Any]) -> List[TradeMistake]:
        """
        تحليل صفقة خاسرة.
        
        Args:
            trade: بيانات الصفقة
            
        Returns:
            قائمة الأخطاء المكتشفة
        """
        if trade.get("pnl", 0) >= 0:
            return []
        
        mistakes = []
        
        self.logger.info(f"تحليل صفقة خاسرة: {trade.get('id', 'unknown')}")
        
        # تحليل كل جانب
        for category, analyzer in self.analyzers.items():
            mistake = await analyzer(trade)
            if mistake:
                mistakes.append(mistake)
        
        # تحليل شامل
        comprehensive_mistakes = await self._comprehensive_analysis(trade, mistakes)
        mistakes.extend(comprehensive_mistakes)
        
        # حفظ الأخطاء
        for mistake in mistakes:
            self.mistakes[mistake.id] = mistake
            self.stats["total_mistakes"] += 1
            self.stats["by_category"][mistake.category.value] += 1
        
        # تحديث الأنماط
        await self._update_patterns(mistakes)
        
        # استخراج الدروس
        await self._extract_lessons(mistakes)
        
        return mistakes
    
    async def _analyze_entry_timing(self, trade: Dict[str, Any]) -> Optional[TradeMistake]:
        """تحليل توقيت الدخول."""
        entry_price = trade.get("entry_price", 0)
        high_after_entry = trade.get("high_after_entry", entry_price)
        low_after_entry = trade.get("low_after_entry", entry_price)
        
        # للشراء: هل دخلنا عند القمة؟
        if trade.get("side") == "buy":
            if entry_price > low_after_entry * 1.02:  # دخلنا أعلى من القاع بـ 2%
                return TradeMistake(
                    id=self._generate_id(trade, "entry_timing"),
                    trade_id=trade.get("id", ""),
                    category=MistakeCategory.ENTRY_TIMING,
                    severity=MistakeSeverity.MODERATE,
                    description="الدخول تم عند سعر مرتفع نسبياً",
                    timestamp=datetime.utcnow(),
                    trade_details=trade,
                    root_cause="عدم انتظار تصحيح السعر",
                    contributing_factors=["FOMO", "إشارة مبكرة"],
                    lessons=["انتظار تأكيد الدعم قبل الدخول"],
                    prevention_rules=["لا تدخل فوراً عند الإشارة"]
                )
        
        return None
    
    async def _analyze_exit_timing(self, trade: Dict[str, Any]) -> Optional[TradeMistake]:
        """تحليل توقيت الخروج."""
        exit_price = trade.get("exit_price", 0)
        high_before_exit = trade.get("high_before_exit", exit_price)
        
        # هل خرجنا متأخرين؟
        if trade.get("side") == "buy":
            if high_before_exit > exit_price * 1.05:  # كان السعر أعلى بـ 5%
                return TradeMistake(
                    id=self._generate_id(trade, "exit_timing"),
                    trade_id=trade.get("id", ""),
                    category=MistakeCategory.EXIT_TIMING,
                    severity=MistakeSeverity.MAJOR,
                    description="الخروج تأخر كثيراً بعد القمة",
                    timestamp=datetime.utcnow(),
                    trade_details=trade,
                    root_cause="عدم استخدام وقف متحرك فعال",
                    contributing_factors=["طمع", "أمل في المزيد"],
                    lessons=["استخدام وقف متحرك عند الربح"],
                    prevention_rules=["تفعيل وقف متحرك عند ربح 3%"]
                )
        
        return None
    
    async def _analyze_position_size(self, trade: Dict[str, Any]) -> Optional[TradeMistake]:
        """تحليل حجم الصفقة."""
        position_size = trade.get("position_size", 0)
        portfolio_value = trade.get("portfolio_value", 1)
        loss = abs(trade.get("pnl", 0))
        
        position_ratio = position_size / portfolio_value if portfolio_value > 0 else 0
        loss_ratio = loss / portfolio_value if portfolio_value > 0 else 0
        
        # هل الحجم كان كبيراً جداً؟
        if position_ratio > 0.2 or loss_ratio > 0.05:
            return TradeMistake(
                id=self._generate_id(trade, "position_size"),
                trade_id=trade.get("id", ""),
                category=MistakeCategory.POSITION_SIZE,
                severity=MistakeSeverity.MAJOR if loss_ratio > 0.05 else MistakeSeverity.MODERATE,
                description=f"حجم الصفقة كبير ({position_ratio:.1%} من المحفظة)",
                timestamp=datetime.utcnow(),
                trade_details=trade,
                root_cause="عدم احترام قواعد إدارة المخاطر",
                contributing_factors=["ثقة زائدة", "تجاهل الحدود"],
                lessons=["الالتزام بحد أقصى 10% لكل صفقة"],
                prevention_rules=["رفض الصفقات > 10% من المحفظة"]
            )
        
        return None
    
    async def _analyze_stop_loss(self, trade: Dict[str, Any]) -> Optional[TradeMistake]:
        """تحليل وقف الخسارة."""
        stop_loss = trade.get("stop_loss")
        entry_price = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        
        if not stop_loss:
            return TradeMistake(
                id=self._generate_id(trade, "stop_loss_missing"),
                trade_id=trade.get("id", ""),
                category=MistakeCategory.STOP_LOSS,
                severity=MistakeSeverity.CRITICAL,
                description="لم يتم تعيين وقف خسارة",
                timestamp=datetime.utcnow(),
                trade_details=trade,
                root_cause="إهمال إدارة المخاطر",
                contributing_factors=["ثقة زائدة", "نسيان"],
                lessons=["وقف الخسارة إلزامي لكل صفقة"],
                prevention_rules=["رفض الصفقات بدون وقف خسارة"]
            )
        
        # هل وقف الخسارة كان بعيداً جداً؟
        if trade.get("side") == "buy":
            stop_distance = (entry_price - stop_loss) / entry_price
            if stop_distance > 0.05:  # أكثر من 5%
                return TradeMistake(
                    id=self._generate_id(trade, "stop_loss_wide"),
                    trade_id=trade.get("id", ""),
                    category=MistakeCategory.STOP_LOSS,
                    severity=MistakeSeverity.MODERATE,
                    description=f"وقف الخسارة بعيد ({stop_distance:.1%})",
                    timestamp=datetime.utcnow(),
                    trade_details=trade,
                    root_cause="وقف خسارة غير محسوب",
                    contributing_factors=["خوف من الإيقاف المبكر"],
                    lessons=["وقف الخسارة يجب أن يكون منطقياً"],
                    prevention_rules=["وقف الخسارة لا يتجاوز 3%"]
                )
        
        return None
    
    async def _analyze_trend(self, trade: Dict[str, Any]) -> Optional[TradeMistake]:
        """تحليل الاتجاه."""
        trade_direction = trade.get("side")
        market_trend = trade.get("market_trend")
        
        # التداول عكس الاتجاه
        if trade_direction == "buy" and market_trend == "bearish":
            return TradeMistake(
                id=self._generate_id(trade, "trend_against"),
                trade_id=trade.get("id", ""),
                category=MistakeCategory.TREND_ANALYSIS,
                severity=MistakeSeverity.MAJOR,
                description="شراء في سوق هابط",
                timestamp=datetime.utcnow(),
                trade_details=trade,
                root_cause="تجاهل الاتجاه العام",
                contributing_factors=["محاولة اصطياد القاع", "تفاؤل زائد"],
                lessons=["لا تتداول عكس الاتجاه الرئيسي"],
                prevention_rules=["تأكد من توافق الصفقة مع الاتجاه"]
            )
        
        return None
    
    async def _analyze_risk(self, trade: Dict[str, Any]) -> Optional[TradeMistake]:
        """تحليل إدارة المخاطر."""
        risk_reward = trade.get("risk_reward_ratio", 1)
        
        if risk_reward < 1:
            return TradeMistake(
                id=self._generate_id(trade, "risk_reward"),
                trade_id=trade.get("id", ""),
                category=MistakeCategory.RISK_MANAGEMENT,
                severity=MistakeSeverity.MODERATE,
                description=f"نسبة مخاطرة/عائد ضعيفة ({risk_reward:.2f})",
                timestamp=datetime.utcnow(),
                trade_details=trade,
                root_cause="صفقة غير متوازنة",
                contributing_factors=["هدف قريب جداً", "وقف خسارة بعيد"],
                lessons=["نسبة مخاطرة/عائد لا تقل عن 1:2"],
                prevention_rules=["رفض الصفقات بنسبة أقل من 1:1.5"]
            )
        
        return None
    
    async def _analyze_market_conditions(self, trade: Dict[str, Any]) -> Optional[TradeMistake]:
        """تحليل ظروف السوق."""
        volatility = trade.get("volatility", 0.02)
        liquidity = trade.get("liquidity", 1)
        
        # تقلب عالي
        if volatility > 0.05:
            return TradeMistake(
                id=self._generate_id(trade, "high_volatility"),
                trade_id=trade.get("id", ""),
                category=MistakeCategory.MARKET_CONDITIONS,
                severity=MistakeSeverity.MODERATE,
                description=f"التداول في تقلب عالي ({volatility:.1%})",
                timestamp=datetime.utcnow(),
                trade_details=trade,
                root_cause="تجاهل ظروف السوق",
                contributing_factors=["طمع", "عدم الصبر"],
                lessons=["تجنب التداول في التقلب العالي"],
                prevention_rules=["تقليل الحجم عند التقلب > 4%"]
            )
        
        # سيولة منخفضة
        if liquidity < 0.5:
            return TradeMistake(
                id=self._generate_id(trade, "low_liquidity"),
                trade_id=trade.get("id", ""),
                category=MistakeCategory.MARKET_CONDITIONS,
                severity=MistakeSeverity.MODERATE,
                description="التداول في سيولة منخفضة",
                timestamp=datetime.utcnow(),
                trade_details=trade,
                root_cause="تجاهل السيولة",
                contributing_factors=["عملة غير سائلة", "وقت غير مناسب"],
                lessons=["التأكد من السيولة قبل الدخول"],
                prevention_rules=["تجنب العملات منخفضة السيولة"]
            )
        
        return None
    
    async def _comprehensive_analysis(self,
                                     trade: Dict[str, Any],
                                     existing_mistakes: List[TradeMistake]) -> List[TradeMistake]:
        """تحليل شامل للصفقة."""
        additional_mistakes = []
        
        # إذا كان هناك أخطاء متعددة
        if len(existing_mistakes) >= 3:
            additional_mistakes.append(TradeMistake(
                id=self._generate_id(trade, "multiple_errors"),
                trade_id=trade.get("id", ""),
                category=MistakeCategory.RISK_MANAGEMENT,
                severity=MistakeSeverity.CRITICAL,
                description=f"أخطاء متعددة في صفقة واحدة ({len(existing_mistakes)})",
                timestamp=datetime.utcnow(),
                trade_details=trade,
                root_cause="قرار متسرع أو غير مدروس",
                contributing_factors=[m.category.value for m in existing_mistakes],
                lessons=["مراجعة شاملة قبل كل صفقة"],
                prevention_rules=["استخدام قائمة تحقق قبل الدخول"]
            ))
        
        return additional_mistakes
    
    def _generate_id(self, trade: Dict[str, Any], suffix: str) -> str:
        """توليد معرف فريد."""
        data = f"{trade.get('id', '')}_{suffix}_{datetime.utcnow().timestamp()}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
    
    async def _update_patterns(self, mistakes: List[TradeMistake]):
        """تحديث أنماط الأخطاء."""
        for mistake in mistakes:
            pattern_key = f"{mistake.category.value}"
            
            if pattern_key in self.patterns:
                pattern = self.patterns[pattern_key]
                pattern.occurrences += 1
                pattern.last_seen = datetime.utcnow()
            else:
                self.patterns[pattern_key] = MistakePattern(
                    id=pattern_key,
                    category=mistake.category,
                    pattern_description=mistake.description,
                    occurrences=1,
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    common_conditions={},
                    prevention_strategy=mistake.prevention_rules[0] if mistake.prevention_rules else ""
                )
                self.stats["patterns_identified"] += 1
    
    async def _extract_lessons(self, mistakes: List[TradeMistake]):
        """استخراج الدروس من الأخطاء."""
        for mistake in mistakes:
            for lesson_text in mistake.lessons:
                lesson_id = hashlib.md5(lesson_text.encode()).hexdigest()[:12]
                
                if lesson_id in self.lessons:
                    self.lessons[lesson_id].source_mistakes.append(mistake.id)
                else:
                    self.lessons[lesson_id] = LessonLearned(
                        id=lesson_id,
                        title=lesson_text[:50],
                        description=lesson_text,
                        source_mistakes=[mistake.id],
                        category=mistake.category,
                        importance=mistake.severity.value / 4,
                        created_at=datetime.utcnow()
                    )
                    self.stats["lessons_learned"] += 1
    
    async def check_for_repeat_mistake(self,
                                      proposed_trade: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        فحص إذا كانت الصفقة المقترحة قد تكرر خطأ سابق.
        
        Args:
            proposed_trade: الصفقة المقترحة
            
        Returns:
            قائمة التحذيرات
        """
        warnings = []
        
        # فحص كل نمط
        for pattern_id, pattern in self.patterns.items():
            if pattern.occurrences >= 3:  # نمط متكرر
                # فحص إذا كانت الصفقة تطابق النمط
                if self._matches_pattern(proposed_trade, pattern):
                    warnings.append({
                        "type": "repeat_mistake",
                        "pattern": pattern_id,
                        "description": pattern.pattern_description,
                        "occurrences": pattern.occurrences,
                        "prevention": pattern.prevention_strategy,
                        "severity": "high" if pattern.occurrences >= 5 else "medium"
                    })
        
        # فحص قواعد الوقاية
        for rule_id, rule in self.prevention_rules.items():
            if self._violates_rule(proposed_trade, rule):
                warnings.append({
                    "type": "rule_violation",
                    "rule": rule_id,
                    "description": rule.get("description", ""),
                    "severity": rule.get("severity", "medium")
                })
        
        if warnings:
            self.logger.warning(f"تحذيرات للصفقة المقترحة: {len(warnings)}")
        
        return warnings
    
    def _matches_pattern(self, trade: Dict[str, Any], pattern: MistakePattern) -> bool:
        """فحص إذا كانت الصفقة تطابق نمط خطأ."""
        # فحص بسيط بناءً على الفئة
        if pattern.category == MistakeCategory.TREND_ANALYSIS:
            if trade.get("side") == "buy" and trade.get("market_trend") == "bearish":
                return True
        
        if pattern.category == MistakeCategory.MARKET_CONDITIONS:
            if trade.get("volatility", 0) > 0.05:
                return True
        
        if pattern.category == MistakeCategory.POSITION_SIZE:
            portfolio = trade.get("portfolio_value", 1)
            size = trade.get("position_size", 0)
            if size / portfolio > 0.15:
                return True
        
        return False
    
    def _violates_rule(self, trade: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        """فحص إذا كانت الصفقة تخالف قاعدة."""
        condition = rule.get("condition", {})
        
        for key, value in condition.items():
            trade_value = trade.get(key)
            if trade_value is not None:
                operator = value.get("operator", "==")
                threshold = value.get("value")
                
                if operator == ">" and trade_value > threshold:
                    return True
                elif operator == "<" and trade_value < threshold:
                    return True
                elif operator == "==" and trade_value == threshold:
                    return True
        
        return False
    
    async def get_lessons_for_situation(self,
                                       context: Dict[str, Any]) -> List[LessonLearned]:
        """
        الحصول على الدروس المناسبة للموقف.
        
        Args:
            context: سياق الموقف
            
        Returns:
            الدروس المناسبة
        """
        relevant_lessons = []
        
        for lesson in self.lessons.values():
            # تصفية حسب الفئة
            if self._lesson_applies(lesson, context):
                relevant_lessons.append(lesson)
        
        # ترتيب حسب الأهمية
        relevant_lessons.sort(key=lambda l: l.importance, reverse=True)
        
        return relevant_lessons[:5]
    
    def _lesson_applies(self, lesson: LessonLearned, context: Dict[str, Any]) -> bool:
        """تحديد إذا كان الدرس ينطبق على الموقف."""
        # منطق بسيط للمطابقة
        if lesson.category == MistakeCategory.TREND_ANALYSIS:
            if "trend" in context:
                return True
        
        if lesson.category == MistakeCategory.MARKET_CONDITIONS:
            if "volatility" in context or "liquidity" in context:
                return True
        
        if lesson.category == MistakeCategory.POSITION_SIZE:
            if "position_size" in context:
                return True
        
        return False
    
    def add_prevention_rule(self, 
                           rule_id: str,
                           description: str,
                           condition: Dict[str, Any],
                           severity: str = "medium"):
        """إضافة قاعدة وقاية."""
        self.prevention_rules[rule_id] = {
            "description": description,
            "condition": condition,
            "severity": severity,
            "created_at": datetime.utcnow().isoformat()
        }
    
    def record_prevention_success(self, pattern_id: str):
        """تسجيل نجاح الوقاية."""
        if pattern_id in self.patterns:
            self.patterns[pattern_id].effectiveness += 0.1
            self.patterns[pattern_id].effectiveness = min(1.0, self.patterns[pattern_id].effectiveness)
        
        self.stats["mistakes_prevented"] += 1
    
    async def get_mistake_report(self) -> Dict[str, Any]:
        """
        الحصول على تقرير الأخطاء.
        
        Returns:
            تقرير شامل
        """
        # أكثر الأخطاء شيوعاً
        common_mistakes = sorted(
            self.patterns.values(),
            key=lambda p: p.occurrences,
            reverse=True
        )[:5]
        
        # أهم الدروس
        important_lessons = sorted(
            self.lessons.values(),
            key=lambda l: l.importance * len(l.source_mistakes),
            reverse=True
        )[:5]
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_mistakes": self.stats["total_mistakes"],
                "patterns_identified": self.stats["patterns_identified"],
                "lessons_learned": self.stats["lessons_learned"],
                "mistakes_prevented": self.stats["mistakes_prevented"]
            },
            "by_category": dict(self.stats["by_category"]),
            "common_mistakes": [
                {
                    "category": p.category.value,
                    "description": p.pattern_description,
                    "occurrences": p.occurrences,
                    "prevention": p.prevention_strategy
                }
                for p in common_mistakes
            ],
            "important_lessons": [
                {
                    "title": l.title,
                    "description": l.description,
                    "importance": l.importance,
                    "applied_count": l.applied_count
                }
                for l in important_lessons
            ],
            "prevention_rules_count": len(self.prevention_rules)
        }
