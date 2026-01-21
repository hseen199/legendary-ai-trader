"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Reasoning Engine
محرك التفكير المنطقي
═══════════════════════════════════════════════════════════════
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


class ReasoningType(Enum):
    """أنواع التفكير"""
    DEDUCTIVE = "استنتاجي"      # من العام للخاص
    INDUCTIVE = "استقرائي"       # من الخاص للعام
    ABDUCTIVE = "افتراضي"        # أفضل تفسير
    ANALOGICAL = "تشابهي"        # بالمقارنة
    CAUSAL = "سببي"              # السبب والنتيجة


@dataclass
class Premise:
    """مقدمة منطقية"""
    statement: str
    confidence: float
    source: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Conclusion:
    """استنتاج منطقي"""
    statement: str
    confidence: float
    reasoning_type: ReasoningType
    premises: List[Premise]
    supporting_evidence: List[str]
    counter_evidence: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Hypothesis:
    """فرضية"""
    statement: str
    probability: float
    evidence_for: List[str]
    evidence_against: List[str]
    test_criteria: List[str]
    status: str = "pending"  # pending, confirmed, rejected


class ReasoningEngine:
    """
    محرك التفكير المنطقي
    
    يقوم بـ:
    - التفكير الاستنتاجي والاستقرائي
    - توليد واختبار الفرضيات
    - تحليل السبب والنتيجة
    - التعلم من الأنماط
    """
    
    def __init__(self):
        """تهيئة محرك التفكير"""
        self.premises: List[Premise] = []
        self.conclusions: List[Conclusion] = []
        self.hypotheses: List[Hypothesis] = []
        self.learned_patterns: Dict[str, Any] = {}
        self.reasoning_history: List[Dict] = []
        
        # قواعد التفكير المدمجة
        self._init_reasoning_rules()
        
        logger.info("🧠 ReasoningEngine initialized")
    
    def _init_reasoning_rules(self):
        """تهيئة قواعد التفكير"""
        self.rules = {
            # قواعد السوق الصاعد
            'bullish_rules': [
                {
                    'conditions': ['rsi < 30', 'price_above_sma_200', 'volume_increasing'],
                    'conclusion': 'فرصة شراء قوية - تشبع بيعي مع اتجاه صاعد',
                    'confidence': 0.85
                },
                {
                    'conditions': ['macd_bullish_cross', 'adx > 25', 'price_above_ema_50'],
                    'conclusion': 'بداية موجة صعود - تأكيد الزخم',
                    'confidence': 0.80
                },
                {
                    'conditions': ['higher_highs', 'higher_lows', 'volume_confirmation'],
                    'conclusion': 'اتجاه صاعد مؤكد - استمرار متوقع',
                    'confidence': 0.75
                }
            ],
            # قواعد السوق الهابط
            'bearish_rules': [
                {
                    'conditions': ['rsi > 70', 'price_below_sma_200', 'volume_decreasing'],
                    'conclusion': 'فرصة بيع قوية - تشبع شرائي مع اتجاه هابط',
                    'confidence': 0.85
                },
                {
                    'conditions': ['macd_bearish_cross', 'adx > 25', 'price_below_ema_50'],
                    'conclusion': 'بداية موجة هبوط - تأكيد الضعف',
                    'confidence': 0.80
                },
                {
                    'conditions': ['lower_highs', 'lower_lows', 'volume_confirmation'],
                    'conclusion': 'اتجاه هابط مؤكد - استمرار متوقع',
                    'confidence': 0.75
                }
            ],
            # قواعد التقلب
            'volatility_rules': [
                {
                    'conditions': ['bb_squeeze', 'low_atr', 'consolidation'],
                    'conclusion': 'انفجار سعري وشيك - استعد للحركة',
                    'confidence': 0.70
                },
                {
                    'conditions': ['high_atr', 'wide_bb', 'erratic_price'],
                    'conclusion': 'تقلب عالي - قلل حجم الصفقات',
                    'confidence': 0.80
                }
            ],
            # قواعد الانعكاس
            'reversal_rules': [
                {
                    'conditions': ['divergence_rsi', 'support_level', 'volume_spike'],
                    'conclusion': 'انعكاس محتمل - راقب التأكيد',
                    'confidence': 0.65
                },
                {
                    'conditions': ['double_bottom', 'bullish_engulfing', 'volume_increase'],
                    'conclusion': 'انعكاس صعودي مؤكد',
                    'confidence': 0.75
                }
            ]
        }
    
    # ═══════════════════════════════════════════════════════════════
    # DEDUCTIVE REASONING - التفكير الاستنتاجي
    # ═══════════════════════════════════════════════════════════════
    
    def deduce(
        self,
        premises: List[Premise],
        context: Dict[str, Any]
    ) -> Optional[Conclusion]:
        """
        التفكير الاستنتاجي - من العام للخاص
        
        Args:
            premises: المقدمات
            context: السياق
            
        Returns:
            الاستنتاج
        """
        if not premises:
            return None
        
        # البحث عن قاعدة مطابقة
        matched_rule = None
        max_match_score = 0
        
        for category, rules in self.rules.items():
            for rule in rules:
                match_score = self._calculate_rule_match(
                    rule['conditions'], 
                    premises, 
                    context
                )
                if match_score > max_match_score:
                    max_match_score = match_score
                    matched_rule = rule
        
        if matched_rule and max_match_score > 0.5:
            conclusion = Conclusion(
                statement=matched_rule['conclusion'],
                confidence=matched_rule['confidence'] * max_match_score,
                reasoning_type=ReasoningType.DEDUCTIVE,
                premises=premises,
                supporting_evidence=[p.statement for p in premises],
                counter_evidence=[]
            )
            
            self.conclusions.append(conclusion)
            self._log_reasoning('DEDUCTIVE', premises, conclusion)
            
            return conclusion
        
        return None
    
    def _calculate_rule_match(
        self,
        conditions: List[str],
        premises: List[Premise],
        context: Dict[str, Any]
    ) -> float:
        """حساب درجة تطابق القاعدة"""
        if not conditions:
            return 0.0
        
        matched = 0
        premise_texts = [p.statement.lower() for p in premises]
        
        for condition in conditions:
            # البحث في المقدمات
            for text in premise_texts:
                if condition.lower() in text or self._semantic_match(condition, text):
                    matched += 1
                    break
            else:
                # البحث في السياق
                if self._check_context_condition(condition, context):
                    matched += 1
        
        return matched / len(conditions)
    
    def _semantic_match(self, condition: str, text: str) -> bool:
        """تطابق دلالي بسيط"""
        # كلمات مترادفة
        synonyms = {
            'bullish': ['صاعد', 'إيجابي', 'شراء', 'ارتفاع'],
            'bearish': ['هابط', 'سلبي', 'بيع', 'انخفاض'],
            'high': ['عالي', 'مرتفع', 'كبير'],
            'low': ['منخفض', 'صغير', 'ضعيف'],
            'increasing': ['متزايد', 'يرتفع', 'ينمو'],
            'decreasing': ['متناقص', 'يهبط', 'ينخفض']
        }
        
        for key, values in synonyms.items():
            if key in condition.lower():
                for syn in values:
                    if syn in text:
                        return True
        
        return False
    
    def _check_context_condition(self, condition: str, context: Dict) -> bool:
        """التحقق من شرط في السياق"""
        # تحليل الشرط
        parts = condition.replace('_', ' ').split()
        
        for key, value in context.items():
            key_lower = key.lower()
            if any(p in key_lower for p in parts):
                if isinstance(value, bool):
                    return value
                elif isinstance(value, (int, float)):
                    # التحقق من الشروط الرقمية
                    if '>' in condition:
                        threshold = float(condition.split('>')[-1].strip())
                        return value > threshold
                    elif '<' in condition:
                        threshold = float(condition.split('<')[-1].strip())
                        return value < threshold
        
        return False
    
    # ═══════════════════════════════════════════════════════════════
    # INDUCTIVE REASONING - التفكير الاستقرائي
    # ═══════════════════════════════════════════════════════════════
    
    def induce(
        self,
        observations: List[Dict[str, Any]],
        min_pattern_frequency: int = 3
    ) -> List[Dict[str, Any]]:
        """
        التفكير الاستقرائي - من الخاص للعام
        اكتشاف الأنماط من الملاحظات
        
        Args:
            observations: الملاحظات
            min_pattern_frequency: الحد الأدنى لتكرار النمط
            
        Returns:
            الأنماط المكتشفة
        """
        patterns = []
        
        # تجميع الملاحظات حسب النتيجة
        outcome_groups = {}
        for obs in observations:
            outcome = obs.get('outcome', 'unknown')
            if outcome not in outcome_groups:
                outcome_groups[outcome] = []
            outcome_groups[outcome].append(obs)
        
        # البحث عن أنماط مشتركة
        for outcome, group in outcome_groups.items():
            if len(group) < min_pattern_frequency:
                continue
            
            # إيجاد الميزات المشتركة
            common_features = self._find_common_features(group)
            
            if common_features:
                pattern = {
                    'type': 'induced',
                    'outcome': outcome,
                    'conditions': common_features,
                    'frequency': len(group),
                    'confidence': len(group) / len(observations),
                    'discovered_at': datetime.now().isoformat()
                }
                patterns.append(pattern)
                
                # حفظ النمط
                pattern_key = f"{outcome}_{hash(str(common_features))}"
                self.learned_patterns[pattern_key] = pattern
        
        self._log_reasoning('INDUCTIVE', observations, patterns)
        return patterns
    
    def _find_common_features(
        self,
        observations: List[Dict]
    ) -> Dict[str, Any]:
        """إيجاد الميزات المشتركة"""
        if not observations:
            return {}
        
        # جمع كل الميزات
        all_features = {}
        for obs in observations:
            features = obs.get('features', {})
            for key, value in features.items():
                if key not in all_features:
                    all_features[key] = []
                all_features[key].append(value)
        
        # إيجاد الميزات المتسقة
        common = {}
        for key, values in all_features.items():
            if len(values) == len(observations):
                # التحقق من التناسق
                if all(isinstance(v, bool) for v in values):
                    if all(values) or not any(values):
                        common[key] = values[0]
                elif all(isinstance(v, (int, float)) for v in values):
                    mean = np.mean(values)
                    std = np.std(values)
                    if std / (abs(mean) + 1e-10) < 0.3:  # تباين منخفض
                        common[key] = {'mean': mean, 'std': std}
        
        return common
    
    # ═══════════════════════════════════════════════════════════════
    # HYPOTHESIS GENERATION & TESTING
    # ═══════════════════════════════════════════════════════════════
    
    def generate_hypothesis(
        self,
        observation: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Hypothesis:
        """
        توليد فرضية من ملاحظة
        
        Args:
            observation: الملاحظة
            context: السياق
            
        Returns:
            الفرضية
        """
        # تحليل الملاحظة
        features = observation.get('features', {})
        
        # توليد فرضية بناءً على الأنماط المتعلمة
        best_match = None
        best_score = 0
        
        for pattern_key, pattern in self.learned_patterns.items():
            score = self._pattern_match_score(features, pattern['conditions'])
            if score > best_score:
                best_score = score
                best_match = pattern
        
        if best_match and best_score > 0.5:
            hypothesis = Hypothesis(
                statement=f"بناءً على النمط المتعلم، النتيجة المتوقعة: {best_match['outcome']}",
                probability=best_score * best_match['confidence'],
                evidence_for=[f"تطابق {best_score:.0%} مع النمط"],
                evidence_against=[],
                test_criteria=[
                    f"مراقبة السعر لمدة 15 دقيقة",
                    f"التحقق من الحجم",
                    f"مراقبة المؤشرات الفنية"
                ]
            )
        else:
            # فرضية افتراضية
            hypothesis = Hypothesis(
                statement="لا يوجد نمط واضح - السوق في حالة غير محددة",
                probability=0.5,
                evidence_for=["عدم وجود إشارات قوية"],
                evidence_against=["غياب الأنماط المعروفة"],
                test_criteria=["انتظار إشارة واضحة"]
            )
        
        self.hypotheses.append(hypothesis)
        return hypothesis
    
    def test_hypothesis(
        self,
        hypothesis: Hypothesis,
        new_data: Dict[str, Any]
    ) -> Hypothesis:
        """
        اختبار فرضية ببيانات جديدة
        
        Args:
            hypothesis: الفرضية
            new_data: البيانات الجديدة
            
        Returns:
            الفرضية المحدثة
        """
        # التحقق من معايير الاختبار
        tests_passed = 0
        tests_failed = 0
        
        for criterion in hypothesis.test_criteria:
            if self._evaluate_criterion(criterion, new_data):
                tests_passed += 1
                hypothesis.evidence_for.append(f"✓ {criterion}")
            else:
                tests_failed += 1
                hypothesis.evidence_against.append(f"✗ {criterion}")
        
        # تحديث الاحتمالية
        total_tests = tests_passed + tests_failed
        if total_tests > 0:
            pass_rate = tests_passed / total_tests
            hypothesis.probability = hypothesis.probability * 0.7 + pass_rate * 0.3
        
        # تحديث الحالة
        if hypothesis.probability > 0.7:
            hypothesis.status = "confirmed"
        elif hypothesis.probability < 0.3:
            hypothesis.status = "rejected"
        else:
            hypothesis.status = "pending"
        
        return hypothesis
    
    def _pattern_match_score(
        self,
        features: Dict,
        conditions: Dict
    ) -> float:
        """حساب درجة تطابق النمط"""
        if not conditions:
            return 0.0
        
        matched = 0
        total = len(conditions)
        
        for key, expected in conditions.items():
            if key in features:
                actual = features[key]
                if isinstance(expected, dict):
                    # مقارنة رقمية
                    mean = expected.get('mean', 0)
                    std = expected.get('std', 1)
                    if abs(actual - mean) <= 2 * std:
                        matched += 1
                elif actual == expected:
                    matched += 1
        
        return matched / total if total > 0 else 0.0
    
    def _evaluate_criterion(self, criterion: str, data: Dict) -> bool:
        """تقييم معيار اختبار"""
        # تقييم بسيط - يمكن توسيعه
        criterion_lower = criterion.lower()
        
        if 'سعر' in criterion_lower or 'price' in criterion_lower:
            return data.get('price_moved', False)
        elif 'حجم' in criterion_lower or 'volume' in criterion_lower:
            return data.get('volume_confirmed', False)
        elif 'مؤشر' in criterion_lower or 'indicator' in criterion_lower:
            return data.get('indicators_aligned', False)
        
        return True  # افتراضي
    
    # ═══════════════════════════════════════════════════════════════
    # CAUSAL REASONING - التفكير السببي
    # ═══════════════════════════════════════════════════════════════
    
    def analyze_causality(
        self,
        event: Dict[str, Any],
        potential_causes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        تحليل العلاقة السببية
        
        Args:
            event: الحدث
            potential_causes: الأسباب المحتملة
            
        Returns:
            تحليل السببية
        """
        causal_analysis = {
            'event': event,
            'likely_causes': [],
            'unlikely_causes': [],
            'confidence': 0.0
        }
        
        for cause in potential_causes:
            # حساب العلاقة الزمنية
            temporal_score = self._temporal_correlation(cause, event)
            
            # حساب العلاقة المنطقية
            logical_score = self._logical_correlation(cause, event)
            
            # الدرجة الإجمالية
            total_score = temporal_score * 0.4 + logical_score * 0.6
            
            cause_analysis = {
                'cause': cause,
                'temporal_score': temporal_score,
                'logical_score': logical_score,
                'total_score': total_score
            }
            
            if total_score > 0.5:
                causal_analysis['likely_causes'].append(cause_analysis)
            else:
                causal_analysis['unlikely_causes'].append(cause_analysis)
        
        # حساب الثقة الإجمالية
        if causal_analysis['likely_causes']:
            causal_analysis['confidence'] = np.mean([
                c['total_score'] for c in causal_analysis['likely_causes']
            ])
        
        return causal_analysis
    
    def _temporal_correlation(self, cause: Dict, effect: Dict) -> float:
        """حساب العلاقة الزمنية"""
        cause_time = cause.get('timestamp')
        effect_time = effect.get('timestamp')
        
        if cause_time and effect_time:
            if isinstance(cause_time, str):
                cause_time = datetime.fromisoformat(cause_time)
            if isinstance(effect_time, str):
                effect_time = datetime.fromisoformat(effect_time)
            
            # السبب يجب أن يسبق النتيجة
            if cause_time < effect_time:
                time_diff = (effect_time - cause_time).total_seconds()
                # كلما كان الفارق أقل، كانت العلاقة أقوى
                return max(0, 1 - time_diff / 3600)  # تناقص خلال ساعة
        
        return 0.5  # افتراضي
    
    def _logical_correlation(self, cause: Dict, effect: Dict) -> float:
        """حساب العلاقة المنطقية"""
        cause_type = cause.get('type', '')
        effect_type = effect.get('type', '')
        
        # علاقات منطقية معروفة
        logical_relations = {
            ('volume_spike', 'price_move'): 0.8,
            ('news_event', 'volatility'): 0.7,
            ('whale_move', 'price_move'): 0.75,
            ('indicator_signal', 'price_move'): 0.6,
            ('market_open', 'volatility'): 0.65
        }
        
        return logical_relations.get((cause_type, effect_type), 0.3)
    
    # ═══════════════════════════════════════════════════════════════
    # REASONING CHAIN
    # ═══════════════════════════════════════════════════════════════
    
    def build_reasoning_chain(
        self,
        initial_observation: Dict[str, Any],
        context: Dict[str, Any],
        max_steps: int = 5
    ) -> Dict[str, Any]:
        """
        بناء سلسلة تفكير كاملة
        
        Args:
            initial_observation: الملاحظة الأولية
            context: السياق
            max_steps: الحد الأقصى للخطوات
            
        Returns:
            سلسلة التفكير
        """
        chain = {
            'steps': [],
            'final_conclusion': None,
            'confidence': 0.0,
            'reasoning_types_used': set()
        }
        
        current_state = initial_observation
        
        for step in range(max_steps):
            # 1. توليد فرضية
            hypothesis = self.generate_hypothesis(current_state, context)
            
            # 2. محاولة الاستنتاج
            premises = [
                Premise(
                    statement=str(current_state),
                    confidence=0.8,
                    source="observation"
                )
            ]
            conclusion = self.deduce(premises, context)
            
            step_info = {
                'step': step + 1,
                'hypothesis': hypothesis.statement,
                'hypothesis_probability': hypothesis.probability,
                'conclusion': conclusion.statement if conclusion else None,
                'conclusion_confidence': conclusion.confidence if conclusion else 0
            }
            chain['steps'].append(step_info)
            
            if conclusion:
                chain['reasoning_types_used'].add(conclusion.reasoning_type.value)
            
            # التحقق من الوصول لاستنتاج نهائي
            if conclusion and conclusion.confidence > 0.75:
                chain['final_conclusion'] = conclusion
                chain['confidence'] = conclusion.confidence
                break
            
            # تحديث الحالة للخطوة التالية
            if conclusion:
                current_state = {
                    'previous': current_state,
                    'conclusion': conclusion.statement,
                    'confidence': conclusion.confidence
                }
        
        chain['reasoning_types_used'] = list(chain['reasoning_types_used'])
        return chain
    
    # ═══════════════════════════════════════════════════════════════
    # LOGGING & HISTORY
    # ═══════════════════════════════════════════════════════════════
    
    def _log_reasoning(
        self,
        reasoning_type: str,
        inputs: Any,
        output: Any
    ):
        """تسجيل عملية التفكير"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': reasoning_type,
            'inputs_summary': str(inputs)[:200],
            'output_summary': str(output)[:200]
        }
        self.reasoning_history.append(entry)
        
        # الاحتفاظ بآخر 1000 سجل
        if len(self.reasoning_history) > 1000:
            self.reasoning_history = self.reasoning_history[-1000:]
    
    def get_reasoning_summary(self) -> Dict[str, Any]:
        """الحصول على ملخص التفكير"""
        return {
            'total_premises': len(self.premises),
            'total_conclusions': len(self.conclusions),
            'total_hypotheses': len(self.hypotheses),
            'learned_patterns': len(self.learned_patterns),
            'reasoning_history_size': len(self.reasoning_history),
            'confirmed_hypotheses': sum(
                1 for h in self.hypotheses if h.status == 'confirmed'
            ),
            'rejected_hypotheses': sum(
                1 for h in self.hypotheses if h.status == 'rejected'
            )
        }


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار محرك التفكير
    engine = ReasoningEngine()
    
    # اختبار الاستنتاج
    premises = [
        Premise("RSI أقل من 30", 0.9, "technical_analysis"),
        Premise("السعر فوق المتوسط 200", 0.85, "technical_analysis"),
        Premise("الحجم متزايد", 0.8, "volume_analysis")
    ]
    
    context = {
        'rsi': 25,
        'price_above_sma_200': True,
        'volume_trend': 'increasing'
    }
    
    conclusion = engine.deduce(premises, context)
    if conclusion:
        print(f"Conclusion: {conclusion.statement}")
        print(f"Confidence: {conclusion.confidence:.2%}")
    
    # اختبار سلسلة التفكير
    observation = {
        'features': {
            'rsi': 28,
            'macd_signal': 'bullish',
            'volume_spike': True
        }
    }
    
    chain = engine.build_reasoning_chain(observation, context)
    print(f"\nReasoning chain: {len(chain['steps'])} steps")
    print(f"Final confidence: {chain['confidence']:.2%}")
    
    # ملخص
    print(f"\nSummary: {engine.get_reasoning_summary()}")
