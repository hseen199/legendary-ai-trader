"""
Legendary Trading System - Enhanced Strategy Inventor
نظام التداول الخارق - مخترع الاستراتيجيات المحسن

نظام متقدم لتوليد وتطوير استراتيجيات التداول.
"""

import asyncio
import random
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import copy


class StrategyType(Enum):
    """أنواع الاستراتيجيات"""
    TREND_FOLLOWING = "trend_following"       # تتبع الاتجاه
    MEAN_REVERSION = "mean_reversion"         # العودة للمتوسط
    MOMENTUM = "momentum"                      # الزخم
    BREAKOUT = "breakout"                      # الاختراق
    SCALPING = "scalping"                      # المضاربة السريعة
    SWING = "swing"                            # التداول المتأرجح
    ARBITRAGE = "arbitrage"                    # المراجحة
    HYBRID = "hybrid"                          # هجين


@dataclass
class StrategyGene:
    """جين استراتيجية (للخوارزمية الجينية)"""
    name: str
    value: Any
    min_value: Any = None
    max_value: Any = None
    mutation_rate: float = 0.1


@dataclass
class TradingStrategy:
    """استراتيجية تداول"""
    id: str
    name: str
    type: StrategyType
    description: str
    
    # المعاملات
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # شروط الدخول والخروج
    entry_conditions: List[Dict[str, Any]] = field(default_factory=list)
    exit_conditions: List[Dict[str, Any]] = field(default_factory=list)
    
    # إدارة المخاطر
    risk_management: Dict[str, Any] = field(default_factory=dict)
    
    # الأداء
    performance: Dict[str, float] = field(default_factory=dict)
    
    # الجينات (للتطور)
    genes: List[StrategyGene] = field(default_factory=list)
    
    # معلومات إضافية
    created_at: datetime = field(default_factory=datetime.utcnow)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    fitness_score: float = 0.0


@dataclass
class BacktestResult:
    """نتيجة الاختبار الخلفي"""
    strategy_id: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    avg_trade_duration: float
    period: str


class EnhancedStrategyInventor:
    """
    مخترع الاستراتيجيات المحسن.
    
    يوفر:
    - توليد استراتيجيات جديدة بالذكاء الاصطناعي
    - اختبار تلقائي للاستراتيجيات
    - تطور جيني للاستراتيجيات (Genetic Algorithm)
    - دمج استراتيجيات ناجحة
    """
    
    def __init__(self, llm_client=None, backtester=None):
        self.logger = logging.getLogger("EnhancedStrategyInventor")
        self.llm_client = llm_client
        self.backtester = backtester
        
        # مكتبة الاستراتيجيات
        self.strategy_library: Dict[str, TradingStrategy] = {}
        
        # قوالب الاستراتيجيات
        self.strategy_templates = self._init_templates()
        
        # المؤشرات المتاحة
        self.available_indicators = self._init_indicators()
        
        # إعدادات التطور الجيني
        self.genetic_config = {
            "population_size": 20,
            "elite_count": 4,
            "mutation_rate": 0.15,
            "crossover_rate": 0.7,
            "tournament_size": 3,
            "max_generations": 50
        }
        
        # تاريخ التطور
        self.evolution_history: List[Dict[str, Any]] = []
        
        # أفضل الاستراتيجيات
        self.hall_of_fame: List[TradingStrategy] = []
    
    def _init_templates(self) -> Dict[StrategyType, Dict[str, Any]]:
        """تهيئة قوالب الاستراتيجيات."""
        return {
            StrategyType.TREND_FOLLOWING: {
                "name": "تتبع الاتجاه",
                "indicators": ["ema", "sma", "adx"],
                "entry_logic": "price > ema AND adx > 25",
                "exit_logic": "price < ema OR adx < 20",
                "default_params": {
                    "ema_period": 20,
                    "sma_period": 50,
                    "adx_period": 14,
                    "adx_threshold": 25
                }
            },
            StrategyType.MEAN_REVERSION: {
                "name": "العودة للمتوسط",
                "indicators": ["bollinger", "rsi", "sma"],
                "entry_logic": "price < lower_band AND rsi < 30",
                "exit_logic": "price > sma OR rsi > 70",
                "default_params": {
                    "bb_period": 20,
                    "bb_std": 2,
                    "rsi_period": 14,
                    "rsi_oversold": 30,
                    "rsi_overbought": 70
                }
            },
            StrategyType.MOMENTUM: {
                "name": "الزخم",
                "indicators": ["rsi", "macd", "roc"],
                "entry_logic": "rsi > 50 AND macd > signal AND roc > 0",
                "exit_logic": "rsi < 50 OR macd < signal",
                "default_params": {
                    "rsi_period": 14,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "roc_period": 10
                }
            },
            StrategyType.BREAKOUT: {
                "name": "الاختراق",
                "indicators": ["atr", "donchian", "volume"],
                "entry_logic": "price > upper_channel AND volume > avg_volume * 1.5",
                "exit_logic": "price < lower_channel OR trailing_stop",
                "default_params": {
                    "channel_period": 20,
                    "atr_period": 14,
                    "volume_multiplier": 1.5,
                    "trailing_atr": 2
                }
            },
            StrategyType.SCALPING: {
                "name": "المضاربة السريعة",
                "indicators": ["ema", "vwap", "spread"],
                "entry_logic": "price near vwap AND spread < threshold",
                "exit_logic": "profit > target OR loss > stop",
                "default_params": {
                    "ema_period": 9,
                    "profit_target": 0.002,
                    "stop_loss": 0.001,
                    "max_hold_time": 300
                }
            },
            StrategyType.SWING: {
                "name": "التداول المتأرجح",
                "indicators": ["sma", "macd", "support_resistance"],
                "entry_logic": "price at support AND macd bullish",
                "exit_logic": "price at resistance OR macd bearish",
                "default_params": {
                    "sma_period": 50,
                    "sr_lookback": 100,
                    "min_swing": 0.03
                }
            }
        }
    
    def _init_indicators(self) -> Dict[str, Dict[str, Any]]:
        """تهيئة المؤشرات المتاحة."""
        return {
            "sma": {"name": "المتوسط البسيط", "params": ["period"]},
            "ema": {"name": "المتوسط الأسي", "params": ["period"]},
            "rsi": {"name": "مؤشر القوة النسبية", "params": ["period"]},
            "macd": {"name": "MACD", "params": ["fast", "slow", "signal"]},
            "bollinger": {"name": "بولينجر", "params": ["period", "std"]},
            "atr": {"name": "متوسط المدى الحقيقي", "params": ["period"]},
            "adx": {"name": "مؤشر الاتجاه", "params": ["period"]},
            "stochastic": {"name": "ستوكاستيك", "params": ["k_period", "d_period"]},
            "vwap": {"name": "متوسط السعر المرجح بالحجم", "params": []},
            "obv": {"name": "حجم التوازن", "params": []},
            "ichimoku": {"name": "إيشيموكو", "params": ["tenkan", "kijun", "senkou"]},
            "fibonacci": {"name": "فيبوناتشي", "params": ["lookback"]},
            "pivot": {"name": "نقاط المحور", "params": ["type"]},
            "volume_profile": {"name": "ملف الحجم", "params": ["bins"]}
        }
    
    async def generate_strategy(self,
                               strategy_type: StrategyType = None,
                               market_conditions: Dict[str, Any] = None,
                               constraints: Dict[str, Any] = None) -> TradingStrategy:
        """
        توليد استراتيجية جديدة.
        
        Args:
            strategy_type: نوع الاستراتيجية (اختياري)
            market_conditions: ظروف السوق الحالية
            constraints: قيود على الاستراتيجية
            
        Returns:
            استراتيجية جديدة
        """
        # اختيار نوع الاستراتيجية
        if not strategy_type:
            strategy_type = await self._select_best_type(market_conditions)
        
        # الحصول على القالب
        template = self.strategy_templates.get(strategy_type)
        
        # توليد معرف فريد
        strategy_id = hashlib.md5(
            f"{datetime.utcnow().isoformat()}_{random.random()}".encode()
        ).hexdigest()[:12]
        
        # توليد الاستراتيجية
        strategy = TradingStrategy(
            id=strategy_id,
            name=f"{template['name']}_{strategy_id[:6]}",
            type=strategy_type,
            description=f"استراتيجية {template['name']} مولدة تلقائياً"
        )
        
        # تعيين المعاملات
        strategy.parameters = self._generate_parameters(template, constraints)
        
        # توليد شروط الدخول
        strategy.entry_conditions = await self._generate_entry_conditions(
            strategy_type, template, market_conditions
        )
        
        # توليد شروط الخروج
        strategy.exit_conditions = await self._generate_exit_conditions(
            strategy_type, template
        )
        
        # إعداد إدارة المخاطر
        strategy.risk_management = self._generate_risk_management(constraints)
        
        # تحويل لجينات
        strategy.genes = self._parameters_to_genes(strategy.parameters)
        
        # حفظ في المكتبة
        self.strategy_library[strategy_id] = strategy
        
        self.logger.info(f"تم توليد استراتيجية جديدة: {strategy.name}")
        
        return strategy
    
    async def _select_best_type(self, 
                               market_conditions: Dict[str, Any]) -> StrategyType:
        """اختيار أفضل نوع استراتيجية للظروف الحالية."""
        if not market_conditions:
            return random.choice(list(StrategyType))
        
        volatility = market_conditions.get("volatility", 0.02)
        trend_strength = market_conditions.get("trend_strength", 0)
        volume = market_conditions.get("volume_ratio", 1)
        
        # قواعد الاختيار
        if trend_strength > 0.6:
            return StrategyType.TREND_FOLLOWING
        elif volatility > 0.05:
            return StrategyType.BREAKOUT
        elif volatility < 0.01:
            return StrategyType.SCALPING
        elif abs(trend_strength) < 0.2:
            return StrategyType.MEAN_REVERSION
        else:
            return StrategyType.MOMENTUM
    
    def _generate_parameters(self,
                            template: Dict[str, Any],
                            constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """توليد معاملات الاستراتيجية."""
        params = copy.deepcopy(template.get("default_params", {}))
        
        # تعديل عشوائي للمعاملات
        for key, value in params.items():
            if isinstance(value, (int, float)):
                # تغيير بنسبة ±20%
                variation = random.uniform(0.8, 1.2)
                new_value = value * variation
                
                if isinstance(value, int):
                    new_value = int(new_value)
                
                params[key] = new_value
        
        # تطبيق القيود
        if constraints:
            for key, constraint in constraints.items():
                if key in params:
                    if "min" in constraint:
                        params[key] = max(params[key], constraint["min"])
                    if "max" in constraint:
                        params[key] = min(params[key], constraint["max"])
        
        return params
    
    async def _generate_entry_conditions(self,
                                        strategy_type: StrategyType,
                                        template: Dict[str, Any],
                                        market_conditions: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """توليد شروط الدخول."""
        conditions = []
        
        # شروط أساسية من القالب
        indicators = template.get("indicators", [])
        
        for indicator in indicators:
            condition = self._create_indicator_condition(indicator, "entry")
            if condition:
                conditions.append(condition)
        
        # شروط إضافية حسب نوع الاستراتيجية
        if strategy_type == StrategyType.TREND_FOLLOWING:
            conditions.append({
                "type": "trend",
                "indicator": "adx",
                "operator": ">",
                "value": 25,
                "description": "قوة الاتجاه كافية"
            })
        elif strategy_type == StrategyType.MEAN_REVERSION:
            conditions.append({
                "type": "oversold",
                "indicator": "rsi",
                "operator": "<",
                "value": 30,
                "description": "منطقة تشبع بيعي"
            })
        
        # شرط الحجم
        conditions.append({
            "type": "volume",
            "indicator": "volume_ratio",
            "operator": ">",
            "value": 0.8,
            "description": "حجم تداول كافٍ"
        })
        
        return conditions
    
    async def _generate_exit_conditions(self,
                                       strategy_type: StrategyType,
                                       template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """توليد شروط الخروج."""
        conditions = []
        
        # وقف الخسارة
        conditions.append({
            "type": "stop_loss",
            "method": "percentage",
            "value": 0.02,
            "description": "وقف خسارة 2%"
        })
        
        # جني الأرباح
        conditions.append({
            "type": "take_profit",
            "method": "percentage",
            "value": 0.04,
            "description": "جني أرباح 4%"
        })
        
        # وقف متحرك
        conditions.append({
            "type": "trailing_stop",
            "method": "atr",
            "multiplier": 2,
            "description": "وقف متحرك 2 ATR"
        })
        
        # شروط خاصة بالنوع
        if strategy_type == StrategyType.TREND_FOLLOWING:
            conditions.append({
                "type": "trend_reversal",
                "indicator": "adx",
                "operator": "<",
                "value": 20,
                "description": "ضعف الاتجاه"
            })
        
        # حد زمني
        conditions.append({
            "type": "time_limit",
            "max_bars": 100,
            "description": "حد أقصى للاحتفاظ"
        })
        
        return conditions
    
    def _generate_risk_management(self,
                                 constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """توليد إعدادات إدارة المخاطر."""
        risk_mgmt = {
            "max_position_size": 0.1,  # 10% من المحفظة
            "max_risk_per_trade": 0.02,  # 2% مخاطرة لكل صفقة
            "max_daily_loss": 0.05,  # 5% خسارة يومية قصوى
            "max_drawdown": 0.15,  # 15% سحب أقصى
            "position_sizing": "kelly",  # طريقة تحديد الحجم
            "correlation_limit": 0.7,  # حد الارتباط
            "max_open_positions": 5
        }
        
        if constraints:
            risk_mgmt.update(constraints.get("risk", {}))
        
        return risk_mgmt
    
    def _create_indicator_condition(self,
                                   indicator: str,
                                   condition_type: str) -> Optional[Dict[str, Any]]:
        """إنشاء شرط مؤشر."""
        indicator_info = self.available_indicators.get(indicator)
        if not indicator_info:
            return None
        
        return {
            "type": "indicator",
            "indicator": indicator,
            "name": indicator_info["name"],
            "params": indicator_info["params"]
        }
    
    def _parameters_to_genes(self, parameters: Dict[str, Any]) -> List[StrategyGene]:
        """تحويل المعاملات لجينات."""
        genes = []
        
        for name, value in parameters.items():
            if isinstance(value, (int, float)):
                gene = StrategyGene(
                    name=name,
                    value=value,
                    min_value=value * 0.5,
                    max_value=value * 2,
                    mutation_rate=0.1
                )
                genes.append(gene)
        
        return genes
    
    async def evolve_strategies(self,
                               initial_population: List[TradingStrategy] = None,
                               generations: int = None,
                               fitness_function: Callable = None) -> List[TradingStrategy]:
        """
        تطوير الاستراتيجيات باستخدام الخوارزمية الجينية.
        
        Args:
            initial_population: المجتمع الأولي
            generations: عدد الأجيال
            fitness_function: دالة اللياقة
            
        Returns:
            أفضل الاستراتيجيات
        """
        generations = generations or self.genetic_config["max_generations"]
        population_size = self.genetic_config["population_size"]
        
        # إنشاء المجتمع الأولي
        if initial_population:
            population = initial_population
        else:
            population = []
            for _ in range(population_size):
                strategy = await self.generate_strategy()
                population.append(strategy)
        
        self.logger.info(f"بدء التطور الجيني: {len(population)} استراتيجية، {generations} جيل")
        
        for gen in range(generations):
            # تقييم اللياقة
            for strategy in population:
                if fitness_function:
                    strategy.fitness_score = await fitness_function(strategy)
                else:
                    strategy.fitness_score = await self._default_fitness(strategy)
            
            # ترتيب حسب اللياقة
            population.sort(key=lambda s: s.fitness_score, reverse=True)
            
            # حفظ تاريخ التطور
            self.evolution_history.append({
                "generation": gen,
                "best_fitness": population[0].fitness_score,
                "avg_fitness": sum(s.fitness_score for s in population) / len(population),
                "best_strategy": population[0].id
            })
            
            # تحديث قاعة الشرف
            self._update_hall_of_fame(population[0])
            
            self.logger.debug(
                f"الجيل {gen}: أفضل لياقة = {population[0].fitness_score:.4f}"
            )
            
            # إنشاء الجيل التالي
            if gen < generations - 1:
                population = await self._create_next_generation(population)
        
        return population[:self.genetic_config["elite_count"]]
    
    async def _default_fitness(self, strategy: TradingStrategy) -> float:
        """دالة اللياقة الافتراضية."""
        # اختبار خلفي إن وجد
        if self.backtester:
            result = await self.backtester.test(strategy)
            
            # حساب اللياقة المركبة
            fitness = (
                result.sharpe_ratio * 0.3 +
                result.win_rate * 0.2 +
                result.profit_factor * 0.2 +
                (1 - result.max_drawdown) * 0.3
            )
            
            strategy.performance = {
                "sharpe_ratio": result.sharpe_ratio,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "max_drawdown": result.max_drawdown
            }
            
            return fitness
        
        # لياقة عشوائية للاختبار
        return random.uniform(0, 1)
    
    async def _create_next_generation(self,
                                     population: List[TradingStrategy]) -> List[TradingStrategy]:
        """إنشاء الجيل التالي."""
        new_population = []
        elite_count = self.genetic_config["elite_count"]
        
        # الحفاظ على النخبة
        for i in range(elite_count):
            elite = copy.deepcopy(population[i])
            elite.generation += 1
            new_population.append(elite)
        
        # التزاوج والطفرة
        while len(new_population) < len(population):
            # اختيار الآباء
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # التزاوج
            if random.random() < self.genetic_config["crossover_rate"]:
                child = await self._crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)
            
            # الطفرة
            child = await self._mutate(child)
            
            child.generation = population[0].generation + 1
            child.parent_ids = [parent1.id, parent2.id]
            child.id = hashlib.md5(
                f"{datetime.utcnow().isoformat()}_{random.random()}".encode()
            ).hexdigest()[:12]
            
            new_population.append(child)
        
        return new_population
    
    def _tournament_selection(self, 
                             population: List[TradingStrategy]) -> TradingStrategy:
        """اختيار بالبطولة."""
        tournament_size = self.genetic_config["tournament_size"]
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda s: s.fitness_score)
    
    async def _crossover(self,
                        parent1: TradingStrategy,
                        parent2: TradingStrategy) -> TradingStrategy:
        """التزاوج بين استراتيجيتين."""
        child = copy.deepcopy(parent1)
        
        # تبادل الجينات
        for i, gene in enumerate(child.genes):
            if i < len(parent2.genes) and random.random() < 0.5:
                child.genes[i].value = parent2.genes[i].value
        
        # تحديث المعاملات من الجينات
        child.parameters = {
            gene.name: gene.value for gene in child.genes
        }
        
        # مزج شروط الدخول
        if random.random() < 0.5:
            child.entry_conditions = parent2.entry_conditions
        
        # مزج شروط الخروج
        if random.random() < 0.5:
            child.exit_conditions = parent2.exit_conditions
        
        return child
    
    async def _mutate(self, strategy: TradingStrategy) -> TradingStrategy:
        """تطبيق الطفرة."""
        mutation_rate = self.genetic_config["mutation_rate"]
        
        for gene in strategy.genes:
            if random.random() < mutation_rate:
                # طفرة عشوائية
                if gene.min_value is not None and gene.max_value is not None:
                    gene.value = random.uniform(gene.min_value, gene.max_value)
                    
                    if isinstance(strategy.parameters.get(gene.name), int):
                        gene.value = int(gene.value)
        
        # تحديث المعاملات
        strategy.parameters = {
            gene.name: gene.value for gene in strategy.genes
        }
        
        return strategy
    
    def _update_hall_of_fame(self, strategy: TradingStrategy):
        """تحديث قاعة الشرف."""
        # إضافة للقاعة إذا كان من الأفضل
        self.hall_of_fame.append(copy.deepcopy(strategy))
        
        # الاحتفاظ بأفضل 10 فقط
        self.hall_of_fame.sort(key=lambda s: s.fitness_score, reverse=True)
        self.hall_of_fame = self.hall_of_fame[:10]
    
    async def combine_strategies(self,
                                strategies: List[TradingStrategy],
                                weights: List[float] = None) -> TradingStrategy:
        """
        دمج عدة استراتيجيات في استراتيجية هجينة.
        
        Args:
            strategies: الاستراتيجيات للدمج
            weights: أوزان كل استراتيجية
            
        Returns:
            استراتيجية هجينة
        """
        if not strategies:
            raise ValueError("يجب توفير استراتيجية واحدة على الأقل")
        
        if not weights:
            weights = [1.0 / len(strategies)] * len(strategies)
        
        # إنشاء استراتيجية هجينة
        hybrid_id = hashlib.md5(
            f"hybrid_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        
        hybrid = TradingStrategy(
            id=hybrid_id,
            name=f"Hybrid_{hybrid_id[:6]}",
            type=StrategyType.HYBRID,
            description=f"استراتيجية هجينة من {len(strategies)} استراتيجيات"
        )
        
        # دمج المعاملات (متوسط مرجح)
        all_params = {}
        for strategy, weight in zip(strategies, weights):
            for key, value in strategy.parameters.items():
                if isinstance(value, (int, float)):
                    if key not in all_params:
                        all_params[key] = 0
                    all_params[key] += value * weight
        
        hybrid.parameters = all_params
        
        # دمج شروط الدخول (اتحاد)
        seen_conditions = set()
        for strategy in strategies:
            for condition in strategy.entry_conditions:
                cond_key = json.dumps(condition, sort_keys=True)
                if cond_key not in seen_conditions:
                    hybrid.entry_conditions.append(condition)
                    seen_conditions.add(cond_key)
        
        # دمج شروط الخروج
        seen_exits = set()
        for strategy in strategies:
            for condition in strategy.exit_conditions:
                cond_key = json.dumps(condition, sort_keys=True)
                if cond_key not in seen_exits:
                    hybrid.exit_conditions.append(condition)
                    seen_exits.add(cond_key)
        
        # إدارة المخاطر (الأكثر تحفظاً)
        hybrid.risk_management = {
            "max_position_size": min(s.risk_management.get("max_position_size", 0.1) for s in strategies),
            "max_risk_per_trade": min(s.risk_management.get("max_risk_per_trade", 0.02) for s in strategies),
            "max_daily_loss": min(s.risk_management.get("max_daily_loss", 0.05) for s in strategies),
            "max_drawdown": min(s.risk_management.get("max_drawdown", 0.15) for s in strategies)
        }
        
        hybrid.parent_ids = [s.id for s in strategies]
        
        # حفظ في المكتبة
        self.strategy_library[hybrid_id] = hybrid
        
        self.logger.info(f"تم إنشاء استراتيجية هجينة: {hybrid.name}")
        
        return hybrid
    
    async def optimize_strategy(self,
                               strategy: TradingStrategy,
                               optimization_target: str = "sharpe_ratio",
                               iterations: int = 100) -> TradingStrategy:
        """
        تحسين استراتيجية موجودة.
        
        Args:
            strategy: الاستراتيجية للتحسين
            optimization_target: الهدف (sharpe_ratio, win_rate, etc.)
            iterations: عدد التكرارات
            
        Returns:
            الاستراتيجية المحسنة
        """
        best_strategy = copy.deepcopy(strategy)
        best_score = await self._evaluate_strategy(best_strategy, optimization_target)
        
        self.logger.info(f"بدء تحسين الاستراتيجية: {strategy.name}")
        
        for i in range(iterations):
            # إنشاء نسخة معدلة
            candidate = copy.deepcopy(best_strategy)
            candidate = await self._mutate(candidate)
            
            # تقييم
            score = await self._evaluate_strategy(candidate, optimization_target)
            
            # تحديث إذا كان أفضل
            if score > best_score:
                best_strategy = candidate
                best_score = score
                self.logger.debug(f"تحسين {i}: {optimization_target} = {score:.4f}")
        
        best_strategy.name = f"{strategy.name}_optimized"
        
        return best_strategy
    
    async def _evaluate_strategy(self,
                                strategy: TradingStrategy,
                                target: str) -> float:
        """تقييم استراتيجية."""
        if self.backtester:
            result = await self.backtester.test(strategy)
            return getattr(result, target, 0)
        
        return random.uniform(0, 1)
    
    def get_best_strategies(self, count: int = 5) -> List[TradingStrategy]:
        """الحصول على أفضل الاستراتيجيات."""
        all_strategies = list(self.strategy_library.values())
        all_strategies.sort(key=lambda s: s.fitness_score, reverse=True)
        return all_strategies[:count]
    
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على الإحصائيات."""
        return {
            "total_strategies": len(self.strategy_library),
            "hall_of_fame_size": len(self.hall_of_fame),
            "evolution_generations": len(self.evolution_history),
            "best_fitness": self.hall_of_fame[0].fitness_score if self.hall_of_fame else 0,
            "strategy_types": {
                st.value: sum(1 for s in self.strategy_library.values() if s.type == st)
                for st in StrategyType
            }
        }
