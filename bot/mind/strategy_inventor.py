"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Strategy Inventor
مبتكر الاستراتيجيات
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random
import json
from loguru import logger


class StrategyType(Enum):
    """أنواع الاستراتيجيات"""
    TREND_FOLLOWING = "تتبع الاتجاه"
    MEAN_REVERSION = "العودة للمتوسط"
    BREAKOUT = "الاختراق"
    MOMENTUM = "الزخم"
    SCALPING = "المضاربة السريعة"
    SWING = "التداول المتأرجح"
    HYBRID = "هجين"


@dataclass
class TradingRule:
    """قاعدة تداول"""
    name: str
    condition: str
    action: str
    parameters: Dict[str, Any]
    weight: float = 1.0


@dataclass
class Strategy:
    """استراتيجية تداول"""
    id: str
    name: str
    type: StrategyType
    description: str
    entry_rules: List[TradingRule]
    exit_rules: List[TradingRule]
    risk_rules: List[TradingRule]
    parameters: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    performance: Dict[str, float] = field(default_factory=dict)
    status: str = "draft"  # draft, testing, active, retired
    version: int = 1


@dataclass
class BacktestResult:
    """نتيجة الاختبار التاريخي"""
    strategy_id: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Dict]
    period: str


class StrategyInventor:
    """
    مبتكر الاستراتيجيات
    
    يقوم بـ:
    - ابتكار استراتيجيات جديدة
    - تطوير الاستراتيجيات الموجودة
    - اختبار الاستراتيجيات
    - تقييم الأداء
    """
    
    def __init__(self):
        """تهيئة مبتكر الاستراتيجيات"""
        self.strategies: Dict[str, Strategy] = {}
        self.backtest_results: Dict[str, List[BacktestResult]] = {}
        self.strategy_counter = 0
        
        # مكتبة القواعد الأساسية
        self._init_rule_library()
        
        # مكتبة المعاملات
        self._init_parameter_library()
        
        logger.info("💡 StrategyInventor initialized")
    
    def _init_rule_library(self):
        """تهيئة مكتبة القواعد"""
        self.entry_rule_templates = [
            # قواعد الاتجاه
            {
                'name': 'MA_Crossover',
                'condition': 'ema_fast > ema_slow',
                'params': {'fast_period': [9, 12, 21], 'slow_period': [21, 26, 50]}
            },
            {
                'name': 'Price_Above_MA',
                'condition': 'close > sma_period',
                'params': {'period': [20, 50, 100, 200]}
            },
            # قواعد الزخم
            {
                'name': 'RSI_Oversold',
                'condition': 'rsi < threshold',
                'params': {'period': [7, 14, 21], 'threshold': [25, 30, 35]}
            },
            {
                'name': 'MACD_Bullish',
                'condition': 'macd > signal AND macd_prev < signal_prev',
                'params': {'fast': [12], 'slow': [26], 'signal': [9]}
            },
            # قواعد الاختراق
            {
                'name': 'Breakout_High',
                'condition': 'close > highest_high_period',
                'params': {'period': [10, 20, 50]}
            },
            {
                'name': 'BB_Squeeze_Break',
                'condition': 'bb_width < threshold AND close > bb_upper',
                'params': {'period': [20], 'std': [2], 'threshold': [0.1, 0.15, 0.2]}
            },
            # قواعد الحجم
            {
                'name': 'Volume_Spike',
                'condition': 'volume > volume_sma * multiplier',
                'params': {'period': [20], 'multiplier': [1.5, 2.0, 2.5]}
            },
            # قواعد متقدمة
            {
                'name': 'Divergence_RSI',
                'condition': 'price_lower_low AND rsi_higher_low',
                'params': {'lookback': [10, 20, 30]}
            },
            {
                'name': 'Support_Bounce',
                'condition': 'close > support_level AND prev_close < support_level',
                'params': {'lookback': [20, 50, 100]}
            }
        ]
        
        self.exit_rule_templates = [
            {
                'name': 'Take_Profit_Percent',
                'condition': 'profit_percent >= target',
                'params': {'target': [1.5, 2.0, 3.0, 5.0]}
            },
            {
                'name': 'Stop_Loss_Percent',
                'condition': 'loss_percent >= threshold',
                'params': {'threshold': [1.0, 1.5, 2.0, 3.0]}
            },
            {
                'name': 'Trailing_Stop',
                'condition': 'price < highest_since_entry * (1 - trail_percent)',
                'params': {'trail_percent': [0.01, 0.02, 0.03]}
            },
            {
                'name': 'RSI_Overbought',
                'condition': 'rsi > threshold',
                'params': {'threshold': [65, 70, 75, 80]}
            },
            {
                'name': 'MA_Cross_Exit',
                'condition': 'ema_fast < ema_slow',
                'params': {'fast_period': [9, 12], 'slow_period': [21, 26]}
            },
            {
                'name': 'Time_Based_Exit',
                'condition': 'bars_since_entry >= max_bars',
                'params': {'max_bars': [10, 20, 50, 100]}
            }
        ]
        
        self.risk_rule_templates = [
            {
                'name': 'Max_Position_Size',
                'condition': 'position_size <= max_percent',
                'params': {'max_percent': [5, 10, 15, 20]}
            },
            {
                'name': 'Max_Daily_Loss',
                'condition': 'daily_loss < max_loss',
                'params': {'max_loss': [3, 5, 7, 10]}
            },
            {
                'name': 'Max_Open_Positions',
                'condition': 'open_positions <= max_positions',
                'params': {'max_positions': [3, 5, 7, 10]}
            },
            {
                'name': 'Volatility_Filter',
                'condition': 'atr_percent < max_volatility',
                'params': {'max_volatility': [3, 5, 7]}
            }
        ]
    
    def _init_parameter_library(self):
        """تهيئة مكتبة المعاملات"""
        self.parameter_ranges = {
            'rsi_period': (7, 21),
            'rsi_oversold': (20, 35),
            'rsi_overbought': (65, 80),
            'ma_fast': (5, 21),
            'ma_slow': (20, 100),
            'bb_period': (15, 30),
            'bb_std': (1.5, 2.5),
            'atr_period': (10, 21),
            'stop_loss_percent': (1.0, 3.0),
            'take_profit_percent': (1.5, 6.0),
            'trailing_stop_percent': (1.0, 3.0),
            'position_size_percent': (5.0, 15.0)
        }
    
    # ═══════════════════════════════════════════════════════════════
    # STRATEGY INVENTION
    # ═══════════════════════════════════════════════════════════════
    
    def invent_strategy(
        self,
        strategy_type: Optional[StrategyType] = None,
        market_condition: Optional[str] = None,
        risk_level: str = "medium"
    ) -> Strategy:
        """
        ابتكار استراتيجية جديدة
        
        Args:
            strategy_type: نوع الاستراتيجية
            market_condition: حالة السوق
            risk_level: مستوى المخاطر
            
        Returns:
            الاستراتيجية الجديدة
        """
        self.strategy_counter += 1
        strategy_id = f"STR_{self.strategy_counter:04d}"
        
        # اختيار نوع الاستراتيجية
        if strategy_type is None:
            strategy_type = random.choice(list(StrategyType))
        
        # اختيار القواعد بناءً على النوع
        entry_rules = self._select_entry_rules(strategy_type, market_condition)
        exit_rules = self._select_exit_rules(strategy_type, risk_level)
        risk_rules = self._select_risk_rules(risk_level)
        
        # توليد المعاملات
        parameters = self._generate_parameters(strategy_type, risk_level)
        
        # إنشاء الاستراتيجية
        strategy = Strategy(
            id=strategy_id,
            name=f"{strategy_type.value} Strategy #{self.strategy_counter}",
            type=strategy_type,
            description=self._generate_description(strategy_type, entry_rules, exit_rules),
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            risk_rules=risk_rules,
            parameters=parameters
        )
        
        self.strategies[strategy_id] = strategy
        logger.info(f"💡 Invented new strategy: {strategy.name}")
        
        return strategy
    
    def _select_entry_rules(
        self,
        strategy_type: StrategyType,
        market_condition: Optional[str]
    ) -> List[TradingRule]:
        """اختيار قواعد الدخول"""
        rules = []
        
        # قواعد حسب نوع الاستراتيجية
        type_rules = {
            StrategyType.TREND_FOLLOWING: ['MA_Crossover', 'Price_Above_MA'],
            StrategyType.MEAN_REVERSION: ['RSI_Oversold', 'BB_Squeeze_Break'],
            StrategyType.BREAKOUT: ['Breakout_High', 'Volume_Spike'],
            StrategyType.MOMENTUM: ['MACD_Bullish', 'RSI_Oversold'],
            StrategyType.SCALPING: ['RSI_Oversold', 'Volume_Spike'],
            StrategyType.SWING: ['MA_Crossover', 'Support_Bounce'],
            StrategyType.HYBRID: ['MA_Crossover', 'RSI_Oversold', 'Volume_Spike']
        }
        
        preferred_rules = type_rules.get(strategy_type, ['MA_Crossover'])
        
        for template in self.entry_rule_templates:
            if template['name'] in preferred_rules:
                # اختيار معاملات عشوائية
                params = {}
                for param_name, param_values in template['params'].items():
                    params[param_name] = random.choice(param_values)
                
                rule = TradingRule(
                    name=template['name'],
                    condition=template['condition'],
                    action='BUY',
                    parameters=params
                )
                rules.append(rule)
        
        return rules
    
    def _select_exit_rules(
        self,
        strategy_type: StrategyType,
        risk_level: str
    ) -> List[TradingRule]:
        """اختيار قواعد الخروج"""
        rules = []
        
        # دائماً نضيف وقف الخسارة وجني الأرباح
        risk_params = {
            'low': {'stop_loss': 3.0, 'take_profit': 2.0, 'trailing': 0.03},
            'medium': {'stop_loss': 2.0, 'take_profit': 3.5, 'trailing': 0.02},
            'high': {'stop_loss': 1.5, 'take_profit': 5.0, 'trailing': 0.015}
        }
        
        params = risk_params.get(risk_level, risk_params['medium'])
        
        # وقف الخسارة
        rules.append(TradingRule(
            name='Stop_Loss',
            condition='loss_percent >= threshold',
            action='SELL',
            parameters={'threshold': params['stop_loss']}
        ))
        
        # جني الأرباح
        rules.append(TradingRule(
            name='Take_Profit',
            condition='profit_percent >= target',
            action='SELL',
            parameters={'target': params['take_profit']}
        ))
        
        # وقف متحرك
        rules.append(TradingRule(
            name='Trailing_Stop',
            condition='price < highest_since_entry * (1 - trail_percent)',
            action='SELL',
            parameters={'trail_percent': params['trailing']}
        ))
        
        # قواعد إضافية حسب النوع
        if strategy_type in [StrategyType.MOMENTUM, StrategyType.SCALPING]:
            rules.append(TradingRule(
                name='RSI_Exit',
                condition='rsi > threshold',
                action='SELL',
                parameters={'threshold': 70}
            ))
        
        return rules
    
    def _select_risk_rules(self, risk_level: str) -> List[TradingRule]:
        """اختيار قواعد المخاطر"""
        rules = []
        
        risk_params = {
            'low': {'max_position': 10, 'max_daily_loss': 3, 'max_positions': 3},
            'medium': {'max_position': 15, 'max_daily_loss': 5, 'max_positions': 5},
            'high': {'max_position': 20, 'max_daily_loss': 7, 'max_positions': 7}
        }
        
        params = risk_params.get(risk_level, risk_params['medium'])
        
        rules.append(TradingRule(
            name='Position_Size_Limit',
            condition='position_size <= max_percent',
            action='LIMIT',
            parameters={'max_percent': params['max_position']}
        ))
        
        rules.append(TradingRule(
            name='Daily_Loss_Limit',
            condition='daily_loss < max_loss',
            action='STOP_TRADING',
            parameters={'max_loss': params['max_daily_loss']}
        ))
        
        rules.append(TradingRule(
            name='Max_Positions',
            condition='open_positions <= max_positions',
            action='LIMIT',
            parameters={'max_positions': params['max_positions']}
        ))
        
        return rules
    
    def _generate_parameters(
        self,
        strategy_type: StrategyType,
        risk_level: str
    ) -> Dict[str, Any]:
        """توليد معاملات الاستراتيجية"""
        params = {}
        
        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            if isinstance(min_val, int):
                params[param_name] = random.randint(min_val, max_val)
            else:
                params[param_name] = round(random.uniform(min_val, max_val), 2)
        
        # تعديل حسب مستوى المخاطر
        risk_multipliers = {'low': 0.7, 'medium': 1.0, 'high': 1.3}
        multiplier = risk_multipliers.get(risk_level, 1.0)
        
        params['position_size_percent'] *= multiplier
        params['stop_loss_percent'] /= multiplier
        
        return params
    
    def _generate_description(
        self,
        strategy_type: StrategyType,
        entry_rules: List[TradingRule],
        exit_rules: List[TradingRule]
    ) -> str:
        """توليد وصف الاستراتيجية"""
        entry_names = [r.name for r in entry_rules]
        exit_names = [r.name for r in exit_rules]
        
        return (
            f"استراتيجية {strategy_type.value} تعتمد على: "
            f"{', '.join(entry_names)} للدخول، و"
            f"{', '.join(exit_names)} للخروج."
        )
    
    # ═══════════════════════════════════════════════════════════════
    # STRATEGY EVOLUTION
    # ═══════════════════════════════════════════════════════════════
    
    def evolve_strategy(
        self,
        strategy_id: str,
        performance_feedback: Dict[str, float]
    ) -> Strategy:
        """
        تطوير استراتيجية بناءً على الأداء
        
        Args:
            strategy_id: معرف الاستراتيجية
            performance_feedback: ردود فعل الأداء
            
        Returns:
            الاستراتيجية المطورة
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        original = self.strategies[strategy_id]
        
        # إنشاء نسخة جديدة
        self.strategy_counter += 1
        new_id = f"STR_{self.strategy_counter:04d}"
        
        # تحليل الأداء
        win_rate = performance_feedback.get('win_rate', 0.5)
        profit_factor = performance_feedback.get('profit_factor', 1.0)
        max_drawdown = performance_feedback.get('max_drawdown', 0.1)
        
        # تعديل المعاملات
        new_params = self._adjust_parameters(
            original.parameters,
            win_rate,
            profit_factor,
            max_drawdown
        )
        
        # تعديل القواعد إذا لزم الأمر
        new_entry_rules = self._adjust_rules(
            original.entry_rules,
            'entry',
            win_rate
        )
        new_exit_rules = self._adjust_rules(
            original.exit_rules,
            'exit',
            profit_factor
        )
        
        # إنشاء الاستراتيجية المطورة
        evolved = Strategy(
            id=new_id,
            name=f"{original.name} v{original.version + 1}",
            type=original.type,
            description=f"نسخة مطورة من {original.name}",
            entry_rules=new_entry_rules,
            exit_rules=new_exit_rules,
            risk_rules=original.risk_rules.copy(),
            parameters=new_params,
            version=original.version + 1
        )
        
        self.strategies[new_id] = evolved
        logger.info(f"🔄 Evolved strategy: {evolved.name}")
        
        return evolved
    
    def _adjust_parameters(
        self,
        params: Dict[str, Any],
        win_rate: float,
        profit_factor: float,
        max_drawdown: float
    ) -> Dict[str, Any]:
        """تعديل المعاملات بناءً على الأداء"""
        new_params = params.copy()
        
        # إذا كان معدل الفوز منخفض، نشدد على الدخول
        if win_rate < 0.45:
            if 'rsi_oversold' in new_params:
                new_params['rsi_oversold'] = max(20, new_params['rsi_oversold'] - 5)
        
        # إذا كان عامل الربح منخفض، نحسن الخروج
        if profit_factor < 1.2:
            if 'take_profit_percent' in new_params:
                new_params['take_profit_percent'] *= 1.2
            if 'stop_loss_percent' in new_params:
                new_params['stop_loss_percent'] *= 0.9
        
        # إذا كان السحب كبير، نقلل المخاطر
        if max_drawdown > 0.15:
            if 'position_size_percent' in new_params:
                new_params['position_size_percent'] *= 0.8
        
        return new_params
    
    def _adjust_rules(
        self,
        rules: List[TradingRule],
        rule_type: str,
        metric: float
    ) -> List[TradingRule]:
        """تعديل القواعد"""
        new_rules = []
        
        for rule in rules:
            new_rule = TradingRule(
                name=rule.name,
                condition=rule.condition,
                action=rule.action,
                parameters=rule.parameters.copy(),
                weight=rule.weight
            )
            
            # تعديل الأوزان بناءً على الأداء
            if metric < 0.5:
                new_rule.weight *= 0.9
            elif metric > 0.6:
                new_rule.weight *= 1.1
            
            new_rules.append(new_rule)
        
        return new_rules
    
    # ═══════════════════════════════════════════════════════════════
    # BACKTESTING
    # ═══════════════════════════════════════════════════════════════
    
    def backtest_strategy(
        self,
        strategy_id: str,
        data: pd.DataFrame,
        initial_capital: float = 10000.0
    ) -> BacktestResult:
        """
        اختبار الاستراتيجية على بيانات تاريخية
        
        Args:
            strategy_id: معرف الاستراتيجية
            data: البيانات التاريخية
            initial_capital: رأس المال الأولي
            
        Returns:
            نتيجة الاختبار
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        strategy = self.strategies[strategy_id]
        
        # محاكاة التداول
        trades = []
        capital = initial_capital
        position = None
        max_capital = initial_capital
        max_drawdown = 0
        
        for i in range(len(data)):
            row = data.iloc[i]
            
            # التحقق من إشارة الدخول
            if position is None:
                if self._check_entry_signal(strategy, data, i):
                    position = {
                        'entry_price': row['close'],
                        'entry_time': row.name if hasattr(row, 'name') else i,
                        'size': capital * strategy.parameters.get('position_size_percent', 10) / 100
                    }
            
            # التحقق من إشارة الخروج
            elif position is not None:
                exit_signal, exit_reason = self._check_exit_signal(
                    strategy, data, i, position
                )
                
                if exit_signal:
                    exit_price = row['close']
                    pnl = (exit_price - position['entry_price']) / position['entry_price']
                    profit = position['size'] * pnl
                    capital += profit
                    
                    trades.append({
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl_percent': pnl * 100,
                        'profit': profit,
                        'exit_reason': exit_reason
                    })
                    
                    position = None
                    
                    # تحديث السحب الأقصى
                    max_capital = max(max_capital, capital)
                    drawdown = (max_capital - capital) / max_capital
                    max_drawdown = max(max_drawdown, drawdown)
        
        # حساب الإحصائيات
        if trades:
            winning_trades = [t for t in trades if t['pnl_percent'] > 0]
            losing_trades = [t for t in trades if t['pnl_percent'] <= 0]
            
            win_rate = len(winning_trades) / len(trades)
            
            total_profit = sum(t['profit'] for t in winning_trades) if winning_trades else 0
            total_loss = abs(sum(t['profit'] for t in losing_trades)) if losing_trades else 1
            profit_factor = total_profit / total_loss if total_loss > 0 else total_profit
            
            total_return = (capital - initial_capital) / initial_capital
            
            # Sharpe ratio (تقريبي)
            returns = [t['pnl_percent'] for t in trades]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        else:
            win_rate = 0
            profit_factor = 0
            total_return = 0
            sharpe_ratio = 0
        
        result = BacktestResult(
            strategy_id=strategy_id,
            total_trades=len(trades),
            winning_trades=len([t for t in trades if t['pnl_percent'] > 0]),
            losing_trades=len([t for t in trades if t['pnl_percent'] <= 0]),
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            trades=trades,
            period=f"{data.index[0]} to {data.index[-1]}" if hasattr(data.index[0], 'strftime') else "N/A"
        )
        
        # حفظ النتيجة
        if strategy_id not in self.backtest_results:
            self.backtest_results[strategy_id] = []
        self.backtest_results[strategy_id].append(result)
        
        # تحديث أداء الاستراتيجية
        strategy.performance = {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
        
        logger.info(
            f"📊 Backtest completed: {len(trades)} trades, "
            f"Win rate: {win_rate:.1%}, Return: {total_return:.1%}"
        )
        
        return result
    
    def _check_entry_signal(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        index: int
    ) -> bool:
        """التحقق من إشارة الدخول"""
        if index < 50:  # نحتاج بيانات كافية
            return False
        
        row = data.iloc[index]
        
        for rule in strategy.entry_rules:
            if not self._evaluate_rule(rule, data, index):
                return False
        
        return True
    
    def _check_exit_signal(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        index: int,
        position: Dict
    ) -> Tuple[bool, str]:
        """التحقق من إشارة الخروج"""
        row = data.iloc[index]
        current_price = row['close']
        entry_price = position['entry_price']
        
        pnl_percent = (current_price - entry_price) / entry_price * 100
        
        for rule in strategy.exit_rules:
            if 'Stop_Loss' in rule.name:
                threshold = rule.parameters.get('threshold', 2.0)
                if pnl_percent <= -threshold:
                    return True, 'Stop Loss'
            
            elif 'Take_Profit' in rule.name:
                target = rule.parameters.get('target', 3.0)
                if pnl_percent >= target:
                    return True, 'Take Profit'
            
            elif 'Trailing_Stop' in rule.name:
                # تبسيط: نتحقق من الانخفاض من الأعلى
                trail = rule.parameters.get('trail_percent', 0.02)
                highest = data.iloc[index-20:index+1]['high'].max()
                if current_price < highest * (1 - trail):
                    return True, 'Trailing Stop'
        
        return False, ''
    
    def _evaluate_rule(
        self,
        rule: TradingRule,
        data: pd.DataFrame,
        index: int
    ) -> bool:
        """تقييم قاعدة"""
        row = data.iloc[index]
        
        try:
            if 'RSI_Oversold' in rule.name:
                if 'rsi_14' in row:
                    threshold = rule.parameters.get('threshold', 30)
                    return row['rsi_14'] < threshold
            
            elif 'MA_Crossover' in rule.name:
                fast = rule.parameters.get('fast_period', 12)
                slow = rule.parameters.get('slow_period', 26)
                if f'ema_{fast}' in row and f'ema_{slow}' in row:
                    return row[f'ema_{fast}'] > row[f'ema_{slow}']
            
            elif 'Volume_Spike' in rule.name:
                if 'volume' in row and 'volume_sma_20' in row:
                    multiplier = rule.parameters.get('multiplier', 2.0)
                    return row['volume'] > row['volume_sma_20'] * multiplier
            
            elif 'MACD_Bullish' in rule.name:
                if 'macd' in row and 'macd_signal' in row:
                    return row['macd'] > row['macd_signal']
        except Exception:
            pass
        
        return True  # افتراضي
    
    # ═══════════════════════════════════════════════════════════════
    # STRATEGY MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    
    def get_best_strategies(
        self,
        metric: str = 'sharpe_ratio',
        top_n: int = 5
    ) -> List[Strategy]:
        """الحصول على أفضل الاستراتيجيات"""
        strategies_with_performance = [
            s for s in self.strategies.values()
            if s.performance and metric in s.performance
        ]
        
        sorted_strategies = sorted(
            strategies_with_performance,
            key=lambda s: s.performance.get(metric, 0),
            reverse=True
        )
        
        return sorted_strategies[:top_n]
    
    def export_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """تصدير استراتيجية"""
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        strategy = self.strategies[strategy_id]
        
        return {
            'id': strategy.id,
            'name': strategy.name,
            'type': strategy.type.value,
            'description': strategy.description,
            'entry_rules': [
                {
                    'name': r.name,
                    'condition': r.condition,
                    'parameters': r.parameters
                }
                for r in strategy.entry_rules
            ],
            'exit_rules': [
                {
                    'name': r.name,
                    'condition': r.condition,
                    'parameters': r.parameters
                }
                for r in strategy.exit_rules
            ],
            'risk_rules': [
                {
                    'name': r.name,
                    'condition': r.condition,
                    'parameters': r.parameters
                }
                for r in strategy.risk_rules
            ],
            'parameters': strategy.parameters,
            'performance': strategy.performance,
            'version': strategy.version
        }


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار مبتكر الاستراتيجيات
    inventor = StrategyInventor()
    
    # ابتكار استراتيجية
    strategy = inventor.invent_strategy(
        strategy_type=StrategyType.MOMENTUM,
        risk_level='medium'
    )
    
    print(f"Created strategy: {strategy.name}")
    print(f"Type: {strategy.type.value}")
    print(f"Entry rules: {len(strategy.entry_rules)}")
    print(f"Exit rules: {len(strategy.exit_rules)}")
    
    # تصدير
    exported = inventor.export_strategy(strategy.id)
    print(f"\nExported strategy: {json.dumps(exported, indent=2, default=str)[:500]}...")
