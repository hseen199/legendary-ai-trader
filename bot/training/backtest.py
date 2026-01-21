"""
═══════════════════════════════════════════════════════════════
LEGENDARY AGENT - Backtester
نظام الاختبار التاريخي
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger


class TradeAction(Enum):
    """إجراءات التداول"""
    BUY = "شراء"
    SELL = "بيع"
    HOLD = "انتظار"


@dataclass
class Trade:
    """صفقة"""
    symbol: str
    action: TradeAction
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    size: float = 1.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """نتيجة الاختبار"""
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class Backtester:
    """
    نظام الاختبار التاريخي
    
    يختبر استراتيجيات التداول على البيانات التاريخية
    """
    
    def __init__(
        self,
        initial_balance: float = 10000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        config: Dict[str, Any] = None
    ):
        """
        تهيئة المختبر
        
        Args:
            initial_balance: الرصيد الأولي
            commission: العمولة
            slippage: الانزلاق
            config: إعدادات إضافية
        """
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.config = config or {}
        
        # إعدادات التداول
        self.stop_loss = self.config.get('stop_loss', 0.02)
        self.take_profit = self.config.get('take_profit', [0.015, 0.035, 0.06])
        self.max_position_size = self.config.get('max_position_size', 0.15)
        
        # الحالة
        self.balance = initial_balance
        self.equity_curve: List[float] = []
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}
        
        logger.info(f"📊 Backtester initialized with ${initial_balance}")
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN BACKTEST
    # ═══════════════════════════════════════════════════════════════
    
    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        symbol: str = 'BTCUSDT'
    ) -> BacktestResult:
        """
        تشغيل الاختبار التاريخي
        
        Args:
            df: البيانات (يجب أن تحتوي على 'close', 'high', 'low')
            signals: إشارات التداول (-1, 0, 1)
            symbol: رمز العملة
            
        Returns:
            نتيجة الاختبار
        """
        logger.info(f"🔄 Running backtest on {len(df)} candles...")
        
        # إعادة تعيين الحالة
        self.balance = self.initial_balance
        self.equity_curve = [self.initial_balance]
        self.trades = []
        self.open_positions = {}
        
        # التأكد من وجود الأعمدة المطلوبة
        required_cols = ['close', 'high', 'low']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # المرور على البيانات
        for i in range(len(df)):
            row = df.iloc[i]
            timestamp = df.index[i] if isinstance(df.index[i], datetime) else datetime.now()
            signal = signals.iloc[i] if i < len(signals) else 0
            
            current_price = row['close']
            high = row['high']
            low = row['low']
            
            # تحديث المراكز المفتوحة
            self._update_positions(symbol, current_price, high, low, timestamp)
            
            # تنفيذ الإشارات
            if signal == 1 and symbol not in self.open_positions:
                self._open_position(symbol, current_price, timestamp)
            elif signal == -1 and symbol in self.open_positions:
                self._close_position(symbol, current_price, timestamp, "Signal")
            
            # تحديث منحنى الأسهم
            equity = self._calculate_equity(current_price)
            self.equity_curve.append(equity)
        
        # إغلاق المراكز المتبقية
        if self.open_positions:
            final_price = df.iloc[-1]['close']
            final_time = df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()
            for symbol in list(self.open_positions.keys()):
                self._close_position(symbol, final_price, final_time, "End of backtest")
        
        # حساب النتائج
        result = self._calculate_results(df)
        
        logger.info(f"✅ Backtest complete: {result.total_return:.2%} return, {result.win_rate:.1%} win rate")
        
        return result
    
    # ═══════════════════════════════════════════════════════════════
    # POSITION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════
    
    def _open_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime
    ) -> None:
        """فتح مركز"""
        # حساب الحجم
        position_value = self.balance * self.max_position_size
        size = position_value / price
        
        # تطبيق العمولة والانزلاق
        actual_price = price * (1 + self.slippage)
        commission_cost = position_value * self.commission
        
        # إنشاء الصفقة
        trade = Trade(
            symbol=symbol,
            action=TradeAction.BUY,
            entry_time=timestamp,
            entry_price=actual_price,
            size=size
        )
        
        self.open_positions[symbol] = trade
        self.balance -= commission_cost
    
    def _close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        reason: str
    ) -> None:
        """إغلاق مركز"""
        if symbol not in self.open_positions:
            return
        
        trade = self.open_positions[symbol]
        
        # تطبيق الانزلاق
        actual_price = price * (1 - self.slippage)
        
        # حساب الربح/الخسارة
        position_value = trade.size * actual_price
        entry_value = trade.size * trade.entry_price
        commission_cost = position_value * self.commission
        
        pnl = position_value - entry_value - commission_cost
        pnl_percent = (actual_price - trade.entry_price) / trade.entry_price * 100
        
        # تحديث الصفقة
        trade.exit_time = timestamp
        trade.exit_price = actual_price
        trade.pnl = pnl
        trade.pnl_percent = pnl_percent
        trade.exit_reason = reason
        
        # تحديث الرصيد
        self.balance += position_value - commission_cost
        
        # حفظ الصفقة
        self.trades.append(trade)
        del self.open_positions[symbol]
    
    def _update_positions(
        self,
        symbol: str,
        current_price: float,
        high: float,
        low: float,
        timestamp: datetime
    ) -> None:
        """تحديث المراكز (وقف الخسارة وجني الأرباح)"""
        if symbol not in self.open_positions:
            return
        
        trade = self.open_positions[symbol]
        entry_price = trade.entry_price
        
        # حساب التغير
        change_from_entry = (current_price - entry_price) / entry_price
        low_change = (low - entry_price) / entry_price
        high_change = (high - entry_price) / entry_price
        
        # وقف الخسارة
        if low_change <= -self.stop_loss:
            exit_price = entry_price * (1 - self.stop_loss)
            self._close_position(symbol, exit_price, timestamp, "Stop Loss")
            return
        
        # جني الأرباح (المرحلة الأخيرة)
        if isinstance(self.take_profit, list) and len(self.take_profit) > 0:
            final_tp = self.take_profit[-1]
            if high_change >= final_tp:
                exit_price = entry_price * (1 + final_tp)
                self._close_position(symbol, exit_price, timestamp, "Take Profit")
                return
        elif isinstance(self.take_profit, (int, float)):
            if high_change >= self.take_profit:
                exit_price = entry_price * (1 + self.take_profit)
                self._close_position(symbol, exit_price, timestamp, "Take Profit")
                return
    
    def _calculate_equity(self, current_price: float) -> float:
        """حساب الأسهم الحالية"""
        equity = self.balance
        
        for symbol, trade in self.open_positions.items():
            position_value = trade.size * current_price
            equity += position_value
        
        return equity
    
    # ═══════════════════════════════════════════════════════════════
    # RESULTS CALCULATION
    # ═══════════════════════════════════════════════════════════════
    
    def _calculate_results(self, df: pd.DataFrame) -> BacktestResult:
        """حساب النتائج"""
        # التواريخ
        start_date = df.index[0] if isinstance(df.index[0], datetime) else datetime.now()
        end_date = df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()
        
        # الإحصائيات الأساسية
        final_balance = self.equity_curve[-1] if self.equity_curve else self.initial_balance
        total_return = (final_balance - self.initial_balance) / self.initial_balance
        
        # إحصائيات الصفقات
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # متوسط الربح والخسارة
        wins = [t.pnl_percent for t in self.trades if t.pnl > 0]
        losses = [t.pnl_percent for t in self.trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # عامل الربح
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # أقصى تراجع
        max_drawdown = self._calculate_max_drawdown()
        
        # نسبة شارب
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            trades=self.trades,
            equity_curve=self.equity_curve
        )
    
    def _calculate_max_drawdown(self) -> float:
        """حساب أقصى تراجع"""
        if not self.equity_curve:
            return 0
        
        peak = self.equity_curve[0]
        max_dd = 0
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """حساب نسبة شارب"""
        if len(self.equity_curve) < 2:
            return 0
        
        # حساب العوائد اليومية
        returns = []
        for i in range(1, len(self.equity_curve)):
            r = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
            returns.append(r)
        
        if not returns:
            return 0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # تحويل إلى سنوي (افتراض 252 يوم تداول)
        annual_return = avg_return * 252
        annual_std = std_return * np.sqrt(252)
        
        sharpe = (annual_return - risk_free_rate) / annual_std
        
        return sharpe
    
    # ═══════════════════════════════════════════════════════════════
    # REPORTING
    # ═══════════════════════════════════════════════════════════════
    
    def print_report(self, result: BacktestResult) -> None:
        """طباعة تقرير"""
        print("\n" + "="*60)
        print("📊 BACKTEST REPORT")
        print("="*60)
        
        print(f"\n📅 Period: {result.start_date} to {result.end_date}")
        
        print(f"\n💰 Performance:")
        print(f"   Initial Balance: ${result.initial_balance:,.2f}")
        print(f"   Final Balance:   ${result.final_balance:,.2f}")
        print(f"   Total Return:    {result.total_return:+.2%}")
        
        print(f"\n📈 Trading Statistics:")
        print(f"   Total Trades:    {result.total_trades}")
        print(f"   Winning Trades:  {result.winning_trades}")
        print(f"   Losing Trades:   {result.losing_trades}")
        print(f"   Win Rate:        {result.win_rate:.1%}")
        
        print(f"\n📊 Risk Metrics:")
        print(f"   Avg Win:         {result.avg_win:+.2f}%")
        print(f"   Avg Loss:        {result.avg_loss:.2f}%")
        print(f"   Profit Factor:   {result.profit_factor:.2f}")
        print(f"   Max Drawdown:    {result.max_drawdown:.2%}")
        print(f"   Sharpe Ratio:    {result.sharpe_ratio:.2f}")
        
        print("\n" + "="*60)


# ═══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # اختبار المختبر
    np.random.seed(42)
    
    # إنشاء بيانات وهمية
    n_candles = 1000
    dates = pd.date_range(start='2024-01-01', periods=n_candles, freq='1H')
    
    # محاكاة أسعار
    prices = [50000]
    for _ in range(n_candles - 1):
        change = np.random.randn() * 0.01
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'close': prices,
        'high': [p * 1.005 for p in prices],
        'low': [p * 0.995 for p in prices]
    }, index=dates)
    
    # إنشاء إشارات وهمية
    signals = pd.Series(np.random.choice([-1, 0, 1], size=n_candles, p=[0.1, 0.8, 0.1]), index=dates)
    
    # تشغيل الاختبار
    backtester = Backtester(initial_balance=10000)
    result = backtester.run(df, signals)
    
    # طباعة التقرير
    backtester.print_report(result)
