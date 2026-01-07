"""
Legendary Trading System - Auto Trainer
نظام التداول الخارق - نظام التدريب التلقائي

يدير عملية التدريب والتحسين التلقائي للنماذج.
"""

import asyncio
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import pickle
from enum import Enum
import optuna
from concurrent.futures import ThreadPoolExecutor


class TrainingMode(Enum):
    """أوضاع التدريب"""
    INITIAL = "initial"          # تدريب أولي
    CONTINUOUS = "continuous"    # تدريب مستمر
    FINE_TUNE = "fine_tune"      # ضبط دقيق
    RETRAIN = "retrain"          # إعادة تدريب


@dataclass
class TrainingConfig:
    """إعدادات التدريب"""
    # إعدادات عامة
    model_type: str = "ppo"
    training_mode: TrainingMode = TrainingMode.CONTINUOUS
    
    # إعدادات البيانات
    lookback_days: int = 365
    train_split: float = 0.8
    validation_split: float = 0.1
    
    # إعدادات التدريب
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    early_stopping_patience: int = 50
    min_improvement: float = 0.001
    
    # إعدادات التحسين
    optimize_hyperparams: bool = True
    n_trials: int = 50
    optimization_metric: str = "sharpe_ratio"
    
    # إعدادات الحفظ
    save_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    best_model_path: str = "best_model.pt"
    
    # إعدادات إعادة التدريب
    retrain_threshold: float = 0.7  # إعادة التدريب إذا انخفض الأداء
    retrain_interval_hours: int = 24


@dataclass
class TrainingMetrics:
    """مقاييس التدريب"""
    episode: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    profitable_trades: int = 0
    loss: float = 0.0
    validation_reward: float = 0.0


class TradingEnvironment:
    """
    بيئة التداول للتدريب.
    
    تحاكي سوق العملات المشفرة للتدريب.
    """
    
    def __init__(self, data: np.ndarray, config: Dict[str, Any]):
        self.data = data
        self.config = config
        
        self.initial_balance = config.get("initial_balance", 10000)
        self.transaction_fee = config.get("transaction_fee", 0.001)
        self.max_position = config.get("max_position", 1.0)
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """إعادة تعيين البيئة."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []
        self.equity_curve = [self.initial_balance]
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        تنفيذ خطوة في البيئة.
        
        Args:
            action: الإجراء (0=hold, 1=buy, 2=sell)
            
        Returns:
            الحالة الجديدة، المكافأة، انتهاء، معلومات
        """
        current_price = self._get_current_price()
        reward = 0.0
        info = {}
        
        # تنفيذ الإجراء
        if action == 1 and self.position <= 0:  # شراء
            self._execute_buy(current_price)
        elif action == 2 and self.position >= 0:  # بيع
            self._execute_sell(current_price)
        
        # حساب المكافأة
        reward = self._calculate_reward(current_price)
        
        # تحديث الخطوة
        self.current_step += 1
        
        # تحديث منحنى رأس المال
        equity = self._calculate_equity(current_price)
        self.equity_curve.append(equity)
        
        # التحقق من الانتهاء
        done = self.current_step >= len(self.data) - 1
        
        # معلومات إضافية
        info = {
            "equity": equity,
            "position": self.position,
            "trades": len(self.trades)
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """الحصول على الحالة الحالية."""
        if self.current_step >= len(self.data):
            return np.zeros(self.data.shape[1])
        return self.data[self.current_step]
    
    def _get_current_price(self) -> float:
        """الحصول على السعر الحالي."""
        # نفترض أن العمود الأول هو السعر
        return self.data[self.current_step][0]
    
    def _execute_buy(self, price: float):
        """تنفيذ شراء."""
        if self.balance > 0:
            amount = self.balance * (1 - self.transaction_fee)
            self.position = amount / price
            self.entry_price = price
            self.balance = 0
            
            self.trades.append({
                "type": "buy",
                "price": price,
                "amount": self.position,
                "step": self.current_step
            })
    
    def _execute_sell(self, price: float):
        """تنفيذ بيع."""
        if self.position > 0:
            amount = self.position * price * (1 - self.transaction_fee)
            pnl = (price - self.entry_price) / self.entry_price
            
            self.trades.append({
                "type": "sell",
                "price": price,
                "amount": self.position,
                "pnl": pnl,
                "step": self.current_step
            })
            
            self.balance = amount
            self.position = 0
            self.entry_price = 0
    
    def _calculate_reward(self, current_price: float) -> float:
        """حساب المكافأة."""
        # مكافأة بناءً على تغير رأس المال
        if len(self.equity_curve) < 2:
            return 0.0
        
        current_equity = self._calculate_equity(current_price)
        prev_equity = self.equity_curve[-1]
        
        # نسبة التغير
        returns = (current_equity - prev_equity) / prev_equity
        
        # مكافأة معدلة بالمخاطر (Sharpe-like)
        reward = returns * 100  # تكبير للتدريب
        
        # عقوبة على عدم التداول
        if len(self.trades) == 0 and self.current_step > 100:
            reward -= 0.001
        
        return reward
    
    def _calculate_equity(self, current_price: float) -> float:
        """حساب رأس المال الحالي."""
        return self.balance + self.position * current_price
    
    def get_metrics(self) -> Dict[str, float]:
        """الحصول على مقاييس الأداء."""
        if len(self.equity_curve) < 2:
            return {}
        
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        # Sharpe Ratio
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max()
        
        # Win Rate
        profitable = sum(1 for t in self.trades if t.get("pnl", 0) > 0)
        total = len([t for t in self.trades if "pnl" in t])
        win_rate = profitable / total if total > 0 else 0
        
        # Profit Factor
        gains = sum(t["pnl"] for t in self.trades if t.get("pnl", 0) > 0)
        losses = abs(sum(t["pnl"] for t in self.trades if t.get("pnl", 0) < 0))
        profit_factor = gains / losses if losses > 0 else float('inf')
        
        return {
            "total_return": (equity[-1] - equity[0]) / equity[0],
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(self.trades)
        }


class AutoTrainer:
    """
    نظام التدريب التلقائي.
    
    يدير:
    - التدريب الأولي والمستمر
    - تحسين المعاملات تلقائياً
    - مراقبة الأداء وإعادة التدريب
    - حفظ واستعادة النماذج
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger("AutoTrainer")
        
        # إنشاء مجلد الحفظ
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # النموذج الحالي
        self.model = None
        self.best_metrics = None
        self.training_history: List[TrainingMetrics] = []
        
        # حالة التدريب
        self.is_training = False
        self.should_stop = False
        self.last_retrain = datetime.utcnow()
        
        # Executor للتدريب المتوازي
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self, model_class, model_config: Dict) -> bool:
        """
        تهيئة النظام.
        
        Args:
            model_class: فئة النموذج
            model_config: إعدادات النموذج
        """
        self.logger.info("تهيئة نظام التدريب التلقائي...")
        
        try:
            self.model_class = model_class
            self.model_config = model_config
            
            # إنشاء النموذج
            self.model = model_class(model_config)
            
            # محاولة تحميل نموذج محفوظ
            best_path = self.checkpoint_dir / self.config.best_model_path
            if best_path.exists():
                self.model.load(str(best_path))
                self.logger.info("تم تحميل أفضل نموذج محفوظ")
            
            return True
            
        except Exception as e:
            self.logger.error(f"خطأ في التهيئة: {e}")
            return False
    
    async def train(self, train_data: np.ndarray, 
                   val_data: Optional[np.ndarray] = None) -> TrainingMetrics:
        """
        تدريب النموذج.
        
        Args:
            train_data: بيانات التدريب
            val_data: بيانات التحقق
            
        Returns:
            مقاييس التدريب النهائية
        """
        self.is_training = True
        self.should_stop = False
        
        self.logger.info("بدء التدريب...")
        
        # إنشاء بيئة التدريب
        env_config = {
            "initial_balance": 10000,
            "transaction_fee": 0.001
        }
        train_env = TradingEnvironment(train_data, env_config)
        val_env = TradingEnvironment(val_data, env_config) if val_data is not None else None
        
        best_reward = float('-inf')
        patience_counter = 0
        
        for episode in range(self.config.max_episodes):
            if self.should_stop:
                break
            
            # تدريب حلقة واحدة
            metrics = await self._train_episode(train_env, episode)
            
            # التحقق
            if val_env:
                val_metrics = await self._evaluate(val_env)
                metrics.validation_reward = val_metrics.total_reward
            
            # تسجيل التاريخ
            self.training_history.append(metrics)
            
            # التحقق من التحسن
            if metrics.total_reward > best_reward + self.config.min_improvement:
                best_reward = metrics.total_reward
                patience_counter = 0
                
                # حفظ أفضل نموذج
                self._save_best_model(metrics)
            else:
                patience_counter += 1
            
            # التوقف المبكر
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"التوقف المبكر في الحلقة {episode}")
                break
            
            # حفظ دوري
            if episode % self.config.save_interval == 0:
                self._save_checkpoint(episode, metrics)
            
            # تسجيل التقدم
            if episode % 10 == 0:
                self.logger.info(
                    f"الحلقة {episode}: المكافأة={metrics.total_reward:.2f}, "
                    f"Sharpe={metrics.sharpe_ratio:.2f}, "
                    f"الصفقات={metrics.total_trades}"
                )
        
        self.is_training = False
        
        # إرجاع أفضل مقاييس
        return self.best_metrics or metrics
    
    async def _train_episode(self, env: TradingEnvironment, 
                            episode: int) -> TrainingMetrics:
        """تدريب حلقة واحدة."""
        state = env.reset()
        total_reward = 0
        step = 0
        
        while step < self.config.max_steps_per_episode:
            # اختيار إجراء
            action, info = self.model.select_action(state, training=True)
            
            # تنفيذ الخطوة
            next_state, reward, done, env_info = env.step(action)
            
            # تخزين التجربة
            self.model.store_transition(
                state, action, reward,
                info.get("value", 0),
                info.get("log_prob", 0),
                done
            )
            
            total_reward += reward
            state = next_state
            step += 1
            
            if done:
                break
        
        # تحديث النموذج
        update_info = self.model.update()
        
        # جمع المقاييس
        env_metrics = env.get_metrics()
        
        return TrainingMetrics(
            episode=episode,
            total_reward=total_reward,
            avg_reward=total_reward / step if step > 0 else 0,
            sharpe_ratio=env_metrics.get("sharpe_ratio", 0),
            max_drawdown=env_metrics.get("max_drawdown", 0),
            win_rate=env_metrics.get("win_rate", 0),
            profit_factor=env_metrics.get("profit_factor", 0),
            total_trades=env_metrics.get("total_trades", 0),
            loss=update_info.get("total_loss", 0)
        )
    
    async def _evaluate(self, env: TradingEnvironment) -> TrainingMetrics:
        """تقييم النموذج."""
        state = env.reset()
        total_reward = 0
        step = 0
        
        while step < self.config.max_steps_per_episode:
            # اختيار إجراء (بدون استكشاف)
            action, _ = self.model.select_action(state, training=False)
            
            next_state, reward, done, _ = env.step(action)
            
            total_reward += reward
            state = next_state
            step += 1
            
            if done:
                break
        
        env_metrics = env.get_metrics()
        
        return TrainingMetrics(
            total_reward=total_reward,
            sharpe_ratio=env_metrics.get("sharpe_ratio", 0),
            max_drawdown=env_metrics.get("max_drawdown", 0),
            win_rate=env_metrics.get("win_rate", 0)
        )
    
    async def optimize_hyperparameters(self, train_data: np.ndarray,
                                       val_data: np.ndarray) -> Dict[str, Any]:
        """
        تحسين المعاملات تلقائياً باستخدام Optuna.
        
        Args:
            train_data: بيانات التدريب
            val_data: بيانات التحقق
            
        Returns:
            أفضل المعاملات
        """
        self.logger.info("بدء تحسين المعاملات...")
        
        def objective(trial):
            # اقتراح معاملات
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "gamma": trial.suggest_float("gamma", 0.9, 0.999),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            }
            
            # تحديث إعدادات النموذج
            config = {**self.model_config, **params}
            
            # إنشاء نموذج جديد
            model = self.model_class(config)
            
            # تدريب مختصر
            env_config = {"initial_balance": 10000, "transaction_fee": 0.001}
            train_env = TradingEnvironment(train_data, env_config)
            val_env = TradingEnvironment(val_data, env_config)
            
            # تدريب 100 حلقة
            for _ in range(100):
                state = train_env.reset()
                done = False
                while not done:
                    action, info = model.select_action(state, training=True)
                    next_state, reward, done, _ = train_env.step(action)
                    model.store_transition(
                        state, action, reward,
                        info.get("value", 0),
                        info.get("log_prob", 0),
                        done
                    )
                    state = next_state
                model.update()
            
            # تقييم
            state = val_env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = model.select_action(state, training=False)
                next_state, reward, done, _ = val_env.step(action)
                total_reward += reward
                state = next_state
            
            metrics = val_env.get_metrics()
            
            # المقياس المستهدف
            if self.config.optimization_metric == "sharpe_ratio":
                return metrics.get("sharpe_ratio", 0)
            elif self.config.optimization_metric == "total_return":
                return metrics.get("total_return", 0)
            else:
                return total_reward
        
        # إنشاء دراسة Optuna
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        self.logger.info(f"أفضل المعاملات: {best_params}")
        
        # تحديث النموذج بأفضل المعاملات
        self.model_config.update(best_params)
        self.model = self.model_class(self.model_config)
        
        return best_params
    
    async def continuous_learning(self, data_provider) -> None:
        """
        التعلم المستمر من البيانات الجديدة.
        
        Args:
            data_provider: مزود البيانات
        """
        self.logger.info("بدء التعلم المستمر...")
        
        while not self.should_stop:
            try:
                # الحصول على بيانات جديدة
                new_data = await data_provider.get_recent_data()
                
                if new_data is not None and len(new_data) > 0:
                    # تدريب على البيانات الجديدة
                    await self._incremental_train(new_data)
                
                # التحقق من الحاجة لإعادة التدريب
                if await self._should_retrain():
                    self.logger.info("إعادة التدريب بسبب انخفاض الأداء...")
                    full_data = await data_provider.get_full_data()
                    await self.train(full_data)
                    self.last_retrain = datetime.utcnow()
                
                # انتظار قبل التحديث التالي
                await asyncio.sleep(3600)  # ساعة واحدة
                
            except Exception as e:
                self.logger.error(f"خطأ في التعلم المستمر: {e}")
                await asyncio.sleep(60)
    
    async def _incremental_train(self, data: np.ndarray) -> None:
        """تدريب تزايدي على بيانات جديدة."""
        env_config = {"initial_balance": 10000, "transaction_fee": 0.001}
        env = TradingEnvironment(data, env_config)
        
        state = env.reset()
        done = False
        
        while not done:
            action, info = self.model.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            
            self.model.store_transition(
                state, action, reward,
                info.get("value", 0),
                info.get("log_prob", 0),
                done
            )
            
            state = next_state
        
        self.model.update()
    
    async def _should_retrain(self) -> bool:
        """التحقق من الحاجة لإعادة التدريب."""
        # التحقق من الوقت
        hours_since_retrain = (datetime.utcnow() - self.last_retrain).total_seconds() / 3600
        if hours_since_retrain < self.config.retrain_interval_hours:
            return False
        
        # التحقق من الأداء
        if self.best_metrics and len(self.training_history) > 0:
            recent_metrics = self.training_history[-10:]
            avg_recent = np.mean([m.sharpe_ratio for m in recent_metrics])
            
            if avg_recent < self.best_metrics.sharpe_ratio * self.config.retrain_threshold:
                return True
        
        return False
    
    def _save_best_model(self, metrics: TrainingMetrics) -> None:
        """حفظ أفضل نموذج."""
        self.best_metrics = metrics
        path = self.checkpoint_dir / self.config.best_model_path
        self.model.save(str(path))
        
        # حفظ المقاييس
        metrics_path = self.checkpoint_dir / "best_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                "episode": metrics.episode,
                "total_reward": metrics.total_reward,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "win_rate": metrics.win_rate,
                "timestamp": datetime.utcnow().isoformat()
            }, f, indent=2)
        
        self.logger.info(f"تم حفظ أفضل نموذج (Sharpe: {metrics.sharpe_ratio:.2f})")
    
    def _save_checkpoint(self, episode: int, metrics: TrainingMetrics) -> None:
        """حفظ نقطة تفتيش."""
        path = self.checkpoint_dir / f"checkpoint_{episode}.pt"
        self.model.save(str(path))
        
        # حفظ تاريخ التدريب
        history_path = self.checkpoint_dir / "training_history.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
    
    def stop_training(self) -> None:
        """إيقاف التدريب."""
        self.should_stop = True
        self.logger.info("تم طلب إيقاف التدريب")
    
    def get_training_status(self) -> Dict[str, Any]:
        """الحصول على حالة التدريب."""
        return {
            "is_training": self.is_training,
            "episodes_completed": len(self.training_history),
            "best_sharpe": self.best_metrics.sharpe_ratio if self.best_metrics else 0,
            "last_retrain": self.last_retrain.isoformat(),
            "recent_metrics": [
                {
                    "episode": m.episode,
                    "reward": m.total_reward,
                    "sharpe": m.sharpe_ratio
                }
                for m in self.training_history[-10:]
            ]
        }
