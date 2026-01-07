"""
Legendary Trading System - A2C Agent
نظام التداول الخارق - وكيل A2C

Advantage Actor-Critic - خوارزمية فعالة للتعلم المعزز.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging


@dataclass
class A2CConfig:
    """إعدادات A2C"""
    state_dim: int = 64
    action_dim: int = 3  # buy, sell, hold
    hidden_dim: int = 256
    learning_rate: float = 7e-4
    gamma: float = 0.99
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_steps: int = 5  # عدد الخطوات قبل التحديث


class A2CNetwork(nn.Module):
    """شبكة Actor-Critic لـ A2C."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # الطبقات المشتركة مع LSTM
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # LSTM للتسلسل الزمني
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # حالة LSTM
        self.hidden = None
    
    def reset_hidden(self, batch_size: int = 1):
        """إعادة تعيين حالة LSTM."""
        device = next(self.parameters()).device
        self.hidden = (
            torch.zeros(1, batch_size, 256).to(device),
            torch.zeros(1, batch_size, 256).to(device)
        )
    
    def forward(self, state: torch.Tensor, 
                hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        التمرير الأمامي.
        
        Args:
            state: حالة البيئة
            hidden: حالة LSTM السابقة
            
        Returns:
            احتمالات الإجراءات، القيمة، حالة LSTM الجديدة
        """
        features = self.shared(state)
        
        # إضافة بعد التسلسل إذا لم يكن موجوداً
        if features.dim() == 2:
            features = features.unsqueeze(1)
        
        if hidden is None:
            hidden = self.hidden
        
        lstm_out, new_hidden = self.lstm(features, hidden)
        lstm_out = lstm_out[:, -1, :]  # آخر خطوة
        
        action_probs = self.actor(lstm_out)
        value = self.critic(lstm_out)
        
        return action_probs, value, new_hidden
    
    def get_action(self, state: torch.Tensor,
                   hidden: Optional[Tuple] = None) -> Tuple[int, torch.Tensor, torch.Tensor, Tuple]:
        """
        اختيار إجراء.
        
        Returns:
            الإجراء، log probability، القيمة، حالة LSTM
        """
        action_probs, value, new_hidden = self.forward(state, hidden)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value, new_hidden


class A2CAgent:
    """
    وكيل A2C للتداول.
    
    يستخدم Advantage Actor-Critic مع LSTM للتعامل مع التسلسلات الزمنية.
    """
    
    def __init__(self, config: A2CConfig):
        self.config = config
        self.logger = logging.getLogger("A2CAgent")
        
        # تحديد الجهاز
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"استخدام الجهاز: {self.device}")
        
        # إنشاء الشبكة
        self.network = A2CNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        ).to(self.device)
        
        # المحسّن
        self.optimizer = optim.RMSprop(
            self.network.parameters(),
            lr=config.learning_rate,
            alpha=0.99,
            eps=1e-5
        )
        
        # تخزين الخطوات
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # حالة LSTM
        self.hidden = None
        
        # إحصائيات
        self.training_step = 0
    
    def reset(self):
        """إعادة تعيين الوكيل."""
        self.network.reset_hidden()
        self.hidden = None
        self._clear_memory()
    
    def _clear_memory(self):
        """مسح الذاكرة."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def select_action(self, state: np.ndarray,
                     training: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        اختيار إجراء.
        
        Args:
            state: حالة البيئة
            training: هل نحن في وضع التدريب
            
        Returns:
            الإجراء، معلومات إضافية
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value, new_hidden = self.network(state_tensor, self.hidden)
        
        if training:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            self.hidden = new_hidden
            
            return action.item(), {
                "log_prob": log_prob.item(),
                "value": value.item(),
                "action_probs": action_probs.cpu().numpy()[0]
            }
        else:
            action = action_probs.argmax().item()
            return action, {
                "action_probs": action_probs.cpu().numpy()[0],
                "value": value.item()
            }
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        value: float, log_prob: float, done: bool):
        """تخزين انتقال."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def update(self, next_value: float = 0) -> Dict[str, float]:
        """
        تحديث الشبكة.
        
        Args:
            next_value: قيمة الحالة التالية
            
        Returns:
            إحصائيات التدريب
        """
        if len(self.states) < self.config.n_steps:
            return {}
        
        # حساب العوائد والمزايا
        returns = []
        advantages = []
        R = next_value
        
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                R = 0
            R = self.rewards[i] + self.config.gamma * R
            returns.insert(0, R)
            advantages.insert(0, R - self.values[i])
        
        # تحويل إلى tensors
        states_t = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_t = torch.LongTensor(self.actions).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        old_log_probs_t = torch.FloatTensor(self.log_probs).to(self.device)
        
        # تطبيع المزايا
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # التمرير الأمامي
        action_probs, values, _ = self.network(states_t)
        dist = Categorical(action_probs)
        
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()
        
        # خسارة السياسة
        policy_loss = -(log_probs * advantages_t).mean()
        
        # خسارة القيمة
        value_loss = nn.MSELoss()(values.squeeze(), returns_t)
        
        # الخسارة الإجمالية
        loss = (
            policy_loss +
            self.config.value_coef * value_loss -
            self.config.entropy_coef * entropy
        )
        
        # التحديث
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        # مسح الذاكرة
        self._clear_memory()
        self.training_step += 1
        
        return {
            "total_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "training_step": self.training_step
        }
    
    def save(self, path: str):
        """حفظ النموذج."""
        torch.save({
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "training_step": self.training_step
        }, path)
        self.logger.info(f"تم حفظ النموذج في {path}")
    
    def load(self, path: str):
        """تحميل النموذج."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        self.logger.info(f"تم تحميل النموذج من {path}")
