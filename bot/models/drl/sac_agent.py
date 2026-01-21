"""
Legendary Trading System - SAC Agent
نظام التداول الخارق - وكيل SAC

Soft Actor-Critic - خوارزمية متقدمة للتعلم المعزز مع تحسين الإنتروبيا.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging
import random


@dataclass
class SACConfig:
    """إعدادات SAC"""
    state_dim: int = 64
    action_dim: int = 1  # continuous action
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2  # temperature parameter
    auto_alpha: bool = True
    buffer_size: int = 100000
    batch_size: int = 256
    update_interval: int = 1
    target_update_interval: int = 1


class ReplayBuffer:
    """ذاكرة إعادة التشغيل."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool):
        """إضافة تجربة."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """أخذ عينة عشوائية."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class SoftQNetwork(nn.Module):
    """شبكة Q الناعمة."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """التمرير الأمامي."""
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class GaussianPolicy(nn.Module):
    """سياسة غاوسية للإجراءات المستمرة."""
    
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """التمرير الأمامي."""
        features = self.shared(state)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        أخذ عينة من السياسة.
        
        Returns:
            الإجراء، log probability
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # إعادة المعاملة (reparameterization trick)
        normal = Normal(mean, std)
        x = normal.rsample()
        
        # تطبيق tanh للحد من نطاق الإجراء
        action = torch.tanh(x)
        
        # حساب log probability مع تصحيح tanh
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, 
                   deterministic: bool = False) -> torch.Tensor:
        """الحصول على إجراء."""
        mean, log_std = self.forward(state)
        
        if deterministic:
            return torch.tanh(mean)
        
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()
        return torch.tanh(x)


class SACAgent:
    """
    وكيل SAC للتداول.
    
    يستخدم Soft Actor-Critic مع:
    - تحسين الإنتروبيا التلقائي
    - شبكتي Q للاستقرار
    - سياسة غاوسية للإجراءات المستمرة
    """
    
    def __init__(self, config: SACConfig):
        self.config = config
        self.logger = logging.getLogger("SACAgent")
        
        # تحديد الجهاز
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"استخدام الجهاز: {self.device}")
        
        # شبكات Q
        self.q1 = SoftQNetwork(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        self.q2 = SoftQNetwork(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        
        # شبكات Q الهدف
        self.q1_target = SoftQNetwork(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        self.q2_target = SoftQNetwork(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        
        # نسخ الأوزان
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # السياسة
        self.policy = GaussianPolicy(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        
        # المحسّنات
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=config.learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=config.learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # معامل الحرارة (alpha)
        if config.auto_alpha:
            self.target_entropy = -config.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = config.alpha
        
        # ذاكرة إعادة التشغيل
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # إحصائيات
        self.training_step = 0
        self.update_count = 0
    
    def select_action(self, state: np.ndarray, 
                     deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        اختيار إجراء.
        
        Args:
            state: حالة البيئة
            deterministic: استخدام السياسة الحتمية
            
        Returns:
            الإجراء، معلومات إضافية
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.policy.get_action(state_tensor, deterministic)
        
        action_np = action.cpu().numpy()[0]
        
        return action_np, {"alpha": self.alpha}
    
    def store_transition(self, state: np.ndarray, action: np.ndarray,
                        reward: float, next_state: np.ndarray, done: bool):
        """تخزين انتقال."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """
        تحديث الشبكات.
        
        Returns:
            إحصائيات التدريب
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        # أخذ عينة
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )
        
        # تحويل إلى tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # تحديث شبكات Q
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states_t)
            
            q1_next = self.q1_target(next_states_t, next_actions)
            q2_next = self.q2_target(next_states_t, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            
            q_target = rewards_t + (1 - dones_t) * self.config.gamma * q_next
        
        # خسارة Q1
        q1_pred = self.q1(states_t, actions_t)
        q1_loss = F.mse_loss(q1_pred, q_target)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        # خسارة Q2
        q2_pred = self.q2(states_t, actions_t)
        q2_loss = F.mse_loss(q2_pred, q_target)
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # تحديث السياسة
        new_actions, log_probs = self.policy.sample(states_t)
        q1_new = self.q1(states_t, new_actions)
        q2_new = self.q2(states_t, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # تحديث alpha (إذا تلقائي)
        alpha_loss = 0
        if self.config.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # تحديث الشبكات الهدف
        self.update_count += 1
        if self.update_count % self.config.target_update_interval == 0:
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)
        
        self.training_step += 1
        
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item() if self.config.auto_alpha else 0,
            "alpha": self.alpha,
            "training_step": self.training_step
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """تحديث ناعم للشبكة الهدف."""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.config.tau * source_param.data +
                (1 - self.config.tau) * target_param.data
            )
    
    def save(self, path: str):
        """حفظ النموذج."""
        torch.save({
            "q1_state_dict": self.q1.state_dict(),
            "q2_state_dict": self.q2.state_dict(),
            "q1_target_state_dict": self.q1_target.state_dict(),
            "q2_target_state_dict": self.q2_target.state_dict(),
            "policy_state_dict": self.policy.state_dict(),
            "q1_optimizer": self.q1_optimizer.state_dict(),
            "q2_optimizer": self.q2_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "log_alpha": self.log_alpha if self.config.auto_alpha else None,
            "config": self.config,
            "training_step": self.training_step
        }, path)
        self.logger.info(f"تم حفظ النموذج في {path}")
    
    def load(self, path: str):
        """تحميل النموذج."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q1.load_state_dict(checkpoint["q1_state_dict"])
        self.q2.load_state_dict(checkpoint["q2_state_dict"])
        self.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        self.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.q1_optimizer.load_state_dict(checkpoint["q1_optimizer"])
        self.q2_optimizer.load_state_dict(checkpoint["q2_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        
        if checkpoint.get("log_alpha") is not None:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha = self.log_alpha.exp().item()
        
        self.training_step = checkpoint.get("training_step", 0)
        self.logger.info(f"تم تحميل النموذج من {path}")
    
    def action_to_trading_signal(self, action: float) -> str:
        """
        تحويل الإجراء المستمر إلى إشارة تداول.
        
        Args:
            action: الإجراء (-1 إلى 1)
            
        Returns:
            إشارة التداول
        """
        if action > 0.3:
            return "buy"
        elif action < -0.3:
            return "sell"
        else:
            return "hold"
