"""
Legendary Trading System - TD3 Agent
نظام التداول الخارق - وكيل TD3

Twin Delayed Deep Deterministic Policy Gradient - خوارزمية متقدمة للتعلم المعزز.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging
import random


@dataclass
class TD3Config:
    """إعدادات TD3"""
    state_dim: int = 64
    action_dim: int = 1
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    buffer_size: int = 100000
    batch_size: int = 256
    exploration_noise: float = 0.1


class Actor(nn.Module):
    """شبكة Actor لـ TD3."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """التمرير الأمامي."""
        return self.net(state)


class Critic(nn.Module):
    """شبكة Critic مزدوجة لـ TD3."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, 
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """التمرير الأمامي."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """التمرير الأمامي لـ Q1 فقط."""
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


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


class TD3Agent:
    """
    وكيل TD3 للتداول.
    
    يستخدم Twin Delayed DDPG مع:
    - شبكتي Critic للحد من التقدير الزائد
    - تأخير تحديث السياسة
    - ضوضاء مستهدفة مقطوعة
    """
    
    def __init__(self, config: TD3Config):
        self.config = config
        self.logger = logging.getLogger("TD3Agent")
        
        # تحديد الجهاز
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"استخدام الجهاز: {self.device}")
        
        # Actor
        self.actor = Actor(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        self.actor_target = Actor(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic
        self.critic = Critic(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        self.critic_target = Critic(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # المحسّنات
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )
        
        # ذاكرة إعادة التشغيل
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # إحصائيات
        self.training_step = 0
        self.update_count = 0
    
    def select_action(self, state: np.ndarray,
                     add_noise: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        اختيار إجراء.
        
        Args:
            state: حالة البيئة
            add_noise: إضافة ضوضاء للاستكشاف
            
        Returns:
            الإجراء، معلومات إضافية
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if add_noise:
            noise = np.random.normal(0, self.config.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        
        return action, {}
    
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
        
        self.update_count += 1
        
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
        
        # تحديث Critic
        with torch.no_grad():
            # إضافة ضوضاء مقطوعة للإجراء الهدف
            noise = (
                torch.randn_like(actions_t) * self.config.policy_noise
            ).clamp(-self.config.noise_clip, self.config.noise_clip)
            
            next_actions = (
                self.actor_target(next_states_t) + noise
            ).clamp(-1, 1)
            
            # حساب Q الهدف
            q1_next, q2_next = self.critic_target(next_states_t, next_actions)
            q_next = torch.min(q1_next, q2_next)
            q_target = rewards_t + (1 - dones_t) * self.config.gamma * q_next
        
        # خسارة Critic
        q1, q2 = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # تحديث Actor (مؤجل)
        actor_loss = 0
        if self.update_count % self.config.policy_delay == 0:
            # خسارة Actor
            actor_loss = -self.critic.q1_forward(
                states_t, self.actor(states_t)
            ).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # تحديث الشبكات الهدف
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            
            actor_loss = actor_loss.item()
        
        self.training_step += 1
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss,
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
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": self.config,
            "training_step": self.training_step
        }, path)
        self.logger.info(f"تم حفظ النموذج في {path}")
    
    def load(self, path: str):
        """تحميل النموذج."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor_target.load_state_dict(checkpoint["actor_target_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        
        self.training_step = checkpoint.get("training_step", 0)
        self.logger.info(f"تم تحميل النموذج من {path}")
    
    def action_to_trading_signal(self, action: float) -> str:
        """تحويل الإجراء إلى إشارة تداول."""
        if action > 0.3:
            return "buy"
        elif action < -0.3:
            return "sell"
        else:
            return "hold"
