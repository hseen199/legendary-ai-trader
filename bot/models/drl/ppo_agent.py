"""
Legendary Trading System - PPO Agent
نظام التداول الخارق - وكيل PPO

Proximal Policy Optimization - أحد أقوى خوارزميات التعلم المعزز.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging


@dataclass
class PPOConfig:
    """إعدادات PPO"""
    state_dim: int = 64
    action_dim: int = 3  # buy, sell, hold
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    batch_size: int = 64
    buffer_size: int = 2048


class ActorCritic(nn.Module):
    """
    شبكة Actor-Critic للـ PPO.
    
    Actor: يحدد السياسة (الإجراءات)
    Critic: يقيّم القيمة (الحالات)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # الطبقات المشتركة
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Actor (السياسة)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (القيمة)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # تهيئة الأوزان
        self._init_weights()
    
    def _init_weights(self):
        """تهيئة الأوزان بشكل صحيح."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        التمرير الأمامي.
        
        Args:
            state: حالة البيئة
            
        Returns:
            احتمالات الإجراءات، قيمة الحالة
        """
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        اختيار إجراء بناءً على السياسة.
        
        Args:
            state: حالة البيئة
            
        Returns:
            الإجراء، log probability، القيمة
        """
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value
    
    def evaluate(self, states: torch.Tensor, 
                actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        تقييم الإجراءات.
        
        Args:
            states: مجموعة الحالات
            actions: مجموعة الإجراءات
            
        Returns:
            log probabilities، القيم، الإنتروبيا
        """
        action_probs, values = self.forward(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(), entropy


class PPOMemory:
    """ذاكرة تخزين التجارب لـ PPO."""
    
    def __init__(self, buffer_size: int = 2048):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.buffer_size = buffer_size
    
    def store(self, state: np.ndarray, action: int, reward: float,
              value: float, log_prob: float, done: bool):
        """تخزين تجربة."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        """مسح الذاكرة."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def get_batches(self, batch_size: int) -> List[Tuple]:
        """الحصول على دفعات للتدريب."""
        n_samples = len(self.states)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        batches = []
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            batch = (
                np.array([self.states[i] for i in batch_indices]),
                np.array([self.actions[i] for i in batch_indices]),
                np.array([self.rewards[i] for i in batch_indices]),
                np.array([self.values[i] for i in batch_indices]),
                np.array([self.log_probs[i] for i in batch_indices]),
                np.array([self.dones[i] for i in batch_indices])
            )
            batches.append(batch)
        
        return batches
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    وكيل PPO للتداول.
    
    يستخدم Proximal Policy Optimization لتعلم استراتيجية التداول المثلى.
    """
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.logger = logging.getLogger("PPOAgent")
        
        # تحديد الجهاز
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"استخدام الجهاز: {self.device}")
        
        # إنشاء الشبكة
        self.network = ActorCritic(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        ).to(self.device)
        
        # المحسّن
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate
        )
        
        # الذاكرة
        self.memory = PPOMemory(config.buffer_size)
        
        # إحصائيات التدريب
        self.training_step = 0
        self.episode_rewards = deque(maxlen=100)
    
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
            action_probs, value = self.network(state_tensor)
        
        if training:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), {
                "log_prob": log_prob.item(),
                "value": value.item(),
                "action_probs": action_probs.cpu().numpy()[0]
            }
        else:
            # في وضع التقييم، اختر الإجراء الأفضل
            action = action_probs.argmax().item()
            return action, {
                "action_probs": action_probs.cpu().numpy()[0],
                "value": value.item()
            }
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        value: float, log_prob: float, done: bool):
        """تخزين انتقال في الذاكرة."""
        self.memory.store(state, action, reward, value, log_prob, done)
    
    def compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                   dones: np.ndarray, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        حساب Generalized Advantage Estimation.
        
        Args:
            rewards: المكافآت
            values: القيم
            dones: علامات الانتهاء
            next_value: قيمة الحالة التالية
            
        Returns:
            المزايا، العوائد
        """
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        next_val = next_value
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_val = 0
                gae = 0
            
            delta = rewards[t] + self.config.gamma * next_val - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_val = values[t]
        
        return advantages, returns
    
    def update(self, next_value: float = 0) -> Dict[str, float]:
        """
        تحديث الشبكة باستخدام PPO.
        
        Args:
            next_value: قيمة الحالة التالية
            
        Returns:
            إحصائيات التدريب
        """
        if len(self.memory) < self.config.batch_size:
            return {}
        
        # تحويل البيانات
        states = np.array(self.memory.states)
        actions = np.array(self.memory.actions)
        rewards = np.array(self.memory.rewards)
        values = np.array(self.memory.values)
        old_log_probs = np.array(self.memory.log_probs)
        dones = np.array(self.memory.dones)
        
        # حساب المزايا والعوائد
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # تطبيع المزايا
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # تحويل إلى tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # تدريب لعدة epochs
        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0
        entropy_total = 0
        
        for _ in range(self.config.ppo_epochs):
            # خلط البيانات
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                # استخراج الدفعة
                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                
                # تقييم الإجراءات الحالية
                new_log_probs, new_values, entropy = self.network.evaluate(
                    batch_states, batch_actions
                )
                
                # حساب النسبة
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # خسارة السياسة المقطوعة
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # خسارة القيمة
                value_loss = nn.MSELoss()(new_values, batch_returns)
                
                # خسارة الإنتروبيا
                entropy_loss = -entropy.mean()
                
                # الخسارة الإجمالية
                loss = (
                    policy_loss +
                    self.config.value_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )
                
                # التحديث
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                total_loss += loss.item()
                policy_loss_total += policy_loss.item()
                value_loss_total += value_loss.item()
                entropy_total += entropy.mean().item()
        
        # مسح الذاكرة
        self.memory.clear()
        self.training_step += 1
        
        n_updates = self.config.ppo_epochs * (len(states) // self.config.batch_size + 1)
        
        return {
            "total_loss": total_loss / n_updates,
            "policy_loss": policy_loss_total / n_updates,
            "value_loss": value_loss_total / n_updates,
            "entropy": entropy_total / n_updates,
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
    
    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """الحصول على احتمالات الإجراءات."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.network(state_tensor)
        return action_probs.cpu().numpy()[0]
