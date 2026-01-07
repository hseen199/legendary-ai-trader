"""
Legendary Trading System - Advanced Memory System
نظام التداول الخارق - نظام الذاكرة المتقدم

يدير جميع أنواع الذاكرة للوكلاء.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path
import logging
import json
import pickle
import hashlib
from enum import Enum
import sqlite3
from abc import ABC, abstractmethod


class MemoryType(Enum):
    """أنواع الذاكرة"""
    EPISODIC = "episodic"       # ذاكرة الأحداث
    SEMANTIC = "semantic"       # ذاكرة المعرفة
    PROCEDURAL = "procedural"   # ذاكرة الإجراءات
    WORKING = "working"         # ذاكرة العمل
    LONG_TERM = "long_term"     # ذاكرة طويلة المدى


@dataclass
class MemoryEntry:
    """إدخال في الذاكرة"""
    id: str
    type: MemoryType
    content: Any
    timestamp: datetime
    importance: float = 0.5
    access_count: int = 0
    last_accessed: datetime = None
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp


class BaseMemory(ABC):
    """الفئة الأساسية للذاكرة."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> bool:
        """تخزين إدخال."""
        pass
    
    @abstractmethod
    async def retrieve(self, query: Any, k: int = 5) -> List[MemoryEntry]:
        """استرجاع إدخالات."""
        pass
    
    @abstractmethod
    async def forget(self, entry_id: str) -> bool:
        """حذف إدخال."""
        pass
    
    @abstractmethod
    async def consolidate(self) -> None:
        """تجميع وتنظيم الذاكرة."""
        pass


class EpisodicMemory(BaseMemory):
    """
    الذاكرة العرضية - تخزن الأحداث والتجارب.
    
    تستخدم لـ:
    - تذكر الصفقات السابقة
    - تعلم من الأخطاء
    - التعرف على الأنماط المتكررة
    """
    
    def __init__(self, capacity: int = 10000):
        super().__init__(capacity)
        self._episodes: deque = deque(maxlen=capacity)
        self._index: Dict[str, int] = {}
    
    async def store(self, entry: MemoryEntry) -> bool:
        """تخزين حدث."""
        try:
            entry.type = MemoryType.EPISODIC
            self._episodes.append(entry)
            self._index[entry.id] = len(self._episodes) - 1
            return True
        except Exception as e:
            self.logger.error(f"خطأ في تخزين الحدث: {e}")
            return False
    
    async def retrieve(self, query: Any, k: int = 5) -> List[MemoryEntry]:
        """استرجاع أحداث مشابهة."""
        if not self._episodes:
            return []
        
        # البحث بالتشابه
        if isinstance(query, str):
            # بحث نصي بسيط
            results = [
                e for e in self._episodes
                if query.lower() in str(e.content).lower()
            ]
        elif isinstance(query, dict):
            # بحث بالخصائص
            results = self._search_by_attributes(query)
        else:
            results = list(self._episodes)
        
        # ترتيب بالأهمية والحداثة
        results.sort(
            key=lambda x: (x.importance, x.timestamp),
            reverse=True
        )
        
        # تحديث عداد الوصول
        for entry in results[:k]:
            entry.access_count += 1
            entry.last_accessed = datetime.utcnow()
        
        return results[:k]
    
    async def retrieve_recent(self, n: int = 10) -> List[MemoryEntry]:
        """استرجاع أحدث الأحداث."""
        return list(self._episodes)[-n:]
    
    async def retrieve_by_symbol(self, symbol: str, n: int = 10) -> List[MemoryEntry]:
        """استرجاع أحداث لرمز معين."""
        results = [
            e for e in self._episodes
            if e.metadata.get("symbol") == symbol
        ]
        return results[-n:]
    
    async def forget(self, entry_id: str) -> bool:
        """حذف حدث."""
        if entry_id in self._index:
            idx = self._index[entry_id]
            if 0 <= idx < len(self._episodes):
                self._episodes[idx] = None
                del self._index[entry_id]
                return True
        return False
    
    async def consolidate(self) -> None:
        """تجميع الأحداث المتشابهة."""
        # إزالة الأحداث المحذوفة
        self._episodes = deque(
            [e for e in self._episodes if e is not None],
            maxlen=self.capacity
        )
        
        # إعادة بناء الفهرس
        self._index = {e.id: i for i, e in enumerate(self._episodes)}
    
    def _search_by_attributes(self, query: Dict) -> List[MemoryEntry]:
        """بحث بالخصائص."""
        results = []
        for entry in self._episodes:
            if entry is None:
                continue
            
            match = True
            for key, value in query.items():
                if key in entry.metadata:
                    if entry.metadata[key] != value:
                        match = False
                        break
            
            if match:
                results.append(entry)
        
        return results


class SemanticMemory(BaseMemory):
    """
    الذاكرة الدلالية - تخزن المعرفة والحقائق.
    
    تستخدم لـ:
    - معرفة خصائص العملات
    - قواعد التداول
    - الأنماط المعروفة
    """
    
    def __init__(self, capacity: int = 5000, db_path: str = "semantic_memory.db"):
        super().__init__(capacity)
        self.db_path = db_path
        self._knowledge: Dict[str, MemoryEntry] = {}
        self._categories: Dict[str, List[str]] = {}
        self._init_db()
    
    def _init_db(self):
        """تهيئة قاعدة البيانات."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                category TEXT,
                content TEXT,
                importance REAL,
                created_at TEXT,
                updated_at TEXT,
                access_count INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def store(self, entry: MemoryEntry) -> bool:
        """تخزين معرفة."""
        try:
            entry.type = MemoryType.SEMANTIC
            self._knowledge[entry.id] = entry
            
            # تصنيف
            category = entry.metadata.get("category", "general")
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(entry.id)
            
            # حفظ في قاعدة البيانات
            await self._save_to_db(entry)
            
            return True
        except Exception as e:
            self.logger.error(f"خطأ في تخزين المعرفة: {e}")
            return False
    
    async def retrieve(self, query: Any, k: int = 5) -> List[MemoryEntry]:
        """استرجاع معرفة."""
        if isinstance(query, str):
            # بحث بالفئة
            if query in self._categories:
                ids = self._categories[query]
                return [self._knowledge[id] for id in ids[:k] if id in self._knowledge]
            
            # بحث نصي
            results = [
                e for e in self._knowledge.values()
                if query.lower() in str(e.content).lower()
            ]
            return results[:k]
        
        return list(self._knowledge.values())[:k]
    
    async def get_knowledge(self, key: str) -> Optional[MemoryEntry]:
        """الحصول على معرفة محددة."""
        return self._knowledge.get(key)
    
    async def update_knowledge(self, key: str, content: Any) -> bool:
        """تحديث معرفة."""
        if key in self._knowledge:
            entry = self._knowledge[key]
            entry.content = content
            entry.last_accessed = datetime.utcnow()
            await self._save_to_db(entry)
            return True
        return False
    
    async def forget(self, entry_id: str) -> bool:
        """حذف معرفة."""
        if entry_id in self._knowledge:
            entry = self._knowledge[entry_id]
            category = entry.metadata.get("category", "general")
            
            if category in self._categories:
                self._categories[category].remove(entry_id)
            
            del self._knowledge[entry_id]
            
            # حذف من قاعدة البيانات
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM knowledge WHERE id = ?", (entry_id,))
            conn.commit()
            conn.close()
            
            return True
        return False
    
    async def consolidate(self) -> None:
        """تجميع المعرفة."""
        # دمج المعرفة المتشابهة
        pass
    
    async def _save_to_db(self, entry: MemoryEntry):
        """حفظ في قاعدة البيانات."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO knowledge
            (id, category, content, importance, created_at, updated_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.metadata.get("category", "general"),
            json.dumps(entry.content) if not isinstance(entry.content, str) else entry.content,
            entry.importance,
            entry.timestamp.isoformat(),
            entry.last_accessed.isoformat(),
            entry.access_count
        ))
        
        conn.commit()
        conn.close()
    
    async def load_from_db(self):
        """تحميل من قاعدة البيانات."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM knowledge")
        rows = cursor.fetchall()
        
        for row in rows:
            entry = MemoryEntry(
                id=row[0],
                type=MemoryType.SEMANTIC,
                content=json.loads(row[2]) if row[2].startswith('{') else row[2],
                timestamp=datetime.fromisoformat(row[4]),
                importance=row[3],
                access_count=row[6],
                last_accessed=datetime.fromisoformat(row[5]),
                metadata={"category": row[1]}
            )
            self._knowledge[entry.id] = entry
            
            if row[1] not in self._categories:
                self._categories[row[1]] = []
            self._categories[row[1]].append(entry.id)
        
        conn.close()


class WorkingMemory(BaseMemory):
    """
    ذاكرة العمل - للمعالجة الفورية.
    
    تستخدم لـ:
    - تخزين السياق الحالي
    - المعلومات المؤقتة
    - التحليلات الجارية
    """
    
    def __init__(self, capacity: int = 100):
        super().__init__(capacity)
        self._buffer: deque = deque(maxlen=capacity)
        self._context: Dict[str, Any] = {}
    
    async def store(self, entry: MemoryEntry) -> bool:
        """تخزين في ذاكرة العمل."""
        entry.type = MemoryType.WORKING
        self._buffer.append(entry)
        return True
    
    async def retrieve(self, query: Any, k: int = 5) -> List[MemoryEntry]:
        """استرجاع من ذاكرة العمل."""
        return list(self._buffer)[-k:]
    
    async def forget(self, entry_id: str) -> bool:
        """حذف من ذاكرة العمل."""
        self._buffer = deque(
            [e for e in self._buffer if e.id != entry_id],
            maxlen=self.capacity
        )
        return True
    
    async def consolidate(self) -> None:
        """تنظيف ذاكرة العمل."""
        # إزالة العناصر القديمة
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self._buffer = deque(
            [e for e in self._buffer if e.timestamp > cutoff],
            maxlen=self.capacity
        )
    
    def set_context(self, key: str, value: Any):
        """تعيين سياق."""
        self._context[key] = value
    
    def get_context(self, key: str) -> Any:
        """الحصول على سياق."""
        return self._context.get(key)
    
    def clear_context(self):
        """مسح السياق."""
        self._context.clear()


class MemorySystem:
    """
    نظام الذاكرة المتكامل.
    
    يدير جميع أنواع الذاكرة ويوفر واجهة موحدة.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("MemorySystem")
        
        # إنشاء أنواع الذاكرة
        self.episodic = EpisodicMemory(
            capacity=config.get("episodic_capacity", 10000)
        )
        self.semantic = SemanticMemory(
            capacity=config.get("semantic_capacity", 5000),
            db_path=config.get("semantic_db", "data/semantic_memory.db")
        )
        self.working = WorkingMemory(
            capacity=config.get("working_capacity", 100)
        )
        
        # إحصائيات
        self._stats = {
            "total_stores": 0,
            "total_retrievals": 0,
            "cache_hits": 0
        }
    
    async def initialize(self):
        """تهيئة نظام الذاكرة."""
        self.logger.info("تهيئة نظام الذاكرة...")
        
        # تحميل الذاكرة الدلالية
        await self.semantic.load_from_db()
        
        self.logger.info("تم تهيئة نظام الذاكرة")
    
    async def remember_trade(self, trade: Dict[str, Any]) -> str:
        """تذكر صفقة."""
        entry_id = self._generate_id(trade)
        
        entry = MemoryEntry(
            id=entry_id,
            type=MemoryType.EPISODIC,
            content=trade,
            timestamp=datetime.utcnow(),
            importance=self._calculate_importance(trade),
            metadata={
                "symbol": trade.get("symbol"),
                "action": trade.get("action"),
                "result": trade.get("result")
            }
        )
        
        await self.episodic.store(entry)
        self._stats["total_stores"] += 1
        
        return entry_id
    
    async def remember_analysis(self, analysis: Dict[str, Any]) -> str:
        """تذكر تحليل."""
        entry_id = self._generate_id(analysis)
        
        entry = MemoryEntry(
            id=entry_id,
            type=MemoryType.EPISODIC,
            content=analysis,
            timestamp=datetime.utcnow(),
            importance=analysis.get("confidence", 0.5),
            metadata={
                "symbol": analysis.get("symbol"),
                "type": analysis.get("type")
            }
        )
        
        await self.episodic.store(entry)
        return entry_id
    
    async def learn_pattern(self, pattern: Dict[str, Any]) -> str:
        """تعلم نمط جديد."""
        entry_id = self._generate_id(pattern)
        
        entry = MemoryEntry(
            id=entry_id,
            type=MemoryType.SEMANTIC,
            content=pattern,
            timestamp=datetime.utcnow(),
            importance=pattern.get("reliability", 0.5),
            metadata={
                "category": "patterns",
                "pattern_type": pattern.get("type")
            }
        )
        
        await self.semantic.store(entry)
        return entry_id
    
    async def recall_similar_trades(self, context: Dict[str, Any],
                                   k: int = 5) -> List[Dict]:
        """استرجاع صفقات مشابهة."""
        self._stats["total_retrievals"] += 1
        
        entries = await self.episodic.retrieve(context, k)
        return [e.content for e in entries]
    
    async def recall_patterns(self, pattern_type: str = None,
                             k: int = 10) -> List[Dict]:
        """استرجاع أنماط."""
        query = "patterns" if pattern_type is None else pattern_type
        entries = await self.semantic.retrieve(query, k)
        return [e.content for e in entries]
    
    async def get_symbol_history(self, symbol: str,
                                n: int = 20) -> List[Dict]:
        """الحصول على تاريخ رمز."""
        entries = await self.episodic.retrieve_by_symbol(symbol, n)
        return [e.content for e in entries]
    
    async def update_context(self, key: str, value: Any):
        """تحديث السياق الحالي."""
        self.working.set_context(key, value)
    
    async def get_context(self, key: str) -> Any:
        """الحصول على السياق."""
        return self.working.get_context(key)
    
    async def consolidate_all(self):
        """تجميع جميع الذاكرات."""
        self.logger.info("تجميع الذاكرات...")
        
        await self.episodic.consolidate()
        await self.semantic.consolidate()
        await self.working.consolidate()
        
        self.logger.info("تم تجميع الذاكرات")
    
    async def get_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات."""
        return {
            **self._stats,
            "episodic_size": len(self.episodic._episodes),
            "semantic_size": len(self.semantic._knowledge),
            "working_size": len(self.working._buffer)
        }
    
    def _generate_id(self, content: Any) -> str:
        """توليد معرف فريد."""
        content_str = json.dumps(content, sort_keys=True, default=str)
        timestamp = datetime.utcnow().isoformat()
        return hashlib.md5(f"{content_str}{timestamp}".encode()).hexdigest()[:16]
    
    def _calculate_importance(self, trade: Dict) -> float:
        """حساب أهمية الصفقة."""
        importance = 0.5
        
        # زيادة الأهمية للصفقات الكبيرة
        pnl = trade.get("pnl_percentage", 0)
        if abs(pnl) > 5:
            importance += 0.2
        
        # زيادة الأهمية للصفقات الناجحة جداً أو الفاشلة جداً
        if pnl > 10 or pnl < -10:
            importance += 0.2
        
        return min(1.0, importance)
