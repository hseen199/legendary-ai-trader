"""
Legendary Trading System - LLM Integration
نظام التداول الخارق - تكامل نماذج اللغة الكبيرة

يوفر واجهة موحدة للتفاعل مع LLMs.
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import logging
import json
from enum import Enum
from openai import AsyncOpenAI


class LLMProvider(Enum):
    """مزودي LLM"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """إعدادات LLM"""
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: str = ""
    timeout: int = 60
    retry_count: int = 3
    rate_limit_delay: float = 1.0


@dataclass
class LLMResponse:
    """استجابة LLM"""
    content: str
    model: str
    tokens_used: int
    latency: float
    success: bool
    error: Optional[str] = None


class LLMClient:
    """
    عميل LLM الموحد.
    
    يدعم:
    - OpenAI GPT-4
    - تحليل السوق
    - توليد التقارير
    - المناظرات
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger("LLMClient")
        
        # تهيئة العميل
        api_key = config.api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = AsyncOpenAI(api_key=api_key)
        
        # إحصائيات
        self._total_requests = 0
        self._total_tokens = 0
        self._errors = 0
    
    async def chat(self, messages: List[Dict[str, str]],
                  system_prompt: str = None,
                  **kwargs) -> LLMResponse:
        """
        إرسال رسالة للمحادثة.
        
        Args:
            messages: قائمة الرسائل
            system_prompt: رسالة النظام
            **kwargs: معاملات إضافية
            
        Returns:
            استجابة LLM
        """
        start_time = datetime.utcnow()
        
        # إعداد الرسائل
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        # المحاولة مع إعادة المحاولة
        for attempt in range(self.config.retry_count):
            try:
                response = await self.client.chat.completions.create(
                    model=kwargs.get("model", self.config.model),
                    messages=full_messages,
                    temperature=kwargs.get("temperature", self.config.temperature),
                    max_tokens=kwargs.get("max_tokens", self.config.max_tokens)
                )
                
                latency = (datetime.utcnow() - start_time).total_seconds()
                
                self._total_requests += 1
                self._total_tokens += response.usage.total_tokens
                
                return LLMResponse(
                    content=response.choices[0].message.content,
                    model=response.model,
                    tokens_used=response.usage.total_tokens,
                    latency=latency,
                    success=True
                )
                
            except Exception as e:
                self.logger.error(f"خطأ في LLM (محاولة {attempt + 1}): {e}")
                self._errors += 1
                
                if attempt < self.config.retry_count - 1:
                    await asyncio.sleep(self.config.rate_limit_delay * (attempt + 1))
        
        return LLMResponse(
            content="",
            model=self.config.model,
            tokens_used=0,
            latency=(datetime.utcnow() - start_time).total_seconds(),
            success=False,
            error="فشل بعد عدة محاولات"
        )
    
    async def analyze_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        تحليل السوق باستخدام LLM.
        
        Args:
            market_data: بيانات السوق
            
        Returns:
            تحليل السوق
        """
        system_prompt = """
        أنت محلل أسواق مالية خبير متخصص في العملات المشفرة.
        قم بتحليل البيانات المقدمة وأعطِ رأيك المهني.
        أجب بصيغة JSON فقط.
        """
        
        user_message = f"""
        حلل بيانات السوق التالية:
        
        {json.dumps(market_data, indent=2, ensure_ascii=False)}
        
        أجب بصيغة JSON:
        {{
            "overall_sentiment": "bullish" أو "bearish" أو "neutral",
            "confidence": رقم من 0 إلى 1,
            "key_observations": ["ملاحظة 1", "ملاحظة 2"],
            "risks": ["خطر 1", "خطر 2"],
            "opportunities": ["فرصة 1", "فرصة 2"],
            "recommendation": "buy" أو "sell" أو "hold",
            "reasoning": "التفسير"
        }}
        """
        
        response = await self.chat(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            temperature=0.3
        )
        
        if response.success:
            try:
                # استخراج JSON
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                return json.loads(content)
            except json.JSONDecodeError:
                return {"error": "فشل تحليل الاستجابة", "raw": response.content}
        
        return {"error": response.error}
    
    async def analyze_news(self, news_items: List[Dict]) -> Dict[str, Any]:
        """
        تحليل الأخبار.
        
        Args:
            news_items: قائمة الأخبار
            
        Returns:
            تحليل الأخبار
        """
        system_prompt = """
        أنت محلل أخبار متخصص في العملات المشفرة.
        قم بتحليل الأخبار وتحديد تأثيرها على السوق.
        """
        
        news_text = "\n".join([
            f"- {item.get('title', '')}: {item.get('summary', '')[:200]}"
            for item in news_items[:10]
        ])
        
        user_message = f"""
        حلل الأخبار التالية:
        
        {news_text}
        
        أجب بصيغة JSON:
        {{
            "overall_impact": "positive" أو "negative" أو "neutral",
            "impact_score": رقم من -1 إلى 1,
            "key_events": ["حدث 1", "حدث 2"],
            "affected_coins": ["عملة 1", "عملة 2"],
            "time_horizon": "short" أو "medium" أو "long",
            "summary": "ملخص"
        }}
        """
        
        response = await self.chat(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            temperature=0.3
        )
        
        if response.success:
            try:
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                return json.loads(content)
            except:
                return {"error": "فشل تحليل الاستجابة"}
        
        return {"error": response.error}
    
    async def generate_trade_reasoning(self, decision: Dict[str, Any],
                                       analyses: List[Dict]) -> str:
        """
        توليد تفسير لقرار التداول.
        
        Args:
            decision: قرار التداول
            analyses: التحليلات
            
        Returns:
            التفسير
        """
        system_prompt = """
        أنت متداول محترف. اشرح قرار التداول بشكل واضح ومختصر.
        """
        
        user_message = f"""
        قرار التداول: {decision.get('action', 'hold')} على {decision.get('symbol', '')}
        
        التحليلات:
        {json.dumps(analyses, indent=2, ensure_ascii=False)[:2000]}
        
        اكتب تفسيراً مختصراً (3-5 جمل) لهذا القرار.
        """
        
        response = await self.chat(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=500
        )
        
        return response.content if response.success else "تعذر توليد التفسير"
    
    async def conduct_debate(self, topic: str,
                            bullish_args: List[str],
                            bearish_args: List[str],
                            rounds: int = 3) -> Dict[str, Any]:
        """
        إجراء مناظرة بين وجهتي نظر.
        
        Args:
            topic: موضوع المناظرة
            bullish_args: الحجج الصعودية
            bearish_args: الحجج الهبوطية
            rounds: عدد الجولات
            
        Returns:
            نتيجة المناظرة
        """
        debate_history = []
        
        for round_num in range(rounds):
            # رد المتفائل
            bullish_response = await self._debate_turn(
                "bullish", topic, bullish_args, bearish_args, debate_history
            )
            debate_history.append({"side": "bullish", "argument": bullish_response})
            
            # رد المتشائم
            bearish_response = await self._debate_turn(
                "bearish", topic, bearish_args, bullish_args, debate_history
            )
            debate_history.append({"side": "bearish", "argument": bearish_response})
        
        # الحكم النهائي
        verdict = await self._judge_debate(topic, debate_history)
        
        return {
            "topic": topic,
            "rounds": rounds,
            "debate_history": debate_history,
            "verdict": verdict
        }
    
    async def _debate_turn(self, side: str, topic: str,
                          own_args: List[str], opponent_args: List[str],
                          history: List[Dict]) -> str:
        """جولة مناظرة واحدة."""
        stance = "متفائل/صعودي" if side == "bullish" else "متشائم/هبوطي"
        
        system_prompt = f"""
        أنت محلل {stance} في مناظرة حول {topic}.
        دافع عن موقفك بحجج قوية ومنطقية.
        """
        
        history_text = "\n".join([
            f"{'المتفائل' if h['side'] == 'bullish' else 'المتشائم'}: {h['argument']}"
            for h in history[-4:]
        ])
        
        user_message = f"""
        الموضوع: {topic}
        
        حججك الأساسية:
        {chr(10).join(f'- {arg}' for arg in own_args[:3])}
        
        حجج الخصم:
        {chr(10).join(f'- {arg}' for arg in opponent_args[:3])}
        
        تاريخ المناظرة:
        {history_text}
        
        قدم حجتك التالية في فقرة واحدة (50-100 كلمة).
        """
        
        response = await self.chat(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=300
        )
        
        return response.content if response.success else f"حجة {stance} غير متاحة"
    
    async def _judge_debate(self, topic: str, history: List[Dict]) -> Dict[str, Any]:
        """الحكم على المناظرة."""
        system_prompt = """
        أنت حكم محايد في مناظرة مالية.
        قيّم الحجج بموضوعية وأعطِ حكمك النهائي.
        """
        
        history_text = "\n".join([
            f"{'المتفائل' if h['side'] == 'bullish' else 'المتشائم'}: {h['argument']}"
            for h in history
        ])
        
        user_message = f"""
        الموضوع: {topic}
        
        المناظرة:
        {history_text}
        
        أجب بصيغة JSON:
        {{
            "winner": "bullish" أو "bearish" أو "tie",
            "confidence": رقم من 0 إلى 1,
            "key_points": ["نقطة 1", "نقطة 2"],
            "final_recommendation": "buy" أو "sell" أو "hold",
            "reasoning": "التفسير"
        }}
        """
        
        response = await self.chat(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            temperature=0.3
        )
        
        if response.success:
            try:
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                return json.loads(content)
            except:
                return {"winner": "tie", "confidence": 0.5}
        
        return {"winner": "tie", "confidence": 0.5, "error": response.error}
    
    async def generate_report(self, data: Dict[str, Any],
                             report_type: str = "daily") -> str:
        """
        توليد تقرير.
        
        Args:
            data: بيانات التقرير
            report_type: نوع التقرير
            
        Returns:
            التقرير
        """
        system_prompt = """
        أنت كاتب تقارير مالية محترف.
        اكتب تقريراً واضحاً ومهنياً.
        """
        
        user_message = f"""
        اكتب تقرير {report_type} بناءً على البيانات التالية:
        
        {json.dumps(data, indent=2, ensure_ascii=False)[:3000]}
        
        يجب أن يتضمن التقرير:
        1. ملخص تنفيذي
        2. أداء المحفظة
        3. الصفقات الرئيسية
        4. تحليل السوق
        5. التوصيات
        """
        
        response = await self.chat(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=2000
        )
        
        return response.content if response.success else "تعذر توليد التقرير"
    
    def get_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات."""
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "errors": self._errors,
            "success_rate": (self._total_requests - self._errors) / max(1, self._total_requests)
        }


class LLMManager:
    """
    مدير LLM.
    
    يدير عدة عملاء LLM ويوزع الطلبات.
    """
    
    def __init__(self, configs: List[LLMConfig] = None):
        self.logger = logging.getLogger("LLMManager")
        
        if configs is None:
            configs = [LLMConfig()]
        
        self.clients = [LLMClient(config) for config in configs]
        self._current_client = 0
    
    def get_client(self) -> LLMClient:
        """الحصول على عميل."""
        client = self.clients[self._current_client]
        self._current_client = (self._current_client + 1) % len(self.clients)
        return client
    
    async def chat(self, *args, **kwargs) -> LLMResponse:
        """إرسال رسالة."""
        return await self.get_client().chat(*args, **kwargs)
    
    async def analyze_market(self, *args, **kwargs) -> Dict[str, Any]:
        """تحليل السوق."""
        return await self.get_client().analyze_market(*args, **kwargs)
