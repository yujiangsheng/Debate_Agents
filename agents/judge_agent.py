#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
裁判智能体
==========

本模块实现辩论系统中的裁判角色。

裁判智能体的职责：
- 评判每轮辩论中双方的表现
- 指出论证漏洞和逻辑问题
- 判定每轮胜负
- 引导双方正面交锋
- 推动达成共识
- 生成最终辩论总结

使用示例
--------
>>> from agents import JudgeAgent
>>> judge = JudgeAgent()
>>> 
>>> # 评判一轮辩论
>>> evaluation, guidance = judge.evaluate_round(
...     topic="AI是否会取代人类",
...     view_a="正方观点...",
...     view_b="反方观点...",
...     round_num=1
... )
>>> 
>>> # 检查共识
>>> has_consensus, score, reason = judge.check_consensus(view_a, view_b)
>>> 
>>> # 生成最终总结
>>> summary = judge.generate_final_summary("AI是否会取代人类")
"""

from typing import List, Dict, Tuple
from .base_agent import BaseAgent


class JudgeAgent(BaseAgent):
    """
    裁判智能体
    
    负责评判辩论双方的表现，风格犀利直接，不和稀泥。
    
    Attributes
    ----------
    debate_records : List[Dict]
        辩论记录，存储每轮双方的发言内容
        
    使用示例
    --------
    >>> judge = JudgeAgent()
    >>> eval_result, guidance = judge.evaluate_round(
    ...     "人工智能是否会取代人类",
    ...     "正方观点...", "反方观点...",
    ...     round_num=1
    ... )
    """
    
    def __init__(self, use_search: bool = True, use_rag: bool = True):
        """
        初始化裁判智能体
        
        Parameters
        ----------
        use_search : bool, optional
            是否启用网络搜索 (默认: True)
        use_rag : bool, optional
            是否启用 RAG 知识库 (默认: True)
        """
        super().__init__(
            name="裁判C",
            role="辩论裁判",
            use_search=use_search,
            use_rag=use_rag
        )
        # 辩论记录，用于生成最终总结
        self.debate_records: List[Dict] = []
    
    def _build_system_prompt(self) -> str:
        """
        构建裁判的系统提示词
        
        定义裁判的角色特点：严厉、尖锐、眼光毒辣，
        不和稀泥，必须明确判定胜负。
        """
        return """你是裁判智能体C，一位严厉、尖锐、眼光毒辣的辩论裁判。

【你的角色】
你不是和事佬，而是一位犀利的评论家。你的任务是：
1. 毫不留情地指出双方论证的漏洞和逻辑谬误
2. 直接点评谁的论据更有力，谁在回避问题
3. 批评偏离主题的发言
4. 迫使双方正面交锋，不要打太极

【评判原则】
1. 聚焦主题：严厉批评任何偏离辩论主题的发言
2. 逻辑优先：论据必须有事实支撑，空洞的口号要被批评
3. 直面交锋：如果一方没有正面回应对方的质疑，要明确指出
4. 不和稀泥：必须明确指出本轮谁更有说服力，不要含糊其辞
5. 推动共识：在批评的同时，提炼双方可能达成的共识点

【评判格式】
1. 【犀利点评】：直接、尖锐地评价双方本轮表现，不留情面
2. 【逻辑漏洞】：指出双方论证中的逻辑问题或事实错误
3. 【本轮胜负】：明确判定本轮谁更有说服力（不能说"各有优劣"这种废话）
4. 【共识进展】：指出双方已经达成或接近达成的共识
5. 【下轮要求】：明确要求双方在下一轮必须回应的问题

请用中文回答，语言要犀利直接，不要客套。"""
    
    def evaluate_round(self, topic: str, view_a: str, view_b: str, 
                       round_num: int, is_final_round: bool = False) -> Tuple[str, str]:
        """
        评判一轮辩论
        
        Parameters
        ----------
        topic : str
            辩论主题
        view_a : str
            智能体A (正方) 的发言内容
        view_b : str
            智能体B (反方) 的发言内容
        round_num : int
            当前轮次编号
        is_final_round : bool, optional
            是否为最后一轮 (默认: False)
            最后一轮不会生成下轮引导
            
        Returns
        -------
        Tuple[str, str]
            (评判内容, 下轮引导)
            如果是最后一轮，下轮引导为空字符串
        """
        # 记录本轮辩论内容
        self.debate_records.append({
            "round": round_num,
            "agent_a": view_a,
            "agent_b": view_b
        })
        
        # 根据是否为最后一轮构建不同的评判提示
        if is_final_round:
            prompt = f"""【辩论主题】{topic}

【第 {round_num} 轮辩论（最后一轮）】

【智能体A的发言】
{view_a}

【智能体B的发言】
{view_b}

这是最后一轮辩论，请作为犀利的裁判进行最终评判:
1. 【犀利点评】：毫不留情地评价双方表现
2. 【逻辑漏洞】：指出双方的逻辑问题
3. 【本轮胜负】：明确判定谁更有说服力（必须选一方）
4. 【共识进展】：指出已达成的共识

注意：这是最后一轮，不需要提出下轮要求。"""
        else:
            prompt = f"""【辩论主题】{topic}

【第 {round_num} 轮辩论】

【智能体A的发言】
{view_a}

【智能体B的发言】
{view_b}

请作为犀利的裁判进行评判:
1. 【犀利点评】：毫不留情地评价双方表现
2. 【逻辑漏洞】：指出双方的逻辑问题
3. 【本轮胜负】：明确判定谁更有说服力（必须选一方）
4. 【共识进展】：指出已达成的共识
5. 【下轮要求】：要求双方必须回应的具体问题"""
        
        evaluation = self.generate(prompt)
        
        # 生成下轮引导（最后一轮不需要）
        guidance = ""
        if not is_final_round:
            guidance_prompt = "基于你的评判，简要给出下一轮的具体指令：A必须回应什么？B必须回应什么？"
            guidance = self.generate(guidance_prompt)
        
        return evaluation, guidance
    
    def check_consensus(self, view_a: str, view_b: str) -> Tuple[bool, float, str]:
        """
        检查双方是否达成共识
        
        分析双方最新观点的一致程度，用于判断是否可以提前结束辩论。
        
        Parameters
        ----------
        view_a : str
            智能体A的最新观点
        view_b : str
            智能体B的最新观点
            
        Returns
        -------
        Tuple[bool, float, str]
            (是否达成共识, 共识度评分(0-1), 详细说明)
        """
        prompt = f"""分析以下两个观点的一致程度:

【智能体A】
{view_a}

【智能体B】
{view_b}

请回答:
一致性: [是/否]
评分: [0-100]
理由: [简要说明]"""
        
        response = self.generate(prompt)
        
        # 解析模型返回的结果
        has_consensus = "是" in response[:50]
        try:
            score_text = response.split("评分")[1].split("\n")[0] if "评分" in response else "50"
            score = float(''.join(c for c in score_text if c.isdigit() or c == '.')) / 100
            score = min(1.0, max(0.0, score))  # 确保在 [0, 1] 范围内
        except:
            score = 0.5  # 解析失败时使用默认值
        
        return has_consensus, score, response
    
    def generate_final_summary(self, topic: str) -> str:
        """
        生成最终辩论总结
        
        汇总整场辩论，提炼共识点和分歧点，给出综合结论。
        
        Parameters
        ----------
        topic : str
            辩论主题
            
        Returns
        -------
        str
            包含共识点、分歧点、综合结论的总结报告
        """
        # 编译所有轮次的记录
        history = ""
        for r in self.debate_records:
            history += f"\n--- 第{r['round']}轮 ---\n"
            history += f"A: {r['agent_a'][:300]}...\n"
            history += f"B: {r['agent_b'][:300]}...\n"
        
        prompt = f"""【辩论主题】{topic}

【辩论记录】
{history}

请作为裁判进行最终总结:

1. 【共识点】双方达成的共识（具体列举）
2. 【分歧点】仍存在的分歧（具体说明）
3. 【综合结论】综合双方观点，给出你认为最合理的结论
4. 【辩论评价】对整场辩论质量的评价"""
        
        return self.generate(prompt)
    
    def reset(self):
        """
        重置裁判状态
        
        在开始新一场辩论前调用，清空辩论记录和对话历史。
        """
        self.debate_records = []
        self.reset_history()
