#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
辩论智能体 - 正方A和反方B
=========================

本模块实现辩论系统中的两个辩论智能体：
- DebateAgentA: 正方，支持辩题主张
- DebateAgentB: 反方，反对辩题主张

通过 DebateAgent 基类和 stance 参数实现代码复用，
两个具体类只是对立场的简单封装。

使用示例
--------
>>> from agents import DebateAgentA, DebateAgentB
>>> agent_a = DebateAgentA()
>>> agent_b = DebateAgentB()
>>> 
>>> # 第一轮发言 (无对方观点)
>>> view_a = agent_a.debate("人工智能是否会取代人类工作")
>>> 
>>> # 后续轮次 (有对方观点，需反驳)
>>> view_b = agent_b.debate("人工智能是否会取代人类工作", opponent_view=view_a)
"""

from typing import Optional
from .base_agent import BaseAgent


class DebateAgent(BaseAgent):
    """
    辩论智能体基类
    
    通过 stance 参数控制正方/反方立场，实现代码复用。
    正方和反方共享相同的辩论逻辑，只有立场和提示词不同。
    
    Attributes
    ----------
    stance : str
        立场标识，"pro" 表示正方，"con" 表示反方
    config : dict
        当前立场的配置信息
        
    Class Attributes
    ----------------
    STANCES : dict
        立场配置字典，包含正方和反方的所有配置
    """
    
    # =========================================================================
    # 立场配置 - 定义正方和反方的差异化设置
    # =========================================================================
    STANCES = {
        "pro": {
            "name": "智能体A",
            "role": "辩论者A - 正方",
            "position": "正方",
            "opponent": "反方",
            "action": "支持",
            "search_keywords": "优势 好处 支持",
            "examples": [
                '辩题"是否应该堕胎" → 你支持堕胎权',
                '辩题"AI是否会取代人类" → 你认为会取代',
                '辩题"应该禁止吸烟吗" → 你支持禁止'
            ]
        },
        "con": {
            "name": "智能体B",
            "role": "辩论者B - 反方",
            "position": "反方",
            "opponent": "正方",
            "action": "反对",
            "search_keywords": "风险 问题 反对",
            "examples": [
                '辩题"是否应该堕胎" → 你反对堕胎权',
                '辩题"AI是否会取代人类" → 你认为不会取代',
                '辩题"应该禁止吸烟吗" → 你反对禁止'
            ]
        }
    }
    
    def __init__(self, stance: str = "pro", use_search: bool = True, use_rag: bool = True):
        """
        初始化辩论智能体
        
        Parameters
        ----------
        stance : str, optional
            立场选择 (默认: "pro")
            - "pro": 正方，支持辩题主张
            - "con": 反方，反对辩题主张
        use_search : bool, optional
            是否启用网络搜索 (默认: True)
        use_rag : bool, optional
            是否启用 RAG 知识库 (默认: True)
        """
        self.stance = stance
        self.config = self.STANCES[stance]
        super().__init__(
            name=self.config["name"],
            role=self.config["role"],
            use_search=use_search,
            use_rag=use_rag
        )
    
    def _build_system_prompt(self) -> str:
        """
        构建系统提示词
        
        根据立场配置生成对应的角色提示词，
        明确智能体的立场、辩论原则和输出格式。
        """
        cfg = self.config
        examples = "\n".join(f"- {e}" for e in cfg["examples"])
        
        return f"""你是辩论智能体，担任本次辩论的【{cfg["position"]}】立场。

【核心身份】
你是{cfg["position"]}辩手，必须{cfg["action"]}辩题中的主张。无论辩题是什么，你都要从{cfg["action"]}的角度进行论证。
例如：
{examples}

【辩论原则】
1. 坚定{cfg["position"]}：你必须坚持{cfg["position"]}立场，{cfg["action"]}辩题中的主张
2. 聚焦主题：所有论述必须紧扣原始问题，不要偏离话题
3. 针锋相对：如果有对方观点，直接反驳其论点，指出漏洞
4. 有限让步：可以承认对方某些细节合理，但绝不动摇核心立场
5. 回应裁判：如果有裁判反馈，认真对待并回应

请用中文回答，语言犀利有力，坚定捍卫{cfg["position"]}立场！"""
    
    def debate(self, topic: str, opponent_view: Optional[str] = None,
               use_tools: bool = False, judge_feedback: Optional[str] = None) -> str:
        """
        进行辩论发言
        
        Parameters
        ----------
        topic : str
            辩论主题
        opponent_view : str, optional
            对方的观点 (首轮发言时为 None)
        use_tools : bool, optional
            是否使用搜索/RAG工具获取信息 (默认: False)
        judge_feedback : str, optional
            裁判的评判和建议 (首轮发言时为 None)
            
        Returns
        -------
        str
            生成的辩论发言内容
        """
        cfg = self.config
        context = ""
        
        # 使用工具获取外部信息作为论据支撑
        if use_tools:
            if self.search_tool:
                context += self.search(f"{topic} {cfg['search_keywords']}")
            if self.rag_tool:
                context += self.retrieve(topic)
        
        # 根据是否有对方观点构建不同的提示
        if opponent_view:
            # 后续轮次：有对方观点，需要反驳
            prompt = f"【辩论主题】{topic}\n\n【{cfg['opponent']}的观点】\n{opponent_view}\n"
            if judge_feedback:
                prompt += f"\n【裁判C的评判和建议】\n{judge_feedback}\n"
            prompt += f"""
你是{cfg['position']}，请按以下格式回复：
1. 【我的立场】：明确表态{cfg['action']}辩题主张
2. 【核心论据】：2-3个有力论据
3. 【反驳对方】：针对{cfg['opponent']}观点的直接反驳"""
            if judge_feedback:
                prompt += f"\n4. 【回应裁判】：针对裁判反馈的回应"
        else:
            # 第一轮发言：陈述立场和论据，无需反驳
            prompt = f"""【辩论主题】{topic}

你是{cfg['position']}，这是你的首次发言，请按以下格式回复：
1. 【我的立场】：明确表态{cfg['action']}辩题主张
2. 【核心论据】：2-3个{cfg['action']}辩题的有力论据

注意：这是首轮发言，不需要反驳对方或回应裁判。"""
        
        return self.generate(prompt, context)


# =============================================================================
# 具体智能体类 - 为便于使用提供的简单封装
# =============================================================================

class DebateAgentA(DebateAgent):
    """
    正方辩论智能体
    
    支持辩题主张，从积极/肯定的角度进行论证。
    
    使用示例
    --------
    >>> agent = DebateAgentA()
    >>> view = agent.debate("人工智能是否会取代人类工作")
    """
    def __init__(self, use_search: bool = True, use_rag: bool = True):
        super().__init__(stance="pro", use_search=use_search, use_rag=use_rag)


class DebateAgentB(DebateAgent):
    """
    反方辩论智能体
    
    反对辩题主张，从审慎/否定的角度进行论证。
    
    使用示例
    --------
    >>> agent = DebateAgentB()
    >>> view = agent.debate("人工智能是否会取代人类工作", opponent_view="...")
    """
    def __init__(self, use_search: bool = True, use_rag: bool = True):
        super().__init__(stance="con", use_search=use_search, use_rag=use_rag)
