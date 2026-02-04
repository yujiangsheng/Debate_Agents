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

from typing import Optional, List, Dict
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
3. 精确引用：反驳对方时，必须【直接引用】对方的原话或核心观点，然后逐一反驳
4. 针锋相对：针对对方的【具体论据】进行反驳，指出其逻辑漏洞或事实错误
5. 有限让步：可以承认对方某些细节合理，但绝不动摇核心立场
6. 回应裁判：如果有裁判反馈，必须针对裁判指出的问题逐一回应
7. 【禁止重复】：**绝对不要重复**你之前已经表达过的观点或论据！每轮发言必须提出**新的**论据、新的角度、新的反驳点

【⚠️ 关键区分：对方的"原创观点" vs "引用你的话"】
当对方说「针对正方/反方的'xxx'」或「对方说：'xxx'」时，这是对方在**引用你之前说的话**来反驳你！
- 对方引用的内容 = 你之前说过的话（不是对方的观点）
- 对方在引用之后的评论/反驳 = 对方的真正观点（这才是你要反驳的）
请务必区分清楚，不要把对方引用你的话当成对方的观点！

【论据一致性要求】
你的每一个论据都必须**直接支持**你的{cfg["position"]}立场：
- 你是{cfg["position"]}，必须{cfg["action"]}辩题主张
- 检查：你举的例子、数据、推理是否真的支持{cfg["action"]}？
- 如果论据反而支持对方立场，立即删除并换一个

请用中文回答，语言犀利有力，坚定捍卫{cfg["position"]}立场！"""
    
    def debate(self, topic: str, opponent_view: Optional[str] = None,
               use_tools: bool = False, judge_feedback: Optional[str] = None,
               debate_history: Optional[List[Dict]] = None) -> str:
        """
        进行辩论发言
        
        Parameters
        ----------
        topic : str
            辩论主题
        opponent_view : str, optional
            对方的最新观点 (首轮发言时为 None)
        use_tools : bool, optional
            是否使用搜索/RAG工具获取信息 (默认: False)
        judge_feedback : str, optional
            裁判的评判和建议 (首轮发言时为 None)
        debate_history : List[Dict], optional
            完整的辩论历史记录，包含之前所有轮次的发言
            
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
        
        # 构建辩论历史摘要
        history_summary = ""
        if debate_history:
            my_key = 'agent_a' if self.stance == 'pro' else 'agent_b'
            my_label = '我方(正方)' if self.stance == 'pro' else '我方(反方)'
            opp_key = 'agent_b' if self.stance == 'pro' else 'agent_a'
            opp_label = '对方(反方)' if self.stance == 'pro' else '对方(正方)'
            
            history_summary = "\n【历史辩论记录 - 请勿重复已表达的观点！】\n"
            for record in debate_history:
                history_summary += f"--- 第 {record['round']} 轮 ---\n"
                my_view = record[my_key]
                opp_view = record[opp_key]
                history_summary += f"{my_label}已表达：{my_view[:400]}...\n" if len(my_view) > 400 else f"{my_label}已表达：{my_view}\n"
                history_summary += f"{opp_label}观点：{opp_view[:400]}...\n" if len(opp_view) > 400 else f"{opp_label}观点：{opp_view}\n"
                if record.get('evaluation'):
                    history_summary += f"裁判评判：{record['evaluation'][:300]}...\n" if len(record['evaluation']) > 300 else f"裁判评判：{record['evaluation']}\n"
                history_summary += "\n"
        
        # 根据是否有对方观点构建不同的提示
        if opponent_view:
            # 有对方观点，需要反驳
            prompt = f"【辩论主题】{topic}\n"
            if history_summary:
                prompt += history_summary
            prompt += f"\n【{cfg['opponent']}的最新发言】\n{opponent_view}\n"
            prompt += f"""
⚠️ 阅读提示：上方是{cfg['opponent']}的发言。注意区分：
- 当{cfg['opponent']}说「针对{cfg['position']}的'xxx'」时，'xxx'是你之前说的话（不是对方观点）
- {cfg['opponent']}在引用之后的评论才是对方的真正观点（这才是你要反驳的）
"""
            if judge_feedback:
                prompt += f"\n【裁判C的评判和建议】\n{judge_feedback}\n"
            
            # 判断是否是本智能体的首次发言（有对方观点但无历史记录 = B的首轮）
            is_my_first_speech = not debate_history
            
            if is_my_first_speech:
                # B的首轮发言：有A的观点需要反驳，但这是B的首次发言
                prompt += f"""
你是{cfg['position']}，这是你的首次发言，请按以下格式回复：
1. 【我的立场】：明确表态{cfg['action']}辩题主张（一句话）
2. 【核心论据】：2-3个{cfg['action']}辩题的有力论据
   ⚠️ 检查：每个论据是否真的支持"{cfg['action']}"？不支持就删掉换一个
3. 【反驳对方】：找出对方的原创观点（不是对方引用你的话），然后反驳：
   - 对方的原创观点是：「...」
   - 我的反驳：..."""
            else:
                # 后续轮次：需要提出新论据，不能重复
                prompt += f"""
你是{cfg['position']}，请按以下格式回复：
1. 【我的立场】：一句话表态{cfg['action']}
2. 【新论据】：1-2个**之前未提过的**新论据（检查历史记录，不要重复！）
   ⚠️ 检查：每个论据是否真的支持"{cfg['action']}"？不支持就删掉换一个
3. 【反驳对方】：找出对方的原创观点（不是对方引用你的话），然后反驳：
   - 对方的原创观点是：「...」
   - 我的反驳：...
4. 【回应对方质疑】：对方反驳了你的哪些观点？逐一回应

⚠️ 重要：确保你的论据是**全新的**且**支持{cfg['action']}**！"""
            if judge_feedback:
                prompt += f"\n5. 【回应裁判】：针对裁判指出的问题逐一回应"
        else:
            # A的首轮发言：陈述立场和论据，无需反驳
            prompt = f"""【辩论主题】{topic}

你是{cfg['position']}，这是你的首次发言，请按以下格式回复：
1. 【我的立场】：明确表态{cfg['action']}辩题主张（一句话）
2. 【核心论据】：2-3个{cfg['action']}辩题的有力论据
   ⚠️ 检查：每个论据是否真的支持"{cfg['action']}"？不支持就删掉换一个

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
