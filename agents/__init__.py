"""
智能体模块 (agents)
===================

本模块包含辩论系统中所有智能体的实现。

类列表
------
BaseAgent
    智能体抽象基类，定义通用接口
DebateAgentA
    正方辩论智能体，支持辩题主张
DebateAgentB  
    反方辩论智能体，反对辩题主张
JudgeAgent
    裁判智能体，评判双方表现并引导共识

使用示例
--------
>>> from agents import DebateAgentA, DebateAgentB, JudgeAgent
>>> agent_a = DebateAgentA()
>>> agent_b = DebateAgentB()
>>> judge = JudgeAgent()
"""

from .base_agent import BaseAgent
from .debate_agents import DebateAgentA, DebateAgentB
from .judge_agent import JudgeAgent

__all__ = ['BaseAgent', 'DebateAgentA', 'DebateAgentB', 'JudgeAgent']
