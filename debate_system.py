#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
辩论系统 - 核心调度模块
=======================

本模块是智能辩论系统的核心，负责协调多个智能体进行辩论。

核心功能
--------
- 创建并管理辩论智能体 (正方A、反方B、裁判C)
- 调度多轮辩论流程
- 检测共识达成情况
- 生成辩论总结

辩论流程
--------
1. 初始化系统，加载模型和智能体
2. 用户输入辩论主题
3. 每轮辩论: A发言 → B发言 → 裁判评判
4. 检查共识度，决定是否继续
5. 生成最终总结 (共识点 + 分歧点)

使用示例
--------
>>> from debate_system import DebateSystem
>>> system = DebateSystem()
>>> result = system.run_debate("人工智能是否会取代人类工作？")
>>> print(result['final_summary'])

>>> # 交互模式
>>> system.interactive_mode()
"""

import gc
from typing import List, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents import DebateAgentA, DebateAgentB, JudgeAgent
from config import MAX_DEBATE_ROUNDS, CONSENSUS_THRESHOLD
from exceptions import DebateSystemError, AgentError


class DebateSystem:
    """
    智能辩论系统 - 核心调度类
    
    协调正方智能体A、反方智能体B和裁判C进行多轮结构化辩论，
    自动检测共识并生成总结报告。
    
    Attributes
    ----------
    agent_a : DebateAgentA
        正方辩论智能体，支持辩题主张
    agent_b : DebateAgentB
        反方辩论智能体，反对辩题主张
    judge : JudgeAgent
        裁判智能体，评判双方表现
    debate_history : List[Dict]
        辩论历史记录
    console : Console
        Rich 控制台对象，用于美化输出
        
    使用示例
    --------
    >>> # 基础用法
    >>> system = DebateSystem()
    >>> result = system.run_debate("人工智能是否会取代人类工作？")
    >>> print(result['final_summary'])
    
    >>> # 禁用工具，纯辩论模式
    >>> system = DebateSystem(use_search=False, use_rag=False)
    
    >>> # 交互模式，连续进行多场辩论
    >>> system.interactive_mode()
    """
    
    def __init__(self, use_search: bool = True, use_rag: bool = True):
        """
        初始化辩论系统
        
        创建三个智能体实例，它们共享同一个语言模型以节省资源。
        
        Parameters
        ----------
        use_search : bool, optional
            是否为智能体启用网络搜索功能 (默认: True)
        use_rag : bool, optional
            是否为智能体启用 RAG 知识库检索 (默认: True)
        """
        self.console = Console()
        self.console.print("\n[bold blue]🎯 初始化智能辩论系统...[/bold blue]\n")
        
        # 加载智能体 (使用进度指示器)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("正在加载模型...", total=None)
            
            # 创建三个智能体，共享同一个模型实例 (单例模式)
            self.agent_a = DebateAgentA(use_search=use_search, use_rag=use_rag)
            self.agent_b = DebateAgentB(use_search=use_search, use_rag=use_rag)
            self.judge = JudgeAgent(use_search=use_search, use_rag=use_rag)
            
            progress.update(task, description="✓ 智能体加载完成!")
        
        # 辩论历史记录
        self.debate_history: List[Dict] = []
    
    def add_knowledge(self, documents: List[str]):
        """
        向所有智能体的知识库添加文档
        
        Parameters
        ----------
        documents : List[str]
            文档字符串列表，每个字符串为一个文档块
        """
        self.agent_a.add_knowledge(documents)
        self.agent_b.add_knowledge(documents)
        self.judge.add_knowledge(documents)
        self.console.print("[green]✓ 知识库已更新[/green]")
    
    def run_debate(self, topic: str, max_rounds: int = MAX_DEBATE_ROUNDS,
                   use_tools: bool = False, early_stop: bool = True) -> Dict:
        """
        运行一场完整的辩论
        
        Parameters
        ----------
        topic : str
            辩论主题，可以是疑问句或陈述句
        max_rounds : int, optional
            最大辩论轮数 (默认: 5)
        use_tools : bool, optional
            是否在辩论中使用搜索/RAG工具 (默认: False)
        early_stop : bool, optional
            达成共识后是否提前结束 (默认: True)
            
        Returns:
            辩论结果字典，包含 topic, rounds, history, final_summary
        """
        self.debate_history = []
        
        # 重置智能体状态
        self.agent_a.reset_history()
        self.agent_b.reset_history()
        self.judge.reset()
        
        # 显示主题
        self.console.print(Panel(
            f"[bold]{topic}[/bold]",
            title="[bold magenta]📋 辩论主题[/bold magenta]",
            border_style="magenta"
        ))
        
        view_a, view_b, last_eval = None, None, None
        
        for round_num in range(1, max_rounds + 1):
            self._print_round_header(round_num)
            
            # 获取历史记录（当前轮之前的所有记录）
            history_for_agents = self.debate_history.copy() if self.debate_history else None
            
            # 智能体A发言
            self.console.print("[bold green]【🅰️ 智能体A发言】[/bold green]")
            with self.console.status("[green]思考中...[/green]"):
                view_a = self.agent_a.debate(
                    topic, opponent_view=view_b, 
                    use_tools=use_tools, judge_feedback=last_eval,
                    debate_history=history_for_agents
                )
            self.console.print(Panel(Markdown(view_a), title="智能体A", border_style="green"))
            
            # 智能体B发言
            self.console.print("\n[bold yellow]【🅱️ 智能体B发言】[/bold yellow]")
            with self.console.status("[yellow]思考中...[/yellow]"):
                view_b = self.agent_b.debate(
                    topic, opponent_view=view_a,
                    use_tools=use_tools, judge_feedback=last_eval,
                    debate_history=history_for_agents
                )
            self.console.print(Panel(Markdown(view_b), title="智能体B", border_style="yellow"))
            
            # 裁判评判
            self.console.print("\n[bold red]【⚖️ 裁判C评判】[/bold red]")
            is_final_round = (round_num == max_rounds)
            with self.console.status("[red]评判中...[/red]"):
                evaluation, guidance = self.judge.evaluate_round(
                    topic, view_a, view_b, round_num, is_final_round=is_final_round
                )
            self.console.print(Panel(Markdown(evaluation), title="裁判评判", border_style="red"))
            
            last_eval = evaluation
            
            # 记录本轮
            self.debate_history.append({
                "round": round_num,
                "agent_a": view_a,
                "agent_b": view_b,
                "evaluation": evaluation
            })
            
            # 每轮后清理内存，防止 OOM
            gc.collect()
            
            # 检查共识
            if early_stop and round_num > 1:
                _, score, _ = self.judge.check_consensus(view_a, view_b)
                if score >= CONSENSUS_THRESHOLD:
                    self.console.print(f"\n[bold green]✓ 双方达成共识 (共识度: {score*100:.0f}%)[/bold green]")
                    break
                self.console.print(f"\n[dim]共识度: {score*100:.0f}%[/dim]")
            
            # 显示下轮引导（非最后一轮才显示）
            if round_num < max_rounds and guidance:
                self.console.print(Panel(Markdown(guidance), title="下轮引导", border_style="blue"))
        
        # 生成最终总结
        self._print_summary_header()
        with self.console.status("[magenta]生成总结...[/magenta]"):
            final_summary = self.judge.generate_final_summary(topic)
        self.console.print(Panel(
            Markdown(final_summary),
            title="[bold]📊 最终总结 - 共识与分歧[/bold]",
            border_style="magenta"
        ))
        
        return {
            "topic": topic,
            "rounds": len(self.debate_history),
            "history": self.debate_history,
            "final_summary": final_summary
        }
    
    def _print_round_header(self, round_num: int):
        """打印轮次标题"""
        self.console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        self.console.print(f"[bold cyan]🔄 第 {round_num} 轮辩论[/bold cyan]")
        self.console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")
    
    def _print_summary_header(self):
        """打印总结标题"""
        self.console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
        self.console.print("[bold magenta]📊 辩论总结[/bold magenta]")
        self.console.print(f"[bold magenta]{'='*60}[/bold magenta]\n")
    
    def interactive_mode(self):
        """交互模式 - 用户输入主题进行辩论"""
        self.console.print("\n[bold blue]🎯 欢迎使用智能辩论系统![/bold blue]")
        self.console.print("输入辩论主题开始辩论，输入 'quit' 退出\n")
        
        while True:
            topic = self.console.input("[green]请输入辩论主题: [/green]").strip()
            
            if topic.lower() in ['quit', 'exit', '退出', 'q']:
                self.console.print("[blue]再见! 👋[/blue]")
                break
            
            if not topic:
                continue
            
            # 询问参数
            rounds = self.console.input("[dim]辩论轮数 (默认5): [/dim]").strip()
            max_rounds = int(rounds) if rounds.isdigit() else MAX_DEBATE_ROUNDS
            
            # 运行辩论
            self.run_debate(topic=topic, max_rounds=max_rounds)
            self.console.print("\n[dim]辩论结束，可继续输入新主题[/dim]\n")
