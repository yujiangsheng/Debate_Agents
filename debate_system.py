"""辩论系统 - 核心调度模块"""
import gc
from typing import List, Dict
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents import DebateAgentA, DebateAgentB, JudgeAgent
from config import MAX_DEBATE_ROUNDS, CONSENSUS_THRESHOLD
from debate_tracker import DebateTracker


class DebateSystem:
    """智能辩论系统 - 协调正方A、反方B和裁判C进行辩论"""
    
    def __init__(self, use_search: bool = True, use_rag: bool = True):
        self.console = Console()
        self.console.print("\n[bold blue]🎯 初始化智能辩论系统...[/bold blue]\n")
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=self.console) as progress:
            task = progress.add_task("正在加载模型...", total=None)
            self.agent_a = DebateAgentA(use_search=use_search, use_rag=use_rag)
            self.agent_b = DebateAgentB(use_search=use_search, use_rag=use_rag)
            self.judge = JudgeAgent(use_search=use_search, use_rag=use_rag)
            progress.update(task, description="✓ 智能体加载完成!")
        
        self.debate_history: List[Dict] = []
    
    def add_knowledge(self, documents: List[str]):
        """向所有智能体的知识库添加文档"""
        self.agent_a.add_knowledge(documents)
        self.agent_b.add_knowledge(documents)
        self.judge.add_knowledge(documents)
        self.console.print("[green]✓ 知识库已更新[/green]")
    
    def run_debate(self, topic: str, max_rounds: int = MAX_DEBATE_ROUNDS,
                   use_tools: bool = False, early_stop: bool = True) -> Dict:
        """运行一场完整的辩论"""
        self.debate_history = []
        self.tracker = DebateTracker(topic)
        
        self.agent_a.reset_history()
        self.agent_b.reset_history()
        self.judge.reset()
        
        self.console.print(Panel(f"[bold]{topic}[/bold]", title="[bold magenta]📋 辩论主题[/bold magenta]", border_style="magenta"))
        
        view_a, view_b, last_eval = None, None, None
        
        for round_num in range(1, max_rounds + 1):
            self.console.print(f"\n[bold cyan]{'='*50}\n🔄 第 {round_num} 轮辩论\n{'='*50}[/bold cyan]\n")
            
            history_for_agents = self.debate_history.copy() if self.debate_history else None
            structured_history_a = self.tracker.get_structured_history_for_agent('agent_a') if round_num > 1 else None
            
            # 智能体A发言
            self.console.print("[bold green]【🅰️ 智能体A发言】[/bold green]")
            with self.console.status("[green]思考中...[/green]"):
                view_a = self.agent_a.debate(topic, opponent_view=view_b, use_tools=use_tools, 
                                              judge_feedback=last_eval, debate_history=history_for_agents,
                                              structured_history=structured_history_a)
            self.console.print(Panel(Markdown(view_a), title="智能体A", border_style="green"))
            self.tracker.add_speech('agent_a', view_a, round_num)
            
            structured_history_b = self.tracker.get_structured_history_for_agent('agent_b')
            
            # 智能体B发言
            self.console.print("\n[bold yellow]【🅱️ 智能体B发言】[/bold yellow]")
            with self.console.status("[yellow]思考中...[/yellow]"):
                view_b = self.agent_b.debate(topic, opponent_view=view_a, use_tools=use_tools,
                                              judge_feedback=last_eval, debate_history=history_for_agents,
                                              structured_history=structured_history_b)
            self.console.print(Panel(Markdown(view_b), title="智能体B", border_style="yellow"))
            self.tracker.add_speech('agent_b', view_b, round_num)
            
            # 裁判评判
            self.console.print("\n[bold red]【⚖️ 裁判C评判】[/bold red]")
            is_final_round = (round_num == max_rounds)
            with self.console.status("[red]评判中...[/red]"):
                evaluation, guidance = self.judge.evaluate_round(topic, view_a, view_b, round_num, is_final_round=is_final_round)
            self.console.print(Panel(Markdown(evaluation), title="裁判评判", border_style="red"))
            
            last_eval = evaluation
            self.debate_history.append({"round": round_num, "agent_a": view_a, "agent_b": view_b, "evaluation": evaluation})
            gc.collect()
            
            # 检查共识
            if early_stop and round_num > 1:
                _, score, _ = self.judge.check_consensus(view_a, view_b)
                if score >= CONSENSUS_THRESHOLD:
                    self.console.print(f"\n[bold green]✓ 双方达成共识 (共识度: {score*100:.0f}%)[/bold green]")
                    break
                self.console.print(f"\n[dim]共识度: {score*100:.0f}%[/dim]")
            
            if round_num < max_rounds and guidance:
                self.console.print(Panel(Markdown(guidance), title="下轮引导", border_style="blue"))
        
        # 生成最终总结
        self.console.print(f"\n[bold magenta]{'='*50}\n📊 辩论总结\n{'='*50}[/bold magenta]\n")
        with self.console.status("[magenta]生成总结...[/magenta]"):
            final_summary = self.judge.generate_final_summary(topic)
        self.console.print(Panel(Markdown(final_summary), title="[bold]📊 最终总结[/bold]", border_style="magenta"))
        
        return {"topic": topic, "rounds": len(self.debate_history), "history": self.debate_history, 
                "final_summary": final_summary, "tracker_data": self.tracker.data}
    
    def interactive_mode(self):
        """交互模式 - 用户输入主题进行辩论"""
        self.console.print("\n[bold blue]🎯 欢迎使用智能辩论系统![/bold blue]\n输入辩论主题开始辩论，输入 'quit' 退出\n")
        
        while True:
            topic = self.console.input("[green]请输入辩论主题: [/green]").strip()
            if topic.lower() in ['quit', 'exit', '退出', 'q']:
                self.console.print("[blue]再见! 👋[/blue]")
                break
            if not topic:
                continue
            
            rounds = self.console.input("[dim]辩论轮数 (默认5): [/dim]").strip()
            max_rounds = int(rounds) if rounds.isdigit() else MAX_DEBATE_ROUNDS
            
            self.run_debate(topic=topic, max_rounds=max_rounds)
            self.console.print("\n[dim]辩论结束，可继续输入新主题[/dim]\n")
