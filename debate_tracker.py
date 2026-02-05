#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
辩论追踪器 - 结构化记录双方观点
===============================

帮助智能体在后续轮次中准确区分"我方"和"对方"的内容。
"""

import json
import re
from typing import Dict


class DebateTracker:
    """结构化记录辩论双方的所有观点"""
    
    def __init__(self, topic: str):
        self.data = {
            "topic": topic,
            "agent_a": {"role": "正方", "stance": "支持辩题主张", "rounds": []},
            "agent_b": {"role": "反方", "stance": "反对辩题主张", "rounds": []}
        }
    
    def parse_speech(self, speech: str, round_num: int) -> Dict:
        """解析发言内容，提取结构化信息"""
        result = {"round": round_num, "position": "", "arguments": []}
        
        # 提取立场
        pos_match = re.search(r'【我的立场】[：:]\s*(.+?)(?=\n\n|\n【|$)', speech, re.DOTALL)
        if pos_match:
            result["position"] = pos_match.group(1).strip()
        
        # 提取论据
        for section in ['【核心论据】', '【新论据】']:
            match = re.search(rf'{section}[：:]?\s*(.+?)(?=\n【|$)', speech, re.DOTALL)
            if match:
                args = re.findall(r'\d+[.、]\s*(.+?)(?=\n\d+[.、]|\n【|$)', match.group(1), re.DOTALL)
                result["arguments"].extend([a.strip() for a in args if a.strip()])
        
        return result
    
    def add_speech(self, agent: str, speech: str, round_num: int):
        """添加一次发言记录"""
        self.data[agent]["rounds"].append(self.parse_speech(speech, round_num))
    
    def get_structured_history_for_agent(self, agent: str) -> str:
        """为指定智能体生成结构化的历史摘要"""
        my_data = self.data[agent]
        opp_agent = 'agent_b' if agent == 'agent_a' else 'agent_a'
        opp_data = self.data[opp_agent]
        
        my_name, opp_name = my_data["role"], opp_data["role"]
        
        # 收集双方观点
        my_arguments = [a for r in my_data["rounds"] for a in r["arguments"]]
        opp_arguments = [a for r in opp_data["rounds"] for a in r["arguments"]]
        
        # 构建输出
        lines = [
            "=" * 60,
            f"📋 你是【{my_name}】，立场：{my_data['stance']}",
            "=" * 60,
            "",
            "🟢【我方观点】—— 需要坚持和辩护："
        ]
        for i, arg in enumerate(my_arguments[:5], 1):
            lines.append(f"  {i}. {arg[:80]}...")
        
        lines.extend(["", "🔴【对方观点】—— 可以反驳的目标："])
        for i, arg in enumerate(opp_arguments[:5], 1):
            lines.append(f"  {i}. {arg[:80]}...")
        
        # JSON索引
        index = {
            "我方_DO_NOT_REBUT": {"角色": my_name, "论据摘要": [a[:50] + "..." for a in my_arguments[:3]]},
            "对方_CAN_REBUT": {"角色": opp_name, "论据摘要": [a[:50] + "..." for a in opp_arguments[:3]]}
        }
        
        lines.extend([
            "",
            "📊 观点归属索引：",
            json.dumps(index, ensure_ascii=False, indent=2),
            "",
            f"⚠️ 规则：反驳【{opp_name}】的论据，辩护【{my_name}】的论据！",
            "=" * 60
        ])
        
        return "\n".join(lines)
