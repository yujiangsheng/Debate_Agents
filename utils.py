#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实用工具模块"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List


def export_debate_result(result: Dict, output_path: str = None) -> str:
    """导出辩论结果为文本文件"""
    if output_path is None:
        topic_short = result.get("topic", "debate")[:20].replace(" ", "_")
        output_path = f"debate_{topic_short}_{datetime.now().strftime('%Y%m%d')}.txt"
    
    lines = [
        "=" * 60,
        f"辩论记录: {result.get('topic', '未知主题')}",
        "=" * 60,
        f"辩论轮数: {result.get('rounds', 0)}",
        f"记录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 60, ""
    ]
    
    for record in result.get("history", []):
        lines.extend([
            f"【第 {record.get('round', '?')} 轮辩论】", "",
            "▶ 智能体A (正方):", record.get("agent_a", ""), "",
            "▶ 智能体B (反方):", record.get("agent_b", ""), "",
            "▶ 裁判评判:", record.get("evaluation", ""), "",
            "-" * 60, ""
        ])
    
    lines.extend([
        "=" * 60, "【最终总结】", "=" * 60, "",
        result.get("final_summary", ""), ""
    ])
    
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return output_path


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """将长文本切分为重叠的块"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_knowledge_file(file_path: str, chunk_size: int = 500) -> List[str]:
    """加载知识库文件并切分"""
    return chunk_text(Path(file_path).read_text(encoding="utf-8"), chunk_size)
