#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实用工具模块
============

提供辩论系统的辅助功能：
- 辩论结果导出
- 文本处理工具
- 文件操作辅助
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def export_debate_result(result: Dict, output_path: Optional[str] = None, 
                         format: Optional[str] = None) -> str:
    """
    导出辩论结果
    
    Parameters
    ----------
    result : Dict
        辩论结果字典，包含 topic, rounds, history, final_summary
    output_path : str, optional
        输出文件路径，不指定则自动生成
    format : str, optional
        输出格式 ("json" 或 "markdown")
        如果未指定，根据文件扩展名自动判断
        
    Returns
    -------
    str
        实际保存的文件路径
    """
    # 自动判断格式
    if format is None and output_path:
        if output_path.endswith('.md'):
            format = "markdown"
        elif output_path.endswith('.txt'):
            format = "text"
        else:
            format = "json"
    elif format is None:
        format = "json"
    
    # 自动生成文件名
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_short = result.get("topic", "debate")[:20].replace(" ", "_")
        ext = "json" if format == "json" else "md"
        output_path = f"debate_{topic_short}_{timestamp}.{ext}"
    
    path = Path(output_path)
    
    if format == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    elif format == "markdown":
        md_content = _result_to_markdown(result)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_content)
    elif format == "text":
        text_content = _result_to_text(result)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text_content)
    else:
        raise ValueError(f"不支持的格式: {format}")
    
    return str(path)


def _result_to_markdown(result: Dict) -> str:
    """将辩论结果转换为 Markdown 格式"""
    lines = [
        f"# 辩论记录: {result.get('topic', '未知主题')}",
        "",
        f"**辩论轮数**: {result.get('rounds', 0)}",
        f"**记录时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        ""
    ]
    
    # 辩论历史
    for record in result.get("history", []):
        lines.extend([
            f"## 第 {record.get('round', '?')} 轮",
            "",
            "### 智能体A (正方)",
            "",
            record.get("agent_a", ""),
            "",
            "### 智能体B (反方)",
            "",
            record.get("agent_b", ""),
            "",
            "### 裁判评判",
            "",
            record.get("evaluation", ""),
            "",
            "---",
            ""
        ])
    
    # 最终总结
    lines.extend([
        "## 最终总结",
        "",
        result.get("final_summary", ""),
        ""
    ])
    
    return "\n".join(lines)


def _result_to_text(result: Dict) -> str:
    """将辩论结果转换为纯文本格式"""
    lines = [
        "=" * 60,
        f"辩论记录: {result.get('topic', '未知主题')}",
        "=" * 60,
        "",
        f"辩论轮数: {result.get('rounds', 0)}",
        f"记录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 60,
        ""
    ]
    
    # 辩论历史
    for record in result.get("history", []):
        lines.extend([
            f"【第 {record.get('round', '?')} 轮辩论】",
            "",
            "▶ 智能体A (正方):",
            record.get("agent_a", ""),
            "",
            "▶ 智能体B (反方):",
            record.get("agent_b", ""),
            "",
            "▶ 裁判评判:",
            record.get("evaluation", ""),
            "",
            "-" * 60,
            ""
        ])
    
    # 最终总结
    lines.extend([
        "=" * 60,
        "【最终总结】",
        "=" * 60,
        "",
        result.get("final_summary", ""),
        ""
    ])
    
    return "\n".join(lines)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    将长文本切分为重叠的块
    
    Parameters
    ----------
    text : str
        输入文本
    chunk_size : int, optional
        每块的大小 (默认: 500)
    overlap : int, optional
        块之间的重叠字符数 (默认: 50)
        
    Returns
    -------
    List[str]
        文本块列表
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # 忽略空白块
            chunks.append(chunk)
        start = end - overlap
    
    return chunks


def load_knowledge_file(file_path: str, chunk_size: int = 500) -> List[str]:
    """
    加载知识库文件并切分
    
    Parameters
    ----------
    file_path : str
        文件路径
    chunk_size : int, optional
        切分块大小 (默认: 500)
        
    Returns
    -------
    List[str]
        文档块列表
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return chunk_text(content, chunk_size)


def format_duration(seconds: float) -> str:
    """
    格式化时间间隔
    
    Parameters
    ----------
    seconds : float
        秒数
        
    Returns
    -------
    str
        格式化的时间字符串 (如 "2分30秒")
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}分{secs}秒"


def validate_topic(topic: str) -> str:
    """
    验证并清理辩论主题
    
    Parameters
    ----------
    topic : str
        用户输入的主题
        
    Returns
    -------
    str
        清理后的主题
        
    Raises
    ------
    ValueError
        如果主题为空或太短
    """
    topic = topic.strip()
    if not topic:
        raise ValueError("辩论主题不能为空")
    if len(topic) < 3:
        raise ValueError("辩论主题太短，请提供更具体的描述")
    return topic
