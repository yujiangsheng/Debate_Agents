#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能辩论系统 - 主程序入口
==========================

基于 Qwen2.5-7B-Instruct 的多智能体辩论系统。

系统包含三个智能体：
- 智能体A (正方): 支持辩题主张，从积极角度论证
- 智能体B (反方): 反对辩题主张，从审慎角度论证  
- 裁判C (评判): 犀利评判双方表现，引导达成共识

运行方式
--------
1. 交互模式 (默认):
   $ python main.py
   
2. 指定主题:
   $ python main.py -t "人工智能是否会取代人类工作"
   
3. 自定义轮数:
   $ python main.py -t "远程办公的利弊" -r 3
   
4. 启用搜索工具:
   $ python main.py -t "2024年科技趋势" --use-tools

更多参数请使用 python main.py --help 查看。
"""

import argparse
import sys
from pathlib import Path

from debate_system import DebateSystem
from config import MAX_DEBATE_ROUNDS
from utils import load_knowledge_file, export_debate_result


def main():
    """
    主函数 - 解析命令行参数并启动辩论系统
    
    支持两种运行模式：
    1. 交互模式: 不指定主题时进入，可连续进行多场辩论
    2. 单次模式: 指定主题后进行一场辩论并退出
    """
    parser = argparse.ArgumentParser(
        description="🎯 智能辩论系统 - 基于Qwen2.5-7B-Instruct的多智能体辩论",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                                     # 交互模式
  python main.py -t "人工智能是否会取代人类工作"       # 单次辩论
  python main.py -t "远程办公vs办公室办公" -r 3       # 3轮辩论
  python main.py -t "加密货币的未来" --use-tools      # 启用搜索工具

作者: Jiangsheng Yu
        """
    )
    
    parser.add_argument(
        "-t", "--topic",
        type=str, default=None,
        help="辩论主题 (不提供则进入交互模式)"
    )
    
    parser.add_argument(
        "-r", "--rounds",
        type=int, default=MAX_DEBATE_ROUNDS,
        help=f"最大辩论轮数 (默认: {MAX_DEBATE_ROUNDS})"
    )
    
    parser.add_argument(
        "--use-tools",
        action="store_true",
        help="启用搜索和RAG工具"
    )
    
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="禁用网络搜索功能"
    )
    
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="禁用RAG知识库功能"
    )
    
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="禁用达成共识后提前结束"
    )
    
    parser.add_argument(
        "-k", "--knowledge-file",
        type=str, default=None,
        help="知识库文件路径 (用于RAG)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str, default=None,
        help="结果导出文件路径 (.json/.md/.txt)，默认自动导出为txt"
    )
    
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="禁用自动导出辩论结果"
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # 初始化辩论系统
    # =========================================================================
    # 创建系统实例，可选择性禁用搜索和RAG功能
    try:
        system = DebateSystem(
            use_search=not args.no_search,
            use_rag=not args.no_rag
        )
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)
    
    # =========================================================================
    # 加载外部知识库 (可选)
    # =========================================================================
    # 如果指定了知识库文件，将其分块后加入RAG检索系统
    if args.knowledge_file:
        try:
            chunks = load_knowledge_file(args.knowledge_file)
            system.add_knowledge(chunks)
        except Exception as e:
            print(f"⚠ 加载知识库失败: {e}")
    
    # =========================================================================
    # 运行辩论
    # =========================================================================
    if args.topic:
        result = system.run_debate(
            topic=args.topic,
            max_rounds=args.rounds,
            use_tools=args.use_tools,
            early_stop=not args.no_early_stop
        )
        print(f"\n✓ 辩论完成，共 {result['rounds']} 轮")
        
        # 导出结果到文件
        if not args.no_export:
            try:
                if args.output:
                    # 用户指定了输出路径
                    output_file = export_debate_result(result, args.output)
                else:
                    # 默认导出为txt文件
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    topic_short = args.topic[:15].replace(" ", "_").replace("/", "_")
                    default_output = f"debate_{topic_short}_{timestamp}.txt"
                    output_file = export_debate_result(result, default_output, format="text")
                print(f"📄 结果已导出到: {output_file}")
            except Exception as e:
                print(f"⚠ 导出失败: {e}")
    else:
        system.interactive_mode()


if __name__ == "__main__":
    main()
