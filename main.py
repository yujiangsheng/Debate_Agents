#!/usr/bin/env python3
"""智能辩论系统 - 主程序入口"""
import argparse
import sys
import re
from datetime import datetime

from debate_system import DebateSystem
from config import MAX_DEBATE_ROUNDS
from utils import load_knowledge_file, export_debate_result


def main():
    parser = argparse.ArgumentParser(description="🎯 智能辩论系统")
    parser.add_argument("-t", "--topic", type=str, default=None, help="辩论主题")
    parser.add_argument("-r", "--rounds", type=int, default=MAX_DEBATE_ROUNDS, help=f"辩论轮数 (默认: {MAX_DEBATE_ROUNDS})")
    parser.add_argument("--use-tools", action="store_true", help="启用搜索和RAG工具")
    parser.add_argument("--no-search", action="store_true", help="禁用网络搜索")
    parser.add_argument("--no-rag", action="store_true", help="禁用RAG知识库")
    parser.add_argument("--no-early-stop", action="store_true", help="禁用共识提前结束")
    parser.add_argument("-k", "--knowledge-file", type=str, default=None, help="知识库文件路径")
    parser.add_argument("-o", "--output", type=str, default=None, help="结果导出路径")
    parser.add_argument("--no-export", action="store_true", help="禁用自动导出")
    
    args = parser.parse_args()
    
    try:
        system = DebateSystem(use_search=not args.no_search, use_rag=not args.no_rag)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)
    
    if args.knowledge_file:
        try:
            chunks = load_knowledge_file(args.knowledge_file)
            system.add_knowledge(chunks)
        except Exception as e:
            print(f"⚠ 加载知识库失败: {e}")
    
    if args.topic:
        result = system.run_debate(topic=args.topic, max_rounds=args.rounds, 
                                    use_tools=args.use_tools, early_stop=not args.no_early_stop)
        print(f"\n✓ 辩论完成，共 {result['rounds']} 轮")
        
        if not args.no_export:
            try:
                if args.output:
                    output_file = export_debate_result(result, args.output)
                else:
                    date_str = datetime.now().strftime("%Y%m%d")
                    topic_clean = re.sub(r'[\\/*?:"<>|]', '', args.topic).replace(" ", "_")[:50]
                    output_file = export_debate_result(result, f"debate_{topic_clean}_{date_str}.txt")
                print(f"📄 结果已导出到: {output_file}")
            except Exception as e:
                print(f"⚠ 导出失败: {e}")
    else:
        system.interactive_mode()


if __name__ == "__main__":
    main()
