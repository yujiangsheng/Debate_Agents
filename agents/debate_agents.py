"""辩论智能体 - 正方A和反方B"""
import re
import json
from typing import Optional, List, Dict
from .base_agent import BaseAgent


class DebateAgent(BaseAgent):
    """辩论智能体基类，通过stance参数控制正/反方立场"""
    
    STANCES = {
        "pro": {
            "name": "智能体A", "role": "辩论者A - 正方", "position": "正方",
            "opponent": "反方", "action": "支持", "search_keywords": "优势 好处 支持",
            "examples": ['辩题"是否应该堕胎" → 你支持堕胎权', '辩题"AI是否会取代人类" → 你认为会取代']
        },
        "con": {
            "name": "智能体B", "role": "辩论者B - 反方", "position": "反方",
            "opponent": "正方", "action": "反对", "search_keywords": "风险 问题 反对",
            "examples": ['辩题"是否应该堕胎" → 你反对堕胎权', '辩题"AI是否会取代人类" → 你认为不会取代']
        }
    }
    
    def __init__(self, stance: str = "pro", use_search: bool = True, use_rag: bool = True):
        self.stance = stance
        self.config = self.STANCES[stance]
        super().__init__(name=self.config["name"], role=self.config["role"], 
                         use_search=use_search, use_rag=use_rag)
    
    def _build_system_prompt(self) -> str:
        cfg = self.config
        examples = "\n".join(f"- {e}" for e in cfg["examples"])
        opposite_action = "反对" if cfg["action"] == "支持" else "支持"
        return f"""你是辩论智能体，担任本次辩论的【{cfg["position"]}】立场。

【核心身份】
你是{cfg["position"]}辩手，必须{cfg["action"]}辩题中的主张。
例如：
{examples}

【辩论原则】
1. 坚定{cfg["position"]}：必须坚持{cfg["position"]}立场，{cfg["action"]}辩题主张
2. 聚焦主题：所有论述必须紧扣原始问题
3. 针锋相对：针对🔴对方的观点进行反驳
4. 【禁止重复】：绝对不要重复你之前已经表达过的观点！

【⚠️⚠️⚠️ 最最重要：不要帮对方说话！】
┌────────────────────────────────────────────────────────┐
│  你是{cfg["position"]}，你的目标是{cfg["action"]}辩题主张！            │
│  你的对手是{cfg["opponent"]}，他的目标是{opposite_action}辩题主张！      │
│                                                        │
│  ✅ 正确：你的每一句话、每个论据、每次反驳，          │
│     结论都必须是「所以应该{cfg["action"]}辩题」              │
│                                                        │
│  ❌ 错误：说着说着就认同对方观点，或得出                │
│     「所以应该{opposite_action}辩题」的结论                    │
│                                                        │
│  ⛔ 绝对禁止：在反驳中得出支持对方立场的结论！       │
└────────────────────────────────────────────────────────┘

【区分我方vs对方的观点】
🟢【我方】= 你自己之前说过的话 → 要辩护不能反驳！
🔴【对方】= 对方说的话 → 这才是你要反驳的！

请用中文回答，语言犀利有力，坚定捍卫你的{cfg["position"]}立场！"""
    
    def debate(self, topic: str, opponent_view: Optional[str] = None,
               use_tools: bool = False, judge_feedback: Optional[str] = None,
               debate_history: Optional[List[Dict]] = None,
               structured_history: Optional[str] = None) -> str:
        """进行辩论发言"""
        cfg = self.config
        context = ""
        
        if use_tools:
            if self.search_tool:
                context += self.search(f"{topic} {cfg['search_keywords']}")
            if self.rag_tool:
                context += self.retrieve(topic)
        
        # 构建历史摘要
        history_summary = ""
        if structured_history:
            history_summary = f"\n{structured_history}\n"
        elif debate_history:
            history_summary = self._build_history_summary(debate_history)
        
        # 构建提示词
        if opponent_view:
            prompt = self._build_rebuttal_prompt(topic, opponent_view, history_summary, 
                                                  judge_feedback, debate_history)
        else:
            prompt = self._build_opening_prompt(topic)
        
        response = self.generate(prompt, context)
        response = self._verify_and_fix_consistency(topic, response, cfg, debate_history)
        return response
    
    def _build_history_summary(self, debate_history: List[Dict]) -> str:
        """构建辩论历史摘要"""
        my_key = 'agent_a' if self.stance == 'pro' else 'agent_b'
        opp_key = 'agent_b' if self.stance == 'pro' else 'agent_a'
        my_name = '正方A' if self.stance == 'pro' else '反方B'
        opp_name = '反方B' if self.stance == 'pro' else '正方A'
        
        summary = f"\n{'='*50}\n📜 辩论历史记录 | 你是【{my_name}】\n{'='*50}\n"
        
        for record in debate_history:
            summary += f"\n第 {record['round']} 轮:\n"
            summary += f"🟢【我方】：{record[my_key][:300]}...\n"
            summary += f"🔴【对方】：{record[opp_key][:300]}...\n"
        
        return summary
    
    def _build_opening_prompt(self, topic: str) -> str:
        """构建首轮发言提示"""
        cfg = self.config
        return f"""【辩论主题】{topic}

你是{cfg['position']}，你的目标是{cfg['action']}辩题主张。首次发言请按以下格式：
1. 【我的立场】：明确表态{cfg['action']}辩题主张
2. 【核心论据】：2-3个支持{cfg['action']}的有力论据（每个论据的结论都必须指向{cfg['action']}）
3. 【预判并反驳对方可能的论据】：
   - 对方可能说：「...」
   - 我的预先反驳：...（结论必须是「所以应该{cfg['action']}辩题」）

⚠️ 重要：你的每一个论据、每一次反驳，结论都必须支持{cfg['action']}！"""
    
    def _build_rebuttal_prompt(self, topic: str, opponent_view: str, 
                                history_summary: str, judge_feedback: str,
                                debate_history: List[Dict]) -> str:
        """构建反驳发言提示"""
        cfg = self.config
        is_my_first = not debate_history
        
        prompt = f"【辩论主题】{topic}\n{history_summary}"
        prompt += f"\n🔴【对方最新发言】：\n{opponent_view}\n"
        
        if is_my_first:
            prompt += f"""
⚠️ 重要提示：对方发言包含两部分，要区别对待！
┌─────────────────────────────────────────────────────────┐
│  对方的「核心论据」= 对方真正的观点 → 这是你要反驳的！   │
│  对方的「预判」= 对方猜测你会说的 → 这不是对方的观点！   │
├─────────────────────────────────────────────────────────┤
│  ⭐ 对于对方的预判，你应该默认接受它作为你的观点：       │
│  ✅ 承认：「没错，我确实认为...」                       │
│     → 然后给出有力论证，说明为什么这个观点是对的        │
│     → 并说明这个观点如何支持你{cfg['action']}的立场       │
│                                                         │
│  ❌ 绝对禁止：把对方的预判当作对方的观点来反驳！        │
│     → 预判是对方猜你会说的，不是对方自己的主张          │
└─────────────────────────────────────────────────────────┘

你是{cfg['position']}，首次发言请按以下格式：
1. 【我的立场】：表态{cfg['action']}辩题主张
2. 【核心论据】：2-3个支持{cfg['action']}的有力论据
3. 【反驳对方核心论据】：针对对方的「核心论据」部分进行反驳（结论必须支持{cfg['action']}）
4. 【回应对方的预判】：接受对方的预判作为你的观点，并论证它为何正确、为何支持你的立场
   → 格式：「没错，我确实认为[对方预判的内容]，因为...，这恰恰说明应该{cfg['action']}辩题」

⛔ 注意：你是辩论者，不是裁判！不要输出【犀利点评】【逻辑漏洞】【本轮胜负】等评判内容！"""
        else:
            opposite_action = "反对" if cfg['action'] == "支持" else "支持"
            prompt += f"""
你是{cfg['position']}，请按以下格式回复：
1. 【我的立场】：一句话表态{cfg['action']}辩题主张
2. 【新论据】：1-2个支持{cfg['action']}的新论据（之前未提过）
3. 【反驳🔴对方观点】：
   - 指出对方论据的问题
   - ⚠️ 结论必须是「所以应该{cfg['action']}辩题」，不能得出{opposite_action}的结论！
4. 【辩护🟢我方观点】：如果对方批评了我的观点，要辩护并强化{cfg['action']}的立场

⛔ 注意：
- 你是辩论者，不是裁判！不要输出【犀利点评】【逻辑漏洞】【本轮胜负】等评判内容！
- 绝对禁止帮对方说话！你的每一句结论都必须支持{cfg['action']}辩题！"""
        
        if judge_feedback:
            prompt += f"\n【裁判反馈】{judge_feedback}\n请针对裁判指出的问题回应。"
        
        return prompt
    
    def _clean_response(self, response: str) -> str:
        """清理发言中的内部自检内容、裁判评判内容和元评论"""
        # 移除JSON自检内容
        cleaned = re.sub(r'\s*\{[^{}]*(?:一致|consistent)[^{}]*\}\s*', '', response, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'^\s*json\s*\n?|\n\s*json\s*(?=\n*【)', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*[【\(]内部自检[^】\)]*[】\)][^\n]*\n?', '', cleaned)
        
        # 移除智能体错误输出的裁判评判内容
        judge_patterns = ['裁判反馈', '犀利点评', '逻辑漏洞', '本轮胜负', '共识进展', '下轮要求']
        for pattern in judge_patterns:
            cleaned = re.sub(rf'\s*【{pattern}】[\s\S]*?(?=【|$)', '', cleaned)
        
        # 移除内部提示词/反思内容（不应该出现在输出中）
        internal_prompts = [
            r'[⚠️❌✅⛔🔴🟢]\s*结论必须是[^\n]*$',  # 结论提示
            r'[⚠️❌✅⛔]\s*[^\n]*不能得出[^\n]*结论[^\n]*$',  # 不能得出xxx结论
            r'[⚠️❌✅⛔]\s*[^\n]*必须支持[^\n]*$',  # 必须支持xxx
            r'[⚠️❌✅⛔]\s*注意[：:][^\n]*$',  # 注意提示
            r'\n+[-•]\s*[⚠️❌✅⛔]\s*[^\n]+$',  # 带符号的提示行
        ]
        for pattern in internal_prompts:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        # 移除元评论/反思性文字（不应该出现在辩论中）
        meta_patterns = [
            r'\n+这样修改后[^\n]*$',
            r'\n+以上[是为]?[^\n]*修改[^\n]*$',
            r'\n+希望[^\n]*能[^\n]*$',
            r'\n+请根据[^\n]*进行[^\n]*$',
            # 移除"通过以上调整，确保了..."这类反思性总结
            r'\n*通过以上[调整修改][，,][^\n]*[。.]?\s*$',
            r'\n*通过上述[调整修改论据][，,][^\n]*[。.]?\s*$',
            # 移除其他反思性表述
            r'\n*以上[论述内容][确保保证][了]?[^\n]*[。.]?\s*$',
            r'\n*经过[以上]*[调整修改][，,]?[^\n]*避免[^\n]*[。.]?\s*$',
            r'\n*这样[就能够可以]*[确保避免][^\n]*[。.]?\s*$',
            # 移除关于修改/调整的自我评价
            r'\n*[综上所述]*[，,]?[我我们][已经对]*[以上上述][内容论点][进行了]*[修改调整][^\n]*[。.]?\s*$',
        ]
        for pattern in meta_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        # 移除多余的空行
        return re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    
    def _verify_and_fix_consistency(self, topic: str, response: str, cfg: dict, 
                                     debate_history: list = None, max_retries: int = 2) -> str:
        """验证论据与立场的一致性"""
        my_previous = ""
        opp_previous = ""
        if debate_history:
            my_key = 'agent_a' if self.stance == 'pro' else 'agent_b'
            opp_key = 'agent_b' if self.stance == 'pro' else 'agent_a'
            for i, r in enumerate(debate_history, 1):
                my_previous += f"第{i}轮我方：{r[my_key][:150]}...\n"
                opp_previous += f"第{i}轮对方：{r[opp_key][:150]}...\n"
        
        verify_prompt = f"""检查以下辩论发言：
【主题】{topic}
【立场】{cfg['position']}（必须{cfg['action']}）
【我方历史】{my_previous or "无"}
【对方历史】{opp_previous or "无"}
【发言】{response}

检查：1.论据是否支持"{cfg['action']}" 2.是否错误反驳了自己的观点
返回JSON：{{"consistent": true/false, "attribution_correct": true/false, "problems": []}}"""

        for retry in range(max_retries):
            try:
                verification = self.generate(verify_prompt, "")
                json_match = re.search(r'\{[^{}]*\}', verification, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    if result.get("consistent", True) and result.get("attribution_correct", True):
                        return self._clean_response(response)
                    elif retry < max_retries - 1:
                        problems = result.get("problems", [])
                        fix_prompt = f"问题：{problems}\n请重新生成发言，主题：{topic}"
                        response = self.generate(fix_prompt, "")
                return self._clean_response(response)
            except Exception:
                return self._clean_response(response)
        return self._clean_response(response)


class DebateAgentA(DebateAgent):
    """正方辩论智能体"""
    def __init__(self, use_search: bool = True, use_rag: bool = True):
        super().__init__(stance="pro", use_search=use_search, use_rag=use_rag)


class DebateAgentB(DebateAgent):
    """反方辩论智能体"""
    def __init__(self, use_search: bool = True, use_rag: bool = True):
        super().__init__(stance="con", use_search=use_search, use_rag=use_rag)
