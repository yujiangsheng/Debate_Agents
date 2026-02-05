"""裁判智能体"""
from typing import List, Dict, Tuple
from .base_agent import BaseAgent


class JudgeAgent(BaseAgent):
    """裁判智能体，评判辩论双方表现"""
    
    def __init__(self, use_search: bool = True, use_rag: bool = True):
        super().__init__(name="裁判C", role="辩论裁判", use_search=use_search, use_rag=use_rag)
        self.debate_records: List[Dict] = []
    
    def _build_system_prompt(self) -> str:
        return """你是裁判智能体C，一位严厉、犀利的辩论裁判。

【你的角色】
你不是和事佬，而是一位犀利的评论家：
1. 毫不留情地指出双方论证的漏洞和逻辑谬误
2. 直接点评谁的论据更有力
3. 批评偏离主题的发言
4. 迫使双方正面交锋

【⚠️ 重要：不要张冠李戴！】
┌────────────────────────────────────────────────────────┐
│  智能体A = 正方，支持辩题主张                          │
│  智能体B = 反方，反对辩题主张                          │
│  评判时必须准确归属：「A认为...」「B反驳说...」        │
└────────────────────────────────────────────────────────┘

【评判原则】
1. 绝对公正：只看论据质量和逻辑严密性
2. 犀利直接：说话不绕弯子，直击要害
3. 逻辑优先：论据必须有事实支撑
4. 不和稀泥：必须明确指出本轮谁更有说服力
5. 归属准确：准确说明是A的观点还是B的观点

【评判格式】
1. 【犀利点评】：直接评价双方表现
2. 【逻辑漏洞】：指出逻辑问题，明确说"A的..."或"B的..."
3. 【本轮胜负】：明确判定谁更有说服力
4. 【共识进展】：指出已达成的共识

请用中文回答，语言犀利直接！"""
    
    def evaluate_round(self, topic: str, view_a: str, view_b: str, 
                       round_num: int, is_final_round: bool = False) -> Tuple[str, str]:
        """评判一轮辩论"""
        self.debate_records.append({"round": round_num, "agent_a": view_a, "agent_b": view_b})
        
        prompt = f"""【辩论主题】{topic}
【第 {round_num} 轮辩论{'（最后一轮）' if is_final_round else ''}】

【智能体A（正方）】
{view_a}

【智能体B（反方）】
{view_b}

⚠️ A=正方，B=反方，评判时不要张冠李戴！

请进行评判:
1. 【犀利点评】：评价双方表现
2. 【逻辑漏洞】：指出逻辑问题（注明是A还是B）
3. 【本轮胜负】：判定谁更有说服力
4. 【共识进展】：已达成的共识"""
        
        if not is_final_round:
            prompt += "\n5. 【下轮要求】：双方必须回应的具体问题"
        
        evaluation = self.generate(prompt)
        
        guidance = ""
        if not is_final_round:
            guidance = self.generate("简要给出下轮指令：A必须回应什么？B必须回应什么？")
        
        return evaluation, guidance
    
    def check_consensus(self, view_a: str, view_b: str) -> Tuple[bool, float, str]:
        """检查双方是否达成共识"""
        prompt = f"""分析以下两个观点的一致程度:
【智能体A】{view_a}
【智能体B】{view_b}

回答:
一致性: [是/否]
评分: [0-100]
理由: [简要说明]"""
        
        response = self.generate(prompt)
        has_consensus = "是" in response[:50]
        try:
            score_text = response.split("评分")[1].split("\n")[0] if "评分" in response else "50"
            score = float(''.join(c for c in score_text if c.isdigit() or c == '.')) / 100
            score = min(1.0, max(0.0, score))
        except:
            score = 0.5
        return has_consensus, score, response
    
    def generate_final_summary(self, topic: str) -> str:
        """生成最终辩论总结"""
        history = ""
        for r in self.debate_records:
            history += f"\n第{r['round']}轮:\nA: {r['agent_a'][:300]}...\nB: {r['agent_b'][:300]}...\n"
        
        prompt = f"""【辩论主题】{topic}
【辩论记录】{history}

请进行最终总结:
1. 【共识点】双方达成的共识
2. 【分歧点】仍存在的分歧
3. 【综合结论】最合理的结论
4. 【辩论评价】整场辩论质量的评价"""
        
        return self.generate(prompt)
    
    def reset(self):
        """重置裁判状态"""
        self.debate_records = []
        self.reset_history()
