"""裁判智能体"""
import re
from typing import List, Dict, Tuple
from .base_agent import BaseAgent


class JudgeAgent(BaseAgent):
    """裁判智能体，评判辩论双方表现"""
    
    def __init__(self, use_search: bool = True, use_rag: bool = True):
        super().__init__(name="裁判C", role="辩论裁判", use_search=use_search, use_rag=use_rag)
        self.debate_records: List[Dict] = []
        self.previous_judgments: List[Dict] = []  # 记录历史评判，用于一致性检查
    
    def _build_system_prompt(self) -> str:
        return """你是裁判智能体C，一位严厉、犀利且逻辑严谨的辩论裁判。

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

【⚠️⚠️⚠️ 最重要：前后逻辑必须一致！】
┌────────────────────────────────────────────────────────┐
│  1. 你的评判必须前后一致，不能自相矛盾               │
│  2. 如果你之前认为A的论点有问题，后面不能突然说好     │
│  3. 本轮胜负判定必须与【犀利点评】【逻辑漏洞】一致   │
│  4. 不能在同一段评判中既夸又贬同一方                 │
│                                                        │
│  ❌ 错误示例：「A论据充分」→「本轮B更有说服力」      │
│  ✅ 正确示例：「A论据充分」→「本轮A更有说服力」      │
└────────────────────────────────────────────────────────┘

【评判原则】
1. 绝对公正：只看论据质量和逻辑严密性
2. 犀利直接：说话不绕弯子，直击要害
3. 逻辑优先：论据必须有事实支撑
4. 不和稀泥：必须明确指出本轮谁更有说服力
5. 归属准确：准确说明是A的观点还是B的观点
6. 前后一致：评判结论必须与分析内容逻辑一致

【评判格式】
1. 【犀利点评】：直接评价双方表现
2. 【逻辑漏洞】：指出逻辑问题，明确说"A的..."或"B的..."
3. 【本轮胜负】：明确判定谁更有说服力（必须与上述分析一致！）
4. 【共识进展】：指出已达成的共识

请用中文回答，语言犀利直接！"""
    
    def evaluate_round(self, topic: str, view_a: str, view_b: str, 
                       round_num: int, is_final_round: bool = False) -> Tuple[str, str]:
        """评判一轮辩论"""
        self.debate_records.append({"round": round_num, "agent_a": view_a, "agent_b": view_b})
        
        # 构建历史评判摘要，供一致性参考
        history_summary = ""
        if self.previous_judgments:
            history_summary = "\n【你之前的评判记录（保持一致！）】\n"
            for j in self.previous_judgments:
                history_summary += f"第{j['round']}轮: {j['winner']}更有说服力, 原因: {j['reason'][:100]}...\n"
        
        prompt = f"""【辩论主题】{topic}
【第 {round_num} 轮辩论{'（最后一轮）' if is_final_round else ''}】
{history_summary}
【智能体A（正方）】
{view_a}

【智能体B（反方）】
{view_b}

⚠️ A=正方，B=反方，评判时不要张冠李戴！
⚠️ 你的【本轮胜负】判定必须与【犀利点评】【逻辑漏洞】的分析逻辑一致！

请进行评判:
1. 【犀利点评】：评价双方表现（说谁好谁差要前后一致）
2. 【逻辑漏洞】：指出逻辑问题（注明是A还是B）
3. 【本轮胜负】：判定谁更有说服力（必须与上面的分析一致！）
4. 【共识进展】：已达成的共识"""
        
        if not is_final_round:
            prompt += "\n5. 【下轮要求】：双方必须回应的具体问题"
        
        evaluation = self.generate(prompt)
        
        # 验证评判一致性
        evaluation = self._verify_judgment_consistency(evaluation, round_num)
        
        guidance = ""
        if not is_final_round:
            guidance = self.generate("简要给出下轮指令：A必须回应什么？B必须回应什么？")
        
        return evaluation, guidance
    
    def _verify_judgment_consistency(self, evaluation: str, round_num: int) -> str:
        """验证评判的内部一致性"""
        # 提取本轮胜负判定
        winner = None
        if "A更有说服力" in evaluation or "正方A" in evaluation and "更有说服力" in evaluation:
            winner = "A"
        elif "B更有说服力" in evaluation or "反方B" in evaluation and "更有说服力" in evaluation:
            winner = "B"
        
        # 简单提取评判理由
        reason = ""
        if "【本轮胜负】" in evaluation:
            reason = evaluation.split("【本轮胜负】")[1][:200]
        
        # 记录本轮评判
        if winner:
            self.previous_judgments.append({
                "round": round_num,
                "winner": winner,
                "reason": reason
            })
        
        # 检查评判内部一致性
        check_prompt = f"""检查以下评判是否存在逻辑矛盾：
{evaluation}

检查项：
1. 【犀利点评】中对A/B的评价是否与【本轮胜负】判定一致？
2. 【逻辑漏洞】指出的问题是否支持【本轮胜负】的结论？
3. 是否存在自相矛盾的表述（如先夸A后判B胜）？

如果存在矛盾，直接指出哪里矛盾；如果一致，回复"一致"。"""
        
        consistency_check = self.generate(check_prompt)
        
        # 如果发现矛盾，要求重新生成
        if "矛盾" in consistency_check and "一致" not in consistency_check[:10]:
            fix_prompt = f"""你之前的评判存在逻辑矛盾：{consistency_check}

请重新生成一份逻辑一致的评判，确保：
1. 【犀利点评】的分析与【本轮胜负】判定一致
2. 如果你认为A的论据更充分，就判A胜；反之判B胜
3. 不要自相矛盾

原始辩论内容已在上文，请重新评判。"""
            evaluation = self.generate(fix_prompt)
        
        return evaluation
    
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
        self.previous_judgments = []
        self.reset_history()
