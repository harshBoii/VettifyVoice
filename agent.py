import re
import csv
import os
import asyncio
import uuid
from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")


class AgentDecision(BaseModel):
    speak: str = Field(description="What the agent should say out loud next.")
    advance: bool = Field(description="Whether to accept the current answer and move to the next question.")
    dug_deeper: bool = Field(description="Whether this response is a follow-up probe on the same question.")


decision_llm = llm.with_structured_output(AgentDecision, method="json_schema")


def _get_text(response) -> str:
    content = response.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content


class VoiceQuestionSession:
    """
    Async-friendly question session for voice calls.
    Driven directly by WebSocket transcript events from main.py.

    Usage:
        session = VoiceQuestionSession(context)
        first_q = await session.start()
        result  = await session.submit_answer(transcript)

    result shape:
        {
            "speak": str,
            "is_done": bool,
            "summary": str | None
        }
    """

    def __init__(self, context: dict):
        self.context = context
        self.questions: list[str] = []
        self.answers: list[dict] = []
        self._q_index = 0
        self.is_done = False
        self.summary: str | None = None
        self._session_id = context.get("session_id", str(uuid.uuid4())[:8])

        self._conv_history: list[dict] = []
        self._dig_counts: dict[int, int] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> str:
        """Generate 5 questions; return spoken text of Q1."""
        ctx = self.context
        prompt = f"""
Based on the following context about a former employee, generate exactly 5 concise feedback questions to ask their ex-boss.

Context:
- Name: {ctx.get("name", "the employee")}
- Profession: {ctx.get("profession", "unknown")}
- Work Experience: {ctx.get("work_experience", "unknown")}

Rules:
- Return a numbered list only (1. ... 2. ... etc.), one question per line, no extra text
- Question 1 must be open-ended and descriptive
- Questions 2 to 5 should be short-answer type
- Keep every question under 20 words for clean voice delivery
"""
        resp = await asyncio.to_thread(llm.invoke, prompt)
        raw = _get_text(resp)

        self.questions = [
            q.strip() for q in raw.strip().split("\n")
            if q.strip() and re.match(r"^\d+[\.\)]\s", q.strip())
        ]

        if not self.questions:
            self.questions = [l.strip() for l in raw.strip().split("\n") if l.strip()][:5]

        print(f"[Session:{self._session_id}] Generated {len(self.questions)} questions", flush=True)
        for i, q in enumerate(self.questions):
            print(f"  Q{i+1}: {q}", flush=True)

        first_q = self._spoken(0)
        self._conv_history.append({"role": "assistant", "content": first_q})
        return first_q

    async def submit_answer(self, answer: str) -> dict:
        """Process a transcript and decide whether to re-ask, dig deeper, answer briefly, or advance."""
        idx = self._q_index

        if idx >= len(self.questions):
            return {
                "speak": "Thank you so much for your time. Goodbye!",
                "is_done": True,
                "summary": self.summary,
            }

        current_q = self._spoken(idx)
        next_q = self._spoken(idx + 1) if idx + 1 < len(self.questions) else None
        dig_count = self._dig_counts.get(idx, 0)
        force_advance = dig_count >= 1

        self._conv_history.append({"role": "user", "content": answer})

        decision = await self._decide_next_step(
            answer=answer,
            current_q=current_q,
            next_q=next_q,
            force_advance=force_advance,
        )

        speak = self._clean_spoken_text(decision.speak)
        advance = bool(decision.advance)
        dug_deeper = bool(decision.dug_deeper)

        self._conv_history.append({"role": "assistant", "content": speak})

        if not advance:
            if dug_deeper:
                self._dig_counts[idx] = dig_count + 1
            return {
                "speak": speak,
                "is_done": False,
                "summary": None,
            }

        self.answers.append({"question": current_q, "answer": answer})
        self._q_index += 1

        remaining = len(self.questions) - self._q_index
        print(f"[Session:{self._session_id}] Q{idx+1} answered. {remaining} remaining.", flush=True)

        if self._q_index < len(self.questions):
            return {
                "speak": speak,
                "is_done": False,
                "summary": None,
            }

        print(f"[Session:{self._session_id}] All questions answered. Generating summary...", flush=True)
        self.summary = await self._generate_summary()
        self.is_done = True
        self._save_csv()

        final_speak = speak
        if not final_speak.strip():
            final_speak = (
                "That's all the questions I had. Thank you so much for your time and honest feedback. "
                "I really appreciate it. Have a wonderful day!"
            )

        return {
            "speak": final_speak,
            "is_done": True,
            "summary": self.summary,
        }

    # ── Decision engine ───────────────────────────────────────────────────────

    async def _decide_next_step(
        self,
        answer: str,
        current_q: str,
        next_q: str | None,
        force_advance: bool,
    ) -> AgentDecision:
        name = self.context.get("name", "the employee")
        profession = self.context.get("profession", "unknown")
        work_exp = self.context.get("work_experience", "unknown")

        recent_lines = [
            f"{'Agent' if m['role'] == 'assistant' else 'Caller'}: {m['content']}"
            for m in self._conv_history[:-1]
        ][-6:]
        history_block = "\n".join(recent_lines) if recent_lines else "No prior context."

        transition = self._transition(self._q_index + 1)
        next_instruction = (
            f'If you advance, briefly acknowledge and then ask this exact next question: "{next_q}"'
            if next_q
            else "If you advance, give a short warm sign-off thanking them and ending the call."
        )

        force_note = (
            "IMPORTANT: You have already asked one follow-up on this question. "
            "Do not dig deeper again. Either re-ask because it was irrelevant/unclear, "
            "or accept and advance."
            if force_advance else
            "You may ask at most one deeper follow-up on this question if the answer is relevant but vague."
        )

        prompt = f"""
You are a professional HR reference-check caller on a live phone call.

Employee context:
- Name: {name}
- Profession: {profession}
- Work Experience: {work_exp}

Recent conversation:
{history_block}

Current question:
"{current_q}"

Caller's latest reply:
"{answer}"

{force_note}

Your job is to decide the next spoken response.

Primary goal:
- Collect useful answers and move the interview forward.
- Prefer asking the next question over giving long explanations.
- If the caller asks you a question, answer briefly only if you know from the provided employee context.
- If you cannot answer confidently, say exactly: "I'm sorry, I don't have that information."
- If the answer sounds irrelevant, broken, too unclear, or unrelated to the current question, ask them to answer again using a natural phone excuse.
- If the answer is relevant but too vague, ask one smart follow-up to dig a little deeper.
- Keep the tone warm, realistic, brief, and natural.

Decision rules:
1. If caller asked you a direct question:
   - Answer briefly if possible using only the context above.
   - If not possible, say "I'm sorry, I don't have that information."
   - Then politely bring them back to the current question.
   - advance = false
   - dug_deeper = false

2. If the answer is irrelevant / inaudible / unclear / broken / obviously off-topic:
   - Use a natural excuse like:
     "Sorry, I think the line broke for a second."
     "Apologies, I didn't quite catch that."
     "Sorry, your voice broke up there."
   - Then ask them to answer the same question again, slightly rephrased.
   - advance = false
   - dug_deeper = false

3. If the answer is relevant but too vague and follow-up is still allowed:
   - Ask one short, targeted follow-up to get a specific example or clarify.
   - advance = false
   - dug_deeper = true

4. If the answer is relevant enough:
   - Acknowledge briefly.
   - {next_instruction}
   - advance = true
   - dug_deeper = false

Hard rules:
- Max 2 sentences.
- Sound like a real person on a phone call.
- No bullet points, no markdown, no labels.
- Do not over-explain.
- Do not invent facts.
- Do not ask more than one extra probing question on the same question.
- For question 1, accept a reasonably relevant answer even if it is not perfect once one follow-up has already been used.
- For short rating questions, accept short but relevant answers.

Also prefer one of these natural acknowledgment styles when advancing:
- "{transition}"
- "Got it."
- "That's helpful."
- "Understood."

Return only the structured result.
"""
        try:
            result = await asyncio.to_thread(decision_llm.invoke, prompt)
            if not result.speak.strip():
                raise ValueError("Empty speak field")
            return result
        except Exception as e:
            print(f"[Session:{self._session_id}] Decision LLM error: {e}", flush=True)
            fallback = (
                f"Got it. {next_q}" if next_q
                else "That's all from my side. Thank you so much for your time. Have a great day!"
            )
            return AgentDecision(speak=fallback, advance=True, dug_deeper=False)

    # ── Summary ───────────────────────────────────────────────────────────────

    async def _generate_summary(self) -> str:
        name = self.context.get("name", "the employee")
        qa_block = "\n".join(
            f"Q: {item['question']}\nA: {item['answer']}"
            for item in self.answers
        )
        prompt = f"""
Based on the Q&A below from an employee reference check call, write a structured performance report.

Format exactly as:
**{name}'s Performance Profile**
[Overview paragraph — 3-4 sentences]

**Skills & Expertise**
- Technical Proficiency: ...
- Time Management & Delivery: ...
- Domain Knowledge: ...

**Work Ethic & Collaboration**
- Communication Style: ...
- Team Collaboration: ...
- Strengths: ...
- Areas for Growth: ...

**Overall Assessment**
[Final 2-3 sentence recommendation]

Use specific references from the Q&A. Be concise but substantive.

Q&A Transcript:
{qa_block}
"""
        resp = await asyncio.to_thread(llm.invoke, prompt)
        return _get_text(resp)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _spoken(self, idx: int) -> str:
        q = self.questions[idx].split("\n")[0]
        return re.sub(r"^\d+[\.\)]\s*", "", q).strip()

    def _transition(self, next_idx: int) -> str:
        phrases = [
            "Great, thank you.",
            "Got it, thanks.",
            "Noted.",
            "Perfect, thank you.",
        ]
        return phrases[min(max(next_idx - 1, 0), len(phrases) - 1)]

    def _clean_spoken_text(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"\s+", " ", text)
        return text[:350].strip()

    def _save_csv(self):
        os.makedirs("feedbacks", exist_ok=True)
        path = f"feedbacks/feedback_{self._session_id}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerows(self.answers)
        print(f"[Session:{self._session_id}] Saved CSV → {path}", flush=True)