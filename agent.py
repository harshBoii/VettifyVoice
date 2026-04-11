import re
import csv
import os
import asyncio
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")


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
    No LangGraph interrupt() — driven directly by WebSocket transcript events.

    Usage:
        session = VoiceQuestionSession(context)
        first_q = await session.start()           # returns spoken text of Q1
        result  = await session.submit_answer(t)  # call for each transcript

    result shape:
        {
            "speak":   str,        # what the agent should say next
            "is_done": bool,
            "summary": str | None  # populated only on the final turn
        }
    """

    def __init__(self, context: dict):
        self.context        = context
        self.questions:     list[str] = []
        self.answers:       list[dict] = []
        self._q_index       = 0
        self._q1_validated  = False
        self.is_done        = False
        self.summary:       str | None = None
        self._session_id    = context.get("session_id", str(uuid.uuid4())[:8])

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> str:
        """Generate 5 questions; return the spoken text of Q1."""
        ctx = self.context
        prompt = f"""
Based on the following context about a former employee, generate exactly 5 concise feedback questions to ask their ex-boss.

Context:
- Name: {ctx.get("name", "the employee")}
- Profession: {ctx.get("profession", "unknown")}
- Work Experience: {ctx.get("work_experience", "unknown")}

Rules:
- Return a numbered list only (1. ... 2. ... etc.), one question per line, no extra text
- Question 1: open-ended, requires a paragraph-length response
- Questions 2–5: short-answer type (answerable with one word or short phrase like "good", "above average", "yes")
- Keep every question under 20 words for clean voice delivery
"""
        resp = await asyncio.to_thread(llm.invoke, prompt)
        raw  = _get_text(resp)

        self.questions = [
            q.strip() for q in raw.strip().split("\n")
            if q.strip() and re.match(r"^\d+[\.\)]\s", q.strip())
        ]

        # Fallback: take first 5 non-empty lines if regex matched nothing
        if not self.questions:
            self.questions = [l.strip() for l in raw.strip().split("\n") if l.strip()][:5]

        print(f"[Session:{self._session_id}] Generated {len(self.questions)} questions", flush=True)
        for i, q in enumerate(self.questions):
            print(f"  Q{i+1}: {q}", flush=True)

        return self._to_spoken(0)

    async def submit_answer(self, answer: str) -> dict:
        """Feed a voice transcript as the answer to the current question."""
        idx = self._q_index

        if idx >= len(self.questions):
            return {"speak": "Thank you so much for your time. Goodbye!", "is_done": True, "summary": self.summary}

        # ── Q1 word-count gate (must be ≥ 50 words) ──────────────────────────
        if idx == 0 and not self._q1_validated:
            word_count = len(answer.split())
            print(f"[Session:{self._session_id}] Q1 word count: {word_count}", flush=True)
            if word_count < 50:
                return {
                    "speak": (
                        "Thank you for that. Could you elaborate just a bit more? "
                        "A few more sentences would really help me capture the full picture."
                    ),
                    "is_done": False,
                    "summary": None,
                }
            self._q1_validated = True

        # ── Record answer ─────────────────────────────────────────────────────
        q_spoken = self._to_spoken(idx)
        self.answers.append({"question": q_spoken, "answer": answer})
        self._q_index += 1

        remaining = len(self.questions) - self._q_index
        print(f"[Session:{self._session_id}] Q{idx+1} answered. {remaining} remaining.", flush=True)

        # ── More questions? ───────────────────────────────────────────────────
        if self._q_index < len(self.questions):
            next_spoken = self._to_spoken(self._q_index)
            transition  = self._transition(self._q_index)
            return {
                "speak":   f"{transition} {next_spoken}",
                "is_done": False,
                "summary": None,
            }

        # ── All done → summarize ──────────────────────────────────────────────
        print(f"[Session:{self._session_id}] All questions answered. Generating summary...", flush=True)
        self.summary  = await self._generate_summary()
        self.is_done  = True
        self._save_csv()

        return {
            "speak": (
                "That's all the questions I had. Thank you so much for your time and honest feedback. "
                "I really appreciate it. Have a wonderful day!"
            ),
            "is_done": True,
            "summary": self.summary,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _to_spoken(self, idx: int) -> str:
        """Strip question number prefix for clean TTS."""
        q = self.questions[idx].split("\n")[0]
        return re.sub(r"^\d+[\.\)]\s*", "", q).strip()

    def _transition(self, next_idx: int) -> str:
        phrases = ["Great, thank you.", "Got it, thanks.", "Noted.", "Perfect, thank you."]
        return phrases[min(next_idx - 1, len(phrases) - 1)]

    async def _generate_summary(self) -> str:
        name     = self.context.get("name", "the employee")
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

    def _save_csv(self):
        os.makedirs("feedbacks", exist_ok=True)
        path = f"feedbacks/feedback_{self._session_id}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "answer"])
            writer.writeheader()
            writer.writerows(self.answers)
        print(f"[Session:{self._session_id}] Saved CSV → {path}", flush=True)