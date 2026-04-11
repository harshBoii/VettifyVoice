# main.py
import os
import json
import base64
import asyncio
import audioop
import uuid
import websockets
import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client
from groq import AsyncGroq
import anthropic
import google.generativeai as genai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from agent import VoiceQuestionSession

load_dotenv()

# ─── Twilio ───────────────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID  = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN   = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER = os.environ["TWILIO_PHONE_NUMBER"]
PUBLIC_BASE_URL     = os.environ["PUBLIC_BASE_URL"].rstrip("/")

# ─── STT ──────────────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY    = os.environ["DEEPGRAM_API_KEY"]

# ─── TTS ──────────────────────────────────────────────────────────────────────
ELEVENLABS_API_KEY  = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.environ["ELEVENLABS_VOICE_ID"]

# ─── LLM keys (only set the ones you use) ────────────────────────────────────
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "")
OPENAI_API_KEY      = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")

# ─── LLM clients ─────────────────────────────────────────────────────────────
groq_client   = AsyncGroq(api_key=GROQ_API_KEY)                          if GROQ_API_KEY      else None
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)                      if OPENAI_API_KEY    else None
claude_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)      if ANTHROPIC_API_KEY else None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app           = FastAPI()
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ─── Default LLM provider + models ───────────────────────────────────────────
DEFAULT_LLM_PROVIDER = "groq"

DEFAULT_LLM_MODELS = {
    "groq":   "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-4-5-20251001",
    "gemini": "gemini-2.0-flash",
}

# ─── Deepgram ─────────────────────────────────────────────────────────────────
DEEPGRAM_URL_BASE = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-3"
    "&encoding=mulaw"
    "&sample_rate=8000"
    "&channels=1"
    "&punctuate=true"
    "&interim_results=true"
    "&endpointing=300"
    "&utterance_end_ms=1000"
    "&language="
)

# ─── ElevenLabs ───────────────────────────────────────────────────────────────
ELEVENLABS_STREAM_PATH = "/stream?output_format=pcm_8000"
ELEVENLABS_MODEL       = "eleven_flash_v2_5"

def elevenlabs_stream_url(voice_id: str) -> str:
    return f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}{ELEVENLABS_STREAM_PATH}"

def deepgram_ws_url(lang: str) -> str:
    return DEEPGRAM_URL_BASE + lang

# ─── Chat-mode agent defaults ─────────────────────────────────────────────────
LANGUAGE          = "English"
NAME              = "Samaaira"
COMPANY           = "Immortell Company"
PRODUCT           = "GEO optimization services"
PERKS_OF_PRODUCT  = "10% off on the first month"
INFO_ABOUT_LEAD   = "The Lead is a poor guy with a low income and is looking for a cheap way to grow his business."
MIN_WORDS_TO_RESPOND = 3

SYSTEM_PROMPT_TEMPLATE = """You are a warm and persuasive sales representative calling on behalf of {COMPANY}.

## Your Goal
Sell {PRODUCT} to {NAME}. The offer includes {PERKS_OF_PRODUCT}. Close the call with either a confirmed interest or a scheduled follow-up.

## What You Know About This Lead
{INFO_ABOUT_LEAD}
Use this intel subtly — don't reference it directly. Let it shape HOW you pitch, not WHAT you say.

## Voice Call Rules (Critical)
- Speak in {LANGUAGE} only, keep the conversation natural, engaging and concise.
- Max 1-2 sentences per response — this is a phone call, not an email

## Conversation Flow
1. Warm intro → Identify pain point → ask permission and ask one question tied to their situation → Bridge → connect their pain to your product naturally → Offer → present {PRODUCT} + {PERKS_OF_PRODUCT} as the solution

## Tone
Act like the customer is your boss. Humbly but professional. Sound like a real person having a real conversation — with natural pauses and occasional light humor if the vibe allows."""

OPENING_GREETING_TEMPLATE = (
    "Hi, {NAME}, this is Annie calling from {COMPANY}. "
    "I'll keep this quick — I'm reaching out to tell you about an offer on {PRODUCT}. "
    "We have {PERKS_OF_PRODUCT} for you. Is this a good time to talk for two minutes?"
)

# ─── In-memory stores ─────────────────────────────────────────────────────────
pending_call_configs:  dict[str, dict]               = {}
call_configs_by_sid:   dict[str, dict]               = {}
question_sessions:     dict[str, VoiceQuestionSession] = {}
call_results:          dict[str, dict]               = {}


# ─── LLM Router (chat mode only) ──────────────────────────────────────────────

async def ask_llm(
    conversation_history: list,
    system_prompt: str,
    provider: str,
    model: str,
) -> str:
    try:
        if provider == "groq":
            if not groq_client:
                raise ValueError("GROQ_API_KEY not set")
            response = await groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}] + conversation_history,
                temperature=0.7,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()

        elif provider == "openai":
            if not openai_client:
                raise ValueError("OPENAI_API_KEY not set")
            response = await openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}] + conversation_history,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()

        elif provider == "claude":
            if not claude_client:
                raise ValueError("ANTHROPIC_API_KEY not set")
            response = await claude_client.messages.create(
                model=model,
                system=system_prompt,
                messages=conversation_history,
                max_tokens=150,
            )
            return response.content[0].text.strip()

        elif provider == "gemini":
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not set")
            gemini_model = genai.GenerativeModel(model_name=model, system_instruction=system_prompt)
            gemini_history = []
            for msg in conversation_history:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [msg["content"]]})
            chat     = gemini_model.start_chat(history=gemini_history[:-1] if gemini_history else [])
            last_msg = gemini_history[-1]["parts"][0] if gemini_history else ""
            response = await asyncio.to_thread(chat.send_message, last_msg)
            return response.text.strip()

        else:
            raise ValueError(f"Unknown LLM provider: '{provider}'")

    except Exception as e:
        print(f"[LLM/{provider}] Error: {e}", flush=True)
        return "Sorry, give me just a moment."


# ─── Config builders ──────────────────────────────────────────────────────────

def _format_vars(*, language, name, company, product, perks_of_product, info_about_lead) -> dict:
    return {
        "LANGUAGE": language, "NAME": name, "COMPANY": company,
        "PRODUCT": product, "PERKS_OF_PRODUCT": perks_of_product,
        "INFO_ABOUT_LEAD": info_about_lead,
    }

def build_call_config(body: dict | None) -> dict:
    b           = body or {}
    language    = b.get("language", LANGUAGE)
    dg_language = b.get("deepgram_language", "en")
    el_model    = b.get("elevenlabs_model", ELEVENLABS_MODEL)
    name        = b.get("name", NAME)
    company     = b.get("company", COMPANY)
    product     = b.get("product", PRODUCT)
    perks       = b.get("perks_of_product", PERKS_OF_PRODUCT)
    lead_info   = b.get("info_about_lead", INFO_ABOUT_LEAD)
    voice_id    = b.get("voiceId") or ELEVENLABS_VOICE_ID
    provider    = b.get("llm_provider", DEFAULT_LLM_PROVIDER).lower()
    model       = b.get("llm_model", DEFAULT_LLM_MODELS.get(provider, DEFAULT_LLM_MODELS[DEFAULT_LLM_PROVIDER]))

    ctx = _format_vars(language=language, name=name, company=company,
                       product=product, perks_of_product=perks, info_about_lead=lead_info)

    return {
        "__mode":            "chat",
        "language":          language,
        "deepgram_language": dg_language,
        "elevenlabs_model":  el_model,
        "voice_id":          voice_id,
        "name":              name,
        "company":           company,
        "product":           product,
        "perks_of_product":  perks,
        "info_about_lead":   lead_info,
        "system_prompt":     b.get("system_prompt") or SYSTEM_PROMPT_TEMPLATE.format(**ctx),
        "opening_greeting":  b.get("opening_greeting") or OPENING_GREETING_TEMPLATE.format(**ctx),
        "llm_provider":      provider,
        "llm_model":         model,
    }

def build_question_config(body: dict) -> dict:
    return {
        "__mode":            "question",
        "context":           body.get("context", {}),
        "callback_url":      body.get("callback_url"),
        "voice_id":          body.get("voiceId") or ELEVENLABS_VOICE_ID,
        "elevenlabs_model":  body.get("elevenlabs_model", ELEVENLABS_MODEL),
        "deepgram_language": body.get("deepgram_language", "en"),
    }


# ─── TTS ──────────────────────────────────────────────────────────────────────

async def text_to_mulaw_chunks(text: str, model_id: str, voice_id: str):
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8, "style": 0.0, "use_speaker_boost": True},
    }
    url = elevenlabs_stream_url(voice_id)
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                print(f"[ElevenLabs] Error {response.status_code}: {body}", flush=True)
                return
            async for pcm_chunk in response.aiter_bytes(chunk_size=320):
                if not pcm_chunk:
                    continue
                yield base64.b64encode(audioop.lin2ulaw(pcm_chunk, 2)).decode("utf-8")


# ─── Webhook helper ───────────────────────────────────────────────────────────

async def _post_callback(url: str, payload: dict):
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            await client.post(url, json=payload)
        print(f"[Callback] POSTed summary to {url}", flush=True)
    except Exception as e:
        print(f"[Callback] Failed to POST to {url}: {e}", flush=True)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"ok": True}


# ── Chat mode outbound call ────────────────────────────────────────────────────
@app.post("/call/outbound")
async def make_outbound_call(request: Request):
    body      = await request.json()
    to_number = body.get("to")
    if not to_number:
        raise HTTPException(status_code=400, detail="Missing 'to' number")
    cfg_body  = {k: v for k, v in body.items() if k != "to"}
    cfg       = build_call_config(cfg_body)
    cfg_token = str(uuid.uuid4())
    pending_call_configs[cfg_token] = cfg
    call = twilio_client.calls.create(
        to=to_number, from_=TWILIO_PHONE_NUMBER,
        url=f"{PUBLIC_BASE_URL}/voice/incoming?cfg={cfg_token}", method="POST"
    )
    print(f"[{call.sid}] Chat outbound → {to_number} | LLM={cfg['llm_provider']}/{cfg['llm_model']}", flush=True)
    return {"call_sid": call.sid, "status": call.status}


# ── Question mode outbound call ────────────────────────────────────────────────
@app.post("/call/question-outbound")
async def make_question_call(request: Request):
    """
    POST body:
    {
        "to":          "+919876543210",
        "context": {
            "name":            "Rohan Sharma",
            "profession":      "UI/UX Designer",
            "work_experience": "3 years at a fintech startup"
        },
        "callback_url":      "https://your-server.com/webhook/done",   // optional
        "voiceId":           "your_elevenlabs_voice_id",               // optional
        "deepgram_language": "en"                                       // optional
    }
    """
    body      = await request.json()
    to_number = body.get("to")
    context   = body.get("context")

    if not to_number:
        raise HTTPException(status_code=400, detail="Missing 'to' number")
    if not context:
        raise HTTPException(status_code=400, detail="Missing 'context' (name, profession, work_experience)")

    cfg       = build_question_config(body)
    cfg_token = str(uuid.uuid4())
    pending_call_configs[cfg_token] = cfg

    call = twilio_client.calls.create(
        to=to_number, from_=TWILIO_PHONE_NUMBER,
        url=f"{PUBLIC_BASE_URL}/voice/incoming?cfg={cfg_token}", method="POST"
    )
    print(f"[{call.sid}] Question outbound → {to_number} | subject={context.get('name')}", flush=True)
    return {"call_sid": call.sid, "status": call.status}


# ── Shared Twilio webhook ──────────────────────────────────────────────────────
@app.post("/voice/incoming")
async def incoming_call(request: Request):
    form      = await request.form()
    params    = dict(form)
    call_sid  = params.get("CallSid", "unknown")
    caller    = params.get("From", "unknown")
    cfg_token = request.query_params.get("cfg")

    if cfg_token and cfg_token in pending_call_configs:
        call_configs_by_sid[call_sid] = pending_call_configs.pop(cfg_token)

    print(f"[{call_sid}] Incoming from {caller}", flush=True)

    ws_base  = PUBLIC_BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
    response = VoiceResponse()
    connect  = Connect()
    connect.stream(url=f"{ws_base}/media-stream/{call_sid}", name="voice-agent-stream", track="inbound_track")
    response.append(connect)
    return Response(content=str(response), media_type="application/xml")


# ── Results endpoint ───────────────────────────────────────────────────────────
@app.get("/results/{call_sid}")
async def get_call_result(call_sid: str):
    """Poll this after the call ends to retrieve the summary + Q&A answers."""
    result = call_results.get(call_sid)
    if not result:
        raise HTTPException(status_code=404, detail="No result yet for this call_sid. Call may still be in progress.")
    return result


# ─── WebSocket pipeline ───────────────────────────────────────────────────────

@app.websocket("/media-stream/{call_sid}")
async def media_stream(websocket: WebSocket, call_sid: str):
    await websocket.accept()
    print(f"[{call_sid}] Twilio WebSocket connected", flush=True)

    call_cfg  = call_configs_by_sid.pop(call_sid, None) or build_call_config(None)
    mode      = call_cfg.get("__mode", "chat")     # "chat" | "question"
    el_model  = call_cfg["elevenlabs_model"]
    voice_id  = call_cfg["voice_id"]
    dg_url    = deepgram_ws_url(call_cfg["deepgram_language"])

    # chat-mode only
    system_prompt    = call_cfg.get("system_prompt", "")
    opening_greeting = call_cfg.get("opening_greeting", "Hello!")
    llm_provider     = call_cfg.get("llm_provider", DEFAULT_LLM_PROVIDER)
    llm_model        = call_cfg.get("llm_model", DEFAULT_LLM_MODELS[DEFAULT_LLM_PROVIDER])

    print(f"[{call_sid}] Mode: {mode}", flush=True)

    audio_queue          = asyncio.Queue()
    conversation_history = []
    transcript_buffer    = []
    stream_sid           = None
    agent_speaking       = False
    call_ending          = False      # flag: hang-up in progress, ignore new transcripts

    # ── TTS sender ────────────────────────────────────────────────────────────
    async def send_audio_to_twilio(text: str):
        nonlocal agent_speaking
        agent_speaking = True
        print(f"[{call_sid}] 🔊 Speaking: {text[:100]}", flush=True)
        chunk_count = 0
        async for mulaw_b64 in text_to_mulaw_chunks(text, el_model, voice_id):
            if not agent_speaking:
                print(f"[{call_sid}] ⚡ Interrupted — stopping TTS", flush=True)
                break
            try:
                await websocket.send_text(json.dumps({
                    "event":     "media",
                    "streamSid": stream_sid,
                    "media":     {"payload": mulaw_b64},
                }))
                chunk_count += 1
            except Exception as e:
                print(f"[{call_sid}] Send error: {e}", flush=True)
                break
        try:
            await websocket.send_text(json.dumps({
                "event": "mark", "streamSid": stream_sid, "mark": {"name": "agent_done"}
            }))
        except Exception:
            pass
        agent_speaking = False
        print(f"[{call_sid}] ✅ Sent {chunk_count} chunks", flush=True)

    # ── Hang-up helper ────────────────────────────────────────────────────────
    async def end_call_after_speaking():
        nonlocal call_ending
        call_ending = True
        # Wait for TTS to finish streaming
        while agent_speaking:
            await asyncio.sleep(0.3)
        await asyncio.sleep(1.5)   # let last audio fully flush to Twilio
        try:
            await asyncio.to_thread(
                lambda: twilio_client.calls(call_sid).update(status="completed")
            )
            print(f"[{call_sid}] 📵 Call terminated", flush=True)
        except Exception as e:
            print(f"[{call_sid}] Hang-up error: {e}", flush=True)

    # ── Twilio receiver ───────────────────────────────────────────────────────
    async def receive_from_twilio():
        nonlocal stream_sid
        try:
            while True:
                raw   = await websocket.receive_text()
                data  = json.loads(raw)
                event = data.get("event")

                if event == "connected":
                    print(f"[{call_sid}] Twilio connected", flush=True)

                elif event == "start":
                    stream_sid = data["start"]["streamSid"]
                    print(f"[{call_sid}] Stream started → {stream_sid}", flush=True)

                    if mode == "question":
                        # Initialise session and speak the opener + Q1
                        ctx = call_cfg.get("context", {})
                        ctx["session_id"] = call_sid[:8]
                        session = VoiceQuestionSession(ctx)
                        question_sessions[call_sid] = session

                        first_q = await session.start()
                        opener  = (
                            f"Hi, I'm calling on behalf of your former colleague {ctx.get('name', 'the candidate')}. "
                            f"I have just 5 quick questions for you. Here's the first one: {first_q}"
                        )
                        conversation_history.append({"role": "assistant", "content": opener})
                        asyncio.create_task(send_audio_to_twilio(opener))

                    else:
                        # Chat / sales mode
                        conversation_history.append({"role": "assistant", "content": opening_greeting})
                        asyncio.create_task(send_audio_to_twilio(opening_greeting))

                elif event == "media":
                    await audio_queue.put(base64.b64decode(data["media"]["payload"]))

                elif event == "mark":
                    print(f"[{call_sid}] Mark: {data['mark']['name']}", flush=True)

                elif event == "stop":
                    print(f"[{call_sid}] Stream stopped", flush=True)
                    break

        except WebSocketDisconnect:
            print(f"[{call_sid}] Twilio disconnected", flush=True)
        except Exception as e:
            print(f"[{call_sid}] Twilio receiver error: {e}", flush=True)
        finally:
            await audio_queue.put(None)

    # ── Deepgram pipeline ─────────────────────────────────────────────────────
    async def stream_to_deepgram():
        nonlocal agent_speaking
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        try:
            async with websockets.connect(dg_url, additional_headers=headers,
                                          ping_interval=5, ping_timeout=20) as dg_ws:
                print(f"[{call_sid}] Deepgram connected ✅", flush=True)

                # ── Forward audio to Deepgram ──────────────────────────────
                async def send_audio():
                    while True:
                        chunk = await audio_queue.get()
                        if chunk is None:
                            try:
                                await dg_ws.send(json.dumps({"type": "CloseStream"}))
                            except Exception:
                                pass
                            break
                        await dg_ws.send(chunk)

                # ── Receive transcripts from Deepgram ─────────────────────
                async def receive_transcripts():
                    nonlocal agent_speaking
                    async for raw_msg in dg_ws:
                        if call_ending:
                            continue   # call is being torn down — ignore new speech

                        try:
                            msg        = json.loads(raw_msg)
                            if msg.get("type") != "Results":
                                continue
                            alt        = msg["channel"]["alternatives"][0]
                            transcript = alt.get("transcript", "").strip()
                            if not transcript:
                                continue

                            is_final     = msg.get("is_final", False)
                            speech_final = msg.get("speech_final", False)

                            # Human interrupted the agent
                            if agent_speaking:
                                agent_speaking = False
                                print(f"[{call_sid}] ⚡ Human interrupted agent", flush=True)
                                try:
                                    await websocket.send_text(json.dumps({
                                        "event": "clear", "streamSid": stream_sid
                                    }))
                                except Exception:
                                    pass

                            if is_final:
                                label = "FINAL ✅" if speech_final else "FINAL"
                                print(f"[{call_sid}] [{label}] {transcript}", flush=True)
                                transcript_buffer.append(transcript)

                            if speech_final and transcript_buffer:
                                full_turn = " ".join(transcript_buffer)
                                transcript_buffer.clear()

                                if len(full_turn.split()) < MIN_WORDS_TO_RESPOND:
                                    print(f"[{call_sid}] ⏭ Skipping short turn: '{full_turn}'", flush=True)
                                    continue

                                print(f"[{call_sid}] 🎤 Human: {full_turn}", flush=True)
                                conversation_history.append({"role": "user", "content": full_turn})

                                # ── QUESTION MODE ──────────────────────────
                                if mode == "question":
                                    session = question_sessions.get(call_sid)
                                    if not session or session.is_done:
                                        continue

                                    result      = await session.submit_answer(full_turn)
                                    agent_reply = result["speak"]

                                    conversation_history.append({"role": "assistant", "content": agent_reply})
                                    await send_audio_to_twilio(agent_reply)

                                    if result["is_done"]:
                                        # Store results
                                        call_results[call_sid] = {
                                            "call_sid": call_sid,
                                            "context":  session.context,
                                            "answers":  session.answers,
                                            "summary":  result["summary"],
                                        }
                                        print(f"[{call_sid}] 📋 Summary stored", flush=True)

                                        # Fire webhook if configured
                                        cb_url = call_cfg.get("callback_url")
                                        if cb_url:
                                            asyncio.create_task(_post_callback(cb_url, call_results[call_sid]))

                                        # Hang up after TTS finishes
                                        asyncio.create_task(end_call_after_speaking())

                                # ── CHAT MODE ─────────────────────────────
                                else:
                                    agent_reply = await ask_llm(
                                        conversation_history, system_prompt,
                                        llm_provider, llm_model
                                    )
                                    conversation_history.append({"role": "assistant", "content": agent_reply})
                                    print(f"[{call_sid}] 🤖 [{llm_provider}] Agent: {agent_reply}", flush=True)
                                    await send_audio_to_twilio(agent_reply)

                        except Exception as e:
                            print(f"[{call_sid}] Transcript error: {e}", flush=True)

                await asyncio.gather(send_audio(), receive_transcripts())

        except websockets.exceptions.InvalidStatus as e:
            print(f"[{call_sid}] ❌ Deepgram rejected: {e}", flush=True)
        except Exception as e:
            print(f"[{call_sid}] Deepgram error: {type(e).__name__}: {e}", flush=True)

    await asyncio.gather(receive_from_twilio(), stream_to_deepgram())
    print(f"[{call_sid}] Pipeline finished", flush=True)