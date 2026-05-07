import json
import asyncio
import httpx
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from pathlib import Path
from watchfiles import awatch
import uvicorn
import argparse
import ahocorasick as pyahocorasick
import re
from colorama import Fore, Style, init
init()

parser = argparse.ArgumentParser(description="llama.cpp Multi-Token Phrase Filter Proxy")
parser.add_argument("--llama-port", type=int, default=8080, help="Port llama.cpp is running on")
parser.add_argument("--proxy-port", type=int, default=5001, help="Port this proxy listens on")
args = parser.parse_args()

LLAMA_HOST = f"http://127.0.0.1:{args.llama_port}"
PROXY_PORT = args.proxy_port

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MAX_REWINDS      = 999
BAN_BIAS         = -999.0
BANNED_FILE_PATH = "banned_phrases.txt"
VERBOSE          = False

# ─────────────────────────────────────────────
# PHRASE LOADING
# ─────────────────────────────────────────────

CHANNEL_OPEN_RE  = re.compile(r"<\|channel\|?>\s*(thought|analysis|final|assistant)\b", re.IGNORECASE)
CHANNEL_CLOSE_RE = re.compile(r"<channel\|>\s*", re.IGNORECASE)

THINK_OPEN_RE    = re.compile(r"<think>|<thought>|<\|im_start\|>thought", re.IGNORECASE)
THINK_CLOSE_RE   = re.compile(r"</think>|</thought>|<\|im_end\|>", re.IGNORECASE)

class ControlTokenRouter:
    """
    Streaming parser that:
      - detects control markers even when split across tokens
      - removes them from output
      - routes text to mode: "reasoning" or "content"
    """
    def __init__(self, initial_mode: str = "content", hold_len: int = 64):
        self.mode = initial_mode  # "content" or "reasoning"
        self.buf = ""
        self.hold_len = hold_len

    def reset(self):
        self.buf = ""

    def _find_next(self):
        # Return (start, end, kind, value) for the earliest control match in self.buf
        best = None

        candidates = [
            ("channel_open",  CHANNEL_OPEN_RE),
            ("channel_close", CHANNEL_CLOSE_RE),
            ("think_open",    THINK_OPEN_RE),
            ("think_close",   THINK_CLOSE_RE),
        ]

        for kind, rx in candidates:
            m = rx.search(self.buf)
            if not m:
                continue
            tup = (m.start(), m.end(), kind, m.group(1).lower() if m.lastindex else None)
            if best is None or tup[0] < best[0]:
                best = tup

        return best

    def feed(self, s: str) -> list[tuple[str, str]]:
        self.buf += (s or "")
        out: list[tuple[str, str]] = []

        # Consume all complete control markers currently visible in buf
        while True:
            hit = self._find_next()
            if not hit:
                break

            start, end, kind, val = hit

            # Emit text before marker
            if start > 0:
                out.append((self.mode, self.buf[:start]))

            # Apply marker effect (and drop it)
            if kind == "channel_open":
                if val in ("thought", "analysis"):
                    self.mode = "reasoning"
                else:
                    self.mode = "content"
            elif kind == "channel_close":
                self.mode = "content"
            elif kind == "think_open":
                self.mode = "reasoning"
            elif kind == "think_close":
                self.mode = "content"

            self.buf = self.buf[end:]

        # Hold back a possible partial marker (e.g. "<|channel>" without the next token yet)
        lt = self.buf.rfind("<")
        if lt != -1 and (len(self.buf) - lt) <= self.hold_len:
            flush_upto = lt
        else:
            flush_upto = len(self.buf)

        if flush_upto > 0:
            out.append((self.mode, self.buf[:flush_upto]))
            self.buf = self.buf[flush_upto:]

        return out

def load_banned_phrases(file_path: str = BANNED_FILE_PATH) -> list[str]:
    path = Path(file_path)
    if not path.exists():
        print(f"[LOAD] Warning: {file_path} not found, creating empty file")
        path.write_text("", encoding='utf-8')
        return []
    try:
        content = path.read_text(encoding='utf-8').strip()
        if not content:
            print(f"[LOAD] File is empty")
            return []

        phrases = []
        for m in re.finditer(r'"([^"]*)"|\'([^\']*)\'|([^,\'"]+)', content):
            p = (m.group(1) if m.group(1) is not None else
                m.group(2) if m.group(2) is not None else
                m.group(3) or "").strip()
            if p:
                phrases.append(p)

        phrases = sorted(
            {p.lower() for p in phrases if p.strip()},
            key=len, reverse=True
        )
        print(f"[LOAD] Loaded {len(phrases)} banned phrases from {file_path}")
        return phrases
    except Exception as e:
        print(f"[LOAD] Error loading {file_path}: {e}")
        return []


def build_phrase_automaton(phrases: list[str]) -> pyahocorasick.Automaton:
    automaton = pyahocorasick.Automaton()
    for phrase in phrases:
        automaton.add_word(phrase.lower(), phrase)
    automaton.make_automaton()
    return automaton


async def file_watcher(file_path: str = BANNED_FILE_PATH):
    global ban_phrases, n_buffer, phrase_automaton
    path = Path(file_path)
    if not path.exists():
        print(f"[WATCHER] File {file_path} doesn't exist yet, waiting...")
    print(f"[WATCHER] Started watching {file_path}")
    print(f"\n[PROXY] Running on http://127.0.0.1:{PROXY_PORT}\n")
    async for changes in awatch(path.parent):
        for change_type, changed_path in changes:
            if Path(changed_path) == path.absolute():
                print(f"\n[WATCHER] ═══════════════════════════════════════")
                print(f"[WATCHER] File change detected: {file_path}")
                new_phrases = load_banned_phrases(file_path)
                if new_phrases != ban_phrases:
                    ban_phrases = new_phrases
                    phrase_automaton = build_phrase_automaton(ban_phrases) if ban_phrases else None
                    print(f"[WATCHER] Updated banned phrases ({len(ban_phrases)}): {ban_phrases}")
                    n_buffer = await compute_n_buffer()
                    print(f"[WATCHER] Updated token buffer size: {n_buffer}")
                else:
                    print(f"[WATCHER] No changes in phrases")
                print(f"[WATCHER] ═══════════════════════════════════════\n")
                break

# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────

ban_phrases: list[str] = []
n_buffer: int = 0
phrase_automaton: pyahocorasick.Automaton | None = None
model_name: str = "unknown"
system_fingerprint: str = "unknown"
token_id_to_text: dict[int, str] = {}


async def _tokenize_one(text: str) -> list[int]:
    try:
        r = await client.post(
            "/tokenize",
            json={"content": text, "add_special": False, "parse_special": True},
            timeout=10,
        )
        r.raise_for_status()
        return [int(t) for t in r.json().get("tokens", [])]
    except Exception as e:
        print(f"[TOKENIZE] error for {text!r}: {e}")
        return []


async def compute_n_buffer() -> int:
    if not ban_phrases:
        return 0
    
    tasks = [_tokenize_one(p) for p in ban_phrases]
    results = await asyncio.gather(*tasks)
    
    max_len = max((len(toks) for toks in results), default=0)
    return max_len + 3


@asynccontextmanager
async def lifespan(app: FastAPI):
    global n_buffer, ban_phrases, model_name, system_fingerprint, phrase_automaton

    print("\n" + "="*60)
    print("PHRASE FILTER PROXY — string-ban / rewind mode")
    print("="*60)

    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{LLAMA_HOST}/props", timeout=10)
            r.raise_for_status()
            props = r.json()
            model_name = props.get("model_alias", "unknown")
            system_fingerprint = props.get("build_info", "unknown")
            print(f"[STARTUP] model_name: {model_name}")
    except Exception as e:
        print(f"[STARTUP] Warning: could not fetch /props: {e}")

    ban_phrases = load_banned_phrases(BANNED_FILE_PATH)
    phrase_automaton = build_phrase_automaton(ban_phrases) if ban_phrases else None
    print(f"[STARTUP] Banned phrases ({len(ban_phrases)}): {ban_phrases}")

    n_buffer = await compute_n_buffer()
    print("="*60 + "\n")

    watcher_task = asyncio.create_task(file_watcher(BANNED_FILE_PATH))
    yield
    watcher_task.cancel()
    try:
        await watcher_task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)
client = httpx.AsyncClient(base_url=LLAMA_HOST, timeout=120)


class ChunkBuilder:
    def __init__(self, slot_id: int):
        self.slot_id = slot_id
        self.base_chunk = {
            "id": f"chatcmpl-proxy-{slot_id}",
            "object": "chat.completion.chunk",
            "model": model_name,
            "system_fingerprint": system_fingerprint,
        }

    def build(self, slot: 'SlotState', content: str, finish_reason: str | None, 
            raw_data: dict | None = None, reasoning_content: str = None) -> bytes:
        chunk = self.base_chunk.copy()
        chunk["created"] = int(time.time())
        
        delta = {}
        if content:
            delta["content"] = content
        if reasoning_content:
            delta["reasoning_content"] = reasoning_content
        
        chunk["choices"] = [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }]
        
        if raw_data and "timings" in raw_data:
            chunk["timings"] = slot.merged_timings_with(raw_data["timings"])
        
        if finish_reason == "stop" and raw_data:
            prompt_toks = slot.total_prompt_tokens or raw_data.get("tokens_evaluated", 0)
            chunk["usage"] = {
                "prompt_tokens": prompt_toks,
                "completion_tokens": slot.total_tokens_predicted,
                "total_tokens": prompt_toks + slot.total_tokens_predicted,
            }
        
        return f"data: {json.dumps(chunk)}\n\n".encode()


# ─────────────────────────────────────────────
# SLOT STATE
# ─────────────────────────────────────────────

class SlotState:
    def __init__(self, slot_id: int):
        self.slot_id = slot_id
        self.token_buffer: list[dict] = []
        self.logit_bias: dict[str, float] = {}
        self.rewind_count: int = 0
        self.pre_trap_bias: dict[str, float] = {}
        self.in_trap: bool = False

        self.committed_n: int = 0
        self.committed_ms: float = 0.0
        self.committed_prompt_n: int = 0
        self.committed_prompt_ms: float = 0.0
        self.last_timings: dict = {}
        
        self._buffered_text_cache: str = ""
        self._cache_valid: bool = True

    def _invalidate_cache(self):
        self._cache_valid = False

    @property
    def buffered_text(self) -> str:
        if not self._cache_valid:
            self._buffered_text_cache = "".join(t["text"] for t in self.token_buffer).lower()
            self._cache_valid = True
        return self._buffered_text_cache

    def find_ban(self) -> tuple[bool, int, str]:
        if not phrase_automaton:
            return False, 0, ""
        
        text = self.buffered_text
        is_stop = self.token_buffer[-1]["stop"] if self.token_buffer else False
        
        for end_pos, phrase in phrase_automaton.iter(text):
            pos = end_pos - len(phrase) + 1
            
            # Word boundary checks
            if phrase[0].isalnum() or phrase[0] == '_':
                if pos > 0 and (text[pos - 1].isalnum() or text[pos - 1] == '_'):
                    continue
            
            if phrase[-1].isalnum() or phrase[-1] == '_':
                if end_pos + 1 < len(text):
                    if text[end_pos + 1].isalnum() or text[end_pos + 1] == '_':
                        continue
                else:
                    if not is_stop:
                        continue
            
            # Calculate rewind
            char_count = 0
            n_rewind = 0
            for tok in reversed(self.token_buffer):
                char_count += len(tok["text"])
                n_rewind += 1
                if char_count >= len(text) - pos:
                    break
            
            return True, n_rewind, phrase
        
        return False, 0, ""

    def apply_rewind_bias(self, n_rewind: int, phrase: str = ""):
        seen: set[str] = set()
        trigger_bans = []
        
        first_tok = self.token_buffer[-n_rewind]
        tid_str = str(first_tok["tok"])
        if tid_str not in seen and first_tok["tok"] != -1:
            self.logit_bias[tid_str] = BAN_BIAS
            seen.add(tid_str)
            trigger_bans.append(f"{tid_str}({first_tok['text']!r})")
        
        if VERBOSE:
            print(f"[BIAS] trigger tokens banned: {trigger_bans}")
        
        self.token_buffer = self.token_buffer[:-n_rewind]
        self._invalidate_cache()

    def flush_safe_prefix(self) -> str:
        flushed_text = "".join(t["text"] for t in self.token_buffer)
        self.token_buffer = []
        self._invalidate_cache()
        return flushed_text

    def absorb_timings(self, data: dict):
        t = data.get("timings")
        if t:
            self.last_timings = t

    def commit_attempt(self):
        if self.last_timings:
            self.committed_n += self.last_timings.get("predicted_n", 0)
            self.committed_ms += self.last_timings.get("predicted_ms", 0.0)
            self.committed_prompt_n += self.last_timings.get("prompt_n", 0)
            self.committed_prompt_ms += self.last_timings.get("prompt_ms", 0.0)
            self.last_timings = {}

    def merged_timings_with(self, raw_t: dict) -> dict:
        if not raw_t:
            return {}
        t = dict(raw_t)

        total_n = self.committed_n + t.get("predicted_n", 0)
        total_ms = self.committed_ms + t.get("predicted_ms", 0.0)
        t["predicted_n"] = total_n
        t["predicted_ms"] = total_ms
        if total_ms > 0 and total_n > 0:
            t["predicted_per_token_ms"] = total_ms / total_n
            t["predicted_per_second"] = total_n / (total_ms / 1000.0)

        total_prompt_n = self.committed_prompt_n + t.get("prompt_n", 0)
        total_prompt_ms = self.committed_prompt_ms + t.get("prompt_ms", 0.0)
        t["prompt_n"] = total_prompt_n
        t["prompt_ms"] = total_prompt_ms
        if total_prompt_ms > 0 and total_prompt_n > 0:
            t["prompt_per_token_ms"] = total_prompt_ms / total_prompt_n
            t["prompt_per_second"] = total_prompt_n / (total_prompt_ms / 1000.0)

        return t

    @property
    def total_tokens_predicted(self) -> int:
        return self.committed_n + self.last_timings.get("predicted_n", 0)

    @property
    def total_prompt_tokens(self) -> int:
        return self.committed_prompt_n + self.last_timings.get("prompt_n", 0)

    def flush_tokens(self, n_flush: int, chunk_builder: ChunkBuilder, 
                     last_data: dict = None) -> tuple[list[bytes], str]:
        chunks = []
        confirmed = []
        
        for tok_info in self.token_buffer[:n_flush]:
            if tok_info["text"]:
                is_stop = tok_info["stop"]
                finish = "stop" if is_stop else None
                chunks.append(chunk_builder.build(self, tok_info["text"], finish, last_data))
                confirmed.append(tok_info["text"])
        
        self.token_buffer = self.token_buffer[n_flush:]
        self._invalidate_cache()
        
        return chunks, "".join(confirmed)


# ─────────────────────────────────────────────
# STREAMING GENERATOR
# ─────────────────────────────────────────────

async def refresh_model_info():
    global model_name, system_fingerprint
    try:
        r = await client.get("/props", timeout=5)
        r.raise_for_status()
        props = r.json()
        model_name = props.get("model_alias", "unknown")
        system_fingerprint = props.get("build_info", "unknown")
    except Exception as e:
        print(f"[REFRESH] Could not refresh /props: {e}")

async def stream_with_ban(messages: list, body: dict, slot_id: int):
    await refresh_model_info()

    slot = SlotState(slot_id)
    chunk_builder = ChunkBuilder(slot_id)

    confirmed_parts: list[str] = []
    attempt = 0

    user_bias: dict[str, float] = {}
    if "logit_bias" in body:
        raw = body["logit_bias"]
        try:
            if isinstance(raw, dict):
                user_bias = {str(k): float(v) for k, v in raw.items()}
            elif isinstance(raw, list):
                # [[id, bias], ...]
                tmp = {}
                for pair in raw:
                    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                        tmp[str(pair[0])] = float(pair[1])
                user_bias = tmp
        except Exception as e:
            print(f"[STREAM/{slot_id}] WARN: couldn't parse user logit_bias: {e}")

    async def apply_template(msgs: list, template_kwargs: dict | None = None) -> str:
        try:
            req = {"messages": msgs, "add_generation_prompt": True}
            if template_kwargs and isinstance(template_kwargs, dict):
                req["chat_template_kwargs"] = template_kwargs
            r = await client.post("/apply-template", json=req, timeout=15)
            r.raise_for_status()
            prompt = r.json().get("prompt", "")
            return prompt
        except Exception as e:
            print(f"[STREAM/{slot_id}] [TEMPLATE] /apply-template failed: {e}")
            return "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in msgs)

    user_template_kwargs = body.get("chat_template_kwargs", {})
    if not isinstance(user_template_kwargs, dict):
        user_template_kwargs = {}

    # ─────────────────────────────────────────────
    # PHASE 1 — passthrough via /v1/chat/completions
    # ─────────────────────────────────────────────

    oai_body = dict(body)
    oai_body["stream"] = True

    content_mode_started = False
    reasoning_acc =[]

    async with client.stream("POST", "/v1/chat/completions", json=oai_body, timeout=300) as resp:
        if resp.status_code != 200:
            raw_err = await resp.aread()
            txt_err = raw_err.decode("utf-8", errors="replace")
            yield b"data: [DONE]\n\n"
            return

        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue

            raw_payload = line[5:].strip()

            if raw_payload == "[DONE]":
                yield b"data: [DONE]\n\n"
                return

            # Forward unparseable payloads as-is
            try:
                chunk = json.loads(raw_payload)
            except Exception:
                yield (f"data: {raw_payload}\n\n").encode()
                continue

            choices = chunk.get("choices") or []
            delta = (choices[0].get("delta") if choices else {}) or {}
            finish_reason = choices[0].get("finish_reason") if choices else None
            delta_content = delta.get("content", None)
            delta_reasoning = delta.get("reasoning_content", None)

            if isinstance(delta_reasoning, str) and delta_reasoning:
                reasoning_acc.append(delta_reasoning)

            # Detect first visible content
            if isinstance(delta_content, str) and delta_content and not delta_content.isspace():
                content_mode_started = True
                break

            # If finished before any content, passthrough end
            if finish_reason is not None:
                yield (f"data: {raw_payload}\n\n").encode()
                yield b"data: [DONE]\n\n"
                return

            # Otherwise forward transparently
            yield (f"data: {raw_payload}\n\n").encode()

    if not content_mode_started:
        yield b"data: [DONE]\n\n"
        return

    # ─────────────────────────────────────────────
    # PHASE 2 — /completion with rewind + token-id bans
    # ─────────────────────────────────────────────

    base_prompt = await apply_template(messages, user_template_kwargs)
    reasoning_str = "".join(reasoning_acc)

    # Process Phase 1 reasoning into the base_prompt before rewinds
    m_think = re.search(r"(<think>|<thought>|<\|im_start\|>thought)[\s\n]*$", base_prompt, re.IGNORECASE)
    if m_think:
        tag_str = m_think.group(1).lower()
        if "im_start" in tag_str:
            close_tag = "<|im_end|>"
        elif "thought" in tag_str:
            close_tag = "</thought>"
        else:
            close_tag = "</think>"
        
        # Append the gathered reasoning chunks, avoiding double-closing
        if reasoning_str and not reasoning_str.strip().endswith(close_tag):
            reasoning_str += "\n" + close_tag

        if not reasoning_str:
            reasoning_str = close_tag

        base_prompt += reasoning_str + "\n"
    else:
        # If model somehow thought without an opening tag in the template
        if reasoning_str:
            if not reasoning_str.strip().endswith("</think>"):
                reasoning_str += "\n</think>"
            base_prompt += "<think>\n" + reasoning_str + "\n"

    # minimal list of stop strings to avoid special-token leakage
    default_stop =["<|im_end|>", "<|endoftext|>", "</s>"]

    while True:
        attempt += 1
        mute_thoughts = (attempt > 1)
        router = ControlTokenRouter(initial_mode="content")

        # Build prompt = template + fully resolved thoughts + confirmed content
        prompt = base_prompt

        if confirmed_parts:
            prompt += "".join(confirmed_parts)

        # ─────────────────────────────────────────────
        # Build /completion body
        # ─────────────────────────────────────────────
        completion_body = {}

        keep_list_or_dict_keys = {
            "stop",
            "samplers",
            "lora",
            "logit_bias",
            "response_fields",
            "dry_sequence_breakers",
        }

        for k, v in body.items():
            if k in ("messages", "stream", "model", "chat_template_kwargs"):
                continue

            if isinstance(v, (str, int, float, bool)) or v is None:
                completion_body[k] = v
            elif k in keep_list_or_dict_keys and isinstance(v, (list, dict)):
                completion_body[k] = v
            else:
                # drop other list/dict payloads silently
                pass

        completion_body["prompt"] = prompt
        completion_body["stream"] = True
        completion_body["cache_prompt"] = True

        # Handle max_tokens → n_predict (if present)
        if "max_tokens" in completion_body:
            completion_body["n_predict"] = completion_body.pop("max_tokens")

        # Ensure stop list includes default stop strings
        stop_val = completion_body.get("stop", [])
        if isinstance(stop_val, str):
            stop_val = [stop_val]
        if not isinstance(stop_val, list):
            stop_val = []
        for s in default_stop:
            if s not in stop_val:
                stop_val.append(s)
        completion_body["stop"] = stop_val

        # Merge logit bias (token IDs) user + our runtime bans
        merged_bias = {**user_bias, **slot.logit_bias}
        if merged_bias:
            # llama.cpp accepts dict or list; keep list-of-pairs like your original
            lb_pairs = []
            for tid, bias in merged_bias.items():
                try:
                    lb_pairs.append([int(tid), float(bias)])
                except Exception:
                    # if somehow non-int keys exist, skip (we are ID-banning)
                    pass
            completion_body["logit_bias"] = lb_pairs

        rewind_triggered = False

        async with client.stream("POST", "/completion", json=completion_body, timeout=300) as resp:
            if resp.status_code != 200:
                raw_err = await resp.aread()
                yield b"data: [DONE]\n\n"
                return

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue

                msg = line[5:].strip()
                if msg == "[DONE]":

                    # Flush anything buffered
                    if slot.token_buffer:
                        chunks, confirmed_text = slot.flush_tokens(len(slot.token_buffer), chunk_builder, None)
                        for chunk in chunks:
                            yield chunk
                        if confirmed_text:
                            confirmed_parts.append(confirmed_text)

                    yield b"data: [DONE]\n\n"
                    return

                try:
                    data = json.loads(msg)
                except Exception:
                    continue

                content_text = data.get("content", "")
                token_id = data["tokens"][0] if data.get("tokens") else -1
                stop = data.get("stop", False)

                slot.absorb_timings(data)

                # Keep raw mapping behavior (optional, but matches your current behavior)
                if token_id != -1 and content_text and token_id not in token_id_to_text:
                    token_id_to_text[token_id] = content_text

                # Route/strip control tokens (Gemma channels + Qwen think)
                for mode, piece in router.feed(content_text):
                    if not piece:
                        continue

                    # Now strip stop-token text if it leaked (do this AFTER routing,
                    # so <|im_end|> can still be used as a think-close marker)
                    for st in default_stop + ["<|im_start|>"]:
                        piece = piece.replace(st, "")
                    if not piece:
                        continue

                    if mode == "reasoning":
                        if not mute_thoughts:
                            yield chunk_builder.build(slot, "", None, data, reasoning_content=piece)
                        continue

                    # content mode
                    slot.token_buffer.append({
                        "tok": token_id,
                        "text": piece,
                        "stop": False,
                    })

                # stop handling
                if stop:
                    router.reset()  # drop any held partial marker at end
                    slot.token_buffer.append({"tok": -1, "text": "", "stop": True})

                slot._invalidate_cache()

                # ─────────────────────────────────────────────
                # Ban check (rewind)
                # ─────────────────────────────────────────────
                if ban_phrases and n_buffer > 0:
                    found, n_rewind, triggered_phrase = slot.find_ban()
                    if found:
                        culprit_token = slot.token_buffer[-n_rewind]
                        token_repr = f"{culprit_token['tok']}({culprit_token['text']!r})"

                        if VERBOSE:
                            print(Fore.YELLOW + f"[REWIND] #{slot.rewind_count} phrase={triggered_phrase!r} via: {token_repr}" + Style.RESET_ALL)
                            
                            active_bans_with_text = [f"{tid}({token_id_to_text.get(int(tid), '???')!r})" for tid in slot.logit_bias.keys()]
                            print(f"[REWIND] active bans: {active_bans_with_text}")

                        if slot.rewind_count < MAX_REWINDS:
                            if not slot.in_trap:
                                slot.pre_trap_bias = dict(slot.logit_bias)
                                slot.in_trap = True

                            slot.rewind_count += 1
                            slot.commit_attempt()
                            slot.apply_rewind_bias(n_rewind, triggered_phrase)

                            text_to_flush = slot.flush_safe_prefix()
                            if text_to_flush:
                                yield chunk_builder.build(slot, text_to_flush, None, None)
                                confirmed_parts.append(text_to_flush)

                            rewind_triggered = True
                            break
                        else:
                            print(f"[REWIND] ⚠ Max rewinds reached ({MAX_REWINDS})")

                # ─────────────────────────────────────────────
                # Flush safe prefix
                # ─────────────────────────────────────────────
                if stop:
                    n_flush = len(slot.token_buffer)
                else:
                    n_flush = max(0, len(slot.token_buffer) - n_buffer)

                if n_flush > 0:
                    chunks, confirmed_text = slot.flush_tokens(n_flush, chunk_builder, data)
                    for chunk in chunks:
                        yield chunk
                    if confirmed_text:
                        confirmed_parts.append(confirmed_text)

                        if slot.in_trap:
                            if VERBOSE:
                                print(f"[ESCAPE] Escaped, rolling back bias from {len(slot.logit_bias)} to {len(slot.pre_trap_bias)}\n")

                            slot.logit_bias = dict(slot.pre_trap_bias)
                            slot.in_trap = False

                if stop:
                    slot.commit_attempt()

                    # flush remaining
                    if slot.token_buffer:
                        chunks, confirmed_text = slot.flush_tokens(len(slot.token_buffer), chunk_builder, data)
                        for chunk in chunks:
                            yield chunk
                        if confirmed_text:
                            confirmed_parts.append(confirmed_text)

                    yield chunk_builder.build(slot, "", "stop", data)
                    yield b"data: [DONE]\n\n"
                    return

        if rewind_triggered:
            continue

        # If stream ended without stop, flush and end
        if slot.token_buffer:
            chunks, confirmed_text = slot.flush_tokens(len(slot.token_buffer), chunk_builder, None)
            for chunk in chunks:
                yield chunk
            if confirmed_text:
                confirmed_parts.append(confirmed_text)

        yield b"data: [DONE]\n\n"
        return
    
# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    slot_id = hash(json.dumps(messages)) % 10000

    if body.get("stream", False):
        return StreamingResponse(
            stream_with_ban(messages, body, slot_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
    full_text_parts = []
    async for chunk in stream_with_ban(messages, body, slot_id):
        if chunk == b"data: [DONE]\n\n":
            continue
        line = chunk.decode()
        if line.startswith("data:"):
            msg = line[5:].strip()
            if msg == "[DONE]":
                continue
            try:
                data = json.loads(msg)
                content = data["choices"][0]["delta"].get("content", "")
                if content:
                    full_text_parts.append(content)
            except:
                pass

    return Response(
        content=json.dumps({
            "id": f"chatcmpl-proxy-{slot_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "system_fingerprint": system_fingerprint,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "".join(full_text_parts)},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }),
        status_code=200,
        media_type="application/json",
    )

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def passthrough(request: Request, path: str):
    url = f"{LLAMA_HOST}/{path}"
    body = await request.body()
    
    # Use the GLOBAL client to maintain keep-alive TCP connections
    r = await client.request(
        method=request.method, 
        url=url, 
        content=body,
        headers={k: v for k, v in request.headers.items()
                 if k.lower() not in ("host", "connection")},
    )
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type"))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PROXY_PORT, log_level="warning")
