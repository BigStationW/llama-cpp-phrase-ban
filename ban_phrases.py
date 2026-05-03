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

parser = argparse.ArgumentParser(description="llama.cpp Multi-Token Phrase Filter Proxy")
parser.add_argument("--llama-port", type=int, default=8080, help="Port llama.cpp is running on")
parser.add_argument("--proxy-port", type=int, default=5001, help="Port this proxy listens on")
args = parser.parse_args()

LLAMA_HOST  = f"http://127.0.0.1:{args.llama_port}"
PROXY_PORT  = args.proxy_port

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MAX_REWINDS      = 999
BAN_BIAS         = -999.0
BANNED_FILE_PATH = "banned_phrases.txt"
VERBOSE          =  False

# ─────────────────────────────────────────────
# PHRASE LOADING
# ─────────────────────────────────────────────

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

        # Use regex to extract quoted strings and bare tokens separately
        phrases = []
        # Match "..." or '...' (including content with commas), or bare non-comma tokens
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
    """Build Aho-Corasick automaton for fast multi-pattern matching"""
    automaton = pyahocorasick.Automaton()
    for phrase in phrases:
        automaton.add_word(phrase.lower(), phrase)
    automaton.make_automaton()
    return automaton


async def file_watcher(file_path: str = BANNED_FILE_PATH):
    global ban_phrases, n_buffer, phrase_token_variants, phrase_automaton
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
phrase_token_variants: dict[str, list[tuple[int, str]]] = {}
phrase_automaton: pyahocorasick.Automaton | None = None
model_name:         str = "unknown"
system_fingerprint: str = "unknown"
token_id_to_text: dict[int, str] = {}


async def _tokenize_one(text: str, client: httpx.AsyncClient) -> list[int]:
    try:
        r = await client.post(
            f"{LLAMA_HOST}/tokenize",
            json={"content": text, "add_special": False, "parse_special": True},
            timeout=10,
        )
        r.raise_for_status()
        return [int(t) for t in r.json().get("tokens", [])]
    except Exception as e:
        print(f"[TOKENIZE] error for {text!r}: {e}")
        return []

async def compute_n_buffer() -> int:
    global token_id_to_text
    if not ban_phrases:
        return 0
    
    async with httpx.AsyncClient() as c:
        tasks = []
        for p in ban_phrases:
            tasks.append(_tokenize_one(p, c))
        
        results = await asyncio.gather(*tasks)
    
    max_len = 0
    all_first_tids: list[int] = []

    for i, phrase in enumerate(ban_phrases):
        toks = results[i]
        first_toks: list[tuple[int, str]] = []
        if toks:
            first_toks.append((toks[0], phrase))
            all_first_tids.append(toks[0])

        phrase_token_variants[phrase] = first_toks
        max_len = max(max_len, len(toks))

    # Detokenize all first-token IDs to get their display text
    async with httpx.AsyncClient() as c:
        detok_tasks = [
            c.post(f"{LLAMA_HOST}/detokenize", json={"tokens": [tid]}, timeout=10)
            for tid in all_first_tids
        ]
        detok_results = await asyncio.gather(*detok_tasks)
        for tid, r in zip(all_first_tids, detok_results):
            try:
                token_id_to_text[tid] = r.json().get("content", "?")
            except Exception:
                token_id_to_text[tid] = "?"

    return max_len + 3

@asynccontextmanager
async def lifespan(app: FastAPI):
    global n_buffer, ban_phrases, model_name, system_fingerprint, phrase_automaton

    print("\n" + "="*60)
    print("PHRASE FILTER PROXY — string-ban / rewind mode")
    print("="*60)

    # ── fetch model metadata once from /props ──────────────────
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{LLAMA_HOST}/props", timeout=10)
            r.raise_for_status()
            props              = r.json()
            model_name         = props.get("model_alias", "unknown")
            system_fingerprint = props.get("build_info",  "unknown")
            print(f"[STARTUP] model_name        : {model_name}")
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


app    = FastAPI(lifespan=lifespan)
client = httpx.AsyncClient(base_url=LLAMA_HOST, timeout=120)


class ChunkBuilder:
    """Pre-built chunk template to avoid recreation in hot path"""
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
        self.slot_id       = slot_id
        self.token_buffer: list[dict]       = []
        self.logit_bias:   dict[str, float] = {}
        self.rewind_count: int              = 0
        self.pre_trap_bias: dict[str, float] = {}
        self.in_trap: bool = False

        self.committed_n:  int   = 0
        self.committed_ms: float = 0.0

        self.committed_prompt_n:  int   = 0
        self.committed_prompt_ms: float = 0.0

        self.last_timings: dict = {}
        
        self._buffered_text_cache: str = ""
        self._cache_valid: bool = True

    def _invalidate_cache(self):
        """Invalidate buffered text cache when buffer changes"""
        self._cache_valid = False

    @property
    def buffered_text(self) -> str:
        """Cached version of buffered text to avoid repeated string joins"""
        if not self._cache_valid:
            self._buffered_text_cache = "".join(t["text"] for t in self.token_buffer).lower()
            self._cache_valid = True
        return self._buffered_text_cache

    def find_ban(self) -> tuple[bool, int, str]:
        if not phrase_automaton:
            return False, 0, ""
        
        text = self.buffered_text
        is_stop = self.token_buffer[-1]["stop"] if self.token_buffer else False
        
        # Find first match using automaton
        for end_pos, phrase in phrase_automaton.iter(text):
            pos = end_pos - len(phrase) + 1
            
            # ── Word Boundary Checks ───────────────────────────
            # Left boundary: if phrase starts with a word character, 
            # ensure the character right before the match isn't one.
            if phrase[0].isalnum() or phrase[0] == '_':
                if pos > 0 and (text[pos - 1].isalnum() or text[pos - 1] == '_'):
                    continue  # Invalid match (inside a word on the left)
            
            # Right boundary: if phrase ends with a word character,
            # ensure the character right after the match isn't one.
            if phrase[-1].isalnum() or phrase[-1] == '_':
                if end_pos + 1 < len(text):
                    if text[end_pos + 1].isalnum() or text[end_pos + 1] == '_':
                        continue  # Invalid match (inside a word on the right)
                else:
                    # Match is exactly at the end of the current text buffer.
                    # We can't know yet if the next token will continue the word (e.g. "her" -> "here").
                    # We skip for now and wait for the next token, unless generation has stopped.
                    if not is_stop:
                        continue
            
            # ── Calculate Rewind ───────────────────────────────
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
            self.committed_n  += self.last_timings.get("predicted_n",  0)
            self.committed_ms += self.last_timings.get("predicted_ms", 0.0)

            self.committed_prompt_n  += self.last_timings.get("prompt_n",  0)
            self.committed_prompt_ms += self.last_timings.get("prompt_ms", 0.0)
            self.last_timings = {}

    def merged_timings_with(self, raw_t: dict) -> dict:
        if not raw_t:
            return {}
        t = dict(raw_t)

        # ── generation side ────────────────────────────────
        total_n  = self.committed_n  + t.get("predicted_n",  0)
        total_ms = self.committed_ms + t.get("predicted_ms", 0.0)
        t["predicted_n"]  = total_n
        t["predicted_ms"] = total_ms
        if total_ms > 0 and total_n > 0:
            t["predicted_per_token_ms"] = total_ms / total_n
            t["predicted_per_second"]   = total_n / (total_ms / 1000.0)

        total_prompt_n  = self.committed_prompt_n  + t.get("prompt_n",  0)
        total_prompt_ms = self.committed_prompt_ms + t.get("prompt_ms", 0.0)
        t["prompt_n"]  = total_prompt_n
        t["prompt_ms"] = total_prompt_ms
        if total_prompt_ms > 0 and total_prompt_n > 0:
            t["prompt_per_token_ms"] = total_prompt_ms / total_prompt_n
            t["prompt_per_second"]   = total_prompt_n / (total_prompt_ms / 1000.0)

        return t

    @property
    def total_tokens_predicted(self) -> int:
        return self.committed_n + self.last_timings.get("predicted_n", 0)

    @property
    def total_prompt_tokens(self) -> int:
        return self.committed_prompt_n + self.last_timings.get("prompt_n", 0)

    def flush_tokens(self, n_flush: int, chunk_builder: ChunkBuilder, 
                     last_data: dict = None) -> tuple[list[bytes], str]:
        """
        Flush n_flush tokens from buffer and return chunks + confirmed text
        Returns: (list of chunk bytes, confirmed_text_from_flush)
        """
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

async def stream_with_ban(
    messages:   list,
    body:       dict,
    extra_bias: dict,
    slot_id:    int,
):
    slot          = SlotState(slot_id)
    chunk_builder = ChunkBuilder(slot_id)
    confirmed_parts =[]
    attempt       = 0

    def build_chat_body(messages_list: list) -> dict:
        cb = body.copy()
        cb["messages"] = messages_list
        cb["stream"] = True
        cb["cache_prompt"] = True

        if attempt > 1:
            kwargs = cb.get("chat_template_kwargs", {}).copy()
            kwargs["enable_thinking"] = False
            cb["chat_template_kwargs"] = kwargs

        merged_bias = {**extra_bias, **slot.logit_bias}
        if merged_bias:
            cb["logit_bias"] = [[int(tid), bias] for tid, bias in merged_bias.items()]
        else:
            cb.pop("logit_bias", None)

        return cb

    while True:
        attempt += 1
        mute_thoughts = (attempt > 1) 
        current_messages = messages.copy()
        if confirmed_parts:
            if current_messages and current_messages[-1]["role"] == "assistant":
                current_messages[-1]["content"] = current_messages[-1].get("content", "") + "".join(confirmed_parts)
            else:
                current_messages.append({"role": "assistant", "content": "".join(confirmed_parts)})

        chat_body        = build_chat_body(current_messages)
        rewind_triggered = False

        # ── thinking gate state (per attempt) ─────────────────────────────
        in_think_block      = False
        think_close         = ""
        think_tail          = ""
        prelude             = ""
        content_mode_started = False

        async with client.stream("POST", "/v1/chat/completions", json=chat_body, timeout=300) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                msg = line[5:].strip()
                if msg == "[DONE]":
                    break

                try:
                    data = json.loads(msg)
                except Exception as e:
                    print(f"[STREAM/{slot_id}] JSON parse error: {e}")
                    continue

                choice = data.get("choices", [{}])[0]
                delta  = choice.get("delta", {}) or {}

                reasoning_text = (delta.get("reasoning_content") or "")
                content_text   = (delta.get("content") or "")
                stop           = (choice.get("finish_reason") == "stop")

                slot.absorb_timings(data)

                # ─────────────────────────────────────────────────────────
                # 1) Structured reasoning: never ban / never buffer
                # ─────────────────────────────────────────────────────────
                if reasoning_text:
                    if not mute_thoughts:
                        try:
                            yield chunk_builder.build(slot, "", None, data, reasoning_content=reasoning_text)
                        except TypeError:
                            yield chunk_builder.build(slot, reasoning_text, None, data)
                    continue

                # ignore non-content, non-stop deltas
                if not content_text and not stop:
                    continue

                # ─────────────────────────────────────────────────────────
                # 2) Tag-block thinking fallback (Dynamic Tag Detection)
                # ─────────────────────────────────────────────────────────
                if not content_mode_started or in_think_block:
                    if not in_think_block and not content_mode_started and content_text:
                        prelude = (prelude + content_text)[:128]
                        
                        # This covers Gemma 4's <|channel> as well as <think>, <thinking>, etc.
                        m = re.match(r"^\s*<(\|?)([A-Za-z_][\w-]{0,30})(\|?)>", prelude)
                        if m:
                            leading  = m.group(1)
                            tag      = m.group(2)
                            trailing = m.group(3)
                            think_tail  = ""
                            think_close = f"<{tag}|>" if (leading or trailing) else f"</{tag}>"
                            in_think_block = True
                        elif re.match(r"^\s*<[^>]*$", prelude):
                            continue

                    if in_think_block:
                        close   = think_close
                        combined = think_tail + content_text
                        
                        if close in combined:
                            pos     = combined.find(close)
                            end_pos = pos + len(close)

                            end_in_current = max(0, end_pos - len(think_tail))
                            think_part = content_text[:end_in_current]
                            after_part = content_text[end_in_current:]

                            if think_part and not mute_thoughts:
                                yield chunk_builder.build(slot, think_part, None, data)

                            in_think_block = False
                            think_tail = ""
                            prelude = ""

                            content_text = after_part
                            if not content_text and not stop:
                                continue
                        else:
                            if not mute_thoughts:
                                yield chunk_builder.build(slot, content_text, None, data)
                                
                            keep = max(0, len(close) - 1)
                            think_tail = combined[-keep:] if keep else ""
                            continue

                    # if we did not enter a think block and we have non-whitespace content, we are in normal content mode now
                    if not in_think_block and content_text and not content_text.isspace() and not content_mode_started:
                        content_mode_started = True

                # ─────────────────────────────────────────────────────────
                # 3) Normal content: tokenize -> buffer -> ban/rewind -> flush
                # ─────────────────────────────────────────────────────────
                tok_id = -1
                if content_text:
                    toks = await _tokenize_one(content_text, client)
                    if toks:
                        tok_id = toks[0]
                        if tok_id not in token_id_to_text:
                            token_id_to_text[tok_id] = content_text

                tok_entry = {
                    "tok":  tok_id,
                    "text": content_text,
                    "stop": stop,
                }
                slot.token_buffer.append(tok_entry)
                slot._invalidate_cache()

                # ── check for banned phrase (content only) ─────────────────
                if ban_phrases and n_buffer > 0:
                    found, n_rewind, triggered_phrase = slot.find_ban()
                    if found:
                        culprit_tokens = []
                        current_text = ""
                        for token_info in slot.token_buffer[-n_rewind:]:
                            culprit_tokens.append(token_info)
                            current_text += token_info["text"]
                            if triggered_phrase.lower() in current_text.lower():
                                break

                        token_repr = " + ".join(f"{t['tok']}({t['text']!r})" for t in culprit_tokens)

                        if VERBOSE:
                            print(f"[REWIND] #{slot.rewind_count} phrase={triggered_phrase!r} via: {token_repr}")
                            print(f"[REWIND] active bans: {list(slot.logit_bias.keys())}")

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
                            print(f"[STREAM/{slot_id}] ⚠ Max rewinds reached")

                # ── flush safe prefix to client ───────────────────────────
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

                    # Flush any remaining tokens in buffer
                    if slot.token_buffer:
                        chunks, confirmed_text = slot.flush_tokens(len(slot.token_buffer), chunk_builder, data)
                        for chunk in chunks:
                            yield chunk
                        if confirmed_text:
                            confirmed_parts.append(confirmed_text)

                    # Final empty stop chunk with timings
                    yield chunk_builder.build(slot, "", "stop", data)
                    yield b"data: [DONE]\n\n"
                    return

        if rewind_triggered:
            continue

        # Stream ended without stop token
        print(f"[STREAM/{slot_id}] Stream ended without stop. Flushing {len(slot.token_buffer)} remaining.")
        if slot.token_buffer:
            chunks, confirmed_text = slot.flush_tokens(len(slot.token_buffer), chunk_builder, data if 'data' in locals() else None)
            for chunk in chunks:
                yield chunk
            if confirmed_text:
                confirmed_parts.append(confirmed_text)

        yield b"data: [DONE]\n\n"
        return
    
# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def extract_extra_bias(body: dict) -> dict:
    raw = body.get("logit_bias")
    if not raw:
        return {}
    
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    
    # Assume list of [token_id, bias] pairs
    return {str(pair[0]): float(pair[1]) for pair in raw}


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body       = await request.json()
    messages   = body.get("messages", [])
    extra_bias = extract_extra_bias(body)
    slot_id    = hash(json.dumps(messages)) % 10000

    if body.get("stream", False):
        return StreamingResponse(
            stream_with_ban(messages, body, extra_bias, slot_id),
            media_type="text/event-stream",
        )

    full_text_parts = []
    async for chunk in stream_with_ban(messages, body, extra_bias, slot_id):
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
            except Exception:
                pass

    return Response(
        content=json.dumps({
            "id":                 f"chatcmpl-proxy-{slot_id}",
            "object":             "chat.completion",
            "created":            int(time.time()),
            "model":              model_name,
            "system_fingerprint": system_fingerprint,
            "choices": [{
                "index":         0,
                "message":       {"role": "assistant", "content": "".join(full_text_parts)},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }),
        status_code=200,
        media_type="application/json",
    )

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def passthrough(request: Request, path: str):
    url  = f"{LLAMA_HOST}/{path}"
    body = await request.body()
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.request(
            method=request.method, url=url, content=body,
            headers={k: v for k, v in request.headers.items()
                     if k.lower() != "host"},
        )
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type"))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PROXY_PORT, log_level="warning")
