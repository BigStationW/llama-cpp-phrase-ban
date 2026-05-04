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
    if not ban_phrases:
        return 0
    
    async with httpx.AsyncClient() as c:
        tasks = [_tokenize_one(p, c) for p in ban_phrases]
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
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{LLAMA_HOST}/props", timeout=5)
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
    confirmed_parts = []
    attempt = 0

    # Extract user's logit_bias if any
    user_bias = {}
    if "logit_bias" in body:
        raw = body["logit_bias"]
        if isinstance(raw, dict):
            user_bias = {str(k): float(v) for k, v in raw.items()}
        elif isinstance(raw, list):
            user_bias = {str(pair[0]): float(pair[1]) for pair in raw}

    async def apply_template(msgs: list, template_kwargs: dict = None) -> str:
        try:
            body = {"messages": msgs, "add_generation_prompt": True}
            if template_kwargs:
                body["chat_template_kwargs"] = template_kwargs
            
            r = await client.post("/apply-template", json=body, timeout=10)
            r.raise_for_status()
            return r.json().get("prompt", "")
        except Exception as e:
            print(f"[TEMPLATE] /apply-template failed: {e}")
            return "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in msgs)

    # Extract chat_template_kwargs from client request
    user_template_kwargs = body.get("chat_template_kwargs", {})

    while True:
        attempt += 1
        mute_thoughts = (attempt > 1)

        # 1. Update kwargs (e.g., turn off thinking on rewinds)
        template_kwargs = user_template_kwargs.copy()
        if attempt > 1:
            template_kwargs["enable_thinking"] = False

        # 2. Apply template to the ORIGINAL messages only
        prompt = await apply_template(messages, template_kwargs)

        # 3. Safely append the partial generation so far (prevents Jinja EOS corruption)
        if confirmed_parts:
            prompt += "".join(confirmed_parts)

        # Build completion body
        completion_body = {k: v for k, v in body.items() 
                          if k not in ("messages", "model", "stream", "logit_bias", "tools", "tool_choice", "chat_template_kwargs")}
        completion_body["prompt"] = prompt
        completion_body["stream"] = True
        completion_body["cache_prompt"] = True
        
        # Handle max_tokens → n_predict
        if "max_tokens" in completion_body:
            completion_body["n_predict"] = completion_body.pop("max_tokens")

        # Merge logit bias
        merged_bias = {**user_bias, **slot.logit_bias}
        if merged_bias:
            completion_body["logit_bias"] = [[int(tid), bias] for tid, bias in merged_bias.items()]

        rewind_triggered = False

        # Thinking block state
        in_think_block = False
        think_close = ""
        content_mode_started = False
        consuming_think_header = False

        async with client.stream("POST", "/completion", json=completion_body, timeout=300) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                msg = line[5:].strip()
                if msg == "[DONE]":
                    break

                try:
                    data = json.loads(msg)
                except:
                    continue

                content_text = data.get("content", "")
                token_id = data["tokens"][0] if data.get("tokens") else -1
                stop = data.get("stop", False)

                slot.absorb_timings(data)

                if token_id != -1 and content_text and token_id not in token_id_to_text:
                    token_id_to_text[token_id] = content_text

                # ─────────────────────────────────────────────────────────
                # Thinking block detection
                # ─────────────────────────────────────────────────────────
                if not content_mode_started or in_think_block:
                    if not in_think_block and content_text:
                        m = re.match(r"^\s*<(\|?)([A-Za-z_][\w-]{0,30})(\|?)>$", content_text.strip())
                        if m:
                            leading, tag, trailing = m.group(1), m.group(2), m.group(3)
                            think_close = f"<{tag}|>" if (leading or trailing) else f"</{tag}>"
                            in_think_block = True
                            consuming_think_header = True
                            continue

                    if in_think_block:
                        if consuming_think_header:
                            if "\n" in content_text:
                                consuming_think_header = False
                            continue

                        # Detect closing tag
                        if content_text.strip() == think_close:
                            in_think_block = False
                            content_mode_started = True
                        else:
                            if not mute_thoughts:
                                try:
                                    yield chunk_builder.build(slot, "", None, data, reasoning_content=content_text)
                                except TypeError:
                                    yield chunk_builder.build(slot, content_text, None, data)
                        continue
                    
                    if content_text and not content_text.isspace():
                        content_mode_started = True

                # ─────────────────────────────────────────────────────────
                # Buffer token with real ID
                # ─────────────────────────────────────────────────────────
                if content_text:
                    slot.token_buffer.append({
                        "tok": token_id,
                        "text": content_text,
                        "stop": False,
                    })

                if stop:
                    slot.token_buffer.append({
                        "tok": -1,
                        "text": "",
                        "stop": True,
                    })

                slot._invalidate_cache()

                # ─────────────────────────────────────────────────────────
                # Ban check
                # ─────────────────────────────────────────────────────────
                if ban_phrases and n_buffer > 0:
                    found, n_rewind, triggered_phrase = slot.find_ban()
                    if found:
                        culprit_token = slot.token_buffer[-n_rewind]
                        token_repr = f"{culprit_token['tok']}({culprit_token['text']!r})"

                        if VERBOSE:
                            print(Fore.YELLOW + f"[REWIND] #{slot.rewind_count} phrase={triggered_phrase!r} via: {token_repr}" + Style.RESET_ALL)
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

                # ─────────────────────────────────────────────────────────
                # Flush safe prefix
                # ─────────────────────────────────────────────────────────
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
