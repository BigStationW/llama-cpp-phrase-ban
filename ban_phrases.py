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
        for p in content.split(','):
            p = p.strip()
            if p.startswith('"') and p.endswith('"'):
                p = p[1:-1]
            elif p.startswith("'") and p.endswith("'"):
                p = p[1:-1]
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
    print(f"\n[PROXY] Running on http://127.0.0.1:{PROXY_PORT}")
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
phrase_token_variants: dict[str, list[int]] = {}
phrase_automaton: pyahocorasick.Automaton | None = None
model_name:         str = "unknown"
system_fingerprint: str = "unknown"


async def _tokenize_one(text: str, client: httpx.AsyncClient) -> list[int]:
    try:
        r = await client.post(
            f"{LLAMA_HOST}/tokenize",
            json={"content": text, "add_special": False},
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
        # Batch all tokenization requests at once
        tasks = []
        for p in ban_phrases:
            tasks.append(_tokenize_one(" " + p, c))
            tasks.append(_tokenize_one(p, c))
        
        results = await asyncio.gather(*tasks)
    
    # Process results in pairs
    max_len = 0
    for i, phrase in enumerate(ban_phrases):
        spaced = results[i * 2]
        unspaced = results[i * 2 + 1]
        
        first_toks: list[int] = []
        seen: set[int] = set()
        for toks in (spaced, unspaced):
            if toks and toks[0] not in seen:
                first_toks.append(toks[0])
                seen.add(toks[0])
        
        phrase_token_variants[phrase] = first_toks
        longest = max(len(spaced), len(unspaced))
        max_len = max(max_len, longest)
    
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
              raw_data: dict | None = None) -> bytes:
        chunk = self.base_chunk.copy()
        chunk["created"] = int(time.time())
        chunk["choices"] = [{
            "index": 0,
            "delta": {"content": content} if content else {},
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
        
        # Find first match using automaton
        for end_pos, phrase in phrase_automaton.iter(text):
            pos = end_pos - len(phrase) + 1
            
            # Calculate how many tokens to rewind
            char_count = 0
            n_rewind = 0
            for tok in reversed(self.token_buffer):
                char_count += len(tok["text"])
                n_rewind += 1
                if char_count >= len(text) - pos:
                    break
            
            if VERBOSE:
                print(f"[BAN/{self.slot_id}] Phrase detected: {phrase!r}  buffer: {text!r}  rewind: {n_rewind}")
            
            return True, n_rewind, phrase
        
        return False, 0, ""

    def apply_rewind_bias(self, n_rewind: int, phrase: str = ""):
        seen: set[str] = set()
        if n_rewind > 0:
            trigger_tok = self.token_buffer[-1]
            tid = str(trigger_tok["tok"])
            if tid not in seen:
                self.logit_bias[tid] = BAN_BIAS
                seen.add(tid)
        if phrase and phrase in phrase_token_variants:
            for extra_tid in phrase_token_variants[phrase]:
                tid_str = str(extra_tid)
                if tid_str not in seen:
                    self.logit_bias[tid_str] = BAN_BIAS
                    seen.add(tid_str)
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
    prompt:     str,
    body:       dict,
    extra_bias: dict,
    slot_id:    int,
):
    slot          = SlotState(slot_id)
    chunk_builder = ChunkBuilder(slot_id)
    confirmed_parts = []  # Use list instead of string concatenation
    attempt       = 0

    def build_completion_body(current_prompt: str) -> dict:
        cb = body.copy()
        cb["prompt"]       = current_prompt
        cb["stream"]       = True
        cb["cache_prompt"] = True
        merged_bias = {**extra_bias, **slot.logit_bias}
        if merged_bias:
            cb["logit_bias"] = [[int(tid), bias] for tid, bias in merged_bias.items()]
        return cb

    while True:
        attempt += 1
        current_prompt   = prompt + "".join(confirmed_parts)
        completion_body  = build_completion_body(current_prompt)
        rewind_triggered = False

        tokens_this_attempt  = 0

        async with client.stream("POST", "/completion", json=completion_body, timeout=300) as resp:
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

                text    = data.get("content", "")
                tok_ids = data.get("tokens", [])
                stop    = data.get("stop", False)

                slot.absorb_timings(data)

                if not text and not stop:
                    continue

                tok_entry = {
                    "tok":  tok_ids[0] if tok_ids else -1,
                    "text": text,
                    "stop": stop,
                }
                slot.token_buffer.append(tok_entry)
                slot._invalidate_cache()
                tokens_this_attempt += 1

                if VERBOSE:
                    print(f"[TOK/{slot_id}] #{tokens_this_attempt:04d}  "
                          f"id={tok_entry['tok']:6d}  "
                          f"buf={len(slot.token_buffer):3d}  "
                          f"text={text!r}")

                # ── check for banned phrase ───────────────────────────
                if ban_phrases and n_buffer > 0:
                    found, n_rewind, triggered_phrase = slot.find_ban()
                    if found:
                        if slot.rewind_count < MAX_REWINDS:
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

                # ── flush safe prefix to client ───────────────────────
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

                if stop:
                    slot.commit_attempt()
                    # Flush any remaining tokens in buffer
                    if slot.token_buffer:
                        chunks, confirmed_text = slot.flush_tokens(
                            len(slot.token_buffer), 
                            chunk_builder, 
                            data
                        )
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
            chunks, confirmed_text = slot.flush_tokens(
                len(slot.token_buffer), 
                chunk_builder, 
                data
            )
            for chunk in chunks:
                yield chunk
            if confirmed_text:
                confirmed_parts.append(confirmed_text)
        
        yield b"data: [DONE]\n\n"
        return


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

async def render_prompt(body: dict) -> str:
    tmpl_payload = {
        "messages":              body["messages"],
        "add_generation_prompt": True,
        "chat_template_kwargs": {
            "enable_thinking": body.get("chat_template_kwargs", {})
                                   .get("enable_thinking", False)
        },
    }
    r = await client.post("/apply-template", json=tmpl_payload, timeout=10)
    return r.json()["prompt"]

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
    prompt     = await render_prompt(body)
    extra_bias = extract_extra_bias(body)
    slot_id    = hash(json.dumps(body.get("messages", []))) % 10000

    if body.get("stream", False):
        return StreamingResponse(
            stream_with_ban(prompt, body, extra_bias, slot_id),
            media_type="text/event-stream",
        )

    full_text_parts = []
    async for chunk in stream_with_ban(prompt, body, extra_bias, slot_id):
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