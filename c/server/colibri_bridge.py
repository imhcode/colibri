"""
Manages the colibrì C engine subprocess in SERVE mode.

Protocol:
  - Engine reads prompts from stdin (one line per turn)
  - Engine writes generated text to stdout, delimited by sentinel bytes
  - READY sentinel: b'\x01\x01READY\x01\x01\n'
  - END sentinel: b'\x01\x01END\x01\x01\n' followed by 'STAT <tok> <tps> <hit> <rss>\n'
  - RESET command: b'\x02RESET\n'
  - MORE command: b'\x02MORE\n'
"""
import os
import sys
import subprocess
import threading
import re
import time

READY = b"\x01\x01READY\x01\x01\n"
END   = b"\x01\x01END\x01\x01\n"

_STAT_RE = re.compile(rb"STAT (\S+) (\S+) (\S+) (\S+)")


class ColibriEngine:
    def __init__(self, model_path, glm_binary, env_overrides=None):
        self.model_path = model_path
        self.glm_binary = glm_binary
        self.env_overrides = env_overrides or {}
        self.proc = None
        self._lock = threading.Lock()
        self._stderr_thread = None

    # ---------- subprocess lifecycle ----------

    def start(self):
        """Spawn the glm subprocess and wait for READY sentinel."""
        env = dict(os.environ, SNAP=self.model_path, SERVE="1")
        env.update(self.env_overrides)
        self.proc = subprocess.Popen(
            [self.glm_binary, "8"],  # cap=8 slots (same default as coli)
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._wait_for(READY, timeout=600)  # model load can take 30s+ on big models
        self._read_stat_line()  # consume the STAT line after READY

    def _drain_stderr(self):
        """Pass stderr through to our stderr (engine logs)."""
        try:
            for line in iter(self.proc.stderr.readline, b""):
                sys.stderr.buffer.write(line)
                sys.stderr.buffer.flush()
        except Exception:
            pass

    def _wait_for(self, sentinel, timeout=120):
        """Read from stdout byte-by-byte until we see the sentinel bytes."""
        buf = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            b = self.proc.stdout.read(1)
            if b == b"":
                raise RuntimeError("Engine process exited while waiting for sentinel")
            buf += b
            if buf.endswith(sentinel):
                return
        raise TimeoutError(f"Timeout waiting for {sentinel!r}")

    def _read_stat_line(self):
        """Read and parse the STAT line that follows READY/END sentinels."""
        line = self.proc.stdout.readline()
        return line.decode("utf-8", "replace").strip()

    def stop(self):
        """Terminate the subprocess cleanly."""
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

    # ---------- engine commands ----------

    def reset(self):
        """Clear the KV cache (start fresh conversation)."""
        with self._lock:
            self.proc.stdin.write(b"\x02RESET\n")
            self.proc.stdin.flush()
            self._wait_for(END)
            self._read_stat_line()

    def generate(self, prompt, on_chunk=None):
        """
        Send a prompt to the engine and stream back the response.

        Args:
            prompt: plain text string (the engine applies chat template internally,
                    unless CHAT_TEMPLATE=0 is set in env)
            on_chunk: callback(text:str) called for each text chunk as it arrives

        Returns:
            dict: {"tok": int, "tps": float, "hit": float, "rss": float}
        """
        with self._lock:
            data = (prompt.replace("\n", " ") + "\n").encode("utf-8")
            self.proc.stdin.write(data)
            self.proc.stdin.flush()

            # Read response until END sentinel, streaming chunks
            pend = b""
            stats = {"tok": 0, "tps": 0.0, "hit": 0.0, "rss": 0.0}
            while True:
                b = self.proc.stdout.read(1)
                if b == b"":
                    raise RuntimeError("Engine process exited during generation")
                pend += b
                if pend.endswith(END):
                    # Flush any remaining text before the sentinel
                    rest = pend[:-len(END)]
                    if rest and on_chunk:
                        on_chunk(rest.decode("utf-8", "replace"))
                    # Read and parse STAT line
                    stat_line = self.proc.stdout.readline()
                    m = _STAT_RE.search(stat_line)
                    if m:
                        stats = {
                            "tok": int(m.group(1)),
                            "tps": float(m.group(2)),
                            "hit": float(m.group(3)),
                            "rss": float(m.group(4)),
                        }
                    break
                # Stream chunks: keep a sliding window of sentinel length
                if len(pend) > len(END):
                    out = pend[:-len(END)]
                    pend = pend[-len(END):]
                    if out and on_chunk:
                        on_chunk(out.decode("utf-8", "replace"))

            return stats
