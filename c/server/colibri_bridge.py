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


class ColibriEngine:
    def __init__(self, model_path, glm_binary, env_overrides=None):
        self.model_path = model_path
        self.glm_binary = glm_binary
        self.env_overrides = env_overrides or {}
        self.proc = None
        self._lock = threading.Lock()

    def start(self):
        """Spawn the glm subprocess and wait for READY."""
        pass  # implemented in later tasks

    def reset(self):
        """Send RESET command to clear KV cache."""
        pass  # implemented in later tasks

    def generate(self, prompt, on_chunk=None):
        """
        Send prompt, stream response via on_chunk callback.
        Returns dict with stats: {tok, tps, hit, rss}.
        """
        pass  # implemented in later tasks

    def stop(self):
        """Terminate the subprocess."""
        pass  # implemented in later tasks
