from datetime import datetime, timedelta, timezone
from queue import Queue, Empty
from time import sleep
import subprocess
import threading
import numpy as np
import sounddevice as sd
import whisper
from interfaces import InputInterface


def _cuda_available():
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


class Module(InputInterface):
    def __init__(
        self,
        model="medium",
        energy_threshold=0.01,
        record_timeout=2.0,
        phrase_timeout=3.0,
        sample_rate=16000,
    ):
        self.energy_threshold = energy_threshold
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout
        self.sample_rate = sample_rate
        self.model = whisper.load_model(model)
        self.data_queue = Queue()
        self.output_queue = Queue()
        self.buffer = np.array([], dtype=np.float32)
        self.last_audio_time = None
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._stream = None
        self._thread.start()
        
    def _audio_callback(self, indata, frames, time, status):
        chunk = indata[:, 0].copy()
        if np.abs(chunk).mean() > self.energy_threshold:
            self.data_queue.put(chunk)
    def _transcribe(self, audio: np.ndarray):
        result = self.model.transcribe(
            audio,
            language="en",
            task="transcribe",
            fp16=_cuda_available(),
            temperature=0.0,
            condition_on_previous_text=False,
        )
        segments = result.get("segments", [])
        text_parts = [
            seg["text"]
            for seg in segments
            if seg["avg_logprob"] > -1.0
        ]
        return " ".join(text_parts).strip()
    def _get_combined_text(self):
        texts = []
        while True:
            try:
                texts.append(self.output_queue.get_nowait())
            except Empty:
                break
        return " ".join(texts) if texts else None
    def _process_audio(self):
        now = datetime.now(timezone.utc)
        chunks = []
        while True:
            try:
                chunks.append(self.data_queue.get_nowait())
            except Empty:
                break
        if chunks:
            self.buffer = np.concatenate([self.buffer, *chunks])
            self.last_audio_time = now
        if (
            self.buffer.size > 0
            and self.last_audio_time
            and now - self.last_audio_time > timedelta(seconds=self.phrase_timeout)
        ):
            text = self._transcribe(self.buffer)
            if text:
                self.output_queue.put(text)
            self.buffer = np.array([], dtype=np.float32)
            self.last_audio_time = None
    def _run_loop(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        with self._stream:
            while self._running:
                self._process_audio()
                sleep(0.05)
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
    def get_input(self):
        while True:
            text = self._get_combined_text()
            if text:
                return text
            sleep(0.05)
    
    def has_input(self) -> bool:
        return not self.output_queue.empty()