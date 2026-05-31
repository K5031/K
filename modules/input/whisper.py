from datetime import datetime, timedelta, timezone
from queue import Queue, Empty
from time import sleep
import threading
import numpy as np
import speech_recognition as sr
import torch
import whisper
from interfaces import InputInterface


class Module(InputInterface):
    def __init__(
        self,
        model="medium",
        energy_threshold=1000,
        record_timeout=2.0,
        phrase_timeout=3.0,
    ):
        self.energy_threshold = energy_threshold
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout

        self.source = sr.Microphone(sample_rate=16000)
        self.recorder = self._setup_recorder()
        self.model = whisper.load_model(model)

        self.data_queue = Queue()
        self.output_queue = Queue()
        self.buffer = b""
        self.last_audio_time = None
        self.stop_listening = None

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _setup_recorder(self):
        recorder = sr.Recognizer()
        recorder.energy_threshold = self.energy_threshold
        recorder.dynamic_energy_threshold = False
        with self.source:
            recorder.adjust_for_ambient_noise(self.source)
        return recorder

    def _record_callback(self, _, audio: sr.AudioData):
        self.data_queue.put(audio.get_raw_data())

    def _drain_audio_queue(self) -> bytes:
        chunks = []
        while True:
            try:
                chunks.append(self.data_queue.get_nowait())
            except Empty:
                break
        return b"".join(chunks)

    def _audio_to_numpy(self, audio_bytes):
        return (
            np.frombuffer(audio_bytes, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )

    def _transcribe(self, audio_bytes):
        audio_np = self._audio_to_numpy(audio_bytes)
        result = self.model.transcribe(
            audio_np,
            language="en",
            task="transcribe",
            fp16=torch.cuda.is_available(),
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
        audio_data = self._drain_audio_queue()
        if audio_data:
            self.buffer += audio_data
            self.last_audio_time = now
        if (
            self.buffer
            and self.last_audio_time
            and now - self.last_audio_time > timedelta(seconds=self.phrase_timeout)
        ):
            text = self._transcribe(self.buffer)
            if text:
                self.output_queue.put(text)
            self.buffer = b""
            self.last_audio_time = None

    def _run_loop(self):
        self.stop_listening = self.recorder.listen_in_background(
            self.source,
            self._record_callback,
            phrase_time_limit=self.record_timeout,
        )
        while self._running:
            self._process_audio()
            sleep(0.05)

    def stop(self):
        self._running = False
        if self.stop_listening:
            self.stop_listening(wait_for_stop=False)
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