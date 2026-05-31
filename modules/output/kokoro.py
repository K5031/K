import threading
import queue
import sounddevice as sd
from kokoro import KPipeline
from queue import Empty
from interfaces import OutputInterface


class Module(OutputInterface):
    def __init__(self, voice="af_heart", lang_code="a"):
        self.stream = sd.OutputStream(
            samplerate=24000,
            channels=1,
            dtype="float32",
        )
        self.stream.start()
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.buffer = ""
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(
            target=self._worker,
            daemon=True,
        )
        self.thread.start()

    def _worker(self):
        while self.running:
            try:
                text = self.queue.get(timeout=0.2)
            except Empty:
                continue
            for _, _, audio in self.pipeline(text, voice=self.voice):
                if not self.running:
                    break
                self.stream.write(audio.reshape(-1, 1))

    def send(self, token: str):
        self.buffer += token
        if self.buffer.endswith((".", "!", "?", ",")):
            self.queue.put(self.buffer)
            self.buffer = ""

    def interrupt(self):
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        self.stream.stop()
        self.stream.close()