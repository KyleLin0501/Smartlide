import base64
import json
import asyncio
import threading
import logging
import time
import numpy as np

from channels.generic.websocket import AsyncWebsocketConsumer
from queue import Queue

# 相對導入
from .whisper_processor import process_audio, start_command_worker_thread
from slides.ollama_controller import predict_slide_action, generate_meeting_summary

SILENCE_THRESHOLD = 0.005
logger = logging.getLogger("slides.consumers")


class SpeechConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_queue = Queue()
        self.text_queue = Queue(maxsize=5)
        self.audio_thread = None
        self.command_thread = None
        self.stop_event = None
        self.loop = None

        # 儲存整場會議的逐字稿
        self.transcript = []
        self.pdf_id = None

    async def connect(self):
        # 獲取 pdf_id
        self.pdf_id = self.scope['url_route']['kwargs'].get('pdf_id')
        await self.accept()
        logger.info(f"[{self.channel_name}] WebSocket connected. PDF ID: {self.pdf_id}")

        if self.audio_thread and self.audio_thread.is_alive():
            return

        self.stop_event = threading.Event()
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.get_event_loop()

        # 1. 啟動 Command Worker
        self.command_thread = threading.Thread(
            target=start_command_worker_thread,
            name=f"command-worker-{id(self)}",
            args=(self.loop, predict_slide_action, self.stop_event, self.text_queue, self),
            daemon=True
        )
        self.command_thread.start()

        # 2. 啟動 Whisper Audio Processor
        def thread_target_wrapper(loop, stop_ev, aqueue, tqueue):
            try:
                process_audio(self, loop, stop_ev, aqueue, tqueue)
            except Exception:
                logger.exception(f"[{self.channel_name}] Audio processing thread crashed.")

        self.audio_thread = threading.Thread(
            target=thread_target_wrapper,
            name=f"audio-whisper-{id(self)}",
            args=(self.loop, self.stop_event, self.audio_queue, self.text_queue),
            daemon=True
        )
        self.audio_thread.start()

    async def disconnect(self, close_code):
        if self.stop_event:
            self.stop_event.set()
        self._clear_queue(self.audio_queue)
        self._clear_queue(self.text_queue)
        self.audio_thread = None
        self.command_thread = None
        logger.info(f"[{self.channel_name}] Disconnected.")

    def _clear_queue(self, q):
        try:
            while not q.empty():
                q.get_nowait()
                if hasattr(q, 'task_done'): q.task_done()
        except Exception:
            pass

    def is_speech(self, pcm_bytes: bytes) -> bool:
        try:
            audio_float = np.frombuffer(pcm_bytes, dtype=np.float32)
            if audio_float.size == 0: return False
            rms = np.sqrt(np.mean(audio_float ** 2))
            return rms > SILENCE_THRESHOLD
        except Exception:
            return True

    async def receive(self, text_data=None, bytes_data=None):
        if self.stop_event and self.stop_event.is_set(): return
        if not text_data: return

        try:
            data = json.loads(text_data)

            # --- 關鍵修改：接收前端傳來的 PDF 內容 ---
            if data.get("command") == "end_presentation":
                # 從前端獲取 PDF 文字
                pdf_text_from_client = data.get("pdf_content", "")
                logger.info(f"Received end_presentation. PDF text length: {len(pdf_text_from_client)}")

                await self.handle_summary_generation(pdf_text_from_client)
                return
            # ---------------------------------------

            b64 = data.get("audio_data")
            if not b64: return

            audio_bytes_f32 = base64.b64decode(b64)
            if self.is_speech(audio_bytes_f32):
                audio_data_np = np.frombuffer(audio_bytes_f32, dtype=np.float32).copy()
                try:
                    self.audio_queue.put_nowait(audio_data_np)
                except:
                    pass

        except json.JSONDecodeError:
            logger.error("JSON decode error.")
        except Exception as e:
            logger.error(f"Error in receive: {e}")

    async def send_json(self, text, cmd):
        if text and text.strip():
            self.transcript.append(text.strip())
        try:
            await self.send(text_data=json.dumps({
                "text": text if text else "",
                "command": cmd
            }, ensure_ascii=False))
        except Exception:
            pass

    async def handle_summary_generation(self, pdf_text):
        full_transcript_text = "\n".join(self.transcript)

        if not full_transcript_text and not pdf_text:
            await self.send(text_data=json.dumps({
                "type": "info",
                "message": "沒有足夠的內容可供分析。"
            }, ensure_ascii=False))
            return

        # 生成摘要
        summary = await generate_meeting_summary(full_transcript_text, pdf_text)

        # 回傳結果
        await self.send(text_data=json.dumps({
            "type": "summary",
            "content": summary
        }, ensure_ascii=False))