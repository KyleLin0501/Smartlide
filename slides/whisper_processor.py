import os
import queue
import asyncio
import numpy as np
import time
import logging
import threading
import inspect
import requests  # 新增：用於發送 HTTP 請求
from concurrent.futures import ThreadPoolExecutor, Future
from opencc import OpenCC

# 設定 Log
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')

logger = logging.getLogger("slides.whisper_processor")

# ==========================================
# 遠端連線設定 (請修改這裡)
# ==========================================
MAC_IP = "140.134.26.83"  # <--- 請改成 Mac Studio 的 IP
WHISPER_API_URL = f"http://{MAC_IP}:9001/transcribe"

# 用於 LLM 判斷的執行緒池 (保留)
_predict_executor = ThreadPoolExecutor(max_workers=4)

# 初始化繁簡轉換 (保留本地清洗功能)
converter = OpenCC("s2t")

NOISE_WORDS = {
    "thankyou", "thank you", "嗯", "啊", "喔", "呃", "字幕by索蘭婭".lower(),
    "優優獨播劇場", "yoyo television series exclusive", "exclusive", "優優", "yoyo",
    "字幕製作人zither harp", "zither harp", "字幕", "製作人", "harp",
    "請說中文", "請開始說話", "開始說話", "請開始", "請講中文"
}


def clean_text(text: str) -> str:
    """
    保留本地的文字清洗邏輯，方便調整過濾規則
    """
    try:
        text = converter.convert(text.strip())
    except Exception:
        text = text.strip()
    if len(text.strip()) == 0: return ""
    text_lower = text.lower()

    # 過濾重複無意義字元
    if len(text) > 2:
        chars = [c for c in text if c.strip()]
        if len(chars) > 0:
            most_common_char = max(set(chars), key=chars.count)
            ratio = chars.count(most_common_char) / len(chars)
            if ratio > 0.70 and most_common_char in ['鳥', '的', '啦', '啊', '嗯', '呵', '哈', '哦', '了', '吧', '寰',
                                                     '室', '索', '蘭', '婭']:
                return ""

    # 過濾關鍵字
    if text_lower in NOISE_WORDS: return ""
    noise_keywords = ["優優", "yoyo", "exclusive", "zither", "harp", "劇場", "television series"]
    for keyword in noise_keywords:
        if keyword in text_lower:
            if len(text) < 50: return ""

    return text


# =========================================================
# Command Worker (負責將文字送給 Ollama / 前端)
# 這部分保持不變，因為它負責處理邏輯
# =========================================================

def _send_command_async_callback(future: Future, consumer, thread_name: str, consumer_id: str,
                                 loop: asyncio.AbstractEventLoop):
    try:
        cmd = future.result(timeout=5)
    except Exception:
        cmd = "none"
    try:
        coro = consumer.send_json("", cmd)
        target_loop = loop or asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(coro, target_loop)
    except Exception:
        pass


def start_command_worker_thread(loop: asyncio.AbstractEventLoop, predict_cmd_func, stop_event: threading.Event,
                                text_queue: queue.Queue, consumer):
    thread_name = threading.current_thread().name
    consumer_id = getattr(consumer, "channel_name", "<unknown>")
    logger.info("[%s][%s] command_worker_thread started.", thread_name, consumer_id)

    while not stop_event.is_set():
        try:
            texts_to_process = []
            try:
                while True:
                    texts_to_process.append(text_queue.get_nowait())
            except queue.Empty:
                pass

            if not texts_to_process:
                time.sleep(0.2)
                continue

            # A: 發送字幕給前端
            for text in texts_to_process:
                try:
                    target_loop = loop or asyncio.get_event_loop()
                    coro = consumer.send_json(text, "none")
                    asyncio.run_coroutine_threadsafe(coro, target_loop)
                except Exception:
                    pass

            # B: LLM 指令判斷 (呼叫 ollama_controller)
            current_context = " ".join(texts_to_process)
            try:
                if inspect.iscoroutinefunction(predict_cmd_func):
                    target_loop = loop or asyncio.get_event_loop()
                    predict_future = asyncio.run_coroutine_threadsafe(predict_cmd_func(current_context), target_loop)
                else:
                    predict_future = _predict_executor.submit(predict_cmd_func, current_context)

                predict_future.add_done_callback(
                    lambda f: _send_command_async_callback(f, consumer, thread_name, consumer_id, loop)
                )
            except Exception:
                pass
        except Exception:
            time.sleep(1.0)


# =========================================================
# Audio Processor (核心修改處)
# 不再跑模型，而是發送 HTTP Request 到 Mac Studio
# =========================================================

def process_audio(consumer, loop: asyncio.AbstractEventLoop, stop_event: threading.Event, audio_queue: queue.Queue,
                  text_queue: queue.Queue):
    logger.info(f"Starting Remote Whisper Processor connecting to {WHISPER_API_URL}")

    while not stop_event.is_set():
        try:
            # 1. 從 Queue 拿音訊 (設定 timeout 避免卡死)
            chunk = audio_queue.get(timeout=0.2)

            if chunk is None or chunk.size == 0:
                continue

            # 2. 發送給遠端 Mac Studio
            try:
                # 將 numpy array 轉為 bytes 傳輸
                audio_bytes = chunk.tobytes()

                # 發送 POST 請求 (設定 timeout)
                response = requests.post(
                    WHISPER_API_URL,
                    files={"file": audio_bytes},
                    timeout=5
                )

                if response.status_code == 200:
                    result = response.json()
                    raw_text = result.get("text", "").strip()

                    # 3. 本地清洗文字
                    cleaned_text = clean_text(raw_text)

                    if cleaned_text:
                        # logger.info(f"Remote Whisper: {cleaned_text}")
                        # 4. 放入 Queue 給 Command Worker 處理
                        try:
                            text_queue.put_nowait(cleaned_text)
                        except queue.Full:
                            # 佇列滿了嘗試清出空間
                            try:
                                text_queue.get_nowait()
                            except:
                                pass
                            try:
                                text_queue.put_nowait(cleaned_text)
                            except:
                                pass
                else:
                    logger.warning(f"Remote Whisper Error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                logger.error(f"Cannot connect to Mac Studio at {WHISPER_API_URL}. Is the server running?")
                time.sleep(2)
            except Exception as e:
                logger.error(f"Request Failed: {e}")

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Process Audio Error: {e}")
            time.sleep(1)

    # Cleanup
    try:
        while not audio_queue.empty(): audio_queue.get_nowait()
    except:
        pass