import ollama
import re
from opencc import OpenCC
import os
import asyncio
import logging
import threading

logger = logging.getLogger("slides.ollama_controller")

# 初始化 OpenCC 與模型
converter = OpenCC("s2t")
selected_model = "llama3.1"

# ==========================================
# 1. 設定遠端 Mac Studio 連線資訊 (修改這裡)
# ==========================================
MAC_IP = "140.134.26.83"  # <--- 請務必改成 Mac Studio 的真實 IP
OLLAMA_HOST = os.environ.get("OLLAMA_URL", "http://140.134.26.83:11434")



# 初始化異步客戶端
try:
    # 指定 host 連線到遠端
    async_client = ollama.AsyncClient(host=OLLAMA_HOST)
    logger.info(f"Ollama AsyncClient initialized connecting to {OLLAMA_HOST}")
except Exception as e:
    logger.error(f"Failed to initialize Ollama AsyncClient: {e}")
    async_client = None

# ==========================================
# 執行緒局部變數 (用於儲存每個連線的螢光筆狀態)
# 在 consumers.py 中，每個連線都有獨立的 command_worker thread
# 所以這裡使用 threading.local 是安全的
# ==========================================
thread_data = threading.local()

def _get_highlight_buffer():
    """安全地獲取當前執行緒的 highlight buffer"""
    if not hasattr(thread_data, 'highlight_buffer'):
        thread_data.highlight_buffer = None
    return thread_data.highlight_buffer

def _set_highlight_buffer(val):
    thread_data.highlight_buffer = val


def clean_mark_text(text: str) -> str:
    """移除指令關鍵詞，保留要標記的內容"""
    text = converter.convert(text.strip())
    keywords = [
        "畫底線", "畫重點", "標記重點", "底線", "畫線", "重點",
        "畫螢光筆", "用螢光筆", "螢光筆", "highlight", "underline", "mark"
    ]
    for kw in keywords:
        if text.lower().startswith(kw):
            return text[len(kw):].strip()
        if text.lower().endswith(kw):
            return text[:-len(kw)].strip()
    return text


def chinese_to_arabic(cn: str):
    """中文數字轉阿拉伯數字"""
    table = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if cn.isdigit(): return int(cn)
    if cn in table: return table[cn]
    if "十" in cn:
        parts = cn.split("十")
        left = table.get(parts[0], 1 if parts[0] == "" else 0)
        right = table.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return left * 10 + right
    return None


async def predict_slide_action(text: str) -> str:
    """
    結合本地端的 Prompt 與邏輯，判斷簡報操作指令
    """
    if async_client is None:
        return "none"

    # 1. 繁簡轉換與預處理
    text = converter.convert(text.strip())
    text_lower = text.lower()
    text_lower = text_lower.replace("under line", "underline").replace("high light", "highlight")
    text_lower = text_lower.replace("highlighted", "highlight")

    # =================================================================
    # 包夾式螢光筆邏輯 (Wrapper Logic)
    # =================================================================
    parts = re.split(r'(highlight|螢光筆)', text_lower, flags=re.IGNORECASE)
    current_buffer = _get_highlight_buffer()

    if len(parts) > 1 or current_buffer is not None:
        for i, part in enumerate(parts):
            is_keyword = (i % 2 == 1)

            if is_keyword:
                # 遇到關鍵字：切換狀態
                if current_buffer is None:
                    current_buffer = "" # 開始錄製
                else:
                    # 停止錄製並送出
                    final_text = clean_mark_text(current_buffer)
                    _set_highlight_buffer(None)
                    return f"H:{final_text}"
            else:
                # 遇到文字：若在錄製中則累積
                if current_buffer is not None:
                    current_buffer += part

        _set_highlight_buffer(current_buffer)

        # 若還在錄製中，回傳 S 等待下一句
        if current_buffer is not None:
            return "S"

    # =================================================================
    # LLM 判斷邏輯
    # =================================================================
    prompt = (
        "你是簡報輔助系統，請根據使用者的語句判斷是否為操作指令。\n"
        "請嚴格遵守以下規則：\n"
        "1. 若語句只是講述內容（朗讀、解釋），請輸出 'S'。\n"
        "2. 若語句提到「下一頁」「往後」「next page」「continue」等，輸出 'N'。\n"
        "3. 若語句提到「上一頁」「回去」「previous page」「go back」等，輸出 'P'。\n"
        "4. 若語句包含「第X頁」「page X」「go to page X」等，輸出數字 X。\n"
        "5. 若語句包含 'underline' 或 中文的「畫底線」「底線」「畫重點」「標記重點」 → 輸出 'U'。\n"
        "6. 若語句包含 'highlight' 或 中文的「畫螢光筆」「螢光筆」 → 輸出 'H'。\n"
        "7. 若模糊或無法確定，輸出 'S'。\n"
        "輸出只能是以下其中之一：'N'、'P'、'S'、'U'、'H' 或 數字。禁止輸出其他內容。"
    )

    try:
        response = await async_client.chat(
            model=selected_model,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': text}
            ]
        )
        output = response.get('message', {}).get('content', '').strip()
        clean_output = re.sub(r'[^a-zA-Z0-9]', '', output).upper()

    except Exception as e:
        logger.error(f"Ollama error: {e}")
        clean_output = "S"

    # 後處理
    if clean_output == 'U' or 'underline' in text_lower or any(k in text_lower for k in ['畫底線', '底線', '畫重點']):
        return f"U:{clean_mark_text(text)}"
    elif clean_output == 'H' or 'highlight' in text_lower or any(k in text_lower for k in ['螢光筆', '畫螢光筆']):
        return f"H:{clean_mark_text(text)}"
    elif clean_output == 'N':
        return "next"
    elif clean_output == 'P':
        return "prev"
    elif re.match(r'^\d+$', clean_output):
        return f"goto:{clean_output}"

    cn_match = re.search(r"(第)?([零一二兩三四五六七八九十]+)頁", text)
    if cn_match:
        n = chinese_to_arabic(cn_match.group(2))
        if n: return f"goto:{n}"

    return "none"


# --- 摘要生成功能 ---
async def generate_meeting_summary(transcript: str, pdf_text: str) -> str:
    if async_client is None:
        return "錯誤：LLM 用戶端未初始化。"

    logger.info(f"Generating summary. Transcript len: {len(transcript)}, PDF len: {len(pdf_text)}")

    prompt = (
        "你是一位專業的會議記錄員。請根據「PDF 簡報內容」與講者的「語音逐字稿」，"
        "整理出一份 Markdown 格式的會議/簡報摘要。\n\n"
        "--- [PDF 內容] ---\n"
        f"{pdf_text[:10000]}"
        "\n--- [語音逐字稿] ---\n"
        f"{transcript[:10000]}"
        "\n請輸出摘要："
    )

    try:
        response = await async_client.chat(
            model=selected_model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response.get('message', {}).get('content', '')
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return f"摘要生成失敗: {e}"