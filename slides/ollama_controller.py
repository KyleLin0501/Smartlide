import ollama
import re
import os
import logging
import threading
import httpx  # 必須安裝: pip install httpx
from opencc import OpenCC

logger = logging.getLogger("slides.ollama_controller")

# 初始化 OpenCC
converter = OpenCC("s2t")

# ==========================================
# 1. 設定連線資訊 (請修改這裡)
# ==========================================

MAC_IP = "140.134.26.83"

# [A] 遠端 AI Server (用於長摘要生成) - Mac Studio Port 11435
# 優先讀取環境變數 SUMMARY_URL，否則使用預設 IP 直連
SUMMARY_HOST = os.environ.get("SUMMARY_URL", f"http://{MAC_IP}:11435/summarize")

# [B] 遠端 Ollama (用於上一頁/下一頁快速判斷) - Mac Studio Port 11434
# 如果 Django 和 Mac Studio 在不同網域，這裡通常維持用 IP 連線 (需確認 Mac 防火牆有開 11434)
OLLAMA_HOST = os.environ.get("OLLAMA_URL", f"http://{MAC_IP}:11434")
SELECTED_LLM_MODEL = "llama3.1"

# 初始化 Ollama 異步客戶端 (僅用於指令判斷)
try:
    async_client = ollama.AsyncClient(host=OLLAMA_HOST)
    logger.info(f"Ollama AsyncClient initialized connecting to {OLLAMA_HOST}")
except Exception as e:
    logger.error(f"Failed to initialize Ollama AsyncClient: {e}")
    async_client = None

# ==========================================
# 執行緒局部變數
# ==========================================
thread_data = threading.local()


def _get_highlight_buffer():
    if not hasattr(thread_data, 'highlight_buffer'):
        thread_data.highlight_buffer = None
    return thread_data.highlight_buffer


def _set_highlight_buffer(val):
    thread_data.highlight_buffer = val


def clean_mark_text(text: str) -> str:
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


# ==========================================
# 功能 1: 簡報指令判斷 (直接連 Ollama 11434，保持快速)
# ==========================================
async def predict_slide_action(text: str) -> str:
    """
    預測簡報動作指令
    包含：下一頁、上一頁、跳頁、螢光筆(Highlight)、底線(Underline)
    """
    if async_client is None:
        return "none"

    # 1. 基礎清理
    text = converter.convert(text.strip())

    # 2. 【關鍵步驟】使用 Regex 進行「標準化」替換
    # flags=re.IGNORECASE 會忽略大小寫
    # \s* 代表中間可以有 0 到多個空白 (handling "under line", "underline", "High light")

    # 處理 Highlight 系列 (high light, HighLight, highlighted -> highlight)
    text = re.sub(r'high\s*light(ed)?', 'highlight', text, flags=re.IGNORECASE)

    # 處理 Underline 系列 (under line, UnderLine -> underline)
    text = re.sub(r'under\s*line', 'underline', text, flags=re.IGNORECASE)

    # 3. 轉為小寫
    text_lower = text.lower()

    # --- 包夾式邏輯 (支援 Highlight 與 Underline) ---

    # 定義觸發關鍵字 (包含英文標準字與中文)
    triggers = ['highlight', '螢光筆', 'underline', '畫底線', '底線']

    # 製作 Regex Pattern: (highlight|螢光筆|underline|畫底線|底線)
    trigger_pattern = f"({'|'.join(triggers)})"

    # 切分字串，保留分隔符
    parts = re.split(trigger_pattern, text_lower)

    current_buffer = _get_highlight_buffer()

    # 如果切分出超過一段 (代表有關鍵字)，或是目前正在 Buffer 模式中
    if len(parts) > 1 or current_buffer is not None:
        for i, part in enumerate(parts):
            # 檢查 part 是否為關鍵字
            if part in triggers:
                if current_buffer is None:
                    # --- [模式開啟] ---
                    current_buffer = ""
                else:
                    # --- [模式關閉] ---
                    final_text = clean_mark_text(current_buffer)
                    _set_highlight_buffer(None)

                    # 判斷是用什麼關鍵字結束的，決定回傳 U 還是 H
                    if part in ['underline', '畫底線', '底線']:
                        return f"U:{final_text}"
                    else:
                        return f"H:{final_text}"
            else:
                # 累積內容 (非關鍵字的部分)
                if current_buffer is not None:
                    current_buffer += part

        # 更新 Buffer 狀態
        _set_highlight_buffer(current_buffer)

        # 如果還在 buffer 中 (只講了一次關鍵字)，回傳 S 讓文字繼續顯示
        if current_buffer is not None:
            return "S"

    # --- LLM 判斷邏輯 (處理單次指令或其他意圖) ---
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
            model=SELECTED_LLM_MODEL,
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

    # --- 後處理與分派 ---

    # 判斷底線 (LLM 輸出 U 或 關鍵字匹配)
    if clean_output == 'U' or 'underline' in text_lower or any(k in text_lower for k in ['畫底線', '底線', '畫重點']):
        return f"U:{clean_mark_text(text)}"

    # 判斷螢光筆 (LLM 輸出 H 或 關鍵字匹配)
    elif clean_output == 'H' or 'highlight' in text_lower or any(k in text_lower for k in ['螢光筆', '畫螢光筆']):
        return f"H:{clean_mark_text(text)}"

    elif clean_output == 'N':
        return "next"
    elif clean_output == 'P':
        return "prev"
    elif re.match(r'^\d+$', clean_output):
        return f"goto:{clean_output}"

    # 中文數字頁碼判斷 (例如：第二十頁)
    cn_match = re.search(r"(第)?([零一二兩三四五六七八九十]+)頁", text)
    if cn_match:
        n = chinese_to_arabic(cn_match.group(2))
        if n: return f"goto:{n}"

    return "none"

# ==========================================
# 功能 2: 會議摘要 (透過 HTTP 呼叫遠端 AI Server)
# ==========================================
async def generate_meeting_summary(transcript: str, pdf_text: str) -> str:
    """
    將資料打包，發送給 Mac Studio 的 AI Server 進行處理
    """
    logger.info(f"正在請求遠端 AI Server 生成摘要: {SUMMARY_HOST}")
    logger.info(f"資料量 - 逐字稿: {len(transcript)}, PDF: {len(pdf_text)}")

    if not transcript and not pdf_text:
        return "錯誤：沒有內容可以生成摘要。"

    try:
        # 使用 httpx 發送 POST 請求
        # 設定較長的 timeout (例如 180秒)，因為摘要生成需要時間
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                SUMMARY_HOST,
                json={
                    "transcript": transcript,
                    "pdf_text": pdf_text
                }
            )

            # 檢查是否有錯誤狀態碼 (如 404, 500)
            response.raise_for_status()

            # 解析 JSON 回傳
            result = response.json()
            summary = result.get("summary", "")

            if not summary:
                return "錯誤：伺服器回傳了空的摘要。"

            return summary

    except httpx.ConnectError:
        logger.error(f"無法連線至 AI Server: {SUMMARY_HOST}")
        return "錯誤：無法連線至 AI 運算伺服器，請確認伺服器是否已啟動。"

    except httpx.TimeoutException:
        logger.error("AI Server 運算逾時")
        return "錯誤：摘要生成逾時，請稍後再試。"

    except httpx.HTTPStatusError as e:
        logger.error(f"AI Server 回傳錯誤: {e.response.status_code} - {e.response.text}")
        return f"錯誤：伺服器運算失敗 ({e.response.status_code})"

    except Exception as e:
        logger.error(f"摘要生成發生未預期錯誤: {e}")
        return f"發生系統錯誤: {str(e)}"