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
    if async_client is None:
        return "none"

    text = converter.convert(text.strip())
    text_lower = text.lower()
    text_lower = text_lower.replace("under line", "underline").replace("high light", "highlight")
    text_lower = text_lower.replace("highlighted", "highlight")

    # 包夾式螢光筆邏輯
    parts = re.split(r'(highlight|螢光筆)', text_lower, flags=re.IGNORECASE)
    current_buffer = _get_highlight_buffer()

    if len(parts) > 1 or current_buffer is not None:
        for i, part in enumerate(parts):
            is_keyword = (i % 2 == 1)
            if is_keyword:
                if current_buffer is None:
                    current_buffer = ""
                else:
                    final_text = clean_mark_text(current_buffer)
                    _set_highlight_buffer(None)
                    return f"H:{final_text}"
            else:
                if current_buffer is not None:
                    current_buffer += part
        _set_highlight_buffer(current_buffer)
        if current_buffer is not None:
            return "S"

    # LLM 判斷邏輯
    prompt = (
        "你是簡報輔助系統，目標是分析語句判斷簡報操作。請嚴格遵守以下規則：\n"
        "1. 分析語句，判斷操作類型，僅輸出以下符號：\n"
        "   - 翻到下一頁，輸出'N'\n"
        "   - 不翻頁，輸出'S'\n"
        "   - 翻到上一頁，輸出'P'\n"
        "   - 跳到指定頁數，輸出阿拉伯數字（若為中文數字，無論簡體或繁體大寫，請自動轉換為阿拉伯數字）\n"
        "   - 標記重點，輸出'U'\n"
        "   - 畫螢光筆，輸出'H'\n"
        "2. 計算翻頁機率與不翻頁機率，總和為 1。\n"
        "   - 若翻頁機率>=85%，輸出'N'\n"
        "   - 若翻頁機率<70%，則輸出'S'\n"
        "   - 若翻頁機率介於84%~70%，請謹慎根據上下文判斷\n"
        "3. 嚴格限制輸出，僅輸出：阿拉伯數字、'N'、'P'、'S'、'U'、'H'\n"
        "4. 僅在語句明確提到頁數時輸出阿拉伯數字。\n"
        "5. 中文數字轉阿拉伯數字規則如下（以下為1到12之範例）：\n"
        "   - 中文數字「一」或「壹」對應阿拉伯數字'1'\n"
        "   - 中文數字「二」、「兩」或「貳」對應阿拉伯數字'2'\n"
        "   - 中文數字「三」、「叁」或「參」對應阿拉伯數字'3'\n"
        "   - 中文數字「四」或「肆」對應阿拉伯數字'4'\n"
        "   - 中文數字「五」或「伍」對應阿拉伯數字'5'\n"
        "   - 中文數字「六」或「陸」對應阿拉伯數字'6'\n"
        "   - 中文數字「七」或「柒」對應阿拉伯數字'7'\n"
        "   - 中文數字「八」或「捌」對應阿拉伯數字'8'\n"
        "   - 中文數字「九」或「玖」對應阿拉伯數字'9'\n"
        "   - 中文數字「十」或「拾」對應阿拉伯數字'10'\n"
        "   - 中文數字「十一」或「拾壹」對應阿拉伯數字'11'\n"
        "   - 中文數字「十二」或「拾貳」對應阿拉伯數字'12'\n"
        "   - 若語句中出現中文數字，無論簡體或繁體大寫，必須自動轉換為對應阿拉伯數字"
        "6. 不可僅依靠單個詞語判斷，必須結合上下文分析。對語句中可能出現的錯別字或筆誤，仍需根據上下文語意判斷正確操作。\n"
        "7. 上下文判斷翻頁機制指引：\n"
        "   - 高翻頁機率>=99%，包含'下一頁'、'下一張簡報'\n"
        "   - 低翻頁機率<=10%，包含'讓我們再看一下'、'這邊補充一下'、'下課'\n"
        "   - 前一頁指令，包含'上一頁'、'往前翻一頁'、'回到上一頁'\n"
        "8. 劃重點與螢光筆判定規則如下：\n"
        "   a. 抓取語句中出現標記重點或螢光筆關鍵詞的前後完整語句作為待標記內容\n"
        "   b. 僅當語句中出現以下關鍵詞才觸發操作：\n"
        "       - 標記重點: '標記重點', '畫重點', '畫底線', '劃重點'\n"
        "       - 螢光筆: '螢光筆', '畫螢光筆', '加螢光筆'\n"
        "   c. 操作對應輸出：\n"
        "       - 標記重點，輸出'U'\n"
        "       - 畫螢光筆，輸出'H'\n"
        "   d. 若語句同時包含翻頁指令和標記/螢光筆指令，需同時輸出對應符號\n"
        "   e. 模型需掃描整個語句，忽略錯別字或語尾標點，根據上下文判斷操作\n"
        "   f. 若語句中沒有以上操作詞，則不輸出 'U' 或 'H'\n"
        "請分析以下語句，切記僅輸出單行，不要額外解釋保持整潔且正確。"
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