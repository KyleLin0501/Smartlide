import ollama
import re
import os
import logging
import threading
import time  # 用於計算時間差
import httpx  # 必須安裝: pip install httpx
from opencc import OpenCC
from difflib import SequenceMatcher  # 用於模糊比對

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
# [修改] 狀態管理: 帶有時間過期機制的暫存區
# ==========================================
# Key: session_id
# Value: { "text": "上一段文字", "timestamp": 171566... }
USER_LAST_SEGMENT = {}

# 設定暫存存活時間 (秒)，超過此時間則視為過期，不進行合併
BUFFER_TTL = 5.0


def get_last_segment(session_id: str) -> str:
    data = USER_LAST_SEGMENT.get(session_id)
    if not data:
        return ""

    # 檢查是否過期
    time_diff = time.time() - data.get("timestamp", 0)
    if time_diff > BUFFER_TTL:
        # 過期了，清除並回傳空字串
        clear_last_segment(session_id)
        return ""

    return data.get("text", "")


def set_last_segment(session_id: str, text: str):
    # 寫入當前時間
    USER_LAST_SEGMENT[session_id] = {
        "text": text,
        "timestamp": time.time()
    }


def clear_last_segment(session_id: str):
    if session_id in USER_LAST_SEGMENT:
        del USER_LAST_SEGMENT[session_id]


# ==========================================
# 輔助函式
# ==========================================

def add_space_between_zh_en(text: str) -> str:
    """
    強制在中英文之間加入空白。
    注意：這一步是為了讓 'fuzzy_correct_keywords' 能正確辨識黏在一起的指令
    (如 '語音辨識Highlight')。雖然您最後希望沒有空格，但這一步對於
    '找出指令' 是必須的。我們會在最後輸出的步驟把空格全部刪掉。
    """
    # 中文接英文 -> 加空白 (例如：識H)
    text = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z])', r'\1 \2', text)
    # 英文接中文 -> 加空白 (例如：t語)
    text = re.sub(r'([a-zA-Z])([\u4e00-\u9fa5])', r'\1 \2', text)
    return text


def remove_punctuation_and_spaces(text: str) -> str:
    """
    [嚴格清理]
    1. 移除所有標點符號
    2. 移除所有空白 (跨句空格、詞間空格)
    3. 只保留：中文、英文、數字
    """
    # Regex 邏輯：
    # [^\u4e00-\u9fa5a-zA-Z0-9] 代表 "非" (中文 或 英文 或 數字) 的所有字元
    # 將這些字元全部替換為空字串
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)


def clean_mark_text(text: str) -> str:
    # 這是舊的清理函式，保留作為 Fallback
    text = converter.convert(text.strip())
    keywords = [
        "畫底線", "畫重點", "標記重點", "底線", "畫線", "重點",
        "畫螢光筆", "用螢光筆", "螢光筆", "highlight", "underline", "mark"
    ]
    for _ in range(2):
        for kw in keywords:
            if text.lower().startswith(kw):
                text = text[len(kw):].strip()
            if text.lower().endswith(kw):
                text = text[:-len(kw)].strip()

    # [關鍵修改] 最後做嚴格清理 (去符號、去空格、留數字)
    return remove_punctuation_and_spaces(text)


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


def fuzzy_correct_keywords(text: str, threshold=0.8) -> str:
    """
    使用相似度比對，將錯誤的語音辨識單字修正為標準指令。
    """
    targets = {
        "underline": "underline",
        "highlight": "highlight"
    }
    words = text.split()
    corrected_words = []

    for word in words:
        clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())
        best_match = None
        highest_ratio = 0.0

        for target in targets:
            ratio = SequenceMatcher(None, clean_word, target).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = target

        if highest_ratio >= threshold and best_match:
            corrected_words.append(targets[best_match])
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


# ==========================================
# [修改] 夾心內容提取函式
# ==========================================
def extract_sandwiched_content(text: str) -> str:
    """
    從字串中精準提取「第一個關鍵字」與「最後一個關鍵字」中間的內容。
    """
    text = converter.convert(text.strip())
    triggers = ['highlight', '螢光筆', 'underline', '畫底線', '底線']

    # 建立 Regex: (highlight|螢光筆|...)
    pattern = "|".join(map(re.escape, triggers))

    matches = list(re.finditer(pattern, text, re.IGNORECASE))

    if len(matches) < 2:
        return clean_mark_text(text)

    # 取第一個關鍵字的「結束點」
    start_idx = matches[0].end()
    # 取最後一個關鍵字的「開始點」
    end_idx = matches[-1].start()

    # 提取中間內容
    extracted = ""
    if start_idx < end_idx:
        extracted = text[start_idx:end_idx]

    # [關鍵修改] 強制執行嚴格清理 (去空格、去符號、留數字)
    return remove_punctuation_and_spaces(extracted)


# ==========================================
# 功能 1: 簡報指令判斷 (支援滑動視窗邏輯)
# ==========================================
async def predict_slide_action(text: str, session_id: str = "default") -> str:
    """
    預測簡報動作指令
    :param text: 本次語音輸入
    :param session_id: 用戶識別碼
    """
    if async_client is None:
        return "none"

    # 1. 基礎清理
    text = converter.convert(text.strip())

    # 先進行中英文斷開 (為了辨識指令，例如 "語音辨識Highlight")
    text = add_space_between_zh_en(text)

    # 2. 模糊修正 (Underlight -> underline)
    text = fuzzy_correct_keywords(text, threshold=0.8)

    # 3. 標準化替換
    def standardize(t):
        t = re.sub(r'high\s*light(ed)?', 'highlight', t, flags=re.IGNORECASE)
        t = re.sub(r'under\s*line', 'underline', t, flags=re.IGNORECASE)
        return t

    current_text_std = standardize(text)

    # 4. 【滑動視窗邏輯】取出上一段並合併
    # get_last_segment 內部會檢查 TTL
    prev_text = get_last_segment(session_id)

    # [關鍵修改] 組合出「完整文本」
    # 這裡移除原本中間的 " " (空格)，直接相加
    # 雖然前面 add_space 可能加了空格，但那是為了指令辨識
    # 最終內容會透過 remove_punctuation_and_spaces 把所有空格再刪掉
    full_check_text = (prev_text + current_text_std).strip() if prev_text else current_text_std

    # 5. 檢查關鍵字
    triggers = ['highlight', '螢光筆', 'underline', '畫底線', '底線']

    # 計算組合字串中出現的關鍵字總次數
    keyword_count = 0
    last_keyword = ""

    # 轉小寫以利計算
    lower_full_text = full_check_text.lower()
    for t in triggers:
        cnt = lower_full_text.count(t)
        if cnt > 0:
            keyword_count += cnt
            last_keyword = t

    # --- 決策樹 ---

    # [情況 A] 成功組成指令 (關鍵字 >= 2，代表有頭有尾)
    if keyword_count >= 2:
        # 使用 extract_sandwiched_content 提取並自動嚴格清理
        final_content = extract_sandwiched_content(full_check_text)

        clear_last_segment(session_id)  # 任務完成，清除暫存

        if last_keyword in ['underline', '畫底線', '底線']:
            return f"U:{final_content}"
        else:
            return f"H:{final_content}"

    # [情況 B] 尚未完成 (關鍵字 = 1 或是 處於等待狀態)
    elif keyword_count > 0 or prev_text != "":
        # 只要還有未閉合的關鍵字，就保留等待下一句
        set_last_segment(session_id, full_check_text)
        return "S"

    # [情況 C] 完全沒有關鍵字 -> 進入原本的 LLM 判斷
    else:
        clear_last_segment(session_id)  # 確保暫存乾淨

        # 先用 Regex 快速篩選
        cmd_text = current_text_std.lower()
        if re.search(r"(下一頁|next|往後|continue)", cmd_text): return "next"
        if re.search(r"(上一頁|prev|回去|previous|back)", cmd_text): return "prev"

        cn_match = re.search(r"(第)?([零一二兩三四五六七八九十]+)頁", text)
        if cn_match:
            n = chinese_to_arabic(cn_match.group(2))
            if n: return f"goto:{n}"

        # LLM 判斷
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

        # LLM 後處理 (這裡也會經過 remove_punctuation_and_spaces)
        if clean_output == 'U': return f"U:{clean_mark_text(text)}"
        if clean_output == 'H': return f"H:{clean_mark_text(text)}"
        if clean_output == 'N': return "next"
        if clean_output == 'P': return "prev"
        if re.match(r'^\d+$', clean_output): return f"goto:{clean_output}"

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