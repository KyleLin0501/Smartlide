import ollama
import re
import os
import asyncio
import logging
import threading
import jieba
import jieba.analyse
import numpy as np

from opencc import OpenCC
from text2vec import SentenceModel
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 初始化設定
# ==========================================
logger = logging.getLogger("slides.ollama_controller")
converter = OpenCC("s2t")
converter_t2s = OpenCC("t2s")  # 增加轉簡體轉換器，因為 text2vec 對簡體支援較佳

# 設定模型名稱
SELECTED_LLM_MODEL = "llama3.1"
EMBEDDING_MODEL_NAME = 'shibing624/text2vec-base-chinese'

# 全域載入 Sentence Model (避免每次請求都重新載入，這會消耗 RAM)
# 注意：第一次執行會下載模型
logger.info("Loading SentenceModel...")
try:
    sentence_model = SentenceModel(EMBEDDING_MODEL_NAME)
    logger.info("SentenceModel loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load SentenceModel: {e}")
    sentence_model = None

# ==========================================
# 1. 設定遠端 Mac Studio 連線資訊
# ==========================================
MAC_IP = "140.134.26.83"
OLLAMA_HOST = os.environ.get("OLLAMA_URL", f"http://{MAC_IP}:11434")

# 初始化異步客戶端
try:
    async_client = ollama.AsyncClient(host=OLLAMA_HOST)
    logger.info(f"Ollama AsyncClient initialized connecting to {OLLAMA_HOST}")
except Exception as e:
    logger.error(f"Failed to initialize Ollama AsyncClient: {e}")
    async_client = None

# 執行緒局部變數 (用於儲存每個連線的螢光筆狀態)
thread_data = threading.local()


# ==========================================
# 2. 文本處理核心邏輯 (從第一段代碼移植)
# ==========================================

def split_sentences(text):
    """改進後的斷句函數"""

    def clean_heading_prefix(sentence):
        return re.sub(r'^\s*(第?[一二三四五六七八九十\d]+[、.)：:]?|[一二三四五六七八九十]+\s*[、：])\s*', '', sentence)

    # 先轉為簡體進行處理 (text2vec 偏好)
    text = converter_t2s.convert(text)
    raw_sentences = re.split(r'(?<=[。！？……….])\s*', text)
    cleaned_sentences = []

    for s in raw_sentences:
        s = s.strip()
        if not s: continue
        # 過濾純標題
        if re.fullmatch(r'\s*(第?[一二三四五六七八九十\d]+[、.)：:]?|[一二三四五六七八九十]+\s*[、：])\s*', s):
            continue
        cleaned_sentences.append(clean_heading_prefix(s))

    return cleaned_sentences


def classify_by_semantic_labels(sentences, label_names):
    """將句子分類到語義最接近的標籤"""
    if not sentence_model: return []

    label_texts = list(label_names.values())
    label_embeddings = sentence_model.encode(label_texts, normalize_embeddings=True)
    sentence_embeddings = sentence_model.encode(sentences, normalize_embeddings=True)

    labels = []
    for sent_emb in sentence_embeddings:
        sims = cosine_similarity([sent_emb], label_embeddings).flatten()
        best_label_idx = sims.argmax()
        labels.append(best_label_idx)

    return labels, sentence_embeddings


def summarize_by_semantic_labels(sentences, embeddings, labels, label_names, top_n=3):
    """根據分類結果提取重點句子"""
    summary_text = ""
    unique_labels = set(labels)

    for label_id in unique_labels:
        # 找出屬於該群集的句子索引
        indices = [i for i, x in enumerate(labels) if x == label_id]
        if not indices: continue

        cluster_sentences = [sentences[i] for i in indices]
        cluster_embeddings = [embeddings[i] for i in indices]

        # 計算群集中心
        centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
        sims = cosine_similarity(cluster_embeddings, centroid).flatten()

        # 取出最接近中心的 top_n 句子
        top_indices_local = sims.argsort()[-top_n:][::-1]

        label_name = label_names.get(label_id, f"類別{label_id}")
        # 轉回繁體中文以便閱讀
        label_name_tw = converter.convert(label_name)

        summary_text += f"【{label_name_tw}】\n"
        for idx in top_indices_local:
            sent_tw = converter.convert(cluster_sentences[idx])
            summary_text += f"{sent_tw}。\n"
        summary_text += "\n"

    return summary_text


def _sync_process_transcript(full_text: str):
    """
    同步執行的重型運算函式 (將被 asyncio.to_thread 呼叫)
    執行斷句、嵌入、分群、關鍵詞提取
    """
    if not sentence_model or not full_text:
        return full_text  # 若模型未載入或無文字，原樣返回

    # 1. 提取關鍵詞 (作為分類標籤)
    # text2vec 較適合處理簡體，先轉換
    text_s = converter_t2s.convert(full_text)
    keywords = jieba.analyse.textrank(text_s, topK=5, withWeight=False)

    if not keywords:
        return full_text

    label_names = {i: kw for i, kw in enumerate(keywords)}

    # 2. 斷句
    sentences = split_sentences(full_text)
    if len(sentences) < 5:  # 句子太少就不跑分群了
        return full_text

    # 3. 分類與計算向量 (最耗時的部分)
    labels, embeddings = classify_by_semantic_labels(sentences, label_names)

    # 4. 產生結構化摘要草稿
    structured_draft = summarize_by_semantic_labels(sentences, embeddings, labels, label_names, top_n=3)

    return structured_draft


# ==========================================
# 3. 既有輔助函式
# ==========================================

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
# 4. 核心異步功能
# ==========================================

async def predict_slide_action(text: str) -> str:
    """判斷簡報操作指令"""
    if async_client is None: return "none"

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
        if current_buffer is not None: return "S"

    # LLM 判斷邏輯
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


async def generate_meeting_summary(transcript: str, pdf_text: str) -> str:
    """
    生成會議摘要：
    1. 結合 PDF 與 逐字稿。
    2. 使用 text2vec 進行嵌入與分群提取重點 (在 thread 中運行以避免阻塞)。
    3. 使用 Ollama 進行最終潤色。
    """
    if async_client is None:
        return "錯誤：LLM 用戶端未初始化。"

    logger.info(f"Start Summary Generation. Transcript len: {len(transcript)}")

    # 組合文本 (通常逐字稿比較發散，PDF 比較精簡，這裡主要針對大量的逐字稿進行整理)
    # 我們將兩者合併，或是主要處理逐字稿，這裡示範合併處理，但你可以只傳入 transcript
    full_text = f"{pdf_text}\n\n{transcript}"

    # 步驟 1: 結構化提取 (CPU 密集型任務，放入 Thread 執行)
    try:
        # 使用 asyncio.to_thread 避免卡住 Event Loop
        logger.info("Running text clustering in background thread...")
        structured_notes = await asyncio.to_thread(_sync_process_transcript, full_text)
        logger.info("Clustering finished.")
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        # 若分群失敗，退化為使用原始文本的前 2000 字
        structured_notes = full_text[:2000]

    # 步驟 2: LLM 潤色
    prompt = (
        "你是一個文章潤色與總結輔助系統。以下是透過演算法提取的「分類重點筆記」。"
        "請根據這些重點，撰寫一份通順、邏輯清晰的「繁體中文會議摘要」。"
        "請保留核心數據與觀點，並去除重複資訊。不要使用 Markdown 格式，僅輸出純文字。\n\n"
        "--- [演算法提取的重點] ---\n"
        f"{structured_notes}"
    )

    try:
        logger.info("Sending to Ollama for final polish...")
        response = await async_client.chat(
            model=SELECTED_LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        final_output = response.get('message', {}).get('content', '')
        return converter.convert(final_output)  # 確保最終輸出為繁體
    except Exception as e:
        logger.error(f"Summary generation failed at LLM stage: {e}")
        return f"摘要生成失敗: {e}"