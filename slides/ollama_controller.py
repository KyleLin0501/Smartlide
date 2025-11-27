import ollama
import re
from opencc import OpenCC
import os
import asyncio
import logging
import threading
import numpy as np
import jieba
import jieba.analyse
from text2vec import SentenceModel
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("slides.ollama_controller")

# ==========================================
# 初始化全域變數
# ==========================================
converter_s2t = OpenCC("s2t")  # 輸出用
converter_t2s = OpenCC("t2s")  # 演算法內部處理用 (簡體對模型支援較好)
selected_model = "llama3.1"

# ==========================================
# 設定遠端 Mac Studio 連線資訊
# ==========================================
MAC_IP = "140.134.26.83"
OLLAMA_HOST = os.environ.get("OLLAMA_URL", f"http://{MAC_IP}:11434")

try:
    async_client = ollama.AsyncClient(host=OLLAMA_HOST)
    logger.info(f"Ollama AsyncClient initialized connecting to {OLLAMA_HOST}")
except Exception as e:
    logger.error(f"Failed to initialize Ollama AsyncClient: {e}")
    async_client = None

# ==========================================
# 懶加載模型機制 (避免啟動時記憶體爆掉)
# ==========================================
_CACHED_MODEL = None


def get_embed_model():
    """只有在真正需要摘要時才載入模型"""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        logger.info("Initializing SentenceModel (Lazy Loading)...")
        try:
            _CACHED_MODEL = SentenceModel('shibing624/text2vec-base-chinese')
            logger.info("SentenceModel loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceModel (Likely OOM): {e}")
            return None
    return _CACHED_MODEL


# ==========================================
# 簡報控制相關 (保留原功能)
# ==========================================
thread_data = threading.local()


def _get_highlight_buffer():
    if not hasattr(thread_data, 'highlight_buffer'):
        thread_data.highlight_buffer = None
    return thread_data.highlight_buffer


def _set_highlight_buffer(val):
    thread_data.highlight_buffer = val


def clean_mark_text(text: str) -> str:
    text = converter_s2t.convert(text.strip())
    keywords = ["畫底線", "畫重點", "標記重點", "底線", "畫線", "重點", "畫螢光筆", "用螢光筆", "螢光筆", "highlight",
                "underline", "mark"]
    for kw in keywords:
        if text.lower().startswith(kw): return text[len(kw):].strip()
        if text.lower().endswith(kw): return text[:-len(kw)].strip()
    return text


def chinese_to_arabic(cn: str):
    table = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
             "十": 10}
    if cn.isdigit(): return int(cn)
    if cn in table: return table[cn]
    if "十" in cn:
        parts = cn.split("十")
        left = table.get(parts[0], 1 if parts[0] == "" else 0)
        right = table.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return left * 10 + right
    return None


async def predict_slide_action(text: str) -> str:
    """判斷簡報操作指令"""
    if async_client is None: return "none"

    text = converter_s2t.convert(text.strip())
    text_lower = text.lower().replace("under line", "underline").replace("high light", "highlight").replace(
        "highlighted", "highlight")

    # 包夾式邏輯
    parts = re.split(r'(highlight|螢光筆)', text_lower, flags=re.IGNORECASE)
    current_buffer = _get_highlight_buffer()

    if len(parts) > 1 or current_buffer is not None:
        for i, part in enumerate(parts):
            if i % 2 == 1:  # 關鍵字
                if current_buffer is None:
                    current_buffer = ""
                else:
                    final = clean_mark_text(current_buffer)
                    _set_highlight_buffer(None)
                    return f"H:{final}"
            else:
                if current_buffer is not None: current_buffer += part
        _set_highlight_buffer(current_buffer)
        if current_buffer is not None: return "S"

    # LLM 邏輯
    prompt = (
        "你是簡報輔助系統，請根據使用者的語句判斷是否為操作指令。\n"
        "規則：\n1. 純講述輸出 'S'。\n2. 下一頁輸出 'N'。\n3. 上一頁輸出 'P'。\n4. 指定頁數輸出數字。\n"
        "5. 畫底線輸出 'U'。\n6. 螢光筆輸出 'H'。\n7. 不確定輸出 'S'。\n只能輸出代碼。"
    )
    try:
        response = await async_client.chat(model=selected_model, messages=[{'role': 'system', 'content': prompt},
                                                                           {'role': 'user', 'content': text}])
        clean_output = re.sub(r'[^a-zA-Z0-9]', '', response.get('message', {}).get('content', '').strip()).upper()
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        clean_output = "S"

    if clean_output == 'U' or 'underline' in text_lower or '底線' in text_lower:
        return f"U:{clean_mark_text(text)}"
    elif clean_output == 'H' or 'highlight' in text_lower or '螢光筆' in text_lower:
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
# 語義摘要演算法 (CPU 密集型邏輯)
# ==========================================
def split_sentences(text):
    def clean_heading_prefix(sentence):
        return re.sub(r'^\s*(第?[一二三四五六七八九十\d]+[、.)：:]?|[一二三四五六七八九十]+\s*[、：])\s*', '', sentence)

    raw = re.split(r'(?<=[。！？……….])\s*', text)
    cleaned = []
    for s in raw:
        s = s.strip()
        if not s or re.fullmatch(r'\s*(第?[一二三四五六七八九十\d]+[、.)：:]?|[一二三四五六七八九十]+\s*[、：])\s*',
                                 s): continue
        cleaned.append(clean_heading_prefix(s))
    return cleaned


def _run_semantic_extraction(full_text: str, top_k=5, top_n=3) -> str:
    """
    這是同步函數，將被 asyncio.to_thread 呼叫。
    負責：斷詞 -> 向量化 -> 分群 -> 提取重點
    """
    model = get_embed_model()
    # 如果模型載入失敗 (OOM)，回傳截斷後的文字作為備案
    if not model:
        logger.warning("Embedding model not loaded, falling back to truncation.")
        return full_text[:5000]

    try:
        # 1. 轉簡體 (Text2Vec 效果較好)
        text_sim = converter_t2s.convert(full_text)

        # 2. 斷句
        sentences = split_sentences(text_sim)
        if len(sentences) < 5: return full_text  # 內容太少不需摘要

        # 3. 提取關鍵詞 (作為標籤)
        keywords = jieba.analyse.textrank(text_sim, topK=top_k, withWeight=False)
        if not keywords: keywords = ["摘要", "重點", "結論"]

        label_names = {i: kw for i, kw in enumerate(keywords)}
        label_texts = list(label_names.values())

        # 4. 向量化
        label_emb = model.encode(label_texts, normalize_embeddings=True)
        sent_emb = model.encode(sentences, normalize_embeddings=True)

        # 5. 分類 (Cosine Similarity)
        labels = []
        for se in sent_emb:
            sims = cosine_similarity([se], label_emb).flatten()
            labels.append(sims.argmax())

        # 6. 分群摘要
        summary_text = ""
        processed_labels = set(labels)

        for label_id in processed_labels:
            indices = [i for i, x in enumerate(labels) if x == label_id]
            if not indices: continue

            cluster_emb = sent_emb[indices]
            centroid = np.mean(cluster_emb, axis=0).reshape(1, -1)
            sims = cosine_similarity(cluster_emb, centroid).flatten()

            # 取該群組分數最高的 top_n 句
            top_indices = sims.argsort()[-top_n:][::-1]

            topic_name = label_names.get(label_id, "重點")
            # 組合句子並轉回繁體
            topic_content = "".join([sentences[indices[i]] for i in top_indices])

            summary_text += f"【{converter_s2t.convert(topic_name)}】\n{converter_s2t.convert(topic_content)}\n\n"

        return summary_text

    except Exception as e:
        logger.error(f"Algorithm execution failed: {e}")
        return full_text[:5000]  # 出錯時回退


# ==========================================
# 最終摘要入口
# ==========================================
async def generate_meeting_summary(transcript: str, pdf_text: str) -> str:
    if async_client is None:
        return "錯誤：LLM 用戶端未初始化。"

    # 合併文本進行處理
    combined_input = f"{pdf_text}\n{transcript}"
    logger.info(f"Generating summary. Input length: {len(combined_input)}")

    # 1. 執行語義提取 (使用 to_thread 避免卡死 WebSocket)
    # 這裡會跑你提供的 Text2Vec 演算法
    try:
        extracted_content = await asyncio.to_thread(
            _run_semantic_extraction,
            combined_input,
            top_k=6,  # 提取 6 個主題
            top_n=4  # 每個主題保留 4 句話
        )
        logger.info(f"Semantic extraction complete. Output length: {len(extracted_content)}")
    except Exception as e:
        logger.error(f"Extraction thread failed: {e}")
        extracted_content = combined_input[:5000]

    # 2. 丟給 Ollama 整理成 Markdown
    prompt = (
        "你是一位專業的會議記錄員。請根據以下經過演算法提取的「重點語句片段」，"
        "整理出一份通順、邏輯清晰的 Markdown 格式會議摘要。\n\n"
        f"{extracted_content}"
        "\n\n請輸出繁體中文摘要："
    )

    try:
        response = await async_client.chat(
            model=selected_model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response.get('message', {}).get('content', '')
    except Exception as e:
        logger.error(f"Ollama Summary generation failed: {e}")
        return f"摘要生成失敗: {e}"