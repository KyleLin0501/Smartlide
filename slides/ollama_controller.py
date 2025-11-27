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

# 設定 Logger
logger = logging.getLogger("slides.ollama_controller")

# ==========================================
# 初始化全域變數與模型
# ==========================================
converter_s2t = OpenCC("s2t")  # 簡轉繁 (給輸出用)
converter_t2s = OpenCC("t2s")  # 繁轉簡 (給 text2vec 模型內部處理用較佳)
selected_model = "llama3.1"

# 延遲加載 SentenceModel 以避免啟動過慢，或者在此處直接加載
# 注意：text2vec 模型加載會佔用記憶體
logger.info("Loading SentenceModel...")
try:
    embed_model = SentenceModel('shibing624/text2vec-base-chinese')
    logger.info("SentenceModel loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load SentenceModel: {e}")
    embed_model = None

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

# 執行緒局部變數
thread_data = threading.local()


# ... (保留原本的 helper functions: _get_highlight_buffer, clean_mark_text, chinese_to_arabic) ...
def _get_highlight_buffer():
    if not hasattr(thread_data, 'highlight_buffer'):
        thread_data.highlight_buffer = None
    return thread_data.highlight_buffer


def _set_highlight_buffer(val):
    thread_data.highlight_buffer = val


def clean_mark_text(text: str) -> str:
    text = converter_s2t.convert(text.strip())
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


# ... (保留原本的 predict_slide_action 邏輯，此處省略以節省篇幅，內容不變) ...
async def predict_slide_action(text: str) -> str:
    # (維持你原本的代碼不變)
    if async_client is None: return "none"
    text = converter_s2t.convert(text.strip())
    text_lower = text.lower()
    text_lower = text_lower.replace("under line", "underline").replace("high light", "highlight")
    # ... Wrapper Logic & LLM Logic (同原程式碼) ...
    # 這裡請貼上你原本的 predict_slide_action 內容
    # 為節省空間，這裡假設原本邏輯已存在
    return "none"


# ==========================================
# 語義摘要演算法 (CPU 密集型邏輯)
# ==========================================
def split_sentences(text):
    """改進後的斷句函數"""

    def clean_heading_prefix(sentence):
        return re.sub(r'^\s*(第?[一二三四五六七八九十\d]+[、.)：:]?|[一二三四五六七八九十]+\s*[、：])\s*', '', sentence)

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


def _extract_key_content_logic(full_text: str, top_k_keywords=5, top_n_sentences=3) -> str:
    """
    執行 Text2Vec + Clustering 的同步邏輯
    """
    if not embed_model:
        return full_text[:5000]  # 如果模型沒加載成功，退回截斷模式

    # 1. 轉簡體以利模型處理 (text2vec-base-chinese 對簡體支援較好)
    text_sim = converter_t2s.convert(full_text)

    # 2. 斷句
    sentences = split_sentences(text_sim)
    if len(sentences) < 5:
        return full_text  # 句子太少不需要摘要

    # 3. 提取關鍵詞 (作為聚類標籤)
    keywords = jieba.analyse.textrank(text_sim, topK=top_k_keywords, withWeight=False)
    if not keywords:
        keywords = ["摘要", "重點", "內容"]  # fallback

    # 4. 建立標籤對應
    label_names = {i: kw for i, kw in enumerate(keywords)}
    label_texts = list(label_names.values())

    # 5. 編碼
    label_embeddings = embed_model.encode(label_texts, normalize_embeddings=True)
    sentence_embeddings = embed_model.encode(sentences, normalize_embeddings=True)

    # 6. 分類句子 (Cosine Similarity)
    labels = []
    for sent_emb in sentence_embeddings:
        sims = cosine_similarity([sent_emb], label_embeddings).flatten()
        labels.append(sims.argmax())

    # 7. 每個群組挑選代表性句子
    extracted_sections = []
    processed_labels = set(labels)

    for label_id in processed_labels:
        # 找出該群組的所有句子索引
        indices = [i for i, x in enumerate(labels) if x == label_id]
        if not indices: continue

        cluster_embeddings = sentence_embeddings[indices]
        # 計算群組中心
        centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
        # 計算距離中心的相似度
        sims = cosine_similarity(cluster_embeddings, centroid).flatten()
        # 取前 N 個最相似的句子
        top_indices_local = sims.argsort()[-top_n_sentences:][::-1]

        # 組合該主題的重點
        topic_name = label_names.get(label_id, "重點")
        topic_content = "".join([sentences[indices[i]] for i in top_indices_local])

        # 轉回繁體
        topic_content_tw = converter_s2t.convert(topic_content)
        topic_name_tw = converter_s2t.convert(topic_name)

        extracted_sections.append(f"【{topic_name_tw}】\n{topic_content_tw}")

    return "\n\n".join(extracted_sections)


# ==========================================
# 修改後的摘要生成功能
# ==========================================
async def generate_meeting_summary(transcript: str, pdf_text: str) -> str:
    """
    使用 Text2Vec 進行重點提取後，再交由 LLM 生成最終摘要
    """
    if async_client is None:
        return "錯誤：LLM 用戶端未初始化。"

    # 合併文本進行分析 (PDF 提供專有名詞背景，Transcript 提供當下重點)
    # 通常我們會希望分析全部內容，但為了效能，如果真的太長還是可以做初步限制
    combined_input = f"{pdf_text}\n{transcript}"

    logger.info(f"Processing semantic extraction for text length: {len(combined_input)}")

    try:
        # **關鍵步驟**：使用 asyncio.to_thread 在背景執行緒運行耗時的向量計算
        # 這樣才不會卡住你的 slide 控制器
        extracted_content = await asyncio.to_thread(
            _extract_key_content_logic,
            combined_input,
            top_k_keywords=6,  # 提取 6 個主題
            top_n_sentences=4  # 每個主題留 4 句
        )

        logger.info(f"Extraction complete. Context reduced from {len(combined_input)} to {len(extracted_content)}")

    except Exception as e:
        logger.error(f"Semantic extraction failed: {e}")
        # 如果演算法失敗，退回原本的截斷方式
        extracted_content = f"PDF內容:\n{pdf_text[:5000]}\n逐字稿:\n{transcript[:5000]}"

    # 構建 Prompt
    prompt = (
        "請整理出一份流暢、邏輯清晰的 Markdown 格式會議摘要。\n\n"
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
        logger.error(f"Summary generation failed: {e}")
        return f"摘要生成失敗: {e}"