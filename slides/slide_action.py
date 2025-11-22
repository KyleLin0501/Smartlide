import ollama
import re
import sys

selected_model = "llama3.1"

def clean_mark_text(text: str) -> str:
    """
    移除「畫底線」、「畫重點」、「螢光筆」等指令關鍵字，只保留標記內容。
    """
    keywords = ["畫底線", "畫重點", "標記重點", "底線", "畫螢光筆", "螢光筆"]
    for kw in keywords:
        if text.startswith(kw):
            return text[len(kw):].strip()
        elif text.endswith(kw):
            return text[:-len(kw)].strip()
    return text.strip()


def chinese_to_arabic(cn: str):
    """
    將中文數字轉換成阿拉伯數字
    """
    cn_num = {
        '零': 0, '一': 1, '二': 2, '兩': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
    }
    if cn.isdigit():
        return int(cn)
    if cn in cn_num:
        return cn_num[cn]
    if '十' in cn:  # 處理「十一」「二十五」這類
        parts = cn.split('十')
        left = cn_num.get(parts[0], 1 if parts[0] == '' else 0)
        right = cn_num.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return left * 10 + right
    return None


def predict_slide_action(text: str) -> str:
    """
    輸入一段語音文字，回傳簡報控制指令：
    - 'N' = 下一頁
    - 'P' = 上一頁
    - 數字 = 跳轉到某頁
    - 'U:xxx' = 畫底線，xxx 為標記內容
    - 'H:xxx' = 畫螢光筆，xxx 為標記內容
    - 'S' = 停止 / 不翻頁
    """

    prompt = (
        "你是簡報輔助系統，目的是判斷是否需要翻頁或標記。請遵守以下規則：\n"
        "1. 當語句明確表示『下一頁』、『下一張』，輸出 'N'。\n"
        "2. 當語句明確表示『上一頁』、『回到上一頁』，輸出 'P'。\n"
        "3. 當語句包含『第X頁』、『第X張』，輸出數字 X（阿拉伯數字）。\n"
        "4. 當語句包含『畫底線』、『畫重點』、『標記重點』，輸出 'U'。\n"
        "5. 當語句包含『畫螢光筆』、『螢光筆』，輸出 'H'。\n"
        "6. 其他情況一律輸出 'S'（表示不做動作）。\n"
        "7. 僅能輸出 'N'、'P'、'S'、數字、'U'、'H'，不得有其他文字。\n"
    )

    response = ollama.chat(
        model=selected_model,
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': text}
        ]
    )

    output = response.get('message', {}).get('content', '').strip()

    # 底線指令
    if output.upper() == 'U':
        return f"U:{clean_mark_text(text)}"

    # 螢光筆指令
    if output.upper() == 'H':
        return f"H:{clean_mark_text(text)}"

    # 翻頁指令
    if output.upper() in ['N', 'P', 'S']:
        return output.upper()

    # 頁數（阿拉伯數字）
    if re.match(r'^\d+$', output):
        return output

    # 頁數（中文數字）
    if re.match(r'^[零一二兩三四五六七八九十]+$', output):
        arabic = chinese_to_arabic(output)
        if arabic is not None:
            return str(arabic)

    return 'S'  # 預設不動作


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = sys.argv[1]
        result = predict_slide_action(input_text)
        print(result)
    else:
        print("未接收到輸入文字")
