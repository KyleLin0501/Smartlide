# 使用 Python 3.10 Slim 版本 (體積小且穩定)
FROM python:3.10-slim

# 設定環境變數：不緩衝輸出 (方便看 Log)、不產生 pyc 檔
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 設定工作目錄
WORKDIR /app

# --- 關鍵：安裝系統層級依賴 ---
# poppler-utils: 修復 PDFInfoNotInstalledError
# ffmpeg: 修復語音辨識依賴
# libpq-dev, gcc: 編譯某些 Python 套件需要
RUN apt-get update && apt-get install -y \
    poppler-utils \
    ffmpeg \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 複製專案所有程式碼到容器內
COPY . .

# 收集靜態文件 (Render 部署必須)
RUN python manage.py collectstatic --noinput

# 暴露 Port (雖然 Render 會自己管理，但寫著比較清楚)
EXPOSE 8000

# --- 啟動指令 ---
# 使用 daphne 啟動 ASGI 以支援 WebSocket
# ⚠️ 請把下面的 [你的專案資料夾名稱] 改成你 settings.py 所在的資料夾名字！
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "Slide.asgi:application"]