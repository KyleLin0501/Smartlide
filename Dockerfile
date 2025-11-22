# 使用官方 Python 映像檔 (建議與你本地開發版本接近，例如 3.10 或 3.11)
FROM python:3.10-slim

# 設定環境變數
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴 (關鍵步驟！)
# poppler-utils: 解決 PDFInfoNotInstalledError
# build-essential, libpq-dev: 避免安裝某些 Python 套件時編譯失敗
# ffmpeg: 如果你的語音辨識(Whisper)需要處理音訊，通常也需要這個
RUN apt-get update && apt-get install -y \
    poppler-utils \
    build-essential \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝 Python 套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案程式碼
COPY . .

# 收集靜態文件 (如果需要)
# RUN python manage.py collectstatic --noinput

# 暴露 Port (Render 預設會看這個，但實際由環境變數 PORT 決定)
EXPOSE 8000

# 啟動指令
# 注意：因為你有用到 WebSocket (channels)，必須使用 daphne 或 uvicorn 啟動 ASGI
# 請將 'your_project_name' 換成你實際的專案資料夾名稱 (存放 settings.py 的那個)
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "Slide.asgi:application"]