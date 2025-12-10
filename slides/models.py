from django.db import models

# Create your models here.
# models.py
from django.db import models
import platform
from django.db import models
from pdf2image import convert_from_path
from PIL import Image
import os
from django.conf import settings

class UploadedPDF(models.Model):
    file = models.FileField(upload_to='pdfs/')
    thumbnail = models.ImageField(upload_to='thumbnails/', blank=True, null=True)  # 新增欄位
    display_name = models.CharField(max_length=255, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_deleted = models.BooleanField(default=False)
    last_opened = models.DateTimeField(blank=True, null=True)

    def __str__(self):
        return self.display_name or self.file.name

    from pdf2image import convert_from_path
    import os
    from django.conf import settings

    def save(self, *args, **kwargs):
        # 1. 先執行原本的存檔，確保檔案已經寫入磁碟，有路徑可以讀取
        super().save(*args, **kwargs)

        # 2. 如果有檔案但還沒有縮圖，才執行轉換
        if self.file and not self.thumbnail:
            pdf_path = self.file.path  # 直接用 self.file.path 比較穩
            thumbnail_dir = os.path.join(settings.MEDIA_ROOT, 'thumbnails')
            os.makedirs(thumbnail_dir, exist_ok=True)

            # --- 關鍵修正：自動判斷路徑 ---
            poppler_path = None
            system_os = platform.system()

            if system_os == 'Linux':
                # Render 伺服器 (Docker 環境)
                poppler_path = '/usr/bin'
            elif system_os == 'Darwin':
                # 本地 macOS 環境
                if os.path.exists("/opt/homebrew/bin"):
                    poppler_path = "/opt/homebrew/bin"  # Apple Silicon (M1/M2/M3)
                else:
                    poppler_path = "/usr/local/bin"  # Intel Mac
            # Windows 通常不需要指定，會自動抓 PATH

            try:
                images = convert_from_path(
                    pdf_path,
                    first_page=1,
                    last_page=1,
                    poppler_path=poppler_path  # 使用動態判斷的路徑
                )

                if images:
                    thumb_path = os.path.join('thumbnails', f'{self.pk}_thumb.jpg')
                    full_thumb_path = os.path.join(settings.MEDIA_ROOT, thumb_path)

                    # 儲存圖片
                    images[0].save(full_thumb_path, 'JPEG')

                    # 更新資料庫欄位
                    self.thumbnail = thumb_path
                    # 注意：一定要用 update_fields，否則會造成無窮遞迴 save()
                    super().save(update_fields=['thumbnail'])

            except Exception as e:
                # 建議印出錯誤但不要讓程式崩潰 (Crash)，這樣使用者至少還能上傳成功
                print(f"⚠️ 縮圖生成失敗 (ID: {self.pk}): {e}")
    class Meta:
        verbose_name = "上傳 PDF"
        verbose_name_plural = "上傳 PDF 列表"


class Mark(models.Model):
    # 這是用來儲存畫記座標的
    pdf = models.ForeignKey(UploadedPDF, on_delete=models.CASCADE, related_name='marks')
    page = models.IntegerField()  # 頁碼 (0 開始)
    type = models.CharField(max_length=5)  # 'H' (螢光筆) 或 'R' (紅框)
    rect = models.JSONField() # 儲存座標 [x1, y1, x2, y2] 的比例
    content = models.TextField(blank=True, null=True) # 標記文字
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Mark: Page {self.page} - {self.content}"

