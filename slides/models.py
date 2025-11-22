from django.db import models

# Create your models here.
# models.py
from django.db import models

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
        super().save(*args, **kwargs)  # 先儲存 PDF

        if self.file and not self.thumbnail:
            pdf_path = os.path.join(settings.MEDIA_ROOT, self.file.name)
            thumbnail_dir = os.path.join(settings.MEDIA_ROOT, 'thumbnails')
            os.makedirs(thumbnail_dir, exist_ok=True)

            # ✅ 指定 poppler 的安裝路徑（請依實際情況調整）
            poppler_path = "/opt/homebrew/bin"  # Apple Silicon 預設路徑
            # 或 Intel Mac：
            # poppler_path = "/usr/local/bin"

            images = convert_from_path(
                pdf_path,
                first_page=1,
                last_page=1,
                poppler_path=poppler_path
            )

            if images:
                thumb_path = os.path.join('thumbnails', f'{self.pk}_thumb.jpg')
                full_thumb_path = os.path.join(settings.MEDIA_ROOT, thumb_path)
                images[0].save(full_thumb_path, 'JPEG')
                self.thumbnail = thumb_path
                super().save(update_fields=['thumbnail'])

    class Meta:
        verbose_name = "上傳 PDF"
        verbose_name_plural = "上傳 PDF 列表"

