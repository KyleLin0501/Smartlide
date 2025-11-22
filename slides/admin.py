from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import UploadedPDF

from django.utils.html import format_html

@admin.register(UploadedPDF)
class UploadedPDFAdmin(admin.ModelAdmin):
    list_display = ('display_name', 'uploaded_at', 'thumbnail_preview')

    def thumbnail_preview(self, obj):
        if obj.thumbnail:
            return format_html('<img src="{}" width="100" />', obj.thumbnail.url)
        return '無縮圖'
    thumbnail_preview.short_description = '縮圖預覽'
