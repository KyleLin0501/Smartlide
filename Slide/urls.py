"""
URL configuration for Slide project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from slides import views

from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve # <--- 新增這個
from django.urls import re_path

urlpatterns = [
    path('admin/', admin.site.urls),

    path('recent/', views.recent, name='recent'),
    path('open/<int:pdf_id>/', views.open_pdf, name='open_pdf'),
    path('overview/', views.overview, name='overview'),
    path('rename_pdf/', views.rename_pdf, name='rename_pdf'),
    path('delete_pdf/', views.delete_pdf, name='delete_pdf'),
    path('upload/', views.upload, name='upload'),
    path('trash/', views.trash, name='trash'),
    path('trash/delete/', views.delete_permanently, name='delete_permanently'),
    path('', views.recent, name='recent'),
    path('register', views.register, name='register'),
    path('test', views.Test, name='Test'),
    path('home', views.home, name='home'),
    path('preview', views.viewer, name='viewer'),
    # 圖片獲取 API
    path('api/page/<int:pdf_id>/<int:page_number>/', views.get_annotated_page_image, name='get_page_image'),
    # 畫記 API
    path('api/mark/<int:pdf_id>/', views.apply_page_action, name='apply_mark_action'),
    path('api/mark/<int:pdf_id>/', views.apply_page_action, name='apply_page_action'),
    path('api/get_mark_positions/<int:pdf_id>/<int:page_number>/', views.get_mark_positions, name='get_mark_positions'),
    path('report/', views.report_view, name='report_view'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# 加入這段：強制 Django 在生產環境也能讀取 media 檔案
urlpatterns += [
    re_path(r'^media/(?P<path>.*)$', serve, {
        'document_root': settings.MEDIA_ROOT,
    }),
]