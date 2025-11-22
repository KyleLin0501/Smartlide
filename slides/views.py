import os
import sys
import json
import re
import fitz  # PyMuPDF
import logging
from io import BytesIO
from PIL import Image, ImageDraw

from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.core.files.storage import FileSystemStorage
from django.core.cache import cache
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from datetime import timedelta
from django.conf import settings

from .models import UploadedPDF
from .pdf_converter import convert_pdf_to_markdown

# 設定 Logger
logger = logging.getLogger(__name__)

# 設定 olmocr 路徑 (如果需要)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'olmocr'))


# ==========================================
# 輔助函式 (畫記與文字處理)
# ==========================================

def get_page_cache_key(pdf_id):
    """取得用於儲存畫記狀態的快取鍵"""
    return f"pdf_annotations_{pdf_id}"


def get_drawing_state(pdf_id):
    """從快取獲取畫記狀態"""
    state = cache.get(get_page_cache_key(pdf_id))
    if state is None:
        state = {'annotations': {}, 'highlights': {}, 'rects': {}}

    # 確保 rects 存在
    state.setdefault('rects', {})

    # 正規化 rects 的 key 為整數 (JSON 序列化後 key 會變字串)
    rects = state.get('rects', {})
    new_rects = {}
    for k, v in rects.items():
        try:
            new_rects[int(k)] = v
        except:
            new_rects[k] = v
    state['rects'] = new_rects
    return state


def save_drawing_state(pdf_id, state):
    """儲存畫記狀態到快取 (24小時)"""
    cache.set(get_page_cache_key(pdf_id), state, 60 * 60 * 24)


def _normalize_text_for_match(s: str) -> str:
    """去除標點與空白並小寫化，供精準比對使用"""
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r'[\s\u00A0]+', '', s)  # 移除空白
    s = re.sub(r'[^\w\u4e00-\u9fff]', '', s)  # 只保留中文字與英數字
    return s.lower()


def clean_mark_text(text: str) -> str:
    """清理標記文字"""
    keywords = ["畫底線", "化底線", "畫重點", "標記重點", "底線", "畫螢光筆", "螢光筆"]
    for kw in keywords:
        if text.startswith(kw):
            return text[len(kw):].strip()
        elif text.endswith(kw):
            return text[:-len(kw)].strip()
    return text.strip()


def compute_mark_rects(pdf_path, page_number, target_text):
    """
    使用 PyMuPDF 在 PDF 中搜尋文字座標
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"compute_mark_rects: open pdf failed: {e}")
        return []

    rects = []
    try:
        if page_number < 0 or page_number >= len(doc):
            return []
        page = doc[page_number]

        # 取得頁面尺寸
        page_w = page.rect.width
        page_h = page.rect.height

        target = (target_text or "").strip()
        if not target:
            return []

        # 1. 嘗試原生搜尋
        found = page.search_for(target)
        if found:
            for r in found:
                # 儲存正規化座標 (0.0 ~ 1.0)
                rects.append([r.x0 / page_w, r.y0 / page_h, r.x1 / page_w, r.y1 / page_h])
            return rects

        # 2. Fallback: 簡單模糊搜尋 (比對去空白後的文字)
        words = page.get_text("words")  # (x0, y0, x1, y1, "text", ...)
        norm_target = _normalize_text_for_match(target)

        if norm_target:
            for w in words:
                if len(w) > 4:
                    raw = w[4]
                    if norm_target in _normalize_text_for_match(raw):
                        rects.append([w[0] / page_w, w[1] / page_h, w[2] / page_w, w[3] / page_h])

        return rects

    except Exception as e:
        logger.exception(f"Search error: {e}")
        return []
    finally:
        doc.close()


def render_page_with_marks(pdf_path, page_number, rects_state):
    """
    後端渲染圖片 (Fallback 用)
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Error opening PDF: {e}")
        return None

    if page_number >= len(doc) or page_number < 0:
        doc.close()
        return None

    page = doc[page_number]
    scale = 2.0
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")

        page_rects = rects_state.get(page_number, [])
        if page_rects:
            for item in page_rects:
                rect = item.get('rect')
                typ = item.get('type', 'U')
                if not rect: continue

                x0 = int(rect[0] * pix.width)
                y0 = int(rect[1] * pix.height)
                x1 = int(rect[2] * pix.width)
                y1 = int(rect[3] * pix.height)

                if typ == 'H':
                    highlight_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
                    h_draw = ImageDraw.Draw(highlight_layer)
                    h_draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 0, 100))
                    image = Image.alpha_composite(image, highlight_layer)
                else:
                    underline_y = y1 - 2
                    draw.line((x0, underline_y, x1, underline_y), fill="red", width=3)

        img_io = BytesIO()
        image.save(img_io, format='PNG')
        img_io.seek(0)
        doc.close()
        return img_io
    except Exception as e:
        logger.error(f"Render error: {e}")
        doc.close()
        return None


# ==========================================
# Views (視圖函式)
# ==========================================

def recent(request):
    """顯示最近開啟的檔案"""
    now = timezone.now()
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)

    last_week_pdfs = UploadedPDF.objects.filter(
        is_deleted=False,
        last_opened__gte=seven_days_ago
    ).order_by('-last_opened')

    last_month_pdfs = UploadedPDF.objects.filter(
        is_deleted=False,
        last_opened__gte=thirty_days_ago,
        last_opened__lt=seven_days_ago
    ).order_by('-last_opened')

    return render(request, 'recent.html', {
        'last_week_pdfs': last_week_pdfs,
        'last_month_pdfs': last_month_pdfs,
    })


def open_pdf(request, pdf_id):
    """開啟 PDF 檢視頁面"""
    pdf = get_object_or_404(UploadedPDF, pk=pdf_id, is_deleted=False)
    pdf.last_opened = timezone.now()
    pdf.save()

    fs = FileSystemStorage()
    # 取得實體路徑計算頁數
    pdf_path = fs.path(pdf.file.name)
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
    except Exception:
        page_count = 1

    # 取得前端可存取的 URL
    try:
        relative_url = fs.url(pdf.file.name)
        pdf_url = request.build_absolute_uri(relative_url)
    except Exception:
        pdf_url = ""

    return render(request, 'open_pdf.html', {
        'pdf_id': pdf_id,
        'initial_page_count': page_count,
        'pdf_url': pdf_url
    })


# API: 前端 Canvas 請求畫記座標
def get_mark_positions(request, pdf_id, page_number):
    try:
        state = get_drawing_state(pdf_id)
        rects_state = state.get('rects', {})

        # 支援 int key
        page_rects = rects_state.get(page_number, [])

        return JsonResponse({'status': 'success', 'rects': page_rects})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})


# API: 接收指令並計算座標儲存
@csrf_exempt
def apply_page_action(request, pdf_id):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'POST only'}, status=405)

    try:
        data = json.loads(request.body)
        page_number = int(data.get('page', 0))
        cmd = data.get('cmd', '')
        mark_type = data.get('type', 'U')  # U or H
        text = data.get('text', '')

        # 如果前端沒傳解析好的文字，嘗試從 cmd 解析
        if not text and (cmd.startswith("U:") or cmd.startswith("H:")):
            text = clean_mark_text(cmd[2:])
            mark_type = cmd[0]

        if not text:
            return JsonResponse({'status': 'fail', 'message': 'No text to mark'})

        pdf_record = get_object_or_404(UploadedPDF, pk=pdf_id)
        fs = FileSystemStorage()
        pdf_path = fs.path(pdf_record.file.name)

        # 計算座標
        new_rects_normalized = compute_mark_rects(pdf_path, page_number, text)

        if new_rects_normalized:
            state = get_drawing_state(pdf_id)
            rects_state = state['rects']

            if page_number not in rects_state:
                rects_state[page_number] = []

            for r in new_rects_normalized:
                rects_state[page_number].append({
                    'type': mark_type,
                    'rect': r,
                    'text': text
                })

            save_drawing_state(pdf_id, state)
            return JsonResponse(
                {'status': 'success', 'marks': [{'rect': r, 'type': mark_type} for r in new_rects_normalized]})
        else:
            return JsonResponse({'status': 'not_found', 'message': 'Text not found'})

    except Exception as e:
        logger.exception("Apply action failed")
        return JsonResponse({'status': 'error', 'message': str(e)})


# 取得帶畫記的靜態圖片 (Fallback)
def get_annotated_page_image(request, pdf_id, page_number):
    pdf_record = get_object_or_404(UploadedPDF, pk=pdf_id, is_deleted=False)
    fs = FileSystemStorage()
    pdf_path = fs.path(pdf_record.file.name)

    state = get_drawing_state(pdf_id)
    rects_state = state.get('rects', {})

    # 確保使用 int key
    if page_number not in rects_state and str(page_number) in rects_state:
        rects_state = {**rects_state}
        rects_state[page_number] = rects_state[str(page_number)]

    image_stream = render_page_with_marks(pdf_path, page_number, rects_state)

    if image_stream is None:
        return HttpResponse("Page error", status=404)

    return HttpResponse(image_stream.read(), content_type="image/png")


def overview(request):
    pdfs = UploadedPDF.objects.filter(is_deleted=False).order_by('-uploaded_at')
    return render(request, 'overview.html', {'pdfs': pdfs})


def rename_pdf(request):
    if request.method == 'POST':
        pdf_id = request.POST.get('id')
        new_name = request.POST.get('name')
        pdf = get_object_or_404(UploadedPDF, id=pdf_id)
        pdf.display_name = new_name
        pdf.save()
        return JsonResponse({'status': 'success', 'new_name': new_name})
    return JsonResponse({'status': 'fail'})


def delete_pdf(request):
    if request.method == 'POST':
        pdf_id = request.POST.get('id')
        pdf = get_object_or_404(UploadedPDF, id=pdf_id)
        pdf.is_deleted = True
        pdf.save()
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'fail'})


def trash(request):
    deleted_pdfs = UploadedPDF.objects.filter(is_deleted=True)
    return render(request, 'trash.html', {'deleted_pdfs': deleted_pdfs})


def delete_permanently(request):
    if request.method == 'POST':
        ids = request.POST.getlist('ids[]')
        UploadedPDF.objects.filter(id__in=ids).delete()
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'})


def upload(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('pdf_file'):
        uploaded_file = request.FILES['pdf_file']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        UploadedPDF.objects.create(
            file=filename,
            display_name=uploaded_file.name
        )

        try:
            markdown_output = convert_pdf_to_markdown(file_path)
        except Exception as e:
            markdown_output = f"❌ 轉換失敗：{e}"

        context['uploaded_file_url'] = fs.url(filename)
        context['markdown_output'] = markdown_output

    return render(request, 'upload.html', context)


# 其他基本頁面
def register(request): return render(request, 'register.html')


def index(request): return render(request, 'index.html')


def home(request): return render(request, 'home.html')


def viewer(request): return render(request, 'viewer.html')


def Test(request): return render(request, 'mic.html')

def report_view(request):
    if request.method == "POST":
        # 從 form 接收 content
        content = request.POST.get('content', '')
        return render(request, 'report.html', {'content': content})
    return redirect('home')