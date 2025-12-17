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

import io
import fitz  # PyMuPDF
from django.http import FileResponse, HttpResponse
from django.shortcuts import get_object_or_404
from slides.models import UploadedPDF, Mark  # ★ 請替換成您實際的 Model 名稱

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
# views.py

def get_mark_positions(request, pdf_id, page_number):
    """
    從資料庫讀取畫記位置，回傳給前端渲染
    """
    try:
        # 1. 改為查詢資料庫 (Mark Model)
        # 注意：前端傳來的 page_number 是 int，資料庫存的也是 int
        marks = Mark.objects.filter(pdf__id=pdf_id, page=page_number)

        rects_data = []
        for m in marks:
            # 將資料庫物件轉換為前端看得懂的 JSON 格式
            rects_data.append({
                'rect': m.rect,  # 比例座標 [x1, y1, x2, y2]
                'type': m.type,  # 'H' 或 'U' 或 'R'
                'text': m.content
            })

        # 2. 回傳結果
        return JsonResponse({'status': 'success', 'rects': rects_data})

    except Exception as e:
        print(f"❌ 讀取畫記錯誤: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})


# API: 接收指令並計算座標儲存
@csrf_exempt
def apply_page_action_OLD_UNUSED(request, pdf_id):
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
        pdf_url = request.POST.get('pdf_url')  # ★ 接收 pdf_url
        pdf_id = request.POST.get('pdf_id')  # ★ 接收 ID
        context = {
            'content': content,
            'pdf_url': pdf_url,  # ★ 將 pdf_url 加入 context 傳給模板
            'pdf_id': pdf_id  # ★ 傳給模板
        }
        return render(request, 'report.html', context)
    return redirect('home')


@csrf_exempt
def mark_pdf_api(request, pdf_id):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print(f"🔥 [API] 收到畫記請求 PDF ID: {pdf_id}, Data: {data}")

            pdf_record = UploadedPDF.objects.get(pk=pdf_id)
            page_num = int(data.get('page', 0))
            # strip() 會去除前後空白，這是正確的，避免語音多產生空白導致不匹配
            text_to_find = data.get('text', '').strip()
            mark_type = data.get('type', 'R')

            # 開啟 PDF 計算座標
            doc = fitz.open(pdf_record.file.path)

            if page_num < 0 or page_num >= len(doc):
                doc.close()
                return JsonResponse({'status': 'error', 'message': '頁碼錯誤'})

            page = doc[page_num]
            w, h = page.rect.width, page.rect.height

            # 1. 搜尋文字 (PyMuPDF 預設是不分大小寫的，這對語音控制很好)
            found_instances = page.search_for(text_to_find)

            created_marks = []

            # 2. 判斷結果
            if found_instances:
                print(f"✅ 找到 {len(found_instances)} 處完全匹配，存入資料庫...")
                for inst in found_instances:
                    # 轉成比例座標 (0.0 ~ 1.0)
                    rect_ratio = [inst.x0 / w, inst.y0 / h, inst.x1 / w, inst.y1 / h]

                    Mark.objects.create(
                        pdf=pdf_record,
                        page=page_num,
                        type=mark_type,
                        rect=rect_ratio,
                        content=text_to_find
                    )
                    created_marks.append({'rect': rect_ratio, 'type': mark_type})

                doc.close()
                # 回傳成功，前端會畫出框框
                return JsonResponse({'status': 'success', 'marks': created_marks})

            else:
                # ★★★ 這裡做了修改 ★★★
                # 如果找不到完全相符的字，就回傳錯誤訊息，不要強制存檔
                print(f"❌ 找不到精確文字 '{text_to_find}'，略過不存檔。")
                doc.close()

                # 回傳 fail 或 error，讓前端知道沒畫成功
                return JsonResponse({
                    'status': 'fail',
                    'message': f'在第 {page_num + 1} 頁找不到文字：{text_to_find}'
                })

        except Exception as e:
            print(f"❌ API Error: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)})

    return JsonResponse({'status': 'error'})

# ------------------------------------------------------------------
# API 2: 下載合成後的 PDF
# ------------------------------------------------------------------
def download_annotated_pdf(request, pdf_id):
    print(f"📥 [Download] 開始準備下載 PDF ID: {pdf_id}")

    pdf_record = get_object_or_404(UploadedPDF, pk=pdf_id)
    marks = Mark.objects.filter(pdf=pdf_record)

    count = marks.count()
    print(f"📊 資料庫中共有 {count} 筆標記")

    if count == 0:
        print("⚠️ 無標記資料，將下載原始檔")

    # 開啟原始 PDF
    try:
        pdf_doc = fitz.open(pdf_record.file.path)
    except Exception as e:
        return HttpResponse(f"找不到原始檔案: {e}", status=404)

    draw_count = 0

    # 開始繪圖
    for i, mark in enumerate(marks):
        try:
            page_idx = int(mark.page)
            if 0 <= page_idx < len(pdf_doc):
                page = pdf_doc[page_idx]
                w, h = page.rect.width, page.rect.height

                r = mark.rect  # 取出比例座標 [x1, y1, x2, y2]

                # 防呆：確保座標格式正確
                if not isinstance(r, list) or len(r) != 4:
                    print(f"❌ 第 {i + 1} 筆標記座標格式錯誤，跳過。")
                    continue

                # 轉回絕對座標
                rect_coords = fitz.Rect(r[0] * w, r[1] * h, r[2] * w, r[3] * h)
                shape = page.new_shape()

                # --- 判斷標記類型並繪圖 ---
                if mark.type == 'H':
                    # 螢光筆 (Highlight)
                    print(f"   [{i + 1}/{count}] Page {page_idx}: 🖊️ 繪製螢光筆 (Highlight)")
                    shape.draw_rect(rect_coords)
                    shape.finish(color=(1, 1, 0), fill=(1, 1, 0), fill_opacity=0.3, width=0)

                elif mark.type == 'U':
                    # 底線 (Underline)
                    print(f"   [{i + 1}/{count}] Page {page_idx}: 🖊️ 繪製底線 (Underline)")
                    # 從左下畫到右下
                    p1 = fitz.Point(rect_coords.x0, rect_coords.y1)
                    p2 = fitz.Point(rect_coords.x1, rect_coords.y1)
                    shape.draw_line(p1, p2)
                    # 設定線條顏色為紅色，寬度 2
                    shape.finish(color=(1, 0, 0), width=2)

                else:
                    # 預設：紅框 (Red Box)
                    print(f"   [{i + 1}/{count}] Page {page_idx}: 🖊️ 繪製紅框 (Red Box)")
                    shape.draw_rect(rect_coords)
                    shape.finish(color=(1, 0, 0), width=3)

                # 提交繪圖
                shape.commit()
                draw_count += 1
            else:
                print(f"⚠️ Page {page_idx} 超出範圍，跳過。")

        except Exception as e:
            print(f"❌ 繪圖錯誤 (Mark ID {mark.id}): {e}")

    print(f"✅ 成功繪製了 {draw_count} 個標記，正在打包檔案...")

    # 輸出檔案
    buffer = io.BytesIO()
    pdf_doc.save(buffer, garbage=4, deflate=True)
    pdf_doc.close()
    buffer.seek(0)

    filename = f"Annotated_{pdf_record.display_name}.pdf"
    return FileResponse(buffer, as_attachment=True, filename=filename)