import base64
import json
import os
import tempfile
import uuid
from pathlib import Path

from django.conf import settings
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods

from crowd_app.can_inference import load_can_model, predict_crowd, select_model, DENSITY_THRESHOLD


_MODEL_A = load_can_model(Path(settings.BASE_DIR) / 'model' / 'part_a.pth')
_MODEL_B = load_can_model(Path(settings.BASE_DIR) / 'model' / 'part_b.pth')


def run_model(image_path, expected_count):
    model, model_label = select_model(_MODEL_A, _MODEL_B, expected_count)

    dm_filename = "results/{stem}_{uid}_heatmap.png".format(
        stem=image_path.stem, uid=uuid.uuid4().hex[:8]
    )
    dm_save_path = Path(settings.MEDIA_ROOT) / dm_filename

    result = predict_crowd(model, image_path, density_map_save_path=dm_save_path)

    density_map_url = (
        settings.MEDIA_URL + dm_filename.replace('\\', '/')
        if result['density_map_saved'] else None
    )

    return {
        'crowd_count': result['crowd_count'],
        'mae': None,
        'mse': None,
        'model_name': model_label,
        'density_map_url': density_map_url,
    }


@require_http_methods(["GET", "POST"])
def upload_view(request):
    context = {'error': None, 'density_threshold': DENSITY_THRESHOLD}

    if request.method == 'POST':
        file_obj = request.FILES.get('file_data')

        if not file_obj:
            context['error'] = 'No file was selected. Please choose an image or video.'
            return render(request, 'crowd_app/upload.html', context)

        ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.mp4'}
        ext = Path(file_obj.name).suffix.lower()
        if ext not in ALLOWED_EXTS:
            context['error'] = 'Unsupported file type "{}". Please upload a jpg, png, or mp4.'.format(ext)
            return render(request, 'crowd_app/upload.html', context)

        try:
            expected_count = int(request.POST.get('expected_count', 0))
            if expected_count < 0:
                expected_count = 0
        except (ValueError, TypeError):
            expected_count = 0

        stem = Path(file_obj.name).stem
        unique = uuid.uuid4().hex[:8]
        safe_name = "{stem}_{unique}{ext}".format(stem=stem, unique=unique, ext=ext)

        rel_path = Path('uploads') / safe_name
        with default_storage.open(str(rel_path), 'wb+') as dest:
            for chunk in file_obj.chunks():
                dest.write(chunk)

        request.session['cv_input_image_url'] = settings.MEDIA_URL + str(rel_path).replace('\\', '/')
        request.session['cv_original_filename'] = file_obj.name
        request.session['cv_expected_count'] = expected_count

        return redirect('crowd_app:results')

    return render(request, 'crowd_app/upload.html', context)


@require_http_methods(["GET"])
def results_view(request):
    input_image_url = request.session.get('cv_input_image_url')
    original_filename = request.session.get('cv_original_filename', '')
    expected_count = request.session.get('cv_expected_count', 0)

    if not input_image_url:
        return redirect('crowd_app:upload')

    rel_path = input_image_url.replace(settings.MEDIA_URL, '', 1)
    image_path = Path(settings.MEDIA_ROOT) / rel_path

    predictions = run_model(image_path, expected_count)

    context = {
        'input_image_url': input_image_url,
        'original_filename': original_filename,
        'expected_count': expected_count,
        **predictions,
    }

    return render(request, 'crowd_app/results.html', context)


@require_http_methods(["GET"])
def live_view(request):
    return render(request, 'crowd_app/live.html')


@require_http_methods(["GET"])
def about_view(request):
    return render(request, 'about/about.html')


@require_http_methods(["POST"])
def live_infer_view(request):
    try:
        payload = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({'error': 'invalid JSON'}, status=400)

    image_data = payload.get('image', '')
    if not image_data or ',' not in image_data:
        return JsonResponse({'error': 'no image'}, status=400)

    _, encoded = image_data.split(',', 1)
    try:
        img_bytes = base64.b64decode(encoded)
    except Exception:
        return JsonResponse({'error': 'bad base64'}, status=400)

    try:
        expected_count = int(payload.get('expected_count', 0))
        if expected_count < 0:
            expected_count = 0
    except (ValueError, TypeError):
        expected_count = 0

    model, model_label = select_model(_MODEL_A, _MODEL_B, expected_count)

    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix='.jpg')
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, 'wb') as f:
            f.write(img_bytes)

        dm_save_path = Path(settings.MEDIA_ROOT) / 'results' / 'live_heatmap.png'
        result = predict_crowd(model, tmp_path, density_map_save_path=dm_save_path)

        density_map_url = (
            settings.MEDIA_URL + 'results/live_heatmap.png'
            if result['density_map_saved'] else None
        )
        return JsonResponse({
            'crowd_count': result['crowd_count'],
            'model_name': model_label,
            'density_map_url': density_map_url,
        })
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)
    finally:
        tmp_path.unlink(missing_ok=True)
