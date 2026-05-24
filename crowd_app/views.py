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

from crowd_app.can_inference import load_can_model, predict_crowd


# load model once at startup
_WEIGHTS_PATH = Path(settings.BASE_DIR) / 'model' / 'best_model.pth'
_MODEL = load_can_model(_WEIGHTS_PATH)


def run_model(image_path):
    dm_filename = f"results/{image_path.stem}_{uuid.uuid4().hex[:8]}_heatmap.png"
    dm_save_path = Path(settings.MEDIA_ROOT) / dm_filename

    result = predict_crowd(_MODEL, image_path, density_map_save_path=dm_save_path)

    density_map_url = (
        settings.MEDIA_URL + dm_filename.replace('\\', '/')
        if result['density_map_saved'] else None
    )

    return {
        'crowd_count':     result['crowd_count'],
        'mae':             None,
        'mse':             None,
        'model_name':      'CAN (Context-Aware Network)',
        'density_map_url': density_map_url,
    }


@require_http_methods(["GET", "POST"])
def upload_view(request):
    context = {'error': None}

    if request.method == 'POST':
        file_obj = request.FILES.get('file_data')

        if not file_obj:
            context['error'] = 'No file was selected. Please choose an image or video.'
            return render(request, 'crowd_app/upload.html', context)

        ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.mp4'}
        ext = Path(file_obj.name).suffix.lower()
        if ext not in ALLOWED_EXTS:
            context['error'] = f'Unsupported file type "{ext}". Please upload a jpg, png, or mp4.'
            return render(request, 'crowd_app/upload.html', context)

        stem = Path(file_obj.name).stem
        unique = uuid.uuid4().hex[:8]
        safe_name = f"{stem}_{unique}{ext}"

        rel_path = Path('uploads') / safe_name
        with default_storage.open(str(rel_path), 'wb+') as dest:
            for chunk in file_obj.chunks():
                dest.write(chunk)

        request.session['cv_input_image_url'] = settings.MEDIA_URL + str(rel_path).replace('\\', '/')
        request.session['cv_original_filename'] = file_obj.name

        return redirect('crowd_app:results')

    return render(request, 'crowd_app/upload.html', context)


@require_http_methods(["GET"])
def results_view(request):
    input_image_url = request.session.get('cv_input_image_url')
    original_filename = request.session.get('cv_original_filename', '')

    if not input_image_url:
        return redirect('crowd_app:upload')

    rel_path = input_image_url.replace(settings.MEDIA_URL, '', 1)
    image_path = Path(settings.MEDIA_ROOT) / rel_path

    predictions = run_model(image_path)

    context = {
        'input_image_url':   input_image_url,
        'original_filename': original_filename,
        **predictions,
    }

    return render(request, 'crowd_app/results.html', context)


@require_http_methods(["GET"])
def live_view(request):
    return render(request, 'crowd_app/live.html')


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

    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix='.jpg')
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(tmp_fd, 'wb') as f:
            f.write(img_bytes)

        dm_save_path = Path(settings.MEDIA_ROOT) / 'results' / 'live_heatmap.png'
        result = predict_crowd(_MODEL, tmp_path, density_map_save_path=dm_save_path)

        density_map_url = (
            settings.MEDIA_URL + 'results/live_heatmap.png'
            if result['density_map_saved'] else None
        )
        return JsonResponse({
            'crowd_count':     result['crowd_count'],
            'density_map_url': density_map_url,
        })
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)
    finally:
        tmp_path.unlink(missing_ok=True)
