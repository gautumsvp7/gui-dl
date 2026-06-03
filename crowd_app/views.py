import base64
import io
import json
import os
import uuid
from pathlib import Path

from django.conf import settings
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from PIL import Image

from crowd_app.can_inference import load_can_model, predict_crowd, select_model, DENSITY_THRESHOLD


# load both models at startup so we're not reloading on every request
MODEL_A = load_can_model(Path(settings.BASE_DIR) / 'model' / 'part_a.pth')
MODEL_B = load_can_model(Path(settings.BASE_DIR) / 'model' / 'part_b.pth')


def upload_view(request):
    context = {'error': None, 'density_threshold': DENSITY_THRESHOLD}

    if request.method == 'POST':
        file_obj = request.FILES.get('file_data')

        if not file_obj:
            context['error'] = 'No file selected.'
            return render(request, 'crowd_app/upload.html', context)

        # only accept jpg/png
        ext = Path(file_obj.name).suffix.lower()
        if ext not in ['.jpg', '.jpeg', '.png']:
            context['error'] = 'File type not supported. Please upload a jpg or png image.'
            return render(request, 'crowd_app/upload.html', context)

        # get the expected count from the form (used to choose Part A vs Part B)
        try:
            expected_count = int(request.POST.get('expected_count', 0))
        except ValueError:
            expected_count = 0

        if expected_count < 0:
            expected_count = 0

        # save the uploaded file with a unique name to avoid collisions
        original_stem = Path(file_obj.name).stem
        unique_id = uuid.uuid4().hex[:8]
        save_name = '{}_{}{}'.format(original_stem, unique_id, ext)
        save_path = os.path.join('uploads', save_name)

        with default_storage.open(save_path, 'wb+') as f:
            for chunk in file_obj.chunks():
                f.write(chunk)

        # store in session so the results page can pick it up
        image_url = settings.MEDIA_URL + save_path.replace('\\', '/')
        request.session['input_image_url'] = image_url
        request.session['original_filename'] = file_obj.name
        request.session['expected_count'] = expected_count

        return redirect('crowd_app:results')

    return render(request, 'crowd_app/upload.html', context)


def results_view(request):
    input_image_url = request.session.get('input_image_url')
    original_filename = request.session.get('original_filename', '')
    expected_count = request.session.get('expected_count', 0)

    # if they navigate here directly without uploading, send them back
    if not input_image_url:
        return redirect('crowd_app:upload')

    # reconstruct the file path from the URL
    rel_path = input_image_url.replace(settings.MEDIA_URL, '', 1)
    image_path = Path(settings.MEDIA_ROOT) / rel_path

    # choose model based on expected crowd size
    model, model_label = select_model(MODEL_A, MODEL_B, expected_count)

    # run inference and save the density map
    heatmap_name = 'results/{}_heatmap.png'.format(uuid.uuid4().hex[:8])
    heatmap_path = Path(settings.MEDIA_ROOT) / heatmap_name

    crowd_count = predict_crowd(model, image_path, density_map_save_path=heatmap_path)
    density_map_url = settings.MEDIA_URL + heatmap_name

    context = {
        'input_image_url': input_image_url,
        'original_filename': original_filename,
        'expected_count': expected_count,
        'crowd_count': crowd_count,
        'model_name': model_label,
        'density_map_url': density_map_url,
        # could add MAE/MSE here if we had ground truth
        'mae': None,
        'mse': None,
    }

    return render(request, 'crowd_app/results.html', context)


def live_view(request):
    return render(request, 'crowd_app/live.html')


def about_view(request):
    return render(request, 'about/about.html')


def live_infer_view(request):
    """
    Called by the live page via fetch() — receives a base64 webcam frame,
    runs inference, and returns the count + heatmap URL as JSON.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'invalid JSON'}, status=400)

    image_data = payload.get('image', '')
    if not image_data or ',' not in image_data:
        return JsonResponse({'error': 'no image data'}, status=400)

    # strip the data URL header (e.g. "data:image/jpeg;base64,")
    header, encoded = image_data.split(',', 1)

    try:
        img_bytes = base64.b64decode(encoded)
    except Exception:
        return JsonResponse({'error': 'could not decode image'}, status=400)

    try:
        expected_count = int(payload.get('expected_count', 0))
        if expected_count < 0:
            expected_count = 0
    except (ValueError, TypeError):
        expected_count = 0

    model, model_label = select_model(MODEL_A, MODEL_B, expected_count)

    # save the frame temporarily so predict_crowd can open it
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tmp_path = Path(settings.MEDIA_ROOT) / 'results' / 'live_tmp.jpg'
    img.save(str(tmp_path))

    heatmap_path = Path(settings.MEDIA_ROOT) / 'results' / 'live_heatmap.png'
    crowd_count = predict_crowd(model, tmp_path, density_map_save_path=heatmap_path)

    return JsonResponse({
        'crowd_count': crowd_count,
        'model_name': model_label,
        'density_map_url': settings.MEDIA_URL + 'results/live_heatmap.png',
    })
