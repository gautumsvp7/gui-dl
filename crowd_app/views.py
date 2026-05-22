"""
crowd_app/views.py
------------------
Two function-based Django views that implement the CrowdVision GUI.

upload_view  →  GET  : render the upload form (Screen 1)
             →  POST : save the file, store URL in session, redirect to results

results_view →  GET  : read session data, render results page (Screen 2)

WHY function-based views instead of Wagtail pages?
  The upload / results workflow is stateful (file upload + redirect) and
  tightly coupled to POST logic.  Wagtail's Page.serve() can handle this
  but it requires a CMS page entry in the database — meaning a superuser
  must manually create the page in /admin after every fresh db.sqlite3.
  Using plain Django views keeps the screens available at a fixed URL
  (/crowd/upload/ and /crowd/results/) with zero CMS setup, while still
  living inside the same Wagtail project.
"""

import os
import uuid
from pathlib import Path

from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods

from crowd_app.can_inference import load_can_model, predict_crowd


# ─────────────────────────────────────────────────────────────────────────────
# ML model — loaded once when the module is first imported
# ─────────────────────────────────────────────────────────────────────────────

# Weights live at  crowdvision/model/best_model.pth
# BASE_DIR is defined in mysite/settings/base.py and points to the crowdvision/
# directory, so joining 'model/best_model.pth' gives the correct absolute path.
_WEIGHTS_PATH = Path(settings.BASE_DIR) / 'model' / 'best_model.pth'
_MODEL = load_can_model(_WEIGHTS_PATH)


def run_model(image_path: Path) -> dict:
    """
    Run CAN crowd-counting inference on the uploaded image.

    Returns a dict with keys expected by results.html:
        crowd_count     (int)
        mae             (None)   — ground-truth not available at inference time
        mse             (None)   — ground-truth not available at inference time
        model_name      (str)
        density_map_url (str | None)  — None shows the SVG placeholder
    """
    # Save the density map to media/results/<stem>_<uid>_heatmap.png
    dm_filename  = f"results/{image_path.stem}_{uuid.uuid4().hex[:8]}_heatmap.png"
    dm_save_path = Path(settings.MEDIA_ROOT) / dm_filename

    result = predict_crowd(_MODEL, image_path, density_map_save_path=dm_save_path)

    density_map_url = (
        settings.MEDIA_URL + dm_filename.replace('\\', '/')
        if result['density_map_saved']
        else None
    )

    return {
        'crowd_count':     result['crowd_count'],
        'mae':             None,
        'mse':             None,
        'model_name':      'CAN (Context-Aware Network)',
        'density_map_url': density_map_url,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Screen 1 — Upload page
# ─────────────────────────────────────────────────────────────────────────────

@require_http_methods(["GET", "POST"])
def upload_view(request):
    """
    GET  → show the drag-and-drop upload form (upload.html).
    POST → validate and save the uploaded file, then redirect to results.

    File-saving logic:
      1. A UUID is prepended to the original filename to prevent collisions
         when two users upload files with the same name simultaneously.
      2. default_storage.open() is used rather than raw open() so the code
         stays compatible with any Django storage backend (local disk, S3, …).
      3. The public media URL (MEDIA_URL + relative path) is stored in the
         Django session so results_view can read it without a database query.
         Sessions are server-side; the client only holds an opaque session key
         in a cookie, so the file path is never exposed to the browser.
    """
    context = {'error': None}

    if request.method == 'POST':
        file_obj = request.FILES.get('file_data')

        if not file_obj:
            context['error'] = 'No file was selected. Please choose an image or video.'
            return render(request, 'crowd_app/upload.html', context)

        # Validate file extension
        ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.mp4'}
        ext = Path(file_obj.name).suffix.lower()
        if ext not in ALLOWED_EXTS:
            context['error'] = f'Unsupported file type "{ext}". Please upload a jpg, png, or mp4.'
            return render(request, 'crowd_app/upload.html', context)

        # Build a collision-safe filename: <original_stem>_<uuid><ext>
        stem   = Path(file_obj.name).stem
        unique = uuid.uuid4().hex[:8]
        safe_name = f"{stem}_{unique}{ext}"

        # Write to MEDIA_ROOT/uploads/<safe_name>
        rel_path = Path('uploads') / safe_name
        with default_storage.open(str(rel_path), 'wb+') as dest:
            for chunk in file_obj.chunks():
                dest.write(chunk)

        # Store the public URL in the session so results_view can access it.
        # MEDIA_URL is typically '/media/', so the full browser URL becomes
        # /media/uploads/<safe_name>.
        request.session['cv_input_image_url'] = settings.MEDIA_URL + str(rel_path).replace('\\', '/')
        request.session['cv_original_filename'] = file_obj.name

        # POST → Redirect → GET pattern prevents form re-submission on refresh.
        return redirect('crowd_app:results')

    # GET — just show the empty form
    return render(request, 'crowd_app/upload.html', context)


# ─────────────────────────────────────────────────────────────────────────────
# Screen 2 — Results page
# ─────────────────────────────────────────────────────────────────────────────

@require_http_methods(["GET"])
def results_view(request):
    """
    Reads the uploaded image URL from the session and renders the results page.

    Mock stats are used until the real ML model is integrated.
    When the model is wired up:
      1. Load the image from MEDIA_ROOT + relative path.
      2. Run inference → crowd_count, density_map numpy array.
      3. Save the density map to MEDIA_ROOT/results/<name>_heatmap.png.
      4. Compute MAE / MSE against the test set mean or pass them from the
         model's evaluate() method.
      5. Replace the mock values below with the real ones.

    If no session data exists (user navigates to /crowd/results/ directly
    without uploading), we redirect back to the upload page so the flow
    is always: Upload → Results, never Results without an image.
    """
    input_image_url = request.session.get('cv_input_image_url')
    original_filename = request.session.get('cv_original_filename', '')

    # Guard: no file uploaded yet → send back to upload
    if not input_image_url:
        return redirect('crowd_app:upload')

    # Resolve the absolute path for the saved image
    rel_path   = input_image_url.replace(settings.MEDIA_URL, '', 1)
    image_path = Path(settings.MEDIA_ROOT) / rel_path

    # Run CAN inference — returns crowd_count, density_map_url, model_name
    predictions = run_model(image_path)

    context = {
        'input_image_url':   input_image_url,
        'original_filename': original_filename,
        **predictions,
    }

    return render(request, 'crowd_app/results.html', context)
