# CrowdVision — Crowd Management Using Computer Vision
42028 Deep Learning and Convolutional Neural Network — Assignment 3

## How to run (Option 2 from the GUI Workshop PDF — start from the example)

```bash
# 1. Create and activate a virtual environment
python -m venv crowd_venv
source crowd_venv/bin/activate          # Windows: crowd_venv\Scripts\activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Apply database migrations
cd crowdvision
python manage.py makemigrations
python manage.py migrate

# 4. Create a Wagtail admin superuser (username + password of your choice)
python manage.py createsuperuser

# 5. Start the development server
python manage.py runserver
```

Then open:
| URL | What you see |
|-----|-------------|
| `http://localhost:8000/crowd/upload/` | Screen 1 — Upload page |
| `http://localhost:8000/crowd/results/` | Screen 2 — Results page |
| `http://localhost:8000/admin/` | Wagtail CMS admin |

---

## Project structure (new files added for CrowdVision)

```
crowdvision/
├── crowd_app/                  ← NEW Django app
│   ├── __init__.py
│   ├── apps.py                 ← AppConfig (name = 'crowd_app')
│   ├── models.py               ← No DB models yet; see file for future schema
│   ├── urls.py                 ← /crowd/upload/  and  /crowd/results/
│   └── views.py                ← upload_view (GET+POST)  +  results_view (GET)
│
├── mysite/
│   ├── settings/base.py        ← crowd_app added to INSTALLED_APPS
│   ├── urls.py                 ← path('crowd/', include('crowd_app.urls', ...))
│   ├── static/css/
│   │   └── crowdvision.css     ← Wireframe-accurate styles (beige, monospace)
│   └── templates/
│       ├── base.html           ← CrowdVision navbar + wireframe base layout
│       └── crowd_app/
│           ├── upload.html     ← Screen 1: drag-and-drop upload
│           └── results.html    ← Screen 2: images + stat cards
│
└── media/
    ├── uploads/                ← Uploaded crowd images stored here
    └── results/                ← Generated density maps stored here (future)
```

---

## Connecting the ML model

When your model is trained, edit `crowd_app/views.py` → `results_view`:

```python
# Replace the mock block with real inference:
from your_model_module import load_model, predict

model = load_model('path/to/weights.pth')
img_path = os.path.join(settings.MEDIA_ROOT, relative_path)
crowd_count, density_map_array = predict(model, img_path)

# Save the density map
heatmap_filename = f"results/{uuid.uuid4().hex}_heatmap.png"
save_heatmap(density_map_array, os.path.join(settings.MEDIA_ROOT, heatmap_filename))
density_map_url = settings.MEDIA_URL + heatmap_filename
```

---

## Team
| Name | Student ID |
|------|-----------|
| Dongyeon Kim | 14490585 |
| Gautum Vaisiam Parambil | 25526979 |
| Manaswini Doma | 25517970 |
