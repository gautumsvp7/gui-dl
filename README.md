# CrowdVision
42028 Deep Learning and Convolutional Neural Network — Assignment 3

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Place `part_a.pth` and `part_b.pth` in the `model/` directory before starting the server.

- Upload page: `http://localhost:8000/crowd/upload/`
- Results page: `http://localhost:8000/crowd/results/`
