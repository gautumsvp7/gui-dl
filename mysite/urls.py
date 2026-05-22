from django.conf import settings
from django.urls import include, path
from django.contrib import admin

from wagtail.admin import urls as wagtailadmin_urls
from wagtail.core import urls as wagtail_urls
from wagtail.documents import urls as wagtaildocs_urls

urlpatterns = [

    path('django-admin/', admin.site.urls),

    path('admin/', include(wagtailadmin_urls)),
    path('documents/', include(wagtaildocs_urls)),

    # ── CrowdVision app ─────────────────────────────────────────────
    # namespace='crowd_app' lets templates use {% url 'crowd_app:upload' %}
    # and {% url 'crowd_app:results' %} without hard-coding paths.
    path('crowd/', include('crowd_app.urls', namespace='crowd_app')),

    path('', include(wagtail_urls)),

]


if settings.DEBUG:
    from django.conf.urls.static import static
    from django.contrib.staticfiles.urls import staticfiles_urlpatterns

    # Serve static and media files from development server
    urlpatterns += staticfiles_urlpatterns()
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
