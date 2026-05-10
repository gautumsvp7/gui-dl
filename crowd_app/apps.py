from django.apps import AppConfig


class CrowdAppConfig(AppConfig):
    """
    AppConfig for the CrowdVision upload / results module.

    Django requires every app to have an AppConfig so it can be
    discovered when listed in INSTALLED_APPS.  The 'name' attribute
    must match the dotted path used in INSTALLED_APPS ('crowd_app').
    """
    name = 'crowd_app'
    verbose_name = 'CrowdVision'
