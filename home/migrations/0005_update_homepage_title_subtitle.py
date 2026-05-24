from django.db import migrations


def update_homepage_text(apps, schema_editor):
    HomePage = apps.get_model('home', 'HomePage')
    HomePage.objects.filter(slug='home').update(
        banner_title='CrowdVision',
        banner_subtitle='<p>Safety, powered by AI</p>',
    )


def revert_homepage_text(apps, schema_editor):
    HomePage = apps.get_model('home', 'HomePage')
    HomePage.objects.filter(slug='home').update(
        banner_title='My Application',
        banner_subtitle='<p>My Application Subtitle</p>',
    )


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0004_homepage_banner_image'),
    ]

    operations = [
        migrations.RunPython(update_homepage_text, revert_homepage_text),
    ]
