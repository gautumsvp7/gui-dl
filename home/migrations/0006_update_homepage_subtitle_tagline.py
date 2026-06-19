from django.db import migrations


def update_subtitle(apps, schema_editor):
    HomePage = apps.get_model('home', 'HomePage')
    HomePage.objects.filter(slug='home').update(
        banner_subtitle='<p>See Everything. Protect Everyone.</p>',
    )


def revert_subtitle(apps, schema_editor):
    HomePage = apps.get_model('home', 'HomePage')
    HomePage.objects.filter(slug='home').update(
        banner_subtitle='<p>Safety, powered by AI</p>',
    )


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0005_update_homepage_title_subtitle'),
    ]

    operations = [
        migrations.RunPython(update_subtitle, revert_subtitle),
    ]
