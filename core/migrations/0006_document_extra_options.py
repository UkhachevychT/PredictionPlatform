# Generated by Django 3.0.6 on 2020-05-28 20:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0005_auto_20200528_0504'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='extra_options',
            field=models.TextField(blank=True, null=True),
        ),
    ]
