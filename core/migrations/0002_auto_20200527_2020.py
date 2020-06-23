# Generated by Django 3.0.6 on 2020-05-27 20:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='is_model',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='document',
            name='model_accuracy',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='document',
            name='name',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='document',
            name='uploaded_date',
            field=models.DateField(auto_now_add=True, null=True),
        ),
    ]