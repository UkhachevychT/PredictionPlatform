from django.contrib import admin

from .models import Document, Model


class DocumentAdmin(admin.ModelAdmin):
    list_display = ['pk', 'name', 'uploaded_date', 'docfile']


class PredictionModelAdmin(admin.ModelAdmin):
    list_display = ['pk', 'name', 'uploaded_date', 'model_accuracy', 'docfile']


admin.site.register(Document, DocumentAdmin)
admin.site.register(Model, PredictionModelAdmin)
