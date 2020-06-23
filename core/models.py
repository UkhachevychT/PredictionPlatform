import json
import os

import joblib
from django.db import models
from sklearn.neural_network import MLPClassifier

from core.ann.ann import process_data, test


class Document(models.Model):
    name = models.CharField(null=True, blank=True, max_length=100)
    description = models.TextField(null=True, blank=True)
    docfile = models.FileField(upload_to='documents/%Y/%m/%d')
    uploaded_date = models.DateField(auto_now_add=True, null=True, blank=True)
    extra_options = models.TextField(null=True, blank=True)

    def __str__(self):
        return str(self.name) if self.name else ''

    def get_extra(self):
        if self.extra_options:
            extra = json.loads(self.extra_options)
            return extra
        return {}

    def make_model(self):
        def _get_layers(data):
            layers = (30, 30, 30)
            if data.shape[1] > 1:
                layers = (data.shape[1], 2*data.shape[1], 2)
            return layers

        cat = None
        target = 'target'
        if self.extra_options:
            extra = json.loads(self.extra_options)
            cat = extra.get('extra_categor', None)
            target = extra.get('extra_target', 'target')

        data, Y, header_names = process_data(self.docfile.file, target=target, categorical_values=cat)
        model = MLPClassifier(hidden_layer_sizes=_get_layers(data), activation='relu', solver='sgd',
                              learning_rate_init=0.01, max_iter=500)
        accuracy = test(data, Y, model)

        if self.extra_options:
            extra = json.loads(self.extra_options)
            extra['header_names'] = header_names
            self.extra_options = json.dumps(extra)
            self.save()

        model_name = f'{self.name}'
        filename = f'{model_name}.sav'

        if not os.path.exists('./media/models'):
            os.makedirs('./media/models')

        full_path = os.path.join(os.path.abspath("media/models/"), filename)
        joblib.dump(model, full_path)
        Model.objects.create(name=model_name, description=self.description,
                             data=self, docfile=full_path, model_accuracy=accuracy)


class Model(models.Model):
    name = models.CharField(null=True, blank=True, max_length=100)
    description = models.TextField(null=True, blank=True)
    data = models.ForeignKey(Document, on_delete=models.CASCADE)
    docfile = models.FileField(upload_to='documents/%Y/%m/%d')
    uploaded_date = models.DateField(auto_now_add=True, null=True, blank=True)
    model_accuracy = models.FloatField(null=True, blank=True)

    def __str__(self):
        return str(self.name) if self.name else ''

    def predict(self, data_X):
        if data_X is None:
            return None

        loaded_model = joblib.load(self.docfile.file)
        result = loaded_model.predict(data_X)
        return result
