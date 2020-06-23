import json

from django import forms


class DocumentForm(forms.Form):
    name = forms.CharField(label='Назва вибірки')
    description = forms.CharField(label='Опис вибірки', widget=forms.Textarea)
    docfile = forms.FileField(
        label='Виберіть вибірку',
        help_text='Формат csv, max - 42mb'
    )
    extra_categor = forms.CharField(label='Категоріальні ознаки', required=False,
                                    help_text='Не обов\'язкове поле. Введіть назви категоріальних ознак \
                                               (через пробіл). Наприклад, "озанка1 оазнака2 ознака3"')
    extra_skip = forms.CharField(label='Ознаки, які можна пропустити', required=False,
                                 help_text='Не обов\'язкове поле. Введіть назви ознак, які можна ігнорувати \
                                            (через пробіл). Наприклад, "озанка1 оазнака2 ознака3"')
    extra_target = forms.CharField(label='Ознака - мітка', required=False,
                                   help_text='Введіть назву ознаки-мітки, яка позначає клас')
    extra_map = forms.CharField(label='Словник кодів міток класу', widget=forms.Textarea, required=False,
                                help_text='Не обов\'язкове поле. Введіть назви кодів міток класу наступним чином: \
                                               "target": {"0": "хвороба присутня", "1": "хвороба відсутня"}')

    def clean_extra_map(self):
        jdata = self.cleaned_data['extra_map']
        try:
            json_data = json.loads(jdata)
        except json.decoder.JSONDecodeError as err:
            if 'Expecting property name enclosed in double quotes' in err.msg:
                jdata = jdata.replace("'", "+").replace('"', "'").replace("+", '"')
                try:
                    json_data = json.loads(jdata)
                except:
                    raise forms.ValidationError("Invalid data in extra_map")
        except:
            raise forms.ValidationError("Invalid data in extra_map")
        # if json data not valid:
        # raise forms.ValidationError("Invalid data in extra_map")
        return jdata


class ModelPredictForm(forms.Form):
    data = forms.FileField(label='Виберіть файл',
                           help_text='Select file with data to predict',
                           required=False)


class DataUploadForm(forms.Form):
    data = forms.FileField(
        label='Виберіть файл',
        help_text='max. 42 megabytes'
    )


class ModelPredictDynamicForm(forms.Form):
    data = forms.FileField(label='Виберіть файл, або введіть дані вручну',
                           required=False)

    def __init__(self, header_names=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if header_names:
            for f in header_names:
                if f == 'csrfmiddlewaretoken':
                    continue
                self.fields[f] = forms.CharField(required=False)
