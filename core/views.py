import json

from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse

from core.ann.ann import process_data
from core.forms import DocumentForm, ModelPredictDynamicForm
from core.models import Document, Model


def index(request):
    models = Model.objects.all()
    return render(
        request,
        'core/index.html',
        {'models': models, 'form': None},
    )


def list(request):

    def process_extra(**kwargs):
        extra = {}
        if kwargs.get('extra_categor'):
            extra['extra_categor'] = kwargs['extra_categor'].split()
        if kwargs.get('extra_skip'):
            extra['extra_skip'] = kwargs['extra_skip'].split()
        if kwargs.get('extra_target'):
            if len(kwargs['extra_target'].split()) > 1:
                raise Exception('Unproperly configured field: extra_target')
            extra['extra_target'] = kwargs['extra_target']
        if kwargs.get('extra_map'):
            extra['extra_map'] = kwargs['extra_map']
        return json.dumps(extra)

    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            extra = process_extra(extra_categor=form.cleaned_data['extra_categor'],
                                  extra_skip=form.cleaned_data['extra_skip'],
                                  extra_target=form.cleaned_data['extra_target'],
                                  extra_map=form.cleaned_data['extra_map'])

            newdoc = Document(name=form.cleaned_data['name'], description=form.cleaned_data['description'],
                              docfile=form.cleaned_data['docfile'], extra_options=extra)
            newdoc.save()
            newdoc.make_model()

            return HttpResponseRedirect(reverse('index'))
    else:
        form = DocumentForm()

    documents = Document.objects.all()

    return render(
        request,
        'core/list.html',
        {'documents': documents, 'form': form},
    )


def predict(request, *args, **kwargs):
    if request.method == 'POST':
        form = ModelPredictDynamicForm(request.POST, request.FILES)
        if form.is_valid():
            model = Model.objects.get(pk=kwargs['model_id'])
            data = {}

            if not request.FILES:
                for hn in model.data.get_extra()['header_names']:
                    data[hn] = request.POST[hn]
            else:
                data = request.FILES['data']

            cat = None
            target = 'target'
            if model.data.get_extra():
                extra = model.data.get_extra()
                cat = extra.get('extra_categor', None)
                target = extra.get('extra_target', 'target')

            data_to_predict, data = process_data(data, is_training_data=False, target=target, categorical_values=cat,
                                                 initial_data=model.data.docfile)

            res = model.predict(data_to_predict)
            extra = json.loads(model.data.extra_options)
            maps = extra.get('extra_map')
            if maps and target:
                if type(maps) == str:
                    maps = json.loads(maps)

            if len(res) == 1:
                res = maps.get(target)[str(res[0])] if maps and target else res
            else:
                new_res = []
                for idx, val in enumerate(res):
                    new_res.append(maps.get(target)[str(val)] if maps and target else val)
                data["result"] = new_res
                data = data.to_csv(index=False)
                response = HttpResponse(content_type='text/csv')
                response['Content-Disposition'] = 'attachment; filename="result.csv"'
                response.write(data)
                return response

            return render(request, 'core/model_page.html', {'form': form, 'model_id': kwargs['model_id'],
                                                            'model': model, 'prediction_result': res})
    else:
        model = Model.objects.get(pk=kwargs['model_id'])
        extra = model.data.get_extra()
        form = ModelPredictDynamicForm(header_names=extra.get('header_names', None))

    return render(
        request,
        'core/model_page.html',
        {'form': form, "model_id": kwargs['model_id'], 'model': model},
    )
