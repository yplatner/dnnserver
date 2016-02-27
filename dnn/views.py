import threading
from django.utils import timezone
from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.core.exceptions import PermissionDenied
from models import Network, Dataset, Result      # .models ?
from forms import *
from logic import load, train, validate, predict, str_to_obj, reset

def index(request):
    obj_list = Network.objects.order_by('-creation_date')
    if request.method == 'POST':
        form = NetworkForm(request.POST)
        if form.is_valid():
            new_network = form.save()
            return HttpResponseRedirect('/dnn/')
    else:
        form = NetworkForm()
    return render(request, 'index.html', {'obj_list': obj_list, 'form': form})
    
def network_reset(request, network_id):
    network = Network.objects.get(id=network_id)
    reset(network)
    return HttpResponseRedirect('/dnn')      
    
def network_edit(request, network_id):
    obj_list = Network.objects.order_by('-creation_date')
    network = Network.objects.get(id=network_id)
    if request.method == 'POST':
        form = NetworkForm(request.POST, instance=network)
        if form.is_valid():
            new_network = form.save()
            return HttpResponseRedirect('/dnn/')
    else:
        form = NetworkForm(instance=network)
    return render(request, 'index.html', {'obj_list': obj_list, 'form': form})   
    
def updates(request):
    obj_list = Update.objects.order_by('-name')
    if request.method == 'POST':
        form = UpdateForm(request.POST)
        if form.is_valid():
            new_update = form.save()
            return HttpResponseRedirect('/dnn/updates/')
    else:
        form = UpdateForm()
    return render(request, 'updates.html', {'obj_list': obj_list, 'form': form})
    
def update_edit(request, update_id):
    obj_list = Update.objects.order_by('-name')
    update = Update.objects.get(id=update_id)
    if request.method == 'POST':
        form = UpdateForm(request.POST, instance=update)
        if form.is_valid():
            new_update = form.save()
            return HttpResponseRedirect('/dnn/updates/')
    else:
        form = UpdateForm(instance=update)
    return render(request, 'updates.html', {'obj_list': obj_list, 'form': form})    
    
def weights(request):
    obj_list = Weight.objects.order_by('-name')
    if request.method == 'POST':
        form = WeightForm(request.POST)
        if form.is_valid():
            new_weight = form.save()
            return HttpResponseRedirect('/dnn/weights/')
    else:
        form = WeightForm()
    return render(request, 'weights.html', {'obj_list': obj_list, 'form': form})
    
def weight_edit(request, weight_id):
    obj_list = Weight.objects.order_by('-name')
    weight = Weight.objects.get(id=weight_id)
    if request.method == 'POST':
        form = UpdateForm(request.POST, instance=weight)
        if form.is_valid():
            new_weight = form.save()
            return HttpResponseRedirect('/dnn/weights/')
    else:
        form = WeightForm(instance=weight)
    return render(request, 'weights.html', {'obj_list': obj_list, 'form': form})       
    
def datasets(request):
    obj_list = Dataset.objects.order_by('-name')
    if request.method == 'POST':
        form = DatasetForm(request.POST)
        if form.is_valid():
            new_dataset = form.save()
            return HttpResponseRedirect('/dnn/datasets/')
    else:
        form = DatasetForm()
    return render(request, 'datasets.html', {'obj_list': obj_list, 'form': form})    
    
def dataset_edit(request, dataset_id):
    obj_list = Dataset.objects.order_by('-name')
    dataset = Dataset.objects.get(id=dataset_id)
    if request.method == 'POST':
        form = DatasetForm(request.POST, instance=dataset)
        if form.is_valid():
            new_dataset = form.save()
            return HttpResponseRedirect('/dnn/datasets/')
    else:
        form = DatasetForm(instance=dataset)
    return render(request, 'datasets.html', {'obj_list': obj_list, 'form': form})       
    
def layers(request, network_id):
    network = get_object_or_404(Network, pk=network_id)
    layers = network.layer_set.order_by('-level').reverse().all
    if request.method == 'POST':
        form = LayerForm(request.POST)
        if form.is_valid():
            new_layer = form.save()
            return HttpResponseRedirect('/dnn/' + str(network.id))
    else:
        form = LayerForm()
    return render(request, 'layers.html', {'network': network, 'layers': layers, 'form': form})   
    
def layer_edit(request, layer_id):
    layer = Layer.objects.get(id=layer_id)
    network = layer.network
    layers = network.layer_set.order_by('-level').reverse().all
    if request.method == 'POST':
        form = LayerForm(request.POST, instance=layer)
        if form.is_valid():
            new_layer = form.save()
            return HttpResponseRedirect('/dnn/' + str(network.id))
    else:
        form = LayerForm(instance=layer)
    return render(request, 'layers.html', {'network': network, 'layers': layers, 'form': form})          
    
def actions(request, network_id):
    network = get_object_or_404(Network, pk=network_id)
    if request.method == 'POST':
        form = ActionForm(request.POST)
        if form.is_valid():
            if form.cleaned_data['action'] == "predict":
                url_str = '/dnn/' + form.cleaned_data['action'] + '/' + str(network_id) + '/' + str(form.cleaned_data['inputs'].id)
            else:
                url_str = '/dnn/' + form.cleaned_data['action'] + '/' + str(network_id) + '/' + str(form.cleaned_data['inputs'].id) + '/' + str(form.cleaned_data['targets'].id) + '/' + str(form.cleaned_data['batch_size']) + '/' + str(form.cleaned_data['epochs'])
            return HttpResponseRedirect(url_str)
    else:
        form = ActionForm()
    return render(request, 'actions.html', {'network': network, 'form': form})    
        
def do_train(request, network_id, inputs_dataset_id, targets_dataset_id, batch_size, epochs):
    network = Network.objects.get(pk=network_id)
    if request.user == network.user :
        result = Result()
        result.action = 'training'
        result.updated_at = timezone.now()
        result.save()
        t = threading.Thread(target=train, args=(network_id, result.id, inputs_dataset_id, targets_dataset_id, int(batch_size), int(epochs)))
        t.setDaemon(True)
        t.start()
        return redirect('/dnn/result/' + str(result.id))
    else :
        raise PermissionDenied
        
def do_validate(request, network_id, inputs_dataset_id, targets_dataset_id, batch_size, epochs):
    network = Network.objects.get(pk=network_id)
    if request.user == network.user :
        result = Result()
        result.action = 'validation'
        result.updated_at = timezone.now()
        result.save()
        #raise Exception("hello")
        t = threading.Thread(target=validate, args=(network_id, result.id, inputs_dataset_id, targets_dataset_id, int(batch_size)))
        t.setDaemon(True)
        t.start()
        return redirect('/dnn/result/' + str(result.id))
    else :
        raise PermissionDenied
        
def do_predict(request, network_id, inputs_dataset_id):
    network = Network.objects.get(pk=network_id)
    if request.user == network.user :
        result = Result()
        result.action = 'prediction'
        result.updated_at = timezone.now()
        result.save()
        t = threading.Thread(target=predict, args=(network_id, result.id, inputs_dataset_id))
        t.setDaemon(True)
        t.start()
        return redirect('/dnn/result/' + str(result.id))
    else :
        raise PermissionDenied

def do_run_mnist(request):
    import threading
    t = threading.Thread(target=run_mnist)
    t.setDaemon(True)
    t.start()
    return HttpResponse()
    
def display_result(request, result_id):
    result = Result.objects.get(pk=result_id)
    # TODO convert inputs and predictions from tensors/vectors to readable images/list
    context = {
        'action'            : result.action,
        'loaded'            : result.loaded,
        'error'             : result.error,
        'accuracy'          : result.accuracy,
        'classification'    : result.classification,
        'updated_at'        : result.updated_at,
    }
    return render(request, 'result.html', context)