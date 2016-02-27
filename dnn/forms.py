from django import forms
from models import Dataset, Network, Layer, Update, Weight, Dataset

class ActionForm(forms.Form):
    ACTIONS = (
        ('train',       'Train network'),
        ('validate',    'Validate network'),
        ('predict',     'Predict'),
    )
    action = forms.ChoiceField(choices=ACTIONS)
    inputs = forms.ModelChoiceField(queryset=Dataset.objects.all())
    targets = forms.ModelChoiceField(queryset=Dataset.objects.all(), required=False)
    batch_size = forms.IntegerField()
    epochs = forms.IntegerField()
    
class NetworkForm(forms.ModelForm):
    class Meta:
        model = Network
        #fields = '__all__'
        fields = ['name', 'user', 'update', 'loss_objective', 'loss_delta', 'regularization', 'penalty', 'coefficient']
        
class LayerForm(forms.ModelForm):
    class Meta:
        model = Layer
        fields = '__all__'
        
class UpdateForm(forms.ModelForm):
    class Meta:
        model = Update
        fields = '__all__'
        
class WeightForm(forms.ModelForm):
    class Meta:
        model = Weight
        fields = '__all__'
        
class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = '__all__'