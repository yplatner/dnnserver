from django.contrib import admin
from .models import Update, Network, Weight, Layer, Dataset

admin.site.register(Update)
admin.site.register(Network)
admin.site.register(Weight)
admin.site.register(Layer)
admin.site.register(Dataset)
