# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dnn', '0013_remove_dataset_loaded'),
    ]

    operations = [
        migrations.AddField(
            model_name='dataset',
            name='split',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='dataset',
            name='split_size_first',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='dataset',
            name='split_size_second',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='dataset',
            name='split_take_first',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='layer',
            name='alpha',
            field=models.FloatField(default=0.0001),
        ),
        migrations.AddField(
            model_name='layer',
            name='beta',
            field=models.FloatField(default=0.75),
        ),
        migrations.AddField(
            model_name='layer',
            name='k',
            field=models.FloatField(default=2),
        ),
        migrations.AddField(
            model_name='layer',
            name='n',
            field=models.IntegerField(default=5),
        ),
        migrations.AddField(
            model_name='network',
            name='coefficient',
            field=models.FloatField(default=1),
        ),
        migrations.AddField(
            model_name='network',
            name='penalty',
            field=models.CharField(default=b'l1', max_length=50, choices=[(b'l1', b'Binary Crossentropy'), (b'l2', b'Categorical Crossentropy')]),
        ),
        migrations.AddField(
            model_name='network',
            name='regularization',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='layer',
            name='base_type',
            field=models.CharField(default=b'InputLayer', max_length=200, choices=[(b'InputLayer', b'A network input layer'), (b'DenseLayer', b'A fully connected layer.'), (b'Conv2DLayer', b'2D convolutional layer'), (b'MaxPool2DLayer', b'2D max-pooling layer'), (b'DropoutLayer', b'Dropout layer'), (b'LocalResponseNormalization2DLayer', b'Cross-channel Local Response Normalization for 2D feature maps.')]),
        ),
    ]
