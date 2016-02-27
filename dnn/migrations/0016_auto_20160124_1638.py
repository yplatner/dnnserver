# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dnn', '0015_weight_init_gain_relu'),
    ]

    operations = [
        migrations.AlterField(
            model_name='layer',
            name='base_type',
            field=models.CharField(default=b'InputLayer', max_length=200, choices=[(b'InputLayer', b'A network input layer'), (b'DenseLayer', b'A fully connected layer.'), (b'Conv2DLayer', b'2D convolutional layer'), (b'MaxPool2DLayer', b'2D max-pooling layer'), (b'DropoutLayer', b'Dropout layer'), (b'LocalResponseNormalization2DLayer', b'Cross-channel Local Response Normalization for 2D feature maps.'), (b'NonlinearityLayer', b'A layer that just applies a nonlinearity.')]),
        ),
        migrations.AlterField(
            model_name='layer',
            name='ignore_border',
            field=models.BooleanField(default=True),
        ),
        migrations.AlterField(
            model_name='layer',
            name='nonlinearity',
            field=models.CharField(default=b'rectify', max_length=200, choices=[(b'None', b'Linear'), (b'sigmoid', b'Sigmoid activation function'), (b'softmax', b'Softmax activation function'), (b'tanh', b'Tanh activation function'), (b'rectify', b'Rectify activation function'), (b'leaky_rectify', b'Instance of LeakyRectify with leakiness = 0.01'), (b'very_leaky_rectify', b'Instance of LeakyRectify with leakiness = 1/3'), (b'softplus', b'Softplus activation function'), (b'linear', b'Linear activation function'), (b'identity', b'Linear activation function')]),
        ),
        migrations.AlterField(
            model_name='network',
            name='penalty',
            field=models.CharField(default=b'l1', max_length=50, choices=[(b'l1', b'L1'), (b'l2', b'L2')]),
        ),
    ]
