# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dnn', '0005_auto_20151206_1621'),
    ]

    operations = [
        migrations.RenameField(
            model_name='layer',
            old_name='pad',
            new_name='pad_x',
        ),
        migrations.AddField(
            model_name='layer',
            name='pad_y',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='layer',
            name='nonlinearity',
            field=models.CharField(default=b'rectify', max_length=200, choices=[(b'sigmoid', b'Sigmoid activation function'), (b'softmax', b'Softmax activation function'), (b'tanh', b'Tanh activation function'), (b'rectify', b'Rectify activation function'), (b'leaky_rectify', b'Instance of LeakyRectify with leakiness = 0.01'), (b'very_leaky_rectify', b'Instance of LeakyRectify with leakiness = 1/3'), (b'softplus', b'Softplus activation function'), (b'linear', b'Linear activation function'), (b'identity', b'Linear activation function')]),
        ),
        migrations.AlterField(
            model_name='update',
            name='beta1',
            field=models.FloatField(default=0.9),
        ),
        migrations.AlterField(
            model_name='update',
            name='beta2',
            field=models.FloatField(default=0.999),
        ),
        migrations.AlterField(
            model_name='update',
            name='epsilon',
            field=models.FloatField(default=1e-06),
        ),
        migrations.AlterField(
            model_name='update',
            name='learning_rate',
            field=models.FloatField(default=1),
        ),
        migrations.AlterField(
            model_name='update',
            name='momentum',
            field=models.FloatField(default=0.9),
        ),
        migrations.AlterField(
            model_name='update',
            name='rho',
            field=models.FloatField(default=0.9),
        ),
        migrations.AlterField(
            model_name='weight',
            name='init',
            field=models.CharField(default=b'GlorotUniform', max_length=200, choices=[(b'Constant', b'Initialize weights with constant value.'), (b'Normal', b'Sample initial weights from the Gaussian distribution.'), (b'Uniform', b'Sample initial weights from the uniform distribution.'), (b'GlorotNormal', b'Glorot with weights sampled from the Normal distribution.'), (b'GlorotUniform', b'Glorot with weights sampled from the Uniform distribution.'), (b'HeNormal', b'He initializer with weights sampled from the Normal distribution.'), (b'HeUniform', b'He initializer with weights sampled from the Uniform distribution.'), (b'Orthogonal', b'Intialize weights as Orthogonal matrix.'), (b'Sparse', b'Initialize weights as sparse matrix.')]),
        ),
        migrations.AlterField(
            model_name='weight',
            name='init_gain',
            field=models.FloatField(default=1.0),
        ),
        migrations.AlterField(
            model_name='weight',
            name='init_range',
            field=models.FloatField(default=-0.01),
        ),
        migrations.AlterField(
            model_name='weight',
            name='init_range_high',
            field=models.FloatField(default=0.01),
        ),
        migrations.AlterField(
            model_name='weight',
            name='init_sparsity',
            field=models.FloatField(default=0.1),
        ),
        migrations.AlterField(
            model_name='weight',
            name='init_std',
            field=models.FloatField(default=0.01),
        ),
    ]
