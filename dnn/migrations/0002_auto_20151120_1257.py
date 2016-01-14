# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('dnn', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Update',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('strategy', models.CharField(default=b'sgd', max_length=50, choices=[(b'sgd', b'Stochastic Gradient Descent'), (b'momentum', b'Stochastic Gradient Descent with Momentum'), (b'nesterov_momentum', b'Stochastic Gradient Descent with Noesterov Momentum'), (b'adagrad', b'Adagrad'), (b'rmsprop', b'RMSProp'), (b'adadelta', b'Adadelta'), (b'adam', b'Adam')])),
                ('learning_rate', models.FloatField(default=0)),
                ('momentum', models.FloatField(default=0)),
                ('rho', models.FloatField(default=0)),
                ('epsilon', models.FloatField(default=0)),
                ('beta1', models.FloatField(default=0)),
                ('beta2', models.FloatField(default=0)),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(default=b'', max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='Weight',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('init', models.CharField(default=b'Constant', max_length=200, choices=[(b'Constant', b'Initialize weights with constant value.'), (b'Normal', b'Sample initial weights from the Gaussian distribution.'), (b'Uniform', b'Sample initial weights from the uniform distribution.'), (b'Glorot', b'Glorot weight initialization.'), (b'GlorotNormal', b'Glorot with weights sampled from the Normal distribution.'), (b'GlorotUniform', b'Glorot with weights sampled from the Uniform distribution.'), (b'He', b'He weight initialization.'), (b'HeNormal', b'He initializer with weights sampled from the Normal distribution.'), (b'HeUniform', b'He initializer with weights sampled from the Uniform distribution.'), (b'Orthogonal', b'Intialize weights as Orthogonal matrix.'), (b'Sparse', b'Initialize weights as sparse matrix.')])),
                ('init_val', models.FloatField(default=0)),
                ('init_std', models.FloatField(default=0)),
                ('init_mean', models.FloatField(default=0)),
                ('init_range', models.FloatField(default=0)),
                ('init_range_high', models.FloatField(default=0)),
                ('init_gain', models.FloatField(default=0)),
                ('init_c01b', models.BooleanField(default=False)),
                ('init_sparsity', models.FloatField(default=0)),
                ('generated', models.BooleanField(default=False)),
                ('value', models.TextField(default=b'EMPTY')),
            ],
        ),
        migrations.RenameField(
            model_name='layer',
            old_name='some_num',
            new_name='level',
        ),
        migrations.RemoveField(
            model_name='layer',
            name='name',
        ),
        migrations.AddField(
            model_name='layer',
            name='base_type',
            field=models.CharField(default=b'InputLayer', max_length=200, choices=[(b'InputLayer', b'A network input layer'), (b'DenseLayer', b'A fully connected layer.'), (b'Conv2DLayer', b'2D convolutional layer'), (b'MaxPool2DLayer', b'2D max-pooling layer'), (b'DropoutLayer', b'Dropout layer')]),
        ),
        migrations.AddField(
            model_name='layer',
            name='filter_size',
            field=models.CommaSeparatedIntegerField(default=0, max_length=2),
        ),
        migrations.AddField(
            model_name='layer',
            name='ignore_border',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='layer',
            name='nonlinearity',
            field=models.CharField(default=b'sigmoid', max_length=200, choices=[(b'sigmoid', b'Sigmoid activation function'), (b'softmax', b'Softmax activation function'), (b'tanh', b'Tanh activation function'), (b'rectify', b'Rectify activation function'), (b'leaky_rectify', b'Instance of LeakyRectify with leakiness = 0.01'), (b'very_leaky_rectify', b'Instance of LeakyRectify with leakiness = 1/3'), (b'softplus', b'Softplus activation function'), (b'linear', b'Linear activation function'), (b'identity', b'Linear activation function')]),
        ),
        migrations.AddField(
            model_name='layer',
            name='num_filters',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='layer',
            name='num_units',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='layer',
            name='p',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='layer',
            name='pad',
            field=models.CommaSeparatedIntegerField(default=0, max_length=2),
        ),
        migrations.AddField(
            model_name='layer',
            name='pool_size',
            field=models.CommaSeparatedIntegerField(default=0, max_length=2),
        ),
        migrations.AddField(
            model_name='layer',
            name='rescale',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='layer',
            name='shape',
            field=models.CommaSeparatedIntegerField(default=0, max_length=200),
        ),
        migrations.AddField(
            model_name='layer',
            name='stride',
            field=models.CommaSeparatedIntegerField(default=0, max_length=2),
        ),
        migrations.AddField(
            model_name='layer',
            name='untie_biases',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='network',
            name='loss_delta',
            field=models.FloatField(default=1),
        ),
        migrations.AddField(
            model_name='network',
            name='loss_objective',
            field=models.CharField(default=b'binary_crossentropy', max_length=50, choices=[(b'binary_crossentropy', b'Binary Crossentropy'), (b'categorical_crossentropy', b'Categorical Crossentropy'), (b'squared_error', b'Squared Error'), (b'binary_hinge_loss', b'Binary Hinge Loss'), (b'multiclass_hinge_loss', b'Multiclass Hinge Loss')]),
        ),
        migrations.AlterField(
            model_name='layer',
            name='network',
            field=models.ForeignKey(on_delete=django.db.models.deletion.SET_NULL, blank=True, to='dnn.Network', null=True),
        ),
        migrations.AlterField(
            model_name='network',
            name='creation_date',
            field=models.DateTimeField(null=True),
        ),
        migrations.AlterField(
            model_name='network',
            name='name',
            field=models.CharField(default=b'', max_length=200),
        ),
        migrations.AddField(
            model_name='layer',
            name='b',
            field=models.ForeignKey(related_name='b', on_delete=django.db.models.deletion.SET_NULL, blank=True, to='dnn.Weight', null=True),
        ),
        migrations.AddField(
            model_name='layer',
            name='w',
            field=models.ForeignKey(related_name='w', on_delete=django.db.models.deletion.SET_NULL, blank=True, to='dnn.Weight', null=True),
        ),
        migrations.AddField(
            model_name='network',
            name='update',
            field=models.ForeignKey(on_delete=django.db.models.deletion.SET_NULL, blank=True, to='dnn.Update', null=True),
        ),
        migrations.AddField(
            model_name='network',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.SET_NULL, blank=True, to='dnn.User', null=True),
        ),
    ]
