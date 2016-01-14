# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dnn', '0004_network_user'),
    ]

    operations = [
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(default=b'', max_length=200)),
                ('source', models.CharField(default=b'', max_length=200)),
                ('filename', models.CharField(default=b'', max_length=200)),
                ('zipped', models.BooleanField(default=True)),
                ('data_type', models.CharField(default=b'uint8', max_length=200, choices=[(b'uint8', b'integer BYTE'), (b'float32', b'float DWORD')])),
                ('offset', models.IntegerField(default=0)),
                ('reshape', models.BooleanField(default=False)),
                ('shape_channels', models.IntegerField(default=1)),
                ('shape_x', models.IntegerField(default=1)),
                ('shape_y', models.IntegerField(default=1)),
                ('divide', models.BooleanField(default=False)),
                ('divide_type', models.CharField(default=b'float32', max_length=200, choices=[(b'uint8', b'integer BYTE'), (b'float32', b'float DWORD')])),
                ('divide_num', models.IntegerField(default=1)),
                ('trim', models.CharField(default=b':-0', max_length=200)),
                ('loaded', models.BooleanField(default=False)),
                ('value', models.TextField(default=b'EMPTY')),
            ],
        ),
        migrations.RemoveField(
            model_name='layer',
            name='filter_size',
        ),
        migrations.RemoveField(
            model_name='layer',
            name='pool_size',
        ),
        migrations.RemoveField(
            model_name='layer',
            name='shape',
        ),
        migrations.RemoveField(
            model_name='layer',
            name='stride',
        ),
        migrations.RemoveField(
            model_name='weight',
            name='generated',
        ),
        migrations.AddField(
            model_name='layer',
            name='filter_size_x',
            field=models.IntegerField(default=1),
        ),
        migrations.AddField(
            model_name='layer',
            name='filter_size_y',
            field=models.IntegerField(default=1),
        ),
        migrations.AddField(
            model_name='layer',
            name='pool_size_x',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='layer',
            name='pool_size_y',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='layer',
            name='shape_channels',
            field=models.IntegerField(default=1),
        ),
        migrations.AddField(
            model_name='layer',
            name='shape_x',
            field=models.IntegerField(default=1),
        ),
        migrations.AddField(
            model_name='layer',
            name='shape_y',
            field=models.IntegerField(default=1),
        ),
        migrations.AddField(
            model_name='layer',
            name='stride_x',
            field=models.IntegerField(default=1),
        ),
        migrations.AddField(
            model_name='layer',
            name='stride_y',
            field=models.IntegerField(default=1),
        ),
        migrations.AddField(
            model_name='network',
            name='generated',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='update',
            name='name',
            field=models.CharField(default=b'', max_length=200),
        ),
        migrations.AlterField(
            model_name='layer',
            name='pad',
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='weight',
            name='init',
            field=models.CharField(default=b'Constant', max_length=200, choices=[(b'Constant', b'Initialize weights with constant value.'), (b'Normal', b'Sample initial weights from the Gaussian distribution.'), (b'Uniform', b'Sample initial weights from the uniform distribution.'), (b'GlorotNormal', b'Glorot with weights sampled from the Normal distribution.'), (b'GlorotUniform', b'Glorot with weights sampled from the Uniform distribution.'), (b'HeNormal', b'He initializer with weights sampled from the Normal distribution.'), (b'HeUniform', b'He initializer with weights sampled from the Uniform distribution.'), (b'Orthogonal', b'Intialize weights as Orthogonal matrix.'), (b'Sparse', b'Initialize weights as sparse matrix.')]),
        ),
    ]
