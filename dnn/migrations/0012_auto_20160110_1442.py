# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dnn', '0011_auto_20151231_1318'),
    ]

    operations = [
        migrations.AddField(
            model_name='weight',
            name='name',
            field=models.CharField(default=b'', max_length=200),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='data_type',
            field=models.CharField(default=b'uint8', max_length=200, choices=[(b'uint8', b'integer BYTE'), (b'uint16', b'integer WORD'), (b'uint32', b'integer DWORD')]),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='divide_num',
            field=models.IntegerField(default=256),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='divide_type',
            field=models.CharField(default=b'float32', max_length=200, choices=[(b'float32', b'float DWORD')]),
        ),
    ]
