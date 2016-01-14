# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dnn', '0010_auto_20151227_1458'),
    ]

    operations = [
        migrations.AddField(
            model_name='dataset',
            name='grayscale',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='dataset',
            name='image',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='zipped',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='result',
            name='classification',
            field=models.TextField(default=b''),
        ),
    ]
