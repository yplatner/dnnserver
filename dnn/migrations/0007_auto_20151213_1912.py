# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dnn', '0006_auto_20151206_1701'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='weight',
            name='value',
        ),
        migrations.AddField(
            model_name='network',
            name='weights',
            field=models.TextField(default=b'EMPTY'),
        ),
    ]
