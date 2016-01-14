# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dnn', '0008_auto_20151224_1247'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='dataset',
            name='value',
        ),
        migrations.RemoveField(
            model_name='network',
            name='weights',
        ),
        migrations.RemoveField(
            model_name='result',
            name='inputs',
        ),
    ]
