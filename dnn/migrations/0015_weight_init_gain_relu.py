# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dnn', '0014_auto_20160115_1433'),
    ]

    operations = [
        migrations.AddField(
            model_name='weight',
            name='init_gain_relu',
            field=models.BooleanField(default=False),
        ),
    ]
