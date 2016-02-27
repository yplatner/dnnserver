# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dnn', '0016_auto_20160124_1638'),
    ]

    operations = [
        migrations.AddField(
            model_name='dataset',
            name='pickle',
            field=models.BooleanField(default=False),
        ),
    ]
