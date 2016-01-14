# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dnn', '0002_auto_20151120_1257'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='network',
            name='user',
        ),
        migrations.DeleteModel(
            name='User',
        ),
    ]
