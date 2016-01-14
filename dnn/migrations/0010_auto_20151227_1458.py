# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dnn', '0009_auto_20151224_1840'),
    ]

    operations = [
        migrations.RenameField(
            model_name='result',
            old_name='prediction',
            new_name='classification',
        ),
    ]
