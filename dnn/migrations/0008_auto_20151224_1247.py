# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('dnn', '0007_auto_20151213_1912'),
    ]

    operations = [
        migrations.CreateModel(
            name='Result',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('action', models.CharField(default=b'validation', max_length=200, choices=[(b'training', b'Training'), (b'validation', b'Validation'), (b'prediction', b'Prediction')])),
                ('loaded', models.BooleanField(default=False)),
                ('error', models.FloatField(default=0)),
                ('accuracy', models.FloatField(default=0)),
                ('inputs', models.TextField(default=b'EMPTY')),
                ('prediction', models.TextField(default=b'EMPTY')),
                ('updated_at', models.DateTimeField()),
            ],
        ),
        migrations.AlterField(
            model_name='dataset',
            name='trim',
            field=models.CharField(default=b'', max_length=200, blank=True),
        ),
    ]
