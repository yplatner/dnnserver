from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$',                                                                                                                          views.index,            name='index'),
    url(r'^reset/(?P<network_id>[0-9]+)/$',                                                                                             views.network_reset,    name='network_reset'),
    url(r'^network/(?P<network_id>[0-9]+)/$',                                                                                           views.network_edit,     name='network_edit'),
    url(r'^updates/$',                                                                                                                  views.updates,          name='updates'),
    url(r'^update/(?P<update_id>[0-9]+)/$',                                                                                             views.update_edit,      name='update_edit'),
    url(r'^weights/$',                                                                                                                  views.weights,          name='weights'),
    url(r'^weight/(?P<weight_id>[0-9]+)/$',                                                                                             views.weight_edit,      name='weight_edit'),
    url(r'^datasets/$',                                                                                                                 views.datasets,         name='datasets'),
    url(r'^dataset/(?P<dataset_id>[0-9]+)/$',                                                                                           views.dataset_edit,     name='dataset_edit'),
    #url(r'^load_all/$',                                                                                                                views.load_all,         name='load all'),
    url(r'^layer/(?P<layer_id>[0-9]+)/$',                                                                                               views.layer_edit,       name='layer_edit'),
    url(r'^action/(?P<network_id>[0-9]+)/$',                                                                                            views.actions,          name='actions'),
    url(r'^(?P<network_id>[0-9]+)/$',                                                                                                   views.layers,           name='layers'),
    url(r'^predict/(?P<network_id>[0-9]+)/(?P<inputs_dataset_id>[0-9]+)/$',                                                             views.do_predict,       name='predict'),
    url(r'^train/(?P<network_id>[0-9]+)/(?P<inputs_dataset_id>[0-9]+)/(?P<targets_dataset_id>[0-9]+)/(?P<batch_size>[0-9]+)/(?P<epochs>[0-9]+)/$',         views.do_train,         name='train'),
    url(r'^validate/(?P<network_id>[0-9]+)/(?P<inputs_dataset_id>[0-9]+)/(?P<targets_dataset_id>[0-9]+)/(?P<batch_size>[0-9]+)/(?P<epochs>[0-9]+)/$',      views.do_validate,      name='validate'),
    url(r'^result/(?P<result_id>[0-9]+)/$',                                                                                             views.display_result,   name='result'),
    #url(r'^mnist/$',                                                                                                                    views.do_run_mnist,     name='run mnist'),
]