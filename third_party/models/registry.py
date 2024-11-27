
import sys
import re
import fnmatch
from collections import defaultdict
from copy import deepcopy

__all__ = ['list_models', 'is_model', 'model_entrypoint', 'list_modules', 'is_model_in_modules',
           'is_model_default_key', 'has_model_default_key', 'get_model_default_value', 'is_model_pretrained']

_module_to_models = defaultdict(set)  
_model_to_module = {}  
_model_entrypoints = {}  
_model_has_pretrained = set() 
_model_default_cfgs = dict()  


def register_model(fn):
    mod = sys.modules[fn.__module__]
    module_name_split = fn.__module__.split('.')
    module_name = module_name_split[-1] if len(module_name_split) else ''

    model_name = fn.__name__
    if hasattr(mod, '__all__'):
        mod.__all__.append(model_name)
    else:
        mod.__all__ = [model_name]


    _model_entrypoints[model_name] = fn
    _model_to_module[model_name] = module_name
    _module_to_models[module_name].add(model_name)
    has_pretrained = False  
    if hasattr(mod, 'default_cfgs') and model_name in mod.default_cfgs:
        has_pretrained = 'url' in mod.default_cfgs[model_name] and 'http' in mod.default_cfgs[model_name]['url']
        _model_default_cfgs[model_name] = deepcopy(mod.default_cfgs[model_name])
    if has_pretrained:
        _model_has_pretrained.add(model_name)
    return fn


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def list_models(filter='', module='', pretrained=False, exclude_filters=''):
    
    if module:
        models = list(_module_to_models[module])
    else:
        models = _model_entrypoints.keys()
    if filter:
        models = fnmatch.filter(models, filter) 
    if exclude_filters:
        if not isinstance(exclude_filters, list):
            exclude_filters = [exclude_filters]
        for xf in exclude_filters:
            exclude_models = fnmatch.filter(models, xf) 
            if len(exclude_models):
                models = set(models).difference(exclude_models)
    if pretrained:
        models = _model_has_pretrained.intersection(models)
    return list(sorted(models, key=_natural_key))


def is_model(model_name):
    return model_name in _model_entrypoints


def model_entrypoint(model_name):
    return _model_entrypoints[model_name]


def list_modules():
    modules = _module_to_models.keys()
    return list(sorted(modules))


def is_model_in_modules(model_name, module_names):
    assert isinstance(module_names, (tuple, list, set))
    return any(model_name in _module_to_models[n] for n in module_names)


def has_model_default_key(model_name, cfg_key):
    if model_name in _model_default_cfgs and cfg_key in _model_default_cfgs[model_name]:
        return True
    return False


def is_model_default_key(model_name, cfg_key):
    if model_name in _model_default_cfgs and _model_default_cfgs[model_name].get(cfg_key, False):
        return True
    return False


def get_model_default_value(model_name, cfg_key):
    if model_name in _model_default_cfgs:
        return _model_default_cfgs[model_name].get(cfg_key, None)
    else:
        return None


def is_model_pretrained(model_name):
    return model_name in _model_has_pretrained
