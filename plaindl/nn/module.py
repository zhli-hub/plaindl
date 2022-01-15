from collections import OrderedDict
from plaindl.tensor import Tensor
from plaindl.ops.ops import Op
import pickle


class Module(object):

    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self._buffers = OrderedDict()

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input):
        return self.forward(*input)

    def error_init(self, sth):
        raise RuntimeError(
            "cannot assign '{}' before Module.__init__() call".format(sth)
        )

    def save_state_dict(self, path):
        with open(path, "wb") as f:

            pickle.dump(self.state_dict(), f)

    def get_local_state_name(self):

        params_name = [name for name in self._parameters.keys()]
        buffers_name = [name for name in self._buffers.keys()]
        return params_name + buffers_name

    def load_local_state_dict(self,state_dict, prefix=''):
        local_state_name = self.get_local_state_name()
        for name in local_state_name:
            if name in self._parameters:
                self._parameters[name].value = state_dict[prefix + name]
            else:
                self._buffers[name].value = state_dict[prefix + name]
        for name, module in self._modules.items():
            module.load_local_state_dict(state_dict, prefix+name+'.')

    def load_state_dict(self, path):
        with open(path, "rb") as f:
            state_dict = pickle.load(f)

        # print(state_dict)
        self.load_local_state_dict(state_dict)

    def register_buffer(self, key, value):
        pass

    def register_parameter(self, key, value):
        pass

    def add_module(self, name, module):
        modules = self.__dict__.get("_modules")
        if modules is None:
            self.error_init("module")
        if isinstance(module, Op):
            return
        self._modules[name] = module


    def set_buffers_dict(self, buffers):
        buffers = self.__dict__.get('_buffers')
        if buffers is None:
            self.error_init("buffers")
        if type(buffers).__name__ is not 'dict':
            raise RuntimeError(
                "the type of the input must be the dict"
            )
        for k, v in buffers.items():
            if not isinstance((v, Tensor)):
                if v is not None:
                    raise RuntimeError("the buffer must be a Tensor or None!")
            buffers[k] = None if v is None else v

    def set_parameters_dict(self, params):
        parameters = self.__dict__.get('_parameters')
        if parameters is None:
            self.error_init("parameters")

        assert type(params).__name__ == 'dict', "the type of the input must be dict!"
        for k, v in params.items():
            if not isinstance(v, Tensor):
                if v is not None:
                    raise RuntimeError(
                        "the parameter must be a Tensor or None!"
                    )
            parameters[k] = None if v is None else v

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))


    def __setattr__(self, key, value):
        # make sure that the attribute just belong to one type dict(parameter, module or buffer)
        def remove_from(*dicts):
            for d in dicts:
                if key in d:
                    del d[key]
        modules = self.__dict__.get('_modules')
        if isinstance(value, Module):
            if modules is None:
                self.error_init("modules")
            remove_from(self.__dict__, self._parameters, self._buffers)
            modules[key] = value
        elif modules is not None and key in modules:
            if value is not None:
                raise TypeError("cannot assign '{}' as child module '{}' "
                                "(plaindl.nn.Module or None expected)"
                                .format(type(value).__name__, key))
            modules[key] = value
        else:
            buffers = self.__dict__.get('_buffers')
            if buffers is not None and key in buffers:
                if value is not None and not isinstance(value, Tensor):
                    raise TypeError("cannot assign '{}' as buffer '{}' "
                                    "(plaindl.Tensor or None expected)"
                                    .format(type(value).__name__, key))
                buffers[key] = value
            else:
                object.__setattr__(self, key, value)


    def state_dict(self, destination=None, prefix=''):
        if destination is None:
            destination = OrderedDict()

        params = self.__dict__.get("_parameters")
        if params is None:
            self.error_init("parameters")
        else:
            for name, param in self._parameters.items():
                destination[prefix+name] = param.value if param is not None else None

        buffers = self.__dict__.get("_buffers")
        if buffers is None:
            self.error_init("buffers")
        else:
            for name, buffer in self._buffers.items():
                destination[prefix+name] = buffer.value

        modules = self.__dict__.get("_modules")
        if modules is None:
            self.error_init("modules")
        else:
            for name, module in self._modules.items():
                module.state_dict(destination, prefix+name+'.')

        return destination
