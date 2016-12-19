import numpy

def attrs_to_dict(obj, prefix):
    if not hasattr(obj, 'attrs'):
        return {}

    d = {}
    for key, value in obj.attrs.items():
        d[prefix + key] = value
    return d

import json
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return {'__complex__': [obj.real, obj.imag ]}

        elif isinstance(obj, numpy.ndarray):
            value = obj
            dtype = obj.dtype
            d = {
                '__dtype__' :
                    dtype.str if dtype.names is None else dtype.descr,
                '__shape__' : value.shape,
                '__data__': value.tolist(),
            }
            return d
        return json.JSONEncoder.default(self, obj)

class JSONDecoder(json.JSONDecoder):
    @staticmethod
    def hook(value):
        def fixdtype(dtype):
            if isinstance(dtype, list):
                true_dtype = []
                for field in dtype:
                    if len(field) == 3:
                        true_dtype.append((str(field[0]), str(field[1]), field[2]))
                    if len(field) == 2:
                        true_dtype.append((str(field[0]), str(field[1])))
                return true_dtype
            return dtype

        def fixdata(data, N, dtype):
            if not isinstance(dtype, list):
                return data

            # for structured array,
            # the last dimension shall be a tuple
            if N > 0:
                return [fixdata(i, N - 1, dtype) for i in data]
            else:
                assert len(data) == len(dtype)
                return tuple(data)

        if '__dtype__' in value:
            dtype = fixdtype(value['__dtype__'])
            shape = value['__shape__']
            a = fixdata(value['__data__'], len(shape), dtype)
            return numpy.array(a, dtype=dtype)

        if '__complex__' in value:
            real, imag = value['__complex__']
            return real + 1j * imag

        return value

    def __init__(self, *args, **kwargs):
        kwargs['object_hook'] = JSONDecoder.hook
        json.JSONDecoder.__init__(self, *args, **kwargs)
