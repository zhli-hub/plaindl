from collections import Iterable
from itertools import product


def get_analytical_jacobian(input, output):
    diff_input_list = list(iter_tensors(input))
    output_size = output.value.reshape(-1).shape
    jacobian = make_jacobian(input, *output_size)
    jacobian_reentrant = make_jacobian(input, *output_size)
    grad_output = Tensor(np.zeros(output.value.shape))
    flat_grad_output = Tensor(grad_output.value.reshape(-1))
    reentrant = True
    correct_grad_sizes = True

    flat_grad_output_numel = flat_grad_output.value.shape
    for i in range(*flat_grad_output_numel):
        flat_grad_output.value.fill(0)
        flat_grad_output.value[i] = 1
        for jacobian_c in (jacobian, jacobian_reentrant):
            grads_input = output.op.backward(output.input2values(), grad_output.value)

            for jacobian_x, d_x, x in zip(jacobian_c, grads_input, diff_input_list):
                jaconbian_numel = jacobian_x.value.reshape(-1).shape[0]
                if d_x is not None and d_x.size != x.value.size:
                    correct_grad_sizes = False
                elif jaconbian_numel != 0:
                    if d_x is None:
                        jacobian_x.value[:, i].fill(0)
                    else:
                        jacobian_x.value[:, i] = d_x.reshape(-1)

    for jacobian_x, jacobian_reentrant_x in zip(jacobian, jacobian_reentrant):
        if jacobian_x.value.reshape(-1)[0] != 0 and abs(jacobian_x.value - jacobian_reentrant_x.value).max() != 0:
            reentrant = False

    return jacobian, reentrant, correct_grad_sizes


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,



def _differentiable_outputs(x):
    return tuple(o for o in _as_tuple(x) if o.requires_grad)


def iter_tensors(x):
    if isinstance(x, Tensor):
        yield x
    elif isinstance(x, Iterable):
        for elem in x:
            for result in iter_tensors(elem):
                yield result

def make_jacobian(input, num_out):
    if isinstance(input, Tensor):
        # if not input.is_floating_point():
        #     return None
        # if not input.requires_grad:
        #     return None
        # return torch.zeros(input.nelement(), num_out, dtype=input.dtype)
        input_size = input.value.reshape(-1).shape[0]
        return Tensor(np.zeros((input_size, num_out)))
        # return torch.zeros(input.nelement(), num_out, dtype=input.dtype)
    elif isinstance(input, Iterable):
        jacobians = list(filter(
            lambda x: x is not None, (make_jacobian(elem, num_out) for elem in input)))
        if not jacobians:
            return None
        return type(input)(jacobians)
    else:
        return None


def get_numerical_jacobian(fn, input, target=None, eps=1e-3):
    if target is None:
        target = input

    output_size = fn(input).value.reshape(-1).shape
    jacobian = make_jacobian(target, *output_size)

    x_tensors = [t for t in iter_tensors(target)]
    j_tensors = [t for t in iter_tensors(jacobian)]

    for x_tensor, d_tensor in zip(x_tensors, j_tensors):

        for d_idx, x_idx in enumerate(product(*[range(m) for m in x_tensor.shape])):
            orig = x_tensor.value[x_idx]
            x_tensor.value[x_idx] = orig - eps
            outa = fn(input).value.copy()
            x_tensor.value[x_idx] = orig + eps
            outb = fn(input).value.copy()
            x_tensor.value[x_idx] = orig

            r = (outb - outa) / (2 * eps)
            d_tensor.value[d_idx] = r.reshape(-1)

    return jacobian



def gradcheck(func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True):

    tupled_inputs = _as_tuple(inputs)
    output = _as_tuple(func(*inputs))

    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)
        return False

    for i, o in enumerate(output):

        def fn(input):
            return _as_tuple(func(*input))[i]

        numerical = get_numerical_jacobian(fn, inputs, eps=eps)
        analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(tupled_inputs, o)

        if not correct_grad_sizes:

            return fail_test('Analytical gradient has incorrect size')

        for j, (a, n) in enumerate(zip(analytical, numerical)):
            if a.value.reshape(-1)[0] != 0 or n.value.reshape(-1)[0] != 0:
                if not (abs(a.value - n.value) <= (atol + rtol * abs(n.value))).all():
                    return fail_test('Jacobian mismatch for output %d with respect to input %d,\n'
                                     'numerical:%s\nanalytical:%s\n' % (i, j, n.value, a.value))

        if not reentrant:
            return fail_test('Backward is not reentrant, i.e., running backward with same '
                             'input and grad_output multiple times gives different values, '
                             'although analytical gradient matches numerical gradient')

    return True


if __name__ =='__main__':
    from plaindl.ops import MatMul
    from plaindl import Tensor
    import numpy as np

    a = Tensor(np.random.randn(10, 3))
    b = Tensor(np.random.randn(3, 2))
    matmul = MatMul()

    test = gradcheck(matmul, (a, b), eps=1e-8)
    print(test)
