# Copyright (c) 2023, DeepLink.
import io
from typing import Callable, List
import torch
from utils.stdout_redirector import stdout_redirector
from utils.local_eviron import local_eviron
from utils.test_in_subprocess import run_individual_test_cases


def test_fallback(
    op_names: List[str],
    diopi_protos: List[str],
    test_fn: Callable[[], None],
    extra_check_str_in_output: List[str] = [],
    skip_devs: List[str] = [],
) -> None:
    captured = io.BytesIO()
    with stdout_redirector(captured):
        with local_eviron(
            {
                "DIPU_FORCE_FALLBACK_OPS_LIST": ",".join(op_names),
                "DIPU_DUMP_OP_ARGS": "1",
                "DIPU_LOG_FALLBACK_INFO": "1",
            }
        ):
            import torch_dipu

            if torch_dipu.dipu.vendor_type in skip_devs:
                return

            test_fn()
    output = captured.getvalue().decode()
    print(output, end="", flush=True)
    assert all(
        f"force fallback has been set, {name} will be fallback to cpu" in output
        for name in op_names
    )
    assert all(item not in output for item in diopi_protos)
    if extra_check_str_in_output is not None:
        assert all(item in output for item in extra_check_str_in_output)


def _test_dipu_fallback():
    def fn():
        x = torch.randn(3, 4).cuda()
        _ = x + x
        _ = x - x

    test_fallback(
        ["add.out", "sub.out"], ["diopiAdd", "diopiSub"], fn, ["dipu_fallback"]
    )


def _test_cpu_fallback():
    def fn():
        device = "cuda"
        m = torch.nn.BatchNorm2d(100, affine=False).to(device)
        input = torch.randn(20, 100, 35, 45).to(device)
        m(input)

    test_fallback(
        ["native_batch_norm"],
        ["diopiBatchNorm"],
        fn,
        ["cpu_fallback:\taten::native_batch_norm", "dipu_fallback"],
    )


def _test_dipu_index_put_impl_fallback():
    def fn():
        dipu_tensor = torch.tensor([1, 2, 3, 4, 5]).cuda()
        indices = torch.tensor([1, 3]).cuda()
        values = torch.tensor([10, 40]).cuda()
        torch._index_put_impl_(dipu_tensor, (indices,), values, accumulate=False)

        tensor = dipu_tensor.cpu()
        indices = indices.cpu()
        values = values.cpu()
        torch._index_put_impl_(tensor, (indices,), values, accumulate=False)

        assert torch.allclose(tensor, dipu_tensor.cpu())

    test_fallback(
        ["_index_put_impl_"],
        ["diopiIndexPut"],
        fn,
        ["custom fallback to cpu, name=_index_put_impl_"],
    )


def _test_dipu_copy_fallback_():
    def fn():
        source_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        target_dipu = torch.zeros_like(source_tensor).cuda()
        target_dipu.copy_(source_tensor)

        source_tensor = source_tensor.cpu()
        target_tensor = torch.zeros_like(source_tensor)
        target_tensor.copy_(source_tensor)

        assert torch.allclose(target_tensor, target_dipu.cpu())

        # detect the occasional occurrence problem of copy_
        for i in range(100):
            N, C, H, W = 1, 3, 2, 2
            x = torch.randn(N, C, H, W)
            x1 = x.to(memory_format=torch.channels_last).cuda()
            x2stride = list(x1.stride())
            x2stride[0] = 4
            x2 = x.as_strided(x1.size(), tuple(x2stride)).cuda()
            y = torch.randn(N, C, H, W).cuda()
            y.copy_(x2)

        assert torch.allclose(y, x2)

    test_fallback(
        ["copy_"],
        ["diopiCopyInp"],
        fn,
        ["custom fallback to dipu copy, name=copy_"],
    )


def _test_dipu_convolution_backward_overrideable_fallback():
    def fn():
        torch.manual_seed(42)
        device = torch.device("dipu")
        m = torch.nn.Conv2d(2, 3, 3, stride=2).to(device)
        m.weight = torch.nn.Parameter(torch.ones_like(m.weight))
        m.bias = torch.nn.Parameter(torch.ones_like(m.bias))
        input_dipu = torch.randn(2, 2, 5, 5).to(device).requires_grad_(True)
        output_dipu = m(input_dipu)
        output_dipu.backward(torch.ones_like(output_dipu))

        torch.manual_seed(42)
        m = torch.nn.Conv2d(2, 3, 3, stride=2)
        m.weight = torch.nn.Parameter(torch.ones_like(m.weight))
        m.bias = torch.nn.Parameter(torch.ones_like(m.bias))
        input_cpu = torch.randn(2, 2, 5, 5, requires_grad=True)
        output_cpu = m(input_cpu)
        output_cpu.backward(torch.ones_like(output_cpu))

        assert torch.allclose(output_dipu.cpu(), output_cpu)
        assert torch.allclose(input_dipu.grad.cpu(), input_cpu.grad)

    test_fallback(
        ["convolution_backward_overrideable"],
        ["diopiConvolution2dBackward"],
        fn,
        ["custom fallback to cpu, name=convolution_backward_overrideable"],
    )


def _test_dipu_convolution_overrideable_fallback():
    def fn():
        m = torch.nn.Conv2d(2, 3, 3, stride=2).cuda()
        m.weight = torch.nn.Parameter(torch.ones_like(m.weight))
        m.bias = torch.nn.Parameter(torch.ones_like(m.bias))
        input_dipu = torch.randn(2, 2, 5, 5).cuda()
        output_dipu = m(input_dipu)

        m = m.cpu()
        m.weight = torch.nn.Parameter(torch.ones_like(m.weight))
        m.bias = torch.nn.Parameter(torch.ones_like(m.bias))
        input_cpu = input_dipu.cpu()
        output_cpu = m(input_cpu)

        assert torch.allclose(output_dipu.cpu(), output_cpu)

    test_fallback(
        ["convolution_overrideable"],
        ["diopiConvolution2d"],
        fn,
        ["custom fallback to cpu, name=convolution_overrideable"],
    )


def _test_dipu_silu_fallback():
    def fn():
        m = torch.nn.SiLU().cuda()
        input_dipu = torch.tensor([1.0, 2.0, 3.0, 4.0]).cuda()
        out_dipu = m(input_dipu)

        m = m.cpu()
        input_cpu = input_dipu.cpu()
        out_cpu = m(input_cpu)

        assert torch.allclose(out_dipu.cpu(), out_cpu)

    test_fallback(
        ["silu.out"],
        ["diopiSilu"],
        fn,
        ["custom fallback to cpu, name=silu_out"],
    )


def _test_dipu_linear_backward_fallback():
    def fn():
        l_cpu = torch.nn.Linear(2, 4)
        l_dev = torch.nn.Linear(2, 4)
        l_dev.weight.data.copy_(l_cpu.weight)
        l_dev.bias.data.copy_(l_cpu.bias)
        l_dev = l_dev.cuda()

        x_cpu = torch.rand(3, 2)
        x_dev = x_cpu.cuda().requires_grad_()
        x_cpu.requires_grad_()

        y_cpu = l_cpu(x_cpu)
        y_dev = l_dev(x_dev)

        grad_cpu = torch.rand_like(y_cpu)
        grad_dev = grad_cpu.cuda()

        y_cpu.backward(grad_cpu)
        y_dev.backward(grad_dev)

        assert x_dev.grad.device == x_dev.device
        assert l_dev.weight.grad.device == x_dev.device
        assert l_dev.bias.grad.device == x_dev.device

        assert torch.allclose(x_dev.grad.cpu(), x_cpu.grad)
        assert torch.allclose(l_dev.weight.grad.cpu(), l_cpu.weight.grad)
        assert torch.allclose(l_dev.bias.grad.cpu(), l_cpu.bias.grad)
        assert x_dev.grad.shape == x_cpu.grad.shape
        assert l_dev.weight.grad.shape == l_cpu.weight.grad.shape
        assert l_dev.bias.grad.shape == l_cpu.bias.grad.shape

    test_fallback(
        ["linear_backward"],
        ["diopiLinearBackward"],
        fn,
        ["custom fallback to cpu, name=linear_backward"],
        # linear_backward isn't registered on cuda
        ["CUDA"],
    )


if __name__ == "__main__":
    run_individual_test_cases(
        [
            _test_dipu_fallback,
            _test_cpu_fallback,
            _test_dipu_index_put_impl_fallback,
            _test_dipu_copy_fallback_,
            _test_dipu_convolution_backward_overrideable_fallback,
            _test_dipu_convolution_overrideable_fallback,
            _test_dipu_silu_fallback,
            _test_dipu_linear_backward_fallback,
        ],
        in_parallel=True,
    )
