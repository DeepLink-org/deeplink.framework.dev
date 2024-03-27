# Copyright (c) 2023, DeepLink.
import time
import torch
from typing import cast


def test_empty_tensor():
    import torch_dipu

    # dev1 = torch_dipu.diputype
    # device = torch.device(dev1)

    device = torch.device("dipu")
    data1 = torch.empty(2, 4)
    input1 = data1.to(device)

    input2 = torch.empty(2, 4).to(device)
    res1 = input1.add(input2)
    res2 = torch.reshape(res1, (-1,))
    res3 = res2.to("cpu")
    print(res3)


def testdevice():
    import torch_dipu

    device = torch.device(0)
    device = torch.device("cuda:0")
    device = torch.device("cuda", index=0)
    device = torch.device("cuda", 0)
    # device= torch.device(0)
    device = torch.device(index=0, type="cuda")
    ret1 = isinstance(device, torch.device)
    input0 = torch.ones((2, 4))
    input0.to(0)
    input1 = torch.ones((2, 4), device=device)
    print(input1)


def testDevice1():
    import torch_dipu
    from torch_dipu import dipu

    ret1 = dipu.device_count()
    ret1 = dipu.current_device()
    ret1 = dipu.set_device("cuda:2")
    ret1 = dipu.current_device()
    print(ret1)


def testDevice2():
    import torch_dipu
    from torch_dipu import dipu

    device = "cuda"
    in0 = torch.range(1, 12).reshape(1, 3, 2, 2)
    in1 = in0.to(device)
    print(in1)

    with dipu.device(torch.device(2)) as dev1:
        in1 = in0.to(device)
        print(torch.sum(in1))
        dipu.synchronize()
        ret1 = dipu.current_device()
        print(ret1)
        # need implment device guard in op
        # in1 = torch.range(1, 12).reshape(1, 3, 2, 2).to("cuda:0")
        # print(in1)
    in1 = torch.range(1, 12).reshape(1, 3, 2, 2).to(device)
    print(in1)


def testStream():
    import torch_dipu
    from torch import cuda

    st0 = cuda.Stream(0)
    res1 = st0.priority_range()
    res1 = st0.priority
    res1 = st0.synchronize()
    res1 = st0.query()
    res1 = st0.stream_id
    res1 = st0.dipu_stream
    res1 = st0.device_index
    res1 = st0.device_type
    res1 = st0.device
    res1 = st0._as_parameter_
    cuda.set_stream(st0)
    st1 = cuda.default_stream()
    res2 = st1.query()
    res2 = st1._as_parameter_
    st2 = cuda.current_stream(0)

    with cuda.StreamContext(st2) as s:
        print("in cur ctx")

    # st1 = dipu.current_stream()
    print(st1)


def test_record_stream():
    stream = torch.cuda.Stream()
    s1 = torch.cuda.current_stream()
    src1 = torch.ones((2, 4)).to(0)
    src1.record_stream(cast(torch._C.Stream, stream))
    src1.record_stream(s1)


def testevent():
    import torch_dipu
    from torch import cuda

    st0 = cuda.Stream(0)
    ev1 = cuda.Event()
    ev1.record(st0)
    time.sleep(1)
    ev2 = cuda.Event()
    ev1.synchronize()
    ev1.wait(st0)
    ev2.record(st0)
    # sync before call elapse
    ev2.synchronize()
    elapsed = ev1.elapsed_time(ev2)
    print(elapsed)


def testDeviceProperties():
    print("device properties: ", torch.cuda.get_device_properties(0))
    print("device capability: ", torch.cuda.get_device_capability(0))
    print("device name: ", torch.cuda.get_device_name(0))


def test_mem_get_info():
    import torch_dipu
    from torch import cuda

    minfo = cuda.mem_get_info()
    d1 = torch.ones((1024, 1024 * 30), device="cuda")
    minfo = cuda.mem_get_info()
    print(minfo)


def test_type():
    import torch_dipu

    dev1 = "cuda"
    s1 = torch.arange(1, 12, dtype=torch.float32, device=dev1)
    s2 = torch.Tensor.new(s1)
    s3 = s2.new((4, 3))
    assert s3.shape == torch.Size((2,))
    s4 = s3.new(size=(4, 3))
    res = isinstance(s4, torch.cuda.FloatTensor)
    assert res == True

    assert dev1 in s1.type()
    assert s1.device.type == dev1


def test_device_copy():
    import torch_dipu

    dev2 = "cuda:2"
    t1 = torch.randn((2, 2), dtype=torch.float64, device=dev2)
    t1 = torch.zero_(t1)

    tsrc = torch.ones((2, 2), dtype=torch.float64, device="cuda")
    print(tsrc)

    t1.copy_(tsrc)
    # cpu fallback func not support device guard now! (enhance?)
    print(t1.cpu())

    tc1 = torch.randn((2, 2), dtype=torch.float64)
    tc1.copy_(t1)
    print(tc1)

    t0 = torch.tensor([980], dtype=torch.int64).cuda()
    t2 = t0.expand(2)
    t2.to(torch.float)


def test_complex_type():
    import torch_dipu
    from torch_dipu import dipu
    import numpy as np

    dev2 = "cuda:0"

    # manually set device
    dipu.set_device(0)
    abs = torch.tensor((1, 2), dtype=torch.float64, device=dev2)
    angle = torch.tensor([np.pi / 2, 5 * np.pi / 4], dtype=torch.float64, device=dev2)
    # z1 = torch.polar(abs, angle)
    z2 = torch.polar(abs, angle)
    print(z2)
    zr = torch.view_as_real(z2)
    print(zr.cpu)


# 　env DIPU_PYTHON_DEVICE_AS_CUDA　is default true！
def test_dipu_as_cuda_type():
    import torch_dipu

    d1 = torch.device("cuda", 0)
    t1 = torch.ones((1024, 1), device=0)
    print(t1)
    assert d1.type == "cuda"
    assert t1.is_cuda == True
    assert t1.device.type == "cuda"
    s1 = t1.storage()
    assert s1.device.type == "cuda"

    gen = torch.Generator("dipu")
    gen.manual_seed(1)
    assert gen.device.type == "cuda"


def test_torch_ver():
    from torch_dipu import dipu

    # torch version must be compatible
    assert dipu.check_dipu_torch_compatiable() == True


if __name__ == "__main__":
    for i in range(1, 2):
        test_torch_ver()
        test_empty_tensor()
        testdevice()
        testDeviceProperties()
        test_mem_get_info()
        testStream()
        test_record_stream()
        testevent()
        test_type()
        test_complex_type()
        test_dipu_as_cuda_type()

        # need more 2 device to run
        # testDevice1()
        # testDevice2()
        # test_device_copy()
