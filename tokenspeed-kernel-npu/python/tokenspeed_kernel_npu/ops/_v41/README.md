# V4.1 MoE providers

The Ascend plugin registers sqrt-softplus router top-k for plain and biased
routing. Bias affects selection only; returned weights use unbiased scores,
optional renormalization, and routed scaling. Ties choose the lowest expert
index. Hash routing and requests for all scores retain their existing provider.

The W4A4 expert sequence keeps vendor FlatQuant and both grouped GEMMs.
`grouped_experts` receives an explicit `route_mapping`: `None` means weights
already follow expanded expert rows; a tensor maps router-order weights to
those rows. SwiGLU consumes the mapping directly, eliminating the separate
route-weight scatter and buffer. MC2 supplies already expanded weights through
the same expert sequence. Clipping, FP32 route weighting, and the BF16
activation boundary remain before the second FlatQuant operation.

Build the native providers with the active CANN environment:

```sh
python tokenspeed-kernel-npu/csrc/dsv41/build.py
```

The route-weight SwiGLU provider checks its fixed-buffer ABI before launch and
uses the Triton implementation when the native geometry is unsupported or the
library is absent. Active expert counts stay on the device; inactive rows are
not consumed. Run `test_router_topk_native.py` and
`test_swiglu_route_native.py` under `tokenspeed-kernel-npu/test` on an NPU.
