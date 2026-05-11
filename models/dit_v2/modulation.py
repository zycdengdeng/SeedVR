# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Callable, List, Optional
import torch
from einops import rearrange
from torch import nn

from common.cache import Cache
from common.distributed.ops import slice_inputs

# (dim: int, emb_dim: int)
ada_layer_type = Callable[[int, int], nn.Module]


def get_ada_layer(ada_layer: str) -> ada_layer_type:
    if ada_layer == "single":
        return AdaSingle
    raise NotImplementedError(f"{ada_layer} is not supported")


def expand_dims(x: torch.Tensor, dim: int, ndim: int):
    """
    Expand tensor "x" to "ndim" by adding empty dims at "dim".
    Example: x is (b d), target ndim is 5, add dim at 1, return (b 1 1 1 d).
    """
    shape = x.shape
    shape = shape[:dim] + (1,) * (ndim - len(shape)) + shape[dim:]
    return x.reshape(shape)


class AdaSingle(nn.Module):
    def __init__(
        self,
        dim: int,
        emb_dim: int,
        layers: List[str],
        modes: List[str] = ["in", "out"],
    ):
        assert emb_dim == 6 * dim, "AdaSingle requires emb_dim == 6 * dim"
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.layers = layers
        for l in layers:
            if "in" in modes:
                self.register_parameter(f"{l}_shift", nn.Parameter(torch.randn(dim) / dim**0.5))
                self.register_parameter(
                    f"{l}_scale", nn.Parameter(torch.randn(dim) / dim**0.5 + 1)
                )
            if "out" in modes:
                self.register_parameter(f"{l}_gate", nn.Parameter(torch.randn(dim) / dim**0.5))

    def forward(
        self,
        hid: torch.FloatTensor,  # b ... c
        emb: torch.FloatTensor,  # b d
        layer: str,
        mode: str,
        cache: Cache = Cache(disable=True),
        branch_tag: str = "",
        hid_len: Optional[torch.LongTensor] = None,  # b
    ) -> torch.FloatTensor:
        idx = self.layers.index(layer)
        emb = rearrange(emb, "b (d l g) -> b d l g", l=len(self.layers), g=3)[..., idx, :]
        emb = expand_dims(emb, 1, hid.ndim + 1)

        if hid_len is not None:
            # [DBG] temporary diagnostic block to localize CUDA "no kernel image" error.
            # Remove after debugging.
            torch.cuda.synchronize()
            print(
                f"[DBG] before repeat: idx={idx} branch_tag={branch_tag} "
                f"emb.shape={tuple(emb.shape)} emb.dtype={emb.dtype} emb.device={emb.device} "
                f"emb.is_contiguous={emb.is_contiguous()} "
                f"hid_len={hid_len.tolist()} hid.shape={tuple(hid.shape)} hid.dtype={hid.dtype}",
                flush=True,
            )
            try:
                torch.cuda.synchronize(); print("[DBG] A: before for-loop", flush=True)
                _repeated = []
                _list_e = list(emb)
                torch.cuda.synchronize(); print(f"[DBG] B: list(emb) len={len(_list_e)} first.shape={tuple(_list_e[0].shape)} first.contig={_list_e[0].is_contiguous()}", flush=True)
                _list_l = list(hid_len)
                torch.cuda.synchronize(); print(f"[DBG] C: list(hid_len) len={len(_list_l)} first.shape={tuple(_list_l[0].shape)} first.dtype={_list_l[0].dtype}", flush=True)
                for _i in range(len(_list_e)):
                    _e = _list_e[_i]
                    _l = _list_l[_i]
                    torch.cuda.synchronize(); print(f"[DBG] D[{_i}]: got _e and _l, _e.stride={_e.stride()}", flush=True)
                    _li = _l.item()
                    torch.cuda.synchronize(); print(f"[DBG] E[{_i}]: _li={_li}", flush=True)
                    # Try 4 alternatives to make _e contiguous, log which work.
                    _e_c = None
                    # (1) plain .contiguous()
                    try:
                        _tmp = _e.contiguous(); torch.cuda.synchronize()
                        print(f"[DBG] E1[{_i}]: .contiguous() OK", flush=True); _e_c = _e_c or _tmp
                    except Exception as ex:
                        print(f"[DBG] E1[{_i}]: .contiguous() FAIL: {type(ex).__name__}: {ex}", flush=True)
                    # (2) .clone()
                    try:
                        _tmp = _e.clone(); torch.cuda.synchronize()
                        print(f"[DBG] E2[{_i}]: .clone() OK", flush=True); _e_c = _e_c or _tmp
                    except Exception as ex:
                        print(f"[DBG] E2[{_i}]: .clone() FAIL: {type(ex).__name__}: {ex}", flush=True)
                    # (3) empty + copy_
                    try:
                        _tmp = torch.empty(_e.shape, dtype=_e.dtype, device=_e.device).copy_(_e); torch.cuda.synchronize()
                        print(f"[DBG] E3[{_i}]: empty+copy_ OK", flush=True); _e_c = _e_c or _tmp
                    except Exception as ex:
                        print(f"[DBG] E3[{_i}]: empty+copy_ FAIL: {type(ex).__name__}: {ex}", flush=True)
                    # (4) dtype roundtrip
                    try:
                        _tmp = _e.float().to(_e.dtype); torch.cuda.synchronize()
                        print(f"[DBG] E4[{_i}]: float-roundtrip OK", flush=True); _e_c = _e_c or _tmp
                    except Exception as ex:
                        print(f"[DBG] E4[{_i}]: float-roundtrip FAIL: {type(ex).__name__}: {ex}", flush=True)
                    if _e_c is None:
                        raise RuntimeError("all 4 contiguous strategies failed")
                    torch.cuda.synchronize(); print(f"[DBG] F[{_i}]: picked _e_c contig={_e_c.is_contiguous()}", flush=True)
                    _r = _e_c.repeat(_li, *([1] * _e_c.ndim))
                    torch.cuda.synchronize(); print(f"[DBG] H[{_i}]: repeat done shape={tuple(_r.shape)}", flush=True)
                    _repeated.append(_r)
                print(f"[DBG] repeat loop done, n={len(_repeated)}", flush=True)
                _cat = torch.cat(_repeated)
                torch.cuda.synchronize(); print(f"[DBG] I: cat done shape={tuple(_cat.shape)}", flush=True)
            except Exception as _ex:
                print(f"[DBG] FAIL at idx={idx}/{branch_tag}: {_ex!r}", flush=True)
                raise
            emb = cache(
                f"emb_repeat_{idx}_{branch_tag}",
                lambda: slice_inputs(_cat, dim=0),
            )

        shiftA, scaleA, gateA = emb.unbind(-1)
        shiftB, scaleB, gateB = (
            getattr(self, f"{layer}_shift", None),
            getattr(self, f"{layer}_scale", None),
            getattr(self, f"{layer}_gate", None),
        )

        if mode == "in":
            return hid.mul_(scaleA + scaleB).add_(shiftA + shiftB)
        if mode == "out":
            return hid.mul_(gateA + gateB)
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"dim={self.dim}, emb_dim={self.emb_dim}, layers={self.layers}"