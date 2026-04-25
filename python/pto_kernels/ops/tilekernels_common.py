from __future__ import annotations

from mlir.dialects import arith, math
from mlir.dialects import pto as _pto_dialect
from mlir.ir import IntegerType

from ptodsl import pto
from ptodsl import scalar as s


DTYPES = {
    "bf16": lambda: pto.bfloat16,
    "f32": lambda: pto.float32,
    "i32": lambda: pto.int32,
    "i64": lambda: IntegerType.get_signless(64),
}


def dtype_from_name(name: str):
    try:
        return DTYPES[name]()
    except KeyError as exc:
        supported = ", ".join(sorted(DTYPES))
        raise ValueError(f"Unsupported dtype {name!r}; supported: {supported}") from exc


def ptr(dtype):
    return pto.PtrType(dtype)


def int64_type():
    return IntegerType.get_signless(64)


def uint64_type():
    return IntegerType.get_signless(64)


def tile_type(dtype, shape: list[int], *, valid_shape: list[int] | None = None):
    if valid_shape is None:
        valid_shape = shape
    return pto.TileBufType(
        shape=shape,
        valid_shape=valid_shape,
        dtype=dtype,
        memory_space="VEC",
        config=pto.TileBufConfig(),
    )


def load_scalar(dtype, ptr_value, offset):
    return s.wrap_value(
        _pto_dialect.load_scalar(dtype, s._unwrap(ptr_value), s._unwrap(offset))
    )


def store_scalar(ptr_value, offset, value) -> None:
    _pto_dialect.store_scalar(
        s._unwrap(ptr_value), s._unwrap(offset), s._unwrap(value)
    )


def f32_const(value: float):
    return s.wrap_value(arith.ConstantOp(pto.float32, float(value)).result)


def fadd(lhs, rhs):
    return s.wrap_value(arith.AddFOp(s._unwrap(lhs), s._unwrap(rhs)).result)


def fsub(lhs, rhs):
    return s.wrap_value(arith.SubFOp(s._unwrap(lhs), s._unwrap(rhs)).result)


def fmul(lhs, rhs):
    return s.wrap_value(arith.MulFOp(s._unwrap(lhs), s._unwrap(rhs)).result)


def fdiv(lhs, rhs):
    return s.wrap_value(arith.DivFOp(s._unwrap(lhs), s._unwrap(rhs)).result)


def fexp(value):
    return s.wrap_value(math.exp(s._unwrap(value)))


def fsqrt(value):
    return s.wrap_value(math.sqrt(s._unwrap(value)))


def frsqrt(value):
    return s.wrap_value(math.rsqrt(s._unwrap(value)))


def fsigmoid(value):
    one = f32_const(1.0)
    neg = fsub(f32_const(0.0), value)
    return fdiv(one, fadd(one, fexp(neg)))


def fcmp_ogt(lhs, rhs):
    return s.wrap_value(
        arith.CmpFOp(arith.CmpFPredicate.OGT, s._unwrap(lhs), s._unwrap(rhs)).result
    )


def fcmp_olt(lhs, rhs):
    return s.wrap_value(
        arith.CmpFOp(arith.CmpFPredicate.OLT, s._unwrap(lhs), s._unwrap(rhs)).result
    )


def iadd(lhs, rhs):
    return s.wrap_value(arith.AddIOp(s._unwrap(lhs), s._unwrap(rhs)).result)


def imul(lhs, rhs):
    return s.wrap_value(arith.MulIOp(s._unwrap(lhs), s._unwrap(rhs)).result)


def irem(lhs, rhs):
    return s.wrap_value(arith.RemSIOp(s._unwrap(lhs), s._unwrap(rhs)).result)


def ixor(lhs, rhs):
    return s.wrap_value(arith.XOrIOp(s._unwrap(lhs), s._unwrap(rhs)).result)


def iand(lhs, rhs):
    return s.wrap_value(arith.AndIOp(s._unwrap(lhs), s._unwrap(rhs)).result)


def ior(lhs, rhs):
    return s.wrap_value(arith.OrIOp(s._unwrap(lhs), s._unwrap(rhs)).result)


def sext(value, dst_type):
    return s.wrap_value(arith.ExtSIOp(dst_type, s._unwrap(value)).result)


def trunc(value, dst_type):
    return s.wrap_value(arith.TruncIOp(dst_type, s._unwrap(value)).result)
