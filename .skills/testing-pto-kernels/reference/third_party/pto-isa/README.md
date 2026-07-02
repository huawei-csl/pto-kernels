# PTO-ISA Headers

A PTO kernel needs `pto/pto-inst.hpp`.

## Default: CANN-Shipped Headers

Use this when the kernel only needs stable APIs:

```bash
export PTO_LIB_PATH="${ASCEND_HOME_PATH}"
```

Compile with:

```bash
-I${PTO_LIB_PATH}/include
```

## Pinned Checkout

Use a local checkout when a sample needs newer APIs than CANN provides:

```bash
export PTO_LIB_PATH=/path/to/pto-isa
```

Examples that may need newer headers include `TALLOC`, `TPOP(GlobalData)`, `TFREE`, or recent A5 direct-copy helpers.

## Submodule Pattern

For deliverable code, prefer a pinned submodule or documented commit:

```bash
git submodule add https://gitcode.com/cann/pto-isa.git third_party/pto-isa
git submodule update --init --recursive
export PTO_LIB_PATH=$PWD/third_party/pto-isa
```

Do not vendor a large PTO-ISA source tree into this skill reference unless a future task explicitly needs an offline snapshot.
