# AICPU Hello world example

Copied from

https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_00050.html

## `--cce-aicpu-no-firmware` is required

The compile commands in the documentation linked above omit this flag, and without
it the kernel never executes. Measured on Ascend 910B2, CANN 9.0.0, driver
26.0.rc1, 10 runs per build:

| build | runs that print |
|---|---|
| exactly as documented | 0/10 |
| as documented, plus `-fPIC` | 0/10 |
| plus `--cce-aicpu-no-firmware` | 9/10 |

`-fPIC` and `-g` make no difference; only the flag does.

When the flag is missing, `aclrtSynchronizeStream` returns `507018`
(`aicpu exception`) and the device reports `errCode=11003` ("get kernel failed") in
`~/ascend/atrace/*/schedule_event_*/schedule_tracer_*.txt`. A kernel with an empty
body fails identically, so this is unrelated to `printf`.

**Why the flag is needed is not established.** Both builds embed an `ET_DYN`
AArch64 shared object with identical symbols and no `SONAME`, differing only in
size, so the compiler's own help text is the only available explanation:

```
--cce-aicpu-no-firmware
      Compile AICpu code with operating system support instead
      of for firmware-only device
```

The flag is undocumented, and the documentation section that would cover AI CPU
compilation -- 2.3.3 "AI CPU算子编译", referenced by 2.2.5 -- is absent from the
CANN 9.0.0 documentation set. Maintainer input welcome.

### Note on printf reliability

With the flag applied, roughly 6% of runs still lose the output (47/50 printed over
50 runs). `AscendC::printf` writes into a dump buffer drained by a host-side thread,
and `DataStoreBarrier()` is only `dsb st` on the device, so teardown can race the
drain. Inserting a wait before `aclrtDestroyStream` makes it 50/50, but the
documentation prescribes no such wait, so this looks like a CANN-side issue rather
than a missing step here.




### Second example

Interesting example from https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_00188.html



```
struct TilingInfo {
    uint64_t lock; // AI CPU/AI Core之间同步的锁
    int8_t type;
    int8_t mode;
    int8_t len;
};
struct KernelArgs {
    uint32_t *xDevice;
    uint32_t *yDevice;
    uint32_t *zDevice;
    TilingInfo *ti; // 与AI Core共享的参数，用于同步tiling选择
};

template<typename T, int8_t mode, int8_t len>
__aicore__ void hello_world_impl(GM_ADDR m)
{
    if constexpr (std::is_same_v<T, float>) {
       AscendC::printf("Hello World: float mode %u len %u.\n", mode, len);
    } else if constexpr (std::is_same_v<T, int>) {
       AscendC::printf("Hello World: int mode %u len %u.\n", mode, len);
    }
}

// AI Core算子总入口
// tilingInfo: 和AI CPU算子共同传递的参数，用于数据共享
template<typename T, int8_t mode, int8_t len>
__mix__(1,2) __global__ __aicore__ void hello_world(GM_ADDR m, GM_ADDR TilingPtr)
{
     __gm__ struct KernelInfo::TilingInfo *ti = (__gm__ struct KernelInfo::TilingInfo *)TilingPtr;
    AscendC::GlobalTensor<uint64_t> lock;
    lock.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(&ti->lock));
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            while (*reinterpret_cast<volatile __gm__ uint64_t*>(lock.GetPhyAddr(0)) == 0) {   // 下沉模式，AI Core等待AICPU tiling计算完成
                AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                    AscendC::DcciDst::CACHELINE_OUT>(lock);    //直接访问Global Memory，获取最新数据
            }
        }
    }
    // 上面是1个核等待AI CPU tiling计算完成，这里进行核间同步
    AscendC::SyncAll<false>();
    // 根据tiling参数值选择不同模板
    if (ti->type ==0 && ti->mode == 1 && ti->len == 2) {
        hello_world_impl<float, 1, 2>(m);
    } else if (ti->type == 1 && ti->mode == 2 && ti->len == 4) {
        hello_world_impl<int, 2, 4>(m);
    }
    // 执行完留一个核释放lock
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            lock.SetValue(0, 0);  // 刷新 lock
            AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(lock);    //刷新Dcache，同步与GM之间的数据
        }
    }
}

extern "C" __global__ __aicpu__ uint32_t MyAicpuKernel(void *arg)
{
    KernelArgs* cfg = (KernelArgs*)arg;
    AscendC::printf("MyAicpuKernel inited!\n");
    cfg->ti->lock = 1;
    cfg->ti->type = 1;
    cfg->ti->mode = 2;
    cfg->ti->len = 4;
    AscendC::DataStoreBarrier(); // 对tilingInfo进行写同步
    AscendC::printf("MyAicpuKernel inited type %u mode %u len %u end!\n", cfg->ti->type, cfg->ti->mode, cfg->ti->len);
    return 0;
}
```
