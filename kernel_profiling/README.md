
# 参数指标
FLOPS:每秒的浮点运算次数 = 【GPU核数】*【单核主频】*【GPU单个周期浮点计算能力】

Memory bandwidth 带宽: 一个计算平台每秒所能完成的内存交换量。Byte/s
bandwidth = 【内存频率】*【Prefetch】*【内存带宽】/【8】

带宽：理论值
吞吐量：实际值

# 使用ncu
/usr/local/NVIDIA-Nsight-Compute-2025.1/ncu --set full -o profile ./add_profile