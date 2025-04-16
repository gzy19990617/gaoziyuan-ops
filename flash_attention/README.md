# flash_attention

## HBM (High Bandwidth Memory)
芯片外的显存
GPU的主内存，相比SRAM慢约5-10倍
存储模型权重、激活值、中间结果等大量数据，显存

## SRAM (Static Random-Access Memory)
芯片内的显存，通常为几十MB, 一般为Shared_memory \ L1 Cache \寄存器文件


## 原始Attention实现：
矩阵QKV存在HBM中
1.从HBM加载Q、K到SRAM
2.计算 S = QK^T
3.将S写到HBM
4.将S加载到SRAM
5.计算P = softmax(S)
6.将P写出到HBM
7.从HBM加载P和V到SRAM
8.计算O = PV
9.把O写出到HBM
10.返回O

缺点：中间有很多临时变量，比如P和O矩阵，他们的缓存随着序列长度N增大，缓存N^2增长。

## flash_attention目标
flash_attention着眼于减少IO操作，目标是避免Attention从HBM的读写
1.通过分块计算，融合多个操作，减少中间结果缓存
2.反向传播时，重新计算中间结果

随着序列长度N增大，缓存N^2增长变为随着N线性增长。

## softmax分块进行
计算公式：
https://github.com/gzy19990617/gaoziyuan-ops/blob/main/flash_attention/tiling_softmax.jpg




