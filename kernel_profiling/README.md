# cuda缓存&访存流程

## 内存层次
```
graph TB
    A[寄存器(Register)] --> B[Shared Memory]
    B --> C[L1 Cache / Texture Cache]
    C --> D[L2 Cache]
    D --> E[全局内存(Global Memory)]
    E --> F[主机内存(Host Memory)]
```


| 缓存类型      | 作用范围          | 特点                                                                 |
|---------------|-------------------|----------------------------------------------------------------------|
| 寄存器        | 单个线程私有      | 最快，但容量极小（每个线程约255个32-bit寄存器）                     |
| Shared Memory | Block内共享       | 用户可编程，延迟≈1 cycle，带宽高（需避免bank conflict）             |
| L1 Cache      | SM内共享          | 自动缓存全局/局部内存，缓存行128字节                                |
| L2 Cache      | 整个GPU共享       | 所有SM共用，缓存全局/常量/纹理内存                                  |
| 常量缓存      | Warp内共享        | 对同一地址的访问会广播到所有线程                                    |



Sector:通常以 32 字节（Byte） 为一个扇区（Sector），这是最小访问粒度。

一个 Warp（32 线程）访问连续的 128 字节对齐内存块（例如 float 类型，32 线程 × 4 字节 = 128 字节），若warp内线程访问连续128字节区域→合并为1次事务

仅需 1 次 128B 内存事务（即 4 个 32B 扇区）。

当前一定会从Dram到L2 cache 再到 L1 cache。



## 使用ncu
/usr/local/NVIDIA-Nsight-Compute-2025.1/ncu --set full -o profile ./add_profile



# occupancy

Occupancy指的是在任意给定时刻，GPU上活跃的线程束（warps）数量与GPU理论上支持的最大活跃线程束数量的比率。简单说，它表示GPU计算资源的利用程度。

Occupancy = 活跃线程束数量 / 最大可能的活跃线程束数量

1.用户设置的block大小会影响实际sm上可以容纳的warp数量：每一个block必须在一个sm中

2.CUDA 的隐藏延迟：一个warp处于等待状态时候会立刻切到下一个Warp，与cpu不同，不需要切换上文，因为gpu有大量寄存器，此时每个sm的寄存器数量有上限，也会成为开启多个warp的一个指标

3.共享内存也有固定大小，也影响warp数量


# warp scheduler
(todo)