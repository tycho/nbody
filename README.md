n-body
=======

[![Build status](https://travis-ci.org/tycho/nbody.svg?branch=master)](https://travis-ci.org/tycho/nbody)

This is a fast and portable implementation of an [n-body
simulation](https://en.wikipedia.org/wiki/N-body_simulation). It can optionally
take advantage of compiler-provided auto-vectorization for single-thread
throughput  as well as [OpenMP](https://en.wikipedia.org/wiki/OpenMP) for
parallelism, and even has support for [CUDA](https://www.nvidia.com/object/cuda_home_new.html).


About
------

There are multiple different algorithms included, with varying levels of
performance. They are divided into two basic designs: SOA (Structure-of-Arrays)
and AOS (Array-of-Structures). You can read more about the differences between
the two approaches [on Wikipedia](https://en.wikipedia.org/wiki/AOS_and_SOA)
and [Stack Overflow](https://stackoverflow.com/questions/40163722/is-my-understanding-of-aos-vs-soa-advantages-disadvantages-correct).

There's a key tradeoff between the two: compilers have a much less difficult
time auto-vectorizing the SOA implementations, but the AOS implementations are
easier for programmers to read and write. It's not uncommon to see AOS
algorithms applied in real-world applications (consider, for example, how most
object-oriented programs are constructed).

Most of the code in this project is ordinary C++11, with some added pragmas to
provide compilers with appropriate hints to vectorize/parallelize the code
properly.


Building
--------

Install [Meson](https://mesonbuild.com/) in your build environment, and then
configure the project:

```
$ meson . build
```

There are some additional options available to enable or disable certain
features in the build:

- To enable CUDA support, you can specify `-Duse_cuda=true` on the meson
  command line.

- Newer versions of Clang and LLVM support building CUDA using LLVM's code
  generation, which can sometimes be faster than the NVIDIA official `nvcc`
  binary. You can specify to use Clang instead with `-Dnvcc=clang++`. Note that
  there are some [restrictions](https://gist.github.com/ax3l/9489132) on what
  CUDA versions can be used with Clang. At the time of this writing, Clang 9.0
  is the latest released version available and it only supports through CUDA
  10.1.

- To enable an OpenGL based renderer for the simulation: `-Dopengl=true`

- To disable OpenMP support: `-Dopenmp=false`

- To disable the SIMD intrinsics variants: `-Dintrinsics=false`


Example Runs
------------

Here's an example run of the included `compare-compilers` script in the
`scripts/` directory. Note that the arguments passed to `compare-compilers` are
the same ones you can put on the `nbody` command-line.

```
$ scripts/compare-compilers

Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
Address sizes:                   43 bits physical, 48 bits virtual
CPU(s):                          32
On-line CPU(s) list:             0-31
Thread(s) per core:              2
Core(s) per socket:              16
Socket(s):                       1
NUMA node(s):                    1
Vendor ID:                       AuthenticAMD
CPU family:                      23
Model:                           113
Model name:                      AMD Ryzen 9 3950X 16-Core Processor
Stepping:                        0
Frequency boost:                 enabled
CPU MHz:                         4144.101
CPU max MHz:                     3500.0000
CPU min MHz:                     2200.0000
BogoMIPS:                        7000.31
Virtualization:                  AMD-V
L1d cache:                       512 KiB
L1i cache:                       512 KiB
L2 cache:                        8 MiB
L3 cache:                        64 MiB
NUMA node0 CPU(s):               0-31
Vulnerability Itlb multihit:     Not affected
Vulnerability L1tf:              Not affected
Vulnerability Mds:               Not affected
Vulnerability Meltdown:          Not affected
Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:        Mitigation; Full AMD retpoline, IBPB conditional, STIBP conditional, RSB filling
Vulnerability Tsx async abort:   Not affected
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxs
                                 r sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl n
                                 onstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse
                                 4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy ab
                                 m sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_
                                 nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate sme ssbd mba sev ibpb stibp vmm
                                 call fsgsbase bmi1 avx2 smep bmi2 cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsav
                                 eopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero irp
                                 erf xsaveerptr wbnoinvd arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyas
                                 id decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif umip rdpid overflow_
                                 recov succor smca

Linux 5.4.7-1-hsw x86_64

============================================

g++ (GCC) 9.2.1 20200102
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Apr_24_19:10:27_PDT_2019
Cuda compilation tools, release 10.1, V10.1.168

n-body version: 1.6.0-109-g8f6ea8a

Binary size: 1284632 bytes

+ env OMP_NUM_THREADS=16 KMP_AFFINITY=scatter build-gcc/nbody --bodies 96 --cycle-after 4 --iterations 1 --verbose --no-crosscheck
Running simulation with 98304 particles, 16 CPU threads, up to 1 GPUs
      CPU_SOA:   255.37 ms =   37.843x10^9 interactions/s (   756.85 GFLOPS)
      CPU_SOA:   253.51 ms =   38.120x10^9 interactions/s (   762.39 GFLOPS)
      CPU_SOA:   254.90 ms =   37.911x10^9 interactions/s (   758.22 GFLOPS)
      CPU_SOA:   258.31 ms =   37.412x10^9 interactions/s (   748.23 GFLOPS)
CPU_SOA_tiled:   230.47 ms =   41.930x10^9 interactions/s (   838.61 GFLOPS)
CPU_SOA_tiled:   231.30 ms =   41.779x10^9 interactions/s (   835.58 GFLOPS)
CPU_SOA_tiled:   230.95 ms =   41.844x10^9 interactions/s (   836.87 GFLOPS)
CPU_SOA_tiled:   230.93 ms =   41.846x10^9 interactions/s (   836.93 GFLOPS)
          AVX:   241.47 ms =   40.020x10^9 interactions/s (   800.39 GFLOPS)
          AVX:   243.60 ms =   39.670x10^9 interactions/s (   793.39 GFLOPS)
          AVX:   239.93 ms =   40.276x10^9 interactions/s (   805.52 GFLOPS)
          AVX:   241.74 ms =   39.975x10^9 interactions/s (   799.51 GFLOPS)
      CPU_AOS:   669.42 ms =   14.436x10^9 interactions/s (   288.72 GFLOPS)
      CPU_AOS:   670.25 ms =   14.418x10^9 interactions/s (   288.36 GFLOPS)
      CPU_AOS:   671.33 ms =   14.395x10^9 interactions/s (   287.90 GFLOPS)
      CPU_AOS:   671.82 ms =   14.384x10^9 interactions/s (   287.68 GFLOPS)
CPU_AOS_tiled:   659.75 ms =   14.647x10^9 interactions/s (   292.95 GFLOPS)
CPU_AOS_tiled:   659.34 ms =   14.657x10^9 interactions/s (   293.13 GFLOPS)
CPU_AOS_tiled:   661.97 ms =   14.598x10^9 interactions/s (   291.97 GFLOPS)
CPU_AOS_tiled:   659.19 ms =   14.660x10^9 interactions/s (   293.20 GFLOPS)
      GPU_AOS:    31.93 ms =  302.662x10^9 interactions/s (  6053.25 GFLOPS)
      GPU_AOS:    32.93 ms =  293.504x10^9 interactions/s (  5870.08 GFLOPS)
      GPU_AOS:    32.93 ms =  293.490x10^9 interactions/s (  5869.81 GFLOPS)
      GPU_AOS:    32.21 ms =  300.066x10^9 interactions/s (  6001.31 GFLOPS)
   GPU_Shared:    21.95 ms =  440.250x10^9 interactions/s (  8805.00 GFLOPS)
   GPU_Shared:    21.62 ms =  447.074x10^9 interactions/s (  8941.48 GFLOPS)
   GPU_Shared:    21.41 ms =  451.283x10^9 interactions/s (  9025.66 GFLOPS)
   GPU_Shared:    21.25 ms =  454.859x10^9 interactions/s (  9097.18 GFLOPS)
    GPU_Const:    21.81 ms =  443.019x10^9 interactions/s (  8860.37 GFLOPS)
    GPU_Const:    21.67 ms =  445.996x10^9 interactions/s (  8919.93 GFLOPS)
    GPU_Const:    21.24 ms =  455.013x10^9 interactions/s (  9100.26 GFLOPS)
    GPU_Const:    21.69 ms =  445.444x10^9 interactions/s (  8908.89 GFLOPS)
     MultiGPU:    20.42 ms =  473.315x10^9 interactions/s (  9466.31 GFLOPS)
     MultiGPU:    20.73 ms =  466.177x10^9 interactions/s (  9323.54 GFLOPS)
     MultiGPU:    20.83 ms =  463.966x10^9 interactions/s (  9279.32 GFLOPS)
     MultiGPU:    21.06 ms =  458.940x10^9 interactions/s (  9178.80 GFLOPS)
  GPU_Shuffle:    25.45 ms =  379.778x10^9 interactions/s (  7595.56 GFLOPS)
  GPU_Shuffle:    25.79 ms =  374.707x10^9 interactions/s (  7494.14 GFLOPS)
  GPU_Shuffle:    25.58 ms =  377.709x10^9 interactions/s (  7554.18 GFLOPS)
  GPU_Shuffle:    26.45 ms =  365.367x10^9 interactions/s (  7307.34 GFLOPS)


============================================

clang version 10.0.0 (https://github.com/llvm/llvm-project.git a5615f5fe8c4b23795c478c3271a7fd0fb04653d)
Target: x86_64-pc-linux-gnu
Thread model: posix
InstalledDir: /home/steven/.apps/llvm-master/bin

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Apr_24_19:10:27_PDT_2019
Cuda compilation tools, release 10.1, V10.1.168

n-body version: 1.6.0-109-g8f6ea8a

Binary size: 582264 bytes

+ env OMP_NUM_THREADS=16 KMP_AFFINITY=scatter build-clang/nbody --bodies 96 --cycle-after 4 --iterations 1 --verbose --no-crosscheck
Running simulation with 98304 particles, 16 CPU threads, up to 1 GPUs
      CPU_SOA:   263.99 ms =   36.606x10^9 interactions/s (   732.12 GFLOPS)
      CPU_SOA:   265.18 ms =   36.442x10^9 interactions/s (   728.83 GFLOPS)
      CPU_SOA:   264.38 ms =   36.553x10^9 interactions/s (   731.05 GFLOPS)
      CPU_SOA:   265.02 ms =   36.464x10^9 interactions/s (   729.29 GFLOPS)
CPU_SOA_tiled:   246.33 ms =   39.231x10^9 interactions/s (   784.62 GFLOPS)
CPU_SOA_tiled:   246.95 ms =   39.132x10^9 interactions/s (   782.64 GFLOPS)
CPU_SOA_tiled:   246.95 ms =   39.131x10^9 interactions/s (   782.63 GFLOPS)
CPU_SOA_tiled:   246.96 ms =   39.131x10^9 interactions/s (   782.62 GFLOPS)
          AVX:   262.43 ms =   36.824x10^9 interactions/s (   736.47 GFLOPS)
          AVX:   264.51 ms =   36.534x10^9 interactions/s (   730.69 GFLOPS)
          AVX:   263.24 ms =   36.710x10^9 interactions/s (   734.21 GFLOPS)
          AVX:   262.93 ms =   36.754x10^9 interactions/s (   735.08 GFLOPS)
      CPU_AOS:   656.69 ms =   14.716x10^9 interactions/s (   294.32 GFLOPS)
      CPU_AOS:   656.52 ms =   14.720x10^9 interactions/s (   294.39 GFLOPS)
      CPU_AOS:   656.70 ms =   14.715x10^9 interactions/s (   294.31 GFLOPS)
      CPU_AOS:   657.52 ms =   14.697x10^9 interactions/s (   293.94 GFLOPS)
CPU_AOS_tiled:   631.12 ms =   15.312x10^9 interactions/s (   306.24 GFLOPS)
CPU_AOS_tiled:   631.14 ms =   15.311x10^9 interactions/s (   306.23 GFLOPS)
CPU_AOS_tiled:   631.07 ms =   15.313x10^9 interactions/s (   306.26 GFLOPS)
CPU_AOS_tiled:   631.09 ms =   15.313x10^9 interactions/s (   306.25 GFLOPS)
      GPU_AOS:    31.14 ms =  310.311x10^9 interactions/s (  6206.22 GFLOPS)
      GPU_AOS:    31.16 ms =  310.116x10^9 interactions/s (  6202.33 GFLOPS)
      GPU_AOS:    31.17 ms =  310.048x10^9 interactions/s (  6200.96 GFLOPS)
      GPU_AOS:    27.93 ms =  345.939x10^9 interactions/s (  6918.79 GFLOPS)
   GPU_Shared:    19.61 ms =  492.806x10^9 interactions/s (  9856.11 GFLOPS)
   GPU_Shared:    19.82 ms =  487.682x10^9 interactions/s (  9753.64 GFLOPS)
   GPU_Shared:    19.27 ms =  501.490x10^9 interactions/s ( 10029.80 GFLOPS)
   GPU_Shared:    19.98 ms =  483.760x10^9 interactions/s (  9675.19 GFLOPS)
    GPU_Const:    26.78 ms =  360.804x10^9 interactions/s (  7216.08 GFLOPS)
    GPU_Const:    25.91 ms =  373.026x10^9 interactions/s (  7460.52 GFLOPS)
    GPU_Const:    25.89 ms =  373.328x10^9 interactions/s (  7466.55 GFLOPS)
    GPU_Const:    25.91 ms =  372.923x10^9 interactions/s (  7458.46 GFLOPS)
     MultiGPU:    20.19 ms =  478.727x10^9 interactions/s (  9574.55 GFLOPS)
     MultiGPU:    20.19 ms =  478.658x10^9 interactions/s (  9573.15 GFLOPS)
     MultiGPU:    20.33 ms =  475.334x10^9 interactions/s (  9506.67 GFLOPS)
     MultiGPU:    20.37 ms =  474.370x10^9 interactions/s (  9487.41 GFLOPS)
  GPU_Shuffle:    25.24 ms =  382.938x10^9 interactions/s (  7658.77 GFLOPS)
  GPU_Shuffle:    26.21 ms =  368.707x10^9 interactions/s (  7374.13 GFLOPS)
  GPU_Shuffle:    25.86 ms =  373.691x10^9 interactions/s (  7473.81 GFLOPS)
  GPU_Shuffle:    26.02 ms =  371.338x10^9 interactions/s (  7426.76 GFLOPS)
```
