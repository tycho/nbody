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

Most of the code in this project is plain C, with some added pragmas to provide
compilers with appropriate hints to vectorize/parallelize the code properly.


Building
--------

The simplest way to build n-body is to clone the repository and simply run:

```
$ make
```

But there are a few options you can use to customize the build as you see fit.
OpenMP is enabled automatically if the build system detects your system
supports it, but you can disable the feature with a command-line option:

```
$ make NO_OPENMP=1
```

n-body also supports CUDA, and you can enable it with a flag on the build
command line:

```
$ make CUDA=1
```

If you want to have pretty-pretty graphics to look at while n-body is running,
you can also enable OpenGL support:

```
$ make OPENGL=1
```

Note also that multiple build options can be combined on the command line as
well:

```
$ make CUDA=1 OPENGL=1
```

The intrinsics-based algorithms can be disabled with:

```
$ make NO_SIMD=1
```

You can select your compiler by specifying it on the command line if you want
to:

```
$ make CC="clang"
```

And you can choose specific optimization flags if the defaults don't suit your
needs:

```
$ make CFOPTIMIZE="-09 -march=s6000 -funroll-every-loop -mrice -omg-optimized"
```


Example Runs
------------

Here's an example run of the included `compare-compilers` script in the
`scripts/` directory. Note that the arguments passed to `compare-compilers` are
the same ones you can put on the `nbody` command-line.

```
$ scripts/compare-compilers --bodies 128 --no-crosscheck --iterations 1 --cycle-after 3

CPU information:
    Architecture:        x86_64
    CPU op-mode(s):      32-bit, 64-bit
    Byte Order:          Little Endian
    Address sizes:       46 bits physical, 48 bits virtual
    CPU(s):              32
    On-line CPU(s) list: 0-31
    Thread(s) per core:  2
    Core(s) per socket:  8
    Socket(s):           2
    NUMA node(s):        2
    Vendor ID:           GenuineIntel
    CPU family:          6
    Model:               63
    Model name:          Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz
    Stepping:            2
    CPU MHz:             1200.062
    CPU max MHz:         2401.0000
    CPU min MHz:         1200.0000
    BogoMIPS:            4800.05
    Virtualization:      VT-x
    L1d cache:           32K
    L1i cache:           32K
    L2 cache:            256K
    L3 cache:            20480K
    NUMA node0 CPU(s):   0-7,16-23
    NUMA node1 CPU(s):   8-15,24-31
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr
       pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm
       pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts
       rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq
       dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm
       pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes
       xsave avx f16c rdrand lahf_lm abm cpuid_fault epb invpcid_single
       pti intel_ppin ssbd ibrs ibpb stibp tpr_shadow vnmi flexpriority
       ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid
       cqm xsaveopt cqm_llc cqm_occup_llc dtherm ida arat pln pts flush_l1d

OS version: Linux 4.20.10-2-hsw x86_64

============================================

CC        = gcc
NVCC      = nvcc
CUDA      = 1
LINK      = gcc
CFLAGS    = -O3 -march=native -ffast-math -std=gnu11 -fno-strict-aliasing
   -D_GNU_SOURCE -DLIBTIME_STATIC -DUSE_OPENMP -DHAVE_SIMD -DUSE_CUDA
   -DUSE_LIBC11 -I../subprojects/c11/include -I../subprojects/time/include
   -Wall -Wdeclaration-after-statement -Werror=implicit
   -Werror=undef -Wmissing-declarations -Wmissing-prototypes
   -Wno-declaration-after-statement -Wno-long-long -Wno-overlength-strings
   -Wno-unknown-pragmas -Wold-style-definition -Wstrict-prototypes -pthread
   -fPIC -fopenmp
NVCCFLAGS = -O3 -Drestrict= --ftz true -Xcompiler -fPIC
   -gencode=arch=compute_50,code="sm_50,compute_50"
   -gencode=arch=compute_52,code="sm_52,compute_52"
   -gencode=arch=compute_61,code="sm_61,compute_61"
   -gencode=arch=compute_70,code="sm_70,compute_70"
   -gencode=arch=compute_75,code="sm_75,compute_75"
LDFLAGS   = -pthread -fPIC -lomp -L/opt/cuda/bin/../lib
   -L/opt/cuda/bin/../lib64 -lcudart ../subprojects/c11/libc11.a
   ../subprojects/time/libtime.a -lm -lrt

Compiler version: gcc (GCC) 8.3.1 20190223

Binary size: 1809272 bytes

Running simulation with 131072 particles, 32 CPU threads, up to 1 GPUs
      CPU_SOA:   861.46 ms =   19.943x10^9 interactions/s (   398.85 GFLOPS)
      CPU_SOA:   862.15 ms =   19.927x10^9 interactions/s (   398.54 GFLOPS)
      CPU_SOA:   860.67 ms =   19.961x10^9 interactions/s (   399.22 GFLOPS)
CPU_SOA_tiled:   850.96 ms =   20.189x10^9 interactions/s (   403.78 GFLOPS)
CPU_SOA_tiled:   851.41 ms =   20.178x10^9 interactions/s (   403.56 GFLOPS)
CPU_SOA_tiled:   851.66 ms =   20.172x10^9 interactions/s (   403.44 GFLOPS)
          AVX:   819.19 ms =   20.972x10^9 interactions/s (   419.43 GFLOPS)
          AVX:   817.93 ms =   21.004x10^9 interactions/s (   420.08 GFLOPS)
          AVX:   818.85 ms =   20.981x10^9 interactions/s (   419.61 GFLOPS)
      CPU_AOS:  2471.56 ms =    6.951x10^9 interactions/s (   139.02 GFLOPS)
      CPU_AOS:  2471.15 ms =    6.952x10^9 interactions/s (   139.04 GFLOPS)
      CPU_AOS:  2472.85 ms =    6.947x10^9 interactions/s (   138.95 GFLOPS)
CPU_AOS_tiled:  2492.77 ms =    6.892x10^9 interactions/s (   137.84 GFLOPS)
CPU_AOS_tiled:  2493.42 ms =    6.890x10^9 interactions/s (   137.80 GFLOPS)
CPU_AOS_tiled:  2493.97 ms =    6.889x10^9 interactions/s (   137.77 GFLOPS)
      GPU_AOS:   266.27 ms =   64.521x10^9 interactions/s (  1290.43 GFLOPS)
      GPU_AOS:   231.43 ms =   74.234x10^9 interactions/s (  1484.68 GFLOPS)
      GPU_AOS:   220.75 ms =   77.825x10^9 interactions/s (  1556.50 GFLOPS)
   GPU_Shared:   113.93 ms =  150.798x10^9 interactions/s (  3015.96 GFLOPS)
   GPU_Shared:   113.92 ms =  150.809x10^9 interactions/s (  3016.18 GFLOPS)
   GPU_Shared:   114.04 ms =  150.647x10^9 interactions/s (  3012.93 GFLOPS)
    GPU_Const:   100.82 ms =  170.393x10^9 interactions/s (  3407.87 GFLOPS)
    GPU_Const:   101.36 ms =  169.494x10^9 interactions/s (  3389.88 GFLOPS)
    GPU_Const:   101.38 ms =  169.460x10^9 interactions/s (  3389.21 GFLOPS)
     MultiGPU:   100.32 ms =  171.256x10^9 interactions/s (  3425.13 GFLOPS)
     MultiGPU:   103.65 ms =  165.752x10^9 interactions/s (  3315.04 GFLOPS)
     MultiGPU:   103.67 ms =  165.712x10^9 interactions/s (  3314.23 GFLOPS)
  GPU_Shuffle:   107.87 ms =  159.259x10^9 interactions/s (  3185.18 GFLOPS)
  GPU_Shuffle:   111.85 ms =  153.599x10^9 interactions/s (  3071.98 GFLOPS)
  GPU_Shuffle:   114.10 ms =  150.566x10^9 interactions/s (  3011.32 GFLOPS)


============================================

CC        = clang
NVCC      = clang++
CUDA      = 1
LINK      = clang
CFLAGS    = -O3 -march=native -ffast-math -std=gnu11 -fno-strict-aliasing
   -D_GNU_SOURCE -DLIBTIME_STATIC -DUSE_OPENMP -DHAVE_SIMD -DUSE_CUDA
   -DUSE_LIBC11 -I../subprojects/c11/include -I../subprojects/time/include
   -Wall -Wdeclaration-after-statement -Werror=implicit
   -Werror=undef -Wmissing-declarations -Wmissing-prototypes
   -Wno-declaration-after-statement -Wno-long-long -Wno-overlength-strings
   -Wno-unknown-pragmas -Wold-style-definition -Wstrict-prototypes
   -pthread -fPIC -idirafter /usr/lib/gcc/x86_64-pc-linux-gnu/8.3.1/include
   -fopenmp=libomp
NVCCFLAGS = -O3 -Drestrict=__restrict -fcuda-flush-denormals-to-zero
   --cuda-gpu-arch=sm_52 --cuda-gpu-arch=sm_61 --cuda-gpu-arch=sm_70
   --cuda-gpu-arch=sm_75
LDFLAGS   = -pthread -fPIC -lomp -L/opt/cuda/bin/../lib
   -L/opt/cuda/bin/../lib64 -lcudart ../subprojects/c11/libc11.a
   ../subprojects/time/libtime.a -lm -lrt

Compiler version: clang version 8.0.0 (/startdir/clang 90903f44d639117b8c57d061291f4ea2b542bd83) (/startdir/llvm 795ca0111f5edb3135c35285ac3f684d570af73e)

Binary size: 525136 bytes

Running simulation with 131072 particles, 32 CPU threads, up to 1 GPUs
      CPU_SOA:   894.90 ms =   19.197x10^9 interactions/s (   383.95 GFLOPS)
      CPU_SOA:   895.24 ms =   19.190x10^9 interactions/s (   383.81 GFLOPS)
      CPU_SOA:   895.92 ms =   19.176x10^9 interactions/s (   383.51 GFLOPS)
CPU_SOA_tiled:   909.04 ms =   18.899x10^9 interactions/s (   377.98 GFLOPS)
CPU_SOA_tiled:   904.92 ms =   18.985x10^9 interactions/s (   379.70 GFLOPS)
CPU_SOA_tiled:   906.51 ms =   18.952x10^9 interactions/s (   379.03 GFLOPS)
          AVX:   901.69 ms =   19.053x10^9 interactions/s (   381.06 GFLOPS)
          AVX:   900.02 ms =   19.088x10^9 interactions/s (   381.77 GFLOPS)
          AVX:   901.47 ms =   19.058x10^9 interactions/s (   381.15 GFLOPS)
      CPU_AOS:  1637.30 ms =   10.493x10^9 interactions/s (   209.86 GFLOPS)
      CPU_AOS:  1638.01 ms =   10.488x10^9 interactions/s (   209.77 GFLOPS)
      CPU_AOS:  1637.67 ms =   10.490x10^9 interactions/s (   209.81 GFLOPS)
CPU_AOS_tiled:  1623.69 ms =   10.581x10^9 interactions/s (   211.61 GFLOPS)
CPU_AOS_tiled:  1625.07 ms =   10.572x10^9 interactions/s (   211.44 GFLOPS)
CPU_AOS_tiled:  1624.71 ms =   10.574x10^9 interactions/s (   211.48 GFLOPS)
      GPU_AOS:   220.89 ms =   77.774x10^9 interactions/s (  1555.49 GFLOPS)
      GPU_AOS:   220.94 ms =   77.758x10^9 interactions/s (  1555.16 GFLOPS)
      GPU_AOS:   221.01 ms =   77.733x10^9 interactions/s (  1554.67 GFLOPS)
   GPU_Shared:    97.26 ms =  176.640x10^9 interactions/s (  3532.79 GFLOPS)
   GPU_Shared:    99.08 ms =  173.389x10^9 interactions/s (  3467.78 GFLOPS)
    GPU_Const:   145.80 ms =  117.831x10^9 interactions/s (  2356.62 GFLOPS)
    GPU_Const:   143.75 ms =  119.510x10^9 interactions/s (  2390.20 GFLOPS)
    GPU_Const:   142.09 ms =  120.905x10^9 interactions/s (  2418.10 GFLOPS)
     MultiGPU:   103.16 ms =  166.534x10^9 interactions/s (  3330.68 GFLOPS)
     MultiGPU:    96.43 ms =  178.161x10^9 interactions/s (  3563.22 GFLOPS)
  GPU_Shuffle:   110.94 ms =  154.854x10^9 interactions/s (  3097.07 GFLOPS)
  GPU_Shuffle:   112.40 ms =  152.845x10^9 interactions/s (  3056.90 GFLOPS)
  GPU_Shuffle:   112.28 ms =  153.015x10^9 interactions/s (  3060.29 GFLOPS)


============================================

CC        = icc
NVCC      = nvcc
CUDA      = 1
LINK      = icc
CFLAGS    = -Ofast -xHOST -no-prec-sqrt -std=gnu11 -fno-strict-aliasing
   -D_GNU_SOURCE -DLIBTIME_STATIC -DUSE_OPENMP -DHAVE_SIMD -DUSE_CUDA
   -DUSE_LIBC11 -I../subprojects/c11/include -I../subprojects/time/include
   -pthread -fPIC -fopenmp
NVCCFLAGS = -O3 -Drestrict= --ftz true -Xcompiler -fPIC
   -gencode=arch=compute_50,code="sm_50,compute_50"
   -gencode=arch=compute_52,code="sm_52,compute_52"
   -gencode=arch=compute_61,code="sm_61,compute_61"
   -gencode=arch=compute_70,code="sm_70,compute_70"
   -gencode=arch=compute_75,code="sm_75,compute_75"
LDFLAGS   = -pthread -fPIC -lomp -static-intel -L/opt/cuda/bin/../lib
   -L/opt/cuda/bin/../lib64 -lcudart ../subprojects/c11/libc11.a
   ../subprojects/time/libtime.a -lm -lrt

Compiler version: icc (ICC) 19.0.1.144 20181018

Binary size: 1810256 bytes

Running simulation with 131072 particles, 32 CPU threads, up to 1 GPUs
      CPU_SOA:   802.22 ms =   21.415x10^9 interactions/s (   428.31 GFLOPS)
      CPU_SOA:   804.57 ms =   21.353x10^9 interactions/s (   427.06 GFLOPS)
      CPU_SOA:   800.29 ms =   21.467x10^9 interactions/s (   429.34 GFLOPS)
CPU_SOA_tiled:   806.50 ms =   21.302x10^9 interactions/s (   426.04 GFLOPS)
CPU_SOA_tiled:   806.47 ms =   21.302x10^9 interactions/s (   426.05 GFLOPS)
CPU_SOA_tiled:   809.66 ms =   21.219x10^9 interactions/s (   424.37 GFLOPS)
          AVX:   799.74 ms =   21.482x10^9 interactions/s (   429.64 GFLOPS)
          AVX:   797.20 ms =   21.550x10^9 interactions/s (   431.01 GFLOPS)
          AVX:   798.18 ms =   21.524x10^9 interactions/s (   430.47 GFLOPS)
      CPU_AOS:  1164.06 ms =   14.759x10^9 interactions/s (   295.17 GFLOPS)
      CPU_AOS:  1161.97 ms =   14.785x10^9 interactions/s (   295.70 GFLOPS)
      CPU_AOS:  1163.00 ms =   14.772x10^9 interactions/s (   295.44 GFLOPS)
CPU_AOS_tiled:  1091.29 ms =   15.743x10^9 interactions/s (   314.85 GFLOPS)
CPU_AOS_tiled:  1090.70 ms =   15.751x10^9 interactions/s (   315.03 GFLOPS)
CPU_AOS_tiled:  1093.98 ms =   15.704x10^9 interactions/s (   314.08 GFLOPS)
      GPU_AOS:   226.54 ms =   75.837x10^9 interactions/s (  1516.75 GFLOPS)
      GPU_AOS:   221.23 ms =   77.657x10^9 interactions/s (  1553.14 GFLOPS)
      GPU_AOS:   222.11 ms =   77.349x10^9 interactions/s (  1546.98 GFLOPS)
   GPU_Shared:   111.88 ms =  153.551x10^9 interactions/s (  3071.01 GFLOPS)
   GPU_Shared:   112.03 ms =  153.351x10^9 interactions/s (  3067.03 GFLOPS)
   GPU_Shared:   112.09 ms =  153.266x10^9 interactions/s (  3065.32 GFLOPS)
    GPU_Const:   104.63 ms =  164.193x10^9 interactions/s (  3283.85 GFLOPS)
    GPU_Const:   104.12 ms =  165.004x10^9 interactions/s (  3300.08 GFLOPS)
    GPU_Const:   103.79 ms =  165.531x10^9 interactions/s (  3310.61 GFLOPS)
     MultiGPU:   106.23 ms =  161.716x10^9 interactions/s (  3234.32 GFLOPS)
     MultiGPU:   105.89 ms =  162.241x10^9 interactions/s (  3244.83 GFLOPS)
     MultiGPU:   106.58 ms =  161.185x10^9 interactions/s (  3223.70 GFLOPS)
  GPU_Shuffle:   109.10 ms =  157.473x10^9 interactions/s (  3149.47 GFLOPS)
  GPU_Shuffle:   111.75 ms =  153.742x10^9 interactions/s (  3074.83 GFLOPS)
  GPU_Shuffle:   114.36 ms =  150.232x10^9 interactions/s (  3004.64 GFLOPS)
```
