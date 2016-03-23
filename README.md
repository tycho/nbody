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
and AOS (Array-of-Structures). An excellent explanation of the difference
between the two can be found [here](http://divergentcoder.com/Coding/2011/02/22/aos-soa-explorations-part-1.html).

There's a key tradeoff between the two: compilers have a much less difficult
time auto-vectorizing the SOA implementations, but the AOS implementations are
easier to read and write. It's not uncommon to see AOS algorithms applied in
real-world applications (consider, for example, how most object-oriented
programs are constructed).

Most of the code is plain C, with some added pragmas to provide compilers with
appropriate hints to vectorize/parallelize the code properly.


Building
--------

The simplest way to build n-body is to clone the repository and simply run:

```
$ make
```

But there are a few options you can use to customize the build as you see fit.
OpenMP and CUDA are enabled automatically if the build system detects your
system supports them, but you can disable one or both of them with some command
line options:

```
$ make NO_CUDA=1 NO_OPENMP=1
```

The intrinsics-based implementation can also be disabled with:

```
$ make NO_SIMD=1
```

You can select your compiler by specifying it on the command line as well:

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
$ scripts/compare-compilers --bodies 64 --no-crosscheck --iterations 1 --cycle-after 3

CPU information:
    Architecture:          x86_64
    CPU op-mode(s):        32-bit, 64-bit
    Byte Order:            Little Endian
    CPU(s):                32
    On-line CPU(s) list:   0-31
    Thread(s) per core:    2
    Core(s) per socket:    8
    Socket(s):             2
    NUMA node(s):          2
    Vendor ID:             GenuineIntel
    CPU family:            6
    Model:                 63
    Model name:            Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz
    Stepping:              2
    CPU MHz:               2401.000
    CPU max MHz:           2401.0000
    CPU min MHz:           1200.0000
    BogoMIPS:              4802.19
    Virtualization:        VT-x
    L1d cache:             32K
    L1i cache:             32K
    L2 cache:              256K
    L3 cache:              20480K
    NUMA node0 CPU(s):     0-7,16-23
    NUMA node1 CPU(s):     8-15,24-31
    Flags:                 fpu vme de pse tsc msr pae mce cx8 apic sep
       mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss
       ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon
       pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu
       pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg
       fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt
       tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm epb
       tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1
       avx2 smep bmi2 erms invpcid cqm xsaveopt cqm_llc cqm_occup_llc
       dtherm ida arat pln pts

OS version: Linux 4.5.0-2-ec2 x86_64

============================================

CC      = gcc
LINK    = gcc
CFLAGS  = -O3 -march=native -ffast-math -std=gnu11 -fno-strict-aliasing
   -D_GNU_SOURCE -DLIBTIME_STATIC -DUSE_OPENMP -DHAVE_SIMD
   -DNO_CUDA -DUSE_LIBC11 -I../libc11/include -I../libtime/include
   -Wall -Wdeclaration-after-statement -Werror=implicit -Werror=undef
   -Wmissing-declarations -Wmissing-prototypes -Wno-declaration-after-statement
   -Wno-long-long -Wno-overlength-strings -Wno-unknown-pragmas
   -Wold-style-definition -Wstrict-prototypes -pthread -fopenmp
LDFLAGS = -pthread -L/usr/lib -liomp5 -lm -lrt ../libc11/libc11.a
   ../libtime/libtime.a

Compiler version: gcc (GCC) 5.3.0

Running simulation with 65536 particles, crosscheck disabled, CPU enabled, 32 threads
      CPU_SOA:   213.18 ms =   20.147x10^9 interactions/s (   402.95 GFLOPS)
      CPU_SOA:   214.09 ms =   20.061x10^9 interactions/s (   401.22 GFLOPS)
      CPU_SOA:   214.35 ms =   20.037x10^9 interactions/s (   400.74 GFLOPS)
CPU_SOA_tiled:   204.92 ms =   20.959x10^9 interactions/s (   419.18 GFLOPS)
CPU_SOA_tiled:   205.96 ms =   20.853x10^9 interactions/s (   417.07 GFLOPS)
CPU_SOA_tiled:   204.93 ms =   20.958x10^9 interactions/s (   419.17 GFLOPS)
   AVX intrin:   202.31 ms =   21.230x10^9 interactions/s (   424.60 GFLOPS)
   AVX intrin:   200.21 ms =   21.452x10^9 interactions/s (   429.05 GFLOPS)
   AVX intrin:   202.52 ms =   21.208x10^9 interactions/s (   424.15 GFLOPS)
      CPU_AOS:   610.13 ms =    7.039x10^9 interactions/s (   140.79 GFLOPS)
      CPU_AOS:   610.03 ms =    7.041x10^9 interactions/s (   140.81 GFLOPS)
      CPU_AOS:   610.22 ms =    7.038x10^9 interactions/s (   140.77 GFLOPS)
CPU_AOS_tiled:   611.14 ms =    7.028x10^9 interactions/s (   140.56 GFLOPS)
CPU_AOS_tiled:   611.12 ms =    7.028x10^9 interactions/s (   140.56 GFLOPS)
CPU_AOS_tiled:   611.48 ms =    7.024x10^9 interactions/s (   140.48 GFLOPS)


============================================

CC      = /usr/bin/clang
LINK    = /usr/bin/clang
CFLAGS  = -O3 -march=native -ffast-math -std=gnu11 -fno-strict-aliasing
   -D_GNU_SOURCE -DLIBTIME_STATIC -DUSE_OPENMP -DHAVE_SIMD
   -DNO_CUDA -DUSE_LIBC11 -I../libc11/include -I../libtime/include
   -Wall -Wdeclaration-after-statement -Werror=implicit -Werror=undef
   -Wmissing-declarations -Wmissing-prototypes -Wno-declaration-after-statement
   -Wno-long-long -Wno-overlength-strings -Wno-unknown-pragmas
   -Wold-style-definition -Wstrict-prototypes -pthread -fopenmp=libiomp5
   -D_OPENMP=201107 -Wno-macro-redefined
LDFLAGS = -pthread -L/usr/lib -liomp5 -lm -lrt ../libc11/libc11.a
   ../libtime/libtime.a

Compiler version: clang version 3.7.0 (tags/RELEASE_370/final)

Running simulation with 65536 particles, crosscheck disabled, CPU enabled, 32 threads
      CPU_SOA:   180.74 ms =   23.763x10^9 interactions/s (   475.27 GFLOPS)
      CPU_SOA:   181.26 ms =   23.696x10^9 interactions/s (   473.91 GFLOPS)
      CPU_SOA:   181.42 ms =   23.674x10^9 interactions/s (   473.48 GFLOPS)
CPU_SOA_tiled:   177.51 ms =   24.195x10^9 interactions/s (   483.91 GFLOPS)
CPU_SOA_tiled:   177.49 ms =   24.198x10^9 interactions/s (   483.97 GFLOPS)
CPU_SOA_tiled:   178.94 ms =   24.003x10^9 interactions/s (   480.06 GFLOPS)
   AVX intrin:   207.13 ms =   20.735x10^9 interactions/s (   414.71 GFLOPS)
   AVX intrin:   206.88 ms =   20.761x10^9 interactions/s (   415.21 GFLOPS)
   AVX intrin:   206.92 ms =   20.756x10^9 interactions/s (   415.13 GFLOPS)
      CPU_AOS:   565.27 ms =    7.598x10^9 interactions/s (   151.96 GFLOPS)
      CPU_AOS:   565.41 ms =    7.596x10^9 interactions/s (   151.92 GFLOPS)
      CPU_AOS:   565.55 ms =    7.594x10^9 interactions/s (   151.89 GFLOPS)
CPU_AOS_tiled:   568.02 ms =    7.561x10^9 interactions/s (   151.23 GFLOPS)
CPU_AOS_tiled:   568.01 ms =    7.561x10^9 interactions/s (   151.23 GFLOPS)
CPU_AOS_tiled:   568.00 ms =    7.562x10^9 interactions/s (   151.23 GFLOPS)


============================================

CC      = icc
LINK    = icc
CFLAGS  = -Ofast -xHOST -no-prec-sqrt -std=gnu11 -fno-strict-aliasing
   -D_GNU_SOURCE -DLIBTIME_STATIC -DUSE_OPENMP -DHAVE_SIMD -DNO_CUDA
   -DUSE_LIBC11 -I../libc11/include -I../libtime/include  -pthread -fopenmp
LDFLAGS = -pthread -L/usr/lib -liomp5 -lm -lrt -static-intel ../libc11/libc11.a
   ../libtime/libtime.a

Compiler version: icc (ICC) 16.0.2 20160204

Running simulation with 65536 particles, crosscheck disabled, CPU enabled, 32 threads
      CPU_SOA:   195.51 ms =   21.968x10^9 interactions/s (   439.35 GFLOPS)
      CPU_SOA:   195.43 ms =   21.977x10^9 interactions/s (   439.55 GFLOPS)
      CPU_SOA:   196.90 ms =   21.813x10^9 interactions/s (   436.25 GFLOPS)
CPU_SOA_tiled:   198.60 ms =   21.626x10^9 interactions/s (   432.53 GFLOPS)
CPU_SOA_tiled:   198.59 ms =   21.628x10^9 interactions/s (   432.55 GFLOPS)
CPU_SOA_tiled:   198.70 ms =   21.615x10^9 interactions/s (   432.31 GFLOPS)
   AVX intrin:   198.88 ms =   21.596x10^9 interactions/s (   431.91 GFLOPS)
   AVX intrin:   199.02 ms =   21.581x10^9 interactions/s (   431.61 GFLOPS)
   AVX intrin:   198.47 ms =   21.641x10^9 interactions/s (   432.81 GFLOPS)
      CPU_AOS:   271.50 ms =   15.819x10^9 interactions/s (   316.39 GFLOPS)
      CPU_AOS:   270.97 ms =   15.850x10^9 interactions/s (   317.00 GFLOPS)
      CPU_AOS:   271.09 ms =   15.844x10^9 interactions/s (   316.87 GFLOPS)
CPU_AOS_tiled:   206.50 ms =   20.799x10^9 interactions/s (   415.97 GFLOPS)
CPU_AOS_tiled:   204.26 ms =   21.027x10^9 interactions/s (   420.55 GFLOPS)
CPU_AOS_tiled:   204.21 ms =   21.033x10^9 interactions/s (   420.65 GFLOPS)
```
