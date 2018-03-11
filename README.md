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
$ scripts/compare-compilers --bodies 64 --no-crosscheck --iterations 1 --cycle-after 3

CPU information:
    Architecture:        x86_64
    CPU op-mode(s):      32-bit, 64-bit
    Byte Order:          Little Endian
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
    CPU MHz:             1418.351
    CPU max MHz:         2401.0000
    CPU min MHz:         1200.0000
    BogoMIPS:            4800.03
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
       pti intel_ppin tpr_shadow vnmi flexpriority ept vpid fsgsbase
       tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm xsaveopt cqm_llc
       cqm_occup_llc dtherm ida arat pln pts

OS version: Linux 4.15.7-1-hsw x86_64

============================================

CC      = gcc
LINK    = gcc
CFLAGS  = -O3 -march=native -ffast-math -std=gnu11 -fno-strict-aliasing
   -D_GNU_SOURCE -DLIBTIME_STATIC -DUSE_OPENMP -DHAVE_SIMD -DUSE_LIBC11
   -I../subprojects/c11/include -I../subprojects/time/include
   -Wall -Wdeclaration-after-statement -Werror=implicit
   -Werror=undef -Wmissing-declarations -Wmissing-prototypes
   -Wno-declaration-after-statement -Wno-long-long -Wno-overlength-strings
   -Wno-unknown-pragmas -Wold-style-definition -Wstrict-prototypes -pthread
   -fPIC -fopenmp
LDFLAGS = -pthread -fPIC -lomp ../subprojects/c11/libc11.a
   ../subprojects/time/libtime.a -lm -lrt

Compiler version: gcc (GCC) 7.3.1 20180307

Binary size: 50568 bytes

Running simulation with 65536 particles, 32 CPU threads
      CPU_SOA:   212.08 ms =   20.251x10^9 interactions/s (   405.03 GFLOPS)
      CPU_SOA:   211.41 ms =   20.316x10^9 interactions/s (   406.32 GFLOPS)
      CPU_SOA:   212.42 ms =   20.219x10^9 interactions/s (   404.38 GFLOPS)
CPU_SOA_tiled:   214.81 ms =   19.994x10^9 interactions/s (   399.88 GFLOPS)
CPU_SOA_tiled:   218.04 ms =   19.698x10^9 interactions/s (   393.96 GFLOPS)
CPU_SOA_tiled:   223.74 ms =   19.196x10^9 interactions/s (   383.92 GFLOPS)
          AVX:   204.43 ms =   21.009x10^9 interactions/s (   420.18 GFLOPS)
          AVX:   204.25 ms =   21.028x10^9 interactions/s (   420.57 GFLOPS)
          AVX:   204.19 ms =   21.034x10^9 interactions/s (   420.68 GFLOPS)
      CPU_AOS:   619.51 ms =    6.933x10^9 interactions/s (   138.66 GFLOPS)
      CPU_AOS:   619.09 ms =    6.938x10^9 interactions/s (   138.75 GFLOPS)
      CPU_AOS:   617.79 ms =    6.952x10^9 interactions/s (   139.04 GFLOPS)
CPU_AOS_tiled:   624.33 ms =    6.879x10^9 interactions/s (   137.59 GFLOPS)
CPU_AOS_tiled:   616.61 ms =    6.966x10^9 interactions/s (   139.31 GFLOPS)
CPU_AOS_tiled:   616.02 ms =    6.972x10^9 interactions/s (   139.44 GFLOPS)


============================================

CC      = clang
LINK    = clang
CFLAGS  = -O3 -march=native -ffast-math -std=gnu11 -fno-strict-aliasing
   -D_GNU_SOURCE -DLIBTIME_STATIC -DUSE_OPENMP -DHAVE_SIMD -DUSE_LIBC11
   -I../subprojects/c11/include -I../subprojects/time/include
   -Wall -Wdeclaration-after-statement -Werror=implicit
   -Werror=undef -Wmissing-declarations -Wmissing-prototypes
   -Wno-declaration-after-statement -Wno-long-long -Wno-overlength-strings
   -Wno-unknown-pragmas -Wold-style-definition -Wstrict-prototypes
   -pthread -fPIC -idirafter /usr/lib/gcc/x86_64-pc-linux-gnu/7.3.1/include
   -fopenmp=libomp
LDFLAGS = -pthread -fPIC -lomp ../subprojects/c11/libc11.a
   ../subprojects/time/libtime.a -lm -lrt

Compiler version: clang version 6.0.0 (tags/RELEASE_600/final)

Binary size: 49712 bytes

Running simulation with 65536 particles, 32 CPU threads
      CPU_SOA:   222.51 ms =   19.302x10^9 interactions/s (   386.04 GFLOPS)
      CPU_SOA:   223.28 ms =   19.236x10^9 interactions/s (   384.72 GFLOPS)
      CPU_SOA:   223.66 ms =   19.203x10^9 interactions/s (   384.06 GFLOPS)
CPU_SOA_tiled:   234.22 ms =   18.338x10^9 interactions/s (   366.75 GFLOPS)
CPU_SOA_tiled:   230.62 ms =   18.624x10^9 interactions/s (   372.48 GFLOPS)
CPU_SOA_tiled:   231.35 ms =   18.564x10^9 interactions/s (   371.29 GFLOPS)
          AVX:   223.30 ms =   19.234x10^9 interactions/s (   384.68 GFLOPS)
          AVX:   223.22 ms =   19.241x10^9 interactions/s (   384.82 GFLOPS)
          AVX:   223.16 ms =   19.246x10^9 interactions/s (   384.93 GFLOPS)
      CPU_AOS:   437.75 ms =    9.811x10^9 interactions/s (   196.23 GFLOPS)
      CPU_AOS:   436.60 ms =    9.837x10^9 interactions/s (   196.74 GFLOPS)
      CPU_AOS:   438.13 ms =    9.803x10^9 interactions/s (   196.06 GFLOPS)
CPU_AOS_tiled:   436.50 ms =    9.840x10^9 interactions/s (   196.79 GFLOPS)
CPU_AOS_tiled:   444.67 ms =    9.659x10^9 interactions/s (   193.18 GFLOPS)
CPU_AOS_tiled:   436.54 ms =    9.839x10^9 interactions/s (   196.78 GFLOPS)


============================================

CC      = icc
LINK    = icc
CFLAGS  = -Ofast -xHOST -no-prec-sqrt -std=gnu11 -fno-strict-aliasing
   -D_GNU_SOURCE -DLIBTIME_STATIC -DUSE_OPENMP -DHAVE_SIMD -DUSE_LIBC11
   -I../subprojects/c11/include -I../subprojects/time/include  -pthread
   -fPIC -fopenmp
LDFLAGS = -pthread -fPIC -lomp -static-intel ../subprojects/c11/libc11.a
   ../subprojects/time/libtime.a -lm -lrt

Compiler version: icc (ICC) 18.0.1 20171018

Binary size: 91640 bytes

Running simulation with 65536 particles, 32 CPU threads
      CPU_SOA:   202.46 ms =   21.214x10^9 interactions/s (   424.27 GFLOPS)
      CPU_SOA:   202.30 ms =   21.231x10^9 interactions/s (   424.61 GFLOPS)
      CPU_SOA:   202.35 ms =   21.226x10^9 interactions/s (   424.52 GFLOPS)
CPU_SOA_tiled:   199.01 ms =   21.582x10^9 interactions/s (   431.64 GFLOPS)
CPU_SOA_tiled:   200.82 ms =   21.387x10^9 interactions/s (   427.74 GFLOPS)
CPU_SOA_tiled:   208.29 ms =   20.620x10^9 interactions/s (   412.40 GFLOPS)
          AVX:   202.08 ms =   21.254x10^9 interactions/s (   425.07 GFLOPS)
          AVX:   202.20 ms =   21.241x10^9 interactions/s (   424.83 GFLOPS)
          AVX:   201.88 ms =   21.275x10^9 interactions/s (   425.51 GFLOPS)
      CPU_AOS:   281.19 ms =   15.274x10^9 interactions/s (   305.48 GFLOPS)
      CPU_AOS:   279.57 ms =   15.363x10^9 interactions/s (   307.25 GFLOPS)
      CPU_AOS:   279.66 ms =   15.358x10^9 interactions/s (   307.16 GFLOPS)
CPU_AOS_tiled:   202.61 ms =   21.198x10^9 interactions/s (   423.97 GFLOPS)
CPU_AOS_tiled:   209.71 ms =   20.480x10^9 interactions/s (   409.60 GFLOPS)
CPU_AOS_tiled:   211.19 ms =   20.337x10^9 interactions/s (   406.74 GFLOPS)
```
