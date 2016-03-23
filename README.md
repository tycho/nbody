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
