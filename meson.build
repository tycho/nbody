project('nbody', 'c',
    default_options: [
        'buildtype=release',
        'c_std=gnu11'
    ]
)

compiler = meson.get_compiler('c')
threads = dependency('threads')
libm = compiler.find_library('m')

incdirs = include_directories('src')
libc11_sp = subproject('c11')
libc11 = libc11_sp.get_variable('c11_dep')
libtime_sp = subproject('time')
libtime = libtime_sp.get_variable('libtime')

enable_cuda = get_option('cuda')
enable_openmp = get_option('openmp')
enable_opengl = get_option('opengl')
nvcc_name = get_option('nvcc')

c_sources = [
    'src/nbody_CPU_AOS.c',
    'src/nbody_CPU_AOS_tiled.c',
    'src/nbody_CPU_AVX.c',
    'src/nbody_CPU_AltiVec.c',
    'src/nbody_CPU_NEON.c',
    'src/nbody_CPU_SOA.c',
    'src/nbody_CPU_SOA_tiled.c',
    'src/nbody_CPU_SSE.c',
    'src/nbody_render_gl.c',
    'src/nbody_util.c',
]
c_args = [
    '-march=native',
    '-ffast-math',
] + compiler.get_supported_arguments([
    '-Wno-unknown-pragmas',
    '-fno-asynchronous-unwind-tables',
])
cpp_args = [
    '-DUSE_LIBC11',
    '-DLIBTIME_STATIC',
    '-DHAVE_SIMD',
    #'-DHIGH_ENTROPY',
]
nvcc_flags = [
    # TODO: These include args should be generated from the nbody target
    # somehow, but meson is a hot mess and doesn't support that kind of thing.
    '-I' + meson.current_source_dir(),
    '-I' + join_paths(meson.current_source_dir(), 'subprojects', 'c11', 'include'),
    '-I' + join_paths(meson.current_source_dir(), 'subprojects', 'time', 'include'),

    '-O3',
]

if nvcc_name == 'clang++'
    nvcc_flags += [
        '-Drestrict=__restrict',
        '--cuda-gpu-arch=sm_52',
        '--cuda-gpu-arch=sm_61',
        '--cuda-gpu-arch=sm_70',
    ]
elif nvcc_name == 'nvcc'
    nvcc_flags += [
        '-Drestrict=',
        '-gencode=arch=compute_52,code="sm_52,compute_52"',
        '-gencode=arch=compute_61,code="sm_61,compute_61"',
        '-gencode=arch=compute_70,code="sm_70,compute_70"'
    ]
endif

extra_deps = []

if enable_opengl
    extra_deps += [
        dependency('gl'),
        dependency('glew'),
        dependency('sdl2'),
    ]
    cpp_args +=['-DUSE_GL']
endif

if enable_cuda
    cpp_args += ['-DUSE_CUDA']
    cuda_sources = [
        'src/nbody.cu',
        'src/nbody_GPU_shared.cu',
        'src/nbody_multiGPU.cu'
    ]
    nvcc_bin = find_program([nvcc_name])
    cuda_gen = generator(nvcc_bin,
        output: '@BASENAME@.o',
        arguments: cpp_args + nvcc_flags +  ['-c', '@INPUT@', '-o', '@OUTPUT@'])
    cuda_obj = cuda_gen.process(cuda_sources)
    cudart_lib = compiler.find_library('cudart', dirs: ['/opt/cuda/lib64', '/usr/local/cuda/lib64'])
    extra_deps += [cudart_lib]
else
    cuda_sources = [
        'src/nbody.cu'
    ]
    copy_bin = find_program('cp', 'copy')
    c_sources += custom_target('cuda-to-c',
        input: ['src/nbody.cu'],
        output: ['nbody.c'],
        command: [copy_bin, '@INPUT@', '@OUTPUT@'])
    cuda_obj = []
endif

if enable_openmp
    cpp_args += ['-DUSE_OPENMP']
    omp_test_code = '''#include <omp.h>
    #include <stdio.h>

    #ifndef _OPENMP
    #error "compiler didn't define _OPENMP"
    #endif

    static inline int threadCount(void)
    {
        int k;
    #  pragma omp parallel
        {
    #  pragma omp master
            {
                k = omp_get_num_threads();
            }
        }
        return k;
    }

    int main(int argc, char **argv)
    {
            printf("thread count: %d\n", threadCount());
            return 0;
    }
    '''
    openmp_cflag_try = [
      '-fopenmp=libomp',
      '-fopenmp=libiomp5',
      '-fopenmp=libgomp',
      '-fopenmp'
    ]
    omp_header = compiler.has_header('omp.h')
    omp_lib = compiler.find_library('omp', required: false)
    if omp_lib.found() == false
        omp_lib = compiler.find_library('iomp5', required: false)
        if omp_lib.found() == false
            omp_lib = compiler.find_library('gomp', required: true)
        endif
    endif
    omp_arg = compiler.first_supported_argument(openmp_cflag_try)
    result = compiler.run(omp_test_code, args: omp_arg, name: 'OpenMP test program', dependencies: omp_lib)
    cpp_args += [omp_arg]
    extra_deps += [omp_lib]
endif

if target_machine.system() == 'windows'
    winmm = compiler.find_library('winmm')
    glu = compiler.find_library('glu32')
    extra_deps += [winmm, glu]
endif

executable('nbody',
    c_sources,
    cuda_obj,
    c_args: c_args + cpp_args,
    include_directories : incdirs,
    dependencies: [threads, libm, libc11, libtime] + extra_deps)

# vim: set ts=4 sts=4 sw=4 et:
