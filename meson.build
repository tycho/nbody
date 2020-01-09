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

enable_cuda = get_option('use_cuda')
enable_openmp = get_option('openmp')
enable_opengl = get_option('opengl')
enable_simd = get_option('intrinsics')
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
    '-ffast-math',
] + compiler.get_supported_arguments([
    '-march=native',
    '-mcpu=native',
    '-Wno-unknown-pragmas',
    '-fno-asynchronous-unwind-tables',
])
cpp_args = [
    '-DUSE_LIBC11',
    '-DLIBTIME_STATIC',
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
        '-fcuda-flush-denormals-to-zero',
        '--cuda-gpu-arch=sm_52',
        '--cuda-gpu-arch=sm_61',
        '--cuda-gpu-arch=sm_70',
    ]
elif nvcc_name == 'nvcc'
    nvcc_flags += [
        '-Drestrict=',
        '--ftz', 'true',
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
    cpp_args += ['-DUSE_GL']
endif

if enable_simd
    cpp_args += ['-DHAVE_SIMD']
endif

if enable_cuda
    cpp_args += ['-DUSE_CUDA']
    cuda_sources = [
        'src/nbody.cu',
        'src/nbody_GPU_AOS.cu',
        'src/nbody_GPU_AOS_const.cu',
        'src/nbody_GPU_AOS_tiled.cu',
        'src/nbody_GPU_AOS_tiled_const.cu',
        'src/nbody_GPU_Atomic.cu',
        'src/nbody_GPU_Shared.cu',
        'src/nbody_GPU_Shuffle.cu',
        'src/nbody_GPU_SOA_tiled.cu',
        'src/nbody_multiGPU.cu',
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
    openmp_cflag_try = [
      '-fopenmp=libomp',
      '-fopenmp=libiomp5',
      '-fopenmp=libgomp',
      '-fopenmp'
    ]
    omp_arg = compiler.first_supported_argument(openmp_cflag_try)
    omp_header = compiler.has_header('omp.h')
    assert(omp_header, 'omp.h header is missing')
    omp_lib = compiler.find_library('omp', required: false)
    omp_ok = omp_lib.found() and run_command('src/openmp-detect-meson.sh', compiler.cmd_array() + [omp_arg, '-lomp']).returncode() == 0
    if not omp_ok
        omp_lib = compiler.find_library('iomp5', required: false)
        omp_ok = omp_lib.found() and run_command('src/openmp-detect-meson.sh', compiler.cmd_array() + [omp_arg, '-liomp5']).returncode() == 0
        if not omp_ok
            omp_lib = compiler.find_library('gomp', required: true)
            omp_ok = omp_lib.found() and run_command('src/openmp-detect-meson.sh', compiler.cmd_array() + [omp_arg, '-lgomp']).returncode() == 0
            if not omp_ok
                omp_lib = []
                omp_ok = run_command('src/openmp-detect-meson.sh', compiler.cmd_array() + [omp_arg, omp_arg]).returncode() == 0
            endif
        endif
    endif
    assert(omp_ok, 'Could not find working OpenMP functionality in your compiler')
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
