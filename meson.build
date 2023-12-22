project('nbody', 'cpp',
    default_options: [
        'buildtype=release',
        'cpp_std=c++11',
        'b_lto=true',
    ]
)

compiler = meson.get_compiler('cpp')
threads = dependency('threads')
libm = compiler.find_library('m')

incdirs = include_directories('src')

enable_cuda = get_option('use_cuda')
enable_openmp = get_option('openmp')
enable_opengl = get_option('opengl')
enable_simd = get_option('intrinsics')
nvcc_name = get_option('nvcc')

cxx_sources = [
    'src/nbody_CPU_AOS.cpp',
    'src/nbody_CPU_AOS_tiled.cpp',
    'src/nbody_CPU_AVX.cpp',
    'src/nbody_CPU_AltiVec.cpp',
    'src/nbody_CPU_NEON.cpp',
    'src/nbody_CPU_SOA.cpp',
    'src/nbody_CPU_SOA_tiled.cpp',
    'src/nbody_CPU_SSE.cpp',
    'src/nbody_render_gl.cpp',
    'src/nbody_util.cpp',
]
cxx_args = [
    '-ffast-math',
    compiler.first_supported_argument([
        '-march=' + get_option('march'),
        '-mcpu=' + get_option('march'),
    ]),
] + compiler.get_supported_arguments([
    '-gdwarf-4',
    '-ggdb3',
    '-fPIC',
    '-fwrapv',
    '-fno-stack-protector',
    '-fno-exceptions',
    '-fno-rtti',
    '-fvisibility=hidden',
    '-fvisibility-inlines-hidden',
    '-Wno-unknown-pragmas',
    '-fno-asynchronous-unwind-tables',
])

if target_machine.cpu_family() == 'aarch64'
    cxx_args += compiler.get_supported_arguments([
        '-mlow-precision-recip-sqrt',
        '-mtune=' + get_option('march'),
    ])
endif

ld_args = [ '-fPIC' ]
cpp_args = [
    #'-DHIGH_ENTROPY',
]
nvcc_flags = [
    '-I' + meson.current_source_dir(),

    '-O3',
]

if nvcc_name == 'clang++'
    nvcc_flags += [
        '-fPIC',
        '-std=c++11',
        '-fcuda-flush-denormals-to-zero',
        '--cuda-gpu-arch=sm_60',
        '--cuda-gpu-arch=sm_61',
        '--cuda-gpu-arch=sm_62',
        '--cuda-gpu-arch=sm_70',
        '--cuda-gpu-arch=sm_72',
        '--cuda-gpu-arch=sm_75',
        '--cuda-gpu-arch=sm_80',
        '--cuda-gpu-arch=sm_86',
        '--cuda-gpu-arch=sm_87',
        '--cuda-gpu-arch=sm_89',
        '--cuda-gpu-arch=sm_90',
        '--cuda-gpu-arch=sm_90a',
    ]
elif nvcc_name == 'nvcc'
    nvcc_flags += [
        '-Xcompiler', '-fPIC',
        '--ftz', 'true',
        '-gencode=arch=compute_60,code="sm_60,compute_60"',
        '-gencode=arch=compute_61,code="sm_61,compute_61"',
        '-gencode=arch=compute_62,code="sm_62,compute_62"',
        '-gencode=arch=compute_70,code="sm_70,compute_70"',
        '-gencode=arch=compute_72,code="sm_72,compute_72"',
        '-gencode=arch=compute_75,code="sm_75,compute_75"',
        '-gencode=arch=compute_80,code="sm_80,compute_80"',
        '-gencode=arch=compute_86,code="sm_86,compute_86"',
        '-gencode=arch=compute_87,code="sm_87,compute_87"',
        '-gencode=arch=compute_89,code="sm_89,compute_89"',
        '-gencode=arch=compute_90,code="sm_90,compute_90"',
        '-gencode=arch=compute_90a,code="sm_90a,compute_90a"',
    ]
endif

extra_deps = []

if enable_opengl
    extra_deps += [
        dependency('glew'),
        dependency('sdl2'),
        dependency('gl'),
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
    cxx_sources += custom_target('cuda-to-cpp',
        input: ['src/nbody.cu'],
        output: ['nbody.cpp'],
        command: [copy_bin, '@INPUT@', '@OUTPUT@'])
    cuda_obj = []
endif

if enable_openmp
    openmp = dependency('openmp', required: false)
    cpp_args += ['-DUSE_OPENMP']
    if openmp.found()
        extra_deps += [openmp]
    else
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
        omp_ok = omp_lib.found() and run_command('src/openmp-detect.sh', compiler.cmd_array() + [omp_arg, '-lomp']).returncode() == 0
        if not omp_ok
            omp_lib = compiler.find_library('iomp5', required: false)
            omp_ok = omp_lib.found() and run_command('src/openmp-detect.sh', compiler.cmd_array() + [omp_arg, '-liomp5']).returncode() == 0
            if not omp_ok
                omp_lib = compiler.find_library('gomp', required: true)
                omp_ok = omp_lib.found() and run_command('src/openmp-detect.sh', compiler.cmd_array() + [omp_arg, '-lgomp']).returncode() == 0
                if not omp_ok
                    omp_lib = []
                    omp_ok = run_command('src/openmp-detect.sh', compiler.cmd_array() + [omp_arg, omp_arg]).returncode() == 0
                endif
            endif
        endif
        assert(omp_ok, 'Could not find working OpenMP functionality in your compiler')
        cpp_args += [omp_arg]
        extra_deps += [omp_lib]
    endif
endif

if target_machine.system() == 'windows'
    winmm = compiler.find_library('winmm')
    extra_deps += [winmm]
endif

executable('nbody',
    cxx_sources,
    cuda_obj,
    cpp_args: cxx_args + cpp_args,
    link_args: ld_args,
    include_directories : incdirs,
    dependencies: [threads, libm] + extra_deps)

# vim: set ts=4 sts=4 sw=4 et:
