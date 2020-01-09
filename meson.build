project('nbody', 'cpp',
    default_options: [
        'buildtype=release',
        'cpp_std=c++11'
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
        '-march=native',
        '-mcpu=native',
    ]),
] + compiler.get_supported_arguments([
    '-fPIC',
    '-fno-exceptions',
    '-fno-rtti',
    '-fvisibility=hidden',
    '-fvisibility-inlines-hidden',
    '-Wno-unknown-pragmas',
    '-fno-asynchronous-unwind-tables',
])
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
        '--cuda-gpu-arch=sm_52',
        '--cuda-gpu-arch=sm_61',
        '--cuda-gpu-arch=sm_70',
    ]
elif nvcc_name == 'nvcc'
    nvcc_flags += [
        '-Xcompiler', '-fPIC',
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
    cxx_sources += custom_target('cuda-to-cpp',
        input: ['src/nbody.cu'],
        output: ['nbody.cpp'],
        command: [copy_bin, '@INPUT@', '@OUTPUT@'])
    cuda_obj = []
endif

if enable_openmp
    openmp = dependency('openmp')
    cpp_args += ['-DUSE_OPENMP']
    extra_deps += [openmp]
endif

if target_machine.system() == 'windows'
    winmm = compiler.find_library('winmm')
    glu = compiler.find_library('glu32')
    extra_deps += [winmm, glu]
endif

executable('nbody',
    cxx_sources,
    cuda_obj,
    cpp_args: cxx_args + cpp_args,
    link_args: ld_args,
    include_directories : incdirs,
    dependencies: [threads, libm] + extra_deps)

# vim: set ts=4 sts=4 sw=4 et:
