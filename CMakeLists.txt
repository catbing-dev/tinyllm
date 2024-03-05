cmake_minimum_required(VERSION 3.14)  # for add_link_options and implicit target directories.
project("llama.cpp" C CXX)
include(CheckIncludeFileCXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LLAMA_STANDALONE ON)

    # configure project version
    # TODO
else()
    set(LLAMA_STANDALONE OFF)
endif()

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(LLAMA_WASM_SINGLE_FILE "llama: embed WASM inside the generated llama.js" ON)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()


#
# Option list
#

# general
option(BUILD_SHARED_LIBS                "build shared libraries"                                OFF)
option(LLAMA_STATIC                     "llama: static link libraries"                          OFF)
option(LLAMA_NATIVE                     "llama: enable -march=native flag"                      ON)
option(LLAMA_LTO                        "llama: enable link time optimization"                  OFF)
option(LLAMA_CCACHE                     "llama: use ccache if available"                        ON)

# debug
option(LLAMA_ALL_WARNINGS               "llama: enable all compiler warnings"                   ON)
option(LLAMA_ALL_WARNINGS_3RD_PARTY     "llama: enable all compiler warnings in 3rd party libs" OFF)
option(LLAMA_GPROF                      "llama: enable gprof"                                   OFF)

# build
option(LLAMA_FATAL_WARNINGS             "llama: enable -Werror flag"                            OFF)

# sanitizers
option(LLAMA_SANITIZE_THREAD            "llama: enable thread sanitizer"                        OFF)
option(LLAMA_SANITIZE_ADDRESS           "llama: enable address sanitizer"                       OFF)
option(LLAMA_SANITIZE_UNDEFINED         "llama: enable undefined sanitizer"                     OFF)

# instruction set specific
if (LLAMA_NATIVE)
    set(INS_ENB OFF)
else()
    set(INS_ENB ON)
endif()

option(LLAMA_AVX                             "llama: enable AVX"                                ${INS_ENB})
option(LLAMA_AVX2                            "llama: enable AVX2"                               ${INS_ENB})
option(LLAMA_AVX512                          "llama: enable AVX512"                             OFF)
option(LLAMA_AVX512_VBMI                     "llama: enable AVX512-VBMI"                        OFF)
option(LLAMA_AVX512_VNNI                     "llama: enable AVX512-VNNI"                        OFF)
option(LLAMA_FMA                             "llama: enable FMA"                                ${INS_ENB})
# in MSVC F16C is implied with AVX2/AVX512
if (NOT MSVC)
    option(LLAMA_F16C                        "llama: enable F16C"                               ${INS_ENB})
endif()

if (WIN32)
    set(LLAMA_WIN_VER "0x602" CACHE STRING "llama: Windows Version")
endif()

# 3rd party libs
option(LLAMA_BLAS                            "llama: use BLAS"                                  OFF)
set(LLAMA_BLAS_VENDOR "Generic" CACHE STRING "llama: BLAS library vendor")
option(LLAMA_CUBLAS                          "llama: use CUDA"                                  OFF)
#option(LLAMA_CUDA_CUBLAS                     "llama: use cuBLAS for prompt processing"          OFF)
option(LLAMA_CUDA_FORCE_DMMV                 "llama: use dmmv instead of mmvq CUDA kernels"     OFF)
option(LLAMA_CUDA_FORCE_MMQ                  "llama: use mmq kernels instead of cuBLAS"         OFF)
set(LLAMA_CUDA_DMMV_X      "32" CACHE STRING "llama: x stride for dmmv CUDA kernels")
set(LLAMA_CUDA_MMV_Y        "1" CACHE STRING "llama: y block size for mmv CUDA kernels")
option(LLAMA_CUDA_F16                        "llama: use 16 bit floats for some calculations"   OFF)
set(LLAMA_CUDA_KQUANTS_ITER "2" CACHE STRING "llama: iters./thread per block for Q2_K/Q6_K")
set(LLAMA_CUDA_PEER_MAX_BATCH_SIZE "128" CACHE STRING
                                             "llama: max. batch size for using peer access")
option(LLAMA_CLBLAST                         "llama: use CLBlast"                               OFF)
option(LLAMA_QKK_64                          "llama: use super-block size of 64 for k-quants"   OFF)
option(LLAMA_CPU_HBM                         "llama: use memkind for CPU HBM"                   OFF)

option(LLAMA_BUILD_EXAMPLES                  "llama: build examples" ${LLAMA_STANDALONE})

# add perf arguments
option(LLAMA_PERF                            "llama: enable perf"                               OFF)

#
# Compile flags
#

set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)
include(CheckCXXCompilerFlag)

# enable libstdc++ assertions for debug builds
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions($<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
endif()

if (NOT MSVC)
    if (LLAMA_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries     (-fsanitize=thread)
    endif()

    if (LLAMA_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries     (-fsanitize=address)
    endif()

    if (LLAMA_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries     (-fsanitize=undefined)
    endif()
endif()

if (LLAMA_BLAS)
    if (LLAMA_STATIC)
        set(BLA_STATIC ON)
    endif()
    if ($(CMAKE_VERSION) VERSION_GREATER_EQUAL 3.22)
        set(BLA_SIZEOF_INTEGER 8)
    endif()

    set(BLA_VENDOR ${LLAMA_BLAS_VENDOR})
    find_package(BLAS)

    if (BLAS_FOUND)
        message(STATUS "BLAS found, Libraries: ${BLAS_LIBRARIES}")

        if ("${BLAS_INCLUDE_DIRS}" STREQUAL "")
            # BLAS_INCLUDE_DIRS is missing in FindBLAS.cmake.
            # see https://gitlab.kitware.com/cmake/cmake/-/issues/20268
            find_package(PkgConfig REQUIRED)
            if (${LLAMA_BLAS_VENDOR} MATCHES "Generic")
                pkg_check_modules(DepBLAS REQUIRED blas)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "OpenBLAS")
                # As of openblas v0.3.22, the 64-bit is named openblas64.pc
                pkg_check_modules(DepBLAS openblas64)
                if (NOT DepBLAS_FOUND)
                    pkg_check_modules(DepBLAS REQUIRED openblas)
                endif()
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "FLAME")
                pkg_check_modules(DepBLAS REQUIRED blis)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "ATLAS")
                pkg_check_modules(DepBLAS REQUIRED blas-atlas)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "FlexiBLAS")
                pkg_check_modules(DepBLAS REQUIRED flexiblas_api)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "Intel")
                # all Intel* libraries share the same include path
                pkg_check_modules(DepBLAS REQUIRED mkl-sdl)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "NVHPC")
                # this doesn't provide pkg-config
                # suggest to assign BLAS_INCLUDE_DIRS on your own
                if ("${NVHPC_VERSION}" STREQUAL "")
                    message(WARNING "Better to set NVHPC_VERSION")
                else()
                    set(DepBLAS_FOUND ON)
                    set(DepBLAS_INCLUDE_DIRS "/opt/nvidia/hpc_sdk/${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}/${NVHPC_VERSION}/math_libs/include")
                endif()
            endif()
            if (DepBLAS_FOUND)
                set(BLAS_INCLUDE_DIRS ${DepBLAS_INCLUDE_DIRS})
            else()
                message(WARNING "BLAS_INCLUDE_DIRS neither been provided nor been automatically"
                " detected by pkgconfig, trying to find cblas.h from possible paths...")
                find_path(BLAS_INCLUDE_DIRS
                    NAMES cblas.h
                    HINTS
                        /usr/include
                        /usr/local/include
                        /usr/include/openblas
                        /opt/homebrew/opt/openblas/include
                        /usr/local/opt/openblas/include
                        /usr/include/x86_64-linux-gnu/openblas/include
                )
            endif()
        endif()

        message(STATUS "BLAS found, Includes: ${BLAS_INCLUDE_DIRS}")

        add_compile_options(${BLAS_LINKER_FLAGS})

        add_compile_definitions(GGML_USE_OPENBLAS)

        if (${BLAS_INCLUDE_DIRS} MATCHES "mkl" AND (${LLAMA_BLAS_VENDOR} MATCHES "Generic" OR ${LLAMA_BLAS_VENDOR} MATCHES "Intel"))
            add_compile_definitions(GGML_BLAS_USE_MKL)
        endif()

        set(LLAMA_EXTRA_LIBS     ${LLAMA_EXTRA_LIBS}     ${BLAS_LIBRARIES})
        set(LLAMA_EXTRA_INCLUDES ${LLAMA_EXTRA_INCLUDES} ${BLAS_INCLUDE_DIRS})
    else()
        message(WARNING "BLAS not found, please refer to "
        "https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors"
        " to set correct LLAMA_BLAS_VENDOR")
    endif()
endif()

if (LLAMA_QKK_64)
    add_compile_definitions(GGML_QKK_64)
endif()

if (LLAMA_CUBLAS)
    cmake_minimum_required(VERSION 3.17)

    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        message(STATUS "cuBLAS found")

        enable_language(CUDA)

        set(GGML_HEADERS_CUDA src/ggml-cuda.h)
        set(GGML_SOURCES_CUDA src/ggml-cuda.cu)

        add_compile_definitions(GGML_USE_CUBLAS)
        if (LLAMA_CUDA_FORCE_DMMV)
            add_compile_definitions(GGML_CUDA_FORCE_DMMV)
        endif()
        if (LLAMA_CUDA_FORCE_MMQ)
            add_compile_definitions(GGML_CUDA_FORCE_MMQ)
        endif()
        add_compile_definitions(GGML_CUDA_DMMV_X=${LLAMA_CUDA_DMMV_X})
        add_compile_definitions(GGML_CUDA_MMV_Y=${LLAMA_CUDA_MMV_Y})
        if (DEFINED LLAMA_CUDA_DMMV_Y)
            add_compile_definitions(GGML_CUDA_MMV_Y=${LLAMA_CUDA_DMMV_Y}) # for backwards compatibility
        endif()
        if (LLAMA_CUDA_F16 OR LLAMA_CUDA_DMMV_F16)
            add_compile_definitions(GGML_CUDA_F16)
        endif()
        add_compile_definitions(K_QUANTS_PER_ITERATION=${LLAMA_CUDA_KQUANTS_ITER})
        add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${LLAMA_CUDA_PEER_MAX_BATCH_SIZE})

        if (LLAMA_STATIC)
            if (WIN32)
                # As of 12.3.1 CUDA Tookit for Windows does not offer a static cublas library
                set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas CUDA::cublasLt)
            else ()
                set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
            endif()
        else()
            set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
        endif()

        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cuda_driver)

    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # 52 == lowest CUDA 12 standard
        # 60 == f16 CUDA intrinsics
        # 61 == integer CUDA intrinsics
        # 70 == compute capability at which unrolling a loop in mul_mat_q kernels is faster
        if (LLAMA_CUDA_F16 OR LLAMA_CUDA_DMMV_F16)
            set(CMAKE_CUDA_ARCHITECTURES "60;61;70") # needed for f16 CUDA intrinsics
        else()
            set(CMAKE_CUDA_ARCHITECTURES "52;61;70") # lowest CUDA 12 standard + lowest for integer intrinsics
            #set(CMAKE_CUDA_ARCHITECTURES "") # use this to compile much faster, but only F16 models work
        endif()
    endif()
    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    else()
        message(WARNING "cuBLAS not found")
    endif()
endif()

if (LLAMA_CLBLAST)
    find_package(CLBlast)
    if (CLBlast_FOUND)
        message(STATUS "CLBlast found")

        set(GGML_HEADERS_OPENCL ggml-opencl.h)
        set(GGML_SOURCES_OPENCL ggml-opencl.cpp)

        add_compile_definitions(GGML_USE_CLBLAST)

        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} clblast)
    else()
        message(WARNING "CLBlast not found")
    endif()
endif()

if (LLAMA_CPU_HBM)
    find_library(memkind memkind REQUIRED)

    add_compile_definitions(GGML_USE_CPU_HBM)

    target_link_libraries(ggml PUBLIC memkind)
endif()

if (LLAMA_PERF)
    add_compile_definitions(GGML_PERF)
endif()

function(get_flags CCID CCVER)
    set(C_FLAGS "")
    set(CXX_FLAGS "")

    if (CCID MATCHES "Clang")
        set(C_FLAGS   -Wunreachable-code-break -Wunreachable-code-return)
        set(CXX_FLAGS -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi)

        if (
            (CCID STREQUAL "Clang"      AND CCVER VERSION_GREATER_EQUAL 3.8.0) OR
            (CCID STREQUAL "AppleClang" AND CCVER VERSION_GREATER_EQUAL 7.3.0)
        )
            list(APPEND C_FLAGS -Wdouble-promotion)
        endif()
    elseif (CCID STREQUAL "GNU")
        set(C_FLAGS   -Wdouble-promotion)
        set(CXX_FLAGS -Wno-array-bounds)

        if (CCVER VERSION_GREATER_EQUAL 7.1.0)
            list(APPEND CXX_FLAGS -Wno-format-truncation)
        endif()
        if (CCVER VERSION_GREATER_EQUAL 8.1.0)
            list(APPEND CXX_FLAGS -Wextra-semi)
        endif()
    endif()

    set(GF_C_FLAGS   ${C_FLAGS}   PARENT_SCOPE)
    set(GF_CXX_FLAGS ${CXX_FLAGS} PARENT_SCOPE)
endfunction()

if (LLAMA_FATAL_WARNINGS)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND C_FLAGS   -Werror)
        list(APPEND CXX_FLAGS -Werror)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(/WX)
    endif()
endif()

if (LLAMA_ALL_WARNINGS)
    if (NOT MSVC)
        list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
        list(APPEND C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                  -Werror=implicit-int -Werror=implicit-function-declaration)
        list(APPEND CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

        list(APPEND C_FLAGS   ${WARNING_FLAGS})
        list(APPEND CXX_FLAGS ${WARNING_FLAGS})

        get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                            "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
    else()
        # todo : msvc
        set(C_FLAGS   "")
        set(CXX_FLAGS "")
    endif()
endif()

set(CUDA_CXX_FLAGS "")

if (LLAMA_CUBLAS)
    set(CUDA_FLAGS -use_fast_math)

    if (LLAMA_FATAL_WARNINGS)
        list(APPEND CUDA_FLAGS -Werror all-warnings)
    endif()

    if (LLAMA_ALL_WARNINGS AND NOT MSVC)
        set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
        if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
            list(APPEND NVCC_CMD -ccbin ${CMAKE_CUDA_HOST_COMPILER})
        endif()

        execute_process(
            COMMAND ${NVCC_CMD} -Xcompiler --version
            OUTPUT_VARIABLE CUDA_CCFULLVER
            ERROR_QUIET
        )

        if (NOT CUDA_CCFULLVER MATCHES clang)
            set(CUDA_CCID "GNU")
            execute_process(
                COMMAND ${NVCC_CMD} -Xcompiler "-dumpfullversion -dumpversion"
                OUTPUT_VARIABLE CUDA_CCVER
                ERROR_QUIET
            )
        else()
            if (CUDA_CCFULLVER MATCHES Apple)
                set(CUDA_CCID "AppleClang")
            else()
                set(CUDA_CCID "Clang")
            endif()
            string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
        endif()

        message("-- CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")

        get_flags(${CUDA_CCID} ${CUDA_CCVER})
        list(APPEND CUDA_CXX_FLAGS ${CXX_FLAGS} ${GF_CXX_FLAGS})  # This is passed to -Xcompiler later
    endif()

    if (NOT MSVC)
        list(APPEND CUDA_CXX_FLAGS -Wno-pedantic)
    endif()
endif()

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

if (LLAMA_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

if (LLAMA_CCACHE)
    find_program(LLAMA_CCACHE_FOUND ccache)
    if (LLAMA_CCACHE_FOUND)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
        set(ENV{CCACHE_SLOPPINESS} time_macros)
        message(STATUS "ccache found, compilation results will be cached. Disable with LLAMA_CCACHE=OFF.")
    else()
        message(STATUS "Warning: ccache not found - consider installing it for faster compilation or disable this warning with LLAMA_CCACHE=OFF")
    endif ()
endif()

# this version of Apple ld64 is buggy
execute_process(
    COMMAND ${CMAKE_C_COMPILER} ${CMAKE_EXE_LINKER_FLAGS} -Wl,-v
    ERROR_VARIABLE output
    OUTPUT_QUIET
)

if (output MATCHES "dyld-1015\.7")
    add_compile_definitions(HAVE_BUGGY_APPLE_LINKER)
endif()

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (MSVC)
    string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" CMAKE_GENERATOR_PLATFORM_LWR)
    message(STATUS "CMAKE_GENERATOR_PLATFORM: ${CMAKE_GENERATOR_PLATFORM}")
else ()
    set(CMAKE_GENERATOR_PLATFORM_LWR "")
endif ()

if (NOT MSVC)
    if (LLAMA_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (LLAMA_GPROF)
        add_compile_options(-pg)
    endif()
endif()

set(ARCH_FLAGS "")

if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR CMAKE_GENERATOR_PLATFORM_LWR STREQUAL "arm64" OR
    (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
     CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm.*|ARM64)$"))
    message(STATUS "ARM detected")
    if (MSVC)
        add_compile_definitions(__aarch64__) # MSVC defines _M_ARM64 instead
        add_compile_definitions(__ARM_NEON)
        add_compile_definitions(__ARM_FEATURE_FMA)

        set(CMAKE_REQUIRED_FLAGS_PREV ${CMAKE_REQUIRED_FLAGS})
        string(JOIN " " CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS} "/arch:armv8.2")
        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int8x16_t _a, _b; int32x4_t _s = vdotq_s32(_s, _a, _b); return 0; }" GGML_COMPILER_SUPPORT_DOTPROD)
        if (GGML_COMPILER_SUPPORT_DOTPROD)
            add_compile_definitions(__ARM_FEATURE_DOTPROD)
        endif ()
        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float16_t _a; float16x8_t _s = vdupq_n_f16(_a); return 0; }" GGML_COMPILER_SUPPORT_FP16_VECTOR_ARITHMETIC)
        if (GGML_COMPILER_SUPPORT_FP16_VECTOR_ARITHMETIC)
            add_compile_definitions(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        endif ()
        set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_PREV})
    else()
        check_cxx_compiler_flag(-mfp16-format=ieee COMPILER_SUPPORTS_FP16_FORMAT_I3E)
        if (NOT "${COMPILER_SUPPORTS_FP16_FORMAT_I3E}" STREQUAL "")
            list(APPEND ARCH_FLAGS -mfp16-format=ieee)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
            # Raspberry Pi 1, Zero
            list(APPEND ARCH_FLAGS -mfpu=neon-fp-armv8 -mno-unaligned-access)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
            if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
                # Android armeabi-v7a
                list(APPEND ARCH_FLAGS -mfpu=neon-vfpv4 -mno-unaligned-access -funsafe-math-optimizations)
            else()
                # Raspberry Pi 2
                list(APPEND ARCH_FLAGS -mfpu=neon-fp-armv8 -mno-unaligned-access -funsafe-math-optimizations)
            endif()
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
            # Android arm64-v8a
            # Raspberry Pi 3, 4, Zero 2 (32-bit)
            list(APPEND ARCH_FLAGS -mno-unaligned-access)
        endif()
    endif()
elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR CMAKE_GENERATOR_PLATFORM_LWR MATCHES "^(x86_64|i686|amd64|x64|win32)$" OR
        (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
         CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|i686|AMD64)$"))
    message(STATUS "x86 detected")
    if (MSVC)
        # instruction set detection for MSVC only
        if (LLAMA_NATIVE)
            include(cmake/FindSIMD.cmake)
        endif ()
        if (LLAMA_AVX512)
            list(APPEND ARCH_FLAGS /arch:AVX512)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
            if (LLAMA_AVX512_VBMI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VBMI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VBMI__>)
            endif()
            if (LLAMA_AVX512_VNNI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VNNI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VNNI__>)
            endif()
        elseif (LLAMA_AVX2)
            list(APPEND ARCH_FLAGS /arch:AVX2)
        elseif (LLAMA_AVX)
            list(APPEND ARCH_FLAGS /arch:AVX)
        endif()
    else()
        if (LLAMA_NATIVE)
            list(APPEND ARCH_FLAGS -march=native)
        endif()
        if (LLAMA_F16C)
            list(APPEND ARCH_FLAGS -mf16c)
        endif()
        if (LLAMA_FMA)
            list(APPEND ARCH_FLAGS -mfma)
        endif()
        if (LLAMA_AVX)
            list(APPEND ARCH_FLAGS -mavx)
        endif()
        if (LLAMA_AVX2)
            list(APPEND ARCH_FLAGS -mavx2)
        endif()
        if (LLAMA_AVX512)
            list(APPEND ARCH_FLAGS -mavx512f)
            list(APPEND ARCH_FLAGS -mavx512bw)
        endif()
        if (LLAMA_AVX512_VBMI)
            list(APPEND ARCH_FLAGS -mavx512vbmi)
        endif()
        if (LLAMA_AVX512_VNNI)
            list(APPEND ARCH_FLAGS -mavx512vnni)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
    message(STATUS "PowerPC detected")
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64le")
        list(APPEND ARCH_FLAGS -mcpu=powerpc64le)
    else()
        list(APPEND ARCH_FLAGS -mcpu=native -mtune=native)
        #TODO: Add  targets for Power8/Power9 (Altivec/VSX) and Power10(MMA) and query for big endian systems (ppc64/le/be)
    endif()
else()
    message(STATUS "Unknown architecture")
endif()

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${ARCH_FLAGS}>")
add_compile_options("$<$<COMPILE_LANGUAGE:C>:${ARCH_FLAGS}>")

if (LLAMA_CUBLAS)
    list(APPEND CUDA_CXX_FLAGS ${ARCH_FLAGS})
    list(JOIN CUDA_CXX_FLAGS " " CUDA_CXX_FLAGS_JOINED)  # pass host compiler flags as a single argument
    if (NOT CUDA_CXX_FLAGS_JOINED STREQUAL "")
        list(APPEND CUDA_FLAGS -Xcompiler ${CUDA_CXX_FLAGS_JOINED})
    endif()
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
endif()

if (MINGW)
    # Target Windows 8 for PrefetchVirtualMemory
    add_compile_definitions(_WIN32_WINNT=${LLAMA_WIN_VER})
endif()

#
# POSIX conformance
#

# clock_gettime came in POSIX.1b (1993)
# CLOCK_MONOTONIC came in POSIX.1-2001 / SUSv3 as optional
# posix_memalign came in POSIX.1-2001 / SUSv3
# M_PI is an XSI extension since POSIX.1-2001 / SUSv3, came in XPG1 (1985)
add_compile_definitions(_XOPEN_SOURCE=600)

# Somehow in OpenBSD whenever POSIX conformance is specified
# some string functions rely on locale_t availability,
# which was introduced in POSIX.1-2008, forcing us to go higher
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    remove_definitions(-D_XOPEN_SOURCE=600)
    add_compile_definitions(_XOPEN_SOURCE=700)
endif()

# Data types, macros and functions related to controlling CPU affinity and
# some memory allocation are available on Linux through GNU extensions in libc
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions(_GNU_SOURCE)
endif()

# RLIMIT_MEMLOCK came in BSD, is not specified in POSIX.1,
# and on macOS its availability depends on enabling Darwin extensions
# similarly on DragonFly, enabling BSD extensions is necessary
if (
    CMAKE_SYSTEM_NAME MATCHES "Darwin" OR
    CMAKE_SYSTEM_NAME MATCHES "iOS" OR
    CMAKE_SYSTEM_NAME MATCHES "tvOS" OR
    CMAKE_SYSTEM_NAME MATCHES "DragonFly"
)
    add_compile_definitions(_DARWIN_C_SOURCE)
endif()

# alloca is a non-standard interface that is not visible on BSDs when
# POSIX conformance is specified, but not all of them provide a clean way
# to enable it in such cases
if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    add_compile_definitions(__BSD_VISIBLE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
    add_compile_definitions(_NETBSD_SOURCE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_BSD_SOURCE)
endif()

#
# libraries
#

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ggml

add_library(ggml OBJECT
            src/ggml.c
            src/ggml.h
            src/ggml-alloc.c
            src/ggml-alloc.h
            src/ggml-backend.c
            src/ggml-backend.h
            src/ggml-quants.c
            src/ggml-quants.h
            ${GGML_SOURCES_CUDA}    ${GGML_HEADERS_CUDA}
            ${GGML_SOURCES_OPENCL}  ${GGML_HEADERS_OPENCL}
            ${GGML_SOURCES_EXTRA}   ${GGML_HEADERS_EXTRA}
            )

target_include_directories(ggml PUBLIC src/ ${LLAMA_EXTRA_INCLUDES})
target_compile_features   (ggml PUBLIC c_std_11) # don't bump

target_link_libraries(ggml PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})

add_library(ggml_static STATIC $<TARGET_OBJECTS:ggml>)
target_link_libraries(ggml_static PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})

if (BUILD_SHARED_LIBS)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(ggml_shared SHARED $<TARGET_OBJECTS:ggml>)
    target_link_libraries(ggml_shared PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})
    install(TARGETS ggml_shared LIBRARY)
endif()

# llama

add_library(llama OBJECT
            src/llama.cpp
            src/llama.h
            )

target_include_directories(llama PUBLIC src/)
target_compile_features   (llama PUBLIC cxx_std_11) # don't bump

target_link_libraries(llama PRIVATE ggml_static ${LLAMA_EXTRA_LIBS})

add_library(llama_static STATIC $<TARGET_OBJECTS:llama>)
target_link_libraries(llama_static PRIVATE ggml_static ${LLAMA_EXTRA_LIBS})

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(llama_shared SHARED $<TARGET_OBJECTS:llama>)
    target_link_libraries(llama_shared PRIVATE ggml_static ${LLAMA_EXTRA_LIBS})
    install(TARGETS llama_shared LIBRARY)
endif()

# common

add_library(common STATIC
    src/common/base64.hpp
    src/common/common.h
    src/common/common.cpp
    src/common/sampling.h
    src/common/sampling.cpp
    src/common/console.h
    src/common/console.cpp
    src/common/grammar-parser.h
    src/common/grammar-parser.cpp
    src/common/train.h
    src/common/train.cpp
    )

if (BUILD_SHARED_LIBS)
    set_target_properties(common PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(common PUBLIC src/common/)
target_compile_features(common PUBLIC cxx_std_11)
target_link_libraries(common PUBLIC llama)

#
# programs, examples and tests
#

if (LLAMA_BUILD_EXAMPLES)
    add_subdirectory(examples/server)
    add_subdirectory(examples/binding)
endif()
