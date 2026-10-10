cmake_minimum_required(VERSION 3.18)

# Provider selection lives here rather than in ggml/CMakeLists.txt so the
# upstream footprint is the single GGML_FLAGOS option. Every provider is
# opt-in: a plain -DGGML_FLAGOS=ON build needs no vendor SDK.
option(GGML_FLAGOS_DENGLIN    "ggml: build the FlagOS Denglin provider"                OFF)
option(GGML_FLAGOS_AMD        "ggml: build the FlagOS AMD HIP provider"                OFF)
option(GGML_FLAGOS_SPACEMIT   "ggml: build the experimental FlagOS SpaceMiT provider"  OFF)
option(GGML_FLAGOS_TRITON_JIT "ggml: build the optional FlagOS Triton JIT runtime"     OFF)

option(GGML_FLAGOS_PROVIDER_MODULES "ggml: build each FlagOS provider as a separately loaded plugin" OFF)

# Common code that both the core and DEVICE-kind provider plugins compile in.
set(FLAGOS_COMMON_SOURCES
    flagos-graph-plan.cpp
    flagos-graph-plan.h
    flagos-provider.cpp
    flagos-provider.h
    flagos-target.cpp
    flagos-target.h
    flagos-plugin.h
    flagos-runtime.h)

# Core-only code: plugin loading, the registry, and the core-owned runtime
# device (L0) with its kernel-package reader and op adapter (L2).
set(FLAGOS_CORE_SOURCES
    flagos-loader.cpp
    flagos-loader.h
    flagos-ops.cpp
    flagos-ops.h
    flagos-package.cpp
    flagos-package.h
    flagos-registry.h
    flagos-registry.cpp
    flagos-rt-device.cpp
    flagos-rt-device.h
    flagos-sha256.h)

set(FLAGOS_SOURCES
    ${FLAGOS_COMMON_SOURCES}
    ${FLAGOS_CORE_SOURCES})

set(FLAGOS_PRIVATE_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/vendor")
set(FLAGOS_PRIVATE_LIBRARIES)
set(FLAGOS_PRIVATE_DEFINITIONS)

set(FLAGOS_TRITON_JIT_SOURCE_DIR "" CACHE PATH "Path to a libtriton_jit source tree")
if (GGML_FLAGOS_TRITON_JIT)
    if (NOT GGML_FLAGOS_AMD)
        message(FATAL_ERROR "The initial FlagOS Triton JIT integration requires GGML_FLAGOS_AMD=ON")
    endif()
    if (NOT EXISTS "${FLAGOS_TRITON_JIT_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "Set FLAGOS_TRITON_JIT_SOURCE_DIR to a libtriton_jit source tree")
    endif()
    set(TRITON_JIT_BACKEND AMD CACHE STRING "libtriton_jit backend" FORCE)
    set(TRITON_JIT_WITH_TORCH OFF CACHE BOOL "Build libtriton_jit without its Torch C++ adapter" FORCE)
    set(TRITON_JIT_BUILD_TESTS OFF CACHE BOOL "Do not build libtriton_jit tests as part of llama.cpp" FORCE)
    set(TRITON_JIT_BUILD_OPERATORS OFF CACHE BOOL "Do not build libtriton_jit operators as part of llama.cpp" FORCE)
    set(TRITON_JIT_INSTALL OFF CACHE BOOL "Do not install libtriton_jit as a separate package" FORCE)
    add_subdirectory("${FLAGOS_TRITON_JIT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/libtriton_jit" EXCLUDE_FROM_ALL)
endif()

# Provider plugins are installed next to the core in a flagos-providers
# directory, which is where the core looks for them at runtime.
if (GGML_BACKEND_DL AND GGML_BACKEND_DIR)
    set(FLAGOS_PROVIDER_INSTALL_DIR "${GGML_BACKEND_DIR}/flagos-providers")
elseif (GGML_BACKEND_DL OR NOT BUILD_SHARED_LIBS)
    set(FLAGOS_PROVIDER_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}/flagos-providers")
else()
    set(FLAGOS_PROVIDER_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/flagos-providers")
endif()
if (BUILD_SHARED_LIBS AND NOT GGML_BACKEND_DL)
    set(FLAGOS_PROVIDER_BUILD_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
else()
    set(FLAGOS_PROVIDER_BUILD_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()
if (NOT FLAGOS_PROVIDER_BUILD_DIR)
    set(FLAGOS_PROVIDER_BUILD_DIR "${CMAKE_BINARY_DIR}/bin")
endif()
set(FLAGOS_PROVIDER_BUILD_DIR "${FLAGOS_PROVIDER_BUILD_DIR}/flagos-providers")

# Include one provider. Without GGML_FLAGOS_PROVIDER_MODULES its sources are
# compiled into the core as before. With it, the provider becomes its own
# MODULE library libflagos-provider-<name>.so that links only its SDK, and
# the core loads it with dlopen(). This is a macro so that variables a
# provider.cmake defines for its tests stay visible to this file.
macro(flagos_add_provider FLAGOS_PROVIDER_NAME FLAGOS_PROVIDER_FILE)
    if (GGML_FLAGOS_PROVIDER_MODULES)
        set(_flagos_core_sources ${FLAGOS_SOURCES})
        set(_flagos_core_libraries ${FLAGOS_PRIVATE_LIBRARIES})
        set(_flagos_core_definitions ${FLAGOS_PRIVATE_DEFINITIONS})
        set(FLAGOS_SOURCES)
        set(FLAGOS_PRIVATE_LIBRARIES)
        set(FLAGOS_PRIVATE_DEFINITIONS)
        include(${FLAGOS_PROVIDER_FILE})
        set(_flagos_target flagos-provider-${FLAGOS_PROVIDER_NAME})
        add_library(${_flagos_target} MODULE ${FLAGOS_SOURCES} ${FLAGOS_COMMON_SOURCES})
        set_target_properties(${_flagos_target} PROPERTIES
            PREFIX "lib"
            LIBRARY_OUTPUT_DIRECTORY "${FLAGOS_PROVIDER_BUILD_DIR}"
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON)
        target_include_directories(${_flagos_target} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/.." ${FLAGOS_PRIVATE_INCLUDE_DIRS})
        target_link_libraries(${_flagos_target} PRIVATE ggml-base ${FLAGOS_PRIVATE_LIBRARIES})
        target_compile_definitions(${_flagos_target} PRIVATE ${FLAGOS_PRIVATE_DEFINITIONS} GGML_FLAGOS_PROVIDER_MODULE)
        install(TARGETS ${_flagos_target} LIBRARY DESTINATION "${FLAGOS_PROVIDER_INSTALL_DIR}")
        list(APPEND FLAGOS_PROVIDER_MODULE_TARGETS ${_flagos_target})
        set(FLAGOS_SOURCES ${_flagos_core_sources})
        set(FLAGOS_PRIVATE_LIBRARIES ${_flagos_core_libraries})
        set(FLAGOS_PRIVATE_DEFINITIONS ${_flagos_core_definitions})
    else()
        include(${FLAGOS_PROVIDER_FILE})
    endif()
endmacro()

set(FLAGOS_PROVIDER_MODULE_TARGETS)

if (GGML_FLAGOS_DENGLIN)
    flagos_add_provider(denglin providers/denglin/provider.cmake)
endif()

if (GGML_FLAGOS_AMD)
    flagos_add_provider(amd providers/amd/provider.cmake)
endif()

if (GGML_FLAGOS_SPACEMIT)
    flagos_add_provider(spacemit providers/spacemit/provider.cmake)
endif()

if (NOT GGML_FLAGOS_DENGLIN AND NOT GGML_FLAGOS_AMD AND NOT GGML_FLAGOS_SPACEMIT)
    message(STATUS "FlagOS: building provider-neutral core with no device provider")
endif()

ggml_add_backend_library(ggml-flagos ${FLAGOS_SOURCES})
target_link_libraries(ggml-flagos PRIVATE ${CMAKE_DL_LIBS})
foreach(_flagos_module IN LISTS FLAGOS_PROVIDER_MODULE_TARGETS)
    add_dependencies(ggml-flagos ${_flagos_module})
endforeach()
target_include_directories(ggml-flagos PRIVATE ${FLAGOS_PRIVATE_INCLUDE_DIRS})
target_link_libraries(ggml-flagos PRIVATE ${FLAGOS_PRIVATE_LIBRARIES})
target_compile_definitions(ggml-flagos PRIVATE ${FLAGOS_PRIVATE_DEFINITIONS})
if (GGML_FLAGOS_SPACEMIT)
    target_link_options(ggml-flagos INTERFACE "$<BUILD_INTERFACE:LINKER:-rpath-link,${SPERT_DIR}/lib>")
endif()


if (GGML_FLAGOS_DENGLIN)
    add_executable(flagos-check-kernels EXCLUDE_FROM_ALL kernels/check_merged_module.cpp)
    target_include_directories(flagos-check-kernels PRIVATE "${FLAGOS_DENGLIN_CUDA_INCLUDE_DIR}")
    target_link_libraries(flagos-check-kernels PRIVATE "${FLAGOS_DENGLIN_CURT_LIBRARY}")
endif()

add_library(flagos-check-public-header OBJECT EXCLUDE_FROM_ALL tests/check_public_header.c)
target_include_directories(flagos-check-public-header PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(flagos-check-public-header PRIVATE ggml-base)

add_executable(flagos-check-graph-plan EXCLUDE_FROM_ALL
    flagos-graph-plan.cpp
    tests/check_graph_plan.cpp)
target_include_directories(flagos-check-graph-plan PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(flagos-check-graph-plan PRIVATE ggml-base)

add_executable(flagos-check-provider EXCLUDE_FROM_ALL
    flagos-provider.cpp
    flagos-target.cpp
    tests/check_provider.cpp)
target_include_directories(flagos-check-provider PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(flagos-check-provider PRIVATE ggml-base)

add_executable(flagos-check-target EXCLUDE_FROM_ALL
    flagos-target.cpp
    tests/check_target.cpp)
target_include_directories(flagos-check-target PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/ggml/include")

add_executable(flagos-check-registry EXCLUDE_FROM_ALL
    flagos-provider.cpp
    flagos-target.cpp
    flagos-graph-plan.cpp
    ${FLAGOS_CORE_SOURCES}
    tests/check_registry.cpp)
target_include_directories(flagos-check-registry PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_SOURCE_DIR}/vendor")
target_compile_definitions(flagos-check-registry PRIVATE GGML_FLAGOS_HAVE_DENGLIN GGML_FLAGOS_HAVE_AMD)
target_link_libraries(flagos-check-registry PRIVATE ggml)

# Provider plugin loader: three mock plugins (C runtime, C++ device, bad ABI)
# plus a file the loader must ignore.
set(FLAGOS_TEST_PLUGIN_DIR "${CMAKE_BINARY_DIR}/flagos-test-plugins")
add_library(flagos-test-plugin-runtime MODULE EXCLUDE_FROM_ALL tests/plugins/mock_runtime_provider.c)
add_library(flagos-test-plugin-bad MODULE EXCLUDE_FROM_ALL tests/plugins/mock_runtime_provider.c)
target_compile_definitions(flagos-test-plugin-bad PRIVATE FLAGOS_MOCK_BAD_ABI)
add_library(flagos-test-plugin-device MODULE EXCLUDE_FROM_ALL
    tests/plugins/mock_device_provider.cpp flagos-provider.cpp flagos-target.cpp)
target_compile_definitions(flagos-test-plugin-device PRIVATE GGML_FLAGOS_PROVIDER_MODULE)
target_link_libraries(flagos-test-plugin-device PRIVATE ggml-base)
foreach(_flagos_plugin runtime bad device)
    set(_flagos_plugin_name ${_flagos_plugin})
    if (_flagos_plugin STREQUAL "runtime")
        set(_flagos_plugin_name mock-runtime)
    elseif (_flagos_plugin STREQUAL "bad")
        set(_flagos_plugin_name mock-bad)
    else()
        set(_flagos_plugin_name mock-device)
    endif()
    target_include_directories(flagos-test-plugin-${_flagos_plugin} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/..")
    set_target_properties(flagos-test-plugin-${_flagos_plugin} PROPERTIES
        PREFIX "lib" OUTPUT_NAME "flagos-provider-${_flagos_plugin_name}"
        LIBRARY_OUTPUT_DIRECTORY "${FLAGOS_TEST_PLUGIN_DIR}"
        C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
endforeach()
file(MAKE_DIRECTORY "${FLAGOS_TEST_PLUGIN_DIR}")
file(WRITE "${FLAGOS_TEST_PLUGIN_DIR}/not-a-provider.so" "")

add_executable(flagos-check-loader EXCLUDE_FROM_ALL
    flagos-provider.cpp
    flagos-target.cpp
    flagos-graph-plan.cpp
    ${FLAGOS_CORE_SOURCES}
    tests/check_loader.cpp)
target_include_directories(flagos-check-loader PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_SOURCE_DIR}/vendor")
target_compile_definitions(flagos-check-loader PRIVATE FLAGOS_TEST_PLUGIN_DIR="${FLAGOS_TEST_PLUGIN_DIR}")
target_link_libraries(flagos-check-loader PRIVATE ggml ${CMAKE_DL_LIBS})
add_dependencies(flagos-check-loader flagos-test-plugin-runtime flagos-test-plugin-bad flagos-test-plugin-device)

# hostsim: reference RUNTIME provider plus a generated kernel package, used to
# test the core runtime device end to end (also with test-backend-ops):
#   GGML_FLAGOS_PROVIDER_PATH=<build>/flagos-hostsim test-backend-ops -b FlagOS-hostsim0
set(FLAGOS_HOSTSIM_DIR "${CMAKE_BINARY_DIR}/flagos-hostsim")
add_library(flagos-provider-hostsim MODULE EXCLUDE_FROM_ALL tests/plugins/hostsim_provider.cpp)
target_include_directories(flagos-provider-hostsim PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/plugins")
set_target_properties(flagos-provider-hostsim PROPERTIES
    PREFIX "lib" OUTPUT_NAME "flagos-provider-hostsim"
    LIBRARY_OUTPUT_DIRECTORY "${FLAGOS_HOSTSIM_DIR}"
    CXX_VISIBILITY_PRESET hidden)
set(_flagos_hostsim_kernels
    "flagos_add_f32 64 4" "flagos_add_repeat_f32 64 5" "flagos_mul_f32 64 5" "flagos_scale_f32 64 5"
    "flagos_rms_norm_f32 4096 4" "flagos_silu_f32 64 3" "flagos_sigmoid_f32 64 3" "flagos_tanh_f32 64 3")
list(APPEND _flagos_hostsim_kernels
    "flagos_copy_strided_f32 64 19" "flagos_set_rows_f32_f16 64 6" "flagos_get_rows_f32 64 4")
foreach(_q q4_0 q4_1 q8_0 q4_k q5_k q6_k)
    list(APPEND _flagos_hostsim_kernels "flagos_mul_mat_${_q}_f32 1 6" "flagos_mul_mat_${_q}_f32_batched 4 7")
endforeach()
set(_flagos_hostsim_module "hostsim-module v1\n")
foreach(_k IN LISTS _flagos_hostsim_kernels)
    string(REPLACE " " ";" _k "${_k}")
    list(GET _k 0 _name)
    list(GET _k 1 _block)
    string(APPEND _flagos_hostsim_module "${_name} ${_block}\n")
endforeach()
file(WRITE "${FLAGOS_HOSTSIM_DIR}/hostsim-kernels/hostsim-elementwise.bin" "${_flagos_hostsim_module}")
file(SHA256 "${FLAGOS_HOSTSIM_DIR}/hostsim-kernels/hostsim-elementwise.bin" _flagos_hostsim_sha)
file(SIZE "${FLAGOS_HOSTSIM_DIR}/hostsim-kernels/hostsim-elementwise.bin" _flagos_hostsim_size)
set(_flagos_hostsim_entries "")
foreach(_k IN LISTS _flagos_hostsim_kernels)
    string(REPLACE " " ";" _k "${_k}")
    list(GET _k 0 _name)
    list(GET _k 1 _block)
    list(GET _k 2 _argc)
    if (_flagos_hostsim_entries)
        string(APPEND _flagos_hostsim_entries ",\n")
    endif()
    string(APPEND _flagos_hostsim_entries "  {\"name\": \"${_name}\", \"file\": \"hostsim-elementwise.bin\", \"sha256\": \"${_flagos_hostsim_sha}\", \"size_bytes\": ${_flagos_hostsim_size}, \"block_size\": ${_block}, \"num_warps\": 1, \"warp_size\": 1, \"argument_count\": ${_argc}}")
endforeach()
file(WRITE "${FLAGOS_HOSTSIM_DIR}/hostsim-kernels/manifest.json"
    "{\"format\": \"flagos-kernel-package-v1\", \"arch\": \"hostsim-v1\", \"kernels\": [\n${_flagos_hostsim_entries}\n]}\n")

add_executable(flagos-check-spacemit-adapter EXCLUDE_FROM_ALL
    flagos-provider.cpp
    flagos-target.cpp
    flagos-graph-plan.cpp
    ${FLAGOS_CORE_SOURCES}
    providers/spacemit/flagos-spacemit.cpp
    tests/check_spacemit_adapter.cpp)
target_include_directories(flagos-check-spacemit-adapter PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_SOURCE_DIR}/vendor")
target_compile_definitions(flagos-check-spacemit-adapter PRIVATE GGML_FLAGOS_HAVE_SPACEMIT)
target_link_libraries(flagos-check-spacemit-adapter PRIVATE ggml)

if (GGML_FLAGOS_SPACEMIT)
    add_executable(flagos-check-spacemit-buffer EXCLUDE_FROM_ALL tests/check_spacemit_buffer.cpp)
    target_include_directories(flagos-check-spacemit-buffer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/..")
    target_link_libraries(flagos-check-spacemit-buffer PRIVATE ggml)
endif()

if (GGML_FLAGOS_AMD)
    add_executable(flagos-check-amd EXCLUDE_FROM_ALL
        flagos-provider.cpp
        flagos-target.cpp
        ${FLAGOS_CORE_SOURCES}
        flagos-graph-plan.cpp
        providers/amd/flagos-amd.cpp
        providers/amd/flagos-amd-hsaco.cpp
        providers/amd/flagos-amd-aot.cpp
        tests/check_amd_qwen35_ffn.cpp
        tests/check_amd.cpp)
    target_include_directories(flagos-check-amd PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
        "${CMAKE_SOURCE_DIR}/vendor")
    target_compile_definitions(flagos-check-amd PRIVATE
        GGML_FLAGOS_HAVE_AMD
        ${FLAGOS_AMD_PRIVATE_DEFINITIONS})
    target_link_libraries(flagos-check-amd PRIVATE ggml hip::host ${CMAKE_DL_LIBS})

    # AMD as a RUNTIME-kind provider under the core runtime device; HIP mocked.
    add_executable(flagos-check-amd-runtime EXCLUDE_FROM_ALL
        providers/amd/flagos-amd-runtime.cpp
        providers/amd/tests/check_amd_runtime.cpp
        flagos-graph-plan.cpp
        flagos-ops.cpp
        flagos-package.cpp
        flagos-rt-device.cpp)
    target_include_directories(flagos-check-amd-runtime PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_SOURCE_DIR}/vendor"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/plugins")
    target_link_libraries(flagos-check-amd-runtime PRIVATE ggml-base ggml-cpu hip::host)

    # Opt-in plugin for the migration period. It is built into a separate
    # directory so it is not auto-loaded next to the DEVICE-kind AMD
    # provider (both would expose the same GPU); point
    # GGML_FLAGOS_PROVIDER_PATH at it to use it instead.
    if (GGML_FLAGOS_PROVIDER_MODULES)
        add_library(flagos-provider-amd-rt MODULE EXCLUDE_FROM_ALL providers/amd/flagos-amd-runtime.cpp)
        set_target_properties(flagos-provider-amd-rt PROPERTIES
            PREFIX "lib" OUTPUT_NAME "flagos-provider-amd-rt"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/flagos-providers-optional"
            CXX_VISIBILITY_PRESET hidden)
        target_link_libraries(flagos-provider-amd-rt PRIVATE hip::host)
        target_compile_definitions(flagos-provider-amd-rt PRIVATE
            GGML_FLAGOS_PROVIDER_MODULE ${FLAGOS_AMD_PRIVATE_DEFINITIONS})
    endif()

    add_executable(flagos-check-amd-gdn-contract EXCLUDE_FROM_ALL
        flagos-graph-plan.cpp tests/check_amd_gdn_contract.cpp)
    target_include_directories(flagos-check-amd-gdn-contract PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
    target_link_libraries(flagos-check-amd-gdn-contract PRIVATE ggml-base)

    add_executable(flagos-check-amd-dense-gather EXCLUDE_FROM_ALL
        tests/check_amd_dense_gather.cpp)
    target_link_libraries(flagos-check-amd-dense-gather PRIVATE ggml)

    add_executable(flagos-check-amd-precision EXCLUDE_FROM_ALL
        tests/check_amd_matmul_precision.cpp)
    target_link_libraries(flagos-check-amd-precision PRIVATE ggml)

    add_executable(flagos-check-amd-aot EXCLUDE_FROM_ALL
        providers/amd/flagos-amd-hsaco.cpp
        providers/amd/flagos-amd-aot.cpp
        tests/check_amd_aot.cpp)
    target_include_directories(flagos-check-amd-aot PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
        "${CMAKE_SOURCE_DIR}/vendor")
    target_compile_definitions(flagos-check-amd-aot PRIVATE
        ${FLAGOS_AMD_PRIVATE_DEFINITIONS})
    target_link_libraries(flagos-check-amd-aot PRIVATE ggml-base hip::host)

    add_executable(flagos-check-amd-qwen35-ffn EXCLUDE_FROM_ALL
        flagos-provider.cpp
        flagos-target.cpp
        flagos-loader.cpp
        flagos-registry.cpp
        flagos-graph-plan.cpp
        providers/amd/flagos-amd.cpp
        providers/amd/flagos-amd-hsaco.cpp
        providers/amd/flagos-amd-aot.cpp
        tests/check_amd_qwen35_ffn.cpp)
    target_include_directories(flagos-check-amd-qwen35-ffn PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
        "${CMAKE_SOURCE_DIR}/vendor")
    target_compile_definitions(flagos-check-amd-qwen35-ffn PRIVATE
        GGML_FLAGOS_HAVE_AMD
        FLAGOS_AMD_QWEN35_FFN_STANDALONE
        ${FLAGOS_AMD_PRIVATE_DEFINITIONS})
    target_link_libraries(flagos-check-amd-qwen35-ffn PRIVATE ggml hip::host ${CMAKE_DL_LIBS})

    if (GGML_FLAGOS_TRITON_JIT)
        add_executable(flagos-check-amd-jit EXCLUDE_FROM_ALL tests/check_amd_jit.cpp)
        target_compile_definitions(flagos-check-amd-jit PRIVATE
            FLAGOS_TRITON_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels/triton")
        target_link_libraries(flagos-check-amd-jit PRIVATE TritonJIT::triton_jit hip::host)
    endif()
endif()

# ggml-flagos.h is installed with the other ggml public headers. The plugin
# ABI header is installed for out-of-tree providers; the remaining FlagOS
# headers are internal.
install(FILES flagos-runtime.h DESTINATION include)
if (GGML_FLAGOS_DENGLIN)
    install(DIRECTORY "${FLAGOS_DENGLIN_AOT_KERNEL_DIR}/" DESTINATION share/ggml/flagos/denglin)
endif()
if (GGML_FLAGOS_AMD AND FLAGOS_AMD_AOT_KERNEL_DIR)
    install(DIRECTORY "${FLAGOS_AMD_AOT_KERNEL_DIR}/" DESTINATION "${CMAKE_INSTALL_DATADIR}/ggml/flagos/amd")
endif()
if (GGML_FLAGOS_TRITON_JIT)
    install(DIRECTORY kernels/triton/ DESTINATION "${CMAKE_INSTALL_DATADIR}/ggml/flagos/triton")
endif()

if (GGML_FLAGOS_AMD)
    add_executable(flagos-check-amd-aligned-gemm EXCLUDE_FROM_ALL
        tests/check_amd_aligned_gemm.cpp)
    target_link_libraries(flagos-check-amd-aligned-gemm PRIVATE ggml)
endif()

if (GGML_FLAGOS_AMD)
    add_executable(flagos-check-amd-tiled-ffn EXCLUDE_FROM_ALL tests/check_amd_tiled_ffn.cpp)
    target_link_libraries(flagos-check-amd-tiled-ffn PRIVATE ggml)
endif()

if (GGML_FLAGOS_AMD)
    add_executable(flagos-check-amd-packed-ffn EXCLUDE_FROM_ALL tests/check_amd_packed_ffn.cpp)
    target_link_libraries(flagos-check-amd-packed-ffn PRIVATE ggml)
endif()

if (GGML_FLAGOS_AMD AND GGML_CPU)
    add_executable(flagos-check-amd-strided-rms EXCLUDE_FROM_ALL tests/check_amd_strided_rms.cpp)
    target_link_libraries(flagos-check-amd-strided-rms PRIVATE ggml)
endif()

if (GGML_FLAGOS_AMD)
    add_executable(flagos-check-amd-gdn-value-tile EXCLUDE_FROM_ALL tests/check_amd_gdn_value_tile.cpp)
    target_link_libraries(flagos-check-amd-gdn-value-tile PRIVATE ggml ${CMAKE_DL_LIBS})
endif()
