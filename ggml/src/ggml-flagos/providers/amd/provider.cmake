if (NOT DEFINED ROCM_PATH)
    if (DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "$ENV{ROCM_PATH}")
    elseif (EXISTS "/opt/rocm")
        set(ROCM_PATH "/opt/rocm")
    else()
        set(ROCM_PATH "/usr")
    endif()
endif()

list(APPEND CMAKE_PREFIX_PATH
    "${ROCM_PATH}"
    "${ROCM_PATH}/lib64/cmake"
    "${ROCM_PATH}/lib/x86_64-linux-gnu/cmake")

find_package(hip CONFIG REQUIRED)

set(FLAGOS_AMD_AOT_KERNEL_DIR "" CACHE PATH "FlagOS AMD AOT package to use and install")
if (FLAGOS_AMD_AOT_KERNEL_DIR)
    get_filename_component(FLAGOS_AMD_AOT_KERNEL_DIR "${FLAGOS_AMD_AOT_KERNEL_DIR}" ABSOLUTE)
    if (NOT EXISTS "${FLAGOS_AMD_AOT_KERNEL_DIR}/manifest.json")
        file(GLOB FLAGOS_AMD_AOT_ARCH_MANIFESTS "${FLAGOS_AMD_AOT_KERNEL_DIR}/*/manifest.json")
        if (NOT FLAGOS_AMD_AOT_ARCH_MANIFESTS)
            message(FATAL_ERROR "FLAGOS_AMD_AOT_KERNEL_DIR must contain manifest.json directly or in target subdirectories: ${FLAGOS_AMD_AOT_KERNEL_DIR}")
        endif()
    endif()
    list(APPEND FLAGOS_AMD_PRIVATE_DEFINITIONS
        FLAGOS_AMD_CONFIGURED_KERNEL_DIR="${FLAGOS_AMD_AOT_KERNEL_DIR}")
endif()
list(APPEND FLAGOS_AMD_PRIVATE_DEFINITIONS
    FLAGOS_AMD_INSTALLED_KERNEL_DIR="${CMAKE_INSTALL_FULL_DATADIR}/ggml/flagos/amd")

if (GGML_BACKEND_DL AND GGML_BACKEND_DIR)
    set(FLAGOS_AMD_RUNTIME_INSTALL_DIR "${GGML_BACKEND_DIR}")
elseif (GGML_BACKEND_DL OR NOT BUILD_SHARED_LIBS)
    set(FLAGOS_AMD_RUNTIME_INSTALL_DIR "${CMAKE_INSTALL_BINDIR}")
else()
    set(FLAGOS_AMD_RUNTIME_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
endif()
get_filename_component(FLAGOS_AMD_RUNTIME_INSTALL_FULL_DIR
    "${FLAGOS_AMD_RUNTIME_INSTALL_DIR}" ABSOLUTE BASE_DIR "${CMAKE_INSTALL_PREFIX}")
get_filename_component(FLAGOS_AMD_DATA_INSTALL_FULL_DIR
    "${CMAKE_INSTALL_DATADIR}/ggml/flagos/amd" ABSOLUTE BASE_DIR "${CMAKE_INSTALL_PREFIX}")
file(RELATIVE_PATH FLAGOS_AMD_RELOCATABLE_KERNEL_DIR
    "${FLAGOS_AMD_RUNTIME_INSTALL_FULL_DIR}"
    "${FLAGOS_AMD_DATA_INSTALL_FULL_DIR}")
if (NOT IS_ABSOLUTE "${FLAGOS_AMD_RELOCATABLE_KERNEL_DIR}")
    list(APPEND FLAGOS_AMD_PRIVATE_DEFINITIONS
        FLAGOS_AMD_RELOCATABLE_KERNEL_DIR="${FLAGOS_AMD_RELOCATABLE_KERNEL_DIR}")
endif()

list(APPEND FLAGOS_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/flagos-amd-api.h"
    "${CMAKE_CURRENT_LIST_DIR}/flagos-amd.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/flagos-amd-hsaco.h"
    "${CMAKE_CURRENT_LIST_DIR}/flagos-amd-hsaco.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/flagos-amd-aot.h"
    "${CMAKE_CURRENT_LIST_DIR}/flagos-amd-attention.inc"
    "${CMAKE_CURRENT_LIST_DIR}/flagos-amd-aot.cpp")
list(APPEND FLAGOS_PRIVATE_LIBRARIES hip::host ${CMAKE_DL_LIBS})
list(APPEND FLAGOS_PRIVATE_DEFINITIONS ${FLAGOS_AMD_PRIVATE_DEFINITIONS} GGML_FLAGOS_HAVE_AMD)

if (UNIX)
    foreach(FLAGOS_AMD_CHECK_NAME IN ITEMS constant-assets policy-preflight repeat-operations tanh gemv f16-projection ffn-fusion-resources batched-gather regular-ffn packed-ffn-preflight gather-resources elementwise-resources silu-mul-fusion tensor-alias copy-resources norm-state-resources rope-resources normalization-fusion-resources rope-fusion-resources swiglu-resources flash-attn-resources)
        string(REPLACE "-" "_" FLAGOS_AMD_CHECK_SOURCE "${FLAGOS_AMD_CHECK_NAME}")
        add_executable(flagos-check-amd-${FLAGOS_AMD_CHECK_NAME} EXCLUDE_FROM_ALL
            "${CMAKE_CURRENT_LIST_DIR}/tests/check_${FLAGOS_AMD_CHECK_SOURCE}.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/flagos-amd-hsaco.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/flagos-amd-aot.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../../flagos-provider.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../../flagos-target.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../../flagos-graph-plan.cpp")
        target_include_directories(flagos-check-amd-${FLAGOS_AMD_CHECK_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/vendor")
        target_compile_definitions(flagos-check-amd-${FLAGOS_AMD_CHECK_NAME} PRIVATE GGML_FLAGOS_HAVE_AMD)
        target_link_libraries(flagos-check-amd-${FLAGOS_AMD_CHECK_NAME} PRIVATE ggml-base hip::host ${CMAKE_DL_LIBS})
    endforeach()
endif()

message(STATUS "FlagOS: enabling AMD HIP provider with ROCM_PATH=${ROCM_PATH}")
