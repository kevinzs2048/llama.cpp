if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|riscv")
    message(FATAL_ERROR "FlagOS SpaceMiT requires a RISC-V Linux target and the SpaceMiT toolchain")
endif()
if (NOT GGML_CPU OR GGML_BACKEND_DL)
    message(FATAL_ERROR "FlagOS SpaceMiT currently requires GGML_CPU=ON and GGML_BACKEND_DL=OFF (native CPU helper dependency)")
endif()
if (GGML_CPU_RISCV64_SPACEMIT OR GGML_SPACEMIT)
    message(FATAL_ERROR "Disable GGML_CPU_RISCV64_SPACEMIT and GGML_SPACEMIT when using GGML_FLAGOS_SPACEMIT; they own the same native symbols")
endif()

set(FLAGOS_SPACEMIT_SOURCE_DIR "" CACHE PATH "SpaceMiT llama.cpp source checkout (see providers/spacemit/README.md)")
set(SPERT_DIR "$ENV{SPERT_DIR}" CACHE PATH "Extracted spine-runtime SDK with include/ and lib/")
set(spacemit_source "${FLAGOS_SPACEMIT_SOURCE_DIR}/ggml/src/ggml-spacemit")
if (NOT EXISTS "${spacemit_source}/ggml-spacemit.cpp")
    message(FATAL_ERROR "Set FLAGOS_SPACEMIT_SOURCE_DIR to the pinned SpaceMiT llama.cpp checkout")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/source-lock.cmake")
if (NOT EXISTS "${SPERT_DIR}/include/spert.hpp")
    message(FATAL_ERROR "Set SPERT_DIR to an extracted spine-runtime SDK")
endif()
find_library(FLAGOS_SPACEMIT_SPERT_LIBRARY NAMES spert
    PATHS "${SPERT_DIR}/lib" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED)

# Copy only the vendor extension headers so their quoted ggml.h includes use this tree.
set(spacemit_headers "${CMAKE_CURRENT_BINARY_DIR}/spacemit-headers")
file(MAKE_DIRECTORY "${spacemit_headers}")
foreach(header ggml-spacemit.h ggml-profile.h)
    configure_file("${FLAGOS_SPACEMIT_SOURCE_DIR}/ggml/include/${header}" "${spacemit_headers}/${header}" COPYONLY)
endforeach()

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(spacemit_march "-march=rv64gcv_zfh_zvfh_zba_zbb_zbs_zicbop_zihintpause_zawrs_zicond_zvbb")
if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
        OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    string(APPEND spacemit_march "_xsmtvdotii")
endif()
set(CMAKE_REQUIRED_FLAGS "${spacemit_march}")
check_cxx_source_compiles("int main() { __asm__ volatile(\"vmadot v2, v0, v1\"); }" FLAGOS_SPACEMIT_HAS_IME1)
check_cxx_source_compiles("int main() { __asm__ volatile(\"vmadot v2, v0, v1, i4\"); __asm__ volatile(\"vpack.vv v2, v0, v1, 2\"); __asm__ volatile(\"vnspack.vv v2, v0, v1, 2\"); }" FLAGOS_SPACEMIT_HAS_IME2)
cmake_pop_check_state()
if (NOT FLAGOS_SPACEMIT_HAS_IME1 AND NOT FLAGOS_SPACEMIT_HAS_IME2)
    message(FATAL_ERROR "SpaceMiT compiler must support IME1 or IME2 and the required RISC-V extensions")
endif()

add_library(flagos-spacemit-native OBJECT
    "${CMAKE_CURRENT_LIST_DIR}/native-backend.cpp"
    "${spacemit_source}/ime.cpp"
    "${spacemit_source}/ime1_kernels.cpp"
    "${spacemit_source}/ime2_kernels.cpp"
    "${spacemit_source}/rvv_kernels.cpp"
    "${spacemit_source}/scalar_kernels.cpp"
    "${spacemit_source}/repack.cpp"
    "${spacemit_source}/spine_mem_pool.cpp")
set_target_properties(flagos-spacemit-native PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(flagos-spacemit-native PRIVATE ggml-base ggml-cpu)
target_include_directories(flagos-spacemit-native PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cpu"
    "${spacemit_headers}" "${spacemit_source}" "${SPERT_DIR}/include")
target_compile_options(flagos-spacemit-native PRIVATE "${spacemit_march}")
if (BUILD_SHARED_LIBS)
    target_compile_definitions(flagos-spacemit-native PRIVATE GGML_BACKEND_BUILD GGML_BACKEND_SHARED)
endif()
foreach(ime IME1 IME2)
    if (FLAGOS_SPACEMIT_HAS_${ime})
        target_compile_definitions(flagos-spacemit-native PRIVATE RISCV64_SPACEMIT_${ime})
    endif()
endforeach()

list(APPEND FLAGOS_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/flagos-spacemit.cpp"
    $<TARGET_OBJECTS:flagos-spacemit-native>)
list(APPEND FLAGOS_PRIVATE_LIBRARIES ggml-cpu "${FLAGOS_SPACEMIT_SPERT_LIBRARY}")
list(APPEND FLAGOS_PRIVATE_DEFINITIONS GGML_FLAGOS_HAVE_SPACEMIT)
message(STATUS "FlagOS: enabling experimental SpaceMiT native provider")
