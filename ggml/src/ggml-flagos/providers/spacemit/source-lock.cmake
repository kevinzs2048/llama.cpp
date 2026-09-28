# Source: spacemit-com/llama.cpp, commit 4e782bcb7e43bc73cf67ab0210a5490369e06a40.
# Fail closed when the native source changes; review the adapter before updating the lock.
set(spacemit_locked_sources
    "ggml/src/ggml-spacemit/ggml-spacemit.cpp|d753b19e8aa10dd026de6041801209093d74b9a14fd4f9fbb0341ac5d8a305a0"
    "ggml/src/ggml-spacemit/ime.cpp|0c23be217aef2b0930082f46b74fde5536d77d7d458b5585096e0cd501e55a2f"
    "ggml/src/ggml-spacemit/ime.h|bcfd2a75eaae19f89e9cba13698c6c8540528991041039a3281a6dc72ce6b2c5"
    "ggml/src/ggml-spacemit/ime1_kernels.cpp|ea1170753bdaca75d629001d7e820ec315bbd54511c37d1ed4c9347eda67bc46"
    "ggml/src/ggml-spacemit/ime2_kernels.cpp|98b9f38041ead963baf59628167979d667f44c52570cb80d9340efce775fae73"
    "ggml/src/ggml-spacemit/ime_kernels.h|84da9984cab8de6430cb99088a1ae85443678e98f99ab09b050804f907deabb2"
    "ggml/src/ggml-spacemit/repack.cpp|65a3e0359ebc9949bac6ea6f9b4b44af811a8160464f394a659e5c4f39cf5a2d"
    "ggml/src/ggml-spacemit/repack.h|70f7c5aae3b7972841bbb68e05acb84f299c08af61ea8608ab233fc432db8920"
    "ggml/src/ggml-spacemit/rvv_kernels.cpp|4d9d08cc4c0bba888567c2e38b06c90340dfe7485c77825d18519b9b7f597904"
    "ggml/src/ggml-spacemit/rvv_kernels.h|81a86bae619360faa03562dd91a12e2e81d995b7fb98f4756e6f2f0a2a328beb"
    "ggml/src/ggml-spacemit/scalar_kernels.cpp|a1e647058529230678f0a106ec90fafe6552680c40d7081ff213db55a874b3ea"
    "ggml/src/ggml-spacemit/scalar_kernels.h|69244bb1ca12ea118431d6333cc56be49b339a71ae7965166c91ca364e96b5b0"
    "ggml/src/ggml-spacemit/spacemit-context.h|7af2c76d778a1d98c200c14f1e76b9fef7ddb40eb980bf6655e50bf7c40d8a4a"
    "ggml/src/ggml-spacemit/spacemit-env.h|4b9f5dd0c97eeab640269531ba38adaf836d0af01fd47e65b4627171a6d726e2"
    "ggml/src/ggml-spacemit/spacemit-kernels.h|02b46be7dc3a0636ab7f3b64cc874b1a78773ef4ab181980b157f613fa6a1940"
    "ggml/src/ggml-spacemit/spacemit-opnode.h|9cedcfa87a31e3d5cddeb43f890dcd2de483150a6b3c87fdaf3dd9248cd19d9e"
    "ggml/src/ggml-spacemit/spacemit-session.h|ff360b98db2db82e72f853138d9c3ed56dd06e9a734683121462c3625d2c1283"
    "ggml/src/ggml-spacemit/spine_barrier.h|ddba450200089601aac492a45d6be7881eeef02cd0a4abedf69db5c4b9dfe134"
    "ggml/src/ggml-spacemit/spine_mem_pool.cpp|de9c9b292c0e809c94c2a7804ab0b1ba6224a01b660f7533f8e95a612e18c9bc"
    "ggml/src/ggml-spacemit/spine_mem_pool.h|72da78e7e0441a75b87c63a88ab51b93002a35d97977f3287150186093683420"
    "ggml/include/ggml-spacemit.h|8ae2aabea5aa7f4097116401c9638957ee1d0e36f80206b57fee89058271cf9b"
    "ggml/include/ggml-profile.h|417617d07b38f564917b1a7599f8349c2a86a23dd10e1dc6f506f7d46d1856f5"
)
foreach(entry IN LISTS spacemit_locked_sources)
    string(REPLACE "|" ";" entry "${entry}")
    list(GET entry 0 path)
    list(GET entry 1 expected)
    if (NOT EXISTS "${FLAGOS_SPACEMIT_SOURCE_DIR}/${path}")
        message(FATAL_ERROR "SpaceMiT source missing: ${path}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${FLAGOS_SPACEMIT_SOURCE_DIR}/${path}")
    file(SHA256 "${FLAGOS_SPACEMIT_SOURCE_DIR}/${path}" actual)
    if (NOT actual STREQUAL expected)
        message(FATAL_ERROR "SpaceMiT source differs from pinned commit: ${path}; see providers/spacemit/README.md")
    endif()
endforeach()
