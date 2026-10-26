#pragma once

#include <cstdint>
#include <initializer_list>

// Capability-only tests: synthetic addresses are never submitted to a kernel.
static void check_amd_gdn_view_alignment(ggml_backend_dev_t dev) {
    const int64_t dimensions[6][4] = {
        {128, 16, 37, 1}, {128, 16, 37, 1}, {128, 32, 37, 1},
        {1, 32, 37, 1}, {1, 32, 37, 1}, {128, 128, 32, 1},
    };
    int positives = 0;
    int negatives = 0;
    int checks = 0;
    for (int nested = 0; nested < 2; ++nested) {
        for (int input = 0; input < 6; ++input) {
            for (size_t offset : {size_t(0), size_t(4), size_t(8), size_t(12), size_t(16)}) {
                ggml_init_params params = {1u << 20, nullptr, true};
                auto * ctx = ggml_init(params);
                CHECK(ctx);
                ggml_tensor * sources[6];
                for (int i = 0; i < 6; ++i) {
                    const auto * dims = dimensions[i];
                    const int64_t count = dims[0] * dims[1] * dims[2] * dims[3];
                    auto * root = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count + 128);
                    auto * source = nested ? ggml_view_1d(ctx, root, count + 64, 16) : root;
                    const size_t view_offset = i == input ? offset : 0;
                    sources[i] = ggml_view_4d(ctx, source, dims[0], dims[1], dims[2], dims[3],
                            dims[0] * 4, dims[0] * dims[1] * 4, dims[0] * dims[1] * dims[2] * 4,
                            view_offset);
                    CHECK(sources[i]->data == nullptr && sources[i]->view_src == root);
                    CHECK(sources[i]->view_offs == view_offset + (nested ? 16 : 0));
                }
                auto * output = ggml_gated_delta_net(ctx, sources[0], sources[1], sources[2],
                        sources[3], sources[4], sources[5], 1);
                CHECK(output->data == nullptr);
                const bool aligned = offset % 16 == 0;
                CHECK(ggml_backend_dev_supports_op(dev, output) == aligned);
                aligned ? ++positives : ++negatives;
                ++checks;

                auto * root = sources[input]->view_src;
                root->data = reinterpret_cast<void *>(uintptr_t(0x10000));
                CHECK(ggml_backend_dev_supports_op(dev, output) == aligned);
                ++checks;
                // A known root can compensate an offset; use its effective alignment.
                root->data = reinterpret_cast<void *>(uintptr_t(0x10000) +
                        (16 - sources[input]->view_offs % 16) % 16);
                CHECK(ggml_backend_dev_supports_op(dev, output));
                ++checks;
                // Once bound, the actual pointer is authoritative.
                sources[input]->data = reinterpret_cast<void *>(uintptr_t(0x20000));
                CHECK(ggml_backend_dev_supports_op(dev, output));
                ++checks;
                sources[input]->data = reinterpret_cast<void *>(uintptr_t(0x20004));
                CHECK(!ggml_backend_dev_supports_op(dev, output));
                ++checks;
                ggml_free(ctx);
            }
        }
    }
    CHECK(positives == 24 && negatives == 36 && checks == 300);
    std::puts("GDN_VIEW_ALIGNMENT_COMPLETE capability_checks=300 "
              "preallocation_positive=24 preallocation_negative=36 no_kernel_launch=1");
}
