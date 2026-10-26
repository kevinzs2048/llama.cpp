#include "ggml.h"
#include "ggml-backend.h"
#include "../providers/amd/flagos-amd-op-contracts.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "CHECK_FAILED %s\n", message); std::exit(2); }
}

static void check_layout() {
    ggml_init_params params { 1u << 20, nullptr, true };
    auto * ctx = ggml_init(params);
    require(ctx != nullptr, "context");
    auto * table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 17, 9);
    auto * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 5);
    auto * op = ggml_get_rows(ctx, table, indices);
    flagos_amd::dense_gather_signature signature;
    require(flagos_amd::describe_dense_gather(op, &signature), "valid layout");
    require(signature.columns == 17 && signature.rows == 5, "shape");
    indices->type = GGML_TYPE_I64;
    require(!flagos_amd::describe_dense_gather(op, &signature), "reject I64 indices");
    indices->type = GGML_TYPE_I32;
    table->ne[1] = INT32_MAX;
    require(!flagos_amd::describe_dense_gather(op, &signature), "reject offset overflow");
    table->ne[1] = 9;
    auto * view = ggml_view_2d(ctx, table, 17, 4, 2 * table->nb[1], 0);
    require(!flagos_amd::describe_dense_gather(ggml_get_rows(ctx, view, indices), &signature), "reject strided table");
    auto * empty = ggml_get_rows(ctx, table, ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 0));
    require(flagos_amd::describe_dense_gather(empty, &signature) && signature.rows == 0, "empty rows");
    ggml_free(ctx);
}

static void check_gpu(ggml_backend_t backend, int columns) {
    constexpr int table_rows = 9, result_rows = 5, guard = 16;
    ggml_init_params params { 1u << 20, nullptr, true };
    auto * ctx = ggml_init(params);
    require(ctx != nullptr, "context");
    auto * table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, table_rows);
    auto * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, result_rows);
    auto * op = ggml_get_rows(ctx, table, indices);
    auto * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, guard * 2 + columns * result_rows);
    auto buft = ggml_backend_get_default_buffer_type(backend);
    auto tb = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(table));
    auto ib = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(indices));
    auto ob = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(storage));
    require(tb && ib && ob, "buffers");
    require(ggml_backend_tensor_alloc(tb, table, ggml_backend_buffer_get_base(tb)) == GGML_STATUS_SUCCESS, "table allocation");
    require(ggml_backend_tensor_alloc(ib, indices, ggml_backend_buffer_get_base(ib)) == GGML_STATUS_SUCCESS, "index allocation");
    require(ggml_backend_tensor_alloc(ob, storage, ggml_backend_buffer_get_base(ob)) == GGML_STATUS_SUCCESS, "storage allocation");
    require(ggml_backend_tensor_alloc(ob, op, static_cast<char *>(storage->data) + guard * sizeof(float)) == GGML_STATUS_SUCCESS, "output allocation");
    auto dev = ggml_backend_get_device(backend);
    require(ggml_backend_dev_supports_op(dev, op), "dense gather supported");
    void * original = op->data;
    op->data = table->data;
    require(!ggml_backend_dev_supports_op(dev, op), "reject table alias");
    op->data = indices->data;
    require(!ggml_backend_dev_supports_op(dev, op), "reject index alias");
    op->data = original;
    std::vector<uint32_t> bits(columns * table_rows);
    for (size_t i = 0; i < bits.size(); ++i) bits[i] = uint32_t(i) * 2654435761u;
    const uint32_t special[] = { 0, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc01234u, 1 };
    for (size_t i = 0; i < bits.size() && i < sizeof(special) / sizeof(special[0]); ++i) bits[i] = special[i];
    const int32_t rows[] = { 8, 0, 4, 4, 1 };
    std::vector<uint32_t> actual(guard * 2 + columns * result_rows, 0x5a5a5a5au);
    ggml_backend_tensor_set(table, bits.data(), 0, ggml_nbytes(table));
    ggml_backend_tensor_set(indices, rows, 0, sizeof(rows));
    ggml_backend_tensor_set(storage, actual.data(), 0, ggml_nbytes(storage));
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, op);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute");
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(storage, actual.data(), 0, ggml_nbytes(storage));
    for (int i = 0; i < guard; ++i) {
        require(actual[i] == 0x5a5a5a5au && actual[actual.size() - 1 - i] == 0x5a5a5a5au, "output guards");
    }
    for (int row = 0; row < result_rows; ++row) {
        require(std::memcmp(actual.data() + guard + row * columns, bits.data() + rows[row] * columns, columns * sizeof(uint32_t)) == 0, "raw bit output");
    }
    std::vector<uint32_t> table_after(bits.size());
    int32_t indices_after[result_rows];
    ggml_backend_tensor_get(table, table_after.data(), 0, ggml_nbytes(table));
    ggml_backend_tensor_get(indices, indices_after, 0, sizeof(indices_after));
    require(table_after == bits && std::memcmp(rows, indices_after, sizeof(rows)) == 0, "inputs unchanged");
    auto * empty_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 0);
    auto * empty = ggml_get_rows(ctx, table, empty_indices);
    require(ggml_backend_dev_supports_op(dev, empty), "empty output supported");
    auto * empty_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(empty_graph, empty);
    require(ggml_backend_graph_compute(backend, empty_graph) == GGML_STATUS_SUCCESS, "empty compute without bindings");
    ggml_backend_buffer_free(ob); ggml_backend_buffer_free(ib); ggml_backend_buffer_free(tb);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    require(argc == 2, "backend directory or --contracts-only");
    check_layout();
    if (std::strcmp(argv[1], "--contracts-only") == 0) return 0;
    ggml_backend_load_all_from_path(argv[1]);
    auto * backend = ggml_backend_init_by_name("FlagOS:AMD:0", nullptr);
    require(backend != nullptr, "AMD backend");
    for (int columns : { 1, 127, 256, 257, 1025 }) check_gpu(backend, columns);
    ggml_backend_free(backend);
    std::puts("AMD dense gather: contracts, raw bits, alias rejection and guards passed");
    return 0;
}
