#include "../providers/amd/flagos-amd-op-contracts.h"

#include <cstdio>
#include <cstdlib>

static void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "CHECK_FAILED %s\n", message); std::exit(2); }
}

int main() {
    ggml_init_params params { 1u << 20, nullptr, false };
    auto * ctx = ggml_init(params);
    require(ctx != nullptr, "context");
    auto * values = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    auto * gdn = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    gdn->op = GGML_OP_GATED_DELTA_NET;
    gdn->src[2] = values;
    gdn->op_params[0] = 1;
    auto * suffix = ggml_view_1d(ctx, gdn, 6, 2 * sizeof(float));
    auto * destination = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6);
    auto * copy = ggml_cpy(ctx, suffix, destination);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, copy);
    flagos_pattern_candidate candidate;
    for (int i = 0; i < graph->n_nodes; ++i) candidate.node_indices.push_back(i);
    candidate.required_output_node_indices = { graph->n_nodes - 1 };
    require(graph->n_nodes == 3, "GDN/suffix/copy fixture");
    require(flagos_amd::gdn_cache_only_outputs_are_safe(graph, candidate, gdn, copy), "closed region remains eligible");
    candidate.required_output_node_indices.insert(candidate.required_output_node_indices.begin(), 0);
    require(!flagos_amd::gdn_cache_only_outputs_are_safe(graph, candidate, gdn, copy), "complete required backing output rejects cache-only");
    candidate.required_output_node_indices = { 2 };
    auto * prefix = ggml_view_1d(ctx, gdn, 2, 0);
    auto * prefix_observer = ggml_dup(ctx, prefix);
    ggml_build_forward_expand(graph, prefix_observer);
    require(flagos_amd::gdn_cache_only_outputs_are_safe(graph, candidate, gdn, copy), "visible attention-prefix observer is safe");
    auto split = *graph;
    split.n_nodes = 3;
    require(!flagos_amd::gdn_cache_only_outputs_are_safe(&split, candidate, gdn, copy), "cross-split backing use rejects cache-only");
    auto * suffix_observer = ggml_dup(ctx, suffix);
    ggml_build_forward_expand(graph, suffix_observer);
    require(!flagos_amd::gdn_cache_only_outputs_are_safe(graph, candidate, gdn, copy), "visible snapshot observer rejects cache-only");
    auto * alias_graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(alias_graph, copy);
    ggml_build_forward_expand(alias_graph, suffix_observer);
    auto alias_split = *alias_graph;
    alias_split.n_nodes = 3;
    require(!flagos_amd::gdn_cache_only_outputs_are_safe(&alias_split, candidate, gdn, copy), "cross-split covered alias rejects cache-only");
    ggml_free(ctx);
    std::puts("AMD GDN cache-only graph contract checks passed");
    return 0;
}
