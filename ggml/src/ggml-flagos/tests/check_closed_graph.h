#pragma once
#include "ggml-impl.h"

// Attach whole-graph operand use counts to an explicitly closed test graph.
// Preserve the test's node order and membership; GGML supplies the metadata.
static void flagos_check_closed_graph(ggml_context * context, ggml_cgraph & graph) {
    ggml_cgraph * complete = ggml_new_graph_custom(context,
        static_cast<size_t>(graph.n_nodes) * (GGML_MAX_SRC + 1) + 1, false);
    for (int index = 0; index < graph.n_nodes; ++index) {
        ggml_build_forward_expand(complete, graph.nodes[index]);
    }
    graph.use_counts = complete->use_counts;
    graph.visited_hash_set = complete->visited_hash_set;
}
