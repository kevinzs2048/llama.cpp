#include "ggml-flagos.h"

ggml_backend_reg_t flagos_check_public_header(void);

ggml_backend_reg_t flagos_check_public_header(void) {
    return ggml_backend_flagos_reg();
}
