// Reference side of the logit comparison: runs a GGUF through llama.cpp on
// the CPU and writes the token ids and the logits for every prompt position.
//
//   llama_dump MODEL PROMPT_FILE OUT_PREFIX [THREADS] [--tokens-only]
//
// PROMPT_FILE is read as raw bytes and tokenized with llama.cpp's tokenizer
// (BOS added, special tokens such as <|turn> parsed). Outputs:
//   OUT_PREFIX.tokens  u32 little-endian token ids
//   OUT_PREFIX.logits  f32 little-endian, n_tokens rows of n_vocab
//                      (skipped with --tokens-only)
//
// The KV cache is kept in f32 and flash attention is off so the reference
// adds no rounding of its own beyond llama.cpp's normal CPU kernels.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static void quiet_log(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL PROMPT_FILE OUT_PREFIX [THREADS] [--tokens-only]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string prompt_path = argv[2];
    const std::string out_prefix = argv[3];
    const int threads = argc > 4 ? atoi(argv[4]) : 8;
    const bool tokens_only = argc > 5 && std::string(argv[5]) == "--tokens-only";

    std::ifstream prompt_file(prompt_path, std::ios::binary);
    if (!prompt_file) {
        fprintf(stderr, "cannot read %s\n", prompt_path.c_str());
        return 1;
    }
    const std::string prompt((std::istreambuf_iterator<char>(prompt_file)), std::istreambuf_iterator<char>());

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> tokens(prompt.size() + 8);
    const int n_tokens = llama_tokenize(vocab, prompt.data(), (int32_t) prompt.size(), tokens.data(),
                                        (int32_t) tokens.size(), /*add_special*/ true, /*parse_special*/ true);
    if (n_tokens <= 0) {
        fprintf(stderr, "tokenization failed (%d)\n", n_tokens);
        return 1;
    }
    tokens.resize(n_tokens);

    std::ofstream token_out(out_prefix + ".tokens", std::ios::binary);
    for (llama_token token : tokens) {
        const uint32_t id = (uint32_t) token;
        token_out.write(reinterpret_cast<const char *>(&id), sizeof(id));
    }
    token_out.close();
    if (tokens_only) {
        llama_model_free(model);
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_tokens < 512 ? 512 : n_tokens;
    cparams.n_batch = cparams.n_ctx;
    cparams.n_ubatch = cparams.n_ctx;
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;
    cparams.type_k = GGML_TYPE_F32;
    cparams.type_v = GGML_TYPE_F32;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "cannot create context\n");
        return 1;
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    std::ofstream logit_out(out_prefix + ".logits", std::ios::binary);
    for (int i = 0; i < n_tokens; ++i) {
        const float * row = llama_get_logits_ith(ctx, i);
        logit_out.write(reinterpret_cast<const char *>(row), sizeof(float) * n_vocab);
    }
    fprintf(stderr, "wrote %d tokens x %d logits\n", n_tokens, n_vocab);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
