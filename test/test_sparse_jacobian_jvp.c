
// We need to define _GNU_SOURCE before
// _any_ headers files are imported to get
// the usage statistics of a thread (i.e. have RUSAGE_THREAD) on GNU/Linux
// https://manpages.courier-mta.org/htmlman2/getrusage.2.html
#ifndef _GNU_SOURCE // Avoid possible double-definition warning.
#define _GNU_SOURCE
#endif

#ifdef __clang__
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-const-variable"
#pragma clang diagnostic ignored "-Wparentheses"
#pragma clang diagnostic ignored "-Wunused-label"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#elif __GNUC__
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-const-variable"
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif

// Headers
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialisation
struct futhark_context_config;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
int futhark_context_config_set_tuning_param(struct futhark_context_config *cfg, const char *param_name, size_t new_value);
struct futhark_context;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg, int flag);
void futhark_context_config_set_profiling(struct futhark_context_config *cfg, int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg, int flag);
int futhark_get_tuning_param_count(void);
const char *futhark_get_tuning_param_name(int);
const char *futhark_get_tuning_param_class(int);

// Arrays
struct futhark_f64_1d;
struct futhark_f64_1d *futhark_new_f64_1d(struct futhark_context *ctx, const double *data, int64_t dim0);
struct futhark_f64_1d *futhark_new_raw_f64_1d(struct futhark_context *ctx, unsigned char *data, int64_t dim0);
int futhark_free_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr);
int futhark_values_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr, double *data);
int futhark_index_f64_1d(struct futhark_context *ctx, double *out, struct futhark_f64_1d *arr, int64_t i0);
unsigned char *futhark_values_raw_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr);
const int64_t *futhark_shape_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr);

// Opaque values



// Entry points
int futhark_entry_test_jvp_csr_from_csr_ex4_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0);
int futhark_entry_test_prepared_jvp_ex4_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0);
int futhark_entry_test_prepared_jvp_reuse_two_points(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0);
int futhark_entry_test_sparse_jvp_ex1_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0);
int futhark_entry_test_sparse_jvp_ex2_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0);
int futhark_entry_test_sparse_jvp_ex4_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0);
int futhark_entry_test_sparse_jvp_ex4_with_colors_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0);
int futhark_entry_test_sparse_jvp_ex5_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0);
int futhark_entry_test_sparse_jvp_zero_pattern_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0);

// Miscellaneous
int futhark_context_sync(struct futhark_context *ctx);
void futhark_context_config_set_cache_file(struct futhark_context_config *cfg, const char *f);
char *futhark_context_get_error(struct futhark_context *ctx);
void futhark_context_set_logging_file(struct futhark_context *ctx, FILE *f);
void futhark_context_pause_profiling(struct futhark_context *ctx);
void futhark_context_unpause_profiling(struct futhark_context *ctx);
char *futhark_context_report(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
#define FUTHARK_BACKEND_c
#define FUTHARK_SUCCESS 0
#define FUTHARK_PROGRAM_ERROR 2
#define FUTHARK_OUT_OF_MEMORY 3

#ifdef __cplusplus
}
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
// If NDEBUG is set, the assert() macro will do nothing. Since Futhark
// (unfortunately) makes use of assert() for error detection (and even some
// side effects), we want to avoid that.
#undef NDEBUG
#include <assert.h>
#include <stdarg.h>
#define SCALAR_FUN_ATTR static inline
// Start of util.h.
//
// Various helper functions that are useful in all generated C code.

#include <errno.h>
#include <string.h>

static const char *fut_progname = "(embedded Futhark)";

static void futhark_panic(int eval, const char *fmt, ...) __attribute__((noreturn));
static char* msgprintf(const char *s, ...);
static void* slurp_file(const char *filename, size_t *size);
static int dump_file(const char *file, const void *buf, size_t n);
struct str_builder;
static void str_builder_init(struct str_builder *b);
static void str_builder(struct str_builder *b, const char *s, ...);
static char *strclone(const char *str);

static void futhark_panic(int eval, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "%s: ", fut_progname);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  exit(eval);
}

// For generating arbitrary-sized error messages.  It is the callers
// responsibility to free the buffer at some point.
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + (size_t)vsnprintf(NULL, 0, s, vl);
  char *buffer = (char*) malloc(needed);
  va_start(vl, s); // Must re-init.
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}

static inline void check_err(int errval, int sets_errno, const char *fun, int line,
                             const char *msg, ...) {
  if (errval) {
    char errnum[10];

    va_list vl;
    va_start(vl, msg);

    fprintf(stderr, "ERROR: ");
    vfprintf(stderr, msg, vl);
    fprintf(stderr, " in %s() at line %d with error code %s\n",
            fun, line,
            sets_errno ? strerror(errno) : errnum);
    exit(errval);
  }
}

#define CHECK_ERR(err, ...) check_err(err, 0, __func__, __LINE__, __VA_ARGS__)
#define CHECK_ERRNO(err, ...) check_err(err, 1, __func__, __LINE__, __VA_ARGS__)

// Read the rest of an open file into a NUL-terminated string; returns
// NULL on error.
static void* fslurp_file(FILE *f, size_t *size) {
  long start = ftell(f);
  fseek(f, 0, SEEK_END);
  long src_size = ftell(f)-start;
  fseek(f, start, SEEK_SET);
  unsigned char *s = (unsigned char*) malloc((size_t)src_size + 1);
  if (fread(s, 1, (size_t)src_size, f) != (size_t)src_size) {
    free(s);
    s = NULL;
  } else {
    s[src_size] = '\0';
  }

  if (size) {
    *size = (size_t)src_size;
  }

  return s;
}

// Read a file into a NUL-terminated string; returns NULL on error.
static void* slurp_file(const char *filename, size_t *size) {
  FILE *f = fopen(filename, "rb"); // To avoid Windows messing with linebreaks.
  if (f == NULL) return NULL;
  unsigned char *s = fslurp_file(f, size);
  fclose(f);
  return s;
}

// Dump 'n' bytes from 'buf' into the file at the designated location.
// Returns 0 on success.
static int dump_file(const char *file, const void *buf, size_t n) {
  FILE *f = fopen(file, "w");

  if (f == NULL) {
    return 1;
  }

  if (fwrite(buf, sizeof(char), n, f) != n) {
    return 1;
  }

  if (fclose(f) != 0) {
    return 1;
  }

  return 0;
}

struct str_builder {
  char *str;
  size_t capacity; // Size of buffer.
  size_t used; // Bytes used, *not* including final zero.
};

static void str_builder_init(struct str_builder *b) {
  b->capacity = 10;
  b->used = 0;
  b->str = malloc(b->capacity);
  b->str[0] = 0;
}

static void str_builder(struct str_builder *b, const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = (size_t)vsnprintf(NULL, 0, s, vl);

  while (b->capacity < b->used + needed + 1) {
    b->capacity *= 2;
    b->str = realloc(b->str, b->capacity);
  }

  va_start(vl, s); // Must re-init.
  vsnprintf(b->str+b->used, b->capacity-b->used, s, vl);
  b->used += needed;
}

static void str_builder_str(struct str_builder *b, const char *s) {
  size_t needed = strlen(s);
  if (b->capacity < b->used + needed + 1) {
    b->capacity *= 2;
    b->str = realloc(b->str, b->capacity);
  }
  strcpy(b->str+b->used, s);
  b->used += needed;
}

static void str_builder_char(struct str_builder *b, char c) {
  size_t needed = 1;
  if (b->capacity < b->used + needed + 1) {
    b->capacity *= 2;
    b->str = realloc(b->str, b->capacity);
  }
  b->str[b->used] = c;
  b->str[b->used+1] = 0;
  b->used += needed;
}

static void str_builder_json_str(struct str_builder* sb, const char* s) {
  str_builder_char(sb, '"');
  for (int j = 0; s[j]; j++) {
    char c = s[j];
    switch (c) {
    case '\n':
      str_builder_str(sb, "\\n");
      break;
    case '"':
      str_builder_str(sb, "\\\"");
      break;
    default:
      str_builder_char(sb, c);
    }
  }
  str_builder_char(sb, '"');
}

static char *strclone(const char *str) {
  size_t size = strlen(str) + 1;
  char *copy = (char*) malloc(size);
  if (copy == NULL) {
    return NULL;
  }

  memcpy(copy, str, size);
  return copy;
}

// Assumes NULL-terminated.
static char *strconcat(const char *src_fragments[]) {
  size_t src_len = 0;
  const char **p;

  for (p = src_fragments; *p; p++) {
    src_len += strlen(*p);
  }

  char *src = (char*) malloc(src_len + 1);
  size_t n = 0;
  for (p = src_fragments; *p; p++) {
    strcpy(src + n, *p);
    n += strlen(*p);
  }

  return src;
}

// End of util.h.
// Start of cache.h

#define CACHE_HASH_SIZE 8 // In 32-bit words.

struct cache_hash {
  uint32_t hash[CACHE_HASH_SIZE];
};

// Initialise a blank cache.
static void cache_hash_init(struct cache_hash *c);

// Hash some bytes and add them to the accumulated hash.
static void cache_hash(struct cache_hash *out, const char *in, size_t n);

// Try to restore cache contents from a file with the given name.
// Assumes the cache is invalid if it contains the given hash.
// Allocates memory and reads the cache conents, which is returned in
// *buf with size *buflen.  If the cache is successfully loaded, this
// function returns 0.  Otherwise it returns nonzero.  Errno is set if
// the failure to load the cache is due to anything except invalid
// cache conents.  Note that failing to restore the cache is not
// necessarily a problem: it might just be invalid or not created yet.
static int cache_restore(const char *fname, const struct cache_hash *hash,
                         unsigned char **buf, size_t *buflen);

// Store cache contents in the given file, with the given hash.
static int cache_store(const char *fname, const struct cache_hash *hash,
                       const unsigned char *buf, size_t buflen);

// Now for the implementation.

static void cache_hash_init(struct cache_hash *c) {
  memset(c->hash, 0, CACHE_HASH_SIZE * sizeof(uint32_t));
}

static void cache_hash(struct cache_hash *out, const char *in, size_t n) {
  // Adaptation of djb2 for larger output size by storing intermediate
  // states.
  uint32_t hash = 5381;
  for (size_t i = 0; i < n; i++) {
    hash = ((hash << 5) + hash) + in[i];
    out->hash[i % CACHE_HASH_SIZE] ^= hash;
  }
}

#define CACHE_HEADER_SIZE 8
static const char cache_header[CACHE_HEADER_SIZE] = "FUTHARK\0";

static int cache_restore(const char *fname, const struct cache_hash *hash,
                         unsigned char **buf, size_t *buflen) {
  FILE *f = fopen(fname, "rb");

  if (f == NULL) {
    return 1;
  }

  char f_header[CACHE_HEADER_SIZE];

  if (fread(f_header, sizeof(char), CACHE_HEADER_SIZE, f) != CACHE_HEADER_SIZE) {
    goto error;
  }

  if (memcmp(f_header, cache_header, CACHE_HEADER_SIZE) != 0) {
    goto error;
  }

  if (fseek(f, 0, SEEK_END) != 0) {
    goto error;
  }
  int64_t f_size = (int64_t)ftell(f);
  if (fseek(f, CACHE_HEADER_SIZE, SEEK_SET) != 0) {
    goto error;
  }

  int64_t expected_size;

  if (fread(&expected_size, sizeof(int64_t), 1, f) != 1) {
    goto error;
  }

  if (f_size != expected_size) {
    errno = 0;
    goto error;
  }

  int32_t f_hash[CACHE_HASH_SIZE];

  if (fread(f_hash, sizeof(int32_t), CACHE_HASH_SIZE, f) != CACHE_HASH_SIZE) {
    goto error;
  }

  if (memcmp(f_hash, hash->hash, CACHE_HASH_SIZE) != 0) {
    errno = 0;
    goto error;
  }

  *buflen = f_size - CACHE_HEADER_SIZE - sizeof(int64_t) - CACHE_HASH_SIZE*sizeof(int32_t);
  *buf = malloc(*buflen);
  if (fread(*buf, sizeof(char), *buflen, f) != *buflen) {
    free(*buf);
    goto error;
  }

  fclose(f);

  return 0;

 error:
  fclose(f);
  return 1;
}

static int cache_store(const char *fname, const struct cache_hash *hash,
                       const unsigned char *buf, size_t buflen) {
  FILE *f = fopen(fname, "wb");

  if (f == NULL) {
    return 1;
  }

  if (fwrite(cache_header, CACHE_HEADER_SIZE, 1, f) != 1) {
    goto error;
  }

  int64_t size = CACHE_HEADER_SIZE + sizeof(int64_t) + CACHE_HASH_SIZE*sizeof(int32_t) + buflen;

  if (fwrite(&size, sizeof(size), 1, f) != 1) {
    goto error;
  }

  if (fwrite(hash->hash, sizeof(int32_t), CACHE_HASH_SIZE, f) != CACHE_HASH_SIZE) {
    goto error;
  }

  if (fwrite(buf, sizeof(unsigned char), buflen, f) != buflen) {
    goto error;
  }

  fclose(f);

  return 0;

 error:
  fclose(f);
  return 1;
}

// End of cache.h
// Start of half.h.

// Conversion functions are from http://half.sourceforge.net/, but
// translated to C.
//
// Copyright (c) 2012-2021 Christian Rau
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef __OPENCL_VERSION__
#define __constant
#endif

__constant static const uint16_t base_table[512] = {
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080, 0x0100,
  0x0200, 0x0400, 0x0800, 0x0C00, 0x1000, 0x1400, 0x1800, 0x1C00, 0x2000, 0x2400, 0x2800, 0x2C00, 0x3000, 0x3400, 0x3800, 0x3C00,
  0x4000, 0x4400, 0x4800, 0x4C00, 0x5000, 0x5400, 0x5800, 0x5C00, 0x6000, 0x6400, 0x6800, 0x6C00, 0x7000, 0x7400, 0x7800, 0x7C00,
  0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
  0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
  0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
  0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
  0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
  0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
  0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
  0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
  0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
  0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
  0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
  0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
  0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
  0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8001, 0x8002, 0x8004, 0x8008, 0x8010, 0x8020, 0x8040, 0x8080, 0x8100,
  0x8200, 0x8400, 0x8800, 0x8C00, 0x9000, 0x9400, 0x9800, 0x9C00, 0xA000, 0xA400, 0xA800, 0xAC00, 0xB000, 0xB400, 0xB800, 0xBC00,
  0xC000, 0xC400, 0xC800, 0xCC00, 0xD000, 0xD400, 0xD800, 0xDC00, 0xE000, 0xE400, 0xE800, 0xEC00, 0xF000, 0xF400, 0xF800, 0xFC00,
  0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
  0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
  0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
  0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
  0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
  0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
  0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00 };

__constant static const unsigned char shift_table[512] = {
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
  13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 13,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
  13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
  24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 13 };

__constant static const uint32_t mantissa_table[2048] = {
  0x00000000, 0x33800000, 0x34000000, 0x34400000, 0x34800000, 0x34A00000, 0x34C00000, 0x34E00000, 0x35000000, 0x35100000, 0x35200000, 0x35300000, 0x35400000, 0x35500000, 0x35600000, 0x35700000,
  0x35800000, 0x35880000, 0x35900000, 0x35980000, 0x35A00000, 0x35A80000, 0x35B00000, 0x35B80000, 0x35C00000, 0x35C80000, 0x35D00000, 0x35D80000, 0x35E00000, 0x35E80000, 0x35F00000, 0x35F80000,
  0x36000000, 0x36040000, 0x36080000, 0x360C0000, 0x36100000, 0x36140000, 0x36180000, 0x361C0000, 0x36200000, 0x36240000, 0x36280000, 0x362C0000, 0x36300000, 0x36340000, 0x36380000, 0x363C0000,
  0x36400000, 0x36440000, 0x36480000, 0x364C0000, 0x36500000, 0x36540000, 0x36580000, 0x365C0000, 0x36600000, 0x36640000, 0x36680000, 0x366C0000, 0x36700000, 0x36740000, 0x36780000, 0x367C0000,
  0x36800000, 0x36820000, 0x36840000, 0x36860000, 0x36880000, 0x368A0000, 0x368C0000, 0x368E0000, 0x36900000, 0x36920000, 0x36940000, 0x36960000, 0x36980000, 0x369A0000, 0x369C0000, 0x369E0000,
  0x36A00000, 0x36A20000, 0x36A40000, 0x36A60000, 0x36A80000, 0x36AA0000, 0x36AC0000, 0x36AE0000, 0x36B00000, 0x36B20000, 0x36B40000, 0x36B60000, 0x36B80000, 0x36BA0000, 0x36BC0000, 0x36BE0000,
  0x36C00000, 0x36C20000, 0x36C40000, 0x36C60000, 0x36C80000, 0x36CA0000, 0x36CC0000, 0x36CE0000, 0x36D00000, 0x36D20000, 0x36D40000, 0x36D60000, 0x36D80000, 0x36DA0000, 0x36DC0000, 0x36DE0000,
  0x36E00000, 0x36E20000, 0x36E40000, 0x36E60000, 0x36E80000, 0x36EA0000, 0x36EC0000, 0x36EE0000, 0x36F00000, 0x36F20000, 0x36F40000, 0x36F60000, 0x36F80000, 0x36FA0000, 0x36FC0000, 0x36FE0000,
  0x37000000, 0x37010000, 0x37020000, 0x37030000, 0x37040000, 0x37050000, 0x37060000, 0x37070000, 0x37080000, 0x37090000, 0x370A0000, 0x370B0000, 0x370C0000, 0x370D0000, 0x370E0000, 0x370F0000,
  0x37100000, 0x37110000, 0x37120000, 0x37130000, 0x37140000, 0x37150000, 0x37160000, 0x37170000, 0x37180000, 0x37190000, 0x371A0000, 0x371B0000, 0x371C0000, 0x371D0000, 0x371E0000, 0x371F0000,
  0x37200000, 0x37210000, 0x37220000, 0x37230000, 0x37240000, 0x37250000, 0x37260000, 0x37270000, 0x37280000, 0x37290000, 0x372A0000, 0x372B0000, 0x372C0000, 0x372D0000, 0x372E0000, 0x372F0000,
  0x37300000, 0x37310000, 0x37320000, 0x37330000, 0x37340000, 0x37350000, 0x37360000, 0x37370000, 0x37380000, 0x37390000, 0x373A0000, 0x373B0000, 0x373C0000, 0x373D0000, 0x373E0000, 0x373F0000,
  0x37400000, 0x37410000, 0x37420000, 0x37430000, 0x37440000, 0x37450000, 0x37460000, 0x37470000, 0x37480000, 0x37490000, 0x374A0000, 0x374B0000, 0x374C0000, 0x374D0000, 0x374E0000, 0x374F0000,
  0x37500000, 0x37510000, 0x37520000, 0x37530000, 0x37540000, 0x37550000, 0x37560000, 0x37570000, 0x37580000, 0x37590000, 0x375A0000, 0x375B0000, 0x375C0000, 0x375D0000, 0x375E0000, 0x375F0000,
  0x37600000, 0x37610000, 0x37620000, 0x37630000, 0x37640000, 0x37650000, 0x37660000, 0x37670000, 0x37680000, 0x37690000, 0x376A0000, 0x376B0000, 0x376C0000, 0x376D0000, 0x376E0000, 0x376F0000,
  0x37700000, 0x37710000, 0x37720000, 0x37730000, 0x37740000, 0x37750000, 0x37760000, 0x37770000, 0x37780000, 0x37790000, 0x377A0000, 0x377B0000, 0x377C0000, 0x377D0000, 0x377E0000, 0x377F0000,
  0x37800000, 0x37808000, 0x37810000, 0x37818000, 0x37820000, 0x37828000, 0x37830000, 0x37838000, 0x37840000, 0x37848000, 0x37850000, 0x37858000, 0x37860000, 0x37868000, 0x37870000, 0x37878000,
  0x37880000, 0x37888000, 0x37890000, 0x37898000, 0x378A0000, 0x378A8000, 0x378B0000, 0x378B8000, 0x378C0000, 0x378C8000, 0x378D0000, 0x378D8000, 0x378E0000, 0x378E8000, 0x378F0000, 0x378F8000,
  0x37900000, 0x37908000, 0x37910000, 0x37918000, 0x37920000, 0x37928000, 0x37930000, 0x37938000, 0x37940000, 0x37948000, 0x37950000, 0x37958000, 0x37960000, 0x37968000, 0x37970000, 0x37978000,
  0x37980000, 0x37988000, 0x37990000, 0x37998000, 0x379A0000, 0x379A8000, 0x379B0000, 0x379B8000, 0x379C0000, 0x379C8000, 0x379D0000, 0x379D8000, 0x379E0000, 0x379E8000, 0x379F0000, 0x379F8000,
  0x37A00000, 0x37A08000, 0x37A10000, 0x37A18000, 0x37A20000, 0x37A28000, 0x37A30000, 0x37A38000, 0x37A40000, 0x37A48000, 0x37A50000, 0x37A58000, 0x37A60000, 0x37A68000, 0x37A70000, 0x37A78000,
  0x37A80000, 0x37A88000, 0x37A90000, 0x37A98000, 0x37AA0000, 0x37AA8000, 0x37AB0000, 0x37AB8000, 0x37AC0000, 0x37AC8000, 0x37AD0000, 0x37AD8000, 0x37AE0000, 0x37AE8000, 0x37AF0000, 0x37AF8000,
  0x37B00000, 0x37B08000, 0x37B10000, 0x37B18000, 0x37B20000, 0x37B28000, 0x37B30000, 0x37B38000, 0x37B40000, 0x37B48000, 0x37B50000, 0x37B58000, 0x37B60000, 0x37B68000, 0x37B70000, 0x37B78000,
  0x37B80000, 0x37B88000, 0x37B90000, 0x37B98000, 0x37BA0000, 0x37BA8000, 0x37BB0000, 0x37BB8000, 0x37BC0000, 0x37BC8000, 0x37BD0000, 0x37BD8000, 0x37BE0000, 0x37BE8000, 0x37BF0000, 0x37BF8000,
  0x37C00000, 0x37C08000, 0x37C10000, 0x37C18000, 0x37C20000, 0x37C28000, 0x37C30000, 0x37C38000, 0x37C40000, 0x37C48000, 0x37C50000, 0x37C58000, 0x37C60000, 0x37C68000, 0x37C70000, 0x37C78000,
  0x37C80000, 0x37C88000, 0x37C90000, 0x37C98000, 0x37CA0000, 0x37CA8000, 0x37CB0000, 0x37CB8000, 0x37CC0000, 0x37CC8000, 0x37CD0000, 0x37CD8000, 0x37CE0000, 0x37CE8000, 0x37CF0000, 0x37CF8000,
  0x37D00000, 0x37D08000, 0x37D10000, 0x37D18000, 0x37D20000, 0x37D28000, 0x37D30000, 0x37D38000, 0x37D40000, 0x37D48000, 0x37D50000, 0x37D58000, 0x37D60000, 0x37D68000, 0x37D70000, 0x37D78000,
  0x37D80000, 0x37D88000, 0x37D90000, 0x37D98000, 0x37DA0000, 0x37DA8000, 0x37DB0000, 0x37DB8000, 0x37DC0000, 0x37DC8000, 0x37DD0000, 0x37DD8000, 0x37DE0000, 0x37DE8000, 0x37DF0000, 0x37DF8000,
  0x37E00000, 0x37E08000, 0x37E10000, 0x37E18000, 0x37E20000, 0x37E28000, 0x37E30000, 0x37E38000, 0x37E40000, 0x37E48000, 0x37E50000, 0x37E58000, 0x37E60000, 0x37E68000, 0x37E70000, 0x37E78000,
  0x37E80000, 0x37E88000, 0x37E90000, 0x37E98000, 0x37EA0000, 0x37EA8000, 0x37EB0000, 0x37EB8000, 0x37EC0000, 0x37EC8000, 0x37ED0000, 0x37ED8000, 0x37EE0000, 0x37EE8000, 0x37EF0000, 0x37EF8000,
  0x37F00000, 0x37F08000, 0x37F10000, 0x37F18000, 0x37F20000, 0x37F28000, 0x37F30000, 0x37F38000, 0x37F40000, 0x37F48000, 0x37F50000, 0x37F58000, 0x37F60000, 0x37F68000, 0x37F70000, 0x37F78000,
  0x37F80000, 0x37F88000, 0x37F90000, 0x37F98000, 0x37FA0000, 0x37FA8000, 0x37FB0000, 0x37FB8000, 0x37FC0000, 0x37FC8000, 0x37FD0000, 0x37FD8000, 0x37FE0000, 0x37FE8000, 0x37FF0000, 0x37FF8000,
  0x38000000, 0x38004000, 0x38008000, 0x3800C000, 0x38010000, 0x38014000, 0x38018000, 0x3801C000, 0x38020000, 0x38024000, 0x38028000, 0x3802C000, 0x38030000, 0x38034000, 0x38038000, 0x3803C000,
  0x38040000, 0x38044000, 0x38048000, 0x3804C000, 0x38050000, 0x38054000, 0x38058000, 0x3805C000, 0x38060000, 0x38064000, 0x38068000, 0x3806C000, 0x38070000, 0x38074000, 0x38078000, 0x3807C000,
  0x38080000, 0x38084000, 0x38088000, 0x3808C000, 0x38090000, 0x38094000, 0x38098000, 0x3809C000, 0x380A0000, 0x380A4000, 0x380A8000, 0x380AC000, 0x380B0000, 0x380B4000, 0x380B8000, 0x380BC000,
  0x380C0000, 0x380C4000, 0x380C8000, 0x380CC000, 0x380D0000, 0x380D4000, 0x380D8000, 0x380DC000, 0x380E0000, 0x380E4000, 0x380E8000, 0x380EC000, 0x380F0000, 0x380F4000, 0x380F8000, 0x380FC000,
  0x38100000, 0x38104000, 0x38108000, 0x3810C000, 0x38110000, 0x38114000, 0x38118000, 0x3811C000, 0x38120000, 0x38124000, 0x38128000, 0x3812C000, 0x38130000, 0x38134000, 0x38138000, 0x3813C000,
  0x38140000, 0x38144000, 0x38148000, 0x3814C000, 0x38150000, 0x38154000, 0x38158000, 0x3815C000, 0x38160000, 0x38164000, 0x38168000, 0x3816C000, 0x38170000, 0x38174000, 0x38178000, 0x3817C000,
  0x38180000, 0x38184000, 0x38188000, 0x3818C000, 0x38190000, 0x38194000, 0x38198000, 0x3819C000, 0x381A0000, 0x381A4000, 0x381A8000, 0x381AC000, 0x381B0000, 0x381B4000, 0x381B8000, 0x381BC000,
  0x381C0000, 0x381C4000, 0x381C8000, 0x381CC000, 0x381D0000, 0x381D4000, 0x381D8000, 0x381DC000, 0x381E0000, 0x381E4000, 0x381E8000, 0x381EC000, 0x381F0000, 0x381F4000, 0x381F8000, 0x381FC000,
  0x38200000, 0x38204000, 0x38208000, 0x3820C000, 0x38210000, 0x38214000, 0x38218000, 0x3821C000, 0x38220000, 0x38224000, 0x38228000, 0x3822C000, 0x38230000, 0x38234000, 0x38238000, 0x3823C000,
  0x38240000, 0x38244000, 0x38248000, 0x3824C000, 0x38250000, 0x38254000, 0x38258000, 0x3825C000, 0x38260000, 0x38264000, 0x38268000, 0x3826C000, 0x38270000, 0x38274000, 0x38278000, 0x3827C000,
  0x38280000, 0x38284000, 0x38288000, 0x3828C000, 0x38290000, 0x38294000, 0x38298000, 0x3829C000, 0x382A0000, 0x382A4000, 0x382A8000, 0x382AC000, 0x382B0000, 0x382B4000, 0x382B8000, 0x382BC000,
  0x382C0000, 0x382C4000, 0x382C8000, 0x382CC000, 0x382D0000, 0x382D4000, 0x382D8000, 0x382DC000, 0x382E0000, 0x382E4000, 0x382E8000, 0x382EC000, 0x382F0000, 0x382F4000, 0x382F8000, 0x382FC000,
  0x38300000, 0x38304000, 0x38308000, 0x3830C000, 0x38310000, 0x38314000, 0x38318000, 0x3831C000, 0x38320000, 0x38324000, 0x38328000, 0x3832C000, 0x38330000, 0x38334000, 0x38338000, 0x3833C000,
  0x38340000, 0x38344000, 0x38348000, 0x3834C000, 0x38350000, 0x38354000, 0x38358000, 0x3835C000, 0x38360000, 0x38364000, 0x38368000, 0x3836C000, 0x38370000, 0x38374000, 0x38378000, 0x3837C000,
  0x38380000, 0x38384000, 0x38388000, 0x3838C000, 0x38390000, 0x38394000, 0x38398000, 0x3839C000, 0x383A0000, 0x383A4000, 0x383A8000, 0x383AC000, 0x383B0000, 0x383B4000, 0x383B8000, 0x383BC000,
  0x383C0000, 0x383C4000, 0x383C8000, 0x383CC000, 0x383D0000, 0x383D4000, 0x383D8000, 0x383DC000, 0x383E0000, 0x383E4000, 0x383E8000, 0x383EC000, 0x383F0000, 0x383F4000, 0x383F8000, 0x383FC000,
  0x38400000, 0x38404000, 0x38408000, 0x3840C000, 0x38410000, 0x38414000, 0x38418000, 0x3841C000, 0x38420000, 0x38424000, 0x38428000, 0x3842C000, 0x38430000, 0x38434000, 0x38438000, 0x3843C000,
  0x38440000, 0x38444000, 0x38448000, 0x3844C000, 0x38450000, 0x38454000, 0x38458000, 0x3845C000, 0x38460000, 0x38464000, 0x38468000, 0x3846C000, 0x38470000, 0x38474000, 0x38478000, 0x3847C000,
  0x38480000, 0x38484000, 0x38488000, 0x3848C000, 0x38490000, 0x38494000, 0x38498000, 0x3849C000, 0x384A0000, 0x384A4000, 0x384A8000, 0x384AC000, 0x384B0000, 0x384B4000, 0x384B8000, 0x384BC000,
  0x384C0000, 0x384C4000, 0x384C8000, 0x384CC000, 0x384D0000, 0x384D4000, 0x384D8000, 0x384DC000, 0x384E0000, 0x384E4000, 0x384E8000, 0x384EC000, 0x384F0000, 0x384F4000, 0x384F8000, 0x384FC000,
  0x38500000, 0x38504000, 0x38508000, 0x3850C000, 0x38510000, 0x38514000, 0x38518000, 0x3851C000, 0x38520000, 0x38524000, 0x38528000, 0x3852C000, 0x38530000, 0x38534000, 0x38538000, 0x3853C000,
  0x38540000, 0x38544000, 0x38548000, 0x3854C000, 0x38550000, 0x38554000, 0x38558000, 0x3855C000, 0x38560000, 0x38564000, 0x38568000, 0x3856C000, 0x38570000, 0x38574000, 0x38578000, 0x3857C000,
  0x38580000, 0x38584000, 0x38588000, 0x3858C000, 0x38590000, 0x38594000, 0x38598000, 0x3859C000, 0x385A0000, 0x385A4000, 0x385A8000, 0x385AC000, 0x385B0000, 0x385B4000, 0x385B8000, 0x385BC000,
  0x385C0000, 0x385C4000, 0x385C8000, 0x385CC000, 0x385D0000, 0x385D4000, 0x385D8000, 0x385DC000, 0x385E0000, 0x385E4000, 0x385E8000, 0x385EC000, 0x385F0000, 0x385F4000, 0x385F8000, 0x385FC000,
  0x38600000, 0x38604000, 0x38608000, 0x3860C000, 0x38610000, 0x38614000, 0x38618000, 0x3861C000, 0x38620000, 0x38624000, 0x38628000, 0x3862C000, 0x38630000, 0x38634000, 0x38638000, 0x3863C000,
  0x38640000, 0x38644000, 0x38648000, 0x3864C000, 0x38650000, 0x38654000, 0x38658000, 0x3865C000, 0x38660000, 0x38664000, 0x38668000, 0x3866C000, 0x38670000, 0x38674000, 0x38678000, 0x3867C000,
  0x38680000, 0x38684000, 0x38688000, 0x3868C000, 0x38690000, 0x38694000, 0x38698000, 0x3869C000, 0x386A0000, 0x386A4000, 0x386A8000, 0x386AC000, 0x386B0000, 0x386B4000, 0x386B8000, 0x386BC000,
  0x386C0000, 0x386C4000, 0x386C8000, 0x386CC000, 0x386D0000, 0x386D4000, 0x386D8000, 0x386DC000, 0x386E0000, 0x386E4000, 0x386E8000, 0x386EC000, 0x386F0000, 0x386F4000, 0x386F8000, 0x386FC000,
  0x38700000, 0x38704000, 0x38708000, 0x3870C000, 0x38710000, 0x38714000, 0x38718000, 0x3871C000, 0x38720000, 0x38724000, 0x38728000, 0x3872C000, 0x38730000, 0x38734000, 0x38738000, 0x3873C000,
  0x38740000, 0x38744000, 0x38748000, 0x3874C000, 0x38750000, 0x38754000, 0x38758000, 0x3875C000, 0x38760000, 0x38764000, 0x38768000, 0x3876C000, 0x38770000, 0x38774000, 0x38778000, 0x3877C000,
  0x38780000, 0x38784000, 0x38788000, 0x3878C000, 0x38790000, 0x38794000, 0x38798000, 0x3879C000, 0x387A0000, 0x387A4000, 0x387A8000, 0x387AC000, 0x387B0000, 0x387B4000, 0x387B8000, 0x387BC000,
  0x387C0000, 0x387C4000, 0x387C8000, 0x387CC000, 0x387D0000, 0x387D4000, 0x387D8000, 0x387DC000, 0x387E0000, 0x387E4000, 0x387E8000, 0x387EC000, 0x387F0000, 0x387F4000, 0x387F8000, 0x387FC000,
  0x38000000, 0x38002000, 0x38004000, 0x38006000, 0x38008000, 0x3800A000, 0x3800C000, 0x3800E000, 0x38010000, 0x38012000, 0x38014000, 0x38016000, 0x38018000, 0x3801A000, 0x3801C000, 0x3801E000,
  0x38020000, 0x38022000, 0x38024000, 0x38026000, 0x38028000, 0x3802A000, 0x3802C000, 0x3802E000, 0x38030000, 0x38032000, 0x38034000, 0x38036000, 0x38038000, 0x3803A000, 0x3803C000, 0x3803E000,
  0x38040000, 0x38042000, 0x38044000, 0x38046000, 0x38048000, 0x3804A000, 0x3804C000, 0x3804E000, 0x38050000, 0x38052000, 0x38054000, 0x38056000, 0x38058000, 0x3805A000, 0x3805C000, 0x3805E000,
  0x38060000, 0x38062000, 0x38064000, 0x38066000, 0x38068000, 0x3806A000, 0x3806C000, 0x3806E000, 0x38070000, 0x38072000, 0x38074000, 0x38076000, 0x38078000, 0x3807A000, 0x3807C000, 0x3807E000,
  0x38080000, 0x38082000, 0x38084000, 0x38086000, 0x38088000, 0x3808A000, 0x3808C000, 0x3808E000, 0x38090000, 0x38092000, 0x38094000, 0x38096000, 0x38098000, 0x3809A000, 0x3809C000, 0x3809E000,
  0x380A0000, 0x380A2000, 0x380A4000, 0x380A6000, 0x380A8000, 0x380AA000, 0x380AC000, 0x380AE000, 0x380B0000, 0x380B2000, 0x380B4000, 0x380B6000, 0x380B8000, 0x380BA000, 0x380BC000, 0x380BE000,
  0x380C0000, 0x380C2000, 0x380C4000, 0x380C6000, 0x380C8000, 0x380CA000, 0x380CC000, 0x380CE000, 0x380D0000, 0x380D2000, 0x380D4000, 0x380D6000, 0x380D8000, 0x380DA000, 0x380DC000, 0x380DE000,
  0x380E0000, 0x380E2000, 0x380E4000, 0x380E6000, 0x380E8000, 0x380EA000, 0x380EC000, 0x380EE000, 0x380F0000, 0x380F2000, 0x380F4000, 0x380F6000, 0x380F8000, 0x380FA000, 0x380FC000, 0x380FE000,
  0x38100000, 0x38102000, 0x38104000, 0x38106000, 0x38108000, 0x3810A000, 0x3810C000, 0x3810E000, 0x38110000, 0x38112000, 0x38114000, 0x38116000, 0x38118000, 0x3811A000, 0x3811C000, 0x3811E000,
  0x38120000, 0x38122000, 0x38124000, 0x38126000, 0x38128000, 0x3812A000, 0x3812C000, 0x3812E000, 0x38130000, 0x38132000, 0x38134000, 0x38136000, 0x38138000, 0x3813A000, 0x3813C000, 0x3813E000,
  0x38140000, 0x38142000, 0x38144000, 0x38146000, 0x38148000, 0x3814A000, 0x3814C000, 0x3814E000, 0x38150000, 0x38152000, 0x38154000, 0x38156000, 0x38158000, 0x3815A000, 0x3815C000, 0x3815E000,
  0x38160000, 0x38162000, 0x38164000, 0x38166000, 0x38168000, 0x3816A000, 0x3816C000, 0x3816E000, 0x38170000, 0x38172000, 0x38174000, 0x38176000, 0x38178000, 0x3817A000, 0x3817C000, 0x3817E000,
  0x38180000, 0x38182000, 0x38184000, 0x38186000, 0x38188000, 0x3818A000, 0x3818C000, 0x3818E000, 0x38190000, 0x38192000, 0x38194000, 0x38196000, 0x38198000, 0x3819A000, 0x3819C000, 0x3819E000,
  0x381A0000, 0x381A2000, 0x381A4000, 0x381A6000, 0x381A8000, 0x381AA000, 0x381AC000, 0x381AE000, 0x381B0000, 0x381B2000, 0x381B4000, 0x381B6000, 0x381B8000, 0x381BA000, 0x381BC000, 0x381BE000,
  0x381C0000, 0x381C2000, 0x381C4000, 0x381C6000, 0x381C8000, 0x381CA000, 0x381CC000, 0x381CE000, 0x381D0000, 0x381D2000, 0x381D4000, 0x381D6000, 0x381D8000, 0x381DA000, 0x381DC000, 0x381DE000,
  0x381E0000, 0x381E2000, 0x381E4000, 0x381E6000, 0x381E8000, 0x381EA000, 0x381EC000, 0x381EE000, 0x381F0000, 0x381F2000, 0x381F4000, 0x381F6000, 0x381F8000, 0x381FA000, 0x381FC000, 0x381FE000,
  0x38200000, 0x38202000, 0x38204000, 0x38206000, 0x38208000, 0x3820A000, 0x3820C000, 0x3820E000, 0x38210000, 0x38212000, 0x38214000, 0x38216000, 0x38218000, 0x3821A000, 0x3821C000, 0x3821E000,
  0x38220000, 0x38222000, 0x38224000, 0x38226000, 0x38228000, 0x3822A000, 0x3822C000, 0x3822E000, 0x38230000, 0x38232000, 0x38234000, 0x38236000, 0x38238000, 0x3823A000, 0x3823C000, 0x3823E000,
  0x38240000, 0x38242000, 0x38244000, 0x38246000, 0x38248000, 0x3824A000, 0x3824C000, 0x3824E000, 0x38250000, 0x38252000, 0x38254000, 0x38256000, 0x38258000, 0x3825A000, 0x3825C000, 0x3825E000,
  0x38260000, 0x38262000, 0x38264000, 0x38266000, 0x38268000, 0x3826A000, 0x3826C000, 0x3826E000, 0x38270000, 0x38272000, 0x38274000, 0x38276000, 0x38278000, 0x3827A000, 0x3827C000, 0x3827E000,
  0x38280000, 0x38282000, 0x38284000, 0x38286000, 0x38288000, 0x3828A000, 0x3828C000, 0x3828E000, 0x38290000, 0x38292000, 0x38294000, 0x38296000, 0x38298000, 0x3829A000, 0x3829C000, 0x3829E000,
  0x382A0000, 0x382A2000, 0x382A4000, 0x382A6000, 0x382A8000, 0x382AA000, 0x382AC000, 0x382AE000, 0x382B0000, 0x382B2000, 0x382B4000, 0x382B6000, 0x382B8000, 0x382BA000, 0x382BC000, 0x382BE000,
  0x382C0000, 0x382C2000, 0x382C4000, 0x382C6000, 0x382C8000, 0x382CA000, 0x382CC000, 0x382CE000, 0x382D0000, 0x382D2000, 0x382D4000, 0x382D6000, 0x382D8000, 0x382DA000, 0x382DC000, 0x382DE000,
  0x382E0000, 0x382E2000, 0x382E4000, 0x382E6000, 0x382E8000, 0x382EA000, 0x382EC000, 0x382EE000, 0x382F0000, 0x382F2000, 0x382F4000, 0x382F6000, 0x382F8000, 0x382FA000, 0x382FC000, 0x382FE000,
  0x38300000, 0x38302000, 0x38304000, 0x38306000, 0x38308000, 0x3830A000, 0x3830C000, 0x3830E000, 0x38310000, 0x38312000, 0x38314000, 0x38316000, 0x38318000, 0x3831A000, 0x3831C000, 0x3831E000,
  0x38320000, 0x38322000, 0x38324000, 0x38326000, 0x38328000, 0x3832A000, 0x3832C000, 0x3832E000, 0x38330000, 0x38332000, 0x38334000, 0x38336000, 0x38338000, 0x3833A000, 0x3833C000, 0x3833E000,
  0x38340000, 0x38342000, 0x38344000, 0x38346000, 0x38348000, 0x3834A000, 0x3834C000, 0x3834E000, 0x38350000, 0x38352000, 0x38354000, 0x38356000, 0x38358000, 0x3835A000, 0x3835C000, 0x3835E000,
  0x38360000, 0x38362000, 0x38364000, 0x38366000, 0x38368000, 0x3836A000, 0x3836C000, 0x3836E000, 0x38370000, 0x38372000, 0x38374000, 0x38376000, 0x38378000, 0x3837A000, 0x3837C000, 0x3837E000,
  0x38380000, 0x38382000, 0x38384000, 0x38386000, 0x38388000, 0x3838A000, 0x3838C000, 0x3838E000, 0x38390000, 0x38392000, 0x38394000, 0x38396000, 0x38398000, 0x3839A000, 0x3839C000, 0x3839E000,
  0x383A0000, 0x383A2000, 0x383A4000, 0x383A6000, 0x383A8000, 0x383AA000, 0x383AC000, 0x383AE000, 0x383B0000, 0x383B2000, 0x383B4000, 0x383B6000, 0x383B8000, 0x383BA000, 0x383BC000, 0x383BE000,
  0x383C0000, 0x383C2000, 0x383C4000, 0x383C6000, 0x383C8000, 0x383CA000, 0x383CC000, 0x383CE000, 0x383D0000, 0x383D2000, 0x383D4000, 0x383D6000, 0x383D8000, 0x383DA000, 0x383DC000, 0x383DE000,
  0x383E0000, 0x383E2000, 0x383E4000, 0x383E6000, 0x383E8000, 0x383EA000, 0x383EC000, 0x383EE000, 0x383F0000, 0x383F2000, 0x383F4000, 0x383F6000, 0x383F8000, 0x383FA000, 0x383FC000, 0x383FE000,
  0x38400000, 0x38402000, 0x38404000, 0x38406000, 0x38408000, 0x3840A000, 0x3840C000, 0x3840E000, 0x38410000, 0x38412000, 0x38414000, 0x38416000, 0x38418000, 0x3841A000, 0x3841C000, 0x3841E000,
  0x38420000, 0x38422000, 0x38424000, 0x38426000, 0x38428000, 0x3842A000, 0x3842C000, 0x3842E000, 0x38430000, 0x38432000, 0x38434000, 0x38436000, 0x38438000, 0x3843A000, 0x3843C000, 0x3843E000,
  0x38440000, 0x38442000, 0x38444000, 0x38446000, 0x38448000, 0x3844A000, 0x3844C000, 0x3844E000, 0x38450000, 0x38452000, 0x38454000, 0x38456000, 0x38458000, 0x3845A000, 0x3845C000, 0x3845E000,
  0x38460000, 0x38462000, 0x38464000, 0x38466000, 0x38468000, 0x3846A000, 0x3846C000, 0x3846E000, 0x38470000, 0x38472000, 0x38474000, 0x38476000, 0x38478000, 0x3847A000, 0x3847C000, 0x3847E000,
  0x38480000, 0x38482000, 0x38484000, 0x38486000, 0x38488000, 0x3848A000, 0x3848C000, 0x3848E000, 0x38490000, 0x38492000, 0x38494000, 0x38496000, 0x38498000, 0x3849A000, 0x3849C000, 0x3849E000,
  0x384A0000, 0x384A2000, 0x384A4000, 0x384A6000, 0x384A8000, 0x384AA000, 0x384AC000, 0x384AE000, 0x384B0000, 0x384B2000, 0x384B4000, 0x384B6000, 0x384B8000, 0x384BA000, 0x384BC000, 0x384BE000,
  0x384C0000, 0x384C2000, 0x384C4000, 0x384C6000, 0x384C8000, 0x384CA000, 0x384CC000, 0x384CE000, 0x384D0000, 0x384D2000, 0x384D4000, 0x384D6000, 0x384D8000, 0x384DA000, 0x384DC000, 0x384DE000,
  0x384E0000, 0x384E2000, 0x384E4000, 0x384E6000, 0x384E8000, 0x384EA000, 0x384EC000, 0x384EE000, 0x384F0000, 0x384F2000, 0x384F4000, 0x384F6000, 0x384F8000, 0x384FA000, 0x384FC000, 0x384FE000,
  0x38500000, 0x38502000, 0x38504000, 0x38506000, 0x38508000, 0x3850A000, 0x3850C000, 0x3850E000, 0x38510000, 0x38512000, 0x38514000, 0x38516000, 0x38518000, 0x3851A000, 0x3851C000, 0x3851E000,
  0x38520000, 0x38522000, 0x38524000, 0x38526000, 0x38528000, 0x3852A000, 0x3852C000, 0x3852E000, 0x38530000, 0x38532000, 0x38534000, 0x38536000, 0x38538000, 0x3853A000, 0x3853C000, 0x3853E000,
  0x38540000, 0x38542000, 0x38544000, 0x38546000, 0x38548000, 0x3854A000, 0x3854C000, 0x3854E000, 0x38550000, 0x38552000, 0x38554000, 0x38556000, 0x38558000, 0x3855A000, 0x3855C000, 0x3855E000,
  0x38560000, 0x38562000, 0x38564000, 0x38566000, 0x38568000, 0x3856A000, 0x3856C000, 0x3856E000, 0x38570000, 0x38572000, 0x38574000, 0x38576000, 0x38578000, 0x3857A000, 0x3857C000, 0x3857E000,
  0x38580000, 0x38582000, 0x38584000, 0x38586000, 0x38588000, 0x3858A000, 0x3858C000, 0x3858E000, 0x38590000, 0x38592000, 0x38594000, 0x38596000, 0x38598000, 0x3859A000, 0x3859C000, 0x3859E000,
  0x385A0000, 0x385A2000, 0x385A4000, 0x385A6000, 0x385A8000, 0x385AA000, 0x385AC000, 0x385AE000, 0x385B0000, 0x385B2000, 0x385B4000, 0x385B6000, 0x385B8000, 0x385BA000, 0x385BC000, 0x385BE000,
  0x385C0000, 0x385C2000, 0x385C4000, 0x385C6000, 0x385C8000, 0x385CA000, 0x385CC000, 0x385CE000, 0x385D0000, 0x385D2000, 0x385D4000, 0x385D6000, 0x385D8000, 0x385DA000, 0x385DC000, 0x385DE000,
  0x385E0000, 0x385E2000, 0x385E4000, 0x385E6000, 0x385E8000, 0x385EA000, 0x385EC000, 0x385EE000, 0x385F0000, 0x385F2000, 0x385F4000, 0x385F6000, 0x385F8000, 0x385FA000, 0x385FC000, 0x385FE000,
  0x38600000, 0x38602000, 0x38604000, 0x38606000, 0x38608000, 0x3860A000, 0x3860C000, 0x3860E000, 0x38610000, 0x38612000, 0x38614000, 0x38616000, 0x38618000, 0x3861A000, 0x3861C000, 0x3861E000,
  0x38620000, 0x38622000, 0x38624000, 0x38626000, 0x38628000, 0x3862A000, 0x3862C000, 0x3862E000, 0x38630000, 0x38632000, 0x38634000, 0x38636000, 0x38638000, 0x3863A000, 0x3863C000, 0x3863E000,
  0x38640000, 0x38642000, 0x38644000, 0x38646000, 0x38648000, 0x3864A000, 0x3864C000, 0x3864E000, 0x38650000, 0x38652000, 0x38654000, 0x38656000, 0x38658000, 0x3865A000, 0x3865C000, 0x3865E000,
  0x38660000, 0x38662000, 0x38664000, 0x38666000, 0x38668000, 0x3866A000, 0x3866C000, 0x3866E000, 0x38670000, 0x38672000, 0x38674000, 0x38676000, 0x38678000, 0x3867A000, 0x3867C000, 0x3867E000,
  0x38680000, 0x38682000, 0x38684000, 0x38686000, 0x38688000, 0x3868A000, 0x3868C000, 0x3868E000, 0x38690000, 0x38692000, 0x38694000, 0x38696000, 0x38698000, 0x3869A000, 0x3869C000, 0x3869E000,
  0x386A0000, 0x386A2000, 0x386A4000, 0x386A6000, 0x386A8000, 0x386AA000, 0x386AC000, 0x386AE000, 0x386B0000, 0x386B2000, 0x386B4000, 0x386B6000, 0x386B8000, 0x386BA000, 0x386BC000, 0x386BE000,
  0x386C0000, 0x386C2000, 0x386C4000, 0x386C6000, 0x386C8000, 0x386CA000, 0x386CC000, 0x386CE000, 0x386D0000, 0x386D2000, 0x386D4000, 0x386D6000, 0x386D8000, 0x386DA000, 0x386DC000, 0x386DE000,
  0x386E0000, 0x386E2000, 0x386E4000, 0x386E6000, 0x386E8000, 0x386EA000, 0x386EC000, 0x386EE000, 0x386F0000, 0x386F2000, 0x386F4000, 0x386F6000, 0x386F8000, 0x386FA000, 0x386FC000, 0x386FE000,
  0x38700000, 0x38702000, 0x38704000, 0x38706000, 0x38708000, 0x3870A000, 0x3870C000, 0x3870E000, 0x38710000, 0x38712000, 0x38714000, 0x38716000, 0x38718000, 0x3871A000, 0x3871C000, 0x3871E000,
  0x38720000, 0x38722000, 0x38724000, 0x38726000, 0x38728000, 0x3872A000, 0x3872C000, 0x3872E000, 0x38730000, 0x38732000, 0x38734000, 0x38736000, 0x38738000, 0x3873A000, 0x3873C000, 0x3873E000,
  0x38740000, 0x38742000, 0x38744000, 0x38746000, 0x38748000, 0x3874A000, 0x3874C000, 0x3874E000, 0x38750000, 0x38752000, 0x38754000, 0x38756000, 0x38758000, 0x3875A000, 0x3875C000, 0x3875E000,
  0x38760000, 0x38762000, 0x38764000, 0x38766000, 0x38768000, 0x3876A000, 0x3876C000, 0x3876E000, 0x38770000, 0x38772000, 0x38774000, 0x38776000, 0x38778000, 0x3877A000, 0x3877C000, 0x3877E000,
  0x38780000, 0x38782000, 0x38784000, 0x38786000, 0x38788000, 0x3878A000, 0x3878C000, 0x3878E000, 0x38790000, 0x38792000, 0x38794000, 0x38796000, 0x38798000, 0x3879A000, 0x3879C000, 0x3879E000,
  0x387A0000, 0x387A2000, 0x387A4000, 0x387A6000, 0x387A8000, 0x387AA000, 0x387AC000, 0x387AE000, 0x387B0000, 0x387B2000, 0x387B4000, 0x387B6000, 0x387B8000, 0x387BA000, 0x387BC000, 0x387BE000,
  0x387C0000, 0x387C2000, 0x387C4000, 0x387C6000, 0x387C8000, 0x387CA000, 0x387CC000, 0x387CE000, 0x387D0000, 0x387D2000, 0x387D4000, 0x387D6000, 0x387D8000, 0x387DA000, 0x387DC000, 0x387DE000,
  0x387E0000, 0x387E2000, 0x387E4000, 0x387E6000, 0x387E8000, 0x387EA000, 0x387EC000, 0x387EE000, 0x387F0000, 0x387F2000, 0x387F4000, 0x387F6000, 0x387F8000, 0x387FA000, 0x387FC000, 0x387FE000 };
__constant static const uint32_t exponent_table[64] = {
  0x00000000, 0x00800000, 0x01000000, 0x01800000, 0x02000000, 0x02800000, 0x03000000, 0x03800000, 0x04000000, 0x04800000, 0x05000000, 0x05800000, 0x06000000, 0x06800000, 0x07000000, 0x07800000,
  0x08000000, 0x08800000, 0x09000000, 0x09800000, 0x0A000000, 0x0A800000, 0x0B000000, 0x0B800000, 0x0C000000, 0x0C800000, 0x0D000000, 0x0D800000, 0x0E000000, 0x0E800000, 0x0F000000, 0x47800000,
  0x80000000, 0x80800000, 0x81000000, 0x81800000, 0x82000000, 0x82800000, 0x83000000, 0x83800000, 0x84000000, 0x84800000, 0x85000000, 0x85800000, 0x86000000, 0x86800000, 0x87000000, 0x87800000,
  0x88000000, 0x88800000, 0x89000000, 0x89800000, 0x8A000000, 0x8A800000, 0x8B000000, 0x8B800000, 0x8C000000, 0x8C800000, 0x8D000000, 0x8D800000, 0x8E000000, 0x8E800000, 0x8F000000, 0xC7800000 };
__constant static const unsigned short offset_table[64] = {
  0, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
  0, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024 };

SCALAR_FUN_ATTR uint16_t float2halfbits(float value) {
  union { float x; uint32_t y; } u;
  u.x = value;
  uint32_t bits = u.y;

  uint16_t hbits = base_table[bits>>23] + (uint16_t)((bits&0x7FFFFF)>>shift_table[bits>>23]);;

  return hbits;
}

SCALAR_FUN_ATTR float halfbits2float(uint16_t value) {
  uint32_t bits = mantissa_table[offset_table[value>>10]+(value&0x3FF)] + exponent_table[value>>10];

  union { uint32_t x; float y; } u;
  u.x = bits;
  return u.y;
}

SCALAR_FUN_ATTR uint16_t halfbitsnextafter(uint16_t from, uint16_t to) {
  int fabs = from & 0x7FFF, tabs = to & 0x7FFF;
  if(fabs > 0x7C00 || tabs > 0x7C00) {
    return ((from&0x7FFF)>0x7C00) ? (from|0x200) : (to|0x200);
  }
  if(from == to || !(fabs|tabs)) {
    return to;
  }
  if(!fabs) {
    return (to&0x8000)+1;
  }
  unsigned int out =
    from +
    (((from>>15)^(unsigned int)((from^(0x8000|(0x8000-(from>>15))))<(to^(0x8000|(0x8000-(to>>15))))))<<1)
    - 1;
  return out;
}

// End of half.h.
// Start of timing.h.

// The function get_wall_time() returns the wall time in microseconds
// (with an unspecified offset).

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

static int64_t get_wall_time_ns(void) {
  return get_wall_time() * 1000;
}

#else
// Assuming POSIX

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time_ns(void) {
  struct timespec time;
  assert(clock_gettime(CLOCK_MONOTONIC, &time) == 0);
  return time.tv_sec * 1000000000 + time.tv_nsec;
}

static int64_t get_wall_time(void) {
  return get_wall_time_ns() / 1000;
}


#endif

// End of timing.h.
// Start of lock.h.

// A very simple cross-platform implementation of locks.  Uses
// pthreads on Unix and some Windows thing there.  Futhark's
// host-level code is not multithreaded, but user code may be, so we
// need some mechanism for ensuring atomic access to API functions.
// This is that mechanism.  It is not exposed to user code at all, so
// we do not have to worry about name collisions.

#ifdef _WIN32

typedef HANDLE lock_t;

static void create_lock(lock_t *lock) {
  *lock = CreateMutex(NULL,  // Default security attributes.
                      FALSE, // Initially unlocked.
                      NULL); // Unnamed.
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
// Assuming POSIX

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  // Nothing to do for pthreads.
  (void)lock;
}

#endif

// End of lock.h.
// Start of free_list.h.

typedef uintptr_t fl_mem;

// An entry in the free list.  May be invalid, to avoid having to
// deallocate entries as soon as they are removed.  There is also a
// tag, to help with memory reuse.
struct free_list_entry {
  size_t size;
  fl_mem mem;
  const char *tag;
  unsigned char valid;
};

struct free_list {
  struct free_list_entry *entries; // Pointer to entries.
  int capacity;                    // Number of entries.
  int used;                        // Number of valid entries.
  lock_t lock;                     // Thread safety.
};

static void free_list_init(struct free_list *l) {
  l->capacity = 30; // Picked arbitrarily.
  l->used = 0;
  l->entries = (struct free_list_entry*) malloc(sizeof(struct free_list_entry) * l->capacity);
  for (int i = 0; i < l->capacity; i++) {
    l->entries[i].valid = 0;
  }
  create_lock(&l->lock);
}

// Remove invalid entries from the free list.
static void free_list_pack(struct free_list *l) {
  lock_lock(&l->lock);
  int p = 0;
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[p] = l->entries[i];
      if (i > p) {
        l->entries[i].valid = 0;
      }
      p++;
    }
  }

  // Now p is the number of used elements.  We don't want it to go
  // less than the default capacity (although in practice it's OK as
  // long as it doesn't become 1).
  if (p < 30) {
    p = 30;
  }
  l->entries = realloc(l->entries, p * sizeof(struct free_list_entry));
  l->capacity = p;
  lock_unlock(&l->lock);
}

static void free_list_destroy(struct free_list *l) {
  assert(l->used == 0);
  free(l->entries);
  free_lock(&l->lock);
}

// Not part of the interface, so no locking.
static int free_list_find_invalid(struct free_list *l) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (!l->entries[i].valid) {
      break;
    }
  }
  return i;
}

static void free_list_insert(struct free_list *l, size_t size, fl_mem mem, const char *tag) {
  lock_lock(&l->lock);
  int i = free_list_find_invalid(l);

  if (i == l->capacity) {
    // List is full; so we have to grow it.
    int new_capacity = l->capacity * 2 * sizeof(struct free_list_entry);
    l->entries = realloc(l->entries, new_capacity);
    for (int j = 0; j < l->capacity; j++) {
      l->entries[j+l->capacity].valid = 0;
    }
    l->capacity *= 2;
  }

  // Now 'i' points to the first invalid entry.
  l->entries[i].valid = 1;
  l->entries[i].size = size;
  l->entries[i].mem = mem;
  l->entries[i].tag = tag;

  l->used++;
  lock_unlock(&l->lock);
}

// Determine whether this entry in the free list is acceptable for
// satisfying the request.  Not public, so no locking.
static bool free_list_acceptable(size_t size, const char* tag, struct free_list_entry *entry) {
  // We check not just the hard requirement (is the entry acceptable
  // and big enough?) but also put a cap on how much wasted space
  // (internal fragmentation) we allow.  This is necessarily a
  // heuristic, and a crude one.

  if (!entry->valid) {
    return false;
  }

  if (size > entry->size) {
    return false;
  }

  // We know the block fits.  Now the question is whether it is too
  // big.  Our policy is as follows:
  //
  // 1) We don't care about wasted space below 4096 bytes (to avoid
  // churn in tiny allocations).
  //
  // 2) If the tag matches, we allow _any_ amount of wasted space.
  //
  // 3) Otherwise we allow up to 50% wasted space.

  if (entry->size < 4096) {
    return true;
  }

  if (entry->tag == tag) {
    return true;
  }

  if (entry->size < size * 2) {
    return true;
  }

  return false;
}

// Find and remove a memory block of the indicated tag, or if that
// does not exist, another memory block with exactly the desired size.
// Returns 0 on success.
static int free_list_find(struct free_list *l, size_t size, const char *tag,
                          size_t *size_out, fl_mem *mem_out) {
  lock_lock(&l->lock);
  int size_match = -1;
  int i;
  int ret = 1;
  for (i = 0; i < l->capacity; i++) {
    if (free_list_acceptable(size, tag, &l->entries[i]) &&
        (size_match < 0 || l->entries[i].size < l->entries[size_match].size)) {
      // If this entry is valid, has sufficient size, and is smaller than the
      // best entry found so far, use this entry.
      size_match = i;
    }
  }

  if (size_match >= 0) {
    l->entries[size_match].valid = 0;
    *size_out = l->entries[size_match].size;
    *mem_out = l->entries[size_match].mem;
    l->used--;
    ret = 0;
  }
  lock_unlock(&l->lock);
  return ret;
}

// Remove the first block in the free list.  Returns 0 if a block was
// removed, and nonzero if the free list was already empty.
static int free_list_first(struct free_list *l, fl_mem *mem_out) {
  lock_lock(&l->lock);
  int ret = 1;
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[i].valid = 0;
      *mem_out = l->entries[i].mem;
      l->used--;
      ret = 0;
      break;
    }
  }
  lock_unlock(&l->lock);
  return ret;
}

// End of free_list.h.
// Start of event_list.h

typedef int (*event_report_fn)(struct str_builder*, void*);

// A collection of key-value associations. Used to associate extra data with
// events.
struct kvs {
  // A buffer that contains all value data. Must be freed when the struct kvs is
  // no longer used.
  char *buf;

  // Size of buf in bytes.
  size_t buf_size;

  // Number of bytes used in buf.
  size_t buf_used;

  // Number of associations stored.
  size_t n;

  // Capacity of vals.
  size_t vals_capacity;

  // An array of keys.
  const char* *keys;

  // Indexes into 'buf' that contains the values as zero-terminated strings.
  size_t *vals;
};

static const size_t KVS_INIT_BUF_SIZE = 128;
static const size_t KVS_INIT_NUMKEYS = 8;

void kvs_init(struct kvs* kvs) {
  kvs->buf = malloc(KVS_INIT_BUF_SIZE);
  kvs->buf_size = KVS_INIT_BUF_SIZE;
  kvs->buf_used = 0;
  kvs->vals_capacity = KVS_INIT_NUMKEYS;
  kvs->keys = calloc(kvs->vals_capacity, sizeof(const char*));
  kvs->vals = calloc(kvs->vals_capacity, sizeof(size_t));
  kvs->n = 0;
}

struct kvs* kvs_new(void) {
  struct kvs *kvs = malloc(sizeof(struct kvs));
  kvs_init(kvs);
  return kvs;
}

void kvs_printf(struct kvs* kvs, const char* key, const char* fmt, ...) {
  va_list vl;
  va_start(vl, fmt);

  size_t needed = 1 + (size_t)vsnprintf(NULL, 0, fmt, vl);

  while (kvs->buf_used+needed > kvs->buf_size) {
    kvs->buf_size *= 2;
    kvs->buf = realloc(kvs->buf, kvs->buf_size * sizeof(const char*));
  }

  if (kvs->n == kvs->vals_capacity) {
    kvs->vals_capacity *= 2;
    kvs->vals = realloc(kvs->vals, kvs->vals_capacity * sizeof(size_t));
    kvs->keys = realloc(kvs->keys, kvs->vals_capacity * sizeof(char*));
  }

  kvs->keys[kvs->n] = key;
  kvs->vals[kvs->n] = kvs->buf_used;
  kvs->buf_used += needed;

  va_start(vl, fmt); // Must re-init.
  vsnprintf(&kvs->buf[kvs->vals[kvs->n]], needed, fmt, vl);

  kvs->n++;
}

void kvs_free(struct kvs* kvs) {
  free(kvs->vals);
  free(kvs->keys);
  free(kvs->buf);
}

// Assumes all of the values are valid JSON objects.
void kvs_json(const struct kvs* kvs, struct str_builder *sb) {
  str_builder_char(sb, '{');
  for (size_t i = 0; i < kvs->n; i++) {
    if (i != 0) {
      str_builder_str(sb, ",");
    }
    str_builder_json_str(sb, kvs->keys[i]);
    str_builder_str(sb, ":");
    str_builder_str(sb, &kvs->buf[kvs->vals[i]]);
  }
  str_builder_char(sb, '}');
}

void kvs_log(const struct kvs* kvs, const char* prefix, FILE* f) {
  for (size_t i = 0; i < kvs->n; i++) {
    fprintf(f, "%s%s: %s\n",
            prefix,
            kvs->keys[i],
            &kvs->buf[kvs->vals[i]]);
  }
}

struct event {
  void* data;
  event_report_fn f;
  const char* name;
  const char *provenance;
  // Key-value information that is also to be printed.
  struct kvs *kvs;
};

struct event_list {
  struct event *events;
  int num_events;
  int capacity;
};

static void event_list_init(struct event_list *l) {
  l->capacity = 100;
  l->num_events = 0;
  l->events = calloc(l->capacity, sizeof(struct event));
}

static void event_list_free(struct event_list *l) {
  free(l->events);
}

static void add_event_to_list(struct event_list *l,
                              const char* name,
                              const char* provenance,
                              struct kvs *kvs,
                              void* data,
                              event_report_fn f) {
  if (l->num_events == l->capacity) {
    l->capacity *= 2;
    l->events = realloc(l->events, l->capacity * sizeof(struct event));
  }
  l->events[l->num_events].name = name;
  l->events[l->num_events].provenance =
    provenance ? provenance : "unknown";
  l->events[l->num_events].kvs = kvs;
  l->events[l->num_events].data = data;
  l->events[l->num_events].f = f;
  l->num_events++;
}

static int report_events_in_list(struct event_list *l,
                                 struct str_builder* sb) {
  int ret = 0;
  for (int i = 0; i < l->num_events; i++) {
    if (i != 0) {
      str_builder_str(sb, ",");
    }
    str_builder_str(sb, "{\"name\":");
    str_builder_json_str(sb, l->events[i].name);
    str_builder_str(sb, ",\"provenance\":");
    str_builder_json_str(sb, l->events[i].provenance);
    if (l->events[i].f(sb, l->events[i].data) != 0) {
      ret = 1;
      break;
    }

    str_builder_str(sb, ",\"details\":");
    if (l->events[i].kvs) {
      kvs_json(l->events[i].kvs, sb);
      kvs_free(l->events[i].kvs);
    } else {
      str_builder_str(sb, "{}");
    }

    str_builder(sb, "}");
  }
  event_list_free(l);
  event_list_init(l);
  return ret;
}

// End of event_list.h
#include <getopt.h>
#include <ctype.h>
#include <inttypes.h>
static const char *entry_point = "main";
// Start of values.h.

//// Text I/O

typedef int (*writer)(FILE*, const void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces(FILE *f) {
  int c;
  do {
    c = getc(f);
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, f);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(FILE *f, char *buf, int bufsize) {
 start:
  skipspaces(f);

  int i = 0;
  while (i < bufsize) {
    int c = getc(f);
    buf[i] = (char)c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getc(f));
      goto start;
    } else if (!constituent((char)c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, f);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(FILE *f, char *buf, int bufsize, const char* expected) {
  next_token(f, buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    (size_t)(reader->n_elems_space * reader->elem_size));
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(FILE *f,
                                char *buf, int bufsize,
                                struct array_reader *reader, int64_t dims) {
  int ret = 1;
  int expect_elem = 1;
  char *knows_dimsize = (char*) calloc((size_t)dims, sizeof(char));
  int cur_dim = (int)dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc((size_t)dims, sizeof(int64_t));

  while (1) {
    next_token(f, buf, bufsize);
    if (strcmp(buf, "]") == 0) {
      expect_elem = 0;
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (!expect_elem && strcmp(buf, ",") == 0) {
      expect_elem = 1;
    } else if (expect_elem) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        expect_elem = 0;
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(FILE *f, char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  if (!next_token_is(f, buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 0; i < dims; i++) {
    if (!next_token_is(f, buf, bufsize, "[")) {
      return 1;
    }

    next_token(f, buf, bufsize);

    if (sscanf(buf, "%"SCNu64, (uint64_t*)&shape[i]) != 1) {
      return 1;
    }

    if (!next_token_is(f, buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(f, buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(f, buf, bufsize, ")")) {
    return 1;
  }

  // Check whether the array really is empty.
  for (int i = 0; i < dims; i++) {
    if (shape[i] == 0) {
      return 0;
    }
  }

  // Not an empty array!
  return 1;
}

static int read_str_array(FILE *f,
                          int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(f, buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(f, buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, (size_t)(elem_size*reader.n_elems_space));
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(f, buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  // Some platforms (WINDOWS) does not support scanf %hhd or its
  // cousin, %SCNi8.  Read into int first to avoid corrupting
  // memory.
  //
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(int8_t*)dest = (int8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  // Some platforms (WINDOWS) does not support scanf %hhd or its
  // cousin, %SCNu8.  Read into int first to avoid corrupting
  // memory.
  //
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%i%n", &x, &j) == 1) {
    *(uint8_t*)dest = (uint8_t)x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f16(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f16.nan") == 0) {
    *(uint16_t*)dest = float2halfbits(NAN);
    return 0;
  } else if (strcmp(buf, "f16.inf") == 0) {
    *(uint16_t*)dest = float2halfbits(INFINITY);
    return 0;
  } else if (strcmp(buf, "-f16.inf") == 0) {
    *(uint16_t*)dest = float2halfbits(-INFINITY);
    return 0;
  } else {
    int j;
    float x;
    if (sscanf(buf, "%f%n", &x, &j) == 1) {
      if (strcmp(buf+j, "") == 0 || strcmp(buf+j, "f16") == 0) {
        *(uint16_t*)dest = float2halfbits(x);
        return 0;
      }
    }
    return 1;
  }
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = (float)NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = (float)INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = (float)-INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = (double)NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = (double)INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = (double)-INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int read_str_unit(char *buf, void* dest) {
  (void)dest;
  if (strcmp(buf, "()") == 0) {
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, const int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, const uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, const int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, const uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, const int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, const uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, const int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, const uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

static int write_str_f16(FILE *out, const uint16_t *src) {
  float x = halfbits2float(*src);
  if (isnan(x)) {
    return fprintf(out, "f16.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f16.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f16.inf");
  } else {
    return fprintf(out, "%.*ff16", FLT_DIG, x);
  }
}

static int write_str_f32(FILE *out, const float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.*ff32", FLT_DIG, x);
  }
}

static int write_str_f64(FILE *out, const double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.*ff64", DBL_DIG, x);
  }
}

static int write_str_bool(FILE *out, const void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

static int write_str_unit(FILE *out, const void *src) {
  (void)src;
  return fprintf(out, "()");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

static void flip_bytes(size_t elem_size, unsigned char *elem) {
  for (size_t j=0; j<elem_size/2; j++) {
    unsigned char head = elem[j];
    size_t tail_index = elem_size-1-j;
    elem[j] = elem[tail_index];
    elem[tail_index] = head;
  }
}

// On Windows we need to explicitly set the file mode to not mangle
// newline characters.  On *nix there is no difference.
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
static void set_binary_mode(FILE *f) {
  setmode(fileno(f), O_BINARY);
}
#else
static void set_binary_mode(FILE *f) {
  (void)f;
}
#endif

static int read_byte(FILE *f, void* dest) {
  size_t num_elems_read = fread(dest, 1, 1, f);
  return num_elems_read == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int64_t size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64};
static const struct primtype_info_t f16_info =
  {.binname = " f16", .type_name = "f16",  .size = 2,
   .write_str = (writer)write_str_f16, .read_str = (str_reader)read_str_f16};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool};
static const struct primtype_info_t unit_info =
  {.binname = "bool", .type_name = "unit",   .size = 1,
   .write_str = (writer)write_str_unit, .read_str = (str_reader)read_str_unit};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f16_info, &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary(FILE *f) {
  skipspaces(f);
  int c = getc(f);
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(f, &bin_version);

    if (ret != 0) { futhark_panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      futhark_panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, f);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum(FILE *f) {
  char read_binname[4];

  int num_matched = fscanf(f, "%4c", read_binname);
  if (num_matched != 1) { futhark_panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  futhark_panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(FILE *f, const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(f, &bin_dims);
  if (ret != 0) { futhark_panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    futhark_panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum(f);
  if (bin_type != expected_type) {
    futhark_panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(FILE *f,
                          const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(f, &bin_dims);
  if (ret != 0) { futhark_panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    futhark_panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum(f);
  if (expected_type != bin_primtype) {
    futhark_panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  int64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    int64_t bin_shape;
    ret = (int)fread(&bin_shape, sizeof(bin_shape), 1, f);
    if (ret != 1) {
      futhark_panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i);
    }
    if (IS_BIG_ENDIAN) {
      flip_bytes(sizeof(bin_shape), (unsigned char*) &bin_shape);
    }
    elem_count *= bin_shape;
    shape[i] = bin_shape;
  }

  int64_t elem_size = expected_type->size;
  void* tmp = realloc(*data, (size_t)(elem_count * elem_size));
  if (tmp == NULL) {
    futhark_panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  int64_t num_elems_read = (int64_t)fread(*data, (size_t)elem_size, (size_t)elem_count, f);
  if (num_elems_read != elem_count) {
    futhark_panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    flip_bytes((size_t)elem_size, (unsigned char*) *data);
  }

  return 0;
}

static int read_array(FILE *f, const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary(f)) {
    return read_str_array(f, expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(f, expected_type, data, shape, dims);
  }
}

static int end_of_input(FILE *f) {
  skipspaces(f);
  char token[2];
  next_token(f, token, sizeof(token));
  if (strcmp(token, "") == 0) {
    return 0;
  } else {
    return 1;
  }
}

static int write_str_array(FILE *out,
                           const struct primtype_info_t *elem_type,
                           const unsigned char *data,
                           const int64_t *shape,
                           int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (const void*)data);
  } else {
    int64_t len = (int64_t)shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int8_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      fprintf(out, "empty(");
      for (int64_t i = 0; i < rank; i++) {
        fprintf(out, "[%"PRIi64"]", shape[i]);
      }
      fprintf(out, "%s", elem_type->type_name);
      fprintf(out, ")");
    } else if (rank==1) {
      fputc('[', out);
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (const void*) (data + i * elem_size));
        if (i != len-1) {
          fprintf(out, ", ");
        }
      }
      fputc(']', out);
    } else {
      fputc('[', out);
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          fprintf(out, ", ");
        }
      }
      fputc(']', out);
    }
  }
  return 0;
}

static int write_bin_array(FILE *out,
                           const struct primtype_info_t *elem_type,
                           const unsigned char *data,
                           const int64_t *shape,
                           int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fwrite(elem_type->binname, 4, 1, out);
  if (shape != NULL) {
    fwrite(shape, sizeof(int64_t), (size_t)rank, out);
  }

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      const unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, (size_t)elem_type->size, (size_t)num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type,
                       const void *data,
                       const int64_t *shape,
                       const int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(FILE *f,
                       const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary(f)) {
    char buf[100];
    next_token(f, buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(f, expected_type);
    size_t elem_size = (size_t)expected_type->size;
    size_t num_elems_read = fread(dest, elem_size, 1, f);
    if (IS_BIG_ENDIAN) {
      flip_bytes(elem_size, (unsigned char*) dest);
    }
    return num_elems_read == 1 ? 0 : 1;
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

// End of values.h.

// Start of server.h.

// Forward declarations of things that we technically don't know until
// the application header file is included, but which we need.
struct futhark_context_config;
struct futhark_context;
char *futhark_context_get_error(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
int futhark_context_config_set_tuning_param(struct futhark_context_config *cfg,
                                            const char *param_name,
                                            size_t new_value);
int futhark_get_tuning_param_count(void);
const char* futhark_get_tuning_param_name(int i);
const char* futhark_get_tuning_param_class(int i);

typedef int (*restore_fn)(const void*, FILE *, struct futhark_context*, void*);
typedef void (*store_fn)(const void*, FILE *, struct futhark_context*, void*);
typedef int (*free_fn)(const void*, struct futhark_context*, void*);
typedef int (*project_fn)(struct futhark_context*, void*, const void*);
typedef int (*new_fn)(struct futhark_context*, void**, const void*[]);

struct field {
  const char *name;
  const struct type *type;
  project_fn project;
};

struct record {
  int num_fields;
  const struct field* fields;
  new_fn new;
};

struct type {
  const char *name;
  restore_fn restore;
  store_fn store;
  free_fn free;
  const void *aux;
  const struct record *record;
};

int free_scalar(const void *aux, struct futhark_context *ctx, void *p) {
  (void)aux;
  (void)ctx;
  (void)p;
  // Nothing to do.
  return 0;
}

#define DEF_SCALAR_TYPE(T)                                      \
  int restore_##T(const void *aux, FILE *f,                     \
                  struct futhark_context *ctx, void *p) {       \
    (void)aux;                                                  \
    (void)ctx;                                                  \
    return read_scalar(f, &T##_info, p);                        \
  }                                                             \
                                                                \
  void store_##T(const void *aux, FILE *f,                      \
                 struct futhark_context *ctx, void *p) {        \
    (void)aux;                                                  \
    (void)ctx;                                                  \
    write_scalar(f, 1, &T##_info, p);                           \
  }                                                             \
                                                                \
  struct type type_##T =                                        \
    { .name = #T,                                               \
      .restore = restore_##T,                                   \
      .store = store_##T,                                       \
      .free = free_scalar                                       \
    }                                                           \

DEF_SCALAR_TYPE(i8);
DEF_SCALAR_TYPE(i16);
DEF_SCALAR_TYPE(i32);
DEF_SCALAR_TYPE(i64);
DEF_SCALAR_TYPE(u8);
DEF_SCALAR_TYPE(u16);
DEF_SCALAR_TYPE(u32);
DEF_SCALAR_TYPE(u64);
DEF_SCALAR_TYPE(f16);
DEF_SCALAR_TYPE(f32);
DEF_SCALAR_TYPE(f64);
DEF_SCALAR_TYPE(bool);

struct value {
  const struct type *type;
  union {
    void *v_ptr;
    int8_t  v_i8;
    int16_t v_i16;
    int32_t v_i32;
    int64_t v_i64;

    uint8_t  v_u8;
    uint16_t v_u16;
    uint32_t v_u32;
    uint64_t v_u64;

    uint16_t v_f16;
    float v_f32;
    double v_f64;

    bool v_bool;
  } value;
};

void* value_ptr(struct value *v) {
  if (v->type == &type_i8) {
    return &v->value.v_i8;
  }
  if (v->type == &type_i16) {
    return &v->value.v_i16;
  }
  if (v->type == &type_i32) {
    return &v->value.v_i32;
  }
  if (v->type == &type_i64) {
    return &v->value.v_i64;
  }
  if (v->type == &type_u8) {
    return &v->value.v_u8;
  }
  if (v->type == &type_u16) {
    return &v->value.v_u16;
  }
  if (v->type == &type_u32) {
    return &v->value.v_u32;
  }
  if (v->type == &type_u64) {
    return &v->value.v_u64;
  }
  if (v->type == &type_f16) {
    return &v->value.v_f16;
  }
  if (v->type == &type_f32) {
    return &v->value.v_f32;
  }
  if (v->type == &type_f64) {
    return &v->value.v_f64;
  }
  if (v->type == &type_bool) {
    return &v->value.v_bool;
  }
  return &v->value.v_ptr;
}

struct variable {
  // NULL name indicates free slot.  Name is owned by this struct.
  char *name;
  struct value value;
};

typedef int (*entry_point_fn)(struct futhark_context*, void**, void**);

struct entry_point {
  const char *name;
  entry_point_fn f;
  const char** tuning_params;
  const struct type **out_types;
  bool *out_unique;
  const struct type **in_types;
  bool *in_unique;
};

int entry_num_ins(struct entry_point *e) {
  int count = 0;
  while (e->in_types[count]) {
    count++;
  }
  return count;
}

int entry_num_outs(struct entry_point *e) {
  int count = 0;
  while (e->out_types[count]) {
    count++;
  }
  return count;
}

struct futhark_prog {
  // Last entry point identified by NULL name.
  struct entry_point *entry_points;
  // Last type identified by NULL name.
  const struct type **types;
};

struct server_state {
  struct futhark_prog prog;
  struct futhark_context_config *cfg;
  struct futhark_context *ctx;
  int variables_capacity;
  struct variable *variables;
};

struct variable* get_variable(struct server_state *s,
                              const char *name) {
  for (int i = 0; i < s->variables_capacity; i++) {
    if (s->variables[i].name != NULL &&
        strcmp(s->variables[i].name, name) == 0) {
      return &s->variables[i];
    }
  }

  return NULL;
}

struct variable* create_variable(struct server_state *s,
                                 const char *name,
                                 const struct type *type) {
  int found = -1;
  for (int i = 0; i < s->variables_capacity; i++) {
    if (found == -1 && s->variables[i].name == NULL) {
      found = i;
    } else if (s->variables[i].name != NULL &&
               strcmp(s->variables[i].name, name) == 0) {
      return NULL;
    }
  }

  if (found != -1) {
    // Found a free spot.
    s->variables[found].name = strdup(name);
    s->variables[found].value.type = type;
    return &s->variables[found];
  }

  // Need to grow the buffer.
  found = s->variables_capacity;
  s->variables_capacity *= 2;
  s->variables = realloc(s->variables,
                         s->variables_capacity * sizeof(struct variable));

  s->variables[found].name = strdup(name);
  s->variables[found].value.type = type;

  for (int i = found+1; i < s->variables_capacity; i++) {
    s->variables[i].name = NULL;
  }

  return &s->variables[found];
}

void drop_variable(struct variable *v) {
  free(v->name);
  v->name = NULL;
}

int arg_exists(const char *args[], int i) {
  return args[i] != NULL;
}

const char* get_arg(const char *args[], int i) {
  if (!arg_exists(args, i)) {
    futhark_panic(1, "Insufficient command args.\n");
  }
  return args[i];
}

const struct type* get_type(struct server_state *s, const char *name) {
  for (int i = 0; s->prog.types[i]; i++) {
    if (strcmp(s->prog.types[i]->name, name) == 0) {
      return s->prog.types[i];
    }
  }

  futhark_panic(1, "Unknown type %s\n", name);
  return NULL;
}

struct entry_point* get_entry_point(struct server_state *s, const char *name) {
  for (int i = 0; s->prog.entry_points[i].name; i++) {
    if (strcmp(s->prog.entry_points[i].name, name) == 0) {
      return &s->prog.entry_points[i];
    }
  }

  return NULL;
}

// Print the command-done marker, indicating that we are ready for
// more input.
void ok(void) {
  printf("%%%%%% OK\n");
  fflush(stdout);
}

// Print the failure marker.  Output is now an error message until the
// next ok().
void failure(void) {
  printf("%%%%%% FAILURE\n");
}

void error_check(struct server_state *s, int err) {
  if (err != 0) {
    failure();
    char *error = futhark_context_get_error(s->ctx);
    if (error != NULL) {
      puts(error);
    }
    free(error);
  }
}

void cmd_call(struct server_state *s, const char *args[]) {
  const char *name = get_arg(args, 0);

  struct entry_point *e = get_entry_point(s, name);

  if (e == NULL) {
    failure();
    printf("Unknown entry point: %s\n", name);
    return;
  }

  int num_outs = entry_num_outs(e);
  int num_ins = entry_num_ins(e);
  // +1 to avoid zero-size arrays, which is UB.
  void* outs[num_outs+1];
  void* ins[num_ins+1];

  for (int i = 0; i < num_ins; i++) {
    const char *in_name = get_arg(args, 1+num_outs+i);
    struct variable *v = get_variable(s, in_name);
    if (v == NULL) {
      failure();
      printf("Unknown variable: %s\n", in_name);
      return;
    }
    if (v->value.type != e->in_types[i]) {
      failure();
      printf("Wrong input type.  Expected %s, got %s.\n",
             e->in_types[i]->name, v->value.type->name);
      return;
    }
    ins[i] = value_ptr(&v->value);
  }

  for (int i = 0; i < num_outs; i++) {
    const char *out_name = get_arg(args, 1+i);
    struct variable *v = create_variable(s, out_name, e->out_types[i]);
    if (v == NULL) {
      failure();
      printf("Variable already exists: %s\n", out_name);
      return;
    }
    outs[i] = value_ptr(&v->value);
  }

  int64_t t_start = get_wall_time();
  int err = e->f(s->ctx, outs, ins);
  err |= futhark_context_sync(s->ctx);
  int64_t t_end = get_wall_time();
  long long int elapsed_usec = t_end - t_start;
  printf("runtime: %lld\n", elapsed_usec);

  error_check(s, err);
  if (err != 0) {
    // Need to uncreate the output variables, which would otherwise be left
    // in an uninitialised state.
    for (int i = 0; i < num_outs; i++) {
      const char *out_name = get_arg(args, 1+i);
      struct variable *v = get_variable(s, out_name);
      if (v) {
        drop_variable(v);
      }
    }
  }
}

void cmd_restore(struct server_state *s, const char *args[]) {
  const char *fname = get_arg(args, 0);

  FILE *f = fopen(fname, "rb");
  if (f == NULL) {
    failure();
    printf("Failed to open %s: %s\n", fname, strerror(errno));
    return;
  }

  int bad = 0;
  int values = 0;
  for (int i = 1; arg_exists(args, i); i+=2, values++) {
    const char *vname = get_arg(args, i);
    const char *type = get_arg(args, i+1);

    const struct type *t = get_type(s, type);
    struct variable *v = create_variable(s, vname, t);

    if (v == NULL) {
      bad = 1;
      failure();
      printf("Variable already exists: %s\n", vname);
      break;
    }

    errno = 0;
    if (t->restore(t->aux, f, s->ctx, value_ptr(&v->value)) != 0) {
      bad = 1;
      failure();
      printf("Failed to restore variable %s.\n"
             "Possibly malformed data in %s (errno: %s)\n",
             vname, fname, strerror(errno));
      drop_variable(v);
      break;
    }
  }

  if (!bad && end_of_input(f) != 0) {
    failure();
    printf("Expected EOF after reading %d values from %s\n",
           values, fname);
  }

  fclose(f);

  if (!bad) {
    int err = futhark_context_sync(s->ctx);
    error_check(s, err);
  }
}

void cmd_store(struct server_state *s, const char *args[]) {
  const char *fname = get_arg(args, 0);

  FILE *f = fopen(fname, "wb");
  if (f == NULL) {
    failure();
    printf("Failed to open %s: %s\n", fname, strerror(errno));
  } else {
    for (int i = 1; arg_exists(args, i); i++) {
      const char *vname = get_arg(args, i);
      struct variable *v = get_variable(s, vname);

      if (v == NULL) {
        failure();
        printf("Unknown variable: %s\n", vname);
        return;
      }

      const struct type *t = v->value.type;
      t->store(t->aux, f, s->ctx, value_ptr(&v->value));
    }
    fclose(f);
  }
}

void cmd_free(struct server_state *s, const char *args[]) {
  for (int i = 0; arg_exists(args, i); i++) {
    const char *name = get_arg(args, i);
    struct variable *v = get_variable(s, name);

    if (v == NULL) {
      failure();
      printf("Unknown variable: %s\n", name);
      return;
    }

    const struct type *t = v->value.type;

    int err = t->free(t->aux, s->ctx, value_ptr(&v->value));
    error_check(s, err);
    drop_variable(v);
  }
}

void cmd_rename(struct server_state *s, const char *args[]) {
  const char *oldname = get_arg(args, 0);
  const char *newname = get_arg(args, 1);
  struct variable *old = get_variable(s, oldname);
  struct variable *new = get_variable(s, newname);

  if (old == NULL) {
    failure();
    printf("Unknown variable: %s\n", oldname);
    return;
  }

  if (new != NULL) {
    failure();
    printf("Variable already exists: %s\n", newname);
    return;
  }

  free(old->name);
  old->name = strdup(newname);
}

void cmd_inputs(struct server_state *s, const char *args[]) {
  const char *name = get_arg(args, 0);
  struct entry_point *e = get_entry_point(s, name);

  if (e == NULL) {
    failure();
    printf("Unknown entry point: %s\n", name);
    return;
  }

  int num_ins = entry_num_ins(e);
  for (int i = 0; i < num_ins; i++) {
    if (e->in_unique[i]) {
      putchar('*');
    }
    puts(e->in_types[i]->name);
  }
}

void cmd_outputs(struct server_state *s, const char *args[]) {
  const char *name = get_arg(args, 0);
  struct entry_point *e = get_entry_point(s, name);

  if (e == NULL) {
    failure();
    printf("Unknown entry point: %s\n", name);
    return;
  }

  int num_outs = entry_num_outs(e);
  for (int i = 0; i < num_outs; i++) {
    if (e->out_unique[i]) {
      putchar('*');
    }
    puts(e->out_types[i]->name);
  }
}

void cmd_clear(struct server_state *s, const char *args[]) {
  (void)args;
  int err = 0;
  for (int i = 0; i < s->variables_capacity; i++) {
    struct variable *v = &s->variables[i];
    if (v->name != NULL) {
      err |= v->value.type->free(v->value.type->aux, s->ctx, value_ptr(&v->value));
      drop_variable(v);
    }
  }
  err |= futhark_context_clear_caches(s->ctx);
  error_check(s, err);
}

void cmd_pause_profiling(struct server_state *s, const char *args[]) {
  (void)args;
  futhark_context_pause_profiling(s->ctx);
}

void cmd_unpause_profiling(struct server_state *s, const char *args[]) {
  (void)args;
  futhark_context_unpause_profiling(s->ctx);
}

void cmd_report(struct server_state *s, const char *args[]) {
  (void)args;
  char *report = futhark_context_report(s->ctx);
  if (report) {
    puts(report);
  } else {
    failure();
    report = futhark_context_get_error(s->ctx);
    if (report) {
      puts(report);
    } else {
      puts("Failed to produce profiling report.\n");
    }
  }
  free(report);
}

void cmd_set_tuning_param(struct server_state *s, const char *args[]) {
  const char *param = get_arg(args, 0);
  const char *val_s = get_arg(args, 1);
  size_t val = atol(val_s);
  int err = futhark_context_config_set_tuning_param(s->cfg, param, val);

  error_check(s, err);

  if (err != 0) {
    printf("Failed to set tuning parameter %s to %ld\n", param, (long)val);
  }
}

void cmd_tuning_params(struct server_state *s, const char *args[]) {
  const char *name = get_arg(args, 0);
  struct entry_point *e = get_entry_point(s, name);

  if (e == NULL) {
    failure();
    printf("Unknown entry point: %s\n", name);
    return;
  }

  const char **params = e->tuning_params;
  for (int i = 0; params[i] != NULL; i++) {
    printf("%s\n", params[i]);
  }
}

void cmd_tuning_param_class(struct server_state *s, const char *args[]) {
  (void)s;
  const char *param = get_arg(args, 0);

  int n = futhark_get_tuning_param_count();

  for (int i = 0; i < n; i++) {
    if (strcmp(futhark_get_tuning_param_name(i), param) == 0) {
      printf("%s\n", futhark_get_tuning_param_class(i));
      return;
    }
  }

  failure();
  printf("Unknown tuning parameter: %s\n", param);
}

void cmd_fields(struct server_state *s, const char *args[]) {
  const char *type = get_arg(args, 0);
  const struct type *t = get_type(s, type);
  const struct record *r = t->record;

  if (r == NULL) {
    failure();
    printf("Not a record type\n");
    return;
  }

  for (int i = 0; i < r->num_fields; i++) {
    const struct field f = r->fields[i];
    printf("%s %s\n", f.name, f.type->name);
  }
}

void cmd_project(struct server_state *s, const char *args[]) {
  const char *to_name = get_arg(args, 0);
  const char *from_name = get_arg(args, 1);
  const char *field_name = get_arg(args, 2);

  struct variable *from = get_variable(s, from_name);

  if (from == NULL) {
    failure();
    printf("Unknown variable: %s\n", from_name);
    return;
  }

  const struct type *from_type = from->value.type;
  const struct record *r = from_type->record;

  if (r == NULL) {
    failure();
    printf("Not a record type\n");
    return;
  }

  const struct field *field = NULL;
  for (int i = 0; i < r->num_fields; i++) {
    if (strcmp(r->fields[i].name, field_name) == 0) {
      field = &r->fields[i];
      break;
    }
  }

  if (field == NULL) {
    failure();
    printf("No such field\n");
  }

  struct variable *to = create_variable(s, to_name, field->type);

  if (to == NULL) {
    failure();
    printf("Variable already exists: %s\n", to_name);
    return;
  }

  field->project(s->ctx, value_ptr(&to->value), from->value.value.v_ptr);
}

void cmd_new(struct server_state *s, const char *args[]) {
  const char *to_name = get_arg(args, 0);
  const char *type_name = get_arg(args, 1);
  const struct type *type = get_type(s, type_name);
  struct variable *to = create_variable(s, to_name, type);

  if (to == NULL) {
    failure();
    printf("Variable already exists: %s\n", to_name);
    return;
  }

  const struct record* r = type->record;

  if (r == NULL) {
    failure();
    printf("Not a record type\n");
    return;
  }

  int num_args = 0;
  for (int i = 2; arg_exists(args, i); i++) {
    num_args++;
  }

  if (num_args != r->num_fields) {
    failure();
    printf("%d fields expected but %d values provided.\n", num_args, r->num_fields);
    return;
  }

  const void** value_ptrs = alloca(num_args * sizeof(void*));

  for (int i = 0; i < num_args; i++) {
    struct variable* v = get_variable(s, args[2+i]);

    if (v == NULL) {
      failure();
      printf("Unknown variable: %s\n", args[2+i]);
      return;
    }

    if (strcmp(v->value.type->name, r->fields[i].type->name) != 0) {
      failure();
      printf("Field %s mismatch: expected type %s, got %s\n",
             r->fields[i].name, r->fields[i].type->name, v->value.type->name);
      return;
    }

    value_ptrs[i] = value_ptr(&v->value);
  }

  r->new(s->ctx, value_ptr(&to->value), value_ptrs);
}

void cmd_entry_points(struct server_state *s, const char *args[]) {
  (void)args;
  for (int i = 0; s->prog.entry_points[i].name; i++) {
    puts(s->prog.entry_points[i].name);
  }
}

void cmd_types(struct server_state *s, const char *args[]) {
  (void)args;
  for (int i = 0; s->prog.types[i] != NULL; i++) {
    puts(s->prog.types[i]->name);
  }
}

char *next_word(char **line) {
  char *p = *line;

  while (isspace(*p)) {
    p++;
  }

  if (*p == 0) {
    return NULL;
  }

  if (*p == '"') {
    char *save = p+1;
    // Skip ahead till closing quote.
    p++;

    while (*p && *p != '"') {
      p++;
    }

    if (*p == '"') {
      *p = 0;
      *line = p+1;
      return save;
    } else {
      return NULL;
    }
  } else {
    char *save = p;
    // Skip ahead till next whitespace.

    while (*p && !isspace(*p)) {
      p++;
    }

    if (*p) {
      *p = 0;
      *line = p+1;
    } else {
      *line = p;
    }
    return save;
  }
}

void process_line(struct server_state *s, char *line) {
  int max_num_tokens = 1000;
  const char* tokens[max_num_tokens];
  int num_tokens = 0;

  while ((tokens[num_tokens] = next_word(&line)) != NULL) {
    num_tokens++;
    if (num_tokens == max_num_tokens) {
      futhark_panic(1, "Line too long.\n");
    }
  }

  const char *command = tokens[0];

  if (command == NULL) {
    failure();
    printf("Empty line\n");
  } else if (strcmp(command, "call") == 0) {
    cmd_call(s, tokens+1);
  } else if (strcmp(command, "restore") == 0) {
    cmd_restore(s, tokens+1);
  } else if (strcmp(command, "store") == 0) {
    cmd_store(s, tokens+1);
  } else if (strcmp(command, "free") == 0) {
    cmd_free(s, tokens+1);
  } else if (strcmp(command, "rename") == 0) {
    cmd_rename(s, tokens+1);
  } else if (strcmp(command, "inputs") == 0) {
    cmd_inputs(s, tokens+1);
  } else if (strcmp(command, "outputs") == 0) {
    cmd_outputs(s, tokens+1);
  } else if (strcmp(command, "clear") == 0) {
    cmd_clear(s, tokens+1);
  } else if (strcmp(command, "pause_profiling") == 0) {
    cmd_pause_profiling(s, tokens+1);
  } else if (strcmp(command, "unpause_profiling") == 0) {
    cmd_unpause_profiling(s, tokens+1);
  } else if (strcmp(command, "report") == 0) {
    cmd_report(s, tokens+1);
  } else if (strcmp(command, "set_tuning_param") == 0) {
    cmd_set_tuning_param(s, tokens+1);
  } else if (strcmp(command, "tuning_params") == 0) {
    cmd_tuning_params(s, tokens+1);
  } else if (strcmp(command, "tuning_param_class") == 0) {
    cmd_tuning_param_class(s, tokens+1);
  } else if (strcmp(command, "fields") == 0) {
    cmd_fields(s, tokens+1);
  } else if (strcmp(command, "new") == 0) {
    cmd_new(s, tokens+1);
  } else if (strcmp(command, "project") == 0) {
    cmd_project(s, tokens+1);
  } else if (strcmp(command, "entry_points") == 0) {
    cmd_entry_points(s, tokens+1);
  } else if (strcmp(command, "types") == 0) {
    cmd_types(s, tokens+1);
  } else {
    futhark_panic(1, "Unknown command: %s\n", command);
  }
}

void run_server(struct futhark_prog *prog,
                struct futhark_context_config *cfg,
                struct futhark_context *ctx) {
  char *line = NULL;
  size_t buflen = 0;
  ssize_t linelen;

  struct server_state s = {
    .cfg = cfg,
    .ctx = ctx,
    .variables_capacity = 100,
    .prog = *prog
  };

  s.variables = malloc(s.variables_capacity * sizeof(struct variable));

  for (int i = 0; i < s.variables_capacity; i++) {
    s.variables[i].name = NULL;
  }

  ok();
  while ((linelen = getline(&line, &buflen, stdin)) > 0) {
    process_line(&s, line);
    ok();
  }

  free(s.variables);
  free(line);
}

// The aux struct lets us write generic method implementations without
// code duplication.

typedef void* (*array_new_fn)(struct futhark_context *, const void*, const int64_t*);
typedef const int64_t* (*array_shape_fn)(struct futhark_context*, void*);
typedef int (*array_values_fn)(struct futhark_context*, void*, void*);
typedef int (*array_free_fn)(struct futhark_context*, void*);

struct array_aux {
  int rank;
  const struct primtype_info_t* info;
  const char *name;
  array_new_fn new;
  array_shape_fn shape;
  array_values_fn values;
  array_free_fn free;
};

int restore_array(const struct array_aux *aux, FILE *f,
                  struct futhark_context *ctx, void *p) {
  void *data = NULL;
  int64_t shape[aux->rank];
  if (read_array(f, aux->info, &data, shape, aux->rank) != 0) {
    return 1;
  }

  void *arr = aux->new(ctx, data, shape);
  if (arr == NULL) {
    return 1;
  }
  int err = futhark_context_sync(ctx);
  *(void**)p = arr;
  free(data);
  return err;
}

void store_array(const struct array_aux *aux, FILE *f,
                 struct futhark_context *ctx, void *p) {
  void *arr = *(void**)p;
  const int64_t *shape = aux->shape(ctx, arr);
  int64_t size = sizeof(aux->info->size);
  for (int i = 0; i < aux->rank; i++) {
    size *= shape[i];
  }
  int32_t *data = malloc(size);
  assert(aux->values(ctx, arr, data) == 0);
  assert(futhark_context_sync(ctx) == 0);
  assert(write_array(f, 1, aux->info, data, shape, aux->rank) == 0);
  free(data);
}

int free_array(const struct array_aux *aux,
               struct futhark_context *ctx, void *p) {
  void *arr = *(void**)p;
  return aux->free(ctx, arr);
}

typedef void* (*opaque_restore_fn)(struct futhark_context*, void*);
typedef int (*opaque_store_fn)(struct futhark_context*, const void*, void **, size_t *);
typedef int (*opaque_free_fn)(struct futhark_context*, void*);

struct opaque_aux {
  opaque_restore_fn restore;
  opaque_store_fn store;
  opaque_free_fn free;
};

int restore_opaque(const struct opaque_aux *aux, FILE *f,
                   struct futhark_context *ctx, void *p) {
  // We have a problem: we need to load data from 'f', since the
  // restore function takes a pointer, but we don't know how much we
  // need (and cannot possibly).  So we do something hacky: we read
  // *all* of the file, pass all of the data to the restore function
  // (which doesn't care if there's extra at the end), then we compute
  // how much space the the object actually takes in serialised form
  // and rewind the file to that position.  The only downside is more IO.
  size_t start = ftell(f);
  size_t size;
  char *bytes = fslurp_file(f, &size);
  void *obj = aux->restore(ctx, bytes);
  free(bytes);
  if (obj != NULL) {
    *(void**)p = obj;
    size_t obj_size;
    (void)aux->store(ctx, obj, NULL, &obj_size);
    fseek(f, start+obj_size, SEEK_SET);
    return 0;
  } else {
    fseek(f, start, SEEK_SET);
    return 1;
  }
}

void store_opaque(const struct opaque_aux *aux, FILE *f,
                  struct futhark_context *ctx, void *p) {
  void *obj = *(void**)p;
  size_t obj_size;
  void *data = NULL;
  (void)aux->store(ctx, obj, &data, &obj_size);
  assert(futhark_context_sync(ctx) == 0);
  fwrite(data, sizeof(char), obj_size, f);
  free(data);
}

int free_opaque(const struct opaque_aux *aux,
                struct futhark_context *ctx, void *p) {
  void *obj = *(void**)p;
  return aux->free(ctx, obj);
}

// End of server.h.

// Start of tuning.h.


int is_blank_line_or_comment(const char *s) {
  size_t i = strspn(s, " \t\n");
  return s[i] == '\0' || // Line is blank.
         strncmp(s + i, "--", 2) == 0; // Line is comment.
}

static char* load_tuning_file(const char *fname,
                              void *cfg,
                              int (*set_tuning_param)(void*, const char*, size_t)) {
  const int max_line_len = 1024;
  char* line = (char*) malloc(max_line_len);

  FILE *f = fopen(fname, "r");

  if (f == NULL) {
    snprintf(line, max_line_len, "Cannot open file: %s", strerror(errno));
    return line;
  }

  int lineno = 0;
  while (fgets(line, max_line_len, f) != NULL) {
    lineno++;
    if (is_blank_line_or_comment(line)) {
      continue;
    }
    char *eql = strstr(line, "=");
    if (eql) {
      *eql = 0;
      char *endptr;
      int value = strtol(eql+1, &endptr, 10);
      if (*endptr && *endptr != '\n') {
        snprintf(line, max_line_len, "Invalid line %d (must be of form 'name=int').",
                 lineno);
        return line;
      }
      if (set_tuning_param(cfg, line, (size_t)value) != 0) {
        char* err = (char*) malloc(max_line_len + 50);
        snprintf(err, max_line_len + 50, "Unknown name '%s' on line %d.", line, lineno);
        free(line);
        return err;
      }
    } else {
      snprintf(line, max_line_len, "Invalid line %d (must be of form 'name=int').",
               lineno);
      return line;
    }
  }

  free(line);

  return NULL;
}

// End of tuning.h.

const struct type type_ZMZNf64;
void *futhark_new_f64_1d_wrap(struct futhark_context *ctx, const void *p, const int64_t *shape)
{
    return futhark_new_f64_1d(ctx, p, shape[0]);
}
const struct array_aux type_ZMZNf64_aux = {.name ="[]f64", .rank =1, .info =&f64_info, .new =(array_new_fn) futhark_new_f64_1d_wrap, .free =(array_free_fn) futhark_free_f64_1d, .shape =(array_shape_fn) futhark_shape_f64_1d, .values =(array_values_fn) futhark_values_f64_1d};
const struct type type_ZMZNf64 = {.name ="[]f64", .restore =(restore_fn) restore_array, .store =(store_fn) store_array, .free =(free_fn) free_array, .aux =&type_ZMZNf64_aux};
const struct type *test_jvp_csr_from_csr_ex4_matches_dense_out_types[] = {&type_bool, NULL};
bool test_jvp_csr_from_csr_ex4_matches_dense_out_unique[] = {false};
const struct type *test_jvp_csr_from_csr_ex4_matches_dense_in_types[] = {&type_ZMZNf64, NULL};
bool test_jvp_csr_from_csr_ex4_matches_dense_in_unique[] = {false};
const char *test_jvp_csr_from_csr_ex4_matches_dense_tuning_params[] = {NULL};
int call_test_jvp_csr_from_csr_ex4_matches_dense(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    struct futhark_f64_1d * in0 = *(struct futhark_f64_1d * *) ins[0];
    
    return futhark_entry_test_jvp_csr_from_csr_ex4_matches_dense(ctx, out0, in0);
}
const struct type *test_prepared_jvp_ex4_matches_dense_out_types[] = {&type_bool, NULL};
bool test_prepared_jvp_ex4_matches_dense_out_unique[] = {false};
const struct type *test_prepared_jvp_ex4_matches_dense_in_types[] = {&type_ZMZNf64, NULL};
bool test_prepared_jvp_ex4_matches_dense_in_unique[] = {false};
const char *test_prepared_jvp_ex4_matches_dense_tuning_params[] = {NULL};
int call_test_prepared_jvp_ex4_matches_dense(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    struct futhark_f64_1d * in0 = *(struct futhark_f64_1d * *) ins[0];
    
    return futhark_entry_test_prepared_jvp_ex4_matches_dense(ctx, out0, in0);
}
const struct type *test_prepared_jvp_reuse_two_points_out_types[] = {&type_bool, NULL};
bool test_prepared_jvp_reuse_two_points_out_unique[] = {false};
const struct type *test_prepared_jvp_reuse_two_points_in_types[] = {&type_ZMZNf64, NULL};
bool test_prepared_jvp_reuse_two_points_in_unique[] = {false};
const char *test_prepared_jvp_reuse_two_points_tuning_params[] = {NULL};
int call_test_prepared_jvp_reuse_two_points(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    struct futhark_f64_1d * in0 = *(struct futhark_f64_1d * *) ins[0];
    
    return futhark_entry_test_prepared_jvp_reuse_two_points(ctx, out0, in0);
}
const struct type *test_sparse_jvp_ex1_matches_dense_out_types[] = {&type_bool, NULL};
bool test_sparse_jvp_ex1_matches_dense_out_unique[] = {false};
const struct type *test_sparse_jvp_ex1_matches_dense_in_types[] = {&type_ZMZNf64, NULL};
bool test_sparse_jvp_ex1_matches_dense_in_unique[] = {false};
const char *test_sparse_jvp_ex1_matches_dense_tuning_params[] = {NULL};
int call_test_sparse_jvp_ex1_matches_dense(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    struct futhark_f64_1d * in0 = *(struct futhark_f64_1d * *) ins[0];
    
    return futhark_entry_test_sparse_jvp_ex1_matches_dense(ctx, out0, in0);
}
const struct type *test_sparse_jvp_ex2_matches_dense_out_types[] = {&type_bool, NULL};
bool test_sparse_jvp_ex2_matches_dense_out_unique[] = {false};
const struct type *test_sparse_jvp_ex2_matches_dense_in_types[] = {&type_ZMZNf64, NULL};
bool test_sparse_jvp_ex2_matches_dense_in_unique[] = {false};
const char *test_sparse_jvp_ex2_matches_dense_tuning_params[] = {NULL};
int call_test_sparse_jvp_ex2_matches_dense(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    struct futhark_f64_1d * in0 = *(struct futhark_f64_1d * *) ins[0];
    
    return futhark_entry_test_sparse_jvp_ex2_matches_dense(ctx, out0, in0);
}
const struct type *test_sparse_jvp_ex4_matches_dense_out_types[] = {&type_bool, NULL};
bool test_sparse_jvp_ex4_matches_dense_out_unique[] = {false};
const struct type *test_sparse_jvp_ex4_matches_dense_in_types[] = {&type_ZMZNf64, NULL};
bool test_sparse_jvp_ex4_matches_dense_in_unique[] = {false};
const char *test_sparse_jvp_ex4_matches_dense_tuning_params[] = {NULL};
int call_test_sparse_jvp_ex4_matches_dense(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    struct futhark_f64_1d * in0 = *(struct futhark_f64_1d * *) ins[0];
    
    return futhark_entry_test_sparse_jvp_ex4_matches_dense(ctx, out0, in0);
}
const struct type *test_sparse_jvp_ex4_with_colors_matches_dense_out_types[] = {&type_bool, NULL};
bool test_sparse_jvp_ex4_with_colors_matches_dense_out_unique[] = {false};
const struct type *test_sparse_jvp_ex4_with_colors_matches_dense_in_types[] = {&type_ZMZNf64, NULL};
bool test_sparse_jvp_ex4_with_colors_matches_dense_in_unique[] = {false};
const char *test_sparse_jvp_ex4_with_colors_matches_dense_tuning_params[] = {NULL};
int call_test_sparse_jvp_ex4_with_colors_matches_dense(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    struct futhark_f64_1d * in0 = *(struct futhark_f64_1d * *) ins[0];
    
    return futhark_entry_test_sparse_jvp_ex4_with_colors_matches_dense(ctx, out0, in0);
}
const struct type *test_sparse_jvp_ex5_matches_dense_out_types[] = {&type_bool, NULL};
bool test_sparse_jvp_ex5_matches_dense_out_unique[] = {false};
const struct type *test_sparse_jvp_ex5_matches_dense_in_types[] = {&type_ZMZNf64, NULL};
bool test_sparse_jvp_ex5_matches_dense_in_unique[] = {false};
const char *test_sparse_jvp_ex5_matches_dense_tuning_params[] = {NULL};
int call_test_sparse_jvp_ex5_matches_dense(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    struct futhark_f64_1d * in0 = *(struct futhark_f64_1d * *) ins[0];
    
    return futhark_entry_test_sparse_jvp_ex5_matches_dense(ctx, out0, in0);
}
const struct type *test_sparse_jvp_zzero_pattern_matches_dense_out_types[] = {&type_bool, NULL};
bool test_sparse_jvp_zzero_pattern_matches_dense_out_unique[] = {false};
const struct type *test_sparse_jvp_zzero_pattern_matches_dense_in_types[] = {&type_ZMZNf64, NULL};
bool test_sparse_jvp_zzero_pattern_matches_dense_in_unique[] = {false};
const char *test_sparse_jvp_zzero_pattern_matches_dense_tuning_params[] = {NULL};
int call_test_sparse_jvp_zzero_pattern_matches_dense(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    struct futhark_f64_1d * in0 = *(struct futhark_f64_1d * *) ins[0];
    
    return futhark_entry_test_sparse_jvp_zero_pattern_matches_dense(ctx, out0, in0);
}
const struct type *types[] = {&type_i8, &type_i16, &type_i32, &type_i64, &type_u8, &type_u16, &type_u32, &type_u64, &type_f16, &type_f32, &type_f64, &type_bool, &type_ZMZNf64, NULL};
struct entry_point entry_points[] = {{.name ="test_jvp_csr_from_csr_ex4_matches_dense", .f =call_test_jvp_csr_from_csr_ex4_matches_dense, .tuning_params =test_jvp_csr_from_csr_ex4_matches_dense_tuning_params, .in_types =test_jvp_csr_from_csr_ex4_matches_dense_in_types, .out_types =test_jvp_csr_from_csr_ex4_matches_dense_out_types, .in_unique =test_jvp_csr_from_csr_ex4_matches_dense_in_unique, .out_unique =test_jvp_csr_from_csr_ex4_matches_dense_out_unique}, {.name ="test_prepared_jvp_ex4_matches_dense", .f =call_test_prepared_jvp_ex4_matches_dense, .tuning_params =test_prepared_jvp_ex4_matches_dense_tuning_params, .in_types =test_prepared_jvp_ex4_matches_dense_in_types, .out_types =test_prepared_jvp_ex4_matches_dense_out_types, .in_unique =test_prepared_jvp_ex4_matches_dense_in_unique, .out_unique =test_prepared_jvp_ex4_matches_dense_out_unique}, {.name ="test_prepared_jvp_reuse_two_points", .f =call_test_prepared_jvp_reuse_two_points, .tuning_params =test_prepared_jvp_reuse_two_points_tuning_params, .in_types =test_prepared_jvp_reuse_two_points_in_types, .out_types =test_prepared_jvp_reuse_two_points_out_types, .in_unique =test_prepared_jvp_reuse_two_points_in_unique, .out_unique =test_prepared_jvp_reuse_two_points_out_unique}, {.name ="test_sparse_jvp_ex1_matches_dense", .f =call_test_sparse_jvp_ex1_matches_dense, .tuning_params =test_sparse_jvp_ex1_matches_dense_tuning_params, .in_types =test_sparse_jvp_ex1_matches_dense_in_types, .out_types =test_sparse_jvp_ex1_matches_dense_out_types, .in_unique =test_sparse_jvp_ex1_matches_dense_in_unique, .out_unique =test_sparse_jvp_ex1_matches_dense_out_unique}, {.name ="test_sparse_jvp_ex2_matches_dense", .f =call_test_sparse_jvp_ex2_matches_dense, .tuning_params =test_sparse_jvp_ex2_matches_dense_tuning_params, .in_types =test_sparse_jvp_ex2_matches_dense_in_types, .out_types =test_sparse_jvp_ex2_matches_dense_out_types, .in_unique =test_sparse_jvp_ex2_matches_dense_in_unique, .out_unique =test_sparse_jvp_ex2_matches_dense_out_unique}, {.name ="test_sparse_jvp_ex4_matches_dense", .f =call_test_sparse_jvp_ex4_matches_dense, .tuning_params =test_sparse_jvp_ex4_matches_dense_tuning_params, .in_types =test_sparse_jvp_ex4_matches_dense_in_types, .out_types =test_sparse_jvp_ex4_matches_dense_out_types, .in_unique =test_sparse_jvp_ex4_matches_dense_in_unique, .out_unique =test_sparse_jvp_ex4_matches_dense_out_unique}, {.name ="test_sparse_jvp_ex4_with_colors_matches_dense", .f =call_test_sparse_jvp_ex4_with_colors_matches_dense, .tuning_params =test_sparse_jvp_ex4_with_colors_matches_dense_tuning_params, .in_types =test_sparse_jvp_ex4_with_colors_matches_dense_in_types, .out_types =test_sparse_jvp_ex4_with_colors_matches_dense_out_types, .in_unique =test_sparse_jvp_ex4_with_colors_matches_dense_in_unique, .out_unique =test_sparse_jvp_ex4_with_colors_matches_dense_out_unique}, {.name ="test_sparse_jvp_ex5_matches_dense", .f =call_test_sparse_jvp_ex5_matches_dense, .tuning_params =test_sparse_jvp_ex5_matches_dense_tuning_params, .in_types =test_sparse_jvp_ex5_matches_dense_in_types, .out_types =test_sparse_jvp_ex5_matches_dense_out_types, .in_unique =test_sparse_jvp_ex5_matches_dense_in_unique, .out_unique =test_sparse_jvp_ex5_matches_dense_out_unique}, {.name ="test_sparse_jvp_zero_pattern_matches_dense", .f =call_test_sparse_jvp_zzero_pattern_matches_dense, .tuning_params =test_sparse_jvp_zzero_pattern_matches_dense_tuning_params, .in_types =test_sparse_jvp_zzero_pattern_matches_dense_in_types, .out_types =test_sparse_jvp_zzero_pattern_matches_dense_out_types, .in_unique =test_sparse_jvp_zzero_pattern_matches_dense_in_unique, .out_unique =test_sparse_jvp_zzero_pattern_matches_dense_out_unique}, {.name =NULL}};
struct futhark_prog prog = {.types =types, .entry_points =entry_points};
int parse_options(struct futhark_context_config *cfg, int argc, char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"debugging", no_argument, NULL, 1}, {"log", no_argument, NULL, 2}, {"profile", no_argument, NULL, 3}, {"help", no_argument, NULL, 4}, {"print-params", no_argument, NULL, 5}, {"param", required_argument, NULL, 6}, {"tuning", required_argument, NULL, 7}, {"cache-file", required_argument, NULL, 8}, {0, 0, 0, 0}};
    static char *option_descriptions = "  -D/--debugging     Perform possibly expensive internal correctness checks and verbose logging.\n  -L/--log           Print various low-overhead logging information while running.\n  -P/--profile       Enable the collection of profiling information.\n  -h/--help          Print help information and exit.\n  --print-params     Print all tuning parameters that can be set with --param or --tuning.\n  --param ASSIGNMENT Set a tuning parameter to the given value.\n  --tuning FILE      Read size=value assignments from the given file.\n  --cache-file FILE  Store program cache here.\n";
    
    while ((ch = getopt_long(argc, argv, ":DLPh", long_options, NULL)) != -1) {
        if (ch == 1 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 2 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 3 || ch == 'P')
            futhark_context_config_set_profiling(cfg, 1);
        if (ch == 4 || ch == 'h') {
            printf("Usage: %s [OPTIONS]...\nOptions:\n\n%s\nFor more information, consult the Futhark User's Guide or the man pages.\n", fut_progname, option_descriptions);
            exit(0);
        }
        if (ch == 5) {
            int n = futhark_get_tuning_param_count();
            
            for (int i = 0; i < n; i++)
                printf("%s (%s)\n", futhark_get_tuning_param_name(i), futhark_get_tuning_param_class(i));
            exit(0);
        }
        if (ch == 6) {
            char *name = optarg;
            char *equals = strstr(optarg, "=");
            char *value_str = equals != NULL ? equals + 1 : optarg;
            int value = atoi(value_str);
            
            if (equals != NULL) {
                *equals = 0;
                if (futhark_context_config_set_tuning_param(cfg, name, value) != 0)
                    futhark_panic(1, "Unknown size: %s\n", name);
            } else
                futhark_panic(1, "Invalid argument for size option: %s\n", optarg);
        }
        if (ch == 7) {
            char *ret = load_tuning_file(optarg, cfg, (int (*)(void *, const char *, size_t)) futhark_context_config_set_tuning_param);
            
            if (ret != NULL)
                futhark_panic(1, "When loading tuning file '%s': %s\n", optarg, ret);
        }
        if (ch == 8)
            futhark_context_config_set_cache_file(cfg, optarg);
        if (ch == ':')
            futhark_panic(-1, "Missing argument for option %s\n", argv[optind - 1]);
        if (ch == '?') {
            fprintf(stderr, "Usage: %s [OPTIONS]...\nOptions:\n\n%s\n", fut_progname, "  -D/--debugging     Perform possibly expensive internal correctness checks and verbose logging.\n  -L/--log           Print various low-overhead logging information while running.\n  -P/--profile       Enable the collection of profiling information.\n  -h/--help          Print help information and exit.\n  --print-params     Print all tuning parameters that can be set with --param or --tuning.\n  --param ASSIGNMENT Set a tuning parameter to the given value.\n  --tuning FILE      Read size=value assignments from the given file.\n  --cache-file FILE  Store program cache here.\n");
            futhark_panic(1, "Unknown option: %s\n", argv[optind - 1]);
        }
    }
    return optind;
}
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        futhark_panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    futhark_context_set_logging_file(ctx, stdout);
    
    char *error = futhark_context_get_error(ctx);
    
    if (error != NULL)
        futhark_panic(1, "Error during context initialisation:\n%s", error);
    if (entry_point != NULL)
        run_server(&prog, cfg, ctx);
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
}

#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <ctype.h>



#define FUTHARK_F64_ENABLED

// Start of scalar.h.

// Implementation of the primitive scalar operations.  Very
// repetitive.  This code is inserted directly into both CUDA and
// OpenCL programs, as well as the CPU code, so it has some #ifdefs to
// work everywhere.  Some operations are defined as macros because
// this allows us to use them as constant expressions in things like
// array sizes and static initialisers.

// Some of the #ifdefs are because OpenCL uses type-generic functions
// for some operations (e.g. sqrt), while C and CUDA sensibly use
// distinct functions for different precisions (e.g. sqrtf() and
// sqrt()).  This is quite annoying.  Due to C's unfortunate casting
// rules, it is also really easy to accidentally implement
// floating-point functions in the wrong precision, so be careful.

// Double-precision definitions are only included if the preprocessor
// macro FUTHARK_F64_ENABLED is set.

#ifndef M_PI
#define M_PI 3.141592653589793
#endif

SCALAR_FUN_ATTR int32_t fptobits_f32_i32(float x);
SCALAR_FUN_ATTR float bitstofp_i32_f32(int32_t x);

SCALAR_FUN_ATTR uint8_t   add8(uint8_t x, uint8_t y)   { return x + y; }
SCALAR_FUN_ATTR uint16_t add16(uint16_t x, uint16_t y) { return x + y; }
SCALAR_FUN_ATTR uint32_t add32(uint32_t x, uint32_t y) { return x + y; }
SCALAR_FUN_ATTR uint64_t add64(uint64_t x, uint64_t y) { return x + y; }

SCALAR_FUN_ATTR uint8_t   sub8(uint8_t x, uint8_t y)   { return x - y; }
SCALAR_FUN_ATTR uint16_t sub16(uint16_t x, uint16_t y) { return x - y; }
SCALAR_FUN_ATTR uint32_t sub32(uint32_t x, uint32_t y) { return x - y; }
SCALAR_FUN_ATTR uint64_t sub64(uint64_t x, uint64_t y) { return x - y; }

SCALAR_FUN_ATTR uint8_t   mul8(uint8_t x, uint8_t y)   { return x * y; }
SCALAR_FUN_ATTR uint16_t mul16(uint16_t x, uint16_t y) { return x * y; }
SCALAR_FUN_ATTR uint32_t mul32(uint32_t x, uint32_t y) { return x * y; }
SCALAR_FUN_ATTR uint64_t mul64(uint64_t x, uint64_t y) { return x * y; }

#if defined(ISPC)

SCALAR_FUN_ATTR uint8_t udiv8(uint8_t x, uint8_t y) {
  // This strange pattern is used to prevent the ISPC compiler from
  // causing SIGFPEs and bogus results on divisions where inactive lanes
  // have 0-valued divisors. It ensures that any inactive lane instead
  // has a divisor of 1. https://github.com/ispc/ispc/issues/2292
  uint8_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR uint16_t udiv16(uint16_t x, uint16_t y) {
  uint16_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR uint32_t udiv32(uint32_t x, uint32_t y) {
  uint32_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR uint64_t udiv64(uint64_t x, uint64_t y) {
  uint64_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR uint8_t udiv_up8(uint8_t x, uint8_t y) {
  uint8_t ys = 1;
  foreach_active(i) { ys = y; }
  return (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint16_t udiv_up16(uint16_t x, uint16_t y) {
  uint16_t ys = 1;
  foreach_active(i) { ys = y; }
  return (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint32_t udiv_up32(uint32_t x, uint32_t y) {
  uint32_t ys = 1;
  foreach_active(i) { ys = y; }
  return (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint64_t udiv_up64(uint64_t x, uint64_t y) {
  uint64_t ys = 1;
  foreach_active(i) { ys = y; }
  return (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint8_t umod8(uint8_t x, uint8_t y) {
  uint8_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR uint16_t umod16(uint16_t x, uint16_t y) {
  uint16_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR uint32_t umod32(uint32_t x, uint32_t y) {
  uint32_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR uint64_t umod64(uint64_t x, uint64_t y) {
  uint64_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR uint8_t udiv_safe8(uint8_t x, uint8_t y) {
  uint8_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR uint16_t udiv_safe16(uint16_t x, uint16_t y) {
  uint16_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR uint32_t udiv_safe32(uint32_t x, uint32_t y) {
  uint32_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR uint64_t udiv_safe64(uint64_t x, uint64_t y) {
  uint64_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR uint8_t udiv_up_safe8(uint8_t x, uint8_t y) {
  uint8_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint16_t udiv_up_safe16(uint16_t x, uint16_t y) {
  uint16_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint32_t udiv_up_safe32(uint32_t x, uint32_t y) {
  uint32_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint64_t udiv_up_safe64(uint64_t x, uint64_t y) {
  uint64_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : (x + y - 1) / ys;
}

SCALAR_FUN_ATTR uint8_t umod_safe8(uint8_t x, uint8_t y) {
  uint8_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

SCALAR_FUN_ATTR uint16_t umod_safe16(uint16_t x, uint16_t y) {
  uint16_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

SCALAR_FUN_ATTR uint32_t umod_safe32(uint32_t x, uint32_t y) {
  uint32_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

SCALAR_FUN_ATTR uint64_t umod_safe64(uint64_t x, uint64_t y) {
  uint64_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

SCALAR_FUN_ATTR int8_t sdiv8(int8_t x, int8_t y) {
  int8_t ys = 1;
  foreach_active(i) { ys = y; }
  int8_t q = x / ys;
  int8_t r = x % ys;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int16_t sdiv16(int16_t x, int16_t y) {
  int16_t ys = 1;
  foreach_active(i) { ys = y; }
  int16_t q = x / ys;
  int16_t r = x % ys;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int32_t sdiv32(int32_t x, int32_t y) {
  int32_t ys = 1;
  foreach_active(i) { ys = y; }
  int32_t q = x / ys;
  int32_t r = x % ys;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int64_t sdiv64(int64_t x, int64_t y) {
  int64_t ys = 1;
  foreach_active(i) { ys = y; }
  int64_t q = x / ys;
  int64_t r = x % ys;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int8_t sdiv_up8(int8_t x, int8_t y) { return sdiv8(x + y - 1, y); }
SCALAR_FUN_ATTR int16_t sdiv_up16(int16_t x, int16_t y) { return sdiv16(x + y - 1, y); }
SCALAR_FUN_ATTR int32_t sdiv_up32(int32_t x, int32_t y) { return sdiv32(x + y - 1, y); }
SCALAR_FUN_ATTR int64_t sdiv_up64(int64_t x, int64_t y) { return sdiv64(x + y - 1, y); }

SCALAR_FUN_ATTR int8_t smod8(int8_t x, int8_t y) {
  int8_t ys = 1;
  foreach_active(i) { ys = y; }
  int8_t r = x % ys;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int16_t smod16(int16_t x, int16_t y) {
  int16_t ys = 1;
  foreach_active(i) { ys = y; }
  int16_t r = x % ys;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int32_t smod32(int32_t x, int32_t y) {
  int32_t ys = 1;
  foreach_active(i) { ys = y; }
  int32_t r = x % ys;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int64_t smod64(int64_t x, int64_t y) {
  int64_t ys = 1;
  foreach_active(i) { ys = y; }
  int64_t r = x % ys;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int8_t   sdiv_safe8(int8_t x, int8_t y)   { return y == 0 ? 0 : sdiv8(x, y); }
SCALAR_FUN_ATTR int16_t sdiv_safe16(int16_t x, int16_t y) { return y == 0 ? 0 : sdiv16(x, y); }
SCALAR_FUN_ATTR int32_t sdiv_safe32(int32_t x, int32_t y) { return y == 0 ? 0 : sdiv32(x, y); }
SCALAR_FUN_ATTR int64_t sdiv_safe64(int64_t x, int64_t y) { return y == 0 ? 0 : sdiv64(x, y); }

SCALAR_FUN_ATTR int8_t sdiv_up_safe8(int8_t x, int8_t y)     { return sdiv_safe8(x + y - 1, y); }
SCALAR_FUN_ATTR int16_t sdiv_up_safe16(int16_t x, int16_t y) { return sdiv_safe16(x + y - 1, y); }
SCALAR_FUN_ATTR int32_t sdiv_up_safe32(int32_t x, int32_t y) { return sdiv_safe32(x + y - 1, y); }
SCALAR_FUN_ATTR int64_t sdiv_up_safe64(int64_t x, int64_t y) { return sdiv_safe64(x + y - 1, y); }

SCALAR_FUN_ATTR int8_t   smod_safe8(int8_t x, int8_t y)   { return y == 0 ? 0 : smod8(x, y); }
SCALAR_FUN_ATTR int16_t smod_safe16(int16_t x, int16_t y) { return y == 0 ? 0 : smod16(x, y); }
SCALAR_FUN_ATTR int32_t smod_safe32(int32_t x, int32_t y) { return y == 0 ? 0 : smod32(x, y); }
SCALAR_FUN_ATTR int64_t smod_safe64(int64_t x, int64_t y) { return y == 0 ? 0 : smod64(x, y); }

SCALAR_FUN_ATTR int8_t squot8(int8_t x, int8_t y) {
  int8_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR int16_t squot16(int16_t x, int16_t y) {
  int16_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR int32_t squot32(int32_t x, int32_t y) {
  int32_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR int64_t squot64(int64_t x, int64_t y) {
  int64_t ys = 1;
  foreach_active(i) { ys = y; }
  return x / ys;
}

SCALAR_FUN_ATTR int8_t srem8(int8_t x, int8_t y) {
  int8_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR int16_t srem16(int16_t x, int16_t y) {
  int16_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR int32_t srem32(int32_t x, int32_t y) {
  int32_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR int64_t srem64(int64_t x, int64_t y) {
  int8_t ys = 1;
  foreach_active(i) { ys = y; }
  return x % ys;
}

SCALAR_FUN_ATTR int8_t squot_safe8(int8_t x, int8_t y) {
  int8_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR int16_t squot_safe16(int16_t x, int16_t y) {
  int16_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR int32_t squot_safe32(int32_t x, int32_t y) {
  int32_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR int64_t squot_safe64(int64_t x, int64_t y) {
  int64_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x / ys;
}

SCALAR_FUN_ATTR int8_t srem_safe8(int8_t x, int8_t y) {
  int8_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

SCALAR_FUN_ATTR int16_t srem_safe16(int16_t x, int16_t y) {
  int16_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

SCALAR_FUN_ATTR int32_t srem_safe32(int32_t x, int32_t y) {
  int32_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

SCALAR_FUN_ATTR int64_t srem_safe64(int64_t x, int64_t y) {
  int64_t ys = 1;
  foreach_active(i) { ys = y; }
  return y == 0 ? 0 : x % ys;
}

#else

SCALAR_FUN_ATTR uint8_t   udiv8(uint8_t x, uint8_t y)   { return x / y; }
SCALAR_FUN_ATTR uint16_t udiv16(uint16_t x, uint16_t y) { return x / y; }
SCALAR_FUN_ATTR uint32_t udiv32(uint32_t x, uint32_t y) { return x / y; }
SCALAR_FUN_ATTR uint64_t udiv64(uint64_t x, uint64_t y) { return x / y; }

SCALAR_FUN_ATTR uint8_t   udiv_up8(uint8_t x, uint8_t y)   { return (x + y - 1) / y; }
SCALAR_FUN_ATTR uint16_t udiv_up16(uint16_t x, uint16_t y) { return (x + y - 1) / y; }
SCALAR_FUN_ATTR uint32_t udiv_up32(uint32_t x, uint32_t y) { return (x + y - 1) / y; }
SCALAR_FUN_ATTR uint64_t udiv_up64(uint64_t x, uint64_t y) { return (x + y - 1) / y; }

SCALAR_FUN_ATTR uint8_t   umod8(uint8_t x, uint8_t y)   { return x % y; }
SCALAR_FUN_ATTR uint16_t umod16(uint16_t x, uint16_t y) { return x % y; }
SCALAR_FUN_ATTR uint32_t umod32(uint32_t x, uint32_t y) { return x % y; }
SCALAR_FUN_ATTR uint64_t umod64(uint64_t x, uint64_t y) { return x % y; }

SCALAR_FUN_ATTR uint8_t   udiv_safe8(uint8_t x, uint8_t y)   { return y == 0 ? 0 : x / y; }
SCALAR_FUN_ATTR uint16_t udiv_safe16(uint16_t x, uint16_t y) { return y == 0 ? 0 : x / y; }
SCALAR_FUN_ATTR uint32_t udiv_safe32(uint32_t x, uint32_t y) { return y == 0 ? 0 : x / y; }
SCALAR_FUN_ATTR uint64_t udiv_safe64(uint64_t x, uint64_t y) { return y == 0 ? 0 : x / y; }

SCALAR_FUN_ATTR uint8_t   udiv_up_safe8(uint8_t x, uint8_t y)   { return y == 0 ? 0 : (x + y - 1) / y; }
SCALAR_FUN_ATTR uint16_t udiv_up_safe16(uint16_t x, uint16_t y) { return y == 0 ? 0 : (x + y - 1) / y; }
SCALAR_FUN_ATTR uint32_t udiv_up_safe32(uint32_t x, uint32_t y) { return y == 0 ? 0 : (x + y - 1) / y; }
SCALAR_FUN_ATTR uint64_t udiv_up_safe64(uint64_t x, uint64_t y) { return y == 0 ? 0 : (x + y - 1) / y; }

SCALAR_FUN_ATTR uint8_t   umod_safe8(uint8_t x, uint8_t y)   { return y == 0 ? 0 : x % y; }
SCALAR_FUN_ATTR uint16_t umod_safe16(uint16_t x, uint16_t y) { return y == 0 ? 0 : x % y; }
SCALAR_FUN_ATTR uint32_t umod_safe32(uint32_t x, uint32_t y) { return y == 0 ? 0 : x % y; }
SCALAR_FUN_ATTR uint64_t umod_safe64(uint64_t x, uint64_t y) { return y == 0 ? 0 : x % y; }

SCALAR_FUN_ATTR int8_t sdiv8(int8_t x, int8_t y) {
  int8_t q = x / y;
  int8_t r = x % y;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int16_t sdiv16(int16_t x, int16_t y) {
  int16_t q = x / y;
  int16_t r = x % y;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int32_t sdiv32(int32_t x, int32_t y) {
  int32_t q = x / y;
  int32_t r = x % y;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int64_t sdiv64(int64_t x, int64_t y) {
  int64_t q = x / y;
  int64_t r = x % y;
  return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}

SCALAR_FUN_ATTR int8_t   sdiv_up8(int8_t x, int8_t y)   { return sdiv8(x + y - 1, y); }
SCALAR_FUN_ATTR int16_t sdiv_up16(int16_t x, int16_t y) { return sdiv16(x + y - 1, y); }
SCALAR_FUN_ATTR int32_t sdiv_up32(int32_t x, int32_t y) { return sdiv32(x + y - 1, y); }
SCALAR_FUN_ATTR int64_t sdiv_up64(int64_t x, int64_t y) { return sdiv64(x + y - 1, y); }

SCALAR_FUN_ATTR int8_t smod8(int8_t x, int8_t y) {
  int8_t r = x % y;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int16_t smod16(int16_t x, int16_t y) {
  int16_t r = x % y;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int32_t smod32(int32_t x, int32_t y) {
  int32_t r = x % y;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int64_t smod64(int64_t x, int64_t y) {
  int64_t r = x % y;
  return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}

SCALAR_FUN_ATTR int8_t   sdiv_safe8(int8_t x, int8_t y)   { return y == 0 ? 0 : sdiv8(x, y); }
SCALAR_FUN_ATTR int16_t sdiv_safe16(int16_t x, int16_t y) { return y == 0 ? 0 : sdiv16(x, y); }
SCALAR_FUN_ATTR int32_t sdiv_safe32(int32_t x, int32_t y) { return y == 0 ? 0 : sdiv32(x, y); }
SCALAR_FUN_ATTR int64_t sdiv_safe64(int64_t x, int64_t y) { return y == 0 ? 0 : sdiv64(x, y); }

SCALAR_FUN_ATTR int8_t   sdiv_up_safe8(int8_t x, int8_t y)   { return sdiv_safe8(x + y - 1, y);}
SCALAR_FUN_ATTR int16_t sdiv_up_safe16(int16_t x, int16_t y) { return sdiv_safe16(x + y - 1, y); }
SCALAR_FUN_ATTR int32_t sdiv_up_safe32(int32_t x, int32_t y) { return sdiv_safe32(x + y - 1, y); }
SCALAR_FUN_ATTR int64_t sdiv_up_safe64(int64_t x, int64_t y) { return sdiv_safe64(x + y - 1, y); }

SCALAR_FUN_ATTR int8_t   smod_safe8(int8_t x, int8_t y)   { return y == 0 ? 0 : smod8(x, y); }
SCALAR_FUN_ATTR int16_t smod_safe16(int16_t x, int16_t y) { return y == 0 ? 0 : smod16(x, y); }
SCALAR_FUN_ATTR int32_t smod_safe32(int32_t x, int32_t y) { return y == 0 ? 0 : smod32(x, y); }
SCALAR_FUN_ATTR int64_t smod_safe64(int64_t x, int64_t y) { return y == 0 ? 0 : smod64(x, y); }

SCALAR_FUN_ATTR int8_t   squot8(int8_t x, int8_t y)   { return x / y; }
SCALAR_FUN_ATTR int16_t squot16(int16_t x, int16_t y) { return x / y; }
SCALAR_FUN_ATTR int32_t squot32(int32_t x, int32_t y) { return x / y; }
SCALAR_FUN_ATTR int64_t squot64(int64_t x, int64_t y) { return x / y; }

SCALAR_FUN_ATTR int8_t   srem8(int8_t x, int8_t y)   { return x % y; }
SCALAR_FUN_ATTR int16_t srem16(int16_t x, int16_t y) { return x % y; }
SCALAR_FUN_ATTR int32_t srem32(int32_t x, int32_t y) { return x % y; }
SCALAR_FUN_ATTR int64_t srem64(int64_t x, int64_t y) { return x % y; }

SCALAR_FUN_ATTR int8_t   squot_safe8(int8_t x, int8_t y)   { return y == 0 ? 0 : x / y; }
SCALAR_FUN_ATTR int16_t squot_safe16(int16_t x, int16_t y) { return y == 0 ? 0 : x / y; }
SCALAR_FUN_ATTR int32_t squot_safe32(int32_t x, int32_t y) { return y == 0 ? 0 : x / y; }
SCALAR_FUN_ATTR int64_t squot_safe64(int64_t x, int64_t y) { return y == 0 ? 0 : x / y; }

SCALAR_FUN_ATTR int8_t   srem_safe8(int8_t x, int8_t y)   { return y == 0 ? 0 : x % y; }
SCALAR_FUN_ATTR int16_t srem_safe16(int16_t x, int16_t y) { return y == 0 ? 0 : x % y; }
SCALAR_FUN_ATTR int32_t srem_safe32(int32_t x, int32_t y) { return y == 0 ? 0 : x % y; }
SCALAR_FUN_ATTR int64_t srem_safe64(int64_t x, int64_t y) { return y == 0 ? 0 : x % y; }

#endif

SCALAR_FUN_ATTR int8_t   smin8(int8_t x, int8_t y)   { return x < y ? x : y; }
SCALAR_FUN_ATTR int16_t smin16(int16_t x, int16_t y) { return x < y ? x : y; }
SCALAR_FUN_ATTR int32_t smin32(int32_t x, int32_t y) { return x < y ? x : y; }
SCALAR_FUN_ATTR int64_t smin64(int64_t x, int64_t y) { return x < y ? x : y; }

SCALAR_FUN_ATTR uint8_t   umin8(uint8_t x, uint8_t y)   { return x < y ? x : y; }
SCALAR_FUN_ATTR uint16_t umin16(uint16_t x, uint16_t y) { return x < y ? x : y; }
SCALAR_FUN_ATTR uint32_t umin32(uint32_t x, uint32_t y) { return x < y ? x : y; }
SCALAR_FUN_ATTR uint64_t umin64(uint64_t x, uint64_t y) { return x < y ? x : y; }

SCALAR_FUN_ATTR int8_t  smax8(int8_t x, int8_t y)    { return x < y ? y : x; }
SCALAR_FUN_ATTR int16_t smax16(int16_t x, int16_t y) { return x < y ? y : x; }
SCALAR_FUN_ATTR int32_t smax32(int32_t x, int32_t y) { return x < y ? y : x; }
SCALAR_FUN_ATTR int64_t smax64(int64_t x, int64_t y) { return x < y ? y : x; }

SCALAR_FUN_ATTR uint8_t   umax8(uint8_t x, uint8_t y)   { return x < y ? y : x; }
SCALAR_FUN_ATTR uint16_t umax16(uint16_t x, uint16_t y) { return x < y ? y : x; }
SCALAR_FUN_ATTR uint32_t umax32(uint32_t x, uint32_t y) { return x < y ? y : x; }
SCALAR_FUN_ATTR uint64_t umax64(uint64_t x, uint64_t y) { return x < y ? y : x; }

SCALAR_FUN_ATTR uint8_t   shl8(uint8_t x, uint8_t y)   { return (uint8_t)(x << y); }
SCALAR_FUN_ATTR uint16_t shl16(uint16_t x, uint16_t y) { return (uint16_t)(x << y); }
SCALAR_FUN_ATTR uint32_t shl32(uint32_t x, uint32_t y) { return x << y; }
SCALAR_FUN_ATTR uint64_t shl64(uint64_t x, uint64_t y) { return x << y; }

SCALAR_FUN_ATTR uint8_t   lshr8(uint8_t x, uint8_t y)   { return x >> y; }
SCALAR_FUN_ATTR uint16_t lshr16(uint16_t x, uint16_t y) { return x >> y; }
SCALAR_FUN_ATTR uint32_t lshr32(uint32_t x, uint32_t y) { return x >> y; }
SCALAR_FUN_ATTR uint64_t lshr64(uint64_t x, uint64_t y) { return x >> y; }

SCALAR_FUN_ATTR int8_t   ashr8(int8_t x, int8_t y)   { return x >> y; }
SCALAR_FUN_ATTR int16_t ashr16(int16_t x, int16_t y) { return x >> y; }
SCALAR_FUN_ATTR int32_t ashr32(int32_t x, int32_t y) { return x >> y; }
SCALAR_FUN_ATTR int64_t ashr64(int64_t x, int64_t y) { return x >> y; }

SCALAR_FUN_ATTR uint8_t   and8(uint8_t x, uint8_t y)   { return x & y; }
SCALAR_FUN_ATTR uint16_t and16(uint16_t x, uint16_t y) { return x & y; }
SCALAR_FUN_ATTR uint32_t and32(uint32_t x, uint32_t y) { return x & y; }
SCALAR_FUN_ATTR uint64_t and64(uint64_t x, uint64_t y) { return x & y; }

SCALAR_FUN_ATTR uint8_t    or8(uint8_t x, uint8_t y)  { return x | y; }
SCALAR_FUN_ATTR uint16_t or16(uint16_t x, uint16_t y) { return x | y; }
SCALAR_FUN_ATTR uint32_t or32(uint32_t x, uint32_t y) { return x | y; }
SCALAR_FUN_ATTR uint64_t or64(uint64_t x, uint64_t y) { return x | y; }

SCALAR_FUN_ATTR uint8_t   xor8(uint8_t x, uint8_t y)   { return x ^ y; }
SCALAR_FUN_ATTR uint16_t xor16(uint16_t x, uint16_t y) { return x ^ y; }
SCALAR_FUN_ATTR uint32_t xor32(uint32_t x, uint32_t y) { return x ^ y; }
SCALAR_FUN_ATTR uint64_t xor64(uint64_t x, uint64_t y) { return x ^ y; }

SCALAR_FUN_ATTR bool ult8(uint8_t x, uint8_t y)    { return x < y; }
SCALAR_FUN_ATTR bool ult16(uint16_t x, uint16_t y) { return x < y; }
SCALAR_FUN_ATTR bool ult32(uint32_t x, uint32_t y) { return x < y; }
SCALAR_FUN_ATTR bool ult64(uint64_t x, uint64_t y) { return x < y; }

SCALAR_FUN_ATTR bool ule8(uint8_t x, uint8_t y)    { return x <= y; }
SCALAR_FUN_ATTR bool ule16(uint16_t x, uint16_t y) { return x <= y; }
SCALAR_FUN_ATTR bool ule32(uint32_t x, uint32_t y) { return x <= y; }
SCALAR_FUN_ATTR bool ule64(uint64_t x, uint64_t y) { return x <= y; }

SCALAR_FUN_ATTR bool  slt8(int8_t x, int8_t y)   { return x < y; }
SCALAR_FUN_ATTR bool slt16(int16_t x, int16_t y) { return x < y; }
SCALAR_FUN_ATTR bool slt32(int32_t x, int32_t y) { return x < y; }
SCALAR_FUN_ATTR bool slt64(int64_t x, int64_t y) { return x < y; }

SCALAR_FUN_ATTR bool  sle8(int8_t x, int8_t y)   { return x <= y; }
SCALAR_FUN_ATTR bool sle16(int16_t x, int16_t y) { return x <= y; }
SCALAR_FUN_ATTR bool sle32(int32_t x, int32_t y) { return x <= y; }
SCALAR_FUN_ATTR bool sle64(int64_t x, int64_t y) { return x <= y; }

SCALAR_FUN_ATTR uint8_t pow8(uint8_t x, uint8_t y) {
  uint8_t res = 1, rem = y;
  while (rem != 0) {
    if (rem & 1)
      res *= x;
    rem >>= 1;
    x *= x;
  }
  return res;
}

SCALAR_FUN_ATTR uint16_t pow16(uint16_t x, uint16_t y) {
  uint16_t res = 1, rem = y;
  while (rem != 0) {
    if (rem & 1)
      res *= x;
    rem >>= 1;
    x *= x;
  }
  return res;
}

SCALAR_FUN_ATTR uint32_t pow32(uint32_t x, uint32_t y) {
  uint32_t res = 1, rem = y;
  while (rem != 0) {
    if (rem & 1)
      res *= x;
    rem >>= 1;
    x *= x;
  }
  return res;
}

SCALAR_FUN_ATTR uint64_t pow64(uint64_t x, uint64_t y) {
  uint64_t res = 1, rem = y;
  while (rem != 0) {
    if (rem & 1)
      res *= x;
    rem >>= 1;
    x *= x;
  }
  return res;
}

SCALAR_FUN_ATTR bool  itob_i8_bool(int8_t x)  { return x != 0; }
SCALAR_FUN_ATTR bool itob_i16_bool(int16_t x) { return x != 0; }
SCALAR_FUN_ATTR bool itob_i32_bool(int32_t x) { return x != 0; }
SCALAR_FUN_ATTR bool itob_i64_bool(int64_t x) { return x != 0; }

SCALAR_FUN_ATTR int8_t btoi_bool_i8(bool x)   { return x; }
SCALAR_FUN_ATTR int16_t btoi_bool_i16(bool x) { return x; }
SCALAR_FUN_ATTR int32_t btoi_bool_i32(bool x) { return x; }
SCALAR_FUN_ATTR int64_t btoi_bool_i64(bool x) { return x; }

#define sext_i8_i8(x) ((int8_t) (int8_t) (x))
#define sext_i8_i16(x) ((int16_t) (int8_t) (x))
#define sext_i8_i32(x) ((int32_t) (int8_t) (x))
#define sext_i8_i64(x) ((int64_t) (int8_t) (x))
#define sext_i16_i8(x) ((int8_t) (int16_t) (x))
#define sext_i16_i16(x) ((int16_t) (int16_t) (x))
#define sext_i16_i32(x) ((int32_t) (int16_t) (x))
#define sext_i16_i64(x) ((int64_t) (int16_t) (x))
#define sext_i32_i8(x) ((int8_t) (int32_t) (x))
#define sext_i32_i16(x) ((int16_t) (int32_t) (x))
#define sext_i32_i32(x) ((int32_t) (int32_t) (x))
#define sext_i32_i64(x) ((int64_t) (int32_t) (x))
#define sext_i64_i8(x) ((int8_t) (int64_t) (x))
#define sext_i64_i16(x) ((int16_t) (int64_t) (x))
#define sext_i64_i32(x) ((int32_t) (int64_t) (x))
#define sext_i64_i64(x) ((int64_t) (int64_t) (x))
#define zext_i8_i8(x) ((int8_t) (uint8_t) (x))
#define zext_i8_i16(x) ((int16_t) (uint8_t) (x))
#define zext_i8_i32(x) ((int32_t) (uint8_t) (x))
#define zext_i8_i64(x) ((int64_t) (uint8_t) (x))
#define zext_i16_i8(x) ((int8_t) (uint16_t) (x))
#define zext_i16_i16(x) ((int16_t) (uint16_t) (x))
#define zext_i16_i32(x) ((int32_t) (uint16_t) (x))
#define zext_i16_i64(x) ((int64_t) (uint16_t) (x))
#define zext_i32_i8(x) ((int8_t) (uint32_t) (x))
#define zext_i32_i16(x) ((int16_t) (uint32_t) (x))
#define zext_i32_i32(x) ((int32_t) (uint32_t) (x))
#define zext_i32_i64(x) ((int64_t) (uint32_t) (x))
#define zext_i64_i8(x) ((int8_t) (uint64_t) (x))
#define zext_i64_i16(x) ((int16_t) (uint64_t) (x))
#define zext_i64_i32(x) ((int32_t) (uint64_t) (x))
#define zext_i64_i64(x) ((int64_t) (uint64_t) (x))

SCALAR_FUN_ATTR int8_t   abs8(int8_t x)  { return (int8_t)abs(x); }
SCALAR_FUN_ATTR int16_t abs16(int16_t x) { return (int16_t)abs(x); }
SCALAR_FUN_ATTR int32_t abs32(int32_t x) { return abs(x); }
SCALAR_FUN_ATTR int64_t abs64(int64_t x) {
#if defined(__OPENCL_VERSION__) || defined(ISPC)
  return abs(x);
#else
  return llabs(x);
#endif
}

#if defined(__OPENCL_VERSION__)

SCALAR_FUN_ATTR int32_t  futrts_popc8(int8_t x)  { return popcount(x); }
SCALAR_FUN_ATTR int32_t futrts_popc16(int16_t x) { return popcount(x); }
SCALAR_FUN_ATTR int32_t futrts_popc32(int32_t x) { return popcount(x); }
SCALAR_FUN_ATTR int32_t futrts_popc64(int64_t x) { return popcount(x); }

#elif defined(__CUDA_ARCH__)

SCALAR_FUN_ATTR int32_t  futrts_popc8(int8_t x)  { return __popc(zext_i8_i32(x)); }
SCALAR_FUN_ATTR int32_t futrts_popc16(int16_t x) { return __popc(zext_i16_i32(x)); }
SCALAR_FUN_ATTR int32_t futrts_popc32(int32_t x) { return __popc(x); }
SCALAR_FUN_ATTR int32_t futrts_popc64(int64_t x) { return __popcll(x); }

#else // Not OpenCL or CUDA, but plain C.

SCALAR_FUN_ATTR int32_t futrts_popc8(uint8_t x) {
  int c = 0;
  for (; x; ++c) { x &= x - 1; }
  return c;
}

SCALAR_FUN_ATTR int32_t futrts_popc16(uint16_t x) {
  int c = 0;
  for (; x; ++c) { x &= x - 1; }
  return c;
}

SCALAR_FUN_ATTR int32_t futrts_popc32(uint32_t x) {
  int c = 0;
  for (; x; ++c) { x &= x - 1; }
  return c;
}

SCALAR_FUN_ATTR int32_t futrts_popc64(uint64_t x) {
  int c = 0;
  for (; x; ++c) { x &= x - 1; }
  return c;
}
#endif

#if defined(__OPENCL_VERSION__)
SCALAR_FUN_ATTR uint8_t  futrts_umul_hi8 ( uint8_t a,  uint8_t b) { return mul_hi(a, b); }
SCALAR_FUN_ATTR uint16_t futrts_umul_hi16(uint16_t a, uint16_t b) { return mul_hi(a, b); }
SCALAR_FUN_ATTR uint32_t futrts_umul_hi32(uint32_t a, uint32_t b) { return mul_hi(a, b); }
SCALAR_FUN_ATTR uint64_t futrts_umul_hi64(uint64_t a, uint64_t b) { return mul_hi(a, b); }
SCALAR_FUN_ATTR uint8_t  futrts_smul_hi8 ( int8_t a,  int8_t b) { return mul_hi(a, b); }
SCALAR_FUN_ATTR uint16_t futrts_smul_hi16(int16_t a, int16_t b) { return mul_hi(a, b); }
SCALAR_FUN_ATTR uint32_t futrts_smul_hi32(int32_t a, int32_t b) { return mul_hi(a, b); }
SCALAR_FUN_ATTR uint64_t futrts_smul_hi64(int64_t a, int64_t b) { return mul_hi(a, b); }
#elif defined(__CUDA_ARCH__)
SCALAR_FUN_ATTR  uint8_t futrts_umul_hi8(uint8_t a, uint8_t b) { return ((uint16_t)a) * ((uint16_t)b) >> 8; }
SCALAR_FUN_ATTR uint16_t futrts_umul_hi16(uint16_t a, uint16_t b) { return ((uint32_t)a) * ((uint32_t)b) >> 16; }
SCALAR_FUN_ATTR uint32_t futrts_umul_hi32(uint32_t a, uint32_t b) { return __umulhi(a, b); }
SCALAR_FUN_ATTR uint64_t futrts_umul_hi64(uint64_t a, uint64_t b) { return __umul64hi(a, b); }
SCALAR_FUN_ATTR  uint8_t futrts_smul_hi8 ( int8_t a, int8_t b) { return ((int16_t)a) * ((int16_t)b) >> 8; }
SCALAR_FUN_ATTR uint16_t futrts_smul_hi16(int16_t a, int16_t b) { return ((int32_t)a) * ((int32_t)b) >> 16; }
SCALAR_FUN_ATTR uint32_t futrts_smul_hi32(int32_t a, int32_t b) { return __mulhi(a, b); }
SCALAR_FUN_ATTR uint64_t futrts_smul_hi64(int64_t a, int64_t b) { return __mul64hi(a, b); }
#elif defined(ISPC)
SCALAR_FUN_ATTR uint8_t futrts_umul_hi8(uint8_t a, uint8_t b) { return ((uint16_t)a) * ((uint16_t)b) >> 8; }
SCALAR_FUN_ATTR uint16_t futrts_umul_hi16(uint16_t a, uint16_t b) { return ((uint32_t)a) * ((uint32_t)b) >> 16; }
SCALAR_FUN_ATTR uint32_t futrts_umul_hi32(uint32_t a, uint32_t b) { return ((uint64_t)a) * ((uint64_t)b) >> 32; }
SCALAR_FUN_ATTR uint64_t futrts_umul_hi64(uint64_t a, uint64_t b) {
  uint64_t ah = a >> 32;
  uint64_t al = a & 0xffffffff;
  uint64_t bh = b >> 32;
  uint64_t bl = b & 0xffffffff;

  uint64_t p1 = al * bl;
  uint64_t p2 = al * bh;
  uint64_t p3 = ah * bl;
  uint64_t p4 = ah * bh;

  uint64_t p1h = p1 >> 32;
  uint64_t p2h = p2 >> 32;
  uint64_t p3h = p3 >> 32;
  uint64_t p2l = p2 & 0xffffffff;
  uint64_t p3l = p3 & 0xffffffff;

  uint64_t l = p1h + p2l + p3l;
  uint64_t m = (p2 >> 32) + (p3 >> 32);
  uint64_t h = (l >> 32) + m + p4;

  return h;
}
SCALAR_FUN_ATTR  int8_t futrts_smul_hi8 ( int8_t a,  int8_t b) { return ((uint16_t)a) * ((uint16_t)b) >> 8; }
SCALAR_FUN_ATTR int16_t futrts_smul_hi16(int16_t a, int16_t b) { return ((uint32_t)a) * ((uint32_t)b) >> 16; }
SCALAR_FUN_ATTR int32_t futrts_smul_hi32(int32_t a, int32_t b) { return ((uint64_t)a) * ((uint64_t)b) >> 32; }
SCALAR_FUN_ATTR int64_t futrts_smul_hi64(int64_t a, int64_t b) {
  uint64_t ah = a >> 32;
  uint64_t al = a & 0xffffffff;
  uint64_t bh = b >> 32;
  uint64_t bl = b & 0xffffffff;

  uint64_t p1 =  al * bl;
  int64_t  p2 = al * bh;
  int64_t  p3 = ah * bl;
  uint64_t p4 =  ah * bh;

  uint64_t p1h = p1 >> 32;
  uint64_t p2h = p2 >> 32;
  uint64_t p3h = p3 >> 32;
  uint64_t p2l = p2 & 0xffffffff;
  uint64_t p3l = p3 & 0xffffffff;

  uint64_t l = p1h + p2l + p3l;
  uint64_t m = (p2 >> 32) + (p3 >> 32);
  uint64_t h = (l >> 32) + m + p4;

  return h;
}

#else // Not OpenCL, ISPC, or CUDA, but plain C.
SCALAR_FUN_ATTR uint8_t futrts_umul_hi8(uint8_t a, uint8_t b) { return ((uint16_t)a) * ((uint16_t)b) >> 8; }
SCALAR_FUN_ATTR uint16_t futrts_umul_hi16(uint16_t a, uint16_t b) { return ((uint32_t)a) * ((uint32_t)b) >> 16; }
SCALAR_FUN_ATTR uint32_t futrts_umul_hi32(uint32_t a, uint32_t b) { return ((uint64_t)a) * ((uint64_t)b) >> 32; }
SCALAR_FUN_ATTR uint64_t futrts_umul_hi64(uint64_t a, uint64_t b) { return ((__uint128_t)a) * ((__uint128_t)b) >> 64; }
SCALAR_FUN_ATTR int8_t futrts_smul_hi8(int8_t a, int8_t b) { return ((int16_t)a) * ((int16_t)b) >> 8; }
SCALAR_FUN_ATTR int16_t futrts_smul_hi16(int16_t a, int16_t b) { return ((int32_t)a) * ((int32_t)b) >> 16; }
SCALAR_FUN_ATTR int32_t futrts_smul_hi32(int32_t a, int32_t b) { return ((int64_t)a) * ((int64_t)b) >> 32; }
SCALAR_FUN_ATTR int64_t futrts_smul_hi64(int64_t a, int64_t b) { return ((__int128_t)a) * ((__int128_t)b) >> 64; }
#endif

#if defined(__OPENCL_VERSION__)
SCALAR_FUN_ATTR  uint8_t futrts_umad_hi8 ( uint8_t a,  uint8_t b,  uint8_t c) { return mad_hi(a, b, c); }
SCALAR_FUN_ATTR uint16_t futrts_umad_hi16(uint16_t a, uint16_t b, uint16_t c) { return mad_hi(a, b, c); }
SCALAR_FUN_ATTR uint32_t futrts_umad_hi32(uint32_t a, uint32_t b, uint32_t c) { return mad_hi(a, b, c); }
SCALAR_FUN_ATTR uint64_t futrts_umad_hi64(uint64_t a, uint64_t b, uint64_t c) { return mad_hi(a, b, c); }
SCALAR_FUN_ATTR  uint8_t futrts_smad_hi8( int8_t a,  int8_t b,   int8_t c) { return mad_hi(a, b, c); }
SCALAR_FUN_ATTR uint16_t futrts_smad_hi16(int16_t a, int16_t b, int16_t c) { return mad_hi(a, b, c); }
SCALAR_FUN_ATTR uint32_t futrts_smad_hi32(int32_t a, int32_t b, int32_t c) { return mad_hi(a, b, c); }
SCALAR_FUN_ATTR uint64_t futrts_smad_hi64(int64_t a, int64_t b, int64_t c) { return mad_hi(a, b, c); }
#else // Not OpenCL

SCALAR_FUN_ATTR  uint8_t futrts_umad_hi8( uint8_t a,  uint8_t b,  uint8_t c) { return futrts_umul_hi8(a, b) + c; }
SCALAR_FUN_ATTR uint16_t futrts_umad_hi16(uint16_t a, uint16_t b, uint16_t c) { return futrts_umul_hi16(a, b) + c; }
SCALAR_FUN_ATTR uint32_t futrts_umad_hi32(uint32_t a, uint32_t b, uint32_t c) { return futrts_umul_hi32(a, b) + c; }
SCALAR_FUN_ATTR uint64_t futrts_umad_hi64(uint64_t a, uint64_t b, uint64_t c) { return futrts_umul_hi64(a, b) + c; }
SCALAR_FUN_ATTR  uint8_t futrts_smad_hi8 ( int8_t a,  int8_t b,  int8_t c) { return futrts_smul_hi8(a, b) + c; }
SCALAR_FUN_ATTR uint16_t futrts_smad_hi16(int16_t a, int16_t b, int16_t c) { return futrts_smul_hi16(a, b) + c; }
SCALAR_FUN_ATTR uint32_t futrts_smad_hi32(int32_t a, int32_t b, int32_t c) { return futrts_smul_hi32(a, b) + c; }
SCALAR_FUN_ATTR uint64_t futrts_smad_hi64(int64_t a, int64_t b, int64_t c) { return futrts_smul_hi64(a, b) + c; }
#endif

#if defined(__OPENCL_VERSION__)
SCALAR_FUN_ATTR int32_t  futrts_clzz8(int8_t x)  { return clz(x); }
SCALAR_FUN_ATTR int32_t futrts_clzz16(int16_t x) { return clz(x); }
SCALAR_FUN_ATTR int32_t futrts_clzz32(int32_t x) { return clz(x); }
SCALAR_FUN_ATTR int32_t futrts_clzz64(int64_t x) { return clz(x); }

#elif defined(__CUDA_ARCH__)

SCALAR_FUN_ATTR int32_t  futrts_clzz8(int8_t x)  { return __clz(zext_i8_i32(x)) - 24; }
SCALAR_FUN_ATTR int32_t futrts_clzz16(int16_t x) { return __clz(zext_i16_i32(x)) - 16; }
SCALAR_FUN_ATTR int32_t futrts_clzz32(int32_t x) { return __clz(x); }
SCALAR_FUN_ATTR int32_t futrts_clzz64(int64_t x) { return __clzll(x); }

#elif defined(ISPC)

SCALAR_FUN_ATTR int32_t  futrts_clzz8(int8_t x)  { return count_leading_zeros((int32_t)(uint8_t)x)-24; }
SCALAR_FUN_ATTR int32_t futrts_clzz16(int16_t x) { return count_leading_zeros((int32_t)(uint16_t)x)-16; }
SCALAR_FUN_ATTR int32_t futrts_clzz32(int32_t x) { return count_leading_zeros(x); }
SCALAR_FUN_ATTR int32_t futrts_clzz64(int64_t x) { return count_leading_zeros(x); }

#else // Not OpenCL, ISPC or CUDA, but plain C.

SCALAR_FUN_ATTR int32_t futrts_clzz8(int8_t x)
{ return x == 0 ? 8 : __builtin_clz((uint32_t)zext_i8_i32(x)) - 24; }
SCALAR_FUN_ATTR int32_t futrts_clzz16(int16_t x)
{ return x == 0 ? 16 : __builtin_clz((uint32_t)zext_i16_i32(x)) - 16; }
SCALAR_FUN_ATTR int32_t futrts_clzz32(int32_t x)
{ return x == 0 ? 32 : __builtin_clz((uint32_t)x); }
SCALAR_FUN_ATTR int32_t futrts_clzz64(int64_t x)
{ return x == 0 ? 64 : __builtin_clzll((uint64_t)x); }
#endif

#if defined(__OPENCL_VERSION__)
SCALAR_FUN_ATTR int32_t futrts_ctzz8(int8_t x) {
  int i = 0;
  for (; i < 8 && (x & 1) == 0; i++, x >>= 1) ;
  return i;
}

SCALAR_FUN_ATTR int32_t futrts_ctzz16(int16_t x) {
  int i = 0;
  for (; i < 16 && (x & 1) == 0; i++, x >>= 1) ;
  return i;
}

SCALAR_FUN_ATTR int32_t futrts_ctzz32(int32_t x) {
  int i = 0;
  for (; i < 32 && (x & 1) == 0; i++, x >>= 1) ;
  return i;
}

SCALAR_FUN_ATTR int32_t futrts_ctzz64(int64_t x) {
  int i = 0;
  for (; i < 64 && (x & 1) == 0; i++, x >>= 1) ;
  return i;
}

#elif defined(__CUDA_ARCH__)

SCALAR_FUN_ATTR int32_t futrts_ctzz8(int8_t x) {
  int y = __ffs(x);
  return y == 0 ? 8 : y - 1;
}

SCALAR_FUN_ATTR int32_t futrts_ctzz16(int16_t x) {
  int y = __ffs(x);
  return y == 0 ? 16 : y - 1;
}

SCALAR_FUN_ATTR int32_t futrts_ctzz32(int32_t x) {
  int y = __ffs(x);
  return y == 0 ? 32 : y - 1;
}

SCALAR_FUN_ATTR int32_t futrts_ctzz64(int64_t x) {
  int y = __ffsll(x);
  return y == 0 ? 64 : y - 1;
}

#elif defined(ISPC)

SCALAR_FUN_ATTR int32_t futrts_ctzz8(int8_t x) { return x == 0 ? 8 : count_trailing_zeros((int32_t)x); }
SCALAR_FUN_ATTR int32_t futrts_ctzz16(int16_t x) { return x == 0 ? 16 : count_trailing_zeros((int32_t)x); }
SCALAR_FUN_ATTR int32_t futrts_ctzz32(int32_t x) { return count_trailing_zeros(x); }
SCALAR_FUN_ATTR int32_t futrts_ctzz64(int64_t x) { return count_trailing_zeros(x); }

#else // Not OpenCL or CUDA, but plain C.

SCALAR_FUN_ATTR int32_t  futrts_ctzz8(int8_t x)  { return x == 0 ? 8 : __builtin_ctz((uint32_t)x); }
SCALAR_FUN_ATTR int32_t futrts_ctzz16(int16_t x) { return x == 0 ? 16 : __builtin_ctz((uint32_t)x); }
SCALAR_FUN_ATTR int32_t futrts_ctzz32(int32_t x) { return x == 0 ? 32 : __builtin_ctz((uint32_t)x); }
SCALAR_FUN_ATTR int32_t futrts_ctzz64(int64_t x) { return x == 0 ? 64 : __builtin_ctzll((uint64_t)x); }
#endif

SCALAR_FUN_ATTR float fdiv32(float x, float y) { return x / y; }
SCALAR_FUN_ATTR float fadd32(float x, float y) { return x + y; }
SCALAR_FUN_ATTR float fsub32(float x, float y) { return x - y; }
SCALAR_FUN_ATTR float fmul32(float x, float y) { return x * y; }
SCALAR_FUN_ATTR bool cmplt32(float x, float y) { return x < y; }
SCALAR_FUN_ATTR bool cmple32(float x, float y) { return x <= y; }
SCALAR_FUN_ATTR float sitofp_i8_f32(int8_t x)  { return (float) x; }

SCALAR_FUN_ATTR float sitofp_i16_f32(int16_t x) { return (float) x; }
SCALAR_FUN_ATTR float sitofp_i32_f32(int32_t x) { return (float) x; }
SCALAR_FUN_ATTR float sitofp_i64_f32(int64_t x) { return (float) x; }
SCALAR_FUN_ATTR float  uitofp_i8_f32(uint8_t x)  { return (float) x; }
SCALAR_FUN_ATTR float uitofp_i16_f32(uint16_t x) { return (float) x; }
SCALAR_FUN_ATTR float uitofp_i32_f32(uint32_t x) { return (float) x; }
SCALAR_FUN_ATTR float uitofp_i64_f32(uint64_t x) { return (float) x; }

#ifdef __OPENCL_VERSION__
SCALAR_FUN_ATTR float fabs32(float x)          { return fabs(x); }
SCALAR_FUN_ATTR float fmax32(float x, float y) { return fmax(x, y); }
SCALAR_FUN_ATTR float fmin32(float x, float y) { return fmin(x, y); }
SCALAR_FUN_ATTR float fpow32(float x, float y) { return pow(x, y); }

#elif defined(ISPC)

SCALAR_FUN_ATTR float fabs32(float x) { return abs(x); }
SCALAR_FUN_ATTR float fmax32(float x, float y) { return isnan(x) ? y : isnan(y) ? x : max(x, y); }
SCALAR_FUN_ATTR float fmin32(float x, float y) { return isnan(x) ? y : isnan(y) ? x : min(x, y); }
SCALAR_FUN_ATTR float fpow32(float a, float b) {
  float ret;
  foreach_active (i) {
      uniform float r = pow(extract(a, i), extract(b, i));
      ret = insert(ret, i, r);
  }
  return ret;
}

#else // Not OpenCL, but CUDA or plain C.

SCALAR_FUN_ATTR float fabs32(float x)          { return fabsf(x); }
SCALAR_FUN_ATTR float fmax32(float x, float y) { return fmaxf(x, y); }
SCALAR_FUN_ATTR float fmin32(float x, float y) { return fminf(x, y); }
SCALAR_FUN_ATTR float fpow32(float x, float y) { return powf(x, y); }
#endif

SCALAR_FUN_ATTR bool futrts_isnan32(float x) { return isnan(x); }

#if defined(ISPC)

SCALAR_FUN_ATTR bool futrts_isinf32(float x) { return !isnan(x) && isnan(x - x); }

SCALAR_FUN_ATTR bool futrts_isfinite32(float x) { return !isnan(x) && !futrts_isinf32(x); }

#else

SCALAR_FUN_ATTR bool futrts_isinf32(float x) { return isinf(x); }

#endif

SCALAR_FUN_ATTR int8_t fptosi_f32_i8(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (int8_t) x;
  }
}

SCALAR_FUN_ATTR int16_t fptosi_f32_i16(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (int16_t) x;
  }
}

SCALAR_FUN_ATTR int32_t fptosi_f32_i32(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (int32_t) x;
  }
}

SCALAR_FUN_ATTR int64_t fptosi_f32_i64(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (int64_t) x;
  };
}

SCALAR_FUN_ATTR uint8_t fptoui_f32_i8(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (uint8_t) (int8_t) x;
  }
}

SCALAR_FUN_ATTR uint16_t fptoui_f32_i16(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (uint16_t) (int16_t) x;
  }
}

SCALAR_FUN_ATTR uint32_t fptoui_f32_i32(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (uint32_t) (int32_t) x;
  }
}

SCALAR_FUN_ATTR uint64_t fptoui_f32_i64(float x) {
  if (futrts_isnan32(x) || futrts_isinf32(x)) {
    return 0;
  } else {
    return (uint64_t) (int64_t) x;
  }
}

SCALAR_FUN_ATTR bool ftob_f32_bool(float x) { return x != 0; }
SCALAR_FUN_ATTR float btof_bool_f32(bool x) { return x ? 1 : 0; }

#ifdef __OPENCL_VERSION__
SCALAR_FUN_ATTR float futrts_log32(float x) { return log(x); }
SCALAR_FUN_ATTR float futrts_log2_32(float x) { return log2(x); }
SCALAR_FUN_ATTR float futrts_log10_32(float x) { return log10(x); }
SCALAR_FUN_ATTR float futrts_log1p_32(float x) { return log1p(x); }
SCALAR_FUN_ATTR float futrts_sqrt32(float x) { return sqrt(x); }
SCALAR_FUN_ATTR float futrts_rsqrt32(float x) { return rsqrt(x); }
SCALAR_FUN_ATTR float futrts_cbrt32(float x) { return cbrt(x); }
SCALAR_FUN_ATTR float futrts_exp32(float x) { return exp(x); }
SCALAR_FUN_ATTR float futrts_cos32(float x) { return cos(x); }
SCALAR_FUN_ATTR float futrts_cospi32(float x) { return cospi(x); }
SCALAR_FUN_ATTR float futrts_sin32(float x) { return sin(x); }
SCALAR_FUN_ATTR float futrts_sinpi32(float x) { return sinpi(x); }
SCALAR_FUN_ATTR float futrts_tan32(float x) { return tan(x); }
SCALAR_FUN_ATTR float futrts_tanpi32(float x) { return tanpi(x); }
SCALAR_FUN_ATTR float futrts_acos32(float x) { return acos(x); }
SCALAR_FUN_ATTR float futrts_acospi32(float x) { return acospi(x); }
SCALAR_FUN_ATTR float futrts_asin32(float x) { return asin(x); }
SCALAR_FUN_ATTR float futrts_asinpi32(float x) { return asinpi(x); }
SCALAR_FUN_ATTR float futrts_atan32(float x) { return atan(x); }
SCALAR_FUN_ATTR float futrts_atanpi32(float x) { return atanpi(x); }
SCALAR_FUN_ATTR float futrts_cosh32(float x) { return cosh(x); }
SCALAR_FUN_ATTR float futrts_sinh32(float x) { return sinh(x); }
SCALAR_FUN_ATTR float futrts_tanh32(float x) { return tanh(x); }
SCALAR_FUN_ATTR float futrts_acosh32(float x) { return acosh(x); }
SCALAR_FUN_ATTR float futrts_asinh32(float x) { return asinh(x); }
SCALAR_FUN_ATTR float futrts_atanh32(float x) { return atanh(x); }
SCALAR_FUN_ATTR float futrts_atan2_32(float x, float y) { return atan2(x, y); }
SCALAR_FUN_ATTR float futrts_atan2pi_32(float x, float y) { return atan2pi(x, y); }
SCALAR_FUN_ATTR float futrts_hypot32(float x, float y) { return hypot(x, y); }
SCALAR_FUN_ATTR float futrts_gamma32(float x) { return tgamma(x); }
SCALAR_FUN_ATTR float futrts_lgamma32(float x) { return lgamma(x); }
SCALAR_FUN_ATTR float futrts_erf32(float x) { return erf(x); }
SCALAR_FUN_ATTR float futrts_erfc32(float x) { return erfc(x); }
SCALAR_FUN_ATTR float fmod32(float x, float y) { return fmod(x, y); }
SCALAR_FUN_ATTR float futrts_round32(float x) { return rint(x); }
SCALAR_FUN_ATTR float futrts_floor32(float x) { return floor(x); }
SCALAR_FUN_ATTR float futrts_ceil32(float x) { return ceil(x); }
SCALAR_FUN_ATTR float futrts_nextafter32(float x, float y) { return nextafter(x, y); }
SCALAR_FUN_ATTR float futrts_lerp32(float v0, float v1, float t) { return mix(v0, v1, t); }
SCALAR_FUN_ATTR float futrts_ldexp32(float x, int32_t y) { return ldexp(x, y); }
SCALAR_FUN_ATTR float futrts_copysign32(float x, float y) { return copysign(x, y); }
SCALAR_FUN_ATTR float futrts_mad32(float a, float b, float c) { return mad(a, b, c); }
SCALAR_FUN_ATTR float futrts_fma32(float a, float b, float c) { return fma(a, b, c); }

#elif defined(ISPC)

SCALAR_FUN_ATTR float futrts_log32(float x) { return futrts_isfinite32(x) || (futrts_isinf32(x) && x < 0)? log(x) : x; }
SCALAR_FUN_ATTR float futrts_log2_32(float x) { return futrts_log32(x) / log(2.0f); }
SCALAR_FUN_ATTR float futrts_log10_32(float x) { return futrts_log32(x) / log(10.0f); }

SCALAR_FUN_ATTR float futrts_log1p_32(float x) {
  if(x == -1.0f || (futrts_isinf32(x) && x > 0.0f)) return x / 0.0f;
  float y = 1.0f + x;
  float z = y - 1.0f;
  return log(y) - (z-x)/y;
}

SCALAR_FUN_ATTR float futrts_sqrt32(float x) { return sqrt(x); }
SCALAR_FUN_ATTR float futrts_rsqrt32(float x) { return 1/sqrt(x); }

extern "C" unmasked uniform float cbrtf(uniform float);
SCALAR_FUN_ATTR float futrts_cbrt32(float x) {
  float res;
  foreach_active (i) {
    uniform float r = cbrtf(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

SCALAR_FUN_ATTR float futrts_exp32(float x) { return exp(x); }
SCALAR_FUN_ATTR float futrts_cos32(float x) { return cos(x); }
SCALAR_FUN_ATTR float futrts_cospi32(float x) { return cos((float)M_PI*x); }
SCALAR_FUN_ATTR float futrts_sin32(float x) { return sin(x); }
SCALAR_FUN_ATTR float futrts_sinpi32(float x) { return sin(M_PI*x); }
SCALAR_FUN_ATTR float futrts_tan32(float x) { return tan(x); }
SCALAR_FUN_ATTR float futrts_tanpi32(float x) { return tan((float)M_PI*x); }
SCALAR_FUN_ATTR float futrts_acos32(float x) { return acos(x); }
SCALAR_FUN_ATTR float futrts_acospi32(float x) { return acos(x)/(float)M_PI; }
SCALAR_FUN_ATTR float futrts_asin32(float x) { return asin(x); }
SCALAR_FUN_ATTR float futrts_asinpi32(float x) { return asin(x)/(float)M_PI; }
SCALAR_FUN_ATTR float futrts_atan32(float x) { return atan(x); }
SCALAR_FUN_ATTR float futrts_atanpi32(float x) { return atan(x)/(float)M_PI; }
SCALAR_FUN_ATTR float futrts_cosh32(float x) { return (exp(x)+exp(-x)) / 2.0f; }
SCALAR_FUN_ATTR float futrts_sinh32(float x) { return (exp(x)-exp(-x)) / 2.0f; }
SCALAR_FUN_ATTR float futrts_tanh32(float x) { return futrts_sinh32(x)/futrts_cosh32(x); }

SCALAR_FUN_ATTR float futrts_acosh32(float x) {
  float f = x+sqrt(x*x-1);
  if (futrts_isfinite32(f)) return log(f);
  return f;
}

SCALAR_FUN_ATTR float futrts_asinh32(float x) {
  float f = x+sqrt(x*x+1);
  if (futrts_isfinite32(f)) return log(f);
  return f;
}

SCALAR_FUN_ATTR float futrts_atanh32(float x) {
  float f = (1+x)/(1-x);
  if (futrts_isfinite32(f)) return log(f)/2.0f;
  return f;
}

SCALAR_FUN_ATTR float futrts_atan2_32(float x, float y)
{ return (x == 0.0f && y == 0.0f) ? 0.0f : atan2(x, y); }
SCALAR_FUN_ATTR float futrts_atan2pi_32(float x, float y)
{ return (x == 0.0f && y == 0.0f) ? 0.0f : atan2(x, y) / (float)M_PI; }

SCALAR_FUN_ATTR float futrts_hypot32(float x, float y) {
  if (futrts_isfinite32(x) && futrts_isfinite32(y)) {
    x = abs(x);
    y = abs(y);
    float a;
    float b;
    if (x >= y){
        a = x;
        b = y;
    } else {
        a = y;
        b = x;
    }
    if(b == 0){
      return a;
    }

    int e;
    float an;
    float bn;
    an = frexp (a, &e);
    bn = ldexp (b, - e);
    float cn;
    cn = sqrt (an * an + bn * bn);
    return ldexp (cn, e);
  } else {
    if (futrts_isinf32(x) || futrts_isinf32(y)) return INFINITY;
    else return x + y;
  }

}

extern "C" unmasked uniform float tgammaf(uniform float x);
SCALAR_FUN_ATTR float futrts_gamma32(float x) {
  float res;
  foreach_active (i) {
    uniform float r = tgammaf(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform float lgammaf(uniform float x);
SCALAR_FUN_ATTR float futrts_lgamma32(float x) {
  float res;
  foreach_active (i) {
    uniform float r = lgammaf(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform float erff(uniform float x);
SCALAR_FUN_ATTR float futrts_erf32(float x) {
  float res;
  foreach_active (i) {
    uniform float r = erff(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform float erfcf(uniform float x);
SCALAR_FUN_ATTR float futrts_erfc32(float x) {
  float res;
  foreach_active (i) {
    uniform float r = erfcf(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

SCALAR_FUN_ATTR float fmod32(float x, float y) { return x - y * trunc(x/y); }
SCALAR_FUN_ATTR float futrts_round32(float x) { return round(x); }
SCALAR_FUN_ATTR float futrts_floor32(float x) { return floor(x); }
SCALAR_FUN_ATTR float futrts_ceil32(float x) { return ceil(x); }

extern "C" unmasked uniform float nextafterf(uniform float x, uniform float y);
SCALAR_FUN_ATTR float futrts_nextafter32(float x, float y) {
  float res;
  foreach_active (i) {
    uniform float r = nextafterf(extract(x, i), extract(y, i));
    res = insert(res, i, r);
  }
  return res;
}

SCALAR_FUN_ATTR float futrts_lerp32(float v0, float v1, float t) {
  return v0 + (v1 - v0) * t;
}

SCALAR_FUN_ATTR float futrts_ldexp32(float x, int32_t y) {
  return x * pow((uniform float)2.0, (float)y);
}

SCALAR_FUN_ATTR float futrts_copysign32(float x, float y) {
  int32_t xb = fptobits_f32_i32(x);
  int32_t yb = fptobits_f32_i32(y);
  return bitstofp_i32_f32((xb & ~(1<<31)) | (yb & (1<<31)));
}

SCALAR_FUN_ATTR float futrts_mad32(float a, float b, float c) {
  return a * b + c;
}

SCALAR_FUN_ATTR float futrts_fma32(float a, float b, float c) {
  return a * b + c;
}

#else // Not OpenCL or ISPC, but CUDA or plain C.

SCALAR_FUN_ATTR float futrts_log32(float x) { return logf(x); }
SCALAR_FUN_ATTR float futrts_log2_32(float x) { return log2f(x); }
SCALAR_FUN_ATTR float futrts_log10_32(float x) { return log10f(x); }
SCALAR_FUN_ATTR float futrts_log1p_32(float x) { return log1pf(x); }
SCALAR_FUN_ATTR float futrts_sqrt32(float x) { return sqrtf(x); }
SCALAR_FUN_ATTR float futrts_rsqrt32(float x) { return 1/sqrtf(x); }
SCALAR_FUN_ATTR float futrts_cbrt32(float x) { return cbrtf(x); }
SCALAR_FUN_ATTR float futrts_exp32(float x) { return expf(x); }
SCALAR_FUN_ATTR float futrts_cos32(float x) { return cosf(x); }

SCALAR_FUN_ATTR float futrts_cospi32(float x) {
#if defined(__CUDA_ARCH__)
  return cospif(x);
#else
  return cosf(((float)M_PI)*x);
#endif
}
SCALAR_FUN_ATTR float futrts_sin32(float x) { return sinf(x); }

SCALAR_FUN_ATTR float futrts_sinpi32(float x) {
#if defined(__CUDA_ARCH__)
  return sinpif(x);
#else
  return sinf((float)M_PI*x);
#endif
}

SCALAR_FUN_ATTR float futrts_tan32(float x) { return tanf(x); }
SCALAR_FUN_ATTR float futrts_tanpi32(float x) { return tanf((float)M_PI*x); }
SCALAR_FUN_ATTR float futrts_acos32(float x) { return acosf(x); }
SCALAR_FUN_ATTR float futrts_acospi32(float x) { return acosf(x)/(float)M_PI; }
SCALAR_FUN_ATTR float futrts_asin32(float x) { return asinf(x); }
SCALAR_FUN_ATTR float futrts_asinpi32(float x) { return asinf(x)/(float)M_PI; }
SCALAR_FUN_ATTR float futrts_atan32(float x) { return atanf(x); }
SCALAR_FUN_ATTR float futrts_atanpi32(float x) { return atanf(x)/(float)M_PI; }
SCALAR_FUN_ATTR float futrts_cosh32(float x) { return coshf(x); }
SCALAR_FUN_ATTR float futrts_sinh32(float x) { return sinhf(x); }
SCALAR_FUN_ATTR float futrts_tanh32(float x) { return tanhf(x); }
SCALAR_FUN_ATTR float futrts_acosh32(float x) { return acoshf(x); }
SCALAR_FUN_ATTR float futrts_asinh32(float x) { return asinhf(x); }
SCALAR_FUN_ATTR float futrts_atanh32(float x) { return atanhf(x); }
SCALAR_FUN_ATTR float futrts_atan2_32(float x, float y) { return atan2f(x, y); }
SCALAR_FUN_ATTR float futrts_atan2pi_32(float x, float y) { return atan2f(x, y) / (float)M_PI; }
SCALAR_FUN_ATTR float futrts_hypot32(float x, float y) { return hypotf(x, y); }
SCALAR_FUN_ATTR float futrts_gamma32(float x) { return tgammaf(x); }
SCALAR_FUN_ATTR float futrts_lgamma32(float x) { return lgammaf(x); }
SCALAR_FUN_ATTR float futrts_erf32(float x) { return erff(x); }
SCALAR_FUN_ATTR float futrts_erfc32(float x) { return erfcf(x); }
SCALAR_FUN_ATTR float fmod32(float x, float y) { return fmodf(x, y); }
SCALAR_FUN_ATTR float futrts_round32(float x) { return rintf(x); }
SCALAR_FUN_ATTR float futrts_floor32(float x) { return floorf(x); }
SCALAR_FUN_ATTR float futrts_ceil32(float x) { return ceilf(x); }
SCALAR_FUN_ATTR float futrts_nextafter32(float x, float y) { return nextafterf(x, y); }
SCALAR_FUN_ATTR float futrts_lerp32(float v0, float v1, float t) { return v0 + (v1 - v0) * t; }
SCALAR_FUN_ATTR float futrts_ldexp32(float x, int32_t y) { return ldexpf(x, y); }
SCALAR_FUN_ATTR float futrts_copysign32(float x, float y) { return copysignf(x, y); }
SCALAR_FUN_ATTR float futrts_mad32(float a, float b, float c) { return a * b + c; }
SCALAR_FUN_ATTR float futrts_fma32(float a, float b, float c) { return fmaf(a, b, c); }

#endif

#if defined(ISPC)

SCALAR_FUN_ATTR int32_t fptobits_f32_i32(float x) { return intbits(x); }
SCALAR_FUN_ATTR float bitstofp_i32_f32(int32_t x) { return floatbits(x); }
SCALAR_FUN_ATTR uniform int32_t fptobits_f32_i32(uniform float x) { return intbits(x); }
SCALAR_FUN_ATTR uniform float bitstofp_i32_f32(uniform int32_t x) { return floatbits(x); }

#else

SCALAR_FUN_ATTR int32_t fptobits_f32_i32(float x) {
  union {
    float f;
    int32_t t;
  } p;

  p.f = x;
  return p.t;
}

SCALAR_FUN_ATTR float bitstofp_i32_f32(int32_t x) {
  union {
    int32_t f;
    float t;
  } p;

  p.f = x;
  return p.t;
}
#endif

SCALAR_FUN_ATTR float fsignum32(float x) {
  return futrts_isnan32(x) ? x : (x > 0 ? 1 : 0) - (x < 0 ? 1 : 0);
}

#ifdef FUTHARK_F64_ENABLED

SCALAR_FUN_ATTR double bitstofp_i64_f64(int64_t x);
SCALAR_FUN_ATTR int64_t fptobits_f64_i64(double x);

#if defined(ISPC)

SCALAR_FUN_ATTR bool futrts_isinf64(float x) { return !isnan(x) && isnan(x - x); }
SCALAR_FUN_ATTR bool futrts_isfinite64(float x) { return !isnan(x) && !futrts_isinf64(x); }
SCALAR_FUN_ATTR double fdiv64(double x, double y) { return x / y; }
SCALAR_FUN_ATTR double fadd64(double x, double y) { return x + y; }
SCALAR_FUN_ATTR double fsub64(double x, double y) { return x - y; }
SCALAR_FUN_ATTR double fmul64(double x, double y) { return x * y; }
SCALAR_FUN_ATTR bool cmplt64(double x, double y) { return x < y; }
SCALAR_FUN_ATTR bool cmple64(double x, double y) { return x <= y; }
SCALAR_FUN_ATTR double sitofp_i8_f64(int8_t x) { return (double) x; }
SCALAR_FUN_ATTR double sitofp_i16_f64(int16_t x) { return (double) x; }
SCALAR_FUN_ATTR double sitofp_i32_f64(int32_t x) { return (double) x; }
SCALAR_FUN_ATTR double sitofp_i64_f64(int64_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i8_f64(uint8_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i16_f64(uint16_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i32_f64(uint32_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i64_f64(uint64_t x) { return (double) x; }
SCALAR_FUN_ATTR double fabs64(double x) { return abs(x); }
SCALAR_FUN_ATTR double fmax64(double x, double y) { return isnan(x) ? y : isnan(y) ? x : max(x, y); }
SCALAR_FUN_ATTR double fmin64(double x, double y) { return isnan(x) ? y : isnan(y) ? x : min(x, y); }

SCALAR_FUN_ATTR double fpow64(double a, double b) {
  float ret;
  foreach_active (i) {
      uniform float r = pow(extract(a, i), extract(b, i));
      ret = insert(ret, i, r);
  }
  return ret;
}
SCALAR_FUN_ATTR double futrts_log64(double x) { return futrts_isfinite64(x) || (futrts_isinf64(x) && x < 0)? log(x) : x; }
SCALAR_FUN_ATTR double futrts_log2_64(double x) { return futrts_log64(x)/log(2.0d); }
SCALAR_FUN_ATTR double futrts_log10_64(double x) { return futrts_log64(x)/log(10.0d); }

SCALAR_FUN_ATTR double futrts_log1p_64(double x) {
  if(x == -1.0d || (futrts_isinf64(x) && x > 0.0d)) return x / 0.0d;
  double y = 1.0d + x;
  double z = y - 1.0d;
  return log(y) - (z-x)/y;
}

SCALAR_FUN_ATTR double futrts_sqrt64(double x) { return sqrt(x); }
SCALAR_FUN_ATTR double futrts_rsqrt64(double x) { return 1/sqrt(x); }

SCALAR_FUN_ATTR double futrts_cbrt64(double x) {
  double res;
  foreach_active (i) {
    uniform double r = cbrtf(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}
SCALAR_FUN_ATTR double futrts_exp64(double x) { return exp(x); }
SCALAR_FUN_ATTR double futrts_cos64(double x) { return cos(x); }
SCALAR_FUN_ATTR double futrts_cospi64(double x) { return cos(M_PI*x); }
SCALAR_FUN_ATTR double futrts_sin64(double x) { return sin(x); }
SCALAR_FUN_ATTR double futrts_sinpi64(double x) { return sin(M_PI*x); }
SCALAR_FUN_ATTR double futrts_tan64(double x) { return tan(x); }
SCALAR_FUN_ATTR double futrts_tanpi64(double x) { return tan(M_PI*x); }
SCALAR_FUN_ATTR double futrts_acos64(double x) { return acos(x); }
SCALAR_FUN_ATTR double futrts_acospi64(double x) { return acos(x)/M_PI; }
SCALAR_FUN_ATTR double futrts_asin64(double x) { return asin(x); }
SCALAR_FUN_ATTR double futrts_asinpi64(double x) { return asin(x)/M_PI; }
SCALAR_FUN_ATTR double futrts_atan64(double x) { return atan(x); }
SCALAR_FUN_ATTR double futrts_atanpi64(double x) { return atan(x)/M_PI; }
SCALAR_FUN_ATTR double futrts_cosh64(double x) { return (exp(x)+exp(-x)) / 2.0d; }
SCALAR_FUN_ATTR double futrts_sinh64(double x) { return (exp(x)-exp(-x)) / 2.0d; }
SCALAR_FUN_ATTR double futrts_tanh64(double x) { return futrts_sinh64(x)/futrts_cosh64(x); }

SCALAR_FUN_ATTR double futrts_acosh64(double x) {
  double f = x+sqrt(x*x-1.0d);
  if(futrts_isfinite64(f)) return log(f);
  return f;
}

SCALAR_FUN_ATTR double futrts_asinh64(double x) {
  double f = x+sqrt(x*x+1.0d);
  if(futrts_isfinite64(f)) return log(f);
  return f;
}

SCALAR_FUN_ATTR double futrts_atanh64(double x) {
  double f = (1.0d+x)/(1.0d-x);
  if(futrts_isfinite64(f)) return log(f)/2.0d;
  return f;
}
SCALAR_FUN_ATTR double futrts_atan2_64(double x, double y) { return atan2(x, y); }

SCALAR_FUN_ATTR double futrts_atan2pi_64(double x, double y) { return atan2(x, y) / M_PI; }

extern "C" unmasked uniform double hypot(uniform double x, uniform double y);
SCALAR_FUN_ATTR double futrts_hypot64(double x, double y) {
  double res;
  foreach_active (i) {
    uniform double r = hypot(extract(x, i), extract(y, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform double tgamma(uniform double x);
SCALAR_FUN_ATTR double futrts_gamma64(double x) {
  double res;
  foreach_active (i) {
    uniform double r = tgamma(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform double lgamma(uniform double x);
SCALAR_FUN_ATTR double futrts_lgamma64(double x) {
  double res;
  foreach_active (i) {
    uniform double r = lgamma(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform double erf(uniform double x);
SCALAR_FUN_ATTR double futrts_erf64(double x) {
  double res;
  foreach_active (i) {
    uniform double r = erf(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform double erfc(uniform double x);
SCALAR_FUN_ATTR double futrts_erfc64(double x) {
  double res;
  foreach_active (i) {
    uniform double r = erfc(extract(x, i));
    res = insert(res, i, r);
  }
  return res;
}

SCALAR_FUN_ATTR double futrts_fma64(double a, double b, double c) { return a * b + c; }
SCALAR_FUN_ATTR double futrts_round64(double x) { return round(x); }
SCALAR_FUN_ATTR double futrts_ceil64(double x) { return ceil(x); }

extern "C" unmasked uniform double nextafter(uniform float x, uniform double y);
SCALAR_FUN_ATTR float futrts_nextafter64(double x, double y) {
  double res;
  foreach_active (i) {
    uniform double r = nextafter(extract(x, i), extract(y, i));
    res = insert(res, i, r);
  }
  return res;
}

SCALAR_FUN_ATTR double futrts_floor64(double x) { return floor(x); }
SCALAR_FUN_ATTR bool futrts_isnan64(double x) { return isnan(x); }

SCALAR_FUN_ATTR int8_t fptosi_f64_i8(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int8_t) x;
  }
}

SCALAR_FUN_ATTR int16_t fptosi_f64_i16(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int16_t) x;
  }
}

SCALAR_FUN_ATTR int32_t fptosi_f64_i32(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int32_t) x;
  }
}

SCALAR_FUN_ATTR int64_t fptosi_f64_i64(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int64_t) x;
  }
}

SCALAR_FUN_ATTR uint8_t fptoui_f64_i8(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint8_t) (int8_t) x;
  }
}

SCALAR_FUN_ATTR uint16_t fptoui_f64_i16(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint16_t) (int16_t) x;
  }
}

SCALAR_FUN_ATTR uint32_t fptoui_f64_i32(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint32_t) (int32_t) x;
  }
}

SCALAR_FUN_ATTR uint64_t fptoui_f64_i64(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint64_t) (int64_t) x;
  }
}

SCALAR_FUN_ATTR bool ftob_f64_bool(double x) { return x != 0.0; }
SCALAR_FUN_ATTR double btof_bool_f64(bool x) { return x ? 1.0 : 0.0; }

SCALAR_FUN_ATTR int64_t fptobits_f64_i64(double x) {
  int64_t res;
  foreach_active (i) {
    uniform double tmp = extract(x, i);
    uniform int64_t r = *((uniform int64_t* uniform)&tmp);
    res = insert(res, i, r);
  }
  return res;
}

SCALAR_FUN_ATTR double bitstofp_i64_f64(int64_t x) {
  double res;
  foreach_active (i) {
    uniform int64_t tmp = extract(x, i);
    uniform double r = *((uniform double* uniform)&tmp);
    res = insert(res, i, r);
  }
  return res;
}

SCALAR_FUN_ATTR uniform int64_t fptobits_f64_i64(uniform double x) {
  return intbits(x);
}

SCALAR_FUN_ATTR uniform double bitstofp_i64_f64(uniform int64_t x) {
  return doublebits(x);
}

SCALAR_FUN_ATTR double fmod64(double x, double y) {
  return x - y * trunc(x/y);
}

SCALAR_FUN_ATTR double fsignum64(double x) {
  return futrts_isnan64(x) ? x : (x > 0 ? 1.0d : 0.0d) - (x < 0 ? 1.0d : 0.0d);
}

SCALAR_FUN_ATTR double futrts_lerp64(double v0, double v1, double t) {
  return v0 + (v1 - v0) * t;
}

SCALAR_FUN_ATTR double futrts_ldexp64(double x, int32_t y) {
  return x * pow((uniform double)2.0, (double)y);
}

SCALAR_FUN_ATTR double futrts_copysign64(double x, double y) {
  int64_t xb = fptobits_f64_i64(x);
  int64_t yb = fptobits_f64_i64(y);
  return bitstofp_i64_f64((xb & ~(((int64_t)1)<<63)) | (yb & (((int64_t)1)<<63)));
}

SCALAR_FUN_ATTR double futrts_mad64(double a, double b, double c) { return a * b + c; }
SCALAR_FUN_ATTR float fpconv_f32_f32(float x) { return (float) x; }
SCALAR_FUN_ATTR double fpconv_f32_f64(float x) { return (double) x; }
SCALAR_FUN_ATTR float fpconv_f64_f32(double x) { return (float) x; }
SCALAR_FUN_ATTR double fpconv_f64_f64(double x) { return (double) x; }

#else

SCALAR_FUN_ATTR double fdiv64(double x, double y) { return x / y; }
SCALAR_FUN_ATTR double fadd64(double x, double y) { return x + y; }
SCALAR_FUN_ATTR double fsub64(double x, double y) { return x - y; }
SCALAR_FUN_ATTR double fmul64(double x, double y) { return x * y; }
SCALAR_FUN_ATTR bool cmplt64(double x, double y) { return x < y; }
SCALAR_FUN_ATTR bool cmple64(double x, double y) { return x <= y; }
SCALAR_FUN_ATTR double sitofp_i8_f64(int8_t x) { return (double) x; }
SCALAR_FUN_ATTR double sitofp_i16_f64(int16_t x) { return (double) x; }
SCALAR_FUN_ATTR double sitofp_i32_f64(int32_t x) { return (double) x; }
SCALAR_FUN_ATTR double sitofp_i64_f64(int64_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i8_f64(uint8_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i16_f64(uint16_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i32_f64(uint32_t x) { return (double) x; }
SCALAR_FUN_ATTR double uitofp_i64_f64(uint64_t x) { return (double) x; }
SCALAR_FUN_ATTR double fabs64(double x) { return fabs(x); }
SCALAR_FUN_ATTR double fmax64(double x, double y) { return fmax(x, y); }
SCALAR_FUN_ATTR double fmin64(double x, double y) { return fmin(x, y); }
SCALAR_FUN_ATTR double fpow64(double x, double y) { return pow(x, y); }
SCALAR_FUN_ATTR double futrts_log64(double x) { return log(x); }
SCALAR_FUN_ATTR double futrts_log2_64(double x) { return log2(x); }
SCALAR_FUN_ATTR double futrts_log10_64(double x) { return log10(x); }
SCALAR_FUN_ATTR double futrts_log1p_64(double x) { return log1p(x); }
SCALAR_FUN_ATTR double futrts_sqrt64(double x) { return sqrt(x); }
SCALAR_FUN_ATTR double futrts_rsqrt64(double x) { return 1/sqrt(x); }
SCALAR_FUN_ATTR double futrts_cbrt64(double x) { return cbrt(x); }
SCALAR_FUN_ATTR double futrts_exp64(double x) { return exp(x); }
SCALAR_FUN_ATTR double futrts_cos64(double x) { return cos(x); }

SCALAR_FUN_ATTR double futrts_cospi64(double x) {
#ifdef __OPENCL_VERSION__
  return cospi(x);
#elif defined(__CUDA_ARCH__)
  return cospi(x);
#else
  return cos(M_PI*x);
#endif
}

SCALAR_FUN_ATTR double futrts_sin64(double x) {
  return sin(x);
}

SCALAR_FUN_ATTR double futrts_sinpi64(double x) {
#ifdef __OPENCL_VERSION__
  return sinpi(x);
#elif defined(__CUDA_ARCH__)
  return sinpi(x);
#else
  return sin(M_PI*x);
#endif
}

SCALAR_FUN_ATTR double futrts_tan64(double x) {
  return tan(x);
}

SCALAR_FUN_ATTR double futrts_tanpi64(double x) {
#ifdef __OPENCL_VERSION__
  return tanpi(x);
#else
  return tan(M_PI*x);
#endif
}

SCALAR_FUN_ATTR double futrts_acos64(double x) {
  return acos(x);
}

SCALAR_FUN_ATTR double futrts_acospi64(double x) {
#ifdef __OPENCL_VERSION__
  return acospi(x);
#else
  return acos(x) / M_PI;
#endif
}

SCALAR_FUN_ATTR double futrts_asin64(double x) {
  return asin(x);
}

SCALAR_FUN_ATTR double futrts_asinpi64(double x) {
#ifdef __OPENCL_VERSION__
  return asinpi(x);
#else
  return asin(x) / M_PI;
#endif
}

SCALAR_FUN_ATTR double futrts_atan64(double x) {
  return atan(x);
}

SCALAR_FUN_ATTR double futrts_atanpi64(double x) {
#ifdef __OPENCL_VERSION__
  return atanpi(x);
#else
  return atan(x) / M_PI;
#endif
}

SCALAR_FUN_ATTR double futrts_cosh64(double x) { return cosh(x); }
SCALAR_FUN_ATTR double futrts_sinh64(double x) { return sinh(x); }
SCALAR_FUN_ATTR double futrts_tanh64(double x) { return tanh(x); }
SCALAR_FUN_ATTR double futrts_acosh64(double x) { return acosh(x); }
SCALAR_FUN_ATTR double futrts_asinh64(double x) { return asinh(x); }
SCALAR_FUN_ATTR double futrts_atanh64(double x) { return atanh(x); }
SCALAR_FUN_ATTR double futrts_atan2_64(double x, double y) { return atan2(x, y); }

SCALAR_FUN_ATTR double futrts_atan2pi_64(double x, double y) {
#ifdef __OPENCL_VERSION__
  return atan2pi(x, y);
#else
  return atan2(x, y) / M_PI;
#endif
}

SCALAR_FUN_ATTR double futrts_hypot64(double x, double y) { return hypot(x, y); }
SCALAR_FUN_ATTR double futrts_gamma64(double x) { return tgamma(x); }
SCALAR_FUN_ATTR double futrts_lgamma64(double x) { return lgamma(x); }
SCALAR_FUN_ATTR double futrts_erf64(double x) { return erf(x); }
SCALAR_FUN_ATTR double futrts_erfc64(double x) { return erfc(x); }
SCALAR_FUN_ATTR double futrts_fma64(double a, double b, double c) { return fma(a, b, c); }
SCALAR_FUN_ATTR double futrts_round64(double x) { return rint(x); }
SCALAR_FUN_ATTR double futrts_ceil64(double x) { return ceil(x); }
SCALAR_FUN_ATTR float futrts_nextafter64(float x, float y) { return nextafter(x, y); }
SCALAR_FUN_ATTR double futrts_floor64(double x) { return floor(x); }
SCALAR_FUN_ATTR bool futrts_isnan64(double x) { return isnan(x); }
SCALAR_FUN_ATTR bool futrts_isinf64(double x) { return isinf(x); }

SCALAR_FUN_ATTR int8_t fptosi_f64_i8(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int8_t) x;
  }
}

SCALAR_FUN_ATTR int16_t fptosi_f64_i16(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int16_t) x;
  }
}

SCALAR_FUN_ATTR int32_t fptosi_f64_i32(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int32_t) x;
  }
}

SCALAR_FUN_ATTR int64_t fptosi_f64_i64(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (int64_t) x;
  }
}

SCALAR_FUN_ATTR uint8_t fptoui_f64_i8(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint8_t) (int8_t) x;
  }
}

SCALAR_FUN_ATTR uint16_t fptoui_f64_i16(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint16_t) (int16_t) x;
  }
}

SCALAR_FUN_ATTR uint32_t fptoui_f64_i32(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint32_t) (int32_t) x;
  }
}

SCALAR_FUN_ATTR uint64_t fptoui_f64_i64(double x) {
  if (futrts_isnan64(x) || futrts_isinf64(x)) {
    return 0;
  } else {
    return (uint64_t) (int64_t) x;
  }
}

SCALAR_FUN_ATTR bool ftob_f64_bool(double x) { return x != 0; }
SCALAR_FUN_ATTR double btof_bool_f64(bool x) { return x ? 1 : 0; }

SCALAR_FUN_ATTR int64_t fptobits_f64_i64(double x) {
  union {
    double f;
    int64_t t;
  } p;

  p.f = x;
  return p.t;
}

SCALAR_FUN_ATTR double bitstofp_i64_f64(int64_t x) {
  union {
    int64_t f;
    double t;
  } p;

  p.f = x;
  return p.t;
}

SCALAR_FUN_ATTR double fmod64(double x, double y) {
  return fmod(x, y);
}

SCALAR_FUN_ATTR double fsignum64(double x) {
  return futrts_isnan64(x) ? x : (x > 0) - (x < 0);
}

SCALAR_FUN_ATTR double futrts_lerp64(double v0, double v1, double t) {
#ifdef __OPENCL_VERSION__
  return mix(v0, v1, t);
#else
  return v0 + (v1 - v0) * t;
#endif
}

SCALAR_FUN_ATTR double futrts_ldexp64(double x, int32_t y) {
  return ldexp(x, y);
}

SCALAR_FUN_ATTR float futrts_copysign64(double x, double y) {
  return copysign(x, y);
}

SCALAR_FUN_ATTR double futrts_mad64(double a, double b, double c) {
#ifdef __OPENCL_VERSION__
  return mad(a, b, c);
#else
  return a * b + c;
#endif
}

SCALAR_FUN_ATTR float fpconv_f32_f32(float x) { return (float) x; }
SCALAR_FUN_ATTR double fpconv_f32_f64(float x) { return (double) x; }
SCALAR_FUN_ATTR float fpconv_f64_f32(double x) { return (float) x; }
SCALAR_FUN_ATTR double fpconv_f64_f64(double x) { return (double) x; }

#endif

#endif

#define futrts_cond_f16(x,y,z) ((x) ? (y) : (z))
#define futrts_cond_f32(x,y,z) ((x) ? (y) : (z))
#define futrts_cond_f64(x,y,z) ((x) ? (y) : (z))

#define futrts_cond_i8(x,y,z) ((x) ? (y) : (z))
#define futrts_cond_i16(x,y,z) ((x) ? (y) : (z))
#define futrts_cond_i32(x,y,z) ((x) ? (y) : (z))
#define futrts_cond_i64(x,y,z) ((x) ? (y) : (z))

#define futrts_cond_bool(x,y,z) ((x) ? (y) : (z))
#define futrts_cond_unit(x,y,z) ((x) ? (y) : (z))

// End of scalar.h.
// Start of scalar_f16.h.

// Half-precision is emulated if needed (e.g. in straight C) with the
// native type used if possible.  The emulation works by typedef'ing
// 'float' to 'f16', and then implementing all operations on single
// precision.  To cut down on duplication, we use the same code for
// those Futhark functions that require just operators or casts.  The
// in-memory representation for arrays will still be 16 bits even
// under emulation, so the compiler will have to be careful when
// generating reads or writes.

#if !defined(cl_khr_fp16) && !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600) && !(defined(ISPC))
#define EMULATE_F16
#endif

#if !defined(EMULATE_F16) && defined(__OPENCL_VERSION__)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifdef EMULATE_F16

// Note that the half-precision storage format is still 16 bits - the
// compiler will have to be real careful!
typedef float f16;

#elif defined(ISPC)
typedef float16 f16;

#else

#ifdef __CUDA_ARCH__
#include <cuda_fp16.h>
#endif

typedef half f16;

#endif

// Some of these functions convert to single precision because half
// precision versions are not available.
SCALAR_FUN_ATTR f16 fadd16(f16 x, f16 y) { return x + y; }
SCALAR_FUN_ATTR f16 fsub16(f16 x, f16 y) { return x - y; }
SCALAR_FUN_ATTR f16 fmul16(f16 x, f16 y) { return x * y; }
SCALAR_FUN_ATTR bool cmplt16(f16 x, f16 y) { return x < y; }
SCALAR_FUN_ATTR bool cmple16(f16 x, f16 y) { return x <= y; }
SCALAR_FUN_ATTR f16 sitofp_i8_f16(int8_t x) { return (f16) x; }
SCALAR_FUN_ATTR f16 sitofp_i16_f16(int16_t x) { return (f16) x; }
SCALAR_FUN_ATTR f16 sitofp_i32_f16(int32_t x) { return (f16) x; }
SCALAR_FUN_ATTR f16 sitofp_i64_f16(int64_t x) { return (f16) x; }
SCALAR_FUN_ATTR f16 uitofp_i8_f16(uint8_t x) { return (f16) x; }
SCALAR_FUN_ATTR f16 uitofp_i16_f16(uint16_t x) { return (f16) x; }
SCALAR_FUN_ATTR f16 uitofp_i32_f16(uint32_t x) { return (f16) x; }
SCALAR_FUN_ATTR f16 uitofp_i64_f16(uint64_t x) { return (f16) x; }
SCALAR_FUN_ATTR int8_t fptosi_f16_i8(f16 x) { return (int8_t) (float) x; }
SCALAR_FUN_ATTR int16_t fptosi_f16_i16(f16 x) { return (int16_t) x; }
SCALAR_FUN_ATTR int32_t fptosi_f16_i32(f16 x) { return (int32_t) x; }
SCALAR_FUN_ATTR int64_t fptosi_f16_i64(f16 x) { return (int64_t) x; }
SCALAR_FUN_ATTR uint8_t fptoui_f16_i8(f16 x) { return (uint8_t) (float) x; }
SCALAR_FUN_ATTR uint16_t fptoui_f16_i16(f16 x) { return (uint16_t) x; }
SCALAR_FUN_ATTR uint32_t fptoui_f16_i32(f16 x) { return (uint32_t) x; }
SCALAR_FUN_ATTR uint64_t fptoui_f16_i64(f16 x) { return (uint64_t) x; }
SCALAR_FUN_ATTR bool ftob_f16_bool(f16 x) { return x != (f16)0; }
SCALAR_FUN_ATTR f16 btof_bool_f16(bool x) { return x ? 1 : 0; }

#ifndef EMULATE_F16

SCALAR_FUN_ATTR bool futrts_isnan16(f16 x) { return isnan((float)x); }

#ifdef __OPENCL_VERSION__

SCALAR_FUN_ATTR f16 fabs16(f16 x) { return fabs(x); }
SCALAR_FUN_ATTR f16 fmax16(f16 x, f16 y) { return fmax(x, y); }
SCALAR_FUN_ATTR f16 fmin16(f16 x, f16 y) { return fmin(x, y); }
SCALAR_FUN_ATTR f16 fpow16(f16 x, f16 y) { return pow(x, y); }

#elif defined(ISPC)

SCALAR_FUN_ATTR f16 fabs16(f16 x) { return abs(x); }
SCALAR_FUN_ATTR f16 fmax16(f16 x, f16 y) { return futrts_isnan16(x) ? y : futrts_isnan16(y) ? x : max(x, y); }
SCALAR_FUN_ATTR f16 fmin16(f16 x, f16 y) { return futrts_isnan16(x) ? y : futrts_isnan16(y) ? x : min(x, y); }
SCALAR_FUN_ATTR f16 fpow16(f16 x, f16 y) { return pow(x, y); }

#else // Assuming CUDA.

SCALAR_FUN_ATTR f16 fabs16(f16 x) { return fabsf(x); }
SCALAR_FUN_ATTR f16 fmax16(f16 x, f16 y) { return fmaxf(x, y); }
SCALAR_FUN_ATTR f16 fmin16(f16 x, f16 y) { return fminf(x, y); }
SCALAR_FUN_ATTR f16 fpow16(f16 x, f16 y) { return powf(x, y); }

#endif

#if defined(ISPC)
SCALAR_FUN_ATTR bool futrts_isinf16(float x) { return !futrts_isnan16(x) && futrts_isnan16(x - x); }
SCALAR_FUN_ATTR bool futrts_isfinite16(float x) { return !futrts_isnan16(x) && !futrts_isinf16(x); }
#else
SCALAR_FUN_ATTR bool futrts_isinf16(f16 x) { return isinf((float)x); }
#endif

#ifdef __OPENCL_VERSION__
SCALAR_FUN_ATTR f16 futrts_log16(f16 x) { return log(x); }
SCALAR_FUN_ATTR f16 futrts_log2_16(f16 x) { return log2(x); }
SCALAR_FUN_ATTR f16 futrts_log10_16(f16 x) { return log10(x); }
SCALAR_FUN_ATTR f16 futrts_log1p_16(f16 x) { return log1p(x); }
SCALAR_FUN_ATTR f16 futrts_sqrt16(f16 x) { return sqrt(x); }
SCALAR_FUN_ATTR f16 futrts_rsqrt16(f16 x) { return rsqrt(x); }
SCALAR_FUN_ATTR f16 futrts_cbrt16(f16 x) { return cbrt(x); }
SCALAR_FUN_ATTR f16 futrts_exp16(f16 x) { return exp(x); }
SCALAR_FUN_ATTR f16 futrts_cos16(f16 x) { return cos(x); }
SCALAR_FUN_ATTR f16 futrts_cospi16(f16 x) { return cospi(x); }
SCALAR_FUN_ATTR f16 futrts_sin16(f16 x) { return sin(x); }
SCALAR_FUN_ATTR f16 futrts_sinpi16(f16 x) { return sinpi(x); }
SCALAR_FUN_ATTR f16 futrts_tan16(f16 x) { return tan(x); }
SCALAR_FUN_ATTR f16 futrts_tanpi16(f16 x) { return tanpi(x); }
SCALAR_FUN_ATTR f16 futrts_acos16(f16 x) { return acos(x); }
SCALAR_FUN_ATTR f16 futrts_acospi16(f16 x) { return acospi(x); }
SCALAR_FUN_ATTR f16 futrts_asin16(f16 x) { return asin(x); }
SCALAR_FUN_ATTR f16 futrts_asinpi16(f16 x) { return asinpi(x); }
SCALAR_FUN_ATTR f16 futrts_atan16(f16 x) { return atan(x); }
SCALAR_FUN_ATTR f16 futrts_atanpi16(f16 x) { return atanpi(x); }
SCALAR_FUN_ATTR f16 futrts_cosh16(f16 x) { return cosh(x); }
SCALAR_FUN_ATTR f16 futrts_sinh16(f16 x) { return sinh(x); }
SCALAR_FUN_ATTR f16 futrts_tanh16(f16 x) { return tanh(x); }
SCALAR_FUN_ATTR f16 futrts_acosh16(f16 x) { return acosh(x); }
SCALAR_FUN_ATTR f16 futrts_asinh16(f16 x) { return asinh(x); }
SCALAR_FUN_ATTR f16 futrts_atanh16(f16 x) { return atanh(x); }
SCALAR_FUN_ATTR f16 futrts_atan2_16(f16 x, f16 y) { return atan2(x, y); }
SCALAR_FUN_ATTR f16 futrts_atan2pi_16(f16 x, f16 y) { return atan2pi(x, y); }
SCALAR_FUN_ATTR f16 futrts_hypot16(f16 x, f16 y) { return hypot(x, y); }
SCALAR_FUN_ATTR f16 futrts_gamma16(f16 x) { return tgamma(x); }
SCALAR_FUN_ATTR f16 futrts_lgamma16(f16 x) { return lgamma(x); }
SCALAR_FUN_ATTR f16 futrts_erf16(f16 x) { return erf(x); }
SCALAR_FUN_ATTR f16 futrts_erfc16(f16 x) { return erfc(x); }
SCALAR_FUN_ATTR f16 fmod16(f16 x, f16 y) { return fmod(x, y); }
SCALAR_FUN_ATTR f16 futrts_round16(f16 x) { return rint(x); }
SCALAR_FUN_ATTR f16 futrts_floor16(f16 x) { return floor(x); }
SCALAR_FUN_ATTR f16 futrts_ceil16(f16 x) { return ceil(x); }
SCALAR_FUN_ATTR f16 futrts_nextafter16(f16 x, f16 y) { return nextafter(x, y); }
SCALAR_FUN_ATTR f16 futrts_lerp16(f16 v0, f16 v1, f16 t) { return mix(v0, v1, t); }
SCALAR_FUN_ATTR f16 futrts_ldexp16(f16 x, int32_t y) { return ldexp(x, y); }
SCALAR_FUN_ATTR f16 futrts_copysign16(f16 x, f16 y) { return copysign(x, y); }
SCALAR_FUN_ATTR f16 futrts_mad16(f16 a, f16 b, f16 c) { return mad(a, b, c); }
SCALAR_FUN_ATTR f16 futrts_fma16(f16 a, f16 b, f16 c) { return fma(a, b, c); }

#elif defined(ISPC)

SCALAR_FUN_ATTR f16 futrts_log16(f16 x) { return futrts_isfinite16(x) || (futrts_isinf16(x) && x < 0) ? log(x) : x; }
SCALAR_FUN_ATTR f16 futrts_log2_16(f16 x) { return futrts_log16(x) / log(2.0f16); }
SCALAR_FUN_ATTR f16 futrts_log10_16(f16 x) { return futrts_log16(x) / log(10.0f16); }
SCALAR_FUN_ATTR f16 futrts_log1p_16(f16 x) {
  if(x == -1.0f16 || (futrts_isinf16(x) && x > 0.0f16)) return x / 0.0f16;
  f16 y = 1.0f16 + x;
  f16 z = y - 1.0f16;
  return log(y) - (z-x)/y;
}
SCALAR_FUN_ATTR f16 futrts_sqrt16(f16 x) { return (float16)sqrt((float)x); }
SCALAR_FUN_ATTR f16 futrts_rsqrt16(f16 x) { return (float16)1/sqrt((float)x); }
SCALAR_FUN_ATTR f16 futrts_exp16(f16 x) { return exp(x); }
SCALAR_FUN_ATTR f16 futrts_cos16(f16 x) { return (float16)cos((float)x); }
SCALAR_FUN_ATTR f16 futrts_cospi16(f16 x) { return (float16)cos((float)M_PI*(float)x); }
SCALAR_FUN_ATTR f16 futrts_sin16(f16 x) { return (float16)sin((float)x); }
SCALAR_FUN_ATTR f16 futrts_sinpi16(f16 x) { return (float16)sin((float)M_PI*(float)x); }
SCALAR_FUN_ATTR f16 futrts_tan16(f16 x) { return (float16)tan((float)x); }
SCALAR_FUN_ATTR f16 futrts_tanpi16(f16 x) { return (float16)(tan((float)M_PI*(float)x)); }
SCALAR_FUN_ATTR f16 futrts_acos16(f16 x) { return (float16)acos((float)x); }
SCALAR_FUN_ATTR f16 futrts_acospi16(f16 x) { return (float16)(acos((float)x)/(float)M_PI); }
SCALAR_FUN_ATTR f16 futrts_asin16(f16 x) { return (float16)asin((float)x); }
SCALAR_FUN_ATTR f16 futrts_asinpi16(f16 x) { return (float16)(asin((float)x)/(float)M_PI); }
SCALAR_FUN_ATTR f16 futrts_atan16(f16 x) { return (float16)atan((float)x); }
SCALAR_FUN_ATTR f16 futrts_atanpi16(f16 x) { return (float16)(atan((float)x)/(float)M_PI); }
SCALAR_FUN_ATTR f16 futrts_cosh16(f16 x) { return (exp(x)+exp(-x)) / 2.0f16; }
SCALAR_FUN_ATTR f16 futrts_sinh16(f16 x) { return (exp(x)-exp(-x)) / 2.0f16; }
SCALAR_FUN_ATTR f16 futrts_tanh16(f16 x) { return futrts_sinh16(x)/futrts_cosh16(x); }
SCALAR_FUN_ATTR f16 futrts_acosh16(f16 x) {
  float16 f = x+(float16)sqrt((float)(x*x-1));
  if(futrts_isfinite16(f)) return log(f);
  return f;
}
SCALAR_FUN_ATTR f16 futrts_asinh16(f16 x) {
  float16 f = x+(float16)sqrt((float)(x*x+1));
  if(futrts_isfinite16(f)) return log(f);
  return f;
}
SCALAR_FUN_ATTR f16 futrts_atanh16(f16 x) {
  float16 f = (1+x)/(1-x);
  if(futrts_isfinite16(f)) return log(f)/2.0f16;
  return f;
}
SCALAR_FUN_ATTR f16 futrts_atan2_16(f16 x, f16 y) { return (float16)atan2((float)x, (float)y); }
SCALAR_FUN_ATTR f16 futrts_atan2pi_16(f16 x, f16 y) { return (float16)(atan2((float)x, (float)y)/(float)M_PI); }
SCALAR_FUN_ATTR f16 futrts_hypot16(f16 x, f16 y) { return (float16)futrts_hypot32((float)x, (float)y); }

extern "C" unmasked uniform float tgammaf(uniform float x);
SCALAR_FUN_ATTR f16 futrts_gamma16(f16 x) {
  f16 res;
  foreach_active (i) {
    uniform f16 r = (f16)tgammaf(extract((float)x, i));
    res = insert(res, i, r);
  }
  return res;
}

extern "C" unmasked uniform float lgammaf(uniform float x);
SCALAR_FUN_ATTR f16 futrts_lgamma16(f16 x) {
  f16 res;
  foreach_active (i) {
    uniform f16 r = (f16)lgammaf(extract((float)x, i));
    res = insert(res, i, r);
  }
  return res;
}
SCALAR_FUN_ATTR f16 futrts_cbrt16(f16 x) { return (f16)futrts_cbrt32((float)x); }
SCALAR_FUN_ATTR f16 futrts_erf16(f16 x) { return (f16)futrts_erf32((float)x); }
SCALAR_FUN_ATTR f16 futrts_erfc16(f16 x) { return (f16)futrts_erfc32((float)x); }
SCALAR_FUN_ATTR f16 fmod16(f16 x, f16 y) { return x - y * (float16)trunc((float) (x/y)); }
SCALAR_FUN_ATTR f16 futrts_round16(f16 x) { return (float16)round((float)x); }
SCALAR_FUN_ATTR f16 futrts_floor16(f16 x) { return (float16)floor((float)x); }
SCALAR_FUN_ATTR f16 futrts_ceil16(f16 x) { return (float16)ceil((float)x); }
SCALAR_FUN_ATTR f16 futrts_nextafter16(f16 x, f16 y) { return (float16)futrts_nextafter32((float)x, (float) y); }
SCALAR_FUN_ATTR f16 futrts_lerp16(f16 v0, f16 v1, f16 t) { return v0 + (v1 - v0) * t; }
SCALAR_FUN_ATTR f16 futrts_ldexp16(f16 x, int32_t y) { return futrts_ldexp32((float)x, y); }
SCALAR_FUN_ATTR f16 futrts_copysign16(f16 x, f16 y) { return futrts_copysign32((float)x, y); }
SCALAR_FUN_ATTR f16 futrts_mad16(f16 a, f16 b, f16 c) { return a * b + c; }
SCALAR_FUN_ATTR f16 futrts_fma16(f16 a, f16 b, f16 c) { return a * b + c; }

#else // Assume CUDA.

SCALAR_FUN_ATTR f16 futrts_log16(f16 x) { return hlog(x); }
SCALAR_FUN_ATTR f16 futrts_log2_16(f16 x) { return hlog2(x); }
SCALAR_FUN_ATTR f16 futrts_log10_16(f16 x) { return hlog10(x); }
SCALAR_FUN_ATTR f16 futrts_log1p_16(f16 x) { return (f16)log1pf((float)x); }
SCALAR_FUN_ATTR f16 futrts_sqrt16(f16 x) { return hsqrt(x); }
SCALAR_FUN_ATTR f16 futrts_rsqrt16(f16 x) { return hrsqrt(x); }
SCALAR_FUN_ATTR f16 futrts_cbrt16(f16 x) { return cbrtf(x); }
SCALAR_FUN_ATTR f16 futrts_exp16(f16 x) { return hexp(x); }
SCALAR_FUN_ATTR f16 futrts_cos16(f16 x) { return hcos(x); }
SCALAR_FUN_ATTR f16 futrts_cospi16(f16 x) { return hcos((f16)M_PI*x); }
SCALAR_FUN_ATTR f16 futrts_sin16(f16 x) { return hsin(x); }
SCALAR_FUN_ATTR f16 futrts_sinpi16(f16 x) { return hsin((f16)M_PI*x); }
SCALAR_FUN_ATTR f16 futrts_tan16(f16 x) { return tanf(x); }
SCALAR_FUN_ATTR f16 futrts_tanpi16(f16 x) { return tanf((f16)M_PI*x); }
SCALAR_FUN_ATTR f16 futrts_acos16(f16 x) { return acosf(x); }
SCALAR_FUN_ATTR f16 futrts_acospi16(f16 x) { return (f16)acosf(x)/(f16)M_PI; }
SCALAR_FUN_ATTR f16 futrts_asin16(f16 x) { return asinf(x); }
SCALAR_FUN_ATTR f16 futrts_asinpi16(f16 x) { return (f16)asinf(x)/(f16)M_PI; }
SCALAR_FUN_ATTR f16 futrts_atan16(f16 x) { return (f16)atanf(x); }
SCALAR_FUN_ATTR f16 futrts_atanpi16(f16 x) { return (f16)atanf(x)/(f16)M_PI; }
SCALAR_FUN_ATTR f16 futrts_cosh16(f16 x) { return coshf(x); }
SCALAR_FUN_ATTR f16 futrts_sinh16(f16 x) { return sinhf(x); }
SCALAR_FUN_ATTR f16 futrts_tanh16(f16 x) { return tanhf(x); }
SCALAR_FUN_ATTR f16 futrts_acosh16(f16 x) { return acoshf(x); }
SCALAR_FUN_ATTR f16 futrts_asinh16(f16 x) { return asinhf(x); }
SCALAR_FUN_ATTR f16 futrts_atanh16(f16 x) { return atanhf(x); }
SCALAR_FUN_ATTR f16 futrts_atan2_16(f16 x, f16 y) { return (f16)atan2f(x, y); }
SCALAR_FUN_ATTR f16 futrts_atan2pi_16(f16 x, f16 y) { return (f16)atan2f(x, y)/(f16)M_PI; }
SCALAR_FUN_ATTR f16 futrts_hypot16(f16 x, f16 y) { return hypotf(x, y); }
SCALAR_FUN_ATTR f16 futrts_gamma16(f16 x) { return tgammaf(x); }
SCALAR_FUN_ATTR f16 futrts_lgamma16(f16 x) { return lgammaf(x); }
SCALAR_FUN_ATTR f16 futrts_erf16(f16 x) { return erff(x); }
SCALAR_FUN_ATTR f16 futrts_erfc16(f16 x) { return erfcf(x); }
SCALAR_FUN_ATTR f16 fmod16(f16 x, f16 y) { return fmodf(x, y); }
SCALAR_FUN_ATTR f16 futrts_round16(f16 x) { return rintf(x); }
SCALAR_FUN_ATTR f16 futrts_floor16(f16 x) { return hfloor(x); }
SCALAR_FUN_ATTR f16 futrts_ceil16(f16 x) { return hceil(x); }
SCALAR_FUN_ATTR f16 futrts_nextafter16(f16 x, f16 y) { return __ushort_as_half(halfbitsnextafter(__half_as_ushort(x), __half_as_ushort(y))); }
SCALAR_FUN_ATTR f16 futrts_lerp16(f16 v0, f16 v1, f16 t) { return v0 + (v1 - v0) * t; }
SCALAR_FUN_ATTR f16 futrts_ldexp16(f16 x, int32_t y) { return futrts_ldexp32((float)x, y); }
SCALAR_FUN_ATTR f16 futrts_copysign16(f16 x, f16 y) { return futrts_copysign32((float)x, y); }
SCALAR_FUN_ATTR f16 futrts_mad16(f16 a, f16 b, f16 c) { return a * b + c; }
SCALAR_FUN_ATTR f16 futrts_fma16(f16 a, f16 b, f16 c) { return fmaf(a, b, c); }

#endif

// The CUDA __half type cannot be put in unions for some reason, so we
// use bespoke conversion functions instead.
#ifdef __CUDA_ARCH__
SCALAR_FUN_ATTR int16_t fptobits_f16_i16(f16 x) { return __half_as_ushort(x); }
SCALAR_FUN_ATTR f16 bitstofp_i16_f16(int16_t x) { return __ushort_as_half(x); }
#elif defined(ISPC)
SCALAR_FUN_ATTR int16_t fptobits_f16_i16(f16 x) { varying int16_t y = *((varying int16_t * uniform)&x); return y;
}
SCALAR_FUN_ATTR f16 bitstofp_i16_f16(int16_t x) { varying f16 y = *((varying f16 * uniform)&x); return y; }
#else
SCALAR_FUN_ATTR int16_t fptobits_f16_i16(f16 x) {
  union {
    f16 f;
    int16_t t;
  } p;

  p.f = x;
  return p.t;
}

SCALAR_FUN_ATTR f16 bitstofp_i16_f16(int16_t x) {
  union {
    int16_t f;
    f16 t;
  } p;

  p.f = x;
  return p.t;
}
#endif

#else // No native f16 - emulate.

SCALAR_FUN_ATTR f16 fabs16(f16 x) { return fabs32(x); }
SCALAR_FUN_ATTR f16 fmax16(f16 x, f16 y) { return fmax32(x, y); }
SCALAR_FUN_ATTR f16 fmin16(f16 x, f16 y) { return fmin32(x, y); }
SCALAR_FUN_ATTR f16 fpow16(f16 x, f16 y) { return fpow32(x, y); }
SCALAR_FUN_ATTR bool futrts_isnan16(f16 x) { return futrts_isnan32(x); }
SCALAR_FUN_ATTR bool futrts_isinf16(f16 x) { return futrts_isinf32(x); }
SCALAR_FUN_ATTR f16 futrts_log16(f16 x) { return futrts_log32(x); }
SCALAR_FUN_ATTR f16 futrts_log2_16(f16 x) { return futrts_log2_32(x); }
SCALAR_FUN_ATTR f16 futrts_log10_16(f16 x) { return futrts_log10_32(x); }
SCALAR_FUN_ATTR f16 futrts_log1p_16(f16 x) { return futrts_log1p_32(x); }
SCALAR_FUN_ATTR f16 futrts_sqrt16(f16 x) { return futrts_sqrt32(x); }
SCALAR_FUN_ATTR f16 futrts_rsqrt16(f16 x) { return futrts_rsqrt32(x); }
SCALAR_FUN_ATTR f16 futrts_cbrt16(f16 x) { return futrts_cbrt32(x); }
SCALAR_FUN_ATTR f16 futrts_exp16(f16 x) { return futrts_exp32(x); }
SCALAR_FUN_ATTR f16 futrts_cos16(f16 x) { return futrts_cos32(x); }
SCALAR_FUN_ATTR f16 futrts_cospi16(f16 x) { return futrts_cospi32(x); }
SCALAR_FUN_ATTR f16 futrts_sin16(f16 x) { return futrts_sin32(x); }
SCALAR_FUN_ATTR f16 futrts_sinpi16(f16 x) { return futrts_sinpi32(x); }
SCALAR_FUN_ATTR f16 futrts_tan16(f16 x) { return futrts_tan32(x); }
SCALAR_FUN_ATTR f16 futrts_tanpi16(f16 x) { return futrts_tanpi32(x); }
SCALAR_FUN_ATTR f16 futrts_acos16(f16 x) { return futrts_acos32(x); }
SCALAR_FUN_ATTR f16 futrts_acospi16(f16 x) { return futrts_acospi32(x); }
SCALAR_FUN_ATTR f16 futrts_asin16(f16 x) { return futrts_asin32(x); }
SCALAR_FUN_ATTR f16 futrts_asinpi16(f16 x) { return futrts_asinpi32(x); }
SCALAR_FUN_ATTR f16 futrts_atan16(f16 x) { return futrts_atan32(x); }
SCALAR_FUN_ATTR f16 futrts_atanpi16(f16 x) { return futrts_atanpi32(x); }
SCALAR_FUN_ATTR f16 futrts_cosh16(f16 x) { return futrts_cosh32(x); }
SCALAR_FUN_ATTR f16 futrts_sinh16(f16 x) { return futrts_sinh32(x); }
SCALAR_FUN_ATTR f16 futrts_tanh16(f16 x) { return futrts_tanh32(x); }
SCALAR_FUN_ATTR f16 futrts_acosh16(f16 x) { return futrts_acosh32(x); }
SCALAR_FUN_ATTR f16 futrts_asinh16(f16 x) { return futrts_asinh32(x); }
SCALAR_FUN_ATTR f16 futrts_atanh16(f16 x) { return futrts_atanh32(x); }
SCALAR_FUN_ATTR f16 futrts_atan2_16(f16 x, f16 y) { return futrts_atan2_32(x, y); }
SCALAR_FUN_ATTR f16 futrts_atan2pi_16(f16 x, f16 y) { return futrts_atan2pi_32(x, y); }
SCALAR_FUN_ATTR f16 futrts_hypot16(f16 x, f16 y) { return futrts_hypot32(x, y); }
SCALAR_FUN_ATTR f16 futrts_gamma16(f16 x) { return futrts_gamma32(x); }
SCALAR_FUN_ATTR f16 futrts_lgamma16(f16 x) { return futrts_lgamma32(x); }
SCALAR_FUN_ATTR f16 futrts_erf16(f16 x) { return futrts_erf32(x); }
SCALAR_FUN_ATTR f16 futrts_erfc16(f16 x) { return futrts_erfc32(x); }
SCALAR_FUN_ATTR f16 fmod16(f16 x, f16 y) { return fmod32(x, y); }
SCALAR_FUN_ATTR f16 futrts_round16(f16 x) { return futrts_round32(x); }
SCALAR_FUN_ATTR f16 futrts_floor16(f16 x) { return futrts_floor32(x); }
SCALAR_FUN_ATTR f16 futrts_ceil16(f16 x) { return futrts_ceil32(x); }
SCALAR_FUN_ATTR f16 futrts_nextafter16(f16 x, f16 y) { return halfbits2float(halfbitsnextafter(float2halfbits(x), float2halfbits(y))); }
SCALAR_FUN_ATTR f16 futrts_lerp16(f16 v0, f16 v1, f16 t) { return futrts_lerp32(v0, v1, t); }
SCALAR_FUN_ATTR f16 futrts_ldexp16(f16 x, int32_t y) { return futrts_ldexp32(x, y); }
SCALAR_FUN_ATTR f16 futrts_copysign16(f16 x, f16 y) { return futrts_copysign32((float)x, y); }
SCALAR_FUN_ATTR f16 futrts_mad16(f16 a, f16 b, f16 c) { return futrts_mad32(a, b, c); }
SCALAR_FUN_ATTR f16 futrts_fma16(f16 a, f16 b, f16 c) { return futrts_fma32(a, b, c); }

// Even when we are using an OpenCL that does not support cl_khr_fp16,
// it must still support vload_half for actually creating a
// half-precision number, which can then be efficiently converted to a
// float.  Similarly for vstore_half.
#ifdef __OPENCL_VERSION__

SCALAR_FUN_ATTR int16_t fptobits_f16_i16(f16 x) {
  int16_t y;
  // Violating strict aliasing here.
  vstore_half((float)x, 0, (half*)&y);
  return y;
}

SCALAR_FUN_ATTR f16 bitstofp_i16_f16(int16_t x) {
  return (f16)vload_half(0, (half*)&x);
}

#else
SCALAR_FUN_ATTR int16_t fptobits_f16_i16(f16 x) { return (int16_t)float2halfbits(x); }
SCALAR_FUN_ATTR f16 bitstofp_i16_f16(int16_t x) { return halfbits2float((uint16_t)x); }
SCALAR_FUN_ATTR f16 fsignum16(f16 x) { return futrts_isnan16(x) ? x : (x > 0 ? 1 : 0) - (x < 0 ? 1 : 0); }

#endif

#endif

SCALAR_FUN_ATTR float fpconv_f16_f16(f16 x) { return x; }
SCALAR_FUN_ATTR float fpconv_f16_f32(f16 x) { return x; }
SCALAR_FUN_ATTR f16 fpconv_f32_f16(float x) { return (f16) x; }

#ifdef FUTHARK_F64_ENABLED
SCALAR_FUN_ATTR double fpconv_f16_f64(f16 x) { return (double) x; }
#if defined(ISPC)
SCALAR_FUN_ATTR f16 fpconv_f64_f16(double x) { return (f16) ((float)x); }
#else
SCALAR_FUN_ATTR f16 fpconv_f64_f16(double x) { return (f16) x; }
#endif
#endif

// End of scalar_f16.h.

// Start of context_prototypes.h
//
// Prototypes for the functions in context.h, or that will be called
// from those functions, that need to be available very early.

struct futhark_context_config;
struct futhark_context;

static void set_error(struct futhark_context* ctx, char *error);

// These are called in context/config new/free functions and contain
// shared setup.  They are generated by the compiler itself.
static int init_constants(struct futhark_context*);
static int free_constants(struct futhark_context*);
static void setup_program(struct futhark_context* ctx);
static void teardown_program(struct futhark_context *ctx);

// Allocate host memory.  Must be freed with host_free().
static void host_alloc(struct futhark_context* ctx, size_t size, const char* tag, size_t* size_out, void** mem_out);
// Allocate memory allocated with host_alloc().
static void host_free(struct futhark_context* ctx, size_t size, const char* tag, void* mem);

// Log that a copy has occurred. The provenance may be NULL, if we do not know
// where this came from.
static void log_copy(struct futhark_context* ctx,
                     const char *kind, const char *provenance,
                     int r,
                     int64_t dst_offset, int64_t dst_strides[r],
                     int64_t src_offset, int64_t src_strides[r],
                     int64_t shape[r]);

static void log_transpose(struct futhark_context* ctx,
                          int64_t k, int64_t m, int64_t n);

static bool lmad_map_tr(int64_t *num_arrays_out, int64_t *n_out, int64_t *m_out,
                        int r,
                        const int64_t dst_strides[r],
                        const int64_t src_strides[r],
                        const int64_t shape[r]);

static bool lmad_contiguous(int r, int64_t strides[r], int64_t shape[r]);

static bool lmad_memcpyable(int r,
                            int64_t dst_strides[r], int64_t src_strides[r], int64_t shape[r]);

static void add_event(struct futhark_context* ctx,
                      const char* name,
                      const char* provenance,
                      struct kvs *kvs,
                      void* data,
                      event_report_fn f);

// Functions that must be defined by the backend.
static void backend_context_config_setup(struct futhark_context_config* cfg);
static void backend_context_config_teardown(struct futhark_context_config* cfg);
static int backend_context_setup(struct futhark_context *ctx);
static void backend_context_teardown(struct futhark_context *ctx);

// End of of context_prototypes.h

struct memblock {
    int *references;
    unsigned char *mem;
    int64_t size;
    const char *desc;
};
struct constants {
    int dummy;
    struct memblock mem_26545;
    struct memblock mem_26548;
    struct memblock mem_26550;
    struct memblock mem_26555;
    struct memblock mem_26556;
    struct memblock mem_26562;
};
static bool static_array_realtype_26839[5] = { 1,0,0,1,0};
static bool static_array_realtype_26840[5] = { 0,1,0,0,0};
static bool static_array_realtype_26841[5] = { 0,0,1,0,0};
static bool static_array_realtype_26842[4] = { 1,1,0,0};
static bool static_array_realtype_26843[4] = { 0,0,1,0};
static bool static_array_realtype_26844[3] = { 0,0,0};
static bool static_array_realtype_26845[6] = { 1,1,0,0,1,0};
static bool static_array_realtype_26846[6] = { 0,0,1,0,0,1};
static bool static_array_realtype_26847[6] = { 0,1,1,1,0,0};
static bool static_array_realtype_26848[6] = { 0,0,0,0,0,0};
static int64_t static_array_realtype_26849[6] = { (int64_t) 0,(int64_t) 1,(int64_t) 0,(int64_t) 2,(int64_t) 2,(int64_t) 1};
static bool static_array_realtype_26850[5] = { 1,0,0,0,1};
static bool static_array_realtype_26851[5] = { 0,1,0,1,0};
static bool static_array_realtype_26852[5] = { 0,0,1,0,1};
static bool static_array_realtype_26853[5] = { 1,0,1,0,0};
static bool static_array_realtype_26854[5] = { 0,0,0,0,0};
struct tuning_params {
    int dummy;
};
static const int num_tuning_params = 0;
static const char *tuning_param_names[] = {NULL};
static const char *tuning_param_vars[] = {NULL};
static const char *tuning_param_classes[] = {NULL};
static int64_t tuning_param_defaults[] = {0};
// Start of backends/c.h

struct futhark_context_config {
  int in_use;
  int debugging;
  int profiling;
  int logging;
  char *cache_fname;
  int num_tuning_params;
  int64_t *tuning_params;
  const char** tuning_param_names;
  const char** tuning_param_vars;
  const char** tuning_param_classes;
};

static void backend_context_config_setup(struct futhark_context_config* cfg) {
  (void)cfg;
}

static void backend_context_config_teardown(struct futhark_context_config* cfg) {
  (void)cfg;
}

int futhark_context_config_set_tuning_param(struct futhark_context_config* cfg, const char *param_name, size_t param_value) {
  (void)cfg; (void)param_name; (void)param_value;
  return 1;
}

struct futhark_context {
  struct futhark_context_config* cfg;
  int detail_memory;
  int debugging;
  int profiling;
  int profiling_paused;
  int logging;
  lock_t lock;
  char *error;
  lock_t error_lock;
  FILE *log;
  struct constants *constants;
  struct free_list free_list;
  struct event_list event_list;
  int64_t peak_mem_usage_default;
  int64_t cur_mem_usage_default;
  struct program* program;
  bool program_initialised;
};

int backend_context_setup(struct futhark_context* ctx) {
  (void)ctx;
  return 0;
}

void backend_context_teardown(struct futhark_context* ctx) {
  (void)ctx;
}

int futhark_context_sync(struct futhark_context* ctx) {
  (void)ctx;
  return 0;
}

// End of backends/c.h

struct program {
    int dummy;
};
static void setup_program(struct futhark_context *ctx)
{
    (void) ctx;
    
    int error = 0;
    
    (void) error;
    ctx->program = malloc(sizeof(struct program));
}
static void teardown_program(struct futhark_context *ctx)
{
    (void) ctx;
    
    int error = 0;
    
    (void) error;
    free(ctx->program);
}
static void set_tuning_params(struct futhark_context *ctx)
{
    (void) ctx;
}
int memblock_unref(struct futhark_context *ctx, struct memblock *block, const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(ctx->log, "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n", desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            host_free(ctx, (size_t) block->size, desc, (void *) block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(ctx->log, "%lld bytes freed (now allocated: %lld bytes)\n", (long long) block->size, (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
    return 0;
}
int memblock_alloc(struct futhark_context *ctx, struct memblock *block, int64_t size, const char *desc)
{
    if (size < 0)
        futhark_panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n", (long long) size, desc, "default space", ctx->cur_mem_usage_default);
    
    int ret = memblock_unref(ctx, block, desc);
    
    if (ret != FUTHARK_SUCCESS)
        return ret;
    if (ctx->detail_memory)
        fprintf(ctx->log, "Allocating %lld bytes for %s in %s (currently allocated: %lld bytes).\n", (long long) size, desc, "default space", (long long) ctx->cur_mem_usage_default);
    host_alloc(ctx, (size_t) size, desc, (size_t *) &size, (void *) &block->mem);
    if (ctx->error == NULL) {
        block->references = (int *) malloc(sizeof(int));
        *block->references = 1;
        block->size = size;
        block->desc = desc;
        
        long long new_usage = ctx->cur_mem_usage_default + size;
        
        if (ctx->detail_memory)
            fprintf(ctx->log, "Received block of %lld bytes; now allocated: %lld bytes", (long long) block->size, new_usage);
        ctx->cur_mem_usage_default = new_usage;
        if (new_usage > ctx->peak_mem_usage_default) {
            ctx->peak_mem_usage_default = new_usage;
            if (ctx->detail_memory)
                fprintf(ctx->log, " (new peak).\n");
        } else if (ctx->detail_memory)
            fprintf(ctx->log, ".\n");
        return FUTHARK_SUCCESS;
    } else {
        // We are naively assuming that any memory allocation error is due to OOM.
        lock_lock(&ctx->error_lock);
        
        char *old_error = ctx->error;
        
        ctx->error = msgprintf("Failed to allocate memory in %s.\nAttempted allocation: %12lld bytes\nCurrently allocated:  %12lld bytes\n%s", "default space", (long long) size, (long long) ctx->cur_mem_usage_default, old_error);
        free(old_error);
        lock_unlock(&ctx->error_lock);
        return FUTHARK_OUT_OF_MEMORY;
    }
}
int memblock_set(struct futhark_context *ctx, struct memblock *lhs, struct memblock *rhs, const char *lhs_desc)
{
    int ret = memblock_unref(ctx, lhs, lhs_desc);
    
    if (rhs->references != NULL)
        (*rhs->references)++;
    *lhs = *rhs;
    return ret;
}
char *futhark_context_report(struct futhark_context *ctx)
{
    if (futhark_context_sync(ctx) != 0)
        return NULL;
    
    struct str_builder builder;
    
    str_builder_init(&builder);
    str_builder_char(&builder, '{');
    str_builder_str(&builder, "\"memory\":{");
    str_builder(&builder, "\"default space\": %lld", (long long) ctx->peak_mem_usage_default);
    str_builder_str(&builder, "},\"events\":[");
    if (report_events_in_list(&ctx->event_list, &builder) != 0) {
        free(builder.str);
        return NULL;
    } else {
        str_builder_str(&builder, "]}");
        return builder.str;
    }
}
int futhark_context_clear_caches(struct futhark_context *ctx)
{
    lock_lock(&ctx->lock);
    ctx->peak_mem_usage_default = 0;
    lock_unlock(&ctx->lock);
    return ctx->error != NULL;
}

// Start of context.h

// Internal functions.

static void set_error(struct futhark_context* ctx, char *error) {
  lock_lock(&ctx->error_lock);
  if (ctx->error == NULL) {
    ctx->error = error;
  } else {
    free(error);
  }
  lock_unlock(&ctx->error_lock);
}

// XXX: should be static, but used in ispc_util.h
void lexical_realloc_error(struct futhark_context* ctx, size_t new_size) {
  set_error(ctx,
            msgprintf("Failed to allocate memory.\nAttempted allocation: %12lld bytes\n",
                      (long long) new_size));
}

static int lexical_realloc(struct futhark_context *ctx,
                           unsigned char **ptr,
                           int64_t *old_size,
                           int64_t new_size) {
  unsigned char *new = realloc(*ptr, (size_t)new_size);
  if (new == NULL) {
    lexical_realloc_error(ctx, new_size);
    return FUTHARK_OUT_OF_MEMORY;
  } else {
    *ptr = new;
    *old_size = new_size;
    return FUTHARK_SUCCESS;
  }
}

static void free_all_in_free_list(struct futhark_context* ctx) {
  fl_mem mem;
  free_list_pack(&ctx->free_list);
  while (free_list_first(&ctx->free_list, (fl_mem*)&mem) == 0) {
    free((void*)mem);
  }
}

static int is_small_alloc(size_t size) {
  return size < 1024*1024;
}

static void host_alloc(struct futhark_context* ctx,
                       size_t size, const char* tag, size_t* size_out, void** mem_out) {
  if (is_small_alloc(size) || free_list_find(&ctx->free_list, size, tag, size_out, (fl_mem*)mem_out) != 0) {
    *size_out = size;
    *mem_out = malloc(size);
  }
}

static void host_free(struct futhark_context* ctx,
                      size_t size, const char* tag, void* mem) {
  // Small allocations are handled by malloc()s own free list.  The
  // threshold here is kind of arbitrary, but seems to work OK.
  // Larger allocations are mmap()ed/munmapped() every time, which is
  // very slow, and Futhark programs tend to use a few very large
  // allocations.
  if (is_small_alloc(size)) {
    free(mem);
  } else {
    free_list_insert(&ctx->free_list, size, (fl_mem)mem, tag);
  }
}

static void add_event(struct futhark_context* ctx,
                      const char* name,
                      const char* provenance,
                      struct kvs *kvs,
                      void* data,
                      event_report_fn f) {
  if (provenance == NULL) {
    provenance = "unknown";
  }
  if (ctx->logging) {
    fprintf(ctx->log, "Event: %s\n  at: %s\n", name, provenance);
    if (kvs) {
      kvs_log(kvs, "  ", ctx->log);
    }
  }
  add_event_to_list(&ctx->event_list, name, provenance, kvs, data, f);
}

char *futhark_context_get_error(struct futhark_context *ctx) {
  char *error = ctx->error;
  ctx->error = NULL;
  return error;
}

void futhark_context_config_set_debugging(struct futhark_context_config *cfg, int flag) {
    cfg->profiling = cfg->logging = cfg->debugging = flag;
}

void futhark_context_config_set_profiling(struct futhark_context_config *cfg, int flag) {
    cfg->profiling = flag;
}

void futhark_context_config_set_logging(struct futhark_context_config *cfg, int flag) {
    cfg->logging = flag;
}

void futhark_context_config_set_cache_file(struct futhark_context_config *cfg, const char *f) {
  cfg->cache_fname = strdup(f);
}

int futhark_get_tuning_param_count(void) {
  return num_tuning_params;
}

const char *futhark_get_tuning_param_name(int i) {
  return tuning_param_names[i];
}

const char *futhark_get_tuning_param_class(int i) {
    return tuning_param_classes[i];
}

void futhark_context_set_logging_file(struct futhark_context *ctx, FILE *f){
  ctx->log = f;
}

void futhark_context_pause_profiling(struct futhark_context *ctx) {
  ctx->profiling_paused = 1;
}

void futhark_context_unpause_profiling(struct futhark_context *ctx) {
  ctx->profiling_paused = 0;
}

struct futhark_context_config* futhark_context_config_new(void) {
  struct futhark_context_config* cfg = malloc(sizeof(struct futhark_context_config));
  if (cfg == NULL) {
    return NULL;
  }
  cfg->in_use = 0;
  cfg->debugging = 0;
  cfg->profiling = 0;
  cfg->logging = 0;
  cfg->cache_fname = NULL;
  cfg->num_tuning_params = num_tuning_params;
  cfg->tuning_params = malloc(cfg->num_tuning_params * sizeof(int64_t));
  memcpy(cfg->tuning_params, tuning_param_defaults,
         cfg->num_tuning_params * sizeof(int64_t));
  cfg->tuning_param_names = tuning_param_names;
  cfg->tuning_param_vars = tuning_param_vars;
  cfg->tuning_param_classes = tuning_param_classes;
  backend_context_config_setup(cfg);
  return cfg;
}

void futhark_context_config_free(struct futhark_context_config* cfg) {
  assert(!cfg->in_use);
  backend_context_config_teardown(cfg);
  free(cfg->cache_fname);
  free(cfg->tuning_params);
  free(cfg);
}

struct futhark_context* futhark_context_new(struct futhark_context_config* cfg) {
  struct futhark_context* ctx = malloc(sizeof(struct futhark_context));
  if (ctx == NULL) {
    return NULL;
  }
  assert(!cfg->in_use);
  ctx->cfg = cfg;
  ctx->cfg->in_use = 1;
  ctx->program_initialised = false;
  create_lock(&ctx->error_lock);
  create_lock(&ctx->lock);
  free_list_init(&ctx->free_list);
  event_list_init(&ctx->event_list);
  ctx->peak_mem_usage_default = 0;
  ctx->cur_mem_usage_default = 0;
  ctx->constants = malloc(sizeof(struct constants));
  ctx->debugging = cfg->debugging;
  ctx->logging = cfg->logging;
  ctx->detail_memory = cfg->logging;
  ctx->profiling = cfg->profiling;
  ctx->profiling_paused = 0;
  ctx->error = NULL;
  ctx->log = stderr;
  set_tuning_params(ctx);
  if (backend_context_setup(ctx) == 0) {
    setup_program(ctx);
    init_constants(ctx);
    ctx->program_initialised = true;
    (void)futhark_context_clear_caches(ctx);
    (void)futhark_context_sync(ctx);
  }
  return ctx;
}

void futhark_context_free(struct futhark_context* ctx) {
  if (ctx->program_initialised) {
    free_constants(ctx);
    teardown_program(ctx);
  }
  backend_context_teardown(ctx);
  free_all_in_free_list(ctx);
  free_list_destroy(&ctx->free_list);
  event_list_free(&ctx->event_list);
  free(ctx->constants);
  free(ctx->error);
  free_lock(&ctx->lock);
  free_lock(&ctx->error_lock);
  ctx->cfg->in_use = 0;
  free(ctx);
}

// End of context.h

// Start of copy.h

// Cache-oblivious map-transpose function.
#define GEN_MAP_TRANSPOSE(NAME, ELEM_TYPE)                              \
  static void map_transpose_##NAME                                      \
  (ELEM_TYPE* dst, ELEM_TYPE* src,                                      \
   int64_t k, int64_t m, int64_t n,                                     \
   int64_t cb, int64_t ce, int64_t rb, int64_t re)                      \
  {                                                                     \
  int32_t r = re - rb;                                                  \
  int32_t c = ce - cb;                                                  \
  if (k == 1) {                                                         \
    if (r <= 64 && c <= 64) {                                           \
      for (int64_t j = 0; j < c; j++) {                                 \
        for (int64_t i = 0; i < r; i++) {                               \
          dst[(j + cb) * n + (i + rb)] = src[(i + rb) * m + (j + cb)];  \
        }                                                               \
      }                                                                 \
    } else if (c <= r) {                                                \
      map_transpose_##NAME(dst, src, k, m, n, cb, ce, rb, rb + r/2);    \
      map_transpose_##NAME(dst, src, k, m, n, cb, ce, rb + r/2, re);    \
    } else {                                                            \
      map_transpose_##NAME(dst, src, k, m, n, cb, cb + c/2, rb, re);    \
      map_transpose_##NAME(dst, src, k, m, n, cb + c/2, ce, rb, re);    \
    }                                                                   \
  } else {                                                              \
  for (int64_t i = 0; i < k; i++) {                                     \
    map_transpose_##NAME(dst + i * m * n, src + i * m * n, 1, m, n, cb, ce, rb, re); \
  }\
} \
}

// Straightforward LMAD copy function.
#define GEN_LMAD_COPY_ELEMENTS(NAME, ELEM_TYPE)                         \
  static void lmad_copy_elements_##NAME(int r,                          \
                                        ELEM_TYPE* dst, int64_t dst_strides[r], \
                                        ELEM_TYPE *src, int64_t src_strides[r], \
                                        int64_t shape[r]) {             \
    if (r == 1) {                                                       \
      for (int i = 0; i < shape[0]; i++) {                              \
        dst[i*dst_strides[0]] = src[i*src_strides[0]];                  \
      }                                                                 \
    } else if (r > 1) {                                                 \
      for (int i = 0; i < shape[0]; i++) {                              \
        lmad_copy_elements_##NAME(r-1,                                  \
                                  dst+i*dst_strides[0], dst_strides+1,  \
                                  src+i*src_strides[0], src_strides+1,  \
                                  shape+1);                             \
      }                                                                 \
    }                                                                   \
  }                                                                     \

// Check whether this LMAD can be seen as a transposed 2D array.  This
// is done by checking every possible splitting point.
static bool lmad_is_tr(int64_t *n_out, int64_t *m_out,
                       int r,
                       const int64_t strides[r],
                       const int64_t shape[r]) {
  for (int i = 1; i < r; i++) {
    int n = 1, m = 1;
    bool ok = true;
    int64_t expected = 1;
    // Check strides before 'i'.
    for (int j = i-1; j >= 0; j--) {
      ok = ok && strides[j] == expected;
      expected *= shape[j];
      n *= shape[j];
    }
    // Check strides after 'i'.
    for (int j = r-1; j >= i; j--) {
      ok = ok && strides[j] == expected;
      expected *= shape[j];
      m *= shape[j];
    }
    if (ok) {
      *n_out = n;
      *m_out = m;
      return true;
    }
  }
  return false;
}

// This function determines whether the a 'dst' LMAD is row-major and
// 'src' LMAD is column-major.  Both LMADs are for arrays of the same
// shape.  Both LMADs are allowed to have additional dimensions "on
// top".  Essentially, this function determines whether a copy from
// 'src' to 'dst' is a "map(transpose)" that we know how to implement
// efficiently.  The LMADs can have arbitrary rank, and the main
// challenge here is checking whether the src LMAD actually
// corresponds to a 2D column-major layout by morally collapsing
// dimensions.  There is a lot of looping here, but the actual trip
// count is going to be very low in practice.
//
// Returns true if this is indeed a map(transpose), and writes the
// number of arrays, and moral array size to appropriate output
// parameters.
static bool lmad_map_tr(int64_t *num_arrays_out, int64_t *n_out, int64_t *m_out,
                        int r,
                        const int64_t dst_strides[r],
                        const int64_t src_strides[r],
                        const int64_t shape[r]) {
  int64_t rowmajor_strides[r];
  rowmajor_strides[r-1] = 1;

  for (int i = r-2; i >= 0; i--) {
    rowmajor_strides[i] = rowmajor_strides[i+1] * shape[i+1];
  }

  // map_r will be the number of mapped dimensions on top.
  int map_r = 0;
  int64_t num_arrays = 1;
  for (int i = 0; i < r; i++) {
    if (dst_strides[i] != rowmajor_strides[i] ||
        src_strides[i] != rowmajor_strides[i]) {
      break;
    } else {
      num_arrays *= shape[i];
      map_r++;
    }
  }

  *num_arrays_out = num_arrays;

  if (r==map_r) {
    return false;
  }

  if (memcmp(&rowmajor_strides[map_r],
             &dst_strides[map_r],
             sizeof(int64_t)*(r-map_r)) == 0) {
    return lmad_is_tr(n_out, m_out, r-map_r, src_strides+map_r, shape+map_r);
  } else if (memcmp(&rowmajor_strides[map_r],
                    &src_strides[map_r],
                    sizeof(int64_t)*(r-map_r)) == 0) {
    return lmad_is_tr(m_out, n_out, r-map_r, dst_strides+map_r, shape+map_r);
  }
  return false;
}

// Check if the strides correspond to row-major strides of *any*
// permutation of the shape.  This is done by recursive search with
// backtracking.  This is worst-case exponential, but hopefully the
// arrays we encounter do not have that many dimensions.
static bool lmad_contiguous_search(int checked, int64_t expected,
                                   int r,
                                   int64_t strides[r], int64_t shape[r], bool used[r]) {
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < r; j++) {
      if (!used[j] && strides[j] == expected && strides[j] >= 0) {
        used[j] = true;
        if (checked+1 == r ||
            lmad_contiguous_search(checked+1, expected * shape[j], r, strides, shape, used)) {
          return true;
        }
        used[j] = false;
      }
    }
  }
  return false;
}

// Does this LMAD correspond to an array with positive strides and no
// holes?
static bool lmad_contiguous(int r, int64_t strides[r], int64_t shape[r]) {
  bool used[r];
  for (int i = 0; i < r; i++) {
    used[i] = false;
  }
  return lmad_contiguous_search(0, 1, r, strides, shape, used);
}

// Does this copy correspond to something that could be done with a
// memcpy()-like operation?  I.e. do the LMADs actually represent the
// same in-memory layout and are they contiguous?
static bool lmad_memcpyable(int r,
                            int64_t dst_strides[r], int64_t src_strides[r], int64_t shape[r]) {
  if (!lmad_contiguous(r, dst_strides, shape)) {
    return false;
  }
  for (int i = 0; i < r; i++) {
    if (dst_strides[i] != src_strides[i] && shape[i] != 1) {
      return false;
    }
  }
  return true;
}


static void log_copy(struct futhark_context* ctx,
                     const char *kind, const char *provenance,
                     int r,
                     int64_t dst_offset, int64_t dst_strides[r],
                     int64_t src_offset, int64_t src_strides[r],
                     int64_t shape[r]) {
  if (ctx->logging) {
    fprintf(ctx->log, "\n# Copy %s\n", kind);
    if (provenance) { fprintf(ctx->log, "At: %s\n", provenance); }
    fprintf(ctx->log, "Shape: ");
    for (int i = 0; i < r; i++) { fprintf(ctx->log, "[%ld]", (long int)shape[i]); }
    fprintf(ctx->log, "\n");
    fprintf(ctx->log, "Dst offset: %ld\n", (long int)dst_offset);
    fprintf(ctx->log, "Dst strides:");
    for (int i = 0; i < r; i++) { fprintf(ctx->log, " %ld", (long int)dst_strides[i]); }
    fprintf(ctx->log, "\n");
    fprintf(ctx->log, "Src offset: %ld\n", (long int)src_offset);
    fprintf(ctx->log, "Src strides:");
    for (int i = 0; i < r; i++) { fprintf(ctx->log, " %ld", (long int)src_strides[i]); }
    fprintf(ctx->log, "\n");
  }
}

static void log_transpose(struct futhark_context* ctx,
                          int64_t k, int64_t n, int64_t m) {
  if (ctx->logging) {
    fprintf(ctx->log, "## Transpose\n");
    fprintf(ctx->log, "Arrays     : %ld\n", (long int)k);
    fprintf(ctx->log, "X elements : %ld\n", (long int)m);
    fprintf(ctx->log, "Y elements : %ld\n", (long int)n);
    fprintf(ctx->log, "\n");
  }
}

#define GEN_LMAD_COPY(NAME, ELEM_TYPE)                                  \
  static void lmad_copy_##NAME                                          \
  (struct futhark_context *ctx, int r,                                  \
   ELEM_TYPE* dst, int64_t dst_offset, int64_t dst_strides[r],          \
   ELEM_TYPE *src, int64_t src_offset, int64_t src_strides[r],          \
   int64_t shape[r]) {                                                  \
    log_copy(ctx, "CPU to CPU", NULL, r, dst_offset, dst_strides,       \
             src_offset, src_strides, shape);                           \
    int64_t size = 1;                                                   \
    for (int i = 0; i < r; i++) { size *= shape[i]; }                   \
    if (size == 0) { return; }                                          \
    int64_t k, n, m;                                                    \
    if (lmad_map_tr(&k, &n, &m,                                         \
                    r, dst_strides, src_strides, shape)) {              \
      log_transpose(ctx, k, n, m);                                      \
      map_transpose_##NAME                                              \
        (dst+dst_offset, src+src_offset, k, n, m, 0, n, 0, m);          \
    } else if (lmad_memcpyable(r, dst_strides, src_strides, shape)) {   \
      if (ctx->logging) {fprintf(ctx->log, "## Flat copy\n\n");}          \
      memcpy(dst+dst_offset, src+src_offset, size*sizeof(*dst));        \
    } else {                                                            \
      if (ctx->logging) {fprintf(ctx->log, "## General copy\n\n");}       \
      lmad_copy_elements_##NAME                                         \
        (r,                                                             \
         dst+dst_offset, dst_strides,                                   \
         src+src_offset, src_strides, shape);                           \
    }                                                                   \
  }

GEN_MAP_TRANSPOSE(1b, uint8_t)
GEN_MAP_TRANSPOSE(2b, uint16_t)
GEN_MAP_TRANSPOSE(4b, uint32_t)
GEN_MAP_TRANSPOSE(8b, uint64_t)

GEN_LMAD_COPY_ELEMENTS(1b, uint8_t)
GEN_LMAD_COPY_ELEMENTS(2b, uint16_t)
GEN_LMAD_COPY_ELEMENTS(4b, uint32_t)
GEN_LMAD_COPY_ELEMENTS(8b, uint64_t)

GEN_LMAD_COPY(1b, uint8_t)
GEN_LMAD_COPY(2b, uint16_t)
GEN_LMAD_COPY(4b, uint32_t)
GEN_LMAD_COPY(8b, uint64_t)

// End of copy.h

#define FUTHARK_FUN_ATTR static

FUTHARK_FUN_ATTR int futrts_csr_rows_from_pattern_7271(struct futhark_context *ctx, struct memblock *mem_out_p_26855, struct memblock *mem_out_p_26856, int64_t *out_prim_out_26857, int64_t *out_prim_out_26858, struct memblock pat_mem_26563, int64_t m_13878, int64_t n_13879);
FUTHARK_FUN_ATTR int futrts_entry_test_jvp_csr_from_csr_ex4_matches_dense(struct futhark_context *ctx, bool *out_prim_out_26862, struct memblock x_mem_26563);
FUTHARK_FUN_ATTR int futrts_entry_test_prepared_jvp_ex4_matches_dense(struct futhark_context *ctx, bool *out_prim_out_26880, struct memblock x_mem_26563);
FUTHARK_FUN_ATTR int futrts_entry_test_prepared_jvp_reuse_two_points(struct futhark_context *ctx, bool *out_prim_out_26898, struct memblock x1_mem_26563);
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_jvp_ex1_matches_dense(struct futhark_context *ctx, bool *out_prim_out_26923, struct memblock x_mem_26563);
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_jvp_ex2_matches_dense(struct futhark_context *ctx, bool *out_prim_out_26941, struct memblock x_mem_26563);
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_jvp_ex4_matches_dense(struct futhark_context *ctx, bool *out_prim_out_26959, struct memblock x_mem_26563);
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_jvp_ex4_with_colors_matches_dense(struct futhark_context *ctx, bool *out_prim_out_26977, struct memblock x_mem_26563);
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_jvp_ex5_matches_dense(struct futhark_context *ctx, bool *out_prim_out_26984, struct memblock x_mem_26563);
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_jvp_zzero_pattern_matches_dense(struct futhark_context *ctx, bool *out_prim_out_27002, struct memblock x_mem_26563);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_26545 (ctx->constants->mem_26545)
    #define mem_26548 (ctx->constants->mem_26548)
    #define mem_26550 (ctx->constants->mem_26550)
    #define mem_26555 (ctx->constants->mem_26555)
    #define mem_26556 (ctx->constants->mem_26556)
    #define mem_26562 (ctx->constants->mem_26562)
    
    struct memblock mem_26561;
    
    mem_26561.references = NULL;
    
    struct memblock mem_26560;
    
    mem_26560.references = NULL;
    
    struct memblock mem_26559;
    
    mem_26559.references = NULL;
    
    struct memblock mem_26558;
    
    mem_26558.references = NULL;
    
    struct memblock mem_26557;
    
    mem_26557.references = NULL;
    
    struct memblock mem_26554;
    
    mem_26554.references = NULL;
    
    struct memblock mem_26553;
    
    mem_26553.references = NULL;
    
    struct memblock mem_26552;
    
    mem_26552.references = NULL;
    
    struct memblock mem_26551;
    
    mem_26551.references = NULL;
    
    struct memblock mem_26549;
    
    mem_26549.references = NULL;
    
    struct memblock mem_26547;
    
    mem_26547.references = NULL;
    
    struct memblock mem_26546;
    
    mem_26546.references = NULL;
    
    struct memblock mem_26544;
    
    mem_26544.references = NULL;
    
    struct memblock mem_26543;
    
    mem_26543.references = NULL;
    
    struct memblock mem_26542;
    
    mem_26542.references = NULL;
    mem_26545.references = NULL;
    mem_26548.references = NULL;
    mem_26550.references = NULL;
    mem_26555.references = NULL;
    mem_26556.references = NULL;
    mem_26562.references = NULL;
    // test/test_sparse_jacobian_jvp.fut:25:5-40
    if (memblock_alloc(ctx, &mem_26542, (int64_t) 5, "mem_26542")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:25:5-40
    
    struct memblock static_array_26822 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26839, 0, "static_array_26822"};
    
    // test/test_sparse_jacobian_jvp.fut:25:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26542.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_26822.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    // test/test_sparse_jacobian_jvp.fut:26:5-40
    if (memblock_alloc(ctx, &mem_26543, (int64_t) 5, "mem_26543")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:26:5-40
    
    struct memblock static_array_26823 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26840, 0, "static_array_26823"};
    
    // test/test_sparse_jacobian_jvp.fut:26:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26543.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_26823.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    // test/test_sparse_jacobian_jvp.fut:27:5-40
    if (memblock_alloc(ctx, &mem_26544, (int64_t) 5, "mem_26544")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:27:5-40
    
    struct memblock static_array_26824 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26841, 0, "static_array_26824"};
    
    // test/test_sparse_jacobian_jvp.fut:27:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26544.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_26824.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    if (memblock_alloc(ctx, &mem_26545, (int64_t) 15, "mem_26545")) {
        err = 1;
        goto cleanup;
    }
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26545.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_26542.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26545.mem, (int64_t) 5, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_26543.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26545.mem, (int64_t) 10, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_26544.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    if (memblock_unref(ctx, &mem_26542, "mem_26542") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26543, "mem_26543") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26544, "mem_26544") != 0)
        return 1;
    // test/test_sparse_jacobian_jvp.fut:47:5-33
    if (memblock_alloc(ctx, &mem_26546, (int64_t) 4, "mem_26546")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:47:5-33
    
    struct memblock static_array_26825 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26842, 0, "static_array_26825"};
    
    // test/test_sparse_jacobian_jvp.fut:47:5-33
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26546.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_26825.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
    // test/test_sparse_jacobian_jvp.fut:48:5-33
    if (memblock_alloc(ctx, &mem_26547, (int64_t) 4, "mem_26547")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:48:5-33
    
    struct memblock static_array_26826 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26843, 0, "static_array_26826"};
    
    // test/test_sparse_jacobian_jvp.fut:48:5-33
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26547.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_26826.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
    if (memblock_alloc(ctx, &mem_26548, (int64_t) 8, "mem_26548")) {
        err = 1;
        goto cleanup;
    }
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26548.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_26546.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26548.mem, (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_26547.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
    if (memblock_unref(ctx, &mem_26546, "mem_26546") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26547, "mem_26547") != 0)
        return 1;
    // test/test_sparse_jacobian_jvp.fut:66:5-26
    if (memblock_alloc(ctx, &mem_26549, (int64_t) 3, "mem_26549")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:66:5-26
    
    struct memblock static_array_26827 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26844, 0, "static_array_26827"};
    
    // test/test_sparse_jacobian_jvp.fut:66:5-26
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26549.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_26827.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 3});
    if (memblock_alloc(ctx, &mem_26550, (int64_t) 6, "mem_26550")) {
        err = 1;
        goto cleanup;
    }
    for (int64_t nest_i_26828 = 0; nest_i_26828 < (int64_t) 2; nest_i_26828++) {
        lmad_copy_1b(ctx, 1, (uint8_t *) mem_26550.mem, nest_i_26828 * (int64_t) 3, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_26549.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 3});
    }
    if (memblock_unref(ctx, &mem_26549, "mem_26549") != 0)
        return 1;
    // test/test_sparse_jacobian_jvp.fut:89:5-47
    if (memblock_alloc(ctx, &mem_26551, (int64_t) 6, "mem_26551")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:89:5-47
    
    struct memblock static_array_26829 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26845, 0, "static_array_26829"};
    
    // test/test_sparse_jacobian_jvp.fut:89:5-47
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26551.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_26829.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    // test/test_sparse_jacobian_jvp.fut:90:5-47
    if (memblock_alloc(ctx, &mem_26552, (int64_t) 6, "mem_26552")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:90:5-47
    
    struct memblock static_array_26830 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26846, 0, "static_array_26830"};
    
    // test/test_sparse_jacobian_jvp.fut:90:5-47
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26552.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_26830.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    // test/test_sparse_jacobian_jvp.fut:91:5-47
    if (memblock_alloc(ctx, &mem_26553, (int64_t) 6, "mem_26553")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:91:5-47
    
    struct memblock static_array_26831 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26847, 0, "static_array_26831"};
    
    // test/test_sparse_jacobian_jvp.fut:91:5-47
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26553.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_26831.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    // test/test_sparse_jacobian_jvp.fut:92:5-47
    if (memblock_alloc(ctx, &mem_26554, (int64_t) 6, "mem_26554")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:92:5-47
    
    struct memblock static_array_26832 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26848, 0, "static_array_26832"};
    
    // test/test_sparse_jacobian_jvp.fut:92:5-47
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26554.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_26832.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    if (memblock_alloc(ctx, &mem_26555, (int64_t) 24, "mem_26555")) {
        err = 1;
        goto cleanup;
    }
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26555.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_26551.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26555.mem, (int64_t) 6, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_26552.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26555.mem, (int64_t) 12, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_26553.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26555.mem, (int64_t) 18, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_26554.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    if (memblock_unref(ctx, &mem_26551, "mem_26551") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26552, "mem_26552") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26553, "mem_26553") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26554, "mem_26554") != 0)
        return 1;
    // test/test_sparse_jacobian_jvp.fut:95:27-63
    if (memblock_alloc(ctx, &mem_26556, (int64_t) 48, "mem_26556")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:95:27-63
    
    struct memblock static_array_26833 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26849, 0, "static_array_26833"};
    
    // test/test_sparse_jacobian_jvp.fut:95:27-63
    lmad_copy_8b(ctx, 1, (uint64_t *) mem_26556.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) static_array_26833.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    // test/test_sparse_jacobian_jvp.fut:127:5-40
    if (memblock_alloc(ctx, &mem_26557, (int64_t) 5, "mem_26557")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:127:5-40
    
    struct memblock static_array_26834 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26850, 0, "static_array_26834"};
    
    // test/test_sparse_jacobian_jvp.fut:127:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26557.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_26834.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    // test/test_sparse_jacobian_jvp.fut:128:5-40
    if (memblock_alloc(ctx, &mem_26558, (int64_t) 5, "mem_26558")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:128:5-40
    
    struct memblock static_array_26835 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26851, 0, "static_array_26835"};
    
    // test/test_sparse_jacobian_jvp.fut:128:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26558.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_26835.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    // test/test_sparse_jacobian_jvp.fut:129:5-40
    if (memblock_alloc(ctx, &mem_26559, (int64_t) 5, "mem_26559")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:129:5-40
    
    struct memblock static_array_26836 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26852, 0, "static_array_26836"};
    
    // test/test_sparse_jacobian_jvp.fut:129:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26559.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_26836.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    // test/test_sparse_jacobian_jvp.fut:130:5-40
    if (memblock_alloc(ctx, &mem_26560, (int64_t) 5, "mem_26560")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:130:5-40
    
    struct memblock static_array_26837 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26853, 0, "static_array_26837"};
    
    // test/test_sparse_jacobian_jvp.fut:130:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26560.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_26837.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    // test/test_sparse_jacobian_jvp.fut:131:5-40
    if (memblock_alloc(ctx, &mem_26561, (int64_t) 5, "mem_26561")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:131:5-40
    
    struct memblock static_array_26838 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_26854, 0, "static_array_26838"};
    
    // test/test_sparse_jacobian_jvp.fut:131:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26561.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_26838.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    if (memblock_alloc(ctx, &mem_26562, (int64_t) 25, "mem_26562")) {
        err = 1;
        goto cleanup;
    }
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26562.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_26557.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26562.mem, (int64_t) 5, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_26558.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26562.mem, (int64_t) 10, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_26559.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26562.mem, (int64_t) 15, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_26560.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_26562.mem, (int64_t) 20, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_26561.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    if (memblock_unref(ctx, &mem_26557, "mem_26557") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26558, "mem_26558") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26559, "mem_26559") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26560, "mem_26560") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26561, "mem_26561") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26561, "mem_26561") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26560, "mem_26560") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26559, "mem_26559") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26558, "mem_26558") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26557, "mem_26557") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26554, "mem_26554") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26553, "mem_26553") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26552, "mem_26552") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26551, "mem_26551") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26549, "mem_26549") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26547, "mem_26547") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26546, "mem_26546") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26544, "mem_26544") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26543, "mem_26543") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26542, "mem_26542") != 0)
        return 1;
    #undef mem_26545
    #undef mem_26548
    #undef mem_26550
    #undef mem_26555
    #undef mem_26556
    #undef mem_26562
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_26545, "ctx->constants->mem_26545") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_26548, "ctx->constants->mem_26548") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_26550, "ctx->constants->mem_26550") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_26555, "ctx->constants->mem_26555") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_26556, "ctx->constants->mem_26556") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_26562, "ctx->constants->mem_26562") != 0)
        return 1;
    return 0;
}
struct futhark_f64_1d {
    struct memblock mem;
    int64_t shape[1];
};
struct futhark_f64_1d *futhark_new_f64_1d(struct futhark_context *ctx, const double *data, int64_t dim0)
{
    int err = 0;
    struct futhark_f64_1d *bad = NULL;
    struct futhark_f64_1d *arr = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    if (memblock_alloc(ctx, &arr->mem, dim0 * 8, "arr->mem"))
        err = 1;
    arr->shape[0] = dim0;
    if ((size_t) dim0 * 8 > 0)
        memmove(arr->mem.mem + 0, (const unsigned char *) data + 0, (size_t) dim0 * 8);
    lock_unlock(&ctx->lock);
    if (err != 0) {
        free(arr);
        return bad;
    }
    return arr;
}
struct futhark_f64_1d *futhark_new_raw_f64_1d(struct futhark_context *ctx, unsigned char *data, int64_t dim0)
{
    int err = 0;
    struct futhark_f64_1d *bad = NULL;
    struct futhark_f64_1d *arr = (struct futhark_f64_1d *) malloc(sizeof(struct futhark_f64_1d));
    
    if (arr == NULL)
        return bad;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    arr->mem.mem = data;
    arr->shape[0] = dim0;
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr)
{
    lock_lock(&ctx->lock);
    if (memblock_unref(ctx, &arr->mem, "arr->mem") != 0)
        return 1;
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr, double *data)
{
    int err = 0;
    
    lock_lock(&ctx->lock);
    if ((size_t) arr->shape[0] * 8 > 0)
        memmove((unsigned char *) data + 0, arr->mem.mem + 0, (size_t) arr->shape[0] * 8);
    lock_unlock(&ctx->lock);
    return err;
}
int futhark_index_f64_1d(struct futhark_context *ctx, double *out, struct futhark_f64_1d *arr, int64_t i0)
{
    int err = 0;
    
    if (i0 >= 0 && i0 < arr->shape[0]) {
        lock_lock(&ctx->lock);
        if (8 > 0)
            memmove((unsigned char *) out + 0, arr->mem.mem + 8 * (i0 * 1), 8);
        lock_unlock(&ctx->lock);
    } else {
        err = 1;
        set_error(ctx, strdup("Index out of bounds."));
    }
    return err;
}
unsigned char *futhark_values_raw_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr)
{
    (void) ctx;
    return arr->mem.mem;
}
const int64_t *futhark_shape_f64_1d(struct futhark_context *ctx, struct futhark_f64_1d *arr)
{
    (void) ctx;
    return arr->shape;
}

FUTHARK_FUN_ATTR int futrts_csr_rows_from_pattern_7271(struct futhark_context *ctx, struct memblock *mem_out_p_26855, struct memblock *mem_out_p_26856, int64_t *out_prim_out_26857, int64_t *out_prim_out_26858, struct memblock pat_mem_26563, int64_t m_13878, int64_t n_13879)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_26565_cached_sizze_26859 = 0;
    unsigned char *mem_26565 = NULL;
    int64_t mem_26575_cached_sizze_26860 = 0;
    unsigned char *mem_26575 = NULL;
    int64_t mem_26577_cached_sizze_26861 = 0;
    unsigned char *mem_26577 = NULL;
    struct memblock mem_26591;
    
    mem_26591.references = NULL;
    
    struct memblock mem_26573;
    
    mem_26573.references = NULL;
    
    struct memblock mem_out_26759;
    
    mem_out_26759.references = NULL;
    
    struct memblock mem_out_26758;
    
    mem_out_26758.references = NULL;
    
    struct memblock mem_26545 = ctx->constants->mem_26545;
    struct memblock mem_26548 = ctx->constants->mem_26548;
    struct memblock mem_26550 = ctx->constants->mem_26550;
    struct memblock mem_26555 = ctx->constants->mem_26555;
    struct memblock mem_26556 = ctx->constants->mem_26556;
    struct memblock mem_26562 = ctx->constants->mem_26562;
    int64_t prim_out_26760;
    int64_t prim_out_26761;
    int64_t dzlz7bUZLztZRz20Umz20Unz7dUzg_13881 = mul64(m_13878, n_13879);
    
    // src/pattern_csr.fut:7:23-40
    
    int64_t bytes_26564 = (int64_t) 8 * m_13878;
    
    // src/pattern_csr.fut:8:17-20
    
    int64_t dzlz7bUZLzpZRz20Umz20U1z7dUzg_21549 = add64((int64_t) 1, m_13878);
    
    // src/pattern_csr.fut:8:6-26
    
    int64_t bytes_26572 = (int64_t) 8 * dzlz7bUZLzpZRz20Umz20U1z7dUzg_21549;
    
    // src/pattern_csr.fut:8:6-46
    
    bool empty_slice_21556 = m_13878 == (int64_t) 0;
    
    // src/pattern_csr.fut:8:6-46
    
    bool i_p_m_t_s_leq_w_21557 = slt64(m_13878, dzlz7bUZLzpZRz20Umz20U1z7dUzg_21549);
    
    // src/pattern_csr.fut:8:6-46
    
    bool i_lte_j_21558 = sle64((int64_t) 1, dzlz7bUZLzpZRz20Umz20U1z7dUzg_21549);
    
    // src/pattern_csr.fut:8:6-46
    
    bool forwards_ok_21559 = i_p_m_t_s_leq_w_21557 && i_lte_j_21558;
    
    // src/pattern_csr.fut:8:6-46
    
    bool ok_or_empty_21560 = empty_slice_21556 || forwards_ok_21559;
    
    // src/pattern_csr.fut:8:6-46
    
    bool index_certs_21561;
    
    if (!ok_or_empty_21560) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) (int64_t) 1, ":", (long long) dzlz7bUZLzpZRz20Umz20U1z7dUzg_21549, "] out of bounds for array of shape [", (long long) dzlz7bUZLzpZRz20Umz20U1z7dUzg_21549, "].", "-> #0  src/pattern_csr.fut:8:6-46\n   #1  src/pattern_csr.fut:12:27-48\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    
    int64_t bytes_26574 = (int64_t) 8 * dzlz7bUZLztZRz20Umz20Unz7dUzg_13881;
    
    // src/pattern_csr.fut:19:34-61
    if (mem_26575_cached_sizze_26860 < bytes_26574) {
        err = lexical_realloc(ctx, &mem_26575, &mem_26575_cached_sizze_26860, bytes_26574);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    if (mem_26577_cached_sizze_26861 < bytes_26574) {
        err = lexical_realloc(ctx, &mem_26577, &mem_26577_cached_sizze_26861, bytes_26574);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    
    int64_t discard_26400;
    int64_t scanacc_26394 = (int64_t) 0;
    
    for (int64_t i_26397 = 0; i_26397 < dzlz7bUZLztZRz20Umz20Unz7dUzg_13881; i_26397++) {
        int64_t new_index_26494 = squot64(i_26397, n_13879);
        int64_t binop_y_26496 = n_13879 * new_index_26494;
        int64_t new_index_26497 = i_26397 - binop_y_26496;
        bool eta_p_24779 = ((bool *) pat_mem_26563.mem)[new_index_26494 * n_13879 + new_index_26497];
        
        // src/pattern_csr.fut:19:34-61
        
        int64_t defunc_0_f_res_24780 = btoi_bool_i64(eta_p_24779);
        
        // src/pattern_csr.fut:19:34-61
        
        int64_t defunc_0_op_res_21674 = add64(defunc_0_f_res_24780, scanacc_26394);
        
        ((int64_t *) mem_26575)[i_26397] = defunc_0_op_res_21674;
        ((int64_t *) mem_26577)[i_26397] = defunc_0_f_res_24780;
        
        int64_t scanacc_tmp_26762 = defunc_0_op_res_21674;
        
        scanacc_26394 = scanacc_tmp_26762;
    }
    discard_26400 = scanacc_26394;
    // src/pattern_csr.fut:16:43-46
    
    bool zzero_21364 = n_13879 == (int64_t) 0;
    
    // src/pattern_csr.fut:16:43-46
    
    bool nonzzero_21365 = !zzero_21364;
    
    // src/pattern_csr.fut:16:43-46
    
    bool nonzzero_cert_21366;
    
    if (!nonzzero_21365) {
        set_error(ctx, msgprintf("Error: %s\n\nBacktrace:\n%s", "division by zero", "-> #0  src/pattern_csr.fut:16:43-46\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    
    int64_t tmp_21677 = sub64(dzlz7bUZLztZRz20Umz20Unz7dUzg_13881, (int64_t) 1);
    
    // src/pattern_csr.fut:19:34-61
    
    bool y_21679 = slt64(tmp_21677, dzlz7bUZLztZRz20Umz20Unz7dUzg_13881);
    
    // src/pattern_csr.fut:19:34-61
    
    bool x_21678 = sle64((int64_t) 0, tmp_21677);
    
    // src/pattern_csr.fut:19:34-61
    
    bool bounds_check_21680 = x_21678 && y_21679;
    
    // src/pattern_csr.fut:19:34-61
    
    bool cond_21675 = dzlz7bUZLztZRz20Umz20Unz7dUzg_13881 == (int64_t) 0;
    
    // src/pattern_csr.fut:19:34-61
    
    bool protect_assert_disj_21681 = cond_21675 || bounds_check_21680;
    
    // src/pattern_csr.fut:19:34-61
    
    bool index_certs_21682;
    
    if (!protect_assert_disj_21681) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_21677, "] out of bounds for array of shape [", (long long) dzlz7bUZLztZRz20Umz20Unz7dUzg_13881, "].", "-> #0  src/pattern_csr.fut:19:34-61\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    
    bool x_21676 = !cond_21675;
    
    // src/pattern_csr.fut:19:34-61
    
    int64_t m_f_res_21683;
    
    if (x_21676) {
        // src/pattern_csr.fut:19:34-61
        
        int64_t x_25981 = ((int64_t *) mem_26575)[tmp_21677];
        
        m_f_res_21683 = x_25981;
    } else {
        m_f_res_21683 = (int64_t) 0;
    }
    // src/pattern_csr.fut:19:34-61
    
    int64_t m_21685;
    
    if (cond_21675) {
        m_21685 = (int64_t) 0;
    } else {
        m_21685 = m_f_res_21683;
    }
    // src/pattern_csr.fut:19:34-61
    
    int64_t bytes_26590 = (int64_t) 8 * m_21685;
    
    // src/pattern_csr.fut:7:23-40
    if (mem_26565_cached_sizze_26859 < bytes_26564) {
        err = lexical_realloc(ctx, &mem_26565, &mem_26565_cached_sizze_26859, bytes_26564);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/pattern_csr.fut:7:23-40
    
    int64_t discard_26391;
    int64_t scanacc_26387 = (int64_t) 0;
    
    for (int64_t i_26389 = 0; i_26389 < m_13878; i_26389++) {
        // src/pattern_csr.fut:4:3-57
        
        int64_t defunc_0_reduce_res_25980;
        int64_t redout_26384 = (int64_t) 0;
        
        for (int64_t i_26385 = 0; i_26385 < n_13879; i_26385++) {
            bool eta_p_24830 = ((bool *) pat_mem_26563.mem)[i_26389 * n_13879 + i_26385];
            
            // src/pattern_csr.fut:4:17-57
            
            int64_t lifted_lambda_res_24831 = btoi_bool_i64(eta_p_24830);
            
            // src/pattern_csr.fut:4:10-13
            
            int64_t defunc_0_op_res_24827 = add64(lifted_lambda_res_24831, redout_26384);
            int64_t redout_tmp_26767 = defunc_0_op_res_24827;
            
            redout_26384 = redout_tmp_26767;
        }
        defunc_0_reduce_res_25980 = redout_26384;
        // src/pattern_csr.fut:7:28-31
        
        int64_t defunc_0_op_res_21554 = add64(defunc_0_reduce_res_25980, scanacc_26387);
        
        ((int64_t *) mem_26565)[i_26389] = defunc_0_op_res_21554;
        
        int64_t scanacc_tmp_26765 = defunc_0_op_res_21554;
        
        scanacc_26387 = scanacc_tmp_26765;
    }
    discard_26391 = scanacc_26387;
    // src/pattern_csr.fut:8:6-26
    if (memblock_alloc(ctx, &mem_26573, bytes_26572, "mem_26573")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:8:6-26
    for (int64_t nest_i_26768 = 0; nest_i_26768 < dzlz7bUZLzpZRz20Umz20U1z7dUzg_21549; nest_i_26768++) {
        ((int64_t *) mem_26573.mem)[nest_i_26768] = (int64_t) 0;
    }
    // src/pattern_csr.fut:8:6-46
    // src/pattern_csr.fut:8:6-46
    lmad_copy_8b(ctx, 1, (uint64_t *) mem_26573.mem, (int64_t) 1, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_26565, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {m_13878});
    // src/pattern_csr.fut:19:34-61
    if (memblock_alloc(ctx, &mem_26591, bytes_26590, "mem_26591")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    
    bool acc_cert_24785;
    
    // src/pattern_csr.fut:16:30-19:61
    for (int64_t i_26402 = 0; i_26402 < dzlz7bUZLztZRz20Umz20Unz7dUzg_13881; i_26402++) {
        int64_t eta_p_24804 = ((int64_t *) mem_26577)[i_26402];
        int64_t eta_p_24805 = ((int64_t *) mem_26575)[i_26402];
        
        // src/pattern_csr.fut:16:43-46
        
        int64_t lifted_lambda_res_24807 = smod64(i_26402, n_13879);
        
        // src/pattern_csr.fut:19:34-61
        
        bool cond_24809 = eta_p_24804 == (int64_t) 1;
        
        // src/pattern_csr.fut:19:34-61
        
        int64_t lifted_lambda_res_24810;
        
        if (cond_24809) {
            // src/pattern_csr.fut:19:34-61
            
            int64_t lifted_lambda_res_t_res_25982 = sub64(eta_p_24805, (int64_t) 1);
            
            lifted_lambda_res_24810 = lifted_lambda_res_t_res_25982;
        } else {
            lifted_lambda_res_24810 = (int64_t) -1;
        }
        // src/pattern_csr.fut:19:34-61
        // UpdateAcc
        if (sle64((int64_t) 0, lifted_lambda_res_24810) && slt64(lifted_lambda_res_24810, m_21685)) {
            ((int64_t *) mem_26591.mem)[lifted_lambda_res_24810] = lifted_lambda_res_24807;
        }
    }
    if (memblock_set(ctx, &mem_out_26758, &mem_26573, "mem_26573") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_26759, &mem_26591, "mem_26591") != 0)
        return 1;
    prim_out_26760 = dzlz7bUZLzpZRz20Umz20U1z7dUzg_21549;
    prim_out_26761 = m_21685;
    if (memblock_set(ctx, &*mem_out_p_26855, &mem_out_26758, "mem_out_26758") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_26856, &mem_out_26759, "mem_out_26759") != 0)
        return 1;
    *out_prim_out_26857 = prim_out_26760;
    *out_prim_out_26858 = prim_out_26761;
    
  cleanup:
    {
        free(mem_26565);
        free(mem_26575);
        free(mem_26577);
        if (memblock_unref(ctx, &mem_26591, "mem_26591") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26573, "mem_26573") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_26759, "mem_out_26759") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_26758, "mem_out_26758") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_jvp_csr_from_csr_ex4_matches_dense(struct futhark_context *ctx, bool *out_prim_out_26862, struct memblock x_mem_26563)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_26564_cached_sizze_26863 = 0;
    unsigned char *mem_26564 = NULL;
    int64_t mem_26569_cached_sizze_26864 = 0;
    unsigned char *mem_26569 = NULL;
    int64_t mem_26578_cached_sizze_26865 = 0;
    unsigned char *mem_26578 = NULL;
    int64_t mem_26579_cached_sizze_26866 = 0;
    unsigned char *mem_26579 = NULL;
    int64_t mem_26588_cached_sizze_26867 = 0;
    unsigned char *mem_26588 = NULL;
    int64_t mem_26600_cached_sizze_26868 = 0;
    unsigned char *mem_26600 = NULL;
    int64_t mem_26604_cached_sizze_26869 = 0;
    unsigned char *mem_26604 = NULL;
    int64_t mem_26621_cached_sizze_26870 = 0;
    unsigned char *mem_26621 = NULL;
    int64_t mem_26623_cached_sizze_26871 = 0;
    unsigned char *mem_26623 = NULL;
    int64_t mem_26624_cached_sizze_26872 = 0;
    unsigned char *mem_26624 = NULL;
    int64_t mem_26644_cached_sizze_26873 = 0;
    unsigned char *mem_26644 = NULL;
    int64_t mem_26646_cached_sizze_26874 = 0;
    unsigned char *mem_26646 = NULL;
    int64_t mem_26648_cached_sizze_26875 = 0;
    unsigned char *mem_26648 = NULL;
    int64_t mem_26656_cached_sizze_26876 = 0;
    unsigned char *mem_26656 = NULL;
    int64_t mem_26667_cached_sizze_26877 = 0;
    unsigned char *mem_26667 = NULL;
    int64_t mem_26672_cached_sizze_26878 = 0;
    unsigned char *mem_26672 = NULL;
    int64_t mem_26698_cached_sizze_26879 = 0;
    unsigned char *mem_26698 = NULL;
    struct memblock mem_param_tmp_26763;
    
    mem_param_tmp_26763.references = NULL;
    
    struct memblock mem_26658;
    
    mem_26658.references = NULL;
    
    struct memblock mem_param_26598;
    
    mem_param_26598.references = NULL;
    
    struct memblock ext_mem_26663;
    
    ext_mem_26663.references = NULL;
    
    struct memblock ext_mem_26591;
    
    ext_mem_26591.references = NULL;
    
    struct memblock ext_mem_26592;
    
    ext_mem_26592.references = NULL;
    
    struct memblock mem_26589;
    
    mem_26589.references = NULL;
    
    struct memblock mem_26587;
    
    mem_26587.references = NULL;
    
    struct memblock ext_mem_26575;
    
    ext_mem_26575.references = NULL;
    
    struct memblock ext_mem_26576;
    
    ext_mem_26576.references = NULL;
    
    struct memblock mem_26545 = ctx->constants->mem_26545;
    struct memblock mem_26548 = ctx->constants->mem_26548;
    struct memblock mem_26550 = ctx->constants->mem_26550;
    struct memblock mem_26555 = ctx->constants->mem_26555;
    struct memblock mem_26556 = ctx->constants->mem_26556;
    struct memblock mem_26562 = ctx->constants->mem_26562;
    bool prim_out_26758;
    
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_23553;
    int64_t csr_bipartite_from_pattern_res_23554;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26576, &ext_mem_26575, &csr_bipartite_from_pattern_res_23553, &csr_bipartite_from_pattern_res_23554, mem_26555, (int64_t) 4, (int64_t) 6) != 0) {
        err = 1;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    
    int64_t bytes_26577 = (int64_t) 8 * csr_bipartite_from_pattern_res_23554;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (mem_26579_cached_sizze_26866 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26579, &mem_26579_cached_sizze_26866, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t discard_26393;
    int64_t scanacc_26389 = (int64_t) 0;
    
    for (int64_t i_26391 = 0; i_26391 < (int64_t) 6; i_26391++) {
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t defunc_0_op_res_23569 = add64((int64_t) 1, scanacc_26389);
        
        ((int64_t *) mem_26579)[i_26391] = defunc_0_op_res_23569;
        
        int64_t scanacc_tmp_26759 = defunc_0_op_res_23569;
        
        scanacc_26389 = scanacc_tmp_26759;
    }
    discard_26393 = scanacc_26389;
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t x_23579 = ((int64_t *) mem_26579)[(int64_t) 5];
    
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    bool loop_cond_23593 = slt64((int64_t) 0, x_23579);
    
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t bytes_26586 = (int64_t) 8 * x_23579;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (memblock_alloc(ctx, &mem_26587, bytes_26586, "mem_26587")) {
        err = 1;
        goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    bool acc_cert_24821;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    for (int64_t i_26395 = 0; i_26395 < (int64_t) 6; i_26395++) {
        int64_t eta_p_24833 = ((int64_t *) mem_26579)[i_26395];
        
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t lifted_lambda_res_t_res_24836 = sub64(eta_p_24833, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:194:20-45
        // UpdateAcc
        if (sle64((int64_t) 0, lifted_lambda_res_t_res_24836) && slt64(lifted_lambda_res_t_res_24836, x_23579)) {
            ((int64_t *) mem_26587.mem)[lifted_lambda_res_t_res_24836] = i_26395;
        }
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    if (mem_26588_cached_sizze_26867 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26588, &mem_26588_cached_sizze_26867, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    for (int64_t nest_i_26762 = 0; nest_i_26762 < (int64_t) 6; nest_i_26762++) {
        ((int64_t *) mem_26588)[nest_i_26762] = (int64_t) -1;
    }
    // src/pattern_csr.fut:25:3-39
    if (memblock_alloc(ctx, &mem_26589, (int64_t) 24, "mem_26589")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:25:3-39
    // src/pattern_csr.fut:25:3-39
    lmad_copy_1b(ctx, 2, (uint8_t *) mem_26589.mem, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint8_t *) mem_26555.mem, (int64_t) 0, (int64_t []) {(int64_t) 1, (int64_t) 6}, (int64_t []) {(int64_t) 6, (int64_t) 4});
    // src/pattern_csr.fut:25:3-39
    
    int64_t csr_cols_from_pattern_res_23558;
    int64_t csr_cols_from_pattern_res_23559;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26592, &ext_mem_26591, &csr_cols_from_pattern_res_23558, &csr_cols_from_pattern_res_23559, mem_26589, (int64_t) 6, (int64_t) 4) != 0) {
        err = 1;
        goto cleanup;
    }
    if (memblock_unref(ctx, &mem_26589, "mem_26589") != 0)
        return 1;
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    int64_t vv_color_side_order_res_23594;
    bool vv_color_side_order_res_23595;
    int64_t vv_color_side_order_res_23598;
    int64_t loop_dz2081Uz2083U_23599;
    bool loop_while_23600;
    int64_t color_bound_23603;
    
    if (memblock_set(ctx, &mem_param_26598, &mem_26587, "mem_26587") != 0)
        return 1;
    loop_dz2081Uz2083U_23599 = x_23579;
    loop_while_23600 = loop_cond_23593;
    color_bound_23603 = (int64_t) 1;
    while (loop_while_23600) {
        // src/bgpc_vv_coloring.fut:62:5-64:15
        
        bool cond_23604 = slt64((int64_t) 0, color_bound_23603);
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t bytes_26599 = (int64_t) 8 * loop_dz2081Uz2083U_23599;
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        if (mem_26600_cached_sizze_26868 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26600, &mem_26600_cached_sizze_26868, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:77:24-51
        if (mem_26604_cached_sizze_26869 < color_bound_23603) {
            err = lexical_realloc(ctx, &mem_26604, &mem_26604_cached_sizze_26869, color_bound_23603);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t defunc_0_reduce_res_26019;
        int64_t redout_26397 = (int64_t) -1;
        
        for (int64_t i_26399 = 0; i_26399 < loop_dz2081Uz2083U_23599; i_26399++) {
            int64_t eta_p_25358 = ((int64_t *) mem_param_26598.mem)[i_26399];
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool x_25360 = sle64((int64_t) 0, eta_p_25358);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool y_25361 = slt64(eta_p_25358, csr_cols_from_pattern_res_23558);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool bounds_check_25362 = x_25360 && y_25361;
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool index_certs_25363;
            
            if (!bounds_check_25362) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25358, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23558, "].", "-> #0  src/bgpc_vv_coloring.fut:46:44-56\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:44-56
            
            int64_t seen_final_25364 = ((int64_t *) ext_mem_26592.mem)[eta_p_25358];
            
            // src/bgpc_vv_coloring.fut:46:70-72
            
            int64_t seen_final_25365 = add64((int64_t) 1, eta_p_25358);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool x_25366 = sle64((int64_t) 0, seen_final_25365);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool y_25367 = slt64(seen_final_25365, csr_cols_from_pattern_res_23558);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool bounds_check_25368 = x_25366 && y_25367;
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool index_certs_25369;
            
            if (!bounds_check_25368) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_25365, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23558, "].", "-> #0  src/bgpc_vv_coloring.fut:46:59-73\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:59-73
            
            int64_t seen_final_25370 = ((int64_t *) ext_mem_26592.mem)[seen_final_25365];
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t j_m_i_25371 = sub64(seen_final_25370, seen_final_25364);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool empty_slice_25372 = j_m_i_25371 == (int64_t) 0;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t m_25373 = sub64(j_m_i_25371, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t i_p_m_t_s_25374 = add64(seen_final_25364, m_25373);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_leq_i_p_m_t_s_25375 = sle64((int64_t) 0, i_p_m_t_s_25374);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_p_m_t_s_leq_w_25376 = slt64(i_p_m_t_s_25374, csr_cols_from_pattern_res_23559);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_lte_i_25377 = sle64((int64_t) 0, seen_final_25364);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_lte_j_25378 = sle64(seen_final_25364, seen_final_25370);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_25379 = i_p_m_t_s_leq_w_25376 && zzero_lte_i_25377;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_25380 = zzero_leq_i_p_m_t_s_25375 && y_25379;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool forwards_ok_25381 = i_lte_j_25378 && y_25380;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool ok_or_empty_25382 = empty_slice_25372 || forwards_ok_25381;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool index_certs_25383;
            
            if (!ok_or_empty_25382) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_25364, ":", (long long) seen_final_25370, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23559, "].", "-> #0  src/bgpc_vv_coloring.fut:46:5-54:19\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:77:24-51
            for (int64_t nest_i_26771 = 0; nest_i_26771 < color_bound_23603; nest_i_26771++) {
                ((bool *) mem_26604)[nest_i_26771] = 0;
            }
            // src/bgpc_vv_coloring.fut:46:5-54:19
            for (int64_t i_25385 = 0; i_25385 < j_m_i_25371; i_25385++) {
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t index_primexp_25387 = seen_final_25364 + i_25385;
                
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t v_25388 = ((int64_t *) ext_mem_26591.mem)[index_primexp_25387];
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool x_25389 = sle64((int64_t) 0, v_25388);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool y_25390 = slt64(v_25388, csr_bipartite_from_pattern_res_23553);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool bounds_check_25391 = x_25389 && y_25390;
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool index_certs_25392;
                
                if (!bounds_check_25391) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_25388, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/bgpc_vv_coloring.fut:49:26-37\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:26-37
                
                int64_t seen_acczq_25393 = ((int64_t *) ext_mem_26576.mem)[v_25388];
                
                // src/bgpc_vv_coloring.fut:49:50-52
                
                int64_t seen_acczq_25394 = add64((int64_t) 1, v_25388);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool x_25395 = sle64((int64_t) 0, seen_acczq_25394);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool y_25396 = slt64(seen_acczq_25394, csr_bipartite_from_pattern_res_23553);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool bounds_check_25397 = x_25395 && y_25396;
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool index_certs_25398;
                
                if (!bounds_check_25397) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_25394, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/bgpc_vv_coloring.fut:49:40-53\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:40-53
                
                int64_t seen_acczq_25399 = ((int64_t *) ext_mem_26576.mem)[seen_acczq_25394];
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t j_m_i_25400 = sub64(seen_acczq_25399, seen_acczq_25393);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool empty_slice_25401 = j_m_i_25400 == (int64_t) 0;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t m_25402 = sub64(j_m_i_25400, (int64_t) 1);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t i_p_m_t_s_25403 = add64(seen_acczq_25393, m_25402);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_leq_i_p_m_t_s_25404 = sle64((int64_t) 0, i_p_m_t_s_25403);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_p_m_t_s_leq_w_25405 = slt64(i_p_m_t_s_25403, csr_bipartite_from_pattern_res_23554);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_lte_i_25406 = sle64((int64_t) 0, seen_acczq_25393);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_lte_j_25407 = sle64(seen_acczq_25393, seen_acczq_25399);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_25408 = i_p_m_t_s_leq_w_25405 && zzero_lte_i_25406;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_25409 = zzero_leq_i_p_m_t_s_25404 && y_25408;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool forwards_ok_25410 = i_lte_j_25407 && y_25409;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool ok_or_empty_25411 = empty_slice_25401 || forwards_ok_25410;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool index_certs_25412;
                
                if (!ok_or_empty_25411) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_25393, ":", (long long) seen_acczq_25399, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23554, "].", "-> #0  src/bgpc_vv_coloring.fut:48:9-53:33\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:48:9-53:33
                for (int64_t i_25414 = 0; i_25414 < j_m_i_25400; i_25414++) {
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t index_primexp_25416 = seen_acczq_25393 + i_25414;
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t u_25417 = ((int64_t *) ext_mem_26575.mem)[index_primexp_25416];
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool x_25418 = sle64((int64_t) 0, u_25417);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool y_25419 = slt64(u_25417, (int64_t) 6);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool bounds_check_25420 = x_25418 && y_25419;
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool index_certs_25421;
                    
                    if (!bounds_check_25420) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_25417, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/bgpc_vv_coloring.fut:50:20-29\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    int64_t cu_25422 = ((int64_t *) mem_26588)[u_25417];
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_25423 = u_25417 == eta_p_25358;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_25424 = !cond_25423;
                    
                    // src/bgpc_vv_coloring.fut:51:30-37
                    
                    bool cond_t_res_25425 = sle64((int64_t) 0, cu_25422);
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    bool x_25426 = cond_25424 && cond_t_res_25425;
                    
                    // src/bgpc_vv_coloring.fut:51:44-67
                    
                    bool cond_t_res_25427 = slt64(cu_25422, color_bound_23603);
                    
                    // src/bgpc_vv_coloring.fut:94:14-103:69
                    
                    bool x_25428 = x_25426 && cond_t_res_25427;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    if (x_25428) {
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool bounds_check_25983 = cond_t_res_25425 && cond_t_res_25427;
                        
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool index_certs_25984;
                        
                        if (!bounds_check_25983) {
                            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) cu_25422, "] out of bounds for array of shape [", (long long) color_bound_23603, "].", "-> #0  src/bgpc_vv_coloring.fut:52:19-50\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                            err = FUTHARK_PROGRAM_ERROR;
                            goto cleanup;
                        }
                        // src/bgpc_vv_coloring.fut:52:19-50
                        ((bool *) mem_26604)[cu_25422] = 1;
                    }
                }
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool loop_cond_25433;
            
            if (cond_23604) {
                // src/bgpc_vv_coloring.fut:63:30-37
                
                bool loop_cond_t_res_25986 = ((bool *) mem_26604)[(int64_t) 0];
                
                loop_cond_25433 = loop_cond_t_res_25986;
            } else {
                loop_cond_25433 = 0;
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool c_final_25435;
            int64_t c_final_25436;
            bool loop_while_25437;
            int64_t c_25438;
            
            loop_while_25437 = loop_cond_25433;
            c_25438 = (int64_t) 0;
            while (loop_while_25437) {
                // src/bgpc_vv_coloring.fut:64:9-15
                
                int64_t loopres_25439 = add64((int64_t) 1, c_25438);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool cond_25440 = slt64(loopres_25439, color_bound_23603);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool loop_cond_25441;
                
                if (cond_25440) {
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool x_25987 = sle64((int64_t) 0, loopres_25439);
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool bounds_check_25988 = cond_25440 && x_25987;
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool index_certs_25989;
                    
                    if (!bounds_check_25988) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) loopres_25439, "] out of bounds for array of shape [", (long long) color_bound_23603, "].", "-> #0  src/bgpc_vv_coloring.fut:63:30-37\n   #1  src/bgpc_vv_coloring.fut:81:6-28\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool loop_cond_t_res_25990 = ((bool *) mem_26604)[loopres_25439];
                    
                    loop_cond_25441 = loop_cond_t_res_25990;
                } else {
                    loop_cond_25441 = 0;
                }
                
                bool loop_while_tmp_26774 = loop_cond_25441;
                int64_t c_tmp_26775 = loopres_25439;
                
                loop_while_25437 = loop_while_tmp_26774;
                c_25438 = c_tmp_26775;
            }
            c_final_25435 = loop_while_25437;
            c_final_25436 = c_25438;
            // src/bgpc_vv_coloring.fut:106:25-32
            
            int64_t max_res_23698 = smax64(c_final_25436, redout_26397);
            
            ((int64_t *) mem_26600)[i_26399] = c_final_25436;
            
            int64_t redout_tmp_26769 = max_res_23698;
            
            redout_26397 = redout_tmp_26769;
        }
        defunc_0_reduce_res_26019 = redout_26397;
        // src/bgpc_vv_coloring.fut:113:17-43
        
        bool acc_cert_23704;
        
        // src/bgpc_vv_coloring.fut:113:17-43
        for (int64_t i_26402 = 0; i_26402 < loop_dz2081Uz2083U_23599; i_26402++) {
            int64_t v_23708 = ((int64_t *) mem_param_26598.mem)[i_26402];
            int64_t v_23709 = ((int64_t *) mem_26600)[i_26402];
            
            // src/bgpc_vv_coloring.fut:113:17-43
            // UpdateAcc
            if (sle64((int64_t) 0, v_23708) && slt64(v_23708, (int64_t) 6)) {
                ((int64_t *) mem_26588)[v_23708] = v_23709;
            }
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26621_cached_sizze_26870 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26621, &mem_26621_cached_sizze_26870, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26623_cached_sizze_26871 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26623, &mem_26623_cached_sizze_26871, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26624_cached_sizze_26872 < loop_dz2081Uz2083U_23599) {
            err = lexical_realloc(ctx, &mem_26624, &mem_26624_cached_sizze_26872, loop_dz2081Uz2083U_23599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t discard_26414;
        int64_t scanacc_26406 = (int64_t) 0;
        
        for (int64_t i_26410 = 0; i_26410 < loop_dz2081Uz2083U_23599; i_26410++) {
            int64_t eta_p_25282 = ((int64_t *) mem_param_26598.mem)[i_26410];
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool x_25283 = sle64((int64_t) 0, eta_p_25282);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool y_25284 = slt64(eta_p_25282, csr_cols_from_pattern_res_23558);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool bounds_check_25285 = x_25283 && y_25284;
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool index_certs_25286;
            
            if (!bounds_check_25285) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25282, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23558, "].", "-> #0  src/bgpc_vv_coloring.fut:126:15-27\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #7  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:127:26-28
            
            int64_t k_end_25288 = add64((int64_t) 1, eta_p_25282);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool x_25289 = sle64((int64_t) 0, k_end_25288);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool y_25290 = slt64(k_end_25288, csr_cols_from_pattern_res_23558);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool bounds_check_25291 = x_25289 && y_25290;
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool index_certs_25292;
            
            if (!bounds_check_25291) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_end_25288, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23558, "].", "-> #0  src/bgpc_vv_coloring.fut:127:15-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #7  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:126:15-27
            
            int64_t k0_25287 = ((int64_t *) ext_mem_26592.mem)[eta_p_25282];
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            int64_t k_end_25293 = ((int64_t *) ext_mem_26592.mem)[k_end_25288];
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool cond_25294 = slt64(k0_25287, k_end_25293);
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool loses_conflict_vertex_res_25295;
            bool loses_conflict_vertex_res_25296;
            int64_t loses_conflict_vertex_res_25297;
            bool loop_while_25298;
            bool lost_25299;
            int64_t k_25300;
            
            loop_while_25298 = cond_25294;
            lost_25299 = 0;
            k_25300 = k0_25287;
            while (loop_while_25298) {
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool x_25301 = sle64((int64_t) 0, k_25300);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool y_25302 = slt64(k_25300, csr_cols_from_pattern_res_23559);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool bounds_check_25303 = x_25301 && y_25302;
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool index_certs_25304;
                
                if (!bounds_check_25303) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_25300, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23559, "].", "-> #0  src/bgpc_vv_coloring.fut:132:15-26\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #7  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:132:15-26
                
                int64_t v_25305 = ((int64_t *) ext_mem_26591.mem)[k_25300];
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool x_25306 = sle64((int64_t) 0, v_25305);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool y_25307 = slt64(v_25305, csr_bipartite_from_pattern_res_23553);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool bounds_check_25308 = x_25306 && y_25307;
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool index_certs_25309;
                
                if (!bounds_check_25308) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_25305, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/bgpc_vv_coloring.fut:134:19-30\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #7  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:135:29-31
                
                int64_t t_end_25311 = add64((int64_t) 1, v_25305);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool x_25312 = sle64((int64_t) 0, t_end_25311);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool y_25313 = slt64(t_end_25311, csr_bipartite_from_pattern_res_23553);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool bounds_check_25314 = x_25312 && y_25313;
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool index_certs_25315;
                
                if (!bounds_check_25314) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_end_25311, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/bgpc_vv_coloring.fut:135:19-32\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #7  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:134:19-30
                
                int64_t t0_25310 = ((int64_t *) ext_mem_26576.mem)[v_25305];
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                int64_t t_end_25316 = ((int64_t *) ext_mem_26576.mem)[t_end_25311];
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool cond_25317 = slt64(t0_25310, t_end_25316);
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool loopres_25318;
                bool loopres_25319;
                int64_t loopres_25320;
                bool loop_while_25321;
                bool lost_in_net_25322;
                int64_t t_25323;
                
                loop_while_25321 = cond_25317;
                lost_in_net_25322 = 0;
                t_25323 = t0_25310;
                while (loop_while_25321) {
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool x_25324 = sle64((int64_t) 0, t_25323);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool y_25325 = slt64(t_25323, csr_bipartite_from_pattern_res_23554);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool bounds_check_25326 = x_25324 && y_25325;
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool index_certs_25327;
                    
                    if (!bounds_check_25326) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_25323, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23554, "].", "-> #0  src/bgpc_vv_coloring.fut:140:19-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #7  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    int64_t u_25328 = ((int64_t *) ext_mem_26575.mem)[t_25323];
                    
                    // src/bgpc_vv_coloring.fut:142:13-71
                    
                    bool lost_in_netzq_25329;
                    
                    if (lost_in_net_25322) {
                        lost_in_netzq_25329 = 1;
                    } else {
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25330 = u_25328 == eta_p_25282;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25331 = !cond_25330;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25332;
                        
                        if (cond_25331) {
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool x_25992 = sle64((int64_t) 0, u_25328);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool y_25993 = slt64(u_25328, (int64_t) 6);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool bounds_check_25994 = x_25992 && y_25993;
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool index_certs_25995;
                            
                            if (!bounds_check_25994) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_25328, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/bgpc_vv_coloring.fut:142:39-48\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #7  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool y_25997 = slt64(eta_p_25282, (int64_t) 6);
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool bounds_check_25998 = x_25283 && y_25997;
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool index_certs_25999;
                            
                            if (!bounds_check_25998) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25282, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/bgpc_vv_coloring.fut:142:52-61\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #7  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            int64_t zeze_lhs_25996 = ((int64_t *) mem_26588)[u_25328];
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            int64_t zeze_rhs_26000 = ((int64_t *) mem_26588)[eta_p_25282];
                            
                            // src/bgpc_vv_coloring.fut:142:49-61
                            
                            bool cond_t_res_26001 = zeze_lhs_25996 == zeze_rhs_26000;
                            
                            cond_25332 = cond_t_res_26001;
                        } else {
                            cond_25332 = 0;
                        }
                        // src/bgpc_vv_coloring.fut:142:67-70
                        
                        bool lost_in_netzq_f_res_t_res_25343 = slt64(u_25328, eta_p_25282);
                        
                        // src/bgpc_vv_coloring.fut:160:14-168:57
                        
                        bool x_25344 = cond_25332 && lost_in_netzq_f_res_t_res_25343;
                        
                        lost_in_netzq_25329 = x_25344;
                    }
                    // src/bgpc_vv_coloring.fut:143:31-37
                    
                    int64_t tmp_25345 = add64((int64_t) 1, t_25323);
                    
                    // src/bgpc_vv_coloring.fut:139:15-43
                    
                    bool cond_25346 = slt64(tmp_25345, t_end_25316);
                    
                    // src/bgpc_vv_coloring.fut:139:28-43
                    
                    bool not_res_25347 = !lost_in_netzq_25329;
                    
                    // src/bgpc_vv_coloring.fut:166:5-169:10
                    
                    bool x_25348 = cond_25346 && not_res_25347;
                    bool loop_while_tmp_26784 = x_25348;
                    bool lost_in_net_tmp_26785 = lost_in_netzq_25329;
                    int64_t t_tmp_26786 = tmp_25345;
                    
                    loop_while_25321 = loop_while_tmp_26784;
                    lost_in_net_25322 = lost_in_net_tmp_26785;
                    t_25323 = t_tmp_26786;
                }
                loopres_25318 = loop_while_25321;
                loopres_25319 = lost_in_net_25322;
                loopres_25320 = t_25323;
                // src/bgpc_vv_coloring.fut:160:14-168:57
                
                bool lostzq_25349 = lost_25299 || loopres_25319;
                
                // src/bgpc_vv_coloring.fut:146:20-26
                
                int64_t tmp_25350 = add64((int64_t) 1, k_25300);
                
                // src/bgpc_vv_coloring.fut:131:11-32
                
                bool cond_25351 = slt64(tmp_25350, k_end_25293);
                
                // src/bgpc_vv_coloring.fut:131:24-32
                
                bool not_res_25352 = !lostzq_25349;
                
                // src/bgpc_vv_coloring.fut:166:5-169:10
                
                bool x_25353 = cond_25351 && not_res_25352;
                bool loop_while_tmp_26781 = x_25353;
                bool lost_tmp_26782 = lostzq_25349;
                int64_t k_tmp_26783 = tmp_25350;
                
                loop_while_25298 = loop_while_tmp_26781;
                lost_25299 = lost_tmp_26782;
                k_25300 = k_tmp_26783;
            }
            loses_conflict_vertex_res_25295 = loop_while_25298;
            loses_conflict_vertex_res_25296 = lost_25299;
            loses_conflict_vertex_res_25297 = k_25300;
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_f_res_25355 = btoi_bool_i64(loses_conflict_vertex_res_25296);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_op_res_23810 = add64(defunc_0_f_res_25355, scanacc_26406);
            
            ((int64_t *) mem_26621)[i_26410] = defunc_0_op_res_23810;
            ((int64_t *) mem_26623)[i_26410] = defunc_0_f_res_25355;
            ((bool *) mem_26624)[i_26410] = loses_conflict_vertex_res_25296;
            
            int64_t scanacc_tmp_26777 = defunc_0_op_res_23810;
            
            scanacc_26406 = scanacc_tmp_26777;
        }
        discard_26414 = scanacc_26406;
        // src/bgpc_vv_coloring.fut:23:25-28
        
        int64_t tmp_23794 = sub64(loop_dz2081Uz2083U_23599, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool y_23796 = slt64(tmp_23794, loop_dz2081Uz2083U_23599);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool x_23795 = sle64((int64_t) 0, tmp_23794);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool bounds_check_23797 = x_23795 && y_23796;
        
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        bool cond_23792 = loop_dz2081Uz2083U_23599 == (int64_t) 0;
        
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool protect_assert_disj_23798 = cond_23792 || bounds_check_23797;
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool index_certs_23799;
        
        if (!protect_assert_disj_23798) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_23794, "] out of bounds for array of shape [", (long long) loop_dz2081Uz2083U_23599, "].", "-> #0  src/bgpc_vv_coloring.fut:23:10-29\n   #1  src/bgpc_vv_coloring.fut:171:23-55\n   #2  src/bgpc_vv_coloring.fut:186:14-205:56\n   #3  src/bgpc_vv_coloring.fut:213:14-216:64\n   #4  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #5  src/sparse_jacobian_jvp.fut:167:15-174:59\n   #6  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool x_23793 = !cond_23792;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_f_res_23811;
        
        if (x_23793) {
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t x_26002 = ((int64_t *) mem_26621)[tmp_23794];
            
            m_f_res_23811 = x_26002;
        } else {
            m_f_res_23811 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_23813;
        
        if (cond_23792) {
            m_23813 = (int64_t) 0;
        } else {
            m_23813 = m_f_res_23811;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t bytes_26643 = (int64_t) 8 * m_23813;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26644_cached_sizze_26873 < bytes_26643) {
            err = lexical_realloc(ctx, &mem_26644, &mem_26644_cached_sizze_26873, bytes_26643);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26646_cached_sizze_26874 < bytes_26643) {
            err = lexical_realloc(ctx, &mem_26646, &mem_26646_cached_sizze_26874, bytes_26643);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:18:24-27:44
        if (mem_26648_cached_sizze_26875 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26648, &mem_26648_cached_sizze_26875, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26656_cached_sizze_26876 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26656, &mem_26656_cached_sizze_26876, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        bool acc_cert_25155;
        bool acc_cert_25156;
        
        // src/bgpc_vv_coloring.fut:18:24-27:44
        
        int64_t inpacc_26010;
        int64_t inpacc_25219 = (int64_t) 0;
        
        for (int64_t i_26441 = 0; i_26441 < loop_dz2081Uz2083U_23599; i_26441++) {
            bool eta_p_26503 = ((bool *) mem_26624)[i_26441];
            
            // src/bgpc_vv_coloring.fut:18:28-39
            
            int64_t bool_to_i64_res_26504 = btoi_bool_i64(eta_p_26503);
            int64_t eta_p_26516 = ((int64_t *) mem_26623)[i_26441];
            int64_t eta_p_26517 = ((int64_t *) mem_26621)[i_26441];
            int64_t v_26520 = ((int64_t *) mem_param_26598.mem)[i_26441];
            
            // src/bgpc_vv_coloring.fut:19:29-32
            
            int64_t defunc_0_op_res_26521 = add64(inpacc_25219, bool_to_i64_res_26504);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            bool cond_26522 = eta_p_26516 == (int64_t) 1;
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t lifted_lambda_res_26523;
            
            if (cond_26522) {
                // src/bgpc_vv_coloring.fut:27:5-44
                
                int64_t lifted_lambda_res_t_res_26524 = sub64(eta_p_26517, (int64_t) 1);
                
                lifted_lambda_res_26523 = lifted_lambda_res_t_res_26524;
            } else {
                lifted_lambda_res_26523 = (int64_t) -1;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_26523) && slt64(lifted_lambda_res_26523, m_23813)) {
                ((int64_t *) mem_26646)[lifted_lambda_res_26523] = v_26520;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_26523) && slt64(lifted_lambda_res_26523, m_23813)) {
                ((int64_t *) mem_26644)[lifted_lambda_res_26523] = defunc_0_op_res_26521;
            }
            ((int64_t *) mem_26648)[i_26441] = defunc_0_op_res_26521;
            
            int64_t inpacc_tmp_26787 = defunc_0_op_res_26521;
            
            inpacc_25219 = inpacc_tmp_26787;
        }
        inpacc_26010 = inpacc_25219;
        // src/bgpc_vv_coloring.fut:27:5-44
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_26656, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_26648, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {loop_dz2081Uz2083U_23599});
        // src/bgpc_vv_coloring.fut:23:10-29
        
        int64_t out_szz_f_res_23800;
        
        if (x_23793) {
            // src/bgpc_vv_coloring.fut:171:23-55
            
            int64_t x_26016 = ((int64_t *) mem_26656)[tmp_23794];
            
            out_szz_f_res_23800 = x_26016;
        } else {
            out_szz_f_res_23800 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        int64_t out_szz_23802;
        
        if (cond_23792) {
            out_szz_23802 = (int64_t) 0;
        } else {
            out_szz_23802 = out_szz_f_res_23800;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        
        int64_t bytes_26657 = (int64_t) 8 * out_szz_23802;
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        bool cond_23699 = slt64(defunc_0_reduce_res_26019, (int64_t) 0);
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        int64_t next_color_bound_23700;
        
        if (cond_23699) {
            next_color_bound_23700 = color_bound_23603;
        } else {
            // src/bgpc_vv_coloring.fut:110:40-46
            
            int64_t max_arg1_23701 = add64((int64_t) 2, defunc_0_reduce_res_26019);
            
            // src/bgpc_vv_coloring.fut:110:10-46
            
            int64_t max_res_23702 = smax64(color_bound_23603, max_arg1_23701);
            
            next_color_bound_23700 = max_res_23702;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        if (memblock_alloc(ctx, &mem_26658, bytes_26657, "mem_26658")) {
            err = 1;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        for (int64_t nest_i_26791 = 0; nest_i_26791 < out_szz_23802; nest_i_26791++) {
            ((int64_t *) mem_26658.mem)[nest_i_26791] = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:33:6-27
        
        bool acc_cert_24928;
        
        // src/bgpc_vv_coloring.fut:29:21-33:27
        for (int64_t i_26446 = 0; i_26446 < m_23813; i_26446++) {
            int64_t eta_p_24940 = ((int64_t *) mem_26644)[i_26446];
            int64_t v_24942 = ((int64_t *) mem_26646)[i_26446];
            
            // src/bgpc_vv_coloring.fut:29:44-50
            
            int64_t lifted_lambda_res_24943 = sub64(eta_p_24940, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:33:6-27
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_24943) && slt64(lifted_lambda_res_24943, out_szz_23802)) {
                ((int64_t *) mem_26658.mem)[lifted_lambda_res_24943] = v_24942;
            }
        }
        // src/bgpc_vv_coloring.fut:175:17-48
        
        bool acc_cert_23850;
        
        // src/bgpc_vv_coloring.fut:175:17-48
        for (int64_t i_26448 = 0; i_26448 < out_szz_23802; i_26448++) {
            int64_t v_23854 = ((int64_t *) mem_26658.mem)[i_26448];
            
            // src/bgpc_vv_coloring.fut:175:17-48
            // UpdateAcc
            if (sle64((int64_t) 0, v_23854) && slt64(v_23854, (int64_t) 6)) {
                ((int64_t *) mem_26588)[v_23854] = (int64_t) -1;
            }
        }
        // src/bgpc_vv_coloring.fut:198:20-23
        
        bool loop_cond_23856 = slt64((int64_t) 0, out_szz_23802);
        
        if (memblock_set(ctx, &mem_param_tmp_26763, &mem_26658, "mem_26658") != 0)
            return 1;
        
        int64_t loop_dz2081Uz2083U_tmp_26764 = out_szz_23802;
        bool loop_while_tmp_26765 = loop_cond_23856;
        int64_t color_bound_tmp_26768 = next_color_bound_23700;
        
        if (memblock_set(ctx, &mem_param_26598, &mem_param_tmp_26763, "mem_param_tmp_26763") != 0)
            return 1;
        loop_dz2081Uz2083U_23599 = loop_dz2081Uz2083U_tmp_26764;
        loop_while_23600 = loop_while_tmp_26765;
        color_bound_23603 = color_bound_tmp_26768;
    }
    if (memblock_set(ctx, &ext_mem_26663, &mem_param_26598, "mem_param_26598") != 0)
        return 1;
    vv_color_side_order_res_23594 = loop_dz2081Uz2083U_23599;
    vv_color_side_order_res_23595 = loop_while_23600;
    vv_color_side_order_res_23598 = color_bound_23603;
    if (memblock_unref(ctx, &mem_26587, "mem_26587") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26591, "ext_mem_26591") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26592, "ext_mem_26592") != 0)
        return 1;
    // src/sparse_jacobian_jvp.fut:107:8-112:40
    
    int64_t x_26031;
    int64_t redout_26449 = (int64_t) 0;
    
    for (int64_t i_26450 = 0; i_26450 < (int64_t) 6; i_26450++) {
        int64_t x_23859 = ((int64_t *) mem_26588)[i_26450];
        
        // src/sparse_jacobian_jvp.fut:14:22-29
        
        int64_t max_res_23862 = smax64(x_23859, redout_26449);
        int64_t redout_tmp_26794 = max_res_23862;
        
        redout_26449 = redout_tmp_26794;
    }
    x_26031 = redout_26449;
    // src/sparse_jacobian_jvp.fut:14:13-41
    
    int64_t num_colors_of_res_f_res_23863 = add64((int64_t) 1, x_26031);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool bounds_invalid_upwards_23865 = slt64(num_colors_of_res_f_res_23863, (int64_t) 0);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool valid_23866 = !bounds_invalid_upwards_23865;
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool range_valid_c_23867;
    
    if (!valid_23866) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Range ", (long long) (int64_t) 0, "..", (long long) (int64_t) 1, "..<", (long long) num_colors_of_res_f_res_23863, " is invalid.", "-> #0  src/sparse_jacobian_jvp.fut:29:11-18\n   #1  src/sparse_jacobian_jvp.fut:107:8-112:40\n   #2  src/sparse_jacobian_jvp.fut:166:8-175:40\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    
    int64_t bytes_26666 = (int64_t) 32 * num_colors_of_res_f_res_23863;
    double zt_lhs_24490 = ((double *) x_mem_26563.mem)[(int64_t) 0];
    double zt_rhs_24491 = ((double *) x_mem_26563.mem)[(int64_t) 1];
    double zt_rhs_24493 = ((double *) x_mem_26563.mem)[(int64_t) 4];
    
    // test/test_sparse_jacobian_jvp.fut:82:33-39
    
    double binop_x_25872 = 0.0 * zt_rhs_24493;
    double zt_lhs_24497 = ((double *) x_mem_26563.mem)[(int64_t) 5];
    
    // src/dense_jacobian.fut:8:40-9:68
    if (mem_26564_cached_sizze_26863 < (int64_t) 192) {
        err = lexical_realloc(ctx, &mem_26564, &mem_26564_cached_sizze_26863, (int64_t) 192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:5:3-21
    if (mem_26569_cached_sizze_26864 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26569, &mem_26569_cached_sizze_26864, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:8:40-9:68
    for (int64_t i_26386 = 0; i_26386 < (int64_t) 6; i_26386++) {
        // src/dense_jacobian.fut:5:3-21
        for (int64_t nest_i_26796 = 0; nest_i_26796 < (int64_t) 6; nest_i_26796++) {
            ((double *) mem_26569)[nest_i_26796] = 0.0;
        }
        // src/dense_jacobian.fut:5:3-39
        ((double *) mem_26569)[i_26386] = 1.0;
        
        double zt_lhs_tan_25865 = ((double *) mem_26569)[(int64_t) 0];
        double zt_rhs_tan_25866 = ((double *) mem_26569)[(int64_t) 1];
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_x_25868 = zt_rhs_24491 * zt_lhs_tan_25865;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_y_25869 = zt_lhs_24490 * zt_rhs_tan_25866;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double zp_lhs_tan_25867 = binop_x_25868 + binop_y_25869;
        double zt_rhs_tan_25870 = ((double *) mem_26569)[(int64_t) 4];
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double binop_y_25873 = 3.0 * zt_rhs_tan_25870;
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double zp_rhs_tan_25871 = binop_x_25872 + binop_y_25873;
        
        // test/test_sparse_jacobian_jvp.fut:82:24-39
        
        double y0_tan_25874 = zp_lhs_tan_25867 + zp_rhs_tan_25871;
        double zm_lhs_tan_25877 = ((double *) mem_26569)[(int64_t) 2];
        double zt_lhs_tan_25878 = ((double *) mem_26569)[(int64_t) 5];
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double binop_x_25880 = zt_lhs_24497 * zt_lhs_tan_25878;
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double zm_rhs_tan_25879 = binop_x_25880 + binop_x_25880;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double binop_y_25884 = -1.0 * zm_rhs_tan_25879;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double y1_tan_25882 = zm_lhs_tan_25877 + binop_y_25884;
        
        // test/test_sparse_jacobian_jvp.fut:84:17-23
        
        double zp_lhs_tan_25885 = zt_rhs_tan_25866 + zm_lhs_tan_25877;
        double zp_rhs_tan_25888 = ((double *) mem_26569)[(int64_t) 3];
        
        // test/test_sparse_jacobian_jvp.fut:84:24-30
        
        double y2_tan_25889 = zp_lhs_tan_25885 + zp_rhs_tan_25888;
        
        // test/test_sparse_jacobian_jvp.fut:185:52-57
        ((double *) mem_26564)[i_26386 * (int64_t) 4] = y0_tan_25874;
        ((double *) mem_26564)[i_26386 * (int64_t) 4 + (int64_t) 1] = y1_tan_25882;
        ((double *) mem_26564)[i_26386 * (int64_t) 4 + (int64_t) 2] = y2_tan_25889;
        ((double *) mem_26564)[i_26386 * (int64_t) 4 + (int64_t) 3] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    if (mem_26578_cached_sizze_26865 < bytes_26577) {
        err = lexical_realloc(ctx, &mem_26578, &mem_26578_cached_sizze_26865, bytes_26577);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    for (int64_t nest_i_26797 = 0; nest_i_26797 < csr_bipartite_from_pattern_res_23554; nest_i_26797++) {
        ((double *) mem_26578)[nest_i_26797] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    if (mem_26667_cached_sizze_26877 < bytes_26666) {
        err = lexical_realloc(ctx, &mem_26667, &mem_26667_cached_sizze_26877, bytes_26666);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:17:3-61
    if (mem_26672_cached_sizze_26878 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26672, &mem_26672_cached_sizze_26878, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    for (int64_t i_26457 = 0; i_26457 < num_colors_of_res_f_res_23863; i_26457++) {
        // src/sparse_jacobian_jvp.fut:17:3-61
        for (int64_t i_26453 = 0; i_26453 < (int64_t) 6; i_26453++) {
            int64_t eta_p_23876 = ((int64_t *) mem_26588)[i_26453];
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            bool cond_23877 = eta_p_23876 == i_26457;
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            double lifted_lambda_res_23878;
            
            if (cond_23877) {
                lifted_lambda_res_23878 = 1.0;
            } else {
                lifted_lambda_res_23878 = 0.0;
            }
            ((double *) mem_26672)[i_26453] = lifted_lambda_res_23878;
        }
        
        double zt_lhs_tan_25894 = ((double *) mem_26672)[(int64_t) 0];
        double zt_rhs_tan_25895 = ((double *) mem_26672)[(int64_t) 1];
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_x_25897 = zt_rhs_24491 * zt_lhs_tan_25894;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_y_25898 = zt_lhs_24490 * zt_rhs_tan_25895;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double zp_lhs_tan_25896 = binop_x_25897 + binop_y_25898;
        double zt_rhs_tan_25899 = ((double *) mem_26672)[(int64_t) 4];
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double binop_y_25902 = 3.0 * zt_rhs_tan_25899;
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double zp_rhs_tan_25900 = binop_x_25872 + binop_y_25902;
        
        // test/test_sparse_jacobian_jvp.fut:82:24-39
        
        double y0_tan_25903 = zp_lhs_tan_25896 + zp_rhs_tan_25900;
        double zm_lhs_tan_25906 = ((double *) mem_26672)[(int64_t) 2];
        double zt_lhs_tan_25907 = ((double *) mem_26672)[(int64_t) 5];
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double binop_x_25909 = zt_lhs_24497 * zt_lhs_tan_25907;
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double zm_rhs_tan_25908 = binop_x_25909 + binop_x_25909;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double binop_y_25913 = -1.0 * zm_rhs_tan_25908;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double y1_tan_25911 = zm_lhs_tan_25906 + binop_y_25913;
        
        // test/test_sparse_jacobian_jvp.fut:84:17-23
        
        double zp_lhs_tan_25914 = zt_rhs_tan_25895 + zm_lhs_tan_25906;
        double zp_rhs_tan_25917 = ((double *) mem_26672)[(int64_t) 3];
        
        // test/test_sparse_jacobian_jvp.fut:84:24-30
        
        double y2_tan_25918 = zp_lhs_tan_25914 + zp_rhs_tan_25917;
        
        // test/test_sparse_jacobian_jvp.fut:191:33-38
        ((double *) mem_26667)[i_26457 * (int64_t) 4] = y0_tan_25903;
        ((double *) mem_26667)[i_26457 * (int64_t) 4 + (int64_t) 1] = y1_tan_25911;
        ((double *) mem_26667)[i_26457 * (int64_t) 4 + (int64_t) 2] = y2_tan_25918;
        ((double *) mem_26667)[i_26457 * (int64_t) 4 + (int64_t) 3] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:45:5-52:27
    
    bool compressed_to_csr_vals_res_23887;
    int64_t compressed_to_csr_vals_res_23889;
    bool loop_while_23890;
    int64_t i_23892;
    
    loop_while_23890 = 1;
    i_23892 = (int64_t) 0;
    while (loop_while_23890) {
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool x_23893 = sle64((int64_t) 0, i_23892);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool y_23894 = slt64(i_23892, csr_bipartite_from_pattern_res_23553);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool bounds_check_23895 = x_23893 && y_23894;
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool index_certs_23896;
        
        if (!bounds_check_23895) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_23892, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/sparse_jacobian_jvp.fut:47:15-26\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:166:8-175:40\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        int64_t s_23897 = ((int64_t *) ext_mem_26576.mem)[i_23892];
        
        // src/sparse_jacobian_jvp.fut:48:25-27
        
        int64_t e_23898 = add64((int64_t) 1, i_23892);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool x_23899 = sle64((int64_t) 0, e_23898);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool y_23900 = slt64(e_23898, csr_bipartite_from_pattern_res_23553);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool bounds_check_23901 = x_23899 && y_23900;
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool index_certs_23902;
        
        if (!bounds_check_23901) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_23898, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/sparse_jacobian_jvp.fut:48:15-28\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:166:8-175:40\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        int64_t e_23903 = ((int64_t *) ext_mem_26576.mem)[e_23898];
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t j_m_i_23904 = sub64(e_23903, s_23897);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool empty_slice_23905 = j_m_i_23904 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t m_23906 = sub64(j_m_i_23904, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t i_p_m_t_s_23907 = add64(s_23897, m_23906);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_23908 = sle64((int64_t) 0, i_p_m_t_s_23907);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_23909 = slt64(i_p_m_t_s_23907, csr_bipartite_from_pattern_res_23554);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_lte_i_23910 = sle64((int64_t) 0, s_23897);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_lte_j_23911 = sle64(s_23897, e_23903);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_23912 = i_p_m_t_s_leq_w_23909 && zzero_lte_i_23910;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_23913 = zzero_leq_i_p_m_t_s_23908 && y_23912;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool forwards_ok_23914 = i_lte_j_23911 && y_23913;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool ok_or_empty_23915 = empty_slice_23905 || forwards_ok_23914;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool index_certs_23916;
        
        if (!ok_or_empty_23915) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_23897, ":", (long long) e_23903, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23554, "].", "-> #0  src/sparse_jacobian_jvp.fut:49:18-30\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:166:8-175:40\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool y_23918 = slt64(i_23892, (int64_t) 4);
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool bounds_check_23919 = x_23893 && y_23918;
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool index_certs_23920;
        
        if (!bounds_check_23919) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_23892, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-45\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:166:8-175:40\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:18-51
        for (int64_t i_26461 = 0; i_26461 < j_m_i_23904; i_26461++) {
            int64_t index_primexp_26496 = s_23897 + i_26461;
            int64_t eta_p_23922 = ((int64_t *) ext_mem_26575.mem)[index_primexp_26496];
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool x_23923 = sle64((int64_t) 0, eta_p_23922);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool y_23924 = slt64(eta_p_23922, (int64_t) 6);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool bounds_check_23925 = x_23923 && y_23924;
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool index_certs_23926;
            
            if (!bounds_check_23925) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_23922, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:32-41\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:166:8-175:40\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            int64_t tmp_23927 = ((int64_t *) mem_26588)[eta_p_23922];
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool x_23928 = sle64((int64_t) 0, tmp_23927);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool y_23929 = slt64(tmp_23927, num_colors_of_res_f_res_23863);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool bounds_check_23930 = x_23928 && y_23929;
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool index_certs_23931;
            
            if (!bounds_check_23930) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_23927, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_23863, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-42\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:166:8-175:40\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-191:74\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:18-51
            
            double lifted_lambda_res_23932 = ((double *) mem_26667)[tmp_23927 * (int64_t) 4 + i_23892];
            
            ((double *) mem_26578)[s_23897 + i_26461] = lifted_lambda_res_23932;
        }
        // src/sparse_jacobian_jvp.fut:46:13-16
        
        bool loop_cond_23934 = slt64(e_23898, (int64_t) 4);
        bool loop_while_tmp_26800 = loop_cond_23934;
        int64_t i_tmp_26802 = e_23898;
        
        loop_while_23890 = loop_while_tmp_26800;
        i_23892 = i_tmp_26802;
    }
    compressed_to_csr_vals_res_23887 = loop_while_23890;
    compressed_to_csr_vals_res_23889 = i_23892;
    // src/sparse_jacobian_jvp.fut:67:29-47
    if (mem_26698_cached_sizze_26879 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26698, &mem_26698_cached_sizze_26879, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:8:19-10:39
    
    bool defunc_0_reduce_res_26033;
    bool redout_26467 = 1;
    
    for (int64_t i_26468 = 0; i_26468 < (int64_t) 4; i_26468++) {
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool y_24851 = slt64(i_26468, csr_bipartite_from_pattern_res_23553);
        
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool index_certs_24853;
        
        if (!y_24851) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_26468, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/sparse_jacobian_jvp.fut:63:17-28\n   #1  test/test_sparse_jacobian_jvp.fut:81:25-194:54\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        int64_t s_24854 = ((int64_t *) ext_mem_26576.mem)[i_26468];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_lte_i_24867 = sle64((int64_t) 0, s_24854);
        
        // src/sparse_jacobian_jvp.fut:64:27-29
        
        int64_t e_24855 = add64((int64_t) 1, i_26468);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool y_24857 = slt64(e_24855, csr_bipartite_from_pattern_res_23553);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool x_24856 = sle64((int64_t) 0, e_24855);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool bounds_check_24858 = x_24856 && y_24857;
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool index_certs_24859;
        
        if (!bounds_check_24858) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_24855, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/sparse_jacobian_jvp.fut:64:17-30\n   #1  test/test_sparse_jacobian_jvp.fut:81:25-194:54\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        int64_t e_24860 = ((int64_t *) ext_mem_26576.mem)[e_24855];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t j_m_i_24861 = sub64(e_24860, s_24854);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t m_24863 = sub64(j_m_i_24861, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t i_p_m_t_s_24864 = add64(s_24854, m_24863);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_p_m_t_s_leq_w_24866 = slt64(i_p_m_t_s_24864, csr_bipartite_from_pattern_res_23554);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24875 = i_p_m_t_s_leq_w_24866 && zzero_lte_i_24867;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_leq_i_p_m_t_s_24865 = sle64((int64_t) 0, i_p_m_t_s_24864);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24876 = zzero_leq_i_p_m_t_s_24865 && y_24875;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_lte_j_24868 = sle64(s_24854, e_24860);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool forwards_ok_24877 = i_lte_j_24868 && y_24876;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool empty_slice_24862 = j_m_i_24861 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool ok_or_empty_24878 = empty_slice_24862 || forwards_ok_24877;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool index_certs_24879;
        
        if (!ok_or_empty_24878) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_24854, ":", (long long) e_24860, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23554, "].", "-> #0  src/sparse_jacobian_jvp.fut:66:20-29\n   #1  test/test_sparse_jacobian_jvp.fut:81:25-194:54\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:67:29-47
        for (int64_t nest_i_26805 = 0; nest_i_26805 < (int64_t) 6; nest_i_26805++) {
            ((double *) mem_26698)[nest_i_26805] = 0.0;
        }
        // src/sparse_jacobian_jvp.fut:68:12-32
        
        bool acc_cert_24883;
        
        // src/sparse_jacobian_jvp.fut:68:12-32
        for (int64_t i_26464 = 0; i_26464 < j_m_i_24861; i_26464++) {
            int64_t index_primexp_26493 = s_24854 + i_26464;
            int64_t v_24887 = ((int64_t *) ext_mem_26575.mem)[index_primexp_26493];
            double v_24888 = ((double *) mem_26578)[index_primexp_26493];
            
            // src/sparse_jacobian_jvp.fut:68:12-32
            // UpdateAcc
            if (sle64((int64_t) 0, v_24887) && slt64(v_24887, (int64_t) 6)) {
                ((double *) mem_26698)[v_24887] = v_24888;
            }
        }
        // test/test_sparse_jacobian_jvp.fut:9:5-66
        
        bool defunc_0_reduce_res_26028;
        bool redout_26465 = 1;
        
        for (int64_t i_26466 = 0; i_26466 < (int64_t) 6; i_26466++) {
            bool eta_p_24911 = ((bool *) mem_26555.mem)[i_26468 * (int64_t) 6 + i_26466];
            double eta_p_24912 = ((double *) mem_26564)[i_26466 * (int64_t) 4 + i_26468];
            double eta_p_24913 = ((double *) mem_26698)[i_26466];
            
            // test/test_sparse_jacobian_jvp.fut:14:25-48
            
            double lifted_lambda_res_24914;
            
            if (eta_p_24911) {
                lifted_lambda_res_24914 = eta_p_24912;
            } else {
                lifted_lambda_res_24914 = 0.0;
            }
            // test/test_sparse_jacobian_jvp.fut:9:48-51
            
            double abs_arg0_24916 = eta_p_24913 - lifted_lambda_res_24914;
            
            // test/test_sparse_jacobian_jvp.fut:9:37-51
            
            double abs_res_24917 = fabs64(abs_arg0_24916);
            
            // test/test_sparse_jacobian_jvp.fut:9:53-59
            
            bool lifted_lambda_res_24918 = abs_res_24917 <= 1.0e-9;
            
            // test/test_sparse_jacobian_jvp.fut:9:5-66
            
            bool x_24901 = lifted_lambda_res_24918 && redout_26465;
            bool redout_tmp_26807 = x_24901;
            
            redout_26465 = redout_tmp_26807;
        }
        defunc_0_reduce_res_26028 = redout_26465;
        // test/test_sparse_jacobian_jvp.fut:10:6-39
        
        bool x_24614 = defunc_0_reduce_res_26028 && redout_26467;
        bool redout_tmp_26804 = x_24614;
        
        redout_26467 = redout_tmp_26804;
    }
    defunc_0_reduce_res_26033 = redout_26467;
    if (memblock_unref(ctx, &ext_mem_26575, "ext_mem_26575") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26576, "ext_mem_26576") != 0)
        return 1;
    prim_out_26758 = defunc_0_reduce_res_26033;
    *out_prim_out_26862 = prim_out_26758;
    
  cleanup:
    {
        free(mem_26564);
        free(mem_26569);
        free(mem_26578);
        free(mem_26579);
        free(mem_26588);
        free(mem_26600);
        free(mem_26604);
        free(mem_26621);
        free(mem_26623);
        free(mem_26624);
        free(mem_26644);
        free(mem_26646);
        free(mem_26648);
        free(mem_26656);
        free(mem_26667);
        free(mem_26672);
        free(mem_26698);
        if (memblock_unref(ctx, &mem_param_tmp_26763, "mem_param_tmp_26763") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26658, "mem_26658") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_26598, "mem_param_26598") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26663, "ext_mem_26663") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26591, "ext_mem_26591") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26592, "ext_mem_26592") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26589, "mem_26589") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26587, "mem_26587") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26575, "ext_mem_26575") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26576, "ext_mem_26576") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_prepared_jvp_ex4_matches_dense(struct futhark_context *ctx, bool *out_prim_out_26880, struct memblock x_mem_26563)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_26564_cached_sizze_26881 = 0;
    unsigned char *mem_26564 = NULL;
    int64_t mem_26569_cached_sizze_26882 = 0;
    unsigned char *mem_26569 = NULL;
    int64_t mem_26578_cached_sizze_26883 = 0;
    unsigned char *mem_26578 = NULL;
    int64_t mem_26579_cached_sizze_26884 = 0;
    unsigned char *mem_26579 = NULL;
    int64_t mem_26588_cached_sizze_26885 = 0;
    unsigned char *mem_26588 = NULL;
    int64_t mem_26600_cached_sizze_26886 = 0;
    unsigned char *mem_26600 = NULL;
    int64_t mem_26604_cached_sizze_26887 = 0;
    unsigned char *mem_26604 = NULL;
    int64_t mem_26621_cached_sizze_26888 = 0;
    unsigned char *mem_26621 = NULL;
    int64_t mem_26623_cached_sizze_26889 = 0;
    unsigned char *mem_26623 = NULL;
    int64_t mem_26624_cached_sizze_26890 = 0;
    unsigned char *mem_26624 = NULL;
    int64_t mem_26644_cached_sizze_26891 = 0;
    unsigned char *mem_26644 = NULL;
    int64_t mem_26646_cached_sizze_26892 = 0;
    unsigned char *mem_26646 = NULL;
    int64_t mem_26648_cached_sizze_26893 = 0;
    unsigned char *mem_26648 = NULL;
    int64_t mem_26656_cached_sizze_26894 = 0;
    unsigned char *mem_26656 = NULL;
    int64_t mem_26667_cached_sizze_26895 = 0;
    unsigned char *mem_26667 = NULL;
    int64_t mem_26672_cached_sizze_26896 = 0;
    unsigned char *mem_26672 = NULL;
    int64_t mem_26698_cached_sizze_26897 = 0;
    unsigned char *mem_26698 = NULL;
    struct memblock mem_param_tmp_26763;
    
    mem_param_tmp_26763.references = NULL;
    
    struct memblock mem_26658;
    
    mem_26658.references = NULL;
    
    struct memblock mem_param_26598;
    
    mem_param_26598.references = NULL;
    
    struct memblock ext_mem_26663;
    
    ext_mem_26663.references = NULL;
    
    struct memblock ext_mem_26591;
    
    ext_mem_26591.references = NULL;
    
    struct memblock ext_mem_26592;
    
    ext_mem_26592.references = NULL;
    
    struct memblock mem_26589;
    
    mem_26589.references = NULL;
    
    struct memblock mem_26587;
    
    mem_26587.references = NULL;
    
    struct memblock ext_mem_26575;
    
    ext_mem_26575.references = NULL;
    
    struct memblock ext_mem_26576;
    
    ext_mem_26576.references = NULL;
    
    struct memblock mem_26545 = ctx->constants->mem_26545;
    struct memblock mem_26548 = ctx->constants->mem_26548;
    struct memblock mem_26550 = ctx->constants->mem_26550;
    struct memblock mem_26555 = ctx->constants->mem_26555;
    struct memblock mem_26556 = ctx->constants->mem_26556;
    struct memblock mem_26562 = ctx->constants->mem_26562;
    bool prim_out_26758;
    
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_23553;
    int64_t csr_bipartite_from_pattern_res_23554;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26576, &ext_mem_26575, &csr_bipartite_from_pattern_res_23553, &csr_bipartite_from_pattern_res_23554, mem_26555, (int64_t) 4, (int64_t) 6) != 0) {
        err = 1;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    
    int64_t bytes_26577 = (int64_t) 8 * csr_bipartite_from_pattern_res_23554;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (mem_26579_cached_sizze_26884 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26579, &mem_26579_cached_sizze_26884, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t discard_26393;
    int64_t scanacc_26389 = (int64_t) 0;
    
    for (int64_t i_26391 = 0; i_26391 < (int64_t) 6; i_26391++) {
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t defunc_0_op_res_23569 = add64((int64_t) 1, scanacc_26389);
        
        ((int64_t *) mem_26579)[i_26391] = defunc_0_op_res_23569;
        
        int64_t scanacc_tmp_26759 = defunc_0_op_res_23569;
        
        scanacc_26389 = scanacc_tmp_26759;
    }
    discard_26393 = scanacc_26389;
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t x_23579 = ((int64_t *) mem_26579)[(int64_t) 5];
    
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    bool loop_cond_23593 = slt64((int64_t) 0, x_23579);
    
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t bytes_26586 = (int64_t) 8 * x_23579;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (memblock_alloc(ctx, &mem_26587, bytes_26586, "mem_26587")) {
        err = 1;
        goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    bool acc_cert_24821;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    for (int64_t i_26395 = 0; i_26395 < (int64_t) 6; i_26395++) {
        int64_t eta_p_24833 = ((int64_t *) mem_26579)[i_26395];
        
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t lifted_lambda_res_t_res_24836 = sub64(eta_p_24833, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:194:20-45
        // UpdateAcc
        if (sle64((int64_t) 0, lifted_lambda_res_t_res_24836) && slt64(lifted_lambda_res_t_res_24836, x_23579)) {
            ((int64_t *) mem_26587.mem)[lifted_lambda_res_t_res_24836] = i_26395;
        }
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    if (mem_26588_cached_sizze_26885 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26588, &mem_26588_cached_sizze_26885, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    for (int64_t nest_i_26762 = 0; nest_i_26762 < (int64_t) 6; nest_i_26762++) {
        ((int64_t *) mem_26588)[nest_i_26762] = (int64_t) -1;
    }
    // src/pattern_csr.fut:25:3-39
    if (memblock_alloc(ctx, &mem_26589, (int64_t) 24, "mem_26589")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:25:3-39
    // src/pattern_csr.fut:25:3-39
    lmad_copy_1b(ctx, 2, (uint8_t *) mem_26589.mem, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint8_t *) mem_26555.mem, (int64_t) 0, (int64_t []) {(int64_t) 1, (int64_t) 6}, (int64_t []) {(int64_t) 6, (int64_t) 4});
    // src/pattern_csr.fut:25:3-39
    
    int64_t csr_cols_from_pattern_res_23558;
    int64_t csr_cols_from_pattern_res_23559;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26592, &ext_mem_26591, &csr_cols_from_pattern_res_23558, &csr_cols_from_pattern_res_23559, mem_26589, (int64_t) 6, (int64_t) 4) != 0) {
        err = 1;
        goto cleanup;
    }
    if (memblock_unref(ctx, &mem_26589, "mem_26589") != 0)
        return 1;
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    int64_t vv_color_side_order_res_23594;
    bool vv_color_side_order_res_23595;
    int64_t vv_color_side_order_res_23598;
    int64_t loop_dz2081Uz2083U_23599;
    bool loop_while_23600;
    int64_t color_bound_23603;
    
    if (memblock_set(ctx, &mem_param_26598, &mem_26587, "mem_26587") != 0)
        return 1;
    loop_dz2081Uz2083U_23599 = x_23579;
    loop_while_23600 = loop_cond_23593;
    color_bound_23603 = (int64_t) 1;
    while (loop_while_23600) {
        // src/bgpc_vv_coloring.fut:62:5-64:15
        
        bool cond_23604 = slt64((int64_t) 0, color_bound_23603);
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t bytes_26599 = (int64_t) 8 * loop_dz2081Uz2083U_23599;
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        if (mem_26600_cached_sizze_26886 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26600, &mem_26600_cached_sizze_26886, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:77:24-51
        if (mem_26604_cached_sizze_26887 < color_bound_23603) {
            err = lexical_realloc(ctx, &mem_26604, &mem_26604_cached_sizze_26887, color_bound_23603);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t defunc_0_reduce_res_26019;
        int64_t redout_26397 = (int64_t) -1;
        
        for (int64_t i_26399 = 0; i_26399 < loop_dz2081Uz2083U_23599; i_26399++) {
            int64_t eta_p_25358 = ((int64_t *) mem_param_26598.mem)[i_26399];
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool x_25360 = sle64((int64_t) 0, eta_p_25358);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool y_25361 = slt64(eta_p_25358, csr_cols_from_pattern_res_23558);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool bounds_check_25362 = x_25360 && y_25361;
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool index_certs_25363;
            
            if (!bounds_check_25362) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25358, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23558, "].", "-> #0  src/bgpc_vv_coloring.fut:46:44-56\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:44-56
            
            int64_t seen_final_25364 = ((int64_t *) ext_mem_26592.mem)[eta_p_25358];
            
            // src/bgpc_vv_coloring.fut:46:70-72
            
            int64_t seen_final_25365 = add64((int64_t) 1, eta_p_25358);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool x_25366 = sle64((int64_t) 0, seen_final_25365);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool y_25367 = slt64(seen_final_25365, csr_cols_from_pattern_res_23558);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool bounds_check_25368 = x_25366 && y_25367;
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool index_certs_25369;
            
            if (!bounds_check_25368) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_25365, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23558, "].", "-> #0  src/bgpc_vv_coloring.fut:46:59-73\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:59-73
            
            int64_t seen_final_25370 = ((int64_t *) ext_mem_26592.mem)[seen_final_25365];
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t j_m_i_25371 = sub64(seen_final_25370, seen_final_25364);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool empty_slice_25372 = j_m_i_25371 == (int64_t) 0;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t m_25373 = sub64(j_m_i_25371, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t i_p_m_t_s_25374 = add64(seen_final_25364, m_25373);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_leq_i_p_m_t_s_25375 = sle64((int64_t) 0, i_p_m_t_s_25374);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_p_m_t_s_leq_w_25376 = slt64(i_p_m_t_s_25374, csr_cols_from_pattern_res_23559);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_lte_i_25377 = sle64((int64_t) 0, seen_final_25364);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_lte_j_25378 = sle64(seen_final_25364, seen_final_25370);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_25379 = i_p_m_t_s_leq_w_25376 && zzero_lte_i_25377;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_25380 = zzero_leq_i_p_m_t_s_25375 && y_25379;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool forwards_ok_25381 = i_lte_j_25378 && y_25380;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool ok_or_empty_25382 = empty_slice_25372 || forwards_ok_25381;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool index_certs_25383;
            
            if (!ok_or_empty_25382) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_25364, ":", (long long) seen_final_25370, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23559, "].", "-> #0  src/bgpc_vv_coloring.fut:46:5-54:19\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:77:24-51
            for (int64_t nest_i_26771 = 0; nest_i_26771 < color_bound_23603; nest_i_26771++) {
                ((bool *) mem_26604)[nest_i_26771] = 0;
            }
            // src/bgpc_vv_coloring.fut:46:5-54:19
            for (int64_t i_25385 = 0; i_25385 < j_m_i_25371; i_25385++) {
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t index_primexp_25387 = seen_final_25364 + i_25385;
                
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t v_25388 = ((int64_t *) ext_mem_26591.mem)[index_primexp_25387];
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool x_25389 = sle64((int64_t) 0, v_25388);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool y_25390 = slt64(v_25388, csr_bipartite_from_pattern_res_23553);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool bounds_check_25391 = x_25389 && y_25390;
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool index_certs_25392;
                
                if (!bounds_check_25391) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_25388, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/bgpc_vv_coloring.fut:49:26-37\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:26-37
                
                int64_t seen_acczq_25393 = ((int64_t *) ext_mem_26576.mem)[v_25388];
                
                // src/bgpc_vv_coloring.fut:49:50-52
                
                int64_t seen_acczq_25394 = add64((int64_t) 1, v_25388);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool x_25395 = sle64((int64_t) 0, seen_acczq_25394);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool y_25396 = slt64(seen_acczq_25394, csr_bipartite_from_pattern_res_23553);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool bounds_check_25397 = x_25395 && y_25396;
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool index_certs_25398;
                
                if (!bounds_check_25397) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_25394, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/bgpc_vv_coloring.fut:49:40-53\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:40-53
                
                int64_t seen_acczq_25399 = ((int64_t *) ext_mem_26576.mem)[seen_acczq_25394];
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t j_m_i_25400 = sub64(seen_acczq_25399, seen_acczq_25393);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool empty_slice_25401 = j_m_i_25400 == (int64_t) 0;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t m_25402 = sub64(j_m_i_25400, (int64_t) 1);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t i_p_m_t_s_25403 = add64(seen_acczq_25393, m_25402);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_leq_i_p_m_t_s_25404 = sle64((int64_t) 0, i_p_m_t_s_25403);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_p_m_t_s_leq_w_25405 = slt64(i_p_m_t_s_25403, csr_bipartite_from_pattern_res_23554);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_lte_i_25406 = sle64((int64_t) 0, seen_acczq_25393);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_lte_j_25407 = sle64(seen_acczq_25393, seen_acczq_25399);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_25408 = i_p_m_t_s_leq_w_25405 && zzero_lte_i_25406;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_25409 = zzero_leq_i_p_m_t_s_25404 && y_25408;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool forwards_ok_25410 = i_lte_j_25407 && y_25409;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool ok_or_empty_25411 = empty_slice_25401 || forwards_ok_25410;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool index_certs_25412;
                
                if (!ok_or_empty_25411) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_25393, ":", (long long) seen_acczq_25399, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23554, "].", "-> #0  src/bgpc_vv_coloring.fut:48:9-53:33\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:48:9-53:33
                for (int64_t i_25414 = 0; i_25414 < j_m_i_25400; i_25414++) {
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t index_primexp_25416 = seen_acczq_25393 + i_25414;
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t u_25417 = ((int64_t *) ext_mem_26575.mem)[index_primexp_25416];
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool x_25418 = sle64((int64_t) 0, u_25417);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool y_25419 = slt64(u_25417, (int64_t) 6);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool bounds_check_25420 = x_25418 && y_25419;
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool index_certs_25421;
                    
                    if (!bounds_check_25420) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_25417, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/bgpc_vv_coloring.fut:50:20-29\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    int64_t cu_25422 = ((int64_t *) mem_26588)[u_25417];
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_25423 = u_25417 == eta_p_25358;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_25424 = !cond_25423;
                    
                    // src/bgpc_vv_coloring.fut:51:30-37
                    
                    bool cond_t_res_25425 = sle64((int64_t) 0, cu_25422);
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    bool x_25426 = cond_25424 && cond_t_res_25425;
                    
                    // src/bgpc_vv_coloring.fut:51:44-67
                    
                    bool cond_t_res_25427 = slt64(cu_25422, color_bound_23603);
                    
                    // src/bgpc_vv_coloring.fut:94:14-103:69
                    
                    bool x_25428 = x_25426 && cond_t_res_25427;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    if (x_25428) {
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool bounds_check_25983 = cond_t_res_25425 && cond_t_res_25427;
                        
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool index_certs_25984;
                        
                        if (!bounds_check_25983) {
                            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) cu_25422, "] out of bounds for array of shape [", (long long) color_bound_23603, "].", "-> #0  src/bgpc_vv_coloring.fut:52:19-50\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                            err = FUTHARK_PROGRAM_ERROR;
                            goto cleanup;
                        }
                        // src/bgpc_vv_coloring.fut:52:19-50
                        ((bool *) mem_26604)[cu_25422] = 1;
                    }
                }
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool loop_cond_25433;
            
            if (cond_23604) {
                // src/bgpc_vv_coloring.fut:63:30-37
                
                bool loop_cond_t_res_25986 = ((bool *) mem_26604)[(int64_t) 0];
                
                loop_cond_25433 = loop_cond_t_res_25986;
            } else {
                loop_cond_25433 = 0;
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool c_final_25435;
            int64_t c_final_25436;
            bool loop_while_25437;
            int64_t c_25438;
            
            loop_while_25437 = loop_cond_25433;
            c_25438 = (int64_t) 0;
            while (loop_while_25437) {
                // src/bgpc_vv_coloring.fut:64:9-15
                
                int64_t loopres_25439 = add64((int64_t) 1, c_25438);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool cond_25440 = slt64(loopres_25439, color_bound_23603);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool loop_cond_25441;
                
                if (cond_25440) {
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool x_25987 = sle64((int64_t) 0, loopres_25439);
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool bounds_check_25988 = cond_25440 && x_25987;
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool index_certs_25989;
                    
                    if (!bounds_check_25988) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) loopres_25439, "] out of bounds for array of shape [", (long long) color_bound_23603, "].", "-> #0  src/bgpc_vv_coloring.fut:63:30-37\n   #1  src/bgpc_vv_coloring.fut:81:6-28\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool loop_cond_t_res_25990 = ((bool *) mem_26604)[loopres_25439];
                    
                    loop_cond_25441 = loop_cond_t_res_25990;
                } else {
                    loop_cond_25441 = 0;
                }
                
                bool loop_while_tmp_26774 = loop_cond_25441;
                int64_t c_tmp_26775 = loopres_25439;
                
                loop_while_25437 = loop_while_tmp_26774;
                c_25438 = c_tmp_26775;
            }
            c_final_25435 = loop_while_25437;
            c_final_25436 = c_25438;
            // src/bgpc_vv_coloring.fut:106:25-32
            
            int64_t max_res_23698 = smax64(c_final_25436, redout_26397);
            
            ((int64_t *) mem_26600)[i_26399] = c_final_25436;
            
            int64_t redout_tmp_26769 = max_res_23698;
            
            redout_26397 = redout_tmp_26769;
        }
        defunc_0_reduce_res_26019 = redout_26397;
        // src/bgpc_vv_coloring.fut:113:17-43
        
        bool acc_cert_23704;
        
        // src/bgpc_vv_coloring.fut:113:17-43
        for (int64_t i_26402 = 0; i_26402 < loop_dz2081Uz2083U_23599; i_26402++) {
            int64_t v_23708 = ((int64_t *) mem_param_26598.mem)[i_26402];
            int64_t v_23709 = ((int64_t *) mem_26600)[i_26402];
            
            // src/bgpc_vv_coloring.fut:113:17-43
            // UpdateAcc
            if (sle64((int64_t) 0, v_23708) && slt64(v_23708, (int64_t) 6)) {
                ((int64_t *) mem_26588)[v_23708] = v_23709;
            }
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26621_cached_sizze_26888 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26621, &mem_26621_cached_sizze_26888, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26623_cached_sizze_26889 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26623, &mem_26623_cached_sizze_26889, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26624_cached_sizze_26890 < loop_dz2081Uz2083U_23599) {
            err = lexical_realloc(ctx, &mem_26624, &mem_26624_cached_sizze_26890, loop_dz2081Uz2083U_23599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t discard_26414;
        int64_t scanacc_26406 = (int64_t) 0;
        
        for (int64_t i_26410 = 0; i_26410 < loop_dz2081Uz2083U_23599; i_26410++) {
            int64_t eta_p_25282 = ((int64_t *) mem_param_26598.mem)[i_26410];
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool x_25283 = sle64((int64_t) 0, eta_p_25282);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool y_25284 = slt64(eta_p_25282, csr_cols_from_pattern_res_23558);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool bounds_check_25285 = x_25283 && y_25284;
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool index_certs_25286;
            
            if (!bounds_check_25285) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25282, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23558, "].", "-> #0  src/bgpc_vv_coloring.fut:126:15-27\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:127:26-28
            
            int64_t k_end_25288 = add64((int64_t) 1, eta_p_25282);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool x_25289 = sle64((int64_t) 0, k_end_25288);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool y_25290 = slt64(k_end_25288, csr_cols_from_pattern_res_23558);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool bounds_check_25291 = x_25289 && y_25290;
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool index_certs_25292;
            
            if (!bounds_check_25291) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_end_25288, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23558, "].", "-> #0  src/bgpc_vv_coloring.fut:127:15-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:126:15-27
            
            int64_t k0_25287 = ((int64_t *) ext_mem_26592.mem)[eta_p_25282];
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            int64_t k_end_25293 = ((int64_t *) ext_mem_26592.mem)[k_end_25288];
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool cond_25294 = slt64(k0_25287, k_end_25293);
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool loses_conflict_vertex_res_25295;
            bool loses_conflict_vertex_res_25296;
            int64_t loses_conflict_vertex_res_25297;
            bool loop_while_25298;
            bool lost_25299;
            int64_t k_25300;
            
            loop_while_25298 = cond_25294;
            lost_25299 = 0;
            k_25300 = k0_25287;
            while (loop_while_25298) {
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool x_25301 = sle64((int64_t) 0, k_25300);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool y_25302 = slt64(k_25300, csr_cols_from_pattern_res_23559);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool bounds_check_25303 = x_25301 && y_25302;
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool index_certs_25304;
                
                if (!bounds_check_25303) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_25300, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23559, "].", "-> #0  src/bgpc_vv_coloring.fut:132:15-26\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:132:15-26
                
                int64_t v_25305 = ((int64_t *) ext_mem_26591.mem)[k_25300];
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool x_25306 = sle64((int64_t) 0, v_25305);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool y_25307 = slt64(v_25305, csr_bipartite_from_pattern_res_23553);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool bounds_check_25308 = x_25306 && y_25307;
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool index_certs_25309;
                
                if (!bounds_check_25308) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_25305, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/bgpc_vv_coloring.fut:134:19-30\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:135:29-31
                
                int64_t t_end_25311 = add64((int64_t) 1, v_25305);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool x_25312 = sle64((int64_t) 0, t_end_25311);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool y_25313 = slt64(t_end_25311, csr_bipartite_from_pattern_res_23553);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool bounds_check_25314 = x_25312 && y_25313;
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool index_certs_25315;
                
                if (!bounds_check_25314) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_end_25311, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/bgpc_vv_coloring.fut:135:19-32\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:134:19-30
                
                int64_t t0_25310 = ((int64_t *) ext_mem_26576.mem)[v_25305];
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                int64_t t_end_25316 = ((int64_t *) ext_mem_26576.mem)[t_end_25311];
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool cond_25317 = slt64(t0_25310, t_end_25316);
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool loopres_25318;
                bool loopres_25319;
                int64_t loopres_25320;
                bool loop_while_25321;
                bool lost_in_net_25322;
                int64_t t_25323;
                
                loop_while_25321 = cond_25317;
                lost_in_net_25322 = 0;
                t_25323 = t0_25310;
                while (loop_while_25321) {
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool x_25324 = sle64((int64_t) 0, t_25323);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool y_25325 = slt64(t_25323, csr_bipartite_from_pattern_res_23554);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool bounds_check_25326 = x_25324 && y_25325;
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool index_certs_25327;
                    
                    if (!bounds_check_25326) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_25323, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23554, "].", "-> #0  src/bgpc_vv_coloring.fut:140:19-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    int64_t u_25328 = ((int64_t *) ext_mem_26575.mem)[t_25323];
                    
                    // src/bgpc_vv_coloring.fut:142:13-71
                    
                    bool lost_in_netzq_25329;
                    
                    if (lost_in_net_25322) {
                        lost_in_netzq_25329 = 1;
                    } else {
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25330 = u_25328 == eta_p_25282;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25331 = !cond_25330;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25332;
                        
                        if (cond_25331) {
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool x_25992 = sle64((int64_t) 0, u_25328);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool y_25993 = slt64(u_25328, (int64_t) 6);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool bounds_check_25994 = x_25992 && y_25993;
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool index_certs_25995;
                            
                            if (!bounds_check_25994) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_25328, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/bgpc_vv_coloring.fut:142:39-48\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool y_25997 = slt64(eta_p_25282, (int64_t) 6);
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool bounds_check_25998 = x_25283 && y_25997;
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool index_certs_25999;
                            
                            if (!bounds_check_25998) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25282, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/bgpc_vv_coloring.fut:142:52-61\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            int64_t zeze_lhs_25996 = ((int64_t *) mem_26588)[u_25328];
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            int64_t zeze_rhs_26000 = ((int64_t *) mem_26588)[eta_p_25282];
                            
                            // src/bgpc_vv_coloring.fut:142:49-61
                            
                            bool cond_t_res_26001 = zeze_lhs_25996 == zeze_rhs_26000;
                            
                            cond_25332 = cond_t_res_26001;
                        } else {
                            cond_25332 = 0;
                        }
                        // src/bgpc_vv_coloring.fut:142:67-70
                        
                        bool lost_in_netzq_f_res_t_res_25343 = slt64(u_25328, eta_p_25282);
                        
                        // src/bgpc_vv_coloring.fut:160:14-168:57
                        
                        bool x_25344 = cond_25332 && lost_in_netzq_f_res_t_res_25343;
                        
                        lost_in_netzq_25329 = x_25344;
                    }
                    // src/bgpc_vv_coloring.fut:143:31-37
                    
                    int64_t tmp_25345 = add64((int64_t) 1, t_25323);
                    
                    // src/bgpc_vv_coloring.fut:139:15-43
                    
                    bool cond_25346 = slt64(tmp_25345, t_end_25316);
                    
                    // src/bgpc_vv_coloring.fut:139:28-43
                    
                    bool not_res_25347 = !lost_in_netzq_25329;
                    
                    // src/bgpc_vv_coloring.fut:166:5-169:10
                    
                    bool x_25348 = cond_25346 && not_res_25347;
                    bool loop_while_tmp_26784 = x_25348;
                    bool lost_in_net_tmp_26785 = lost_in_netzq_25329;
                    int64_t t_tmp_26786 = tmp_25345;
                    
                    loop_while_25321 = loop_while_tmp_26784;
                    lost_in_net_25322 = lost_in_net_tmp_26785;
                    t_25323 = t_tmp_26786;
                }
                loopres_25318 = loop_while_25321;
                loopres_25319 = lost_in_net_25322;
                loopres_25320 = t_25323;
                // src/bgpc_vv_coloring.fut:160:14-168:57
                
                bool lostzq_25349 = lost_25299 || loopres_25319;
                
                // src/bgpc_vv_coloring.fut:146:20-26
                
                int64_t tmp_25350 = add64((int64_t) 1, k_25300);
                
                // src/bgpc_vv_coloring.fut:131:11-32
                
                bool cond_25351 = slt64(tmp_25350, k_end_25293);
                
                // src/bgpc_vv_coloring.fut:131:24-32
                
                bool not_res_25352 = !lostzq_25349;
                
                // src/bgpc_vv_coloring.fut:166:5-169:10
                
                bool x_25353 = cond_25351 && not_res_25352;
                bool loop_while_tmp_26781 = x_25353;
                bool lost_tmp_26782 = lostzq_25349;
                int64_t k_tmp_26783 = tmp_25350;
                
                loop_while_25298 = loop_while_tmp_26781;
                lost_25299 = lost_tmp_26782;
                k_25300 = k_tmp_26783;
            }
            loses_conflict_vertex_res_25295 = loop_while_25298;
            loses_conflict_vertex_res_25296 = lost_25299;
            loses_conflict_vertex_res_25297 = k_25300;
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_f_res_25355 = btoi_bool_i64(loses_conflict_vertex_res_25296);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_op_res_23810 = add64(defunc_0_f_res_25355, scanacc_26406);
            
            ((int64_t *) mem_26621)[i_26410] = defunc_0_op_res_23810;
            ((int64_t *) mem_26623)[i_26410] = defunc_0_f_res_25355;
            ((bool *) mem_26624)[i_26410] = loses_conflict_vertex_res_25296;
            
            int64_t scanacc_tmp_26777 = defunc_0_op_res_23810;
            
            scanacc_26406 = scanacc_tmp_26777;
        }
        discard_26414 = scanacc_26406;
        // src/bgpc_vv_coloring.fut:23:25-28
        
        int64_t tmp_23794 = sub64(loop_dz2081Uz2083U_23599, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool y_23796 = slt64(tmp_23794, loop_dz2081Uz2083U_23599);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool x_23795 = sle64((int64_t) 0, tmp_23794);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool bounds_check_23797 = x_23795 && y_23796;
        
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        bool cond_23792 = loop_dz2081Uz2083U_23599 == (int64_t) 0;
        
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool protect_assert_disj_23798 = cond_23792 || bounds_check_23797;
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool index_certs_23799;
        
        if (!protect_assert_disj_23798) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_23794, "] out of bounds for array of shape [", (long long) loop_dz2081Uz2083U_23599, "].", "-> #0  src/bgpc_vv_coloring.fut:23:10-29\n   #1  src/bgpc_vv_coloring.fut:171:23-55\n   #2  src/bgpc_vv_coloring.fut:186:14-205:56\n   #3  src/bgpc_vv_coloring.fut:213:14-216:64\n   #4  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #5  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #6  test/test_sparse_jacobian_jvp.fut:152:18-44\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool x_23793 = !cond_23792;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_f_res_23811;
        
        if (x_23793) {
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t x_26002 = ((int64_t *) mem_26621)[tmp_23794];
            
            m_f_res_23811 = x_26002;
        } else {
            m_f_res_23811 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_23813;
        
        if (cond_23792) {
            m_23813 = (int64_t) 0;
        } else {
            m_23813 = m_f_res_23811;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t bytes_26643 = (int64_t) 8 * m_23813;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26644_cached_sizze_26891 < bytes_26643) {
            err = lexical_realloc(ctx, &mem_26644, &mem_26644_cached_sizze_26891, bytes_26643);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26646_cached_sizze_26892 < bytes_26643) {
            err = lexical_realloc(ctx, &mem_26646, &mem_26646_cached_sizze_26892, bytes_26643);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:18:24-27:44
        if (mem_26648_cached_sizze_26893 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26648, &mem_26648_cached_sizze_26893, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26656_cached_sizze_26894 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26656, &mem_26656_cached_sizze_26894, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        bool acc_cert_25155;
        bool acc_cert_25156;
        
        // src/bgpc_vv_coloring.fut:18:24-27:44
        
        int64_t inpacc_26010;
        int64_t inpacc_25219 = (int64_t) 0;
        
        for (int64_t i_26441 = 0; i_26441 < loop_dz2081Uz2083U_23599; i_26441++) {
            bool eta_p_26503 = ((bool *) mem_26624)[i_26441];
            
            // src/bgpc_vv_coloring.fut:18:28-39
            
            int64_t bool_to_i64_res_26504 = btoi_bool_i64(eta_p_26503);
            int64_t eta_p_26516 = ((int64_t *) mem_26623)[i_26441];
            int64_t eta_p_26517 = ((int64_t *) mem_26621)[i_26441];
            int64_t v_26520 = ((int64_t *) mem_param_26598.mem)[i_26441];
            
            // src/bgpc_vv_coloring.fut:19:29-32
            
            int64_t defunc_0_op_res_26521 = add64(inpacc_25219, bool_to_i64_res_26504);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            bool cond_26522 = eta_p_26516 == (int64_t) 1;
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t lifted_lambda_res_26523;
            
            if (cond_26522) {
                // src/bgpc_vv_coloring.fut:27:5-44
                
                int64_t lifted_lambda_res_t_res_26524 = sub64(eta_p_26517, (int64_t) 1);
                
                lifted_lambda_res_26523 = lifted_lambda_res_t_res_26524;
            } else {
                lifted_lambda_res_26523 = (int64_t) -1;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_26523) && slt64(lifted_lambda_res_26523, m_23813)) {
                ((int64_t *) mem_26646)[lifted_lambda_res_26523] = v_26520;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_26523) && slt64(lifted_lambda_res_26523, m_23813)) {
                ((int64_t *) mem_26644)[lifted_lambda_res_26523] = defunc_0_op_res_26521;
            }
            ((int64_t *) mem_26648)[i_26441] = defunc_0_op_res_26521;
            
            int64_t inpacc_tmp_26787 = defunc_0_op_res_26521;
            
            inpacc_25219 = inpacc_tmp_26787;
        }
        inpacc_26010 = inpacc_25219;
        // src/bgpc_vv_coloring.fut:27:5-44
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_26656, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_26648, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {loop_dz2081Uz2083U_23599});
        // src/bgpc_vv_coloring.fut:23:10-29
        
        int64_t out_szz_f_res_23800;
        
        if (x_23793) {
            // src/bgpc_vv_coloring.fut:171:23-55
            
            int64_t x_26016 = ((int64_t *) mem_26656)[tmp_23794];
            
            out_szz_f_res_23800 = x_26016;
        } else {
            out_szz_f_res_23800 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        int64_t out_szz_23802;
        
        if (cond_23792) {
            out_szz_23802 = (int64_t) 0;
        } else {
            out_szz_23802 = out_szz_f_res_23800;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        
        int64_t bytes_26657 = (int64_t) 8 * out_szz_23802;
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        bool cond_23699 = slt64(defunc_0_reduce_res_26019, (int64_t) 0);
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        int64_t next_color_bound_23700;
        
        if (cond_23699) {
            next_color_bound_23700 = color_bound_23603;
        } else {
            // src/bgpc_vv_coloring.fut:110:40-46
            
            int64_t max_arg1_23701 = add64((int64_t) 2, defunc_0_reduce_res_26019);
            
            // src/bgpc_vv_coloring.fut:110:10-46
            
            int64_t max_res_23702 = smax64(color_bound_23603, max_arg1_23701);
            
            next_color_bound_23700 = max_res_23702;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        if (memblock_alloc(ctx, &mem_26658, bytes_26657, "mem_26658")) {
            err = 1;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        for (int64_t nest_i_26791 = 0; nest_i_26791 < out_szz_23802; nest_i_26791++) {
            ((int64_t *) mem_26658.mem)[nest_i_26791] = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:33:6-27
        
        bool acc_cert_24928;
        
        // src/bgpc_vv_coloring.fut:29:21-33:27
        for (int64_t i_26446 = 0; i_26446 < m_23813; i_26446++) {
            int64_t eta_p_24940 = ((int64_t *) mem_26644)[i_26446];
            int64_t v_24942 = ((int64_t *) mem_26646)[i_26446];
            
            // src/bgpc_vv_coloring.fut:29:44-50
            
            int64_t lifted_lambda_res_24943 = sub64(eta_p_24940, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:33:6-27
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_24943) && slt64(lifted_lambda_res_24943, out_szz_23802)) {
                ((int64_t *) mem_26658.mem)[lifted_lambda_res_24943] = v_24942;
            }
        }
        // src/bgpc_vv_coloring.fut:175:17-48
        
        bool acc_cert_23850;
        
        // src/bgpc_vv_coloring.fut:175:17-48
        for (int64_t i_26448 = 0; i_26448 < out_szz_23802; i_26448++) {
            int64_t v_23854 = ((int64_t *) mem_26658.mem)[i_26448];
            
            // src/bgpc_vv_coloring.fut:175:17-48
            // UpdateAcc
            if (sle64((int64_t) 0, v_23854) && slt64(v_23854, (int64_t) 6)) {
                ((int64_t *) mem_26588)[v_23854] = (int64_t) -1;
            }
        }
        // src/bgpc_vv_coloring.fut:198:20-23
        
        bool loop_cond_23856 = slt64((int64_t) 0, out_szz_23802);
        
        if (memblock_set(ctx, &mem_param_tmp_26763, &mem_26658, "mem_26658") != 0)
            return 1;
        
        int64_t loop_dz2081Uz2083U_tmp_26764 = out_szz_23802;
        bool loop_while_tmp_26765 = loop_cond_23856;
        int64_t color_bound_tmp_26768 = next_color_bound_23700;
        
        if (memblock_set(ctx, &mem_param_26598, &mem_param_tmp_26763, "mem_param_tmp_26763") != 0)
            return 1;
        loop_dz2081Uz2083U_23599 = loop_dz2081Uz2083U_tmp_26764;
        loop_while_23600 = loop_while_tmp_26765;
        color_bound_23603 = color_bound_tmp_26768;
    }
    if (memblock_set(ctx, &ext_mem_26663, &mem_param_26598, "mem_param_26598") != 0)
        return 1;
    vv_color_side_order_res_23594 = loop_dz2081Uz2083U_23599;
    vv_color_side_order_res_23595 = loop_while_23600;
    vv_color_side_order_res_23598 = color_bound_23603;
    if (memblock_unref(ctx, &mem_26587, "mem_26587") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26591, "ext_mem_26591") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26592, "ext_mem_26592") != 0)
        return 1;
    // src/sparse_jacobian_jvp.fut:107:8-112:40
    
    int64_t x_26031;
    int64_t redout_26449 = (int64_t) 0;
    
    for (int64_t i_26450 = 0; i_26450 < (int64_t) 6; i_26450++) {
        int64_t x_23861 = ((int64_t *) mem_26588)[i_26450];
        
        // src/sparse_jacobian_jvp.fut:14:22-29
        
        int64_t max_res_23864 = smax64(x_23861, redout_26449);
        int64_t redout_tmp_26794 = max_res_23864;
        
        redout_26449 = redout_tmp_26794;
    }
    x_26031 = redout_26449;
    // src/sparse_jacobian_jvp.fut:14:13-41
    
    int64_t num_colors_of_res_f_res_23865 = add64((int64_t) 1, x_26031);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool bounds_invalid_upwards_23867 = slt64(num_colors_of_res_f_res_23865, (int64_t) 0);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool valid_23868 = !bounds_invalid_upwards_23867;
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool range_valid_c_23869;
    
    if (!valid_23868) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Range ", (long long) (int64_t) 0, "..", (long long) (int64_t) 1, "..<", (long long) num_colors_of_res_f_res_23865, " is invalid.", "-> #0  src/sparse_jacobian_jvp.fut:29:11-18\n   #1  src/sparse_jacobian_jvp.fut:107:8-112:40\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-153:59\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    
    int64_t bytes_26666 = (int64_t) 32 * num_colors_of_res_f_res_23865;
    double zt_lhs_24490 = ((double *) x_mem_26563.mem)[(int64_t) 0];
    double zt_rhs_24491 = ((double *) x_mem_26563.mem)[(int64_t) 1];
    double zt_rhs_24493 = ((double *) x_mem_26563.mem)[(int64_t) 4];
    
    // test/test_sparse_jacobian_jvp.fut:82:33-39
    
    double binop_x_25872 = 0.0 * zt_rhs_24493;
    double zt_lhs_24497 = ((double *) x_mem_26563.mem)[(int64_t) 5];
    
    // src/dense_jacobian.fut:8:40-9:68
    if (mem_26564_cached_sizze_26881 < (int64_t) 192) {
        err = lexical_realloc(ctx, &mem_26564, &mem_26564_cached_sizze_26881, (int64_t) 192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:5:3-21
    if (mem_26569_cached_sizze_26882 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26569, &mem_26569_cached_sizze_26882, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:8:40-9:68
    for (int64_t i_26386 = 0; i_26386 < (int64_t) 6; i_26386++) {
        // src/dense_jacobian.fut:5:3-21
        for (int64_t nest_i_26796 = 0; nest_i_26796 < (int64_t) 6; nest_i_26796++) {
            ((double *) mem_26569)[nest_i_26796] = 0.0;
        }
        // src/dense_jacobian.fut:5:3-39
        ((double *) mem_26569)[i_26386] = 1.0;
        
        double zt_lhs_tan_25865 = ((double *) mem_26569)[(int64_t) 0];
        double zt_rhs_tan_25866 = ((double *) mem_26569)[(int64_t) 1];
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_x_25868 = zt_rhs_24491 * zt_lhs_tan_25865;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_y_25869 = zt_lhs_24490 * zt_rhs_tan_25866;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double zp_lhs_tan_25867 = binop_x_25868 + binop_y_25869;
        double zt_rhs_tan_25870 = ((double *) mem_26569)[(int64_t) 4];
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double binop_y_25873 = 3.0 * zt_rhs_tan_25870;
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double zp_rhs_tan_25871 = binop_x_25872 + binop_y_25873;
        
        // test/test_sparse_jacobian_jvp.fut:82:24-39
        
        double y0_tan_25874 = zp_lhs_tan_25867 + zp_rhs_tan_25871;
        double zm_lhs_tan_25877 = ((double *) mem_26569)[(int64_t) 2];
        double zt_lhs_tan_25878 = ((double *) mem_26569)[(int64_t) 5];
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double binop_x_25880 = zt_lhs_24497 * zt_lhs_tan_25878;
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double zm_rhs_tan_25879 = binop_x_25880 + binop_x_25880;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double binop_y_25884 = -1.0 * zm_rhs_tan_25879;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double y1_tan_25882 = zm_lhs_tan_25877 + binop_y_25884;
        
        // test/test_sparse_jacobian_jvp.fut:84:17-23
        
        double zp_lhs_tan_25885 = zt_rhs_tan_25866 + zm_lhs_tan_25877;
        double zp_rhs_tan_25888 = ((double *) mem_26569)[(int64_t) 3];
        
        // test/test_sparse_jacobian_jvp.fut:84:24-30
        
        double y2_tan_25889 = zp_lhs_tan_25885 + zp_rhs_tan_25888;
        
        // test/test_sparse_jacobian_jvp.fut:151:59-64
        ((double *) mem_26564)[i_26386 * (int64_t) 4] = y0_tan_25874;
        ((double *) mem_26564)[i_26386 * (int64_t) 4 + (int64_t) 1] = y1_tan_25882;
        ((double *) mem_26564)[i_26386 * (int64_t) 4 + (int64_t) 2] = y2_tan_25889;
        ((double *) mem_26564)[i_26386 * (int64_t) 4 + (int64_t) 3] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    if (mem_26578_cached_sizze_26883 < bytes_26577) {
        err = lexical_realloc(ctx, &mem_26578, &mem_26578_cached_sizze_26883, bytes_26577);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    for (int64_t nest_i_26797 = 0; nest_i_26797 < csr_bipartite_from_pattern_res_23554; nest_i_26797++) {
        ((double *) mem_26578)[nest_i_26797] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    if (mem_26667_cached_sizze_26895 < bytes_26666) {
        err = lexical_realloc(ctx, &mem_26667, &mem_26667_cached_sizze_26895, bytes_26666);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:17:3-61
    if (mem_26672_cached_sizze_26896 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26672, &mem_26672_cached_sizze_26896, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    for (int64_t i_26457 = 0; i_26457 < num_colors_of_res_f_res_23865; i_26457++) {
        // src/sparse_jacobian_jvp.fut:17:3-61
        for (int64_t i_26453 = 0; i_26453 < (int64_t) 6; i_26453++) {
            int64_t eta_p_23878 = ((int64_t *) mem_26588)[i_26453];
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            bool cond_23879 = eta_p_23878 == i_26457;
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            double lifted_lambda_res_23880;
            
            if (cond_23879) {
                lifted_lambda_res_23880 = 1.0;
            } else {
                lifted_lambda_res_23880 = 0.0;
            }
            ((double *) mem_26672)[i_26453] = lifted_lambda_res_23880;
        }
        
        double zt_lhs_tan_25894 = ((double *) mem_26672)[(int64_t) 0];
        double zt_rhs_tan_25895 = ((double *) mem_26672)[(int64_t) 1];
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_x_25897 = zt_rhs_24491 * zt_lhs_tan_25894;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_y_25898 = zt_lhs_24490 * zt_rhs_tan_25895;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double zp_lhs_tan_25896 = binop_x_25897 + binop_y_25898;
        double zt_rhs_tan_25899 = ((double *) mem_26672)[(int64_t) 4];
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double binop_y_25902 = 3.0 * zt_rhs_tan_25899;
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double zp_rhs_tan_25900 = binop_x_25872 + binop_y_25902;
        
        // test/test_sparse_jacobian_jvp.fut:82:24-39
        
        double y0_tan_25903 = zp_lhs_tan_25896 + zp_rhs_tan_25900;
        double zm_lhs_tan_25906 = ((double *) mem_26672)[(int64_t) 2];
        double zt_lhs_tan_25907 = ((double *) mem_26672)[(int64_t) 5];
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double binop_x_25909 = zt_lhs_24497 * zt_lhs_tan_25907;
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double zm_rhs_tan_25908 = binop_x_25909 + binop_x_25909;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double binop_y_25913 = -1.0 * zm_rhs_tan_25908;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double y1_tan_25911 = zm_lhs_tan_25906 + binop_y_25913;
        
        // test/test_sparse_jacobian_jvp.fut:84:17-23
        
        double zp_lhs_tan_25914 = zt_rhs_tan_25895 + zm_lhs_tan_25906;
        double zp_rhs_tan_25917 = ((double *) mem_26672)[(int64_t) 3];
        
        // test/test_sparse_jacobian_jvp.fut:84:24-30
        
        double y2_tan_25918 = zp_lhs_tan_25914 + zp_rhs_tan_25917;
        
        // test/test_sparse_jacobian_jvp.fut:153:43-48
        ((double *) mem_26667)[i_26457 * (int64_t) 4] = y0_tan_25903;
        ((double *) mem_26667)[i_26457 * (int64_t) 4 + (int64_t) 1] = y1_tan_25911;
        ((double *) mem_26667)[i_26457 * (int64_t) 4 + (int64_t) 2] = y2_tan_25918;
        ((double *) mem_26667)[i_26457 * (int64_t) 4 + (int64_t) 3] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:45:5-52:27
    
    bool compressed_to_csr_vals_res_23889;
    int64_t compressed_to_csr_vals_res_23891;
    bool loop_while_23892;
    int64_t i_23894;
    
    loop_while_23892 = 1;
    i_23894 = (int64_t) 0;
    while (loop_while_23892) {
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool x_23895 = sle64((int64_t) 0, i_23894);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool y_23896 = slt64(i_23894, csr_bipartite_from_pattern_res_23553);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool bounds_check_23897 = x_23895 && y_23896;
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool index_certs_23898;
        
        if (!bounds_check_23897) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_23894, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/sparse_jacobian_jvp.fut:47:15-26\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-153:59\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        int64_t s_23899 = ((int64_t *) ext_mem_26576.mem)[i_23894];
        
        // src/sparse_jacobian_jvp.fut:48:25-27
        
        int64_t e_23900 = add64((int64_t) 1, i_23894);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool x_23901 = sle64((int64_t) 0, e_23900);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool y_23902 = slt64(e_23900, csr_bipartite_from_pattern_res_23553);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool bounds_check_23903 = x_23901 && y_23902;
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool index_certs_23904;
        
        if (!bounds_check_23903) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_23900, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/sparse_jacobian_jvp.fut:48:15-28\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-153:59\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        int64_t e_23905 = ((int64_t *) ext_mem_26576.mem)[e_23900];
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t j_m_i_23906 = sub64(e_23905, s_23899);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool empty_slice_23907 = j_m_i_23906 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t m_23908 = sub64(j_m_i_23906, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t i_p_m_t_s_23909 = add64(s_23899, m_23908);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_23910 = sle64((int64_t) 0, i_p_m_t_s_23909);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_23911 = slt64(i_p_m_t_s_23909, csr_bipartite_from_pattern_res_23554);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_lte_i_23912 = sle64((int64_t) 0, s_23899);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_lte_j_23913 = sle64(s_23899, e_23905);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_23914 = i_p_m_t_s_leq_w_23911 && zzero_lte_i_23912;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_23915 = zzero_leq_i_p_m_t_s_23910 && y_23914;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool forwards_ok_23916 = i_lte_j_23913 && y_23915;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool ok_or_empty_23917 = empty_slice_23907 || forwards_ok_23916;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool index_certs_23918;
        
        if (!ok_or_empty_23917) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_23899, ":", (long long) e_23905, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23554, "].", "-> #0  src/sparse_jacobian_jvp.fut:49:18-30\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-153:59\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool y_23920 = slt64(i_23894, (int64_t) 4);
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool bounds_check_23921 = x_23895 && y_23920;
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool index_certs_23922;
        
        if (!bounds_check_23921) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_23894, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-45\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-153:59\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:18-51
        for (int64_t i_26461 = 0; i_26461 < j_m_i_23906; i_26461++) {
            int64_t index_primexp_26496 = s_23899 + i_26461;
            int64_t eta_p_23924 = ((int64_t *) ext_mem_26575.mem)[index_primexp_26496];
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool x_23925 = sle64((int64_t) 0, eta_p_23924);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool y_23926 = slt64(eta_p_23924, (int64_t) 6);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool bounds_check_23927 = x_23925 && y_23926;
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool index_certs_23928;
            
            if (!bounds_check_23927) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_23924, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:32-41\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-153:59\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            int64_t tmp_23929 = ((int64_t *) mem_26588)[eta_p_23924];
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool x_23930 = sle64((int64_t) 0, tmp_23929);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool y_23931 = slt64(tmp_23929, num_colors_of_res_f_res_23865);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool bounds_check_23932 = x_23930 && y_23931;
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool index_certs_23933;
            
            if (!bounds_check_23932) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_23929, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_23865, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-42\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-153:59\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:18-51
            
            double lifted_lambda_res_23934 = ((double *) mem_26667)[tmp_23929 * (int64_t) 4 + i_23894];
            
            ((double *) mem_26578)[s_23899 + i_26461] = lifted_lambda_res_23934;
        }
        // src/sparse_jacobian_jvp.fut:46:13-16
        
        bool loop_cond_23936 = slt64(e_23900, (int64_t) 4);
        bool loop_while_tmp_26800 = loop_cond_23936;
        int64_t i_tmp_26802 = e_23900;
        
        loop_while_23892 = loop_while_tmp_26800;
        i_23894 = i_tmp_26802;
    }
    compressed_to_csr_vals_res_23889 = loop_while_23892;
    compressed_to_csr_vals_res_23891 = i_23894;
    // src/sparse_jacobian_jvp.fut:67:29-47
    if (mem_26698_cached_sizze_26897 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26698, &mem_26698_cached_sizze_26897, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:8:19-10:39
    
    bool defunc_0_reduce_res_26033;
    bool redout_26467 = 1;
    
    for (int64_t i_26468 = 0; i_26468 < (int64_t) 4; i_26468++) {
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool y_24851 = slt64(i_26468, csr_bipartite_from_pattern_res_23553);
        
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool index_certs_24853;
        
        if (!y_24851) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_26468, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/sparse_jacobian_jvp.fut:63:17-28\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  test/test_sparse_jacobian_jvp.fut:81:15-153:59\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        int64_t s_24854 = ((int64_t *) ext_mem_26576.mem)[i_26468];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_lte_i_24867 = sle64((int64_t) 0, s_24854);
        
        // src/sparse_jacobian_jvp.fut:64:27-29
        
        int64_t e_24855 = add64((int64_t) 1, i_26468);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool y_24857 = slt64(e_24855, csr_bipartite_from_pattern_res_23553);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool x_24856 = sle64((int64_t) 0, e_24855);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool bounds_check_24858 = x_24856 && y_24857;
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool index_certs_24859;
        
        if (!bounds_check_24858) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_24855, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23553, "].", "-> #0  src/sparse_jacobian_jvp.fut:64:17-30\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  test/test_sparse_jacobian_jvp.fut:81:15-153:59\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        int64_t e_24860 = ((int64_t *) ext_mem_26576.mem)[e_24855];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t j_m_i_24861 = sub64(e_24860, s_24854);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t m_24863 = sub64(j_m_i_24861, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t i_p_m_t_s_24864 = add64(s_24854, m_24863);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_p_m_t_s_leq_w_24866 = slt64(i_p_m_t_s_24864, csr_bipartite_from_pattern_res_23554);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24875 = i_p_m_t_s_leq_w_24866 && zzero_lte_i_24867;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_leq_i_p_m_t_s_24865 = sle64((int64_t) 0, i_p_m_t_s_24864);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24876 = zzero_leq_i_p_m_t_s_24865 && y_24875;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_lte_j_24868 = sle64(s_24854, e_24860);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool forwards_ok_24877 = i_lte_j_24868 && y_24876;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool empty_slice_24862 = j_m_i_24861 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool ok_or_empty_24878 = empty_slice_24862 || forwards_ok_24877;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool index_certs_24879;
        
        if (!ok_or_empty_24878) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_24854, ":", (long long) e_24860, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23554, "].", "-> #0  src/sparse_jacobian_jvp.fut:66:20-29\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  test/test_sparse_jacobian_jvp.fut:81:15-153:59\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:67:29-47
        for (int64_t nest_i_26805 = 0; nest_i_26805 < (int64_t) 6; nest_i_26805++) {
            ((double *) mem_26698)[nest_i_26805] = 0.0;
        }
        // src/sparse_jacobian_jvp.fut:68:12-32
        
        bool acc_cert_24883;
        
        // src/sparse_jacobian_jvp.fut:68:12-32
        for (int64_t i_26464 = 0; i_26464 < j_m_i_24861; i_26464++) {
            int64_t index_primexp_26493 = s_24854 + i_26464;
            int64_t v_24887 = ((int64_t *) ext_mem_26575.mem)[index_primexp_26493];
            double v_24888 = ((double *) mem_26578)[index_primexp_26493];
            
            // src/sparse_jacobian_jvp.fut:68:12-32
            // UpdateAcc
            if (sle64((int64_t) 0, v_24887) && slt64(v_24887, (int64_t) 6)) {
                ((double *) mem_26698)[v_24887] = v_24888;
            }
        }
        // test/test_sparse_jacobian_jvp.fut:9:5-66
        
        bool defunc_0_reduce_res_26028;
        bool redout_26465 = 1;
        
        for (int64_t i_26466 = 0; i_26466 < (int64_t) 6; i_26466++) {
            bool eta_p_24911 = ((bool *) mem_26555.mem)[i_26468 * (int64_t) 6 + i_26466];
            double eta_p_24912 = ((double *) mem_26564)[i_26466 * (int64_t) 4 + i_26468];
            double eta_p_24913 = ((double *) mem_26698)[i_26466];
            
            // test/test_sparse_jacobian_jvp.fut:14:25-48
            
            double lifted_lambda_res_24914;
            
            if (eta_p_24911) {
                lifted_lambda_res_24914 = eta_p_24912;
            } else {
                lifted_lambda_res_24914 = 0.0;
            }
            // test/test_sparse_jacobian_jvp.fut:9:48-51
            
            double abs_arg0_24916 = eta_p_24913 - lifted_lambda_res_24914;
            
            // test/test_sparse_jacobian_jvp.fut:9:37-51
            
            double abs_res_24917 = fabs64(abs_arg0_24916);
            
            // test/test_sparse_jacobian_jvp.fut:9:53-59
            
            bool lifted_lambda_res_24918 = abs_res_24917 <= 1.0e-9;
            
            // test/test_sparse_jacobian_jvp.fut:9:5-66
            
            bool x_24901 = lifted_lambda_res_24918 && redout_26465;
            bool redout_tmp_26807 = x_24901;
            
            redout_26465 = redout_tmp_26807;
        }
        defunc_0_reduce_res_26028 = redout_26465;
        // test/test_sparse_jacobian_jvp.fut:10:6-39
        
        bool x_24614 = defunc_0_reduce_res_26028 && redout_26467;
        bool redout_tmp_26804 = x_24614;
        
        redout_26467 = redout_tmp_26804;
    }
    defunc_0_reduce_res_26033 = redout_26467;
    if (memblock_unref(ctx, &ext_mem_26575, "ext_mem_26575") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26576, "ext_mem_26576") != 0)
        return 1;
    prim_out_26758 = defunc_0_reduce_res_26033;
    *out_prim_out_26880 = prim_out_26758;
    
  cleanup:
    {
        free(mem_26564);
        free(mem_26569);
        free(mem_26578);
        free(mem_26579);
        free(mem_26588);
        free(mem_26600);
        free(mem_26604);
        free(mem_26621);
        free(mem_26623);
        free(mem_26624);
        free(mem_26644);
        free(mem_26646);
        free(mem_26648);
        free(mem_26656);
        free(mem_26667);
        free(mem_26672);
        free(mem_26698);
        if (memblock_unref(ctx, &mem_param_tmp_26763, "mem_param_tmp_26763") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26658, "mem_26658") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_26598, "mem_param_26598") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26663, "ext_mem_26663") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26591, "ext_mem_26591") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26592, "ext_mem_26592") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26589, "mem_26589") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26587, "mem_26587") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26575, "ext_mem_26575") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26576, "ext_mem_26576") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_prepared_jvp_reuse_two_points(struct futhark_context *ctx, bool *out_prim_out_26898, struct memblock x1_mem_26563)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_26570_cached_sizze_26899 = 0;
    unsigned char *mem_26570 = NULL;
    int64_t mem_26571_cached_sizze_26900 = 0;
    unsigned char *mem_26571 = NULL;
    int64_t mem_26587_cached_sizze_26901 = 0;
    unsigned char *mem_26587 = NULL;
    int64_t mem_26591_cached_sizze_26902 = 0;
    unsigned char *mem_26591 = NULL;
    int64_t mem_26608_cached_sizze_26903 = 0;
    unsigned char *mem_26608 = NULL;
    int64_t mem_26610_cached_sizze_26904 = 0;
    unsigned char *mem_26610 = NULL;
    int64_t mem_26611_cached_sizze_26905 = 0;
    unsigned char *mem_26611 = NULL;
    int64_t mem_26631_cached_sizze_26906 = 0;
    unsigned char *mem_26631 = NULL;
    int64_t mem_26633_cached_sizze_26907 = 0;
    unsigned char *mem_26633 = NULL;
    int64_t mem_26635_cached_sizze_26908 = 0;
    unsigned char *mem_26635 = NULL;
    int64_t mem_26643_cached_sizze_26909 = 0;
    unsigned char *mem_26643 = NULL;
    int64_t mem_26652_cached_sizze_26910 = 0;
    unsigned char *mem_26652 = NULL;
    int64_t mem_26657_cached_sizze_26911 = 0;
    unsigned char *mem_26657 = NULL;
    int64_t mem_26664_cached_sizze_26912 = 0;
    unsigned char *mem_26664 = NULL;
    int64_t mem_26667_cached_sizze_26913 = 0;
    unsigned char *mem_26667 = NULL;
    int64_t mem_26672_cached_sizze_26914 = 0;
    unsigned char *mem_26672 = NULL;
    int64_t mem_26688_cached_sizze_26915 = 0;
    unsigned char *mem_26688 = NULL;
    int64_t mem_26698_cached_sizze_26916 = 0;
    unsigned char *mem_26698 = NULL;
    int64_t mem_26700_cached_sizze_26917 = 0;
    unsigned char *mem_26700 = NULL;
    int64_t mem_26703_cached_sizze_26918 = 0;
    unsigned char *mem_26703 = NULL;
    int64_t mem_26708_cached_sizze_26919 = 0;
    unsigned char *mem_26708 = NULL;
    int64_t mem_26734_cached_sizze_26920 = 0;
    unsigned char *mem_26734 = NULL;
    int64_t mem_26739_cached_sizze_26921 = 0;
    unsigned char *mem_26739 = NULL;
    int64_t mem_26745_cached_sizze_26922 = 0;
    unsigned char *mem_26745 = NULL;
    struct memblock mem_param_tmp_26763;
    
    mem_param_tmp_26763.references = NULL;
    
    struct memblock mem_26645;
    
    mem_26645.references = NULL;
    
    struct memblock mem_param_26585;
    
    mem_param_26585.references = NULL;
    
    struct memblock ext_mem_26650;
    
    ext_mem_26650.references = NULL;
    
    struct memblock mem_26579;
    
    mem_26579.references = NULL;
    
    struct memblock ext_mem_26568;
    
    ext_mem_26568.references = NULL;
    
    struct memblock ext_mem_26569;
    
    ext_mem_26569.references = NULL;
    
    struct memblock mem_26566;
    
    mem_26566.references = NULL;
    
    struct memblock ext_mem_26564;
    
    ext_mem_26564.references = NULL;
    
    struct memblock ext_mem_26565;
    
    ext_mem_26565.references = NULL;
    
    struct memblock mem_26545 = ctx->constants->mem_26545;
    struct memblock mem_26548 = ctx->constants->mem_26548;
    struct memblock mem_26550 = ctx->constants->mem_26550;
    struct memblock mem_26555 = ctx->constants->mem_26555;
    struct memblock mem_26556 = ctx->constants->mem_26556;
    struct memblock mem_26562 = ctx->constants->mem_26562;
    bool prim_out_26758;
    
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_23526;
    int64_t csr_bipartite_from_pattern_res_23527;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26565, &ext_mem_26564, &csr_bipartite_from_pattern_res_23526, &csr_bipartite_from_pattern_res_23527, mem_26555, (int64_t) 4, (int64_t) 6) != 0) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:25:3-39
    if (memblock_alloc(ctx, &mem_26566, (int64_t) 24, "mem_26566")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:25:3-39
    // src/pattern_csr.fut:25:3-39
    lmad_copy_1b(ctx, 2, (uint8_t *) mem_26566.mem, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint8_t *) mem_26555.mem, (int64_t) 0, (int64_t []) {(int64_t) 1, (int64_t) 6}, (int64_t []) {(int64_t) 6, (int64_t) 4});
    // src/pattern_csr.fut:25:3-39
    
    int64_t csr_cols_from_pattern_res_23531;
    int64_t csr_cols_from_pattern_res_23532;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26569, &ext_mem_26568, &csr_cols_from_pattern_res_23531, &csr_cols_from_pattern_res_23532, mem_26566, (int64_t) 6, (int64_t) 4) != 0) {
        err = 1;
        goto cleanup;
    }
    if (memblock_unref(ctx, &mem_26566, "mem_26566") != 0)
        return 1;
    // src/bgpc_vv_coloring.fut:190:35-57
    if (mem_26570_cached_sizze_26899 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26570, &mem_26570_cached_sizze_26899, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    for (int64_t nest_i_26759 = 0; nest_i_26759 < (int64_t) 6; nest_i_26759++) {
        ((int64_t *) mem_26570)[nest_i_26759] = (int64_t) -1;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    if (mem_26571_cached_sizze_26900 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26571, &mem_26571_cached_sizze_26900, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t discard_26389;
    int64_t scanacc_26385 = (int64_t) 0;
    
    for (int64_t i_26387 = 0; i_26387 < (int64_t) 6; i_26387++) {
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t defunc_0_op_res_23542 = add64((int64_t) 1, scanacc_26385);
        
        ((int64_t *) mem_26571)[i_26387] = defunc_0_op_res_23542;
        
        int64_t scanacc_tmp_26760 = defunc_0_op_res_23542;
        
        scanacc_26385 = scanacc_tmp_26760;
    }
    discard_26389 = scanacc_26385;
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t x_23552 = ((int64_t *) mem_26571)[(int64_t) 5];
    
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t bytes_26578 = (int64_t) 8 * x_23552;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (memblock_alloc(ctx, &mem_26579, bytes_26578, "mem_26579")) {
        err = 1;
        goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    bool acc_cert_24883;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    for (int64_t i_26391 = 0; i_26391 < (int64_t) 6; i_26391++) {
        int64_t eta_p_24895 = ((int64_t *) mem_26571)[i_26391];
        
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t lifted_lambda_res_t_res_24898 = sub64(eta_p_24895, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:194:20-45
        // UpdateAcc
        if (sle64((int64_t) 0, lifted_lambda_res_t_res_24898) && slt64(lifted_lambda_res_t_res_24898, x_23552)) {
            ((int64_t *) mem_26579.mem)[lifted_lambda_res_t_res_24898] = i_26391;
        }
    }
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    bool loop_cond_23566 = slt64((int64_t) 0, x_23552);
    
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    int64_t vv_color_side_order_res_23567;
    bool vv_color_side_order_res_23568;
    int64_t vv_color_side_order_res_23571;
    int64_t loop_dz2081Uz2083U_23572;
    bool loop_while_23573;
    int64_t color_bound_23576;
    
    if (memblock_set(ctx, &mem_param_26585, &mem_26579, "mem_26579") != 0)
        return 1;
    loop_dz2081Uz2083U_23572 = x_23552;
    loop_while_23573 = loop_cond_23566;
    color_bound_23576 = (int64_t) 1;
    while (loop_while_23573) {
        // src/bgpc_vv_coloring.fut:62:5-64:15
        
        bool cond_23577 = slt64((int64_t) 0, color_bound_23576);
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t bytes_26586 = (int64_t) 8 * loop_dz2081Uz2083U_23572;
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        if (mem_26587_cached_sizze_26901 < bytes_26586) {
            err = lexical_realloc(ctx, &mem_26587, &mem_26587_cached_sizze_26901, bytes_26586);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:77:24-51
        if (mem_26591_cached_sizze_26902 < color_bound_23576) {
            err = lexical_realloc(ctx, &mem_26591, &mem_26591_cached_sizze_26902, color_bound_23576);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t defunc_0_reduce_res_26273;
        int64_t redout_26393 = (int64_t) -1;
        
        for (int64_t i_26395 = 0; i_26395 < loop_dz2081Uz2083U_23572; i_26395++) {
            int64_t eta_p_25341 = ((int64_t *) mem_param_26585.mem)[i_26395];
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool x_25343 = sle64((int64_t) 0, eta_p_25341);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool y_25344 = slt64(eta_p_25341, csr_cols_from_pattern_res_23531);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool bounds_check_25345 = x_25343 && y_25344;
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool index_certs_25346;
            
            if (!bounds_check_25345) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25341, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23531, "].", "-> #0  src/bgpc_vv_coloring.fut:46:44-56\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:44-56
            
            int64_t seen_final_25347 = ((int64_t *) ext_mem_26569.mem)[eta_p_25341];
            
            // src/bgpc_vv_coloring.fut:46:70-72
            
            int64_t seen_final_25348 = add64((int64_t) 1, eta_p_25341);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool x_25349 = sle64((int64_t) 0, seen_final_25348);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool y_25350 = slt64(seen_final_25348, csr_cols_from_pattern_res_23531);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool bounds_check_25351 = x_25349 && y_25350;
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool index_certs_25352;
            
            if (!bounds_check_25351) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_25348, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23531, "].", "-> #0  src/bgpc_vv_coloring.fut:46:59-73\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:59-73
            
            int64_t seen_final_25353 = ((int64_t *) ext_mem_26569.mem)[seen_final_25348];
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t j_m_i_25354 = sub64(seen_final_25353, seen_final_25347);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool empty_slice_25355 = j_m_i_25354 == (int64_t) 0;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t m_25356 = sub64(j_m_i_25354, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t i_p_m_t_s_25357 = add64(seen_final_25347, m_25356);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_leq_i_p_m_t_s_25358 = sle64((int64_t) 0, i_p_m_t_s_25357);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_p_m_t_s_leq_w_25359 = slt64(i_p_m_t_s_25357, csr_cols_from_pattern_res_23532);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_lte_i_25360 = sle64((int64_t) 0, seen_final_25347);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_lte_j_25361 = sle64(seen_final_25347, seen_final_25353);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_25362 = i_p_m_t_s_leq_w_25359 && zzero_lte_i_25360;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_25363 = zzero_leq_i_p_m_t_s_25358 && y_25362;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool forwards_ok_25364 = i_lte_j_25361 && y_25363;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool ok_or_empty_25365 = empty_slice_25355 || forwards_ok_25364;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool index_certs_25366;
            
            if (!ok_or_empty_25365) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_25347, ":", (long long) seen_final_25353, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23532, "].", "-> #0  src/bgpc_vv_coloring.fut:46:5-54:19\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:77:24-51
            for (int64_t nest_i_26771 = 0; nest_i_26771 < color_bound_23576; nest_i_26771++) {
                ((bool *) mem_26591)[nest_i_26771] = 0;
            }
            // src/bgpc_vv_coloring.fut:46:5-54:19
            for (int64_t i_25368 = 0; i_25368 < j_m_i_25354; i_25368++) {
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t index_primexp_25370 = seen_final_25347 + i_25368;
                
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t v_25371 = ((int64_t *) ext_mem_26568.mem)[index_primexp_25370];
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool x_25372 = sle64((int64_t) 0, v_25371);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool y_25373 = slt64(v_25371, csr_bipartite_from_pattern_res_23526);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool bounds_check_25374 = x_25372 && y_25373;
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool index_certs_25375;
                
                if (!bounds_check_25374) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_25371, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23526, "].", "-> #0  src/bgpc_vv_coloring.fut:49:26-37\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:26-37
                
                int64_t seen_acczq_25376 = ((int64_t *) ext_mem_26565.mem)[v_25371];
                
                // src/bgpc_vv_coloring.fut:49:50-52
                
                int64_t seen_acczq_25377 = add64((int64_t) 1, v_25371);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool x_25378 = sle64((int64_t) 0, seen_acczq_25377);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool y_25379 = slt64(seen_acczq_25377, csr_bipartite_from_pattern_res_23526);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool bounds_check_25380 = x_25378 && y_25379;
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool index_certs_25381;
                
                if (!bounds_check_25380) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_25377, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23526, "].", "-> #0  src/bgpc_vv_coloring.fut:49:40-53\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:40-53
                
                int64_t seen_acczq_25382 = ((int64_t *) ext_mem_26565.mem)[seen_acczq_25377];
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t j_m_i_25383 = sub64(seen_acczq_25382, seen_acczq_25376);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool empty_slice_25384 = j_m_i_25383 == (int64_t) 0;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t m_25385 = sub64(j_m_i_25383, (int64_t) 1);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t i_p_m_t_s_25386 = add64(seen_acczq_25376, m_25385);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_leq_i_p_m_t_s_25387 = sle64((int64_t) 0, i_p_m_t_s_25386);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_p_m_t_s_leq_w_25388 = slt64(i_p_m_t_s_25386, csr_bipartite_from_pattern_res_23527);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_lte_i_25389 = sle64((int64_t) 0, seen_acczq_25376);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_lte_j_25390 = sle64(seen_acczq_25376, seen_acczq_25382);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_25391 = i_p_m_t_s_leq_w_25388 && zzero_lte_i_25389;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_25392 = zzero_leq_i_p_m_t_s_25387 && y_25391;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool forwards_ok_25393 = i_lte_j_25390 && y_25392;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool ok_or_empty_25394 = empty_slice_25384 || forwards_ok_25393;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool index_certs_25395;
                
                if (!ok_or_empty_25394) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_25376, ":", (long long) seen_acczq_25382, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23527, "].", "-> #0  src/bgpc_vv_coloring.fut:48:9-53:33\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:48:9-53:33
                for (int64_t i_25397 = 0; i_25397 < j_m_i_25383; i_25397++) {
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t index_primexp_25399 = seen_acczq_25376 + i_25397;
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t u_25400 = ((int64_t *) ext_mem_26564.mem)[index_primexp_25399];
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool x_25401 = sle64((int64_t) 0, u_25400);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool y_25402 = slt64(u_25400, (int64_t) 6);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool bounds_check_25403 = x_25401 && y_25402;
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool index_certs_25404;
                    
                    if (!bounds_check_25403) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_25400, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/bgpc_vv_coloring.fut:50:20-29\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    int64_t cu_25405 = ((int64_t *) mem_26570)[u_25400];
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_25406 = u_25400 == eta_p_25341;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_25407 = !cond_25406;
                    
                    // src/bgpc_vv_coloring.fut:51:30-37
                    
                    bool cond_t_res_25408 = sle64((int64_t) 0, cu_25405);
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    bool x_25409 = cond_25407 && cond_t_res_25408;
                    
                    // src/bgpc_vv_coloring.fut:51:44-67
                    
                    bool cond_t_res_25410 = slt64(cu_25405, color_bound_23576);
                    
                    // src/bgpc_vv_coloring.fut:94:14-103:69
                    
                    bool x_25411 = x_25409 && cond_t_res_25410;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    if (x_25411) {
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool bounds_check_26237 = cond_t_res_25408 && cond_t_res_25410;
                        
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool index_certs_26238;
                        
                        if (!bounds_check_26237) {
                            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) cu_25405, "] out of bounds for array of shape [", (long long) color_bound_23576, "].", "-> #0  src/bgpc_vv_coloring.fut:52:19-50\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                            err = FUTHARK_PROGRAM_ERROR;
                            goto cleanup;
                        }
                        // src/bgpc_vv_coloring.fut:52:19-50
                        ((bool *) mem_26591)[cu_25405] = 1;
                    }
                }
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool loop_cond_25416;
            
            if (cond_23577) {
                // src/bgpc_vv_coloring.fut:63:30-37
                
                bool loop_cond_t_res_26240 = ((bool *) mem_26591)[(int64_t) 0];
                
                loop_cond_25416 = loop_cond_t_res_26240;
            } else {
                loop_cond_25416 = 0;
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool c_final_25418;
            int64_t c_final_25419;
            bool loop_while_25420;
            int64_t c_25421;
            
            loop_while_25420 = loop_cond_25416;
            c_25421 = (int64_t) 0;
            while (loop_while_25420) {
                // src/bgpc_vv_coloring.fut:64:9-15
                
                int64_t loopres_25422 = add64((int64_t) 1, c_25421);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool cond_25423 = slt64(loopres_25422, color_bound_23576);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool loop_cond_25424;
                
                if (cond_25423) {
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool x_26241 = sle64((int64_t) 0, loopres_25422);
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool bounds_check_26242 = cond_25423 && x_26241;
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool index_certs_26243;
                    
                    if (!bounds_check_26242) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) loopres_25422, "] out of bounds for array of shape [", (long long) color_bound_23576, "].", "-> #0  src/bgpc_vv_coloring.fut:63:30-37\n   #1  src/bgpc_vv_coloring.fut:81:6-28\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool loop_cond_t_res_26244 = ((bool *) mem_26591)[loopres_25422];
                    
                    loop_cond_25424 = loop_cond_t_res_26244;
                } else {
                    loop_cond_25424 = 0;
                }
                
                bool loop_while_tmp_26774 = loop_cond_25424;
                int64_t c_tmp_26775 = loopres_25422;
                
                loop_while_25420 = loop_while_tmp_26774;
                c_25421 = c_tmp_26775;
            }
            c_final_25418 = loop_while_25420;
            c_final_25419 = c_25421;
            // src/bgpc_vv_coloring.fut:106:25-32
            
            int64_t max_res_23671 = smax64(c_final_25419, redout_26393);
            
            ((int64_t *) mem_26587)[i_26395] = c_final_25419;
            
            int64_t redout_tmp_26769 = max_res_23671;
            
            redout_26393 = redout_tmp_26769;
        }
        defunc_0_reduce_res_26273 = redout_26393;
        // src/bgpc_vv_coloring.fut:113:17-43
        
        bool acc_cert_23677;
        
        // src/bgpc_vv_coloring.fut:113:17-43
        for (int64_t i_26398 = 0; i_26398 < loop_dz2081Uz2083U_23572; i_26398++) {
            int64_t v_23681 = ((int64_t *) mem_param_26585.mem)[i_26398];
            int64_t v_23682 = ((int64_t *) mem_26587)[i_26398];
            
            // src/bgpc_vv_coloring.fut:113:17-43
            // UpdateAcc
            if (sle64((int64_t) 0, v_23681) && slt64(v_23681, (int64_t) 6)) {
                ((int64_t *) mem_26570)[v_23681] = v_23682;
            }
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26608_cached_sizze_26903 < bytes_26586) {
            err = lexical_realloc(ctx, &mem_26608, &mem_26608_cached_sizze_26903, bytes_26586);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26610_cached_sizze_26904 < bytes_26586) {
            err = lexical_realloc(ctx, &mem_26610, &mem_26610_cached_sizze_26904, bytes_26586);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26611_cached_sizze_26905 < loop_dz2081Uz2083U_23572) {
            err = lexical_realloc(ctx, &mem_26611, &mem_26611_cached_sizze_26905, loop_dz2081Uz2083U_23572);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t discard_26410;
        int64_t scanacc_26402 = (int64_t) 0;
        
        for (int64_t i_26406 = 0; i_26406 < loop_dz2081Uz2083U_23572; i_26406++) {
            int64_t eta_p_25265 = ((int64_t *) mem_param_26585.mem)[i_26406];
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool x_25266 = sle64((int64_t) 0, eta_p_25265);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool y_25267 = slt64(eta_p_25265, csr_cols_from_pattern_res_23531);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool bounds_check_25268 = x_25266 && y_25267;
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool index_certs_25269;
            
            if (!bounds_check_25268) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25265, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23531, "].", "-> #0  src/bgpc_vv_coloring.fut:126:15-27\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:127:26-28
            
            int64_t k_end_25271 = add64((int64_t) 1, eta_p_25265);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool x_25272 = sle64((int64_t) 0, k_end_25271);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool y_25273 = slt64(k_end_25271, csr_cols_from_pattern_res_23531);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool bounds_check_25274 = x_25272 && y_25273;
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool index_certs_25275;
            
            if (!bounds_check_25274) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_end_25271, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23531, "].", "-> #0  src/bgpc_vv_coloring.fut:127:15-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:126:15-27
            
            int64_t k0_25270 = ((int64_t *) ext_mem_26569.mem)[eta_p_25265];
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            int64_t k_end_25276 = ((int64_t *) ext_mem_26569.mem)[k_end_25271];
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool cond_25277 = slt64(k0_25270, k_end_25276);
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool loses_conflict_vertex_res_25278;
            bool loses_conflict_vertex_res_25279;
            int64_t loses_conflict_vertex_res_25280;
            bool loop_while_25281;
            bool lost_25282;
            int64_t k_25283;
            
            loop_while_25281 = cond_25277;
            lost_25282 = 0;
            k_25283 = k0_25270;
            while (loop_while_25281) {
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool x_25284 = sle64((int64_t) 0, k_25283);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool y_25285 = slt64(k_25283, csr_cols_from_pattern_res_23532);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool bounds_check_25286 = x_25284 && y_25285;
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool index_certs_25287;
                
                if (!bounds_check_25286) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_25283, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_23532, "].", "-> #0  src/bgpc_vv_coloring.fut:132:15-26\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:132:15-26
                
                int64_t v_25288 = ((int64_t *) ext_mem_26568.mem)[k_25283];
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool x_25289 = sle64((int64_t) 0, v_25288);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool y_25290 = slt64(v_25288, csr_bipartite_from_pattern_res_23526);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool bounds_check_25291 = x_25289 && y_25290;
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool index_certs_25292;
                
                if (!bounds_check_25291) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_25288, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23526, "].", "-> #0  src/bgpc_vv_coloring.fut:134:19-30\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:135:29-31
                
                int64_t t_end_25294 = add64((int64_t) 1, v_25288);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool x_25295 = sle64((int64_t) 0, t_end_25294);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool y_25296 = slt64(t_end_25294, csr_bipartite_from_pattern_res_23526);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool bounds_check_25297 = x_25295 && y_25296;
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool index_certs_25298;
                
                if (!bounds_check_25297) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_end_25294, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23526, "].", "-> #0  src/bgpc_vv_coloring.fut:135:19-32\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:134:19-30
                
                int64_t t0_25293 = ((int64_t *) ext_mem_26565.mem)[v_25288];
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                int64_t t_end_25299 = ((int64_t *) ext_mem_26565.mem)[t_end_25294];
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool cond_25300 = slt64(t0_25293, t_end_25299);
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool loopres_25301;
                bool loopres_25302;
                int64_t loopres_25303;
                bool loop_while_25304;
                bool lost_in_net_25305;
                int64_t t_25306;
                
                loop_while_25304 = cond_25300;
                lost_in_net_25305 = 0;
                t_25306 = t0_25293;
                while (loop_while_25304) {
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool x_25307 = sle64((int64_t) 0, t_25306);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool y_25308 = slt64(t_25306, csr_bipartite_from_pattern_res_23527);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool bounds_check_25309 = x_25307 && y_25308;
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool index_certs_25310;
                    
                    if (!bounds_check_25309) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_25306, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23527, "].", "-> #0  src/bgpc_vv_coloring.fut:140:19-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    int64_t u_25311 = ((int64_t *) ext_mem_26564.mem)[t_25306];
                    
                    // src/bgpc_vv_coloring.fut:142:13-71
                    
                    bool lost_in_netzq_25312;
                    
                    if (lost_in_net_25305) {
                        lost_in_netzq_25312 = 1;
                    } else {
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25313 = u_25311 == eta_p_25265;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25314 = !cond_25313;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25315;
                        
                        if (cond_25314) {
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool x_26246 = sle64((int64_t) 0, u_25311);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool y_26247 = slt64(u_25311, (int64_t) 6);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool bounds_check_26248 = x_26246 && y_26247;
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool index_certs_26249;
                            
                            if (!bounds_check_26248) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_25311, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/bgpc_vv_coloring.fut:142:39-48\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool y_26251 = slt64(eta_p_25265, (int64_t) 6);
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool bounds_check_26252 = x_25266 && y_26251;
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool index_certs_26253;
                            
                            if (!bounds_check_26252) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25265, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/bgpc_vv_coloring.fut:142:52-61\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            int64_t zeze_lhs_26250 = ((int64_t *) mem_26570)[u_25311];
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            int64_t zeze_rhs_26254 = ((int64_t *) mem_26570)[eta_p_25265];
                            
                            // src/bgpc_vv_coloring.fut:142:49-61
                            
                            bool cond_t_res_26255 = zeze_lhs_26250 == zeze_rhs_26254;
                            
                            cond_25315 = cond_t_res_26255;
                        } else {
                            cond_25315 = 0;
                        }
                        // src/bgpc_vv_coloring.fut:142:67-70
                        
                        bool lost_in_netzq_f_res_t_res_25326 = slt64(u_25311, eta_p_25265);
                        
                        // src/bgpc_vv_coloring.fut:160:14-168:57
                        
                        bool x_25327 = cond_25315 && lost_in_netzq_f_res_t_res_25326;
                        
                        lost_in_netzq_25312 = x_25327;
                    }
                    // src/bgpc_vv_coloring.fut:143:31-37
                    
                    int64_t tmp_25328 = add64((int64_t) 1, t_25306);
                    
                    // src/bgpc_vv_coloring.fut:139:15-43
                    
                    bool cond_25329 = slt64(tmp_25328, t_end_25299);
                    
                    // src/bgpc_vv_coloring.fut:139:28-43
                    
                    bool not_res_25330 = !lost_in_netzq_25312;
                    
                    // src/bgpc_vv_coloring.fut:166:5-169:10
                    
                    bool x_25331 = cond_25329 && not_res_25330;
                    bool loop_while_tmp_26784 = x_25331;
                    bool lost_in_net_tmp_26785 = lost_in_netzq_25312;
                    int64_t t_tmp_26786 = tmp_25328;
                    
                    loop_while_25304 = loop_while_tmp_26784;
                    lost_in_net_25305 = lost_in_net_tmp_26785;
                    t_25306 = t_tmp_26786;
                }
                loopres_25301 = loop_while_25304;
                loopres_25302 = lost_in_net_25305;
                loopres_25303 = t_25306;
                // src/bgpc_vv_coloring.fut:160:14-168:57
                
                bool lostzq_25332 = lost_25282 || loopres_25302;
                
                // src/bgpc_vv_coloring.fut:146:20-26
                
                int64_t tmp_25333 = add64((int64_t) 1, k_25283);
                
                // src/bgpc_vv_coloring.fut:131:11-32
                
                bool cond_25334 = slt64(tmp_25333, k_end_25276);
                
                // src/bgpc_vv_coloring.fut:131:24-32
                
                bool not_res_25335 = !lostzq_25332;
                
                // src/bgpc_vv_coloring.fut:166:5-169:10
                
                bool x_25336 = cond_25334 && not_res_25335;
                bool loop_while_tmp_26781 = x_25336;
                bool lost_tmp_26782 = lostzq_25332;
                int64_t k_tmp_26783 = tmp_25333;
                
                loop_while_25281 = loop_while_tmp_26781;
                lost_25282 = lost_tmp_26782;
                k_25283 = k_tmp_26783;
            }
            loses_conflict_vertex_res_25278 = loop_while_25281;
            loses_conflict_vertex_res_25279 = lost_25282;
            loses_conflict_vertex_res_25280 = k_25283;
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_f_res_25338 = btoi_bool_i64(loses_conflict_vertex_res_25279);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_op_res_23783 = add64(defunc_0_f_res_25338, scanacc_26402);
            
            ((int64_t *) mem_26608)[i_26406] = defunc_0_op_res_23783;
            ((int64_t *) mem_26610)[i_26406] = defunc_0_f_res_25338;
            ((bool *) mem_26611)[i_26406] = loses_conflict_vertex_res_25279;
            
            int64_t scanacc_tmp_26777 = defunc_0_op_res_23783;
            
            scanacc_26402 = scanacc_tmp_26777;
        }
        discard_26410 = scanacc_26402;
        // src/bgpc_vv_coloring.fut:23:25-28
        
        int64_t tmp_23767 = sub64(loop_dz2081Uz2083U_23572, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool y_23769 = slt64(tmp_23767, loop_dz2081Uz2083U_23572);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool x_23768 = sle64((int64_t) 0, tmp_23767);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool bounds_check_23770 = x_23768 && y_23769;
        
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        bool cond_23765 = loop_dz2081Uz2083U_23572 == (int64_t) 0;
        
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool protect_assert_disj_23771 = cond_23765 || bounds_check_23770;
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool index_certs_23772;
        
        if (!protect_assert_disj_23771) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_23767, "] out of bounds for array of shape [", (long long) loop_dz2081Uz2083U_23572, "].", "-> #0  src/bgpc_vv_coloring.fut:23:10-29\n   #1  src/bgpc_vv_coloring.fut:171:23-55\n   #2  src/bgpc_vv_coloring.fut:186:14-205:56\n   #3  src/bgpc_vv_coloring.fut:213:14-216:64\n   #4  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #5  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #6  test/test_sparse_jacobian_jvp.fut:166:18-44\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool x_23766 = !cond_23765;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_f_res_23784;
        
        if (x_23766) {
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t x_26256 = ((int64_t *) mem_26608)[tmp_23767];
            
            m_f_res_23784 = x_26256;
        } else {
            m_f_res_23784 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_23786;
        
        if (cond_23765) {
            m_23786 = (int64_t) 0;
        } else {
            m_23786 = m_f_res_23784;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t bytes_26630 = (int64_t) 8 * m_23786;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26631_cached_sizze_26906 < bytes_26630) {
            err = lexical_realloc(ctx, &mem_26631, &mem_26631_cached_sizze_26906, bytes_26630);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26633_cached_sizze_26907 < bytes_26630) {
            err = lexical_realloc(ctx, &mem_26633, &mem_26633_cached_sizze_26907, bytes_26630);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:18:24-27:44
        if (mem_26635_cached_sizze_26908 < bytes_26586) {
            err = lexical_realloc(ctx, &mem_26635, &mem_26635_cached_sizze_26908, bytes_26586);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26643_cached_sizze_26909 < bytes_26586) {
            err = lexical_realloc(ctx, &mem_26643, &mem_26643_cached_sizze_26909, bytes_26586);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        bool acc_cert_25138;
        bool acc_cert_25139;
        
        // src/bgpc_vv_coloring.fut:18:24-27:44
        
        int64_t inpacc_26264;
        int64_t inpacc_25202 = (int64_t) 0;
        
        for (int64_t i_26437 = 0; i_26437 < loop_dz2081Uz2083U_23572; i_26437++) {
            bool eta_p_26514 = ((bool *) mem_26611)[i_26437];
            
            // src/bgpc_vv_coloring.fut:18:28-39
            
            int64_t bool_to_i64_res_26515 = btoi_bool_i64(eta_p_26514);
            int64_t eta_p_26527 = ((int64_t *) mem_26610)[i_26437];
            int64_t eta_p_26528 = ((int64_t *) mem_26608)[i_26437];
            int64_t v_26531 = ((int64_t *) mem_param_26585.mem)[i_26437];
            
            // src/bgpc_vv_coloring.fut:19:29-32
            
            int64_t defunc_0_op_res_26532 = add64(inpacc_25202, bool_to_i64_res_26515);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            bool cond_26533 = eta_p_26527 == (int64_t) 1;
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t lifted_lambda_res_26534;
            
            if (cond_26533) {
                // src/bgpc_vv_coloring.fut:27:5-44
                
                int64_t lifted_lambda_res_t_res_26535 = sub64(eta_p_26528, (int64_t) 1);
                
                lifted_lambda_res_26534 = lifted_lambda_res_t_res_26535;
            } else {
                lifted_lambda_res_26534 = (int64_t) -1;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_26534) && slt64(lifted_lambda_res_26534, m_23786)) {
                ((int64_t *) mem_26633)[lifted_lambda_res_26534] = v_26531;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_26534) && slt64(lifted_lambda_res_26534, m_23786)) {
                ((int64_t *) mem_26631)[lifted_lambda_res_26534] = defunc_0_op_res_26532;
            }
            ((int64_t *) mem_26635)[i_26437] = defunc_0_op_res_26532;
            
            int64_t inpacc_tmp_26787 = defunc_0_op_res_26532;
            
            inpacc_25202 = inpacc_tmp_26787;
        }
        inpacc_26264 = inpacc_25202;
        // src/bgpc_vv_coloring.fut:27:5-44
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_26643, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_26635, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {loop_dz2081Uz2083U_23572});
        // src/bgpc_vv_coloring.fut:23:10-29
        
        int64_t out_szz_f_res_23773;
        
        if (x_23766) {
            // src/bgpc_vv_coloring.fut:171:23-55
            
            int64_t x_26270 = ((int64_t *) mem_26643)[tmp_23767];
            
            out_szz_f_res_23773 = x_26270;
        } else {
            out_szz_f_res_23773 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        int64_t out_szz_23775;
        
        if (cond_23765) {
            out_szz_23775 = (int64_t) 0;
        } else {
            out_szz_23775 = out_szz_f_res_23773;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        
        int64_t bytes_26644 = (int64_t) 8 * out_szz_23775;
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        bool cond_23672 = slt64(defunc_0_reduce_res_26273, (int64_t) 0);
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        int64_t next_color_bound_23673;
        
        if (cond_23672) {
            next_color_bound_23673 = color_bound_23576;
        } else {
            // src/bgpc_vv_coloring.fut:110:40-46
            
            int64_t max_arg1_23674 = add64((int64_t) 2, defunc_0_reduce_res_26273);
            
            // src/bgpc_vv_coloring.fut:110:10-46
            
            int64_t max_res_23675 = smax64(color_bound_23576, max_arg1_23674);
            
            next_color_bound_23673 = max_res_23675;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        if (memblock_alloc(ctx, &mem_26645, bytes_26644, "mem_26645")) {
            err = 1;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        for (int64_t nest_i_26791 = 0; nest_i_26791 < out_szz_23775; nest_i_26791++) {
            ((int64_t *) mem_26645.mem)[nest_i_26791] = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:33:6-27
        
        bool acc_cert_24911;
        
        // src/bgpc_vv_coloring.fut:29:21-33:27
        for (int64_t i_26442 = 0; i_26442 < m_23786; i_26442++) {
            int64_t eta_p_24923 = ((int64_t *) mem_26631)[i_26442];
            int64_t v_24925 = ((int64_t *) mem_26633)[i_26442];
            
            // src/bgpc_vv_coloring.fut:29:44-50
            
            int64_t lifted_lambda_res_24926 = sub64(eta_p_24923, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:33:6-27
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_24926) && slt64(lifted_lambda_res_24926, out_szz_23775)) {
                ((int64_t *) mem_26645.mem)[lifted_lambda_res_24926] = v_24925;
            }
        }
        // src/bgpc_vv_coloring.fut:175:17-48
        
        bool acc_cert_23823;
        
        // src/bgpc_vv_coloring.fut:175:17-48
        for (int64_t i_26444 = 0; i_26444 < out_szz_23775; i_26444++) {
            int64_t v_23827 = ((int64_t *) mem_26645.mem)[i_26444];
            
            // src/bgpc_vv_coloring.fut:175:17-48
            // UpdateAcc
            if (sle64((int64_t) 0, v_23827) && slt64(v_23827, (int64_t) 6)) {
                ((int64_t *) mem_26570)[v_23827] = (int64_t) -1;
            }
        }
        // src/bgpc_vv_coloring.fut:198:20-23
        
        bool loop_cond_23829 = slt64((int64_t) 0, out_szz_23775);
        
        if (memblock_set(ctx, &mem_param_tmp_26763, &mem_26645, "mem_26645") != 0)
            return 1;
        
        int64_t loop_dz2081Uz2083U_tmp_26764 = out_szz_23775;
        bool loop_while_tmp_26765 = loop_cond_23829;
        int64_t color_bound_tmp_26768 = next_color_bound_23673;
        
        if (memblock_set(ctx, &mem_param_26585, &mem_param_tmp_26763, "mem_param_tmp_26763") != 0)
            return 1;
        loop_dz2081Uz2083U_23572 = loop_dz2081Uz2083U_tmp_26764;
        loop_while_23573 = loop_while_tmp_26765;
        color_bound_23576 = color_bound_tmp_26768;
    }
    if (memblock_set(ctx, &ext_mem_26650, &mem_param_26585, "mem_param_26585") != 0)
        return 1;
    vv_color_side_order_res_23567 = loop_dz2081Uz2083U_23572;
    vv_color_side_order_res_23568 = loop_while_23573;
    vv_color_side_order_res_23571 = color_bound_23576;
    if (memblock_unref(ctx, &ext_mem_26568, "ext_mem_26568") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26569, "ext_mem_26569") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_26579, "mem_26579") != 0)
        return 1;
    // src/sparse_jacobian_jvp.fut:42:15-35
    
    int64_t bytes_26663 = (int64_t) 8 * csr_bipartite_from_pattern_res_23527;
    
    // src/sparse_jacobian_jvp.fut:107:8-112:40
    
    int64_t x_26378;
    int64_t x_26379;
    int64_t redout_26449;
    int64_t redout_26450;
    
    redout_26449 = (int64_t) 0;
    redout_26450 = (int64_t) 0;
    for (int64_t i_26451 = 0; i_26451 < (int64_t) 6; i_26451++) {
        int64_t x_24904 = ((int64_t *) mem_26570)[i_26451];
        
        // src/sparse_jacobian_jvp.fut:14:22-29
        
        int64_t max_res_23865 = smax64(x_24904, redout_26449);
        
        // src/sparse_jacobian_jvp.fut:14:22-29
        
        int64_t max_res_23974 = smax64(x_24904, redout_26450);
        int64_t redout_tmp_26794 = max_res_23865;
        int64_t redout_tmp_26795 = max_res_23974;
        
        redout_26449 = redout_tmp_26794;
        redout_26450 = redout_tmp_26795;
    }
    x_26378 = redout_26449;
    x_26379 = redout_26450;
    // src/sparse_jacobian_jvp.fut:14:13-41
    
    int64_t num_colors_of_res_f_res_23866 = add64((int64_t) 1, x_26378);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool bounds_invalid_upwards_23868 = slt64(num_colors_of_res_f_res_23866, (int64_t) 0);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool valid_23869 = !bounds_invalid_upwards_23868;
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool range_valid_c_23870;
    
    if (!valid_23869) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Range ", (long long) (int64_t) 0, "..", (long long) (int64_t) 1, "..<", (long long) num_colors_of_res_f_res_23866, " is invalid.", "-> #0  src/sparse_jacobian_jvp.fut:29:11-18\n   #1  src/sparse_jacobian_jvp.fut:107:8-112:40\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-169:61\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    
    int64_t bytes_26666 = (int64_t) 32 * num_colors_of_res_f_res_23866;
    
    // src/sparse_jacobian_jvp.fut:14:13-41
    
    int64_t num_colors_of_res_f_res_23975 = add64((int64_t) 1, x_26379);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool bounds_invalid_upwards_23977 = slt64(num_colors_of_res_f_res_23975, (int64_t) 0);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool valid_23978 = !bounds_invalid_upwards_23977;
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool range_valid_c_23979;
    
    if (!valid_23978) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Range ", (long long) (int64_t) 0, "..", (long long) (int64_t) 1, "..<", (long long) num_colors_of_res_f_res_23975, " is invalid.", "-> #0  src/sparse_jacobian_jvp.fut:29:11-18\n   #1  src/sparse_jacobian_jvp.fut:107:8-112:40\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-172:61\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    
    int64_t bytes_26702 = (int64_t) 32 * num_colors_of_res_f_res_23975;
    double zt_lhs_24490 = ((double *) x1_mem_26563.mem)[(int64_t) 0];
    double zt_rhs_24491 = ((double *) x1_mem_26563.mem)[(int64_t) 1];
    double zt_rhs_24493 = ((double *) x1_mem_26563.mem)[(int64_t) 4];
    
    // test/test_sparse_jacobian_jvp.fut:82:33-39
    
    double binop_x_25872 = 0.0 * zt_rhs_24493;
    double zt_lhs_24497 = ((double *) x1_mem_26563.mem)[(int64_t) 5];
    
    // src/dense_jacobian.fut:8:40-9:68
    if (mem_26652_cached_sizze_26910 < (int64_t) 192) {
        err = lexical_realloc(ctx, &mem_26652, &mem_26652_cached_sizze_26910, (int64_t) 192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:5:3-21
    if (mem_26657_cached_sizze_26911 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26657, &mem_26657_cached_sizze_26911, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:8:40-9:68
    for (int64_t i_26447 = 0; i_26447 < (int64_t) 6; i_26447++) {
        // src/dense_jacobian.fut:5:3-21
        for (int64_t nest_i_26797 = 0; nest_i_26797 < (int64_t) 6; nest_i_26797++) {
            ((double *) mem_26657)[nest_i_26797] = 0.0;
        }
        // src/dense_jacobian.fut:5:3-39
        ((double *) mem_26657)[i_26447] = 1.0;
        
        double zt_lhs_tan_25865 = ((double *) mem_26657)[(int64_t) 0];
        double zt_rhs_tan_25866 = ((double *) mem_26657)[(int64_t) 1];
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_x_25868 = zt_rhs_24491 * zt_lhs_tan_25865;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_y_25869 = zt_lhs_24490 * zt_rhs_tan_25866;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double zp_lhs_tan_25867 = binop_x_25868 + binop_y_25869;
        double zt_rhs_tan_25870 = ((double *) mem_26657)[(int64_t) 4];
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double binop_y_25873 = 3.0 * zt_rhs_tan_25870;
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double zp_rhs_tan_25871 = binop_x_25872 + binop_y_25873;
        
        // test/test_sparse_jacobian_jvp.fut:82:24-39
        
        double y0_tan_25874 = zp_lhs_tan_25867 + zp_rhs_tan_25871;
        double zm_lhs_tan_25877 = ((double *) mem_26657)[(int64_t) 2];
        double zt_lhs_tan_25878 = ((double *) mem_26657)[(int64_t) 5];
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double binop_x_25880 = zt_lhs_24497 * zt_lhs_tan_25878;
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double zm_rhs_tan_25879 = binop_x_25880 + binop_x_25880;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double binop_y_25884 = -1.0 * zm_rhs_tan_25879;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double y1_tan_25882 = zm_lhs_tan_25877 + binop_y_25884;
        
        // test/test_sparse_jacobian_jvp.fut:84:17-23
        
        double zp_lhs_tan_25885 = zt_rhs_tan_25866 + zm_lhs_tan_25877;
        double zp_rhs_tan_25888 = ((double *) mem_26657)[(int64_t) 3];
        
        // test/test_sparse_jacobian_jvp.fut:84:24-30
        
        double y2_tan_25889 = zp_lhs_tan_25885 + zp_rhs_tan_25888;
        
        // test/test_sparse_jacobian_jvp.fut:168:60-65
        ((double *) mem_26652)[i_26447 * (int64_t) 4] = y0_tan_25874;
        ((double *) mem_26652)[i_26447 * (int64_t) 4 + (int64_t) 1] = y1_tan_25882;
        ((double *) mem_26652)[i_26447 * (int64_t) 4 + (int64_t) 2] = y2_tan_25889;
        ((double *) mem_26652)[i_26447 * (int64_t) 4 + (int64_t) 3] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    if (mem_26664_cached_sizze_26912 < bytes_26663) {
        err = lexical_realloc(ctx, &mem_26664, &mem_26664_cached_sizze_26912, bytes_26663);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    for (int64_t nest_i_26798 = 0; nest_i_26798 < csr_bipartite_from_pattern_res_23527; nest_i_26798++) {
        ((double *) mem_26664)[nest_i_26798] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    if (mem_26667_cached_sizze_26913 < bytes_26666) {
        err = lexical_realloc(ctx, &mem_26667, &mem_26667_cached_sizze_26913, bytes_26666);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:17:3-61
    if (mem_26672_cached_sizze_26914 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26672, &mem_26672_cached_sizze_26914, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    for (int64_t i_26458 = 0; i_26458 < num_colors_of_res_f_res_23866; i_26458++) {
        // src/sparse_jacobian_jvp.fut:17:3-61
        for (int64_t i_26454 = 0; i_26454 < (int64_t) 6; i_26454++) {
            int64_t eta_p_23879 = ((int64_t *) mem_26570)[i_26454];
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            bool cond_23880 = eta_p_23879 == i_26458;
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            double lifted_lambda_res_23881;
            
            if (cond_23880) {
                lifted_lambda_res_23881 = 1.0;
            } else {
                lifted_lambda_res_23881 = 0.0;
            }
            ((double *) mem_26672)[i_26454] = lifted_lambda_res_23881;
        }
        
        double zt_lhs_tan_25894 = ((double *) mem_26672)[(int64_t) 0];
        double zt_rhs_tan_25895 = ((double *) mem_26672)[(int64_t) 1];
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_x_25897 = zt_rhs_24491 * zt_lhs_tan_25894;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_y_25898 = zt_lhs_24490 * zt_rhs_tan_25895;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double zp_lhs_tan_25896 = binop_x_25897 + binop_y_25898;
        double zt_rhs_tan_25899 = ((double *) mem_26672)[(int64_t) 4];
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double binop_y_25902 = 3.0 * zt_rhs_tan_25899;
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double zp_rhs_tan_25900 = binop_x_25872 + binop_y_25902;
        
        // test/test_sparse_jacobian_jvp.fut:82:24-39
        
        double y0_tan_25903 = zp_lhs_tan_25896 + zp_rhs_tan_25900;
        double zm_lhs_tan_25906 = ((double *) mem_26672)[(int64_t) 2];
        double zt_lhs_tan_25907 = ((double *) mem_26672)[(int64_t) 5];
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double binop_x_25909 = zt_lhs_24497 * zt_lhs_tan_25907;
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double zm_rhs_tan_25908 = binop_x_25909 + binop_x_25909;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double binop_y_25913 = -1.0 * zm_rhs_tan_25908;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double y1_tan_25911 = zm_lhs_tan_25906 + binop_y_25913;
        
        // test/test_sparse_jacobian_jvp.fut:84:17-23
        
        double zp_lhs_tan_25914 = zt_rhs_tan_25895 + zm_lhs_tan_25906;
        double zp_rhs_tan_25917 = ((double *) mem_26672)[(int64_t) 3];
        
        // test/test_sparse_jacobian_jvp.fut:84:24-30
        
        double y2_tan_25918 = zp_lhs_tan_25914 + zp_rhs_tan_25917;
        
        // test/test_sparse_jacobian_jvp.fut:169:44-49
        ((double *) mem_26667)[i_26458 * (int64_t) 4] = y0_tan_25903;
        ((double *) mem_26667)[i_26458 * (int64_t) 4 + (int64_t) 1] = y1_tan_25911;
        ((double *) mem_26667)[i_26458 * (int64_t) 4 + (int64_t) 2] = y2_tan_25918;
        ((double *) mem_26667)[i_26458 * (int64_t) 4 + (int64_t) 3] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:45:5-52:27
    
    bool compressed_to_csr_vals_res_23890;
    int64_t compressed_to_csr_vals_res_23892;
    bool loop_while_23893;
    int64_t i_23895;
    
    loop_while_23893 = 1;
    i_23895 = (int64_t) 0;
    while (loop_while_23893) {
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool x_23896 = sle64((int64_t) 0, i_23895);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool y_23897 = slt64(i_23895, csr_bipartite_from_pattern_res_23526);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool bounds_check_23898 = x_23896 && y_23897;
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool index_certs_23899;
        
        if (!bounds_check_23898) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_23895, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23526, "].", "-> #0  src/sparse_jacobian_jvp.fut:47:15-26\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-169:61\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        int64_t s_23900 = ((int64_t *) ext_mem_26565.mem)[i_23895];
        
        // src/sparse_jacobian_jvp.fut:48:25-27
        
        int64_t e_23901 = add64((int64_t) 1, i_23895);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool x_23902 = sle64((int64_t) 0, e_23901);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool y_23903 = slt64(e_23901, csr_bipartite_from_pattern_res_23526);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool bounds_check_23904 = x_23902 && y_23903;
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool index_certs_23905;
        
        if (!bounds_check_23904) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_23901, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23526, "].", "-> #0  src/sparse_jacobian_jvp.fut:48:15-28\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-169:61\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        int64_t e_23906 = ((int64_t *) ext_mem_26565.mem)[e_23901];
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t j_m_i_23907 = sub64(e_23906, s_23900);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool empty_slice_23908 = j_m_i_23907 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t m_23909 = sub64(j_m_i_23907, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t i_p_m_t_s_23910 = add64(s_23900, m_23909);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_23911 = sle64((int64_t) 0, i_p_m_t_s_23910);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_23912 = slt64(i_p_m_t_s_23910, csr_bipartite_from_pattern_res_23527);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_lte_i_23913 = sle64((int64_t) 0, s_23900);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_lte_j_23914 = sle64(s_23900, e_23906);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_23915 = i_p_m_t_s_leq_w_23912 && zzero_lte_i_23913;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_23916 = zzero_leq_i_p_m_t_s_23911 && y_23915;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool forwards_ok_23917 = i_lte_j_23914 && y_23916;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool ok_or_empty_23918 = empty_slice_23908 || forwards_ok_23917;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool index_certs_23919;
        
        if (!ok_or_empty_23918) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_23900, ":", (long long) e_23906, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23527, "].", "-> #0  src/sparse_jacobian_jvp.fut:49:18-30\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-169:61\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool y_23921 = slt64(i_23895, (int64_t) 4);
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool bounds_check_23922 = x_23896 && y_23921;
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool index_certs_23923;
        
        if (!bounds_check_23922) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_23895, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-45\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-169:61\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:18-51
        
        int64_t bytes_26687 = (int64_t) 8 * j_m_i_23907;
        
        // src/sparse_jacobian_jvp.fut:50:18-51
        if (mem_26688_cached_sizze_26915 < bytes_26687) {
            err = lexical_realloc(ctx, &mem_26688, &mem_26688_cached_sizze_26915, bytes_26687);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:18-51
        for (int64_t i_26462 = 0; i_26462 < j_m_i_23907; i_26462++) {
            int64_t index_primexp_26505 = s_23900 + i_26462;
            int64_t eta_p_23925 = ((int64_t *) ext_mem_26564.mem)[index_primexp_26505];
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool x_23926 = sle64((int64_t) 0, eta_p_23925);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool y_23927 = slt64(eta_p_23925, (int64_t) 6);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool bounds_check_23928 = x_23926 && y_23927;
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool index_certs_23929;
            
            if (!bounds_check_23928) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_23925, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:32-41\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-169:61\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            int64_t tmp_23930 = ((int64_t *) mem_26570)[eta_p_23925];
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool x_23931 = sle64((int64_t) 0, tmp_23930);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool y_23932 = slt64(tmp_23930, num_colors_of_res_f_res_23866);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool bounds_check_23933 = x_23931 && y_23932;
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool index_certs_23934;
            
            if (!bounds_check_23933) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_23930, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_23866, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-42\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-169:61\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:18-51
            
            double lifted_lambda_res_23935 = ((double *) mem_26667)[tmp_23930 * (int64_t) 4 + i_23895];
            
            ((double *) mem_26688)[i_26462] = lifted_lambda_res_23935;
        }
        // src/sparse_jacobian_jvp.fut:51:19-40
        // src/sparse_jacobian_jvp.fut:51:19-40
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_26664, s_23900, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_26688, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {j_m_i_23907});
        // src/sparse_jacobian_jvp.fut:46:13-16
        
        bool loop_cond_23937 = slt64(e_23901, (int64_t) 4);
        bool loop_while_tmp_26801 = loop_cond_23937;
        int64_t i_tmp_26803 = e_23901;
        
        loop_while_23893 = loop_while_tmp_26801;
        i_23895 = i_tmp_26803;
    }
    compressed_to_csr_vals_res_23890 = loop_while_23893;
    compressed_to_csr_vals_res_23892 = i_23895;
    // src/sparse_jacobian_jvp.fut:67:29-47
    if (mem_26698_cached_sizze_26916 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26698, &mem_26698_cached_sizze_26916, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:8:19-10:39
    
    bool defunc_0_reduce_res_26381;
    bool redout_26468 = 1;
    
    for (int64_t i_26469 = 0; i_26469 < (int64_t) 4; i_26469++) {
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool y_24830 = slt64(i_26469, csr_bipartite_from_pattern_res_23526);
        
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool index_certs_24832;
        
        if (!y_24830) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_26469, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23526, "].", "-> #0  src/sparse_jacobian_jvp.fut:63:17-28\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  test/test_sparse_jacobian_jvp.fut:81:15-169:61\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        int64_t s_24833 = ((int64_t *) ext_mem_26565.mem)[i_26469];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_lte_i_24846 = sle64((int64_t) 0, s_24833);
        
        // src/sparse_jacobian_jvp.fut:64:27-29
        
        int64_t e_24834 = add64((int64_t) 1, i_26469);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool y_24836 = slt64(e_24834, csr_bipartite_from_pattern_res_23526);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool x_24835 = sle64((int64_t) 0, e_24834);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool bounds_check_24837 = x_24835 && y_24836;
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool index_certs_24838;
        
        if (!bounds_check_24837) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_24834, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23526, "].", "-> #0  src/sparse_jacobian_jvp.fut:64:17-30\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  test/test_sparse_jacobian_jvp.fut:81:15-169:61\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        int64_t e_24839 = ((int64_t *) ext_mem_26565.mem)[e_24834];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t j_m_i_24840 = sub64(e_24839, s_24833);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t m_24842 = sub64(j_m_i_24840, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t i_p_m_t_s_24843 = add64(s_24833, m_24842);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_p_m_t_s_leq_w_24845 = slt64(i_p_m_t_s_24843, csr_bipartite_from_pattern_res_23527);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24854 = i_p_m_t_s_leq_w_24845 && zzero_lte_i_24846;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_leq_i_p_m_t_s_24844 = sle64((int64_t) 0, i_p_m_t_s_24843);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24855 = zzero_leq_i_p_m_t_s_24844 && y_24854;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_lte_j_24847 = sle64(s_24833, e_24839);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool forwards_ok_24856 = i_lte_j_24847 && y_24855;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool empty_slice_24841 = j_m_i_24840 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool ok_or_empty_24857 = empty_slice_24841 || forwards_ok_24856;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool index_certs_24858;
        
        if (!ok_or_empty_24857) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_24833, ":", (long long) e_24839, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23527, "].", "-> #0  src/sparse_jacobian_jvp.fut:66:20-29\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  test/test_sparse_jacobian_jvp.fut:81:15-169:61\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:67:29-47
        for (int64_t nest_i_26806 = 0; nest_i_26806 < (int64_t) 6; nest_i_26806++) {
            ((double *) mem_26698)[nest_i_26806] = 0.0;
        }
        // src/sparse_jacobian_jvp.fut:68:12-32
        
        bool acc_cert_24862;
        
        // src/sparse_jacobian_jvp.fut:68:12-32
        for (int64_t i_26465 = 0; i_26465 < j_m_i_24840; i_26465++) {
            int64_t index_primexp_26502 = s_24833 + i_26465;
            int64_t v_24866 = ((int64_t *) ext_mem_26564.mem)[index_primexp_26502];
            double v_24867 = ((double *) mem_26664)[index_primexp_26502];
            
            // src/sparse_jacobian_jvp.fut:68:12-32
            // UpdateAcc
            if (sle64((int64_t) 0, v_24866) && slt64(v_24866, (int64_t) 6)) {
                ((double *) mem_26698)[v_24866] = v_24867;
            }
        }
        // test/test_sparse_jacobian_jvp.fut:9:5-66
        
        bool defunc_0_reduce_res_26282;
        bool redout_26466 = 1;
        
        for (int64_t i_26467 = 0; i_26467 < (int64_t) 6; i_26467++) {
            bool eta_p_25520 = ((bool *) mem_26555.mem)[i_26469 * (int64_t) 6 + i_26467];
            double eta_p_25521 = ((double *) mem_26652)[i_26467 * (int64_t) 4 + i_26469];
            double eta_p_25522 = ((double *) mem_26698)[i_26467];
            
            // test/test_sparse_jacobian_jvp.fut:14:25-48
            
            double lifted_lambda_res_25523;
            
            if (eta_p_25520) {
                lifted_lambda_res_25523 = eta_p_25521;
            } else {
                lifted_lambda_res_25523 = 0.0;
            }
            // test/test_sparse_jacobian_jvp.fut:9:48-51
            
            double abs_arg0_25525 = eta_p_25522 - lifted_lambda_res_25523;
            
            // test/test_sparse_jacobian_jvp.fut:9:37-51
            
            double abs_res_25526 = fabs64(abs_arg0_25525);
            
            // test/test_sparse_jacobian_jvp.fut:9:53-59
            
            bool lifted_lambda_res_25527 = abs_res_25526 <= 1.0e-9;
            
            // test/test_sparse_jacobian_jvp.fut:9:5-66
            
            bool x_24880 = lifted_lambda_res_25527 && redout_26466;
            bool redout_tmp_26808 = x_24880;
            
            redout_26466 = redout_tmp_26808;
        }
        defunc_0_reduce_res_26282 = redout_26466;
        // test/test_sparse_jacobian_jvp.fut:10:6-39
        
        bool x_24724 = defunc_0_reduce_res_26282 && redout_26468;
        bool redout_tmp_26805 = x_24724;
        
        redout_26468 = redout_tmp_26805;
    }
    defunc_0_reduce_res_26381 = redout_26468;
    // src/sparse_jacobian_jvp.fut:42:15-35
    if (mem_26700_cached_sizze_26917 < bytes_26663) {
        err = lexical_realloc(ctx, &mem_26700, &mem_26700_cached_sizze_26917, bytes_26663);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    for (int64_t nest_i_26809 = 0; nest_i_26809 < csr_bipartite_from_pattern_res_23527; nest_i_26809++) {
        ((double *) mem_26700)[nest_i_26809] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    if (mem_26703_cached_sizze_26918 < bytes_26702) {
        err = lexical_realloc(ctx, &mem_26703, &mem_26703_cached_sizze_26918, bytes_26702);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:17:3-61
    if (mem_26708_cached_sizze_26919 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26708, &mem_26708_cached_sizze_26919, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    for (int64_t i_26476 = 0; i_26476 < num_colors_of_res_f_res_23975; i_26476++) {
        // src/sparse_jacobian_jvp.fut:17:3-61
        for (int64_t i_26472 = 0; i_26472 < (int64_t) 6; i_26472++) {
            int64_t eta_p_23988 = ((int64_t *) mem_26570)[i_26472];
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            bool cond_23989 = eta_p_23988 == i_26476;
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            double lifted_lambda_res_23990;
            
            if (cond_23989) {
                lifted_lambda_res_23990 = 1.0;
            } else {
                lifted_lambda_res_23990 = 0.0;
            }
            ((double *) mem_26708)[i_26472] = lifted_lambda_res_23990;
        }
        
        double zt_lhs_tan_25923 = ((double *) mem_26708)[(int64_t) 0];
        double zt_rhs_tan_25924 = ((double *) mem_26708)[(int64_t) 1];
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_y_25927 = 0.25 * zt_rhs_tan_25924;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double zp_lhs_tan_25925 = zt_lhs_tan_25923 + binop_y_25927;
        double zt_rhs_tan_25928 = ((double *) mem_26708)[(int64_t) 4];
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double binop_y_25931 = 3.0 * zt_rhs_tan_25928;
        
        // test/test_sparse_jacobian_jvp.fut:82:24-39
        
        double y0_tan_25932 = zp_lhs_tan_25925 + binop_y_25931;
        double zm_lhs_tan_25935 = ((double *) mem_26708)[(int64_t) 2];
        double zt_lhs_tan_25936 = ((double *) mem_26708)[(int64_t) 5];
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double binop_x_25938 = 0.75 * zt_lhs_tan_25936;
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double zm_rhs_tan_25937 = binop_x_25938 + binop_x_25938;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double binop_y_25942 = -1.0 * zm_rhs_tan_25937;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double y1_tan_25940 = zm_lhs_tan_25935 + binop_y_25942;
        
        // test/test_sparse_jacobian_jvp.fut:84:17-23
        
        double zp_lhs_tan_25943 = zt_rhs_tan_25924 + zm_lhs_tan_25935;
        double zp_rhs_tan_25946 = ((double *) mem_26708)[(int64_t) 3];
        
        // test/test_sparse_jacobian_jvp.fut:84:24-30
        
        double y2_tan_25947 = zp_lhs_tan_25943 + zp_rhs_tan_25946;
        
        // test/test_sparse_jacobian_jvp.fut:172:44-49
        ((double *) mem_26703)[i_26476 * (int64_t) 4] = y0_tan_25932;
        ((double *) mem_26703)[i_26476 * (int64_t) 4 + (int64_t) 1] = y1_tan_25940;
        ((double *) mem_26703)[i_26476 * (int64_t) 4 + (int64_t) 2] = y2_tan_25947;
        ((double *) mem_26703)[i_26476 * (int64_t) 4 + (int64_t) 3] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:45:5-52:27
    
    bool compressed_to_csr_vals_res_23999;
    int64_t compressed_to_csr_vals_res_24001;
    bool loop_while_24002;
    int64_t i_24004;
    
    loop_while_24002 = 1;
    i_24004 = (int64_t) 0;
    while (loop_while_24002) {
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool x_24005 = sle64((int64_t) 0, i_24004);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool y_24006 = slt64(i_24004, csr_bipartite_from_pattern_res_23526);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool bounds_check_24007 = x_24005 && y_24006;
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool index_certs_24008;
        
        if (!bounds_check_24007) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_24004, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23526, "].", "-> #0  src/sparse_jacobian_jvp.fut:47:15-26\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-172:61\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        int64_t s_24009 = ((int64_t *) ext_mem_26565.mem)[i_24004];
        
        // src/sparse_jacobian_jvp.fut:48:25-27
        
        int64_t e_24010 = add64((int64_t) 1, i_24004);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool x_24011 = sle64((int64_t) 0, e_24010);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool y_24012 = slt64(e_24010, csr_bipartite_from_pattern_res_23526);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool bounds_check_24013 = x_24011 && y_24012;
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool index_certs_24014;
        
        if (!bounds_check_24013) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_24010, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23526, "].", "-> #0  src/sparse_jacobian_jvp.fut:48:15-28\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-172:61\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        int64_t e_24015 = ((int64_t *) ext_mem_26565.mem)[e_24010];
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t j_m_i_24016 = sub64(e_24015, s_24009);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool empty_slice_24017 = j_m_i_24016 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t m_24018 = sub64(j_m_i_24016, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t i_p_m_t_s_24019 = add64(s_24009, m_24018);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_24020 = sle64((int64_t) 0, i_p_m_t_s_24019);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_24021 = slt64(i_p_m_t_s_24019, csr_bipartite_from_pattern_res_23527);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_lte_i_24022 = sle64((int64_t) 0, s_24009);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_lte_j_24023 = sle64(s_24009, e_24015);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_24024 = i_p_m_t_s_leq_w_24021 && zzero_lte_i_24022;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_24025 = zzero_leq_i_p_m_t_s_24020 && y_24024;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool forwards_ok_24026 = i_lte_j_24023 && y_24025;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool ok_or_empty_24027 = empty_slice_24017 || forwards_ok_24026;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool index_certs_24028;
        
        if (!ok_or_empty_24027) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_24009, ":", (long long) e_24015, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23527, "].", "-> #0  src/sparse_jacobian_jvp.fut:49:18-30\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-172:61\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool y_24030 = slt64(i_24004, (int64_t) 4);
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool bounds_check_24031 = x_24005 && y_24030;
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool index_certs_24032;
        
        if (!bounds_check_24031) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_24004, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-45\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-172:61\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:18-51
        for (int64_t i_26480 = 0; i_26480 < j_m_i_24016; i_26480++) {
            int64_t index_primexp_26498 = s_24009 + i_26480;
            int64_t eta_p_24034 = ((int64_t *) ext_mem_26564.mem)[index_primexp_26498];
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool x_24035 = sle64((int64_t) 0, eta_p_24034);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool y_24036 = slt64(eta_p_24034, (int64_t) 6);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool bounds_check_24037 = x_24035 && y_24036;
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool index_certs_24038;
            
            if (!bounds_check_24037) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_24034, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:32-41\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-172:61\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            int64_t tmp_24039 = ((int64_t *) mem_26570)[eta_p_24034];
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool x_24040 = sle64((int64_t) 0, tmp_24039);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool y_24041 = slt64(tmp_24039, num_colors_of_res_f_res_23975);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool bounds_check_24042 = x_24040 && y_24041;
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool index_certs_24043;
            
            if (!bounds_check_24042) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_24039, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_23975, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-42\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-172:61\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:18-51
            
            double lifted_lambda_res_24044 = ((double *) mem_26703)[tmp_24039 * (int64_t) 4 + i_24004];
            
            ((double *) mem_26700)[s_24009 + i_26480] = lifted_lambda_res_24044;
        }
        // src/sparse_jacobian_jvp.fut:46:13-16
        
        bool loop_cond_24046 = slt64(e_24010, (int64_t) 4);
        bool loop_while_tmp_26812 = loop_cond_24046;
        int64_t i_tmp_26814 = e_24010;
        
        loop_while_24002 = loop_while_tmp_26812;
        i_24004 = i_tmp_26814;
    }
    compressed_to_csr_vals_res_23999 = loop_while_24002;
    compressed_to_csr_vals_res_24001 = i_24004;
    // test/test_sparse_jacobian_jvp.fut:174:6-60
    
    bool test_prepared_jvp_reuse_two_points_res_24052;
    
    if (defunc_0_reduce_res_26381) {
        // src/dense_jacobian.fut:8:40-9:68
        if (mem_26734_cached_sizze_26920 < (int64_t) 192) {
            err = lexical_realloc(ctx, &mem_26734, &mem_26734_cached_sizze_26920, (int64_t) 192);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/dense_jacobian.fut:5:3-21
        if (mem_26739_cached_sizze_26921 < (int64_t) 48) {
            err = lexical_realloc(ctx, &mem_26739, &mem_26739_cached_sizze_26921, (int64_t) 48);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/dense_jacobian.fut:8:40-9:68
        for (int64_t i_26484 = 0; i_26484 < (int64_t) 6; i_26484++) {
            // src/dense_jacobian.fut:5:3-21
            for (int64_t nest_i_26817 = 0; nest_i_26817 < (int64_t) 6; nest_i_26817++) {
                ((double *) mem_26739)[nest_i_26817] = 0.0;
            }
            // src/dense_jacobian.fut:5:3-39
            ((double *) mem_26739)[i_26484] = 1.0;
            
            double zt_lhs_tan_26299 = ((double *) mem_26739)[(int64_t) 0];
            double zt_rhs_tan_26300 = ((double *) mem_26739)[(int64_t) 1];
            
            // test/test_sparse_jacobian_jvp.fut:82:17-23
            
            double binop_y_26301 = 0.25 * zt_rhs_tan_26300;
            
            // test/test_sparse_jacobian_jvp.fut:82:17-23
            
            double zp_lhs_tan_26302 = zt_lhs_tan_26299 + binop_y_26301;
            double zt_rhs_tan_26303 = ((double *) mem_26739)[(int64_t) 4];
            
            // test/test_sparse_jacobian_jvp.fut:82:33-39
            
            double binop_y_26304 = 3.0 * zt_rhs_tan_26303;
            
            // test/test_sparse_jacobian_jvp.fut:82:24-39
            
            double y0_tan_26305 = zp_lhs_tan_26302 + binop_y_26304;
            double zm_lhs_tan_26306 = ((double *) mem_26739)[(int64_t) 2];
            double zt_lhs_tan_26307 = ((double *) mem_26739)[(int64_t) 5];
            
            // test/test_sparse_jacobian_jvp.fut:83:24-30
            
            double binop_x_26308 = 0.75 * zt_lhs_tan_26307;
            
            // test/test_sparse_jacobian_jvp.fut:83:24-30
            
            double zm_rhs_tan_26309 = binop_x_26308 + binop_x_26308;
            
            // test/test_sparse_jacobian_jvp.fut:83:17-30
            
            double binop_y_26310 = -1.0 * zm_rhs_tan_26309;
            
            // test/test_sparse_jacobian_jvp.fut:83:17-30
            
            double y1_tan_26311 = zm_lhs_tan_26306 + binop_y_26310;
            
            // test/test_sparse_jacobian_jvp.fut:84:17-23
            
            double zp_lhs_tan_26312 = zt_rhs_tan_26300 + zm_lhs_tan_26306;
            double zp_rhs_tan_26313 = ((double *) mem_26739)[(int64_t) 3];
            
            // test/test_sparse_jacobian_jvp.fut:84:24-30
            
            double y2_tan_26314 = zp_lhs_tan_26312 + zp_rhs_tan_26313;
            
            // test/test_sparse_jacobian_jvp.fut:171:60-65
            ((double *) mem_26734)[i_26484 * (int64_t) 4] = y0_tan_26305;
            ((double *) mem_26734)[i_26484 * (int64_t) 4 + (int64_t) 1] = y1_tan_26311;
            ((double *) mem_26734)[i_26484 * (int64_t) 4 + (int64_t) 2] = y2_tan_26314;
            ((double *) mem_26734)[i_26484 * (int64_t) 4 + (int64_t) 3] = 0.0;
        }
        // src/sparse_jacobian_jvp.fut:67:29-47
        if (mem_26745_cached_sizze_26922 < (int64_t) 48) {
            err = lexical_realloc(ctx, &mem_26745, &mem_26745_cached_sizze_26922, (int64_t) 48);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // test/test_sparse_jacobian_jvp.fut:8:19-10:39
        
        bool defunc_0_reduce_res_26319;
        bool redout_26490 = 1;
        
        for (int64_t i_26491 = 0; i_26491 < (int64_t) 4; i_26491++) {
            // src/sparse_jacobian_jvp.fut:63:17-28
            
            bool y_26323 = slt64(i_26491, csr_bipartite_from_pattern_res_23526);
            
            // src/sparse_jacobian_jvp.fut:63:17-28
            
            bool index_certs_26326;
            
            if (!y_26323) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_26491, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23526, "].", "-> #0  src/sparse_jacobian_jvp.fut:63:17-28\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  test/test_sparse_jacobian_jvp.fut:81:15-172:61\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:63:17-28
            
            int64_t s_26327 = ((int64_t *) ext_mem_26565.mem)[i_26491];
            
            // src/sparse_jacobian_jvp.fut:65:20-32
            
            bool zzero_lte_i_26328 = sle64((int64_t) 0, s_26327);
            
            // src/sparse_jacobian_jvp.fut:64:27-29
            
            int64_t e_26329 = add64((int64_t) 1, i_26491);
            
            // src/sparse_jacobian_jvp.fut:64:17-30
            
            bool y_26330 = slt64(e_26329, csr_bipartite_from_pattern_res_23526);
            
            // src/sparse_jacobian_jvp.fut:64:17-30
            
            bool x_26331 = sle64((int64_t) 0, e_26329);
            
            // src/sparse_jacobian_jvp.fut:64:17-30
            
            bool bounds_check_26332 = y_26330 && x_26331;
            
            // src/sparse_jacobian_jvp.fut:64:17-30
            
            bool index_certs_26333;
            
            if (!bounds_check_26332) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_26329, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23526, "].", "-> #0  src/sparse_jacobian_jvp.fut:64:17-30\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  test/test_sparse_jacobian_jvp.fut:81:15-172:61\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:64:17-30
            
            int64_t e_26334 = ((int64_t *) ext_mem_26565.mem)[e_26329];
            
            // src/sparse_jacobian_jvp.fut:65:20-32
            
            int64_t j_m_i_26335 = sub64(e_26334, s_26327);
            
            // src/sparse_jacobian_jvp.fut:65:20-32
            
            int64_t m_26336 = sub64(j_m_i_26335, (int64_t) 1);
            
            // src/sparse_jacobian_jvp.fut:65:20-32
            
            int64_t i_p_m_t_s_26337 = add64(s_26327, m_26336);
            
            // src/sparse_jacobian_jvp.fut:65:20-32
            
            bool i_p_m_t_s_leq_w_26338 = slt64(i_p_m_t_s_26337, csr_bipartite_from_pattern_res_23527);
            
            // src/sparse_jacobian_jvp.fut:65:20-32
            
            bool y_26339 = zzero_lte_i_26328 && i_p_m_t_s_leq_w_26338;
            
            // src/sparse_jacobian_jvp.fut:65:20-32
            
            bool zzero_leq_i_p_m_t_s_26340 = sle64((int64_t) 0, i_p_m_t_s_26337);
            
            // src/sparse_jacobian_jvp.fut:65:20-32
            
            bool y_26341 = y_26339 && zzero_leq_i_p_m_t_s_26340;
            
            // src/sparse_jacobian_jvp.fut:65:20-32
            
            bool i_lte_j_26342 = sle64(s_26327, e_26334);
            
            // src/sparse_jacobian_jvp.fut:65:20-32
            
            bool forwards_ok_26343 = y_26341 && i_lte_j_26342;
            
            // src/sparse_jacobian_jvp.fut:65:20-32
            
            bool empty_slice_26344 = j_m_i_26335 == (int64_t) 0;
            
            // src/sparse_jacobian_jvp.fut:65:20-32
            
            bool ok_or_empty_26345 = forwards_ok_26343 || empty_slice_26344;
            
            // src/sparse_jacobian_jvp.fut:65:20-32
            
            bool index_certs_26346;
            
            if (!ok_or_empty_26345) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_26327, ":", (long long) e_26334, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_23527, "].", "-> #0  src/sparse_jacobian_jvp.fut:65:20-32\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  test/test_sparse_jacobian_jvp.fut:81:15-172:61\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:67:29-47
            for (int64_t nest_i_26819 = 0; nest_i_26819 < (int64_t) 6; nest_i_26819++) {
                ((double *) mem_26745)[nest_i_26819] = 0.0;
            }
            // src/sparse_jacobian_jvp.fut:68:12-32
            
            bool acc_cert_26351;
            
            // src/sparse_jacobian_jvp.fut:68:12-32
            for (int64_t i_26487 = 0; i_26487 < j_m_i_26335; i_26487++) {
                int64_t index_primexp_26493 = s_26327 + i_26487;
                int64_t v_26355 = ((int64_t *) ext_mem_26564.mem)[index_primexp_26493];
                double v_26356 = ((double *) mem_26700)[index_primexp_26493];
                
                // src/sparse_jacobian_jvp.fut:68:12-32
                // UpdateAcc
                if (sle64((int64_t) 0, v_26355) && slt64(v_26355, (int64_t) 6)) {
                    ((double *) mem_26745)[v_26355] = v_26356;
                }
            }
            // test/test_sparse_jacobian_jvp.fut:9:5-66
            
            bool defunc_0_reduce_res_26359;
            bool redout_26488 = 1;
            
            for (int64_t i_26489 = 0; i_26489 < (int64_t) 6; i_26489++) {
                bool eta_p_26360 = ((bool *) mem_26555.mem)[i_26491 * (int64_t) 6 + i_26489];
                double eta_p_26361 = ((double *) mem_26734)[i_26489 * (int64_t) 4 + i_26491];
                double eta_p_26362 = ((double *) mem_26745)[i_26489];
                
                // test/test_sparse_jacobian_jvp.fut:14:25-48
                
                double lifted_lambda_res_26363;
                
                if (eta_p_26360) {
                    lifted_lambda_res_26363 = eta_p_26361;
                } else {
                    lifted_lambda_res_26363 = 0.0;
                }
                // test/test_sparse_jacobian_jvp.fut:9:48-51
                
                double abs_arg0_26364 = eta_p_26362 - lifted_lambda_res_26363;
                
                // test/test_sparse_jacobian_jvp.fut:9:37-51
                
                double abs_res_26365 = fabs64(abs_arg0_26364);
                
                // test/test_sparse_jacobian_jvp.fut:9:53-59
                
                bool lifted_lambda_res_26366 = abs_res_26365 <= 1.0e-9;
                
                // test/test_sparse_jacobian_jvp.fut:9:5-66
                
                bool x_26369 = lifted_lambda_res_26366 && redout_26488;
                bool redout_tmp_26821 = x_26369;
                
                redout_26488 = redout_tmp_26821;
            }
            defunc_0_reduce_res_26359 = redout_26488;
            // test/test_sparse_jacobian_jvp.fut:10:6-39
            
            bool x_26373 = defunc_0_reduce_res_26359 && redout_26490;
            bool redout_tmp_26818 = x_26373;
            
            redout_26490 = redout_tmp_26818;
        }
        defunc_0_reduce_res_26319 = redout_26490;
        test_prepared_jvp_reuse_two_points_res_24052 = defunc_0_reduce_res_26319;
    } else {
        test_prepared_jvp_reuse_two_points_res_24052 = 0;
    }
    if (memblock_unref(ctx, &ext_mem_26564, "ext_mem_26564") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26565, "ext_mem_26565") != 0)
        return 1;
    prim_out_26758 = test_prepared_jvp_reuse_two_points_res_24052;
    *out_prim_out_26898 = prim_out_26758;
    
  cleanup:
    {
        free(mem_26570);
        free(mem_26571);
        free(mem_26587);
        free(mem_26591);
        free(mem_26608);
        free(mem_26610);
        free(mem_26611);
        free(mem_26631);
        free(mem_26633);
        free(mem_26635);
        free(mem_26643);
        free(mem_26652);
        free(mem_26657);
        free(mem_26664);
        free(mem_26667);
        free(mem_26672);
        free(mem_26688);
        free(mem_26698);
        free(mem_26700);
        free(mem_26703);
        free(mem_26708);
        free(mem_26734);
        free(mem_26739);
        free(mem_26745);
        if (memblock_unref(ctx, &mem_param_tmp_26763, "mem_param_tmp_26763") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26645, "mem_26645") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_26585, "mem_param_26585") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26650, "ext_mem_26650") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26579, "mem_26579") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26568, "ext_mem_26568") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26569, "ext_mem_26569") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26566, "mem_26566") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26564, "ext_mem_26564") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26565, "ext_mem_26565") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_jvp_ex1_matches_dense(struct futhark_context *ctx, bool *out_prim_out_26923, struct memblock x_mem_26563)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_26564_cached_sizze_26924 = 0;
    unsigned char *mem_26564 = NULL;
    int64_t mem_26569_cached_sizze_26925 = 0;
    unsigned char *mem_26569 = NULL;
    int64_t mem_26578_cached_sizze_26926 = 0;
    unsigned char *mem_26578 = NULL;
    int64_t mem_26579_cached_sizze_26927 = 0;
    unsigned char *mem_26579 = NULL;
    int64_t mem_26588_cached_sizze_26928 = 0;
    unsigned char *mem_26588 = NULL;
    int64_t mem_26600_cached_sizze_26929 = 0;
    unsigned char *mem_26600 = NULL;
    int64_t mem_26604_cached_sizze_26930 = 0;
    unsigned char *mem_26604 = NULL;
    int64_t mem_26621_cached_sizze_26931 = 0;
    unsigned char *mem_26621 = NULL;
    int64_t mem_26623_cached_sizze_26932 = 0;
    unsigned char *mem_26623 = NULL;
    int64_t mem_26624_cached_sizze_26933 = 0;
    unsigned char *mem_26624 = NULL;
    int64_t mem_26644_cached_sizze_26934 = 0;
    unsigned char *mem_26644 = NULL;
    int64_t mem_26646_cached_sizze_26935 = 0;
    unsigned char *mem_26646 = NULL;
    int64_t mem_26648_cached_sizze_26936 = 0;
    unsigned char *mem_26648 = NULL;
    int64_t mem_26656_cached_sizze_26937 = 0;
    unsigned char *mem_26656 = NULL;
    int64_t mem_26667_cached_sizze_26938 = 0;
    unsigned char *mem_26667 = NULL;
    int64_t mem_26672_cached_sizze_26939 = 0;
    unsigned char *mem_26672 = NULL;
    int64_t mem_26698_cached_sizze_26940 = 0;
    unsigned char *mem_26698 = NULL;
    struct memblock mem_param_tmp_26763;
    
    mem_param_tmp_26763.references = NULL;
    
    struct memblock mem_26658;
    
    mem_26658.references = NULL;
    
    struct memblock mem_param_26598;
    
    mem_param_26598.references = NULL;
    
    struct memblock ext_mem_26663;
    
    ext_mem_26663.references = NULL;
    
    struct memblock ext_mem_26591;
    
    ext_mem_26591.references = NULL;
    
    struct memblock ext_mem_26592;
    
    ext_mem_26592.references = NULL;
    
    struct memblock mem_26589;
    
    mem_26589.references = NULL;
    
    struct memblock mem_26587;
    
    mem_26587.references = NULL;
    
    struct memblock ext_mem_26575;
    
    ext_mem_26575.references = NULL;
    
    struct memblock ext_mem_26576;
    
    ext_mem_26576.references = NULL;
    
    struct memblock mem_26545 = ctx->constants->mem_26545;
    struct memblock mem_26548 = ctx->constants->mem_26548;
    struct memblock mem_26550 = ctx->constants->mem_26550;
    struct memblock mem_26555 = ctx->constants->mem_26555;
    struct memblock mem_26556 = ctx->constants->mem_26556;
    struct memblock mem_26562 = ctx->constants->mem_26562;
    bool prim_out_26758;
    
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_24086;
    int64_t csr_bipartite_from_pattern_res_24087;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26576, &ext_mem_26575, &csr_bipartite_from_pattern_res_24086, &csr_bipartite_from_pattern_res_24087, mem_26545, (int64_t) 3, (int64_t) 5) != 0) {
        err = 1;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    
    int64_t bytes_26577 = (int64_t) 8 * csr_bipartite_from_pattern_res_24087;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (mem_26579_cached_sizze_26927 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_26579, &mem_26579_cached_sizze_26927, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t discard_26393;
    int64_t scanacc_26389 = (int64_t) 0;
    
    for (int64_t i_26391 = 0; i_26391 < (int64_t) 5; i_26391++) {
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t defunc_0_op_res_24102 = add64((int64_t) 1, scanacc_26389);
        
        ((int64_t *) mem_26579)[i_26391] = defunc_0_op_res_24102;
        
        int64_t scanacc_tmp_26759 = defunc_0_op_res_24102;
        
        scanacc_26389 = scanacc_tmp_26759;
    }
    discard_26393 = scanacc_26389;
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t x_24112 = ((int64_t *) mem_26579)[(int64_t) 4];
    
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    bool loop_cond_24126 = slt64((int64_t) 0, x_24112);
    
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t bytes_26586 = (int64_t) 8 * x_24112;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (memblock_alloc(ctx, &mem_26587, bytes_26586, "mem_26587")) {
        err = 1;
        goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    bool acc_cert_24821;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    for (int64_t i_26395 = 0; i_26395 < (int64_t) 5; i_26395++) {
        int64_t eta_p_24833 = ((int64_t *) mem_26579)[i_26395];
        
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t lifted_lambda_res_t_res_24836 = sub64(eta_p_24833, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:194:20-45
        // UpdateAcc
        if (sle64((int64_t) 0, lifted_lambda_res_t_res_24836) && slt64(lifted_lambda_res_t_res_24836, x_24112)) {
            ((int64_t *) mem_26587.mem)[lifted_lambda_res_t_res_24836] = i_26395;
        }
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    if (mem_26588_cached_sizze_26928 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_26588, &mem_26588_cached_sizze_26928, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    for (int64_t nest_i_26762 = 0; nest_i_26762 < (int64_t) 5; nest_i_26762++) {
        ((int64_t *) mem_26588)[nest_i_26762] = (int64_t) -1;
    }
    // src/pattern_csr.fut:25:3-39
    if (memblock_alloc(ctx, &mem_26589, (int64_t) 15, "mem_26589")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:25:3-39
    // src/pattern_csr.fut:25:3-39
    lmad_copy_1b(ctx, 2, (uint8_t *) mem_26589.mem, (int64_t) 0, (int64_t []) {(int64_t) 3, (int64_t) 1}, (uint8_t *) mem_26545.mem, (int64_t) 0, (int64_t []) {(int64_t) 1, (int64_t) 5}, (int64_t []) {(int64_t) 5, (int64_t) 3});
    // src/pattern_csr.fut:25:3-39
    
    int64_t csr_cols_from_pattern_res_24091;
    int64_t csr_cols_from_pattern_res_24092;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26592, &ext_mem_26591, &csr_cols_from_pattern_res_24091, &csr_cols_from_pattern_res_24092, mem_26589, (int64_t) 5, (int64_t) 3) != 0) {
        err = 1;
        goto cleanup;
    }
    if (memblock_unref(ctx, &mem_26589, "mem_26589") != 0)
        return 1;
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    int64_t vv_color_side_order_res_24127;
    bool vv_color_side_order_res_24128;
    int64_t vv_color_side_order_res_24131;
    int64_t loop_dz2081Uz2083U_24132;
    bool loop_while_24133;
    int64_t color_bound_24136;
    
    if (memblock_set(ctx, &mem_param_26598, &mem_26587, "mem_26587") != 0)
        return 1;
    loop_dz2081Uz2083U_24132 = x_24112;
    loop_while_24133 = loop_cond_24126;
    color_bound_24136 = (int64_t) 1;
    while (loop_while_24133) {
        // src/bgpc_vv_coloring.fut:62:5-64:15
        
        bool cond_24137 = slt64((int64_t) 0, color_bound_24136);
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t bytes_26599 = (int64_t) 8 * loop_dz2081Uz2083U_24132;
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        if (mem_26600_cached_sizze_26929 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26600, &mem_26600_cached_sizze_26929, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:77:24-51
        if (mem_26604_cached_sizze_26930 < color_bound_24136) {
            err = lexical_realloc(ctx, &mem_26604, &mem_26604_cached_sizze_26930, color_bound_24136);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t defunc_0_reduce_res_26019;
        int64_t redout_26397 = (int64_t) -1;
        
        for (int64_t i_26399 = 0; i_26399 < loop_dz2081Uz2083U_24132; i_26399++) {
            int64_t eta_p_25358 = ((int64_t *) mem_param_26598.mem)[i_26399];
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool x_25360 = sle64((int64_t) 0, eta_p_25358);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool y_25361 = slt64(eta_p_25358, csr_cols_from_pattern_res_24091);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool bounds_check_25362 = x_25360 && y_25361;
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool index_certs_25363;
            
            if (!bounds_check_25362) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25358, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24091, "].", "-> #0  src/bgpc_vv_coloring.fut:46:44-56\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:44-56
            
            int64_t seen_final_25364 = ((int64_t *) ext_mem_26592.mem)[eta_p_25358];
            
            // src/bgpc_vv_coloring.fut:46:70-72
            
            int64_t seen_final_25365 = add64((int64_t) 1, eta_p_25358);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool x_25366 = sle64((int64_t) 0, seen_final_25365);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool y_25367 = slt64(seen_final_25365, csr_cols_from_pattern_res_24091);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool bounds_check_25368 = x_25366 && y_25367;
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool index_certs_25369;
            
            if (!bounds_check_25368) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_25365, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24091, "].", "-> #0  src/bgpc_vv_coloring.fut:46:59-73\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:59-73
            
            int64_t seen_final_25370 = ((int64_t *) ext_mem_26592.mem)[seen_final_25365];
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t j_m_i_25371 = sub64(seen_final_25370, seen_final_25364);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool empty_slice_25372 = j_m_i_25371 == (int64_t) 0;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t m_25373 = sub64(j_m_i_25371, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t i_p_m_t_s_25374 = add64(seen_final_25364, m_25373);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_leq_i_p_m_t_s_25375 = sle64((int64_t) 0, i_p_m_t_s_25374);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_p_m_t_s_leq_w_25376 = slt64(i_p_m_t_s_25374, csr_cols_from_pattern_res_24092);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_lte_i_25377 = sle64((int64_t) 0, seen_final_25364);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_lte_j_25378 = sle64(seen_final_25364, seen_final_25370);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_25379 = i_p_m_t_s_leq_w_25376 && zzero_lte_i_25377;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_25380 = zzero_leq_i_p_m_t_s_25375 && y_25379;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool forwards_ok_25381 = i_lte_j_25378 && y_25380;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool ok_or_empty_25382 = empty_slice_25372 || forwards_ok_25381;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool index_certs_25383;
            
            if (!ok_or_empty_25382) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_25364, ":", (long long) seen_final_25370, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24092, "].", "-> #0  src/bgpc_vv_coloring.fut:46:5-54:19\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:77:24-51
            for (int64_t nest_i_26771 = 0; nest_i_26771 < color_bound_24136; nest_i_26771++) {
                ((bool *) mem_26604)[nest_i_26771] = 0;
            }
            // src/bgpc_vv_coloring.fut:46:5-54:19
            for (int64_t i_25385 = 0; i_25385 < j_m_i_25371; i_25385++) {
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t index_primexp_25387 = seen_final_25364 + i_25385;
                
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t v_25388 = ((int64_t *) ext_mem_26591.mem)[index_primexp_25387];
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool x_25389 = sle64((int64_t) 0, v_25388);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool y_25390 = slt64(v_25388, csr_bipartite_from_pattern_res_24086);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool bounds_check_25391 = x_25389 && y_25390;
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool index_certs_25392;
                
                if (!bounds_check_25391) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_25388, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/bgpc_vv_coloring.fut:49:26-37\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:26-37
                
                int64_t seen_acczq_25393 = ((int64_t *) ext_mem_26576.mem)[v_25388];
                
                // src/bgpc_vv_coloring.fut:49:50-52
                
                int64_t seen_acczq_25394 = add64((int64_t) 1, v_25388);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool x_25395 = sle64((int64_t) 0, seen_acczq_25394);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool y_25396 = slt64(seen_acczq_25394, csr_bipartite_from_pattern_res_24086);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool bounds_check_25397 = x_25395 && y_25396;
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool index_certs_25398;
                
                if (!bounds_check_25397) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_25394, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/bgpc_vv_coloring.fut:49:40-53\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:40-53
                
                int64_t seen_acczq_25399 = ((int64_t *) ext_mem_26576.mem)[seen_acczq_25394];
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t j_m_i_25400 = sub64(seen_acczq_25399, seen_acczq_25393);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool empty_slice_25401 = j_m_i_25400 == (int64_t) 0;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t m_25402 = sub64(j_m_i_25400, (int64_t) 1);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t i_p_m_t_s_25403 = add64(seen_acczq_25393, m_25402);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_leq_i_p_m_t_s_25404 = sle64((int64_t) 0, i_p_m_t_s_25403);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_p_m_t_s_leq_w_25405 = slt64(i_p_m_t_s_25403, csr_bipartite_from_pattern_res_24087);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_lte_i_25406 = sle64((int64_t) 0, seen_acczq_25393);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_lte_j_25407 = sle64(seen_acczq_25393, seen_acczq_25399);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_25408 = i_p_m_t_s_leq_w_25405 && zzero_lte_i_25406;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_25409 = zzero_leq_i_p_m_t_s_25404 && y_25408;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool forwards_ok_25410 = i_lte_j_25407 && y_25409;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool ok_or_empty_25411 = empty_slice_25401 || forwards_ok_25410;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool index_certs_25412;
                
                if (!ok_or_empty_25411) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_25393, ":", (long long) seen_acczq_25399, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24087, "].", "-> #0  src/bgpc_vv_coloring.fut:48:9-53:33\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:48:9-53:33
                for (int64_t i_25414 = 0; i_25414 < j_m_i_25400; i_25414++) {
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t index_primexp_25416 = seen_acczq_25393 + i_25414;
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t u_25417 = ((int64_t *) ext_mem_26575.mem)[index_primexp_25416];
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool x_25418 = sle64((int64_t) 0, u_25417);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool y_25419 = slt64(u_25417, (int64_t) 5);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool bounds_check_25420 = x_25418 && y_25419;
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool index_certs_25421;
                    
                    if (!bounds_check_25420) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_25417, "] out of bounds for array of shape [", (long long) (int64_t) 5, "].", "-> #0  src/bgpc_vv_coloring.fut:50:20-29\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    int64_t cu_25422 = ((int64_t *) mem_26588)[u_25417];
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_25423 = u_25417 == eta_p_25358;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_25424 = !cond_25423;
                    
                    // src/bgpc_vv_coloring.fut:51:30-37
                    
                    bool cond_t_res_25425 = sle64((int64_t) 0, cu_25422);
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    bool x_25426 = cond_25424 && cond_t_res_25425;
                    
                    // src/bgpc_vv_coloring.fut:51:44-67
                    
                    bool cond_t_res_25427 = slt64(cu_25422, color_bound_24136);
                    
                    // src/bgpc_vv_coloring.fut:94:14-103:69
                    
                    bool x_25428 = x_25426 && cond_t_res_25427;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    if (x_25428) {
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool bounds_check_25983 = cond_t_res_25425 && cond_t_res_25427;
                        
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool index_certs_25984;
                        
                        if (!bounds_check_25983) {
                            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) cu_25422, "] out of bounds for array of shape [", (long long) color_bound_24136, "].", "-> #0  src/bgpc_vv_coloring.fut:52:19-50\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                            err = FUTHARK_PROGRAM_ERROR;
                            goto cleanup;
                        }
                        // src/bgpc_vv_coloring.fut:52:19-50
                        ((bool *) mem_26604)[cu_25422] = 1;
                    }
                }
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool loop_cond_25433;
            
            if (cond_24137) {
                // src/bgpc_vv_coloring.fut:63:30-37
                
                bool loop_cond_t_res_25986 = ((bool *) mem_26604)[(int64_t) 0];
                
                loop_cond_25433 = loop_cond_t_res_25986;
            } else {
                loop_cond_25433 = 0;
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool c_final_25435;
            int64_t c_final_25436;
            bool loop_while_25437;
            int64_t c_25438;
            
            loop_while_25437 = loop_cond_25433;
            c_25438 = (int64_t) 0;
            while (loop_while_25437) {
                // src/bgpc_vv_coloring.fut:64:9-15
                
                int64_t loopres_25439 = add64((int64_t) 1, c_25438);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool cond_25440 = slt64(loopres_25439, color_bound_24136);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool loop_cond_25441;
                
                if (cond_25440) {
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool x_25987 = sle64((int64_t) 0, loopres_25439);
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool bounds_check_25988 = cond_25440 && x_25987;
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool index_certs_25989;
                    
                    if (!bounds_check_25988) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) loopres_25439, "] out of bounds for array of shape [", (long long) color_bound_24136, "].", "-> #0  src/bgpc_vv_coloring.fut:63:30-37\n   #1  src/bgpc_vv_coloring.fut:81:6-28\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool loop_cond_t_res_25990 = ((bool *) mem_26604)[loopres_25439];
                    
                    loop_cond_25441 = loop_cond_t_res_25990;
                } else {
                    loop_cond_25441 = 0;
                }
                
                bool loop_while_tmp_26774 = loop_cond_25441;
                int64_t c_tmp_26775 = loopres_25439;
                
                loop_while_25437 = loop_while_tmp_26774;
                c_25438 = c_tmp_26775;
            }
            c_final_25435 = loop_while_25437;
            c_final_25436 = c_25438;
            // src/bgpc_vv_coloring.fut:106:25-32
            
            int64_t max_res_24231 = smax64(c_final_25436, redout_26397);
            
            ((int64_t *) mem_26600)[i_26399] = c_final_25436;
            
            int64_t redout_tmp_26769 = max_res_24231;
            
            redout_26397 = redout_tmp_26769;
        }
        defunc_0_reduce_res_26019 = redout_26397;
        // src/bgpc_vv_coloring.fut:113:17-43
        
        bool acc_cert_24237;
        
        // src/bgpc_vv_coloring.fut:113:17-43
        for (int64_t i_26402 = 0; i_26402 < loop_dz2081Uz2083U_24132; i_26402++) {
            int64_t v_24241 = ((int64_t *) mem_param_26598.mem)[i_26402];
            int64_t v_24242 = ((int64_t *) mem_26600)[i_26402];
            
            // src/bgpc_vv_coloring.fut:113:17-43
            // UpdateAcc
            if (sle64((int64_t) 0, v_24241) && slt64(v_24241, (int64_t) 5)) {
                ((int64_t *) mem_26588)[v_24241] = v_24242;
            }
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26621_cached_sizze_26931 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26621, &mem_26621_cached_sizze_26931, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26623_cached_sizze_26932 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26623, &mem_26623_cached_sizze_26932, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26624_cached_sizze_26933 < loop_dz2081Uz2083U_24132) {
            err = lexical_realloc(ctx, &mem_26624, &mem_26624_cached_sizze_26933, loop_dz2081Uz2083U_24132);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t discard_26414;
        int64_t scanacc_26406 = (int64_t) 0;
        
        for (int64_t i_26410 = 0; i_26410 < loop_dz2081Uz2083U_24132; i_26410++) {
            int64_t eta_p_25282 = ((int64_t *) mem_param_26598.mem)[i_26410];
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool x_25283 = sle64((int64_t) 0, eta_p_25282);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool y_25284 = slt64(eta_p_25282, csr_cols_from_pattern_res_24091);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool bounds_check_25285 = x_25283 && y_25284;
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool index_certs_25286;
            
            if (!bounds_check_25285) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25282, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24091, "].", "-> #0  src/bgpc_vv_coloring.fut:126:15-27\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:127:26-28
            
            int64_t k_end_25288 = add64((int64_t) 1, eta_p_25282);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool x_25289 = sle64((int64_t) 0, k_end_25288);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool y_25290 = slt64(k_end_25288, csr_cols_from_pattern_res_24091);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool bounds_check_25291 = x_25289 && y_25290;
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool index_certs_25292;
            
            if (!bounds_check_25291) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_end_25288, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24091, "].", "-> #0  src/bgpc_vv_coloring.fut:127:15-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:126:15-27
            
            int64_t k0_25287 = ((int64_t *) ext_mem_26592.mem)[eta_p_25282];
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            int64_t k_end_25293 = ((int64_t *) ext_mem_26592.mem)[k_end_25288];
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool cond_25294 = slt64(k0_25287, k_end_25293);
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool loses_conflict_vertex_res_25295;
            bool loses_conflict_vertex_res_25296;
            int64_t loses_conflict_vertex_res_25297;
            bool loop_while_25298;
            bool lost_25299;
            int64_t k_25300;
            
            loop_while_25298 = cond_25294;
            lost_25299 = 0;
            k_25300 = k0_25287;
            while (loop_while_25298) {
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool x_25301 = sle64((int64_t) 0, k_25300);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool y_25302 = slt64(k_25300, csr_cols_from_pattern_res_24092);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool bounds_check_25303 = x_25301 && y_25302;
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool index_certs_25304;
                
                if (!bounds_check_25303) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_25300, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24092, "].", "-> #0  src/bgpc_vv_coloring.fut:132:15-26\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:132:15-26
                
                int64_t v_25305 = ((int64_t *) ext_mem_26591.mem)[k_25300];
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool x_25306 = sle64((int64_t) 0, v_25305);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool y_25307 = slt64(v_25305, csr_bipartite_from_pattern_res_24086);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool bounds_check_25308 = x_25306 && y_25307;
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool index_certs_25309;
                
                if (!bounds_check_25308) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_25305, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/bgpc_vv_coloring.fut:134:19-30\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:135:29-31
                
                int64_t t_end_25311 = add64((int64_t) 1, v_25305);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool x_25312 = sle64((int64_t) 0, t_end_25311);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool y_25313 = slt64(t_end_25311, csr_bipartite_from_pattern_res_24086);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool bounds_check_25314 = x_25312 && y_25313;
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool index_certs_25315;
                
                if (!bounds_check_25314) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_end_25311, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/bgpc_vv_coloring.fut:135:19-32\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:134:19-30
                
                int64_t t0_25310 = ((int64_t *) ext_mem_26576.mem)[v_25305];
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                int64_t t_end_25316 = ((int64_t *) ext_mem_26576.mem)[t_end_25311];
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool cond_25317 = slt64(t0_25310, t_end_25316);
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool loopres_25318;
                bool loopres_25319;
                int64_t loopres_25320;
                bool loop_while_25321;
                bool lost_in_net_25322;
                int64_t t_25323;
                
                loop_while_25321 = cond_25317;
                lost_in_net_25322 = 0;
                t_25323 = t0_25310;
                while (loop_while_25321) {
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool x_25324 = sle64((int64_t) 0, t_25323);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool y_25325 = slt64(t_25323, csr_bipartite_from_pattern_res_24087);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool bounds_check_25326 = x_25324 && y_25325;
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool index_certs_25327;
                    
                    if (!bounds_check_25326) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_25323, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24087, "].", "-> #0  src/bgpc_vv_coloring.fut:140:19-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    int64_t u_25328 = ((int64_t *) ext_mem_26575.mem)[t_25323];
                    
                    // src/bgpc_vv_coloring.fut:142:13-71
                    
                    bool lost_in_netzq_25329;
                    
                    if (lost_in_net_25322) {
                        lost_in_netzq_25329 = 1;
                    } else {
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25330 = u_25328 == eta_p_25282;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25331 = !cond_25330;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25332;
                        
                        if (cond_25331) {
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool x_25992 = sle64((int64_t) 0, u_25328);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool y_25993 = slt64(u_25328, (int64_t) 5);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool bounds_check_25994 = x_25992 && y_25993;
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool index_certs_25995;
                            
                            if (!bounds_check_25994) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_25328, "] out of bounds for array of shape [", (long long) (int64_t) 5, "].", "-> #0  src/bgpc_vv_coloring.fut:142:39-48\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool y_25997 = slt64(eta_p_25282, (int64_t) 5);
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool bounds_check_25998 = x_25283 && y_25997;
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool index_certs_25999;
                            
                            if (!bounds_check_25998) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25282, "] out of bounds for array of shape [", (long long) (int64_t) 5, "].", "-> #0  src/bgpc_vv_coloring.fut:142:52-61\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            int64_t zeze_lhs_25996 = ((int64_t *) mem_26588)[u_25328];
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            int64_t zeze_rhs_26000 = ((int64_t *) mem_26588)[eta_p_25282];
                            
                            // src/bgpc_vv_coloring.fut:142:49-61
                            
                            bool cond_t_res_26001 = zeze_lhs_25996 == zeze_rhs_26000;
                            
                            cond_25332 = cond_t_res_26001;
                        } else {
                            cond_25332 = 0;
                        }
                        // src/bgpc_vv_coloring.fut:142:67-70
                        
                        bool lost_in_netzq_f_res_t_res_25343 = slt64(u_25328, eta_p_25282);
                        
                        // src/bgpc_vv_coloring.fut:160:14-168:57
                        
                        bool x_25344 = cond_25332 && lost_in_netzq_f_res_t_res_25343;
                        
                        lost_in_netzq_25329 = x_25344;
                    }
                    // src/bgpc_vv_coloring.fut:143:31-37
                    
                    int64_t tmp_25345 = add64((int64_t) 1, t_25323);
                    
                    // src/bgpc_vv_coloring.fut:139:15-43
                    
                    bool cond_25346 = slt64(tmp_25345, t_end_25316);
                    
                    // src/bgpc_vv_coloring.fut:139:28-43
                    
                    bool not_res_25347 = !lost_in_netzq_25329;
                    
                    // src/bgpc_vv_coloring.fut:166:5-169:10
                    
                    bool x_25348 = cond_25346 && not_res_25347;
                    bool loop_while_tmp_26784 = x_25348;
                    bool lost_in_net_tmp_26785 = lost_in_netzq_25329;
                    int64_t t_tmp_26786 = tmp_25345;
                    
                    loop_while_25321 = loop_while_tmp_26784;
                    lost_in_net_25322 = lost_in_net_tmp_26785;
                    t_25323 = t_tmp_26786;
                }
                loopres_25318 = loop_while_25321;
                loopres_25319 = lost_in_net_25322;
                loopres_25320 = t_25323;
                // src/bgpc_vv_coloring.fut:160:14-168:57
                
                bool lostzq_25349 = lost_25299 || loopres_25319;
                
                // src/bgpc_vv_coloring.fut:146:20-26
                
                int64_t tmp_25350 = add64((int64_t) 1, k_25300);
                
                // src/bgpc_vv_coloring.fut:131:11-32
                
                bool cond_25351 = slt64(tmp_25350, k_end_25293);
                
                // src/bgpc_vv_coloring.fut:131:24-32
                
                bool not_res_25352 = !lostzq_25349;
                
                // src/bgpc_vv_coloring.fut:166:5-169:10
                
                bool x_25353 = cond_25351 && not_res_25352;
                bool loop_while_tmp_26781 = x_25353;
                bool lost_tmp_26782 = lostzq_25349;
                int64_t k_tmp_26783 = tmp_25350;
                
                loop_while_25298 = loop_while_tmp_26781;
                lost_25299 = lost_tmp_26782;
                k_25300 = k_tmp_26783;
            }
            loses_conflict_vertex_res_25295 = loop_while_25298;
            loses_conflict_vertex_res_25296 = lost_25299;
            loses_conflict_vertex_res_25297 = k_25300;
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_f_res_25355 = btoi_bool_i64(loses_conflict_vertex_res_25296);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_op_res_24343 = add64(defunc_0_f_res_25355, scanacc_26406);
            
            ((int64_t *) mem_26621)[i_26410] = defunc_0_op_res_24343;
            ((int64_t *) mem_26623)[i_26410] = defunc_0_f_res_25355;
            ((bool *) mem_26624)[i_26410] = loses_conflict_vertex_res_25296;
            
            int64_t scanacc_tmp_26777 = defunc_0_op_res_24343;
            
            scanacc_26406 = scanacc_tmp_26777;
        }
        discard_26414 = scanacc_26406;
        // src/bgpc_vv_coloring.fut:23:25-28
        
        int64_t tmp_24327 = sub64(loop_dz2081Uz2083U_24132, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool y_24329 = slt64(tmp_24327, loop_dz2081Uz2083U_24132);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool x_24328 = sle64((int64_t) 0, tmp_24327);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool bounds_check_24330 = x_24328 && y_24329;
        
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        bool cond_24325 = loop_dz2081Uz2083U_24132 == (int64_t) 0;
        
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool protect_assert_disj_24331 = cond_24325 || bounds_check_24330;
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool index_certs_24332;
        
        if (!protect_assert_disj_24331) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_24327, "] out of bounds for array of shape [", (long long) loop_dz2081Uz2083U_24132, "].", "-> #0  src/bgpc_vv_coloring.fut:23:10-29\n   #1  src/bgpc_vv_coloring.fut:171:23-55\n   #2  src/bgpc_vv_coloring.fut:186:14-205:56\n   #3  src/bgpc_vv_coloring.fut:213:14-216:64\n   #4  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #5  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #6  src/sparse_jacobian_jvp.fut:158:18-33\n   #7  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool x_24326 = !cond_24325;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_f_res_24344;
        
        if (x_24326) {
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t x_26002 = ((int64_t *) mem_26621)[tmp_24327];
            
            m_f_res_24344 = x_26002;
        } else {
            m_f_res_24344 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_24346;
        
        if (cond_24325) {
            m_24346 = (int64_t) 0;
        } else {
            m_24346 = m_f_res_24344;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t bytes_26643 = (int64_t) 8 * m_24346;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26644_cached_sizze_26934 < bytes_26643) {
            err = lexical_realloc(ctx, &mem_26644, &mem_26644_cached_sizze_26934, bytes_26643);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26646_cached_sizze_26935 < bytes_26643) {
            err = lexical_realloc(ctx, &mem_26646, &mem_26646_cached_sizze_26935, bytes_26643);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:18:24-27:44
        if (mem_26648_cached_sizze_26936 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26648, &mem_26648_cached_sizze_26936, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26656_cached_sizze_26937 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26656, &mem_26656_cached_sizze_26937, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        bool acc_cert_25155;
        bool acc_cert_25156;
        
        // src/bgpc_vv_coloring.fut:18:24-27:44
        
        int64_t inpacc_26010;
        int64_t inpacc_25219 = (int64_t) 0;
        
        for (int64_t i_26441 = 0; i_26441 < loop_dz2081Uz2083U_24132; i_26441++) {
            bool eta_p_26503 = ((bool *) mem_26624)[i_26441];
            
            // src/bgpc_vv_coloring.fut:18:28-39
            
            int64_t bool_to_i64_res_26504 = btoi_bool_i64(eta_p_26503);
            int64_t eta_p_26516 = ((int64_t *) mem_26623)[i_26441];
            int64_t eta_p_26517 = ((int64_t *) mem_26621)[i_26441];
            int64_t v_26520 = ((int64_t *) mem_param_26598.mem)[i_26441];
            
            // src/bgpc_vv_coloring.fut:19:29-32
            
            int64_t defunc_0_op_res_26521 = add64(inpacc_25219, bool_to_i64_res_26504);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            bool cond_26522 = eta_p_26516 == (int64_t) 1;
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t lifted_lambda_res_26523;
            
            if (cond_26522) {
                // src/bgpc_vv_coloring.fut:27:5-44
                
                int64_t lifted_lambda_res_t_res_26524 = sub64(eta_p_26517, (int64_t) 1);
                
                lifted_lambda_res_26523 = lifted_lambda_res_t_res_26524;
            } else {
                lifted_lambda_res_26523 = (int64_t) -1;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_26523) && slt64(lifted_lambda_res_26523, m_24346)) {
                ((int64_t *) mem_26646)[lifted_lambda_res_26523] = v_26520;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_26523) && slt64(lifted_lambda_res_26523, m_24346)) {
                ((int64_t *) mem_26644)[lifted_lambda_res_26523] = defunc_0_op_res_26521;
            }
            ((int64_t *) mem_26648)[i_26441] = defunc_0_op_res_26521;
            
            int64_t inpacc_tmp_26787 = defunc_0_op_res_26521;
            
            inpacc_25219 = inpacc_tmp_26787;
        }
        inpacc_26010 = inpacc_25219;
        // src/bgpc_vv_coloring.fut:27:5-44
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_26656, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_26648, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {loop_dz2081Uz2083U_24132});
        // src/bgpc_vv_coloring.fut:23:10-29
        
        int64_t out_szz_f_res_24333;
        
        if (x_24326) {
            // src/bgpc_vv_coloring.fut:171:23-55
            
            int64_t x_26016 = ((int64_t *) mem_26656)[tmp_24327];
            
            out_szz_f_res_24333 = x_26016;
        } else {
            out_szz_f_res_24333 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        int64_t out_szz_24335;
        
        if (cond_24325) {
            out_szz_24335 = (int64_t) 0;
        } else {
            out_szz_24335 = out_szz_f_res_24333;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        
        int64_t bytes_26657 = (int64_t) 8 * out_szz_24335;
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        bool cond_24232 = slt64(defunc_0_reduce_res_26019, (int64_t) 0);
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        int64_t next_color_bound_24233;
        
        if (cond_24232) {
            next_color_bound_24233 = color_bound_24136;
        } else {
            // src/bgpc_vv_coloring.fut:110:40-46
            
            int64_t max_arg1_24234 = add64((int64_t) 2, defunc_0_reduce_res_26019);
            
            // src/bgpc_vv_coloring.fut:110:10-46
            
            int64_t max_res_24235 = smax64(color_bound_24136, max_arg1_24234);
            
            next_color_bound_24233 = max_res_24235;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        if (memblock_alloc(ctx, &mem_26658, bytes_26657, "mem_26658")) {
            err = 1;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        for (int64_t nest_i_26791 = 0; nest_i_26791 < out_szz_24335; nest_i_26791++) {
            ((int64_t *) mem_26658.mem)[nest_i_26791] = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:33:6-27
        
        bool acc_cert_24928;
        
        // src/bgpc_vv_coloring.fut:29:21-33:27
        for (int64_t i_26446 = 0; i_26446 < m_24346; i_26446++) {
            int64_t eta_p_24940 = ((int64_t *) mem_26644)[i_26446];
            int64_t v_24942 = ((int64_t *) mem_26646)[i_26446];
            
            // src/bgpc_vv_coloring.fut:29:44-50
            
            int64_t lifted_lambda_res_24943 = sub64(eta_p_24940, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:33:6-27
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_24943) && slt64(lifted_lambda_res_24943, out_szz_24335)) {
                ((int64_t *) mem_26658.mem)[lifted_lambda_res_24943] = v_24942;
            }
        }
        // src/bgpc_vv_coloring.fut:175:17-48
        
        bool acc_cert_24383;
        
        // src/bgpc_vv_coloring.fut:175:17-48
        for (int64_t i_26448 = 0; i_26448 < out_szz_24335; i_26448++) {
            int64_t v_24387 = ((int64_t *) mem_26658.mem)[i_26448];
            
            // src/bgpc_vv_coloring.fut:175:17-48
            // UpdateAcc
            if (sle64((int64_t) 0, v_24387) && slt64(v_24387, (int64_t) 5)) {
                ((int64_t *) mem_26588)[v_24387] = (int64_t) -1;
            }
        }
        // src/bgpc_vv_coloring.fut:198:20-23
        
        bool loop_cond_24389 = slt64((int64_t) 0, out_szz_24335);
        
        if (memblock_set(ctx, &mem_param_tmp_26763, &mem_26658, "mem_26658") != 0)
            return 1;
        
        int64_t loop_dz2081Uz2083U_tmp_26764 = out_szz_24335;
        bool loop_while_tmp_26765 = loop_cond_24389;
        int64_t color_bound_tmp_26768 = next_color_bound_24233;
        
        if (memblock_set(ctx, &mem_param_26598, &mem_param_tmp_26763, "mem_param_tmp_26763") != 0)
            return 1;
        loop_dz2081Uz2083U_24132 = loop_dz2081Uz2083U_tmp_26764;
        loop_while_24133 = loop_while_tmp_26765;
        color_bound_24136 = color_bound_tmp_26768;
    }
    if (memblock_set(ctx, &ext_mem_26663, &mem_param_26598, "mem_param_26598") != 0)
        return 1;
    vv_color_side_order_res_24127 = loop_dz2081Uz2083U_24132;
    vv_color_side_order_res_24128 = loop_while_24133;
    vv_color_side_order_res_24131 = color_bound_24136;
    if (memblock_unref(ctx, &mem_26587, "mem_26587") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26591, "ext_mem_26591") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26592, "ext_mem_26592") != 0)
        return 1;
    // src/sparse_jacobian_jvp.fut:107:8-112:40
    
    int64_t x_26031;
    int64_t redout_26449 = (int64_t) 0;
    
    for (int64_t i_26450 = 0; i_26450 < (int64_t) 5; i_26450++) {
        int64_t x_24394 = ((int64_t *) mem_26588)[i_26450];
        
        // src/sparse_jacobian_jvp.fut:14:22-29
        
        int64_t max_res_24397 = smax64(x_24394, redout_26449);
        int64_t redout_tmp_26794 = max_res_24397;
        
        redout_26449 = redout_tmp_26794;
    }
    x_26031 = redout_26449;
    // src/sparse_jacobian_jvp.fut:14:13-41
    
    int64_t num_colors_of_res_f_res_24398 = add64((int64_t) 1, x_26031);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool bounds_invalid_upwards_24400 = slt64(num_colors_of_res_f_res_24398, (int64_t) 0);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool valid_24401 = !bounds_invalid_upwards_24400;
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool range_valid_c_24402;
    
    if (!valid_24401) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Range ", (long long) (int64_t) 0, "..", (long long) (int64_t) 1, "..<", (long long) num_colors_of_res_f_res_24398, " is invalid.", "-> #0  src/sparse_jacobian_jvp.fut:29:11-18\n   #1  src/sparse_jacobian_jvp.fut:107:8-112:40\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    
    int64_t bytes_26666 = (int64_t) 24 * num_colors_of_res_f_res_24398;
    double zt_rhs_24491 = ((double *) x_mem_26563.mem)[(int64_t) 3];
    
    // test/test_sparse_jacobian_jvp.fut:19:26-32
    
    double binop_x_25868 = 0.0 * zt_rhs_24491;
    double zt_rhs_24494 = ((double *) x_mem_26563.mem)[(int64_t) 1];
    
    // test/test_sparse_jacobian_jvp.fut:20:19-25
    
    double binop_x_25875 = 0.0 * zt_rhs_24494;
    double zt_lhs_24496 = ((double *) x_mem_26563.mem)[(int64_t) 2];
    
    // src/dense_jacobian.fut:8:40-9:68
    if (mem_26564_cached_sizze_26924 < (int64_t) 120) {
        err = lexical_realloc(ctx, &mem_26564, &mem_26564_cached_sizze_26924, (int64_t) 120);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:5:3-21
    if (mem_26569_cached_sizze_26925 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_26569, &mem_26569_cached_sizze_26925, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:8:40-9:68
    for (int64_t i_26386 = 0; i_26386 < (int64_t) 5; i_26386++) {
        // src/dense_jacobian.fut:5:3-21
        for (int64_t nest_i_26796 = 0; nest_i_26796 < (int64_t) 5; nest_i_26796++) {
            ((double *) mem_26569)[nest_i_26796] = 0.0;
        }
        // src/dense_jacobian.fut:5:3-39
        ((double *) mem_26569)[i_26386] = 1.0;
        
        double zp_lhs_tan_25865 = ((double *) mem_26569)[(int64_t) 0];
        double zt_rhs_tan_25866 = ((double *) mem_26569)[(int64_t) 3];
        
        // test/test_sparse_jacobian_jvp.fut:19:26-32
        
        double binop_y_25869 = 2.0 * zt_rhs_tan_25866;
        
        // test/test_sparse_jacobian_jvp.fut:19:26-32
        
        double zp_rhs_tan_25867 = binop_x_25868 + binop_y_25869;
        
        // test/test_sparse_jacobian_jvp.fut:19:17-32
        
        double y0_tan_25870 = zp_lhs_tan_25865 + zp_rhs_tan_25867;
        double zt_rhs_tan_25873 = ((double *) mem_26569)[(int64_t) 1];
        
        // test/test_sparse_jacobian_jvp.fut:20:19-25
        
        double binop_y_25876 = 5.0 * zt_rhs_tan_25873;
        
        // test/test_sparse_jacobian_jvp.fut:20:19-25
        
        double y1_tan_25874 = binop_x_25875 + binop_y_25876;
        double zt_lhs_tan_25877 = ((double *) mem_26569)[(int64_t) 2];
        
        // test/test_sparse_jacobian_jvp.fut:21:17-23
        
        double binop_x_25879 = zt_lhs_24496 * zt_lhs_tan_25877;
        
        // test/test_sparse_jacobian_jvp.fut:21:17-23
        
        double y2_tan_25878 = binop_x_25879 + binop_x_25879;
        
        // test/test_sparse_jacobian_jvp.fut:36:59-64
        ((double *) mem_26564)[i_26386 * (int64_t) 3] = y0_tan_25870;
        ((double *) mem_26564)[i_26386 * (int64_t) 3 + (int64_t) 1] = y1_tan_25874;
        ((double *) mem_26564)[i_26386 * (int64_t) 3 + (int64_t) 2] = y2_tan_25878;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    if (mem_26578_cached_sizze_26926 < bytes_26577) {
        err = lexical_realloc(ctx, &mem_26578, &mem_26578_cached_sizze_26926, bytes_26577);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    for (int64_t nest_i_26797 = 0; nest_i_26797 < csr_bipartite_from_pattern_res_24087; nest_i_26797++) {
        ((double *) mem_26578)[nest_i_26797] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    if (mem_26667_cached_sizze_26938 < bytes_26666) {
        err = lexical_realloc(ctx, &mem_26667, &mem_26667_cached_sizze_26938, bytes_26666);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:17:3-61
    if (mem_26672_cached_sizze_26939 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_26672, &mem_26672_cached_sizze_26939, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    for (int64_t i_26457 = 0; i_26457 < num_colors_of_res_f_res_24398; i_26457++) {
        // src/sparse_jacobian_jvp.fut:17:3-61
        for (int64_t i_26453 = 0; i_26453 < (int64_t) 5; i_26453++) {
            int64_t eta_p_24411 = ((int64_t *) mem_26588)[i_26453];
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            bool cond_24412 = eta_p_24411 == i_26457;
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            double lifted_lambda_res_24413;
            
            if (cond_24412) {
                lifted_lambda_res_24413 = 1.0;
            } else {
                lifted_lambda_res_24413 = 0.0;
            }
            ((double *) mem_26672)[i_26453] = lifted_lambda_res_24413;
        }
        
        double zp_lhs_tan_25883 = ((double *) mem_26672)[(int64_t) 0];
        double zt_rhs_tan_25884 = ((double *) mem_26672)[(int64_t) 3];
        
        // test/test_sparse_jacobian_jvp.fut:19:26-32
        
        double binop_y_25887 = 2.0 * zt_rhs_tan_25884;
        
        // test/test_sparse_jacobian_jvp.fut:19:26-32
        
        double zp_rhs_tan_25885 = binop_x_25868 + binop_y_25887;
        
        // test/test_sparse_jacobian_jvp.fut:19:17-32
        
        double y0_tan_25888 = zp_lhs_tan_25883 + zp_rhs_tan_25885;
        double zt_rhs_tan_25891 = ((double *) mem_26672)[(int64_t) 1];
        
        // test/test_sparse_jacobian_jvp.fut:20:19-25
        
        double binop_y_25894 = 5.0 * zt_rhs_tan_25891;
        
        // test/test_sparse_jacobian_jvp.fut:20:19-25
        
        double y1_tan_25892 = binop_x_25875 + binop_y_25894;
        double zt_lhs_tan_25895 = ((double *) mem_26672)[(int64_t) 2];
        
        // test/test_sparse_jacobian_jvp.fut:21:17-23
        
        double binop_x_25897 = zt_lhs_24496 * zt_lhs_tan_25895;
        
        // test/test_sparse_jacobian_jvp.fut:21:17-23
        
        double y2_tan_25896 = binop_x_25897 + binop_x_25897;
        
        // test/test_sparse_jacobian_jvp.fut:37:33-38
        ((double *) mem_26667)[i_26457 * (int64_t) 3] = y0_tan_25888;
        ((double *) mem_26667)[i_26457 * (int64_t) 3 + (int64_t) 1] = y1_tan_25892;
        ((double *) mem_26667)[i_26457 * (int64_t) 3 + (int64_t) 2] = y2_tan_25896;
    }
    // src/sparse_jacobian_jvp.fut:45:5-52:27
    
    bool compressed_to_csr_vals_res_24422;
    int64_t compressed_to_csr_vals_res_24424;
    bool loop_while_24425;
    int64_t i_24427;
    
    loop_while_24425 = 1;
    i_24427 = (int64_t) 0;
    while (loop_while_24425) {
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool x_24428 = sle64((int64_t) 0, i_24427);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool y_24429 = slt64(i_24427, csr_bipartite_from_pattern_res_24086);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool bounds_check_24430 = x_24428 && y_24429;
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool index_certs_24431;
        
        if (!bounds_check_24430) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_24427, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/sparse_jacobian_jvp.fut:47:15-26\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        int64_t s_24432 = ((int64_t *) ext_mem_26576.mem)[i_24427];
        
        // src/sparse_jacobian_jvp.fut:48:25-27
        
        int64_t e_24433 = add64((int64_t) 1, i_24427);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool x_24434 = sle64((int64_t) 0, e_24433);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool y_24435 = slt64(e_24433, csr_bipartite_from_pattern_res_24086);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool bounds_check_24436 = x_24434 && y_24435;
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool index_certs_24437;
        
        if (!bounds_check_24436) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_24433, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/sparse_jacobian_jvp.fut:48:15-28\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        int64_t e_24438 = ((int64_t *) ext_mem_26576.mem)[e_24433];
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t j_m_i_24439 = sub64(e_24438, s_24432);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool empty_slice_24440 = j_m_i_24439 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t m_24441 = sub64(j_m_i_24439, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t i_p_m_t_s_24442 = add64(s_24432, m_24441);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_24443 = sle64((int64_t) 0, i_p_m_t_s_24442);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_24444 = slt64(i_p_m_t_s_24442, csr_bipartite_from_pattern_res_24087);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_lte_i_24445 = sle64((int64_t) 0, s_24432);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_lte_j_24446 = sle64(s_24432, e_24438);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_24447 = i_p_m_t_s_leq_w_24444 && zzero_lte_i_24445;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_24448 = zzero_leq_i_p_m_t_s_24443 && y_24447;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool forwards_ok_24449 = i_lte_j_24446 && y_24448;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool ok_or_empty_24450 = empty_slice_24440 || forwards_ok_24449;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool index_certs_24451;
        
        if (!ok_or_empty_24450) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_24432, ":", (long long) e_24438, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24087, "].", "-> #0  src/sparse_jacobian_jvp.fut:49:18-30\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool y_24453 = slt64(i_24427, (int64_t) 3);
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool bounds_check_24454 = x_24428 && y_24453;
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool index_certs_24455;
        
        if (!bounds_check_24454) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_24427, "] out of bounds for array of shape [", (long long) (int64_t) 3, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-45\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:18-51
        for (int64_t i_26461 = 0; i_26461 < j_m_i_24439; i_26461++) {
            int64_t index_primexp_26496 = s_24432 + i_26461;
            int64_t eta_p_24457 = ((int64_t *) ext_mem_26575.mem)[index_primexp_26496];
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool x_24458 = sle64((int64_t) 0, eta_p_24457);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool y_24459 = slt64(eta_p_24457, (int64_t) 5);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool bounds_check_24460 = x_24458 && y_24459;
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool index_certs_24461;
            
            if (!bounds_check_24460) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_24457, "] out of bounds for array of shape [", (long long) (int64_t) 5, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:32-41\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            int64_t tmp_24462 = ((int64_t *) mem_26588)[eta_p_24457];
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool x_24463 = sle64((int64_t) 0, tmp_24462);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool y_24464 = slt64(tmp_24462, num_colors_of_res_f_res_24398);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool bounds_check_24465 = x_24463 && y_24464;
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool index_certs_24466;
            
            if (!bounds_check_24465) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_24462, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_24398, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-42\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:18-51
            
            double lifted_lambda_res_24467 = ((double *) mem_26667)[tmp_24462 * (int64_t) 3 + i_24427];
            
            ((double *) mem_26578)[s_24432 + i_26461] = lifted_lambda_res_24467;
        }
        // src/sparse_jacobian_jvp.fut:46:13-16
        
        bool loop_cond_24469 = slt64(e_24433, (int64_t) 3);
        bool loop_while_tmp_26800 = loop_cond_24469;
        int64_t i_tmp_26802 = e_24433;
        
        loop_while_24425 = loop_while_tmp_26800;
        i_24427 = i_tmp_26802;
    }
    compressed_to_csr_vals_res_24422 = loop_while_24425;
    compressed_to_csr_vals_res_24424 = i_24427;
    // src/sparse_jacobian_jvp.fut:67:29-47
    if (mem_26698_cached_sizze_26940 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_26698, &mem_26698_cached_sizze_26940, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:8:19-10:39
    
    bool defunc_0_reduce_res_26033;
    bool redout_26467 = 1;
    
    for (int64_t i_26468 = 0; i_26468 < (int64_t) 3; i_26468++) {
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool y_24851 = slt64(i_26468, csr_bipartite_from_pattern_res_24086);
        
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool index_certs_24853;
        
        if (!y_24851) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_26468, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/sparse_jacobian_jvp.fut:63:17-28\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #3  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        int64_t s_24854 = ((int64_t *) ext_mem_26576.mem)[i_26468];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_lte_i_24867 = sle64((int64_t) 0, s_24854);
        
        // src/sparse_jacobian_jvp.fut:64:27-29
        
        int64_t e_24855 = add64((int64_t) 1, i_26468);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool y_24857 = slt64(e_24855, csr_bipartite_from_pattern_res_24086);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool x_24856 = sle64((int64_t) 0, e_24855);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool bounds_check_24858 = x_24856 && y_24857;
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool index_certs_24859;
        
        if (!bounds_check_24858) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_24855, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/sparse_jacobian_jvp.fut:64:17-30\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #3  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        int64_t e_24860 = ((int64_t *) ext_mem_26576.mem)[e_24855];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t j_m_i_24861 = sub64(e_24860, s_24854);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t m_24863 = sub64(j_m_i_24861, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t i_p_m_t_s_24864 = add64(s_24854, m_24863);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_p_m_t_s_leq_w_24866 = slt64(i_p_m_t_s_24864, csr_bipartite_from_pattern_res_24087);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24875 = i_p_m_t_s_leq_w_24866 && zzero_lte_i_24867;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_leq_i_p_m_t_s_24865 = sle64((int64_t) 0, i_p_m_t_s_24864);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24876 = zzero_leq_i_p_m_t_s_24865 && y_24875;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_lte_j_24868 = sle64(s_24854, e_24860);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool forwards_ok_24877 = i_lte_j_24868 && y_24876;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool empty_slice_24862 = j_m_i_24861 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool ok_or_empty_24878 = empty_slice_24862 || forwards_ok_24877;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool index_certs_24879;
        
        if (!ok_or_empty_24878) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_24854, ":", (long long) e_24860, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24087, "].", "-> #0  src/sparse_jacobian_jvp.fut:66:20-29\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #3  test/test_sparse_jacobian_jvp.fut:18:15-37:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:67:29-47
        for (int64_t nest_i_26805 = 0; nest_i_26805 < (int64_t) 5; nest_i_26805++) {
            ((double *) mem_26698)[nest_i_26805] = 0.0;
        }
        // src/sparse_jacobian_jvp.fut:68:12-32
        
        bool acc_cert_24883;
        
        // src/sparse_jacobian_jvp.fut:68:12-32
        for (int64_t i_26464 = 0; i_26464 < j_m_i_24861; i_26464++) {
            int64_t index_primexp_26493 = s_24854 + i_26464;
            int64_t v_24887 = ((int64_t *) ext_mem_26575.mem)[index_primexp_26493];
            double v_24888 = ((double *) mem_26578)[index_primexp_26493];
            
            // src/sparse_jacobian_jvp.fut:68:12-32
            // UpdateAcc
            if (sle64((int64_t) 0, v_24887) && slt64(v_24887, (int64_t) 5)) {
                ((double *) mem_26698)[v_24887] = v_24888;
            }
        }
        // test/test_sparse_jacobian_jvp.fut:9:5-66
        
        bool defunc_0_reduce_res_26028;
        bool redout_26465 = 1;
        
        for (int64_t i_26466 = 0; i_26466 < (int64_t) 5; i_26466++) {
            bool eta_p_24911 = ((bool *) mem_26545.mem)[i_26468 * (int64_t) 5 + i_26466];
            double eta_p_24912 = ((double *) mem_26564)[i_26466 * (int64_t) 3 + i_26468];
            double eta_p_24913 = ((double *) mem_26698)[i_26466];
            
            // test/test_sparse_jacobian_jvp.fut:14:25-48
            
            double lifted_lambda_res_24914;
            
            if (eta_p_24911) {
                lifted_lambda_res_24914 = eta_p_24912;
            } else {
                lifted_lambda_res_24914 = 0.0;
            }
            // test/test_sparse_jacobian_jvp.fut:9:48-51
            
            double abs_arg0_24916 = eta_p_24913 - lifted_lambda_res_24914;
            
            // test/test_sparse_jacobian_jvp.fut:9:37-51
            
            double abs_res_24917 = fabs64(abs_arg0_24916);
            
            // test/test_sparse_jacobian_jvp.fut:9:53-59
            
            bool lifted_lambda_res_24918 = abs_res_24917 <= 1.0e-9;
            
            // test/test_sparse_jacobian_jvp.fut:9:5-66
            
            bool x_24901 = lifted_lambda_res_24918 && redout_26465;
            bool redout_tmp_26807 = x_24901;
            
            redout_26465 = redout_tmp_26807;
        }
        defunc_0_reduce_res_26028 = redout_26465;
        // test/test_sparse_jacobian_jvp.fut:10:6-39
        
        bool x_24600 = defunc_0_reduce_res_26028 && redout_26467;
        bool redout_tmp_26804 = x_24600;
        
        redout_26467 = redout_tmp_26804;
    }
    defunc_0_reduce_res_26033 = redout_26467;
    if (memblock_unref(ctx, &ext_mem_26575, "ext_mem_26575") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26576, "ext_mem_26576") != 0)
        return 1;
    prim_out_26758 = defunc_0_reduce_res_26033;
    *out_prim_out_26923 = prim_out_26758;
    
  cleanup:
    {
        free(mem_26564);
        free(mem_26569);
        free(mem_26578);
        free(mem_26579);
        free(mem_26588);
        free(mem_26600);
        free(mem_26604);
        free(mem_26621);
        free(mem_26623);
        free(mem_26624);
        free(mem_26644);
        free(mem_26646);
        free(mem_26648);
        free(mem_26656);
        free(mem_26667);
        free(mem_26672);
        free(mem_26698);
        if (memblock_unref(ctx, &mem_param_tmp_26763, "mem_param_tmp_26763") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26658, "mem_26658") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_26598, "mem_param_26598") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26663, "ext_mem_26663") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26591, "ext_mem_26591") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26592, "ext_mem_26592") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26589, "mem_26589") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26587, "mem_26587") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26575, "ext_mem_26575") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26576, "ext_mem_26576") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_jvp_ex2_matches_dense(struct futhark_context *ctx, bool *out_prim_out_26941, struct memblock x_mem_26563)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_26564_cached_sizze_26942 = 0;
    unsigned char *mem_26564 = NULL;
    int64_t mem_26569_cached_sizze_26943 = 0;
    unsigned char *mem_26569 = NULL;
    int64_t mem_26578_cached_sizze_26944 = 0;
    unsigned char *mem_26578 = NULL;
    int64_t mem_26579_cached_sizze_26945 = 0;
    unsigned char *mem_26579 = NULL;
    int64_t mem_26588_cached_sizze_26946 = 0;
    unsigned char *mem_26588 = NULL;
    int64_t mem_26600_cached_sizze_26947 = 0;
    unsigned char *mem_26600 = NULL;
    int64_t mem_26604_cached_sizze_26948 = 0;
    unsigned char *mem_26604 = NULL;
    int64_t mem_26621_cached_sizze_26949 = 0;
    unsigned char *mem_26621 = NULL;
    int64_t mem_26623_cached_sizze_26950 = 0;
    unsigned char *mem_26623 = NULL;
    int64_t mem_26624_cached_sizze_26951 = 0;
    unsigned char *mem_26624 = NULL;
    int64_t mem_26644_cached_sizze_26952 = 0;
    unsigned char *mem_26644 = NULL;
    int64_t mem_26646_cached_sizze_26953 = 0;
    unsigned char *mem_26646 = NULL;
    int64_t mem_26648_cached_sizze_26954 = 0;
    unsigned char *mem_26648 = NULL;
    int64_t mem_26656_cached_sizze_26955 = 0;
    unsigned char *mem_26656 = NULL;
    int64_t mem_26667_cached_sizze_26956 = 0;
    unsigned char *mem_26667 = NULL;
    int64_t mem_26672_cached_sizze_26957 = 0;
    unsigned char *mem_26672 = NULL;
    int64_t mem_26698_cached_sizze_26958 = 0;
    unsigned char *mem_26698 = NULL;
    struct memblock mem_param_tmp_26763;
    
    mem_param_tmp_26763.references = NULL;
    
    struct memblock mem_26658;
    
    mem_26658.references = NULL;
    
    struct memblock mem_param_26598;
    
    mem_param_26598.references = NULL;
    
    struct memblock ext_mem_26663;
    
    ext_mem_26663.references = NULL;
    
    struct memblock ext_mem_26591;
    
    ext_mem_26591.references = NULL;
    
    struct memblock ext_mem_26592;
    
    ext_mem_26592.references = NULL;
    
    struct memblock mem_26589;
    
    mem_26589.references = NULL;
    
    struct memblock mem_26587;
    
    mem_26587.references = NULL;
    
    struct memblock ext_mem_26575;
    
    ext_mem_26575.references = NULL;
    
    struct memblock ext_mem_26576;
    
    ext_mem_26576.references = NULL;
    
    struct memblock mem_26545 = ctx->constants->mem_26545;
    struct memblock mem_26548 = ctx->constants->mem_26548;
    struct memblock mem_26550 = ctx->constants->mem_26550;
    struct memblock mem_26555 = ctx->constants->mem_26555;
    struct memblock mem_26556 = ctx->constants->mem_26556;
    struct memblock mem_26562 = ctx->constants->mem_26562;
    bool prim_out_26758;
    
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_24086;
    int64_t csr_bipartite_from_pattern_res_24087;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26576, &ext_mem_26575, &csr_bipartite_from_pattern_res_24086, &csr_bipartite_from_pattern_res_24087, mem_26548, (int64_t) 2, (int64_t) 4) != 0) {
        err = 1;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    
    int64_t bytes_26577 = (int64_t) 8 * csr_bipartite_from_pattern_res_24087;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (mem_26579_cached_sizze_26945 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_26579, &mem_26579_cached_sizze_26945, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t discard_26393;
    int64_t scanacc_26389 = (int64_t) 0;
    
    for (int64_t i_26391 = 0; i_26391 < (int64_t) 4; i_26391++) {
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t defunc_0_op_res_24102 = add64((int64_t) 1, scanacc_26389);
        
        ((int64_t *) mem_26579)[i_26391] = defunc_0_op_res_24102;
        
        int64_t scanacc_tmp_26759 = defunc_0_op_res_24102;
        
        scanacc_26389 = scanacc_tmp_26759;
    }
    discard_26393 = scanacc_26389;
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t x_24112 = ((int64_t *) mem_26579)[(int64_t) 3];
    
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    bool loop_cond_24126 = slt64((int64_t) 0, x_24112);
    
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t bytes_26586 = (int64_t) 8 * x_24112;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (memblock_alloc(ctx, &mem_26587, bytes_26586, "mem_26587")) {
        err = 1;
        goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    bool acc_cert_24821;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    for (int64_t i_26395 = 0; i_26395 < (int64_t) 4; i_26395++) {
        int64_t eta_p_24833 = ((int64_t *) mem_26579)[i_26395];
        
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t lifted_lambda_res_t_res_24836 = sub64(eta_p_24833, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:194:20-45
        // UpdateAcc
        if (sle64((int64_t) 0, lifted_lambda_res_t_res_24836) && slt64(lifted_lambda_res_t_res_24836, x_24112)) {
            ((int64_t *) mem_26587.mem)[lifted_lambda_res_t_res_24836] = i_26395;
        }
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    if (mem_26588_cached_sizze_26946 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_26588, &mem_26588_cached_sizze_26946, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    for (int64_t nest_i_26762 = 0; nest_i_26762 < (int64_t) 4; nest_i_26762++) {
        ((int64_t *) mem_26588)[nest_i_26762] = (int64_t) -1;
    }
    // src/pattern_csr.fut:25:3-39
    if (memblock_alloc(ctx, &mem_26589, (int64_t) 8, "mem_26589")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:25:3-39
    // src/pattern_csr.fut:25:3-39
    lmad_copy_1b(ctx, 2, (uint8_t *) mem_26589.mem, (int64_t) 0, (int64_t []) {(int64_t) 2, (int64_t) 1}, (uint8_t *) mem_26548.mem, (int64_t) 0, (int64_t []) {(int64_t) 1, (int64_t) 4}, (int64_t []) {(int64_t) 4, (int64_t) 2});
    // src/pattern_csr.fut:25:3-39
    
    int64_t csr_cols_from_pattern_res_24091;
    int64_t csr_cols_from_pattern_res_24092;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26592, &ext_mem_26591, &csr_cols_from_pattern_res_24091, &csr_cols_from_pattern_res_24092, mem_26589, (int64_t) 4, (int64_t) 2) != 0) {
        err = 1;
        goto cleanup;
    }
    if (memblock_unref(ctx, &mem_26589, "mem_26589") != 0)
        return 1;
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    int64_t vv_color_side_order_res_24127;
    bool vv_color_side_order_res_24128;
    int64_t vv_color_side_order_res_24131;
    int64_t loop_dz2081Uz2083U_24132;
    bool loop_while_24133;
    int64_t color_bound_24136;
    
    if (memblock_set(ctx, &mem_param_26598, &mem_26587, "mem_26587") != 0)
        return 1;
    loop_dz2081Uz2083U_24132 = x_24112;
    loop_while_24133 = loop_cond_24126;
    color_bound_24136 = (int64_t) 1;
    while (loop_while_24133) {
        // src/bgpc_vv_coloring.fut:62:5-64:15
        
        bool cond_24137 = slt64((int64_t) 0, color_bound_24136);
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t bytes_26599 = (int64_t) 8 * loop_dz2081Uz2083U_24132;
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        if (mem_26600_cached_sizze_26947 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26600, &mem_26600_cached_sizze_26947, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:77:24-51
        if (mem_26604_cached_sizze_26948 < color_bound_24136) {
            err = lexical_realloc(ctx, &mem_26604, &mem_26604_cached_sizze_26948, color_bound_24136);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t defunc_0_reduce_res_26019;
        int64_t redout_26397 = (int64_t) -1;
        
        for (int64_t i_26399 = 0; i_26399 < loop_dz2081Uz2083U_24132; i_26399++) {
            int64_t eta_p_25358 = ((int64_t *) mem_param_26598.mem)[i_26399];
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool x_25360 = sle64((int64_t) 0, eta_p_25358);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool y_25361 = slt64(eta_p_25358, csr_cols_from_pattern_res_24091);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool bounds_check_25362 = x_25360 && y_25361;
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool index_certs_25363;
            
            if (!bounds_check_25362) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25358, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24091, "].", "-> #0  src/bgpc_vv_coloring.fut:46:44-56\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:44-56
            
            int64_t seen_final_25364 = ((int64_t *) ext_mem_26592.mem)[eta_p_25358];
            
            // src/bgpc_vv_coloring.fut:46:70-72
            
            int64_t seen_final_25365 = add64((int64_t) 1, eta_p_25358);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool x_25366 = sle64((int64_t) 0, seen_final_25365);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool y_25367 = slt64(seen_final_25365, csr_cols_from_pattern_res_24091);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool bounds_check_25368 = x_25366 && y_25367;
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool index_certs_25369;
            
            if (!bounds_check_25368) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_25365, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24091, "].", "-> #0  src/bgpc_vv_coloring.fut:46:59-73\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:59-73
            
            int64_t seen_final_25370 = ((int64_t *) ext_mem_26592.mem)[seen_final_25365];
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t j_m_i_25371 = sub64(seen_final_25370, seen_final_25364);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool empty_slice_25372 = j_m_i_25371 == (int64_t) 0;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t m_25373 = sub64(j_m_i_25371, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t i_p_m_t_s_25374 = add64(seen_final_25364, m_25373);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_leq_i_p_m_t_s_25375 = sle64((int64_t) 0, i_p_m_t_s_25374);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_p_m_t_s_leq_w_25376 = slt64(i_p_m_t_s_25374, csr_cols_from_pattern_res_24092);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_lte_i_25377 = sle64((int64_t) 0, seen_final_25364);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_lte_j_25378 = sle64(seen_final_25364, seen_final_25370);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_25379 = i_p_m_t_s_leq_w_25376 && zzero_lte_i_25377;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_25380 = zzero_leq_i_p_m_t_s_25375 && y_25379;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool forwards_ok_25381 = i_lte_j_25378 && y_25380;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool ok_or_empty_25382 = empty_slice_25372 || forwards_ok_25381;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool index_certs_25383;
            
            if (!ok_or_empty_25382) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_25364, ":", (long long) seen_final_25370, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24092, "].", "-> #0  src/bgpc_vv_coloring.fut:46:5-54:19\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:77:24-51
            for (int64_t nest_i_26771 = 0; nest_i_26771 < color_bound_24136; nest_i_26771++) {
                ((bool *) mem_26604)[nest_i_26771] = 0;
            }
            // src/bgpc_vv_coloring.fut:46:5-54:19
            for (int64_t i_25385 = 0; i_25385 < j_m_i_25371; i_25385++) {
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t index_primexp_25387 = seen_final_25364 + i_25385;
                
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t v_25388 = ((int64_t *) ext_mem_26591.mem)[index_primexp_25387];
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool x_25389 = sle64((int64_t) 0, v_25388);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool y_25390 = slt64(v_25388, csr_bipartite_from_pattern_res_24086);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool bounds_check_25391 = x_25389 && y_25390;
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool index_certs_25392;
                
                if (!bounds_check_25391) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_25388, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/bgpc_vv_coloring.fut:49:26-37\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:26-37
                
                int64_t seen_acczq_25393 = ((int64_t *) ext_mem_26576.mem)[v_25388];
                
                // src/bgpc_vv_coloring.fut:49:50-52
                
                int64_t seen_acczq_25394 = add64((int64_t) 1, v_25388);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool x_25395 = sle64((int64_t) 0, seen_acczq_25394);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool y_25396 = slt64(seen_acczq_25394, csr_bipartite_from_pattern_res_24086);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool bounds_check_25397 = x_25395 && y_25396;
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool index_certs_25398;
                
                if (!bounds_check_25397) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_25394, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/bgpc_vv_coloring.fut:49:40-53\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:40-53
                
                int64_t seen_acczq_25399 = ((int64_t *) ext_mem_26576.mem)[seen_acczq_25394];
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t j_m_i_25400 = sub64(seen_acczq_25399, seen_acczq_25393);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool empty_slice_25401 = j_m_i_25400 == (int64_t) 0;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t m_25402 = sub64(j_m_i_25400, (int64_t) 1);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t i_p_m_t_s_25403 = add64(seen_acczq_25393, m_25402);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_leq_i_p_m_t_s_25404 = sle64((int64_t) 0, i_p_m_t_s_25403);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_p_m_t_s_leq_w_25405 = slt64(i_p_m_t_s_25403, csr_bipartite_from_pattern_res_24087);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_lte_i_25406 = sle64((int64_t) 0, seen_acczq_25393);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_lte_j_25407 = sle64(seen_acczq_25393, seen_acczq_25399);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_25408 = i_p_m_t_s_leq_w_25405 && zzero_lte_i_25406;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_25409 = zzero_leq_i_p_m_t_s_25404 && y_25408;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool forwards_ok_25410 = i_lte_j_25407 && y_25409;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool ok_or_empty_25411 = empty_slice_25401 || forwards_ok_25410;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool index_certs_25412;
                
                if (!ok_or_empty_25411) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_25393, ":", (long long) seen_acczq_25399, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24087, "].", "-> #0  src/bgpc_vv_coloring.fut:48:9-53:33\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:48:9-53:33
                for (int64_t i_25414 = 0; i_25414 < j_m_i_25400; i_25414++) {
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t index_primexp_25416 = seen_acczq_25393 + i_25414;
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t u_25417 = ((int64_t *) ext_mem_26575.mem)[index_primexp_25416];
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool x_25418 = sle64((int64_t) 0, u_25417);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool y_25419 = slt64(u_25417, (int64_t) 4);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool bounds_check_25420 = x_25418 && y_25419;
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool index_certs_25421;
                    
                    if (!bounds_check_25420) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_25417, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/bgpc_vv_coloring.fut:50:20-29\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    int64_t cu_25422 = ((int64_t *) mem_26588)[u_25417];
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_25423 = u_25417 == eta_p_25358;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_25424 = !cond_25423;
                    
                    // src/bgpc_vv_coloring.fut:51:30-37
                    
                    bool cond_t_res_25425 = sle64((int64_t) 0, cu_25422);
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    bool x_25426 = cond_25424 && cond_t_res_25425;
                    
                    // src/bgpc_vv_coloring.fut:51:44-67
                    
                    bool cond_t_res_25427 = slt64(cu_25422, color_bound_24136);
                    
                    // src/bgpc_vv_coloring.fut:94:14-103:69
                    
                    bool x_25428 = x_25426 && cond_t_res_25427;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    if (x_25428) {
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool bounds_check_25983 = cond_t_res_25425 && cond_t_res_25427;
                        
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool index_certs_25984;
                        
                        if (!bounds_check_25983) {
                            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) cu_25422, "] out of bounds for array of shape [", (long long) color_bound_24136, "].", "-> #0  src/bgpc_vv_coloring.fut:52:19-50\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                            err = FUTHARK_PROGRAM_ERROR;
                            goto cleanup;
                        }
                        // src/bgpc_vv_coloring.fut:52:19-50
                        ((bool *) mem_26604)[cu_25422] = 1;
                    }
                }
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool loop_cond_25433;
            
            if (cond_24137) {
                // src/bgpc_vv_coloring.fut:63:30-37
                
                bool loop_cond_t_res_25986 = ((bool *) mem_26604)[(int64_t) 0];
                
                loop_cond_25433 = loop_cond_t_res_25986;
            } else {
                loop_cond_25433 = 0;
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool c_final_25435;
            int64_t c_final_25436;
            bool loop_while_25437;
            int64_t c_25438;
            
            loop_while_25437 = loop_cond_25433;
            c_25438 = (int64_t) 0;
            while (loop_while_25437) {
                // src/bgpc_vv_coloring.fut:64:9-15
                
                int64_t loopres_25439 = add64((int64_t) 1, c_25438);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool cond_25440 = slt64(loopres_25439, color_bound_24136);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool loop_cond_25441;
                
                if (cond_25440) {
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool x_25987 = sle64((int64_t) 0, loopres_25439);
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool bounds_check_25988 = cond_25440 && x_25987;
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool index_certs_25989;
                    
                    if (!bounds_check_25988) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) loopres_25439, "] out of bounds for array of shape [", (long long) color_bound_24136, "].", "-> #0  src/bgpc_vv_coloring.fut:63:30-37\n   #1  src/bgpc_vv_coloring.fut:81:6-28\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool loop_cond_t_res_25990 = ((bool *) mem_26604)[loopres_25439];
                    
                    loop_cond_25441 = loop_cond_t_res_25990;
                } else {
                    loop_cond_25441 = 0;
                }
                
                bool loop_while_tmp_26774 = loop_cond_25441;
                int64_t c_tmp_26775 = loopres_25439;
                
                loop_while_25437 = loop_while_tmp_26774;
                c_25438 = c_tmp_26775;
            }
            c_final_25435 = loop_while_25437;
            c_final_25436 = c_25438;
            // src/bgpc_vv_coloring.fut:106:25-32
            
            int64_t max_res_24231 = smax64(c_final_25436, redout_26397);
            
            ((int64_t *) mem_26600)[i_26399] = c_final_25436;
            
            int64_t redout_tmp_26769 = max_res_24231;
            
            redout_26397 = redout_tmp_26769;
        }
        defunc_0_reduce_res_26019 = redout_26397;
        // src/bgpc_vv_coloring.fut:113:17-43
        
        bool acc_cert_24237;
        
        // src/bgpc_vv_coloring.fut:113:17-43
        for (int64_t i_26402 = 0; i_26402 < loop_dz2081Uz2083U_24132; i_26402++) {
            int64_t v_24241 = ((int64_t *) mem_param_26598.mem)[i_26402];
            int64_t v_24242 = ((int64_t *) mem_26600)[i_26402];
            
            // src/bgpc_vv_coloring.fut:113:17-43
            // UpdateAcc
            if (sle64((int64_t) 0, v_24241) && slt64(v_24241, (int64_t) 4)) {
                ((int64_t *) mem_26588)[v_24241] = v_24242;
            }
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26621_cached_sizze_26949 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26621, &mem_26621_cached_sizze_26949, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26623_cached_sizze_26950 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26623, &mem_26623_cached_sizze_26950, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26624_cached_sizze_26951 < loop_dz2081Uz2083U_24132) {
            err = lexical_realloc(ctx, &mem_26624, &mem_26624_cached_sizze_26951, loop_dz2081Uz2083U_24132);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t discard_26414;
        int64_t scanacc_26406 = (int64_t) 0;
        
        for (int64_t i_26410 = 0; i_26410 < loop_dz2081Uz2083U_24132; i_26410++) {
            int64_t eta_p_25282 = ((int64_t *) mem_param_26598.mem)[i_26410];
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool x_25283 = sle64((int64_t) 0, eta_p_25282);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool y_25284 = slt64(eta_p_25282, csr_cols_from_pattern_res_24091);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool bounds_check_25285 = x_25283 && y_25284;
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool index_certs_25286;
            
            if (!bounds_check_25285) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25282, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24091, "].", "-> #0  src/bgpc_vv_coloring.fut:126:15-27\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:127:26-28
            
            int64_t k_end_25288 = add64((int64_t) 1, eta_p_25282);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool x_25289 = sle64((int64_t) 0, k_end_25288);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool y_25290 = slt64(k_end_25288, csr_cols_from_pattern_res_24091);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool bounds_check_25291 = x_25289 && y_25290;
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool index_certs_25292;
            
            if (!bounds_check_25291) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_end_25288, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24091, "].", "-> #0  src/bgpc_vv_coloring.fut:127:15-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:126:15-27
            
            int64_t k0_25287 = ((int64_t *) ext_mem_26592.mem)[eta_p_25282];
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            int64_t k_end_25293 = ((int64_t *) ext_mem_26592.mem)[k_end_25288];
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool cond_25294 = slt64(k0_25287, k_end_25293);
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool loses_conflict_vertex_res_25295;
            bool loses_conflict_vertex_res_25296;
            int64_t loses_conflict_vertex_res_25297;
            bool loop_while_25298;
            bool lost_25299;
            int64_t k_25300;
            
            loop_while_25298 = cond_25294;
            lost_25299 = 0;
            k_25300 = k0_25287;
            while (loop_while_25298) {
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool x_25301 = sle64((int64_t) 0, k_25300);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool y_25302 = slt64(k_25300, csr_cols_from_pattern_res_24092);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool bounds_check_25303 = x_25301 && y_25302;
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool index_certs_25304;
                
                if (!bounds_check_25303) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_25300, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24092, "].", "-> #0  src/bgpc_vv_coloring.fut:132:15-26\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:132:15-26
                
                int64_t v_25305 = ((int64_t *) ext_mem_26591.mem)[k_25300];
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool x_25306 = sle64((int64_t) 0, v_25305);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool y_25307 = slt64(v_25305, csr_bipartite_from_pattern_res_24086);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool bounds_check_25308 = x_25306 && y_25307;
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool index_certs_25309;
                
                if (!bounds_check_25308) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_25305, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/bgpc_vv_coloring.fut:134:19-30\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:135:29-31
                
                int64_t t_end_25311 = add64((int64_t) 1, v_25305);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool x_25312 = sle64((int64_t) 0, t_end_25311);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool y_25313 = slt64(t_end_25311, csr_bipartite_from_pattern_res_24086);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool bounds_check_25314 = x_25312 && y_25313;
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool index_certs_25315;
                
                if (!bounds_check_25314) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_end_25311, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/bgpc_vv_coloring.fut:135:19-32\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:134:19-30
                
                int64_t t0_25310 = ((int64_t *) ext_mem_26576.mem)[v_25305];
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                int64_t t_end_25316 = ((int64_t *) ext_mem_26576.mem)[t_end_25311];
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool cond_25317 = slt64(t0_25310, t_end_25316);
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool loopres_25318;
                bool loopres_25319;
                int64_t loopres_25320;
                bool loop_while_25321;
                bool lost_in_net_25322;
                int64_t t_25323;
                
                loop_while_25321 = cond_25317;
                lost_in_net_25322 = 0;
                t_25323 = t0_25310;
                while (loop_while_25321) {
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool x_25324 = sle64((int64_t) 0, t_25323);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool y_25325 = slt64(t_25323, csr_bipartite_from_pattern_res_24087);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool bounds_check_25326 = x_25324 && y_25325;
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool index_certs_25327;
                    
                    if (!bounds_check_25326) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_25323, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24087, "].", "-> #0  src/bgpc_vv_coloring.fut:140:19-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    int64_t u_25328 = ((int64_t *) ext_mem_26575.mem)[t_25323];
                    
                    // src/bgpc_vv_coloring.fut:142:13-71
                    
                    bool lost_in_netzq_25329;
                    
                    if (lost_in_net_25322) {
                        lost_in_netzq_25329 = 1;
                    } else {
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25330 = u_25328 == eta_p_25282;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25331 = !cond_25330;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25332;
                        
                        if (cond_25331) {
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool x_25992 = sle64((int64_t) 0, u_25328);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool y_25993 = slt64(u_25328, (int64_t) 4);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool bounds_check_25994 = x_25992 && y_25993;
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool index_certs_25995;
                            
                            if (!bounds_check_25994) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_25328, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/bgpc_vv_coloring.fut:142:39-48\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool y_25997 = slt64(eta_p_25282, (int64_t) 4);
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool bounds_check_25998 = x_25283 && y_25997;
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool index_certs_25999;
                            
                            if (!bounds_check_25998) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25282, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/bgpc_vv_coloring.fut:142:52-61\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            int64_t zeze_lhs_25996 = ((int64_t *) mem_26588)[u_25328];
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            int64_t zeze_rhs_26000 = ((int64_t *) mem_26588)[eta_p_25282];
                            
                            // src/bgpc_vv_coloring.fut:142:49-61
                            
                            bool cond_t_res_26001 = zeze_lhs_25996 == zeze_rhs_26000;
                            
                            cond_25332 = cond_t_res_26001;
                        } else {
                            cond_25332 = 0;
                        }
                        // src/bgpc_vv_coloring.fut:142:67-70
                        
                        bool lost_in_netzq_f_res_t_res_25343 = slt64(u_25328, eta_p_25282);
                        
                        // src/bgpc_vv_coloring.fut:160:14-168:57
                        
                        bool x_25344 = cond_25332 && lost_in_netzq_f_res_t_res_25343;
                        
                        lost_in_netzq_25329 = x_25344;
                    }
                    // src/bgpc_vv_coloring.fut:143:31-37
                    
                    int64_t tmp_25345 = add64((int64_t) 1, t_25323);
                    
                    // src/bgpc_vv_coloring.fut:139:15-43
                    
                    bool cond_25346 = slt64(tmp_25345, t_end_25316);
                    
                    // src/bgpc_vv_coloring.fut:139:28-43
                    
                    bool not_res_25347 = !lost_in_netzq_25329;
                    
                    // src/bgpc_vv_coloring.fut:166:5-169:10
                    
                    bool x_25348 = cond_25346 && not_res_25347;
                    bool loop_while_tmp_26784 = x_25348;
                    bool lost_in_net_tmp_26785 = lost_in_netzq_25329;
                    int64_t t_tmp_26786 = tmp_25345;
                    
                    loop_while_25321 = loop_while_tmp_26784;
                    lost_in_net_25322 = lost_in_net_tmp_26785;
                    t_25323 = t_tmp_26786;
                }
                loopres_25318 = loop_while_25321;
                loopres_25319 = lost_in_net_25322;
                loopres_25320 = t_25323;
                // src/bgpc_vv_coloring.fut:160:14-168:57
                
                bool lostzq_25349 = lost_25299 || loopres_25319;
                
                // src/bgpc_vv_coloring.fut:146:20-26
                
                int64_t tmp_25350 = add64((int64_t) 1, k_25300);
                
                // src/bgpc_vv_coloring.fut:131:11-32
                
                bool cond_25351 = slt64(tmp_25350, k_end_25293);
                
                // src/bgpc_vv_coloring.fut:131:24-32
                
                bool not_res_25352 = !lostzq_25349;
                
                // src/bgpc_vv_coloring.fut:166:5-169:10
                
                bool x_25353 = cond_25351 && not_res_25352;
                bool loop_while_tmp_26781 = x_25353;
                bool lost_tmp_26782 = lostzq_25349;
                int64_t k_tmp_26783 = tmp_25350;
                
                loop_while_25298 = loop_while_tmp_26781;
                lost_25299 = lost_tmp_26782;
                k_25300 = k_tmp_26783;
            }
            loses_conflict_vertex_res_25295 = loop_while_25298;
            loses_conflict_vertex_res_25296 = lost_25299;
            loses_conflict_vertex_res_25297 = k_25300;
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_f_res_25355 = btoi_bool_i64(loses_conflict_vertex_res_25296);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_op_res_24343 = add64(defunc_0_f_res_25355, scanacc_26406);
            
            ((int64_t *) mem_26621)[i_26410] = defunc_0_op_res_24343;
            ((int64_t *) mem_26623)[i_26410] = defunc_0_f_res_25355;
            ((bool *) mem_26624)[i_26410] = loses_conflict_vertex_res_25296;
            
            int64_t scanacc_tmp_26777 = defunc_0_op_res_24343;
            
            scanacc_26406 = scanacc_tmp_26777;
        }
        discard_26414 = scanacc_26406;
        // src/bgpc_vv_coloring.fut:23:25-28
        
        int64_t tmp_24327 = sub64(loop_dz2081Uz2083U_24132, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool y_24329 = slt64(tmp_24327, loop_dz2081Uz2083U_24132);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool x_24328 = sle64((int64_t) 0, tmp_24327);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool bounds_check_24330 = x_24328 && y_24329;
        
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        bool cond_24325 = loop_dz2081Uz2083U_24132 == (int64_t) 0;
        
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool protect_assert_disj_24331 = cond_24325 || bounds_check_24330;
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool index_certs_24332;
        
        if (!protect_assert_disj_24331) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_24327, "] out of bounds for array of shape [", (long long) loop_dz2081Uz2083U_24132, "].", "-> #0  src/bgpc_vv_coloring.fut:23:10-29\n   #1  src/bgpc_vv_coloring.fut:171:23-55\n   #2  src/bgpc_vv_coloring.fut:186:14-205:56\n   #3  src/bgpc_vv_coloring.fut:213:14-216:64\n   #4  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #5  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #6  src/sparse_jacobian_jvp.fut:158:18-33\n   #7  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool x_24326 = !cond_24325;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_f_res_24344;
        
        if (x_24326) {
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t x_26002 = ((int64_t *) mem_26621)[tmp_24327];
            
            m_f_res_24344 = x_26002;
        } else {
            m_f_res_24344 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_24346;
        
        if (cond_24325) {
            m_24346 = (int64_t) 0;
        } else {
            m_24346 = m_f_res_24344;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t bytes_26643 = (int64_t) 8 * m_24346;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26644_cached_sizze_26952 < bytes_26643) {
            err = lexical_realloc(ctx, &mem_26644, &mem_26644_cached_sizze_26952, bytes_26643);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26646_cached_sizze_26953 < bytes_26643) {
            err = lexical_realloc(ctx, &mem_26646, &mem_26646_cached_sizze_26953, bytes_26643);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:18:24-27:44
        if (mem_26648_cached_sizze_26954 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26648, &mem_26648_cached_sizze_26954, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26656_cached_sizze_26955 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26656, &mem_26656_cached_sizze_26955, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        bool acc_cert_25155;
        bool acc_cert_25156;
        
        // src/bgpc_vv_coloring.fut:18:24-27:44
        
        int64_t inpacc_26010;
        int64_t inpacc_25219 = (int64_t) 0;
        
        for (int64_t i_26441 = 0; i_26441 < loop_dz2081Uz2083U_24132; i_26441++) {
            bool eta_p_26503 = ((bool *) mem_26624)[i_26441];
            
            // src/bgpc_vv_coloring.fut:18:28-39
            
            int64_t bool_to_i64_res_26504 = btoi_bool_i64(eta_p_26503);
            int64_t eta_p_26516 = ((int64_t *) mem_26623)[i_26441];
            int64_t eta_p_26517 = ((int64_t *) mem_26621)[i_26441];
            int64_t v_26520 = ((int64_t *) mem_param_26598.mem)[i_26441];
            
            // src/bgpc_vv_coloring.fut:19:29-32
            
            int64_t defunc_0_op_res_26521 = add64(inpacc_25219, bool_to_i64_res_26504);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            bool cond_26522 = eta_p_26516 == (int64_t) 1;
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t lifted_lambda_res_26523;
            
            if (cond_26522) {
                // src/bgpc_vv_coloring.fut:27:5-44
                
                int64_t lifted_lambda_res_t_res_26524 = sub64(eta_p_26517, (int64_t) 1);
                
                lifted_lambda_res_26523 = lifted_lambda_res_t_res_26524;
            } else {
                lifted_lambda_res_26523 = (int64_t) -1;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_26523) && slt64(lifted_lambda_res_26523, m_24346)) {
                ((int64_t *) mem_26646)[lifted_lambda_res_26523] = v_26520;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_26523) && slt64(lifted_lambda_res_26523, m_24346)) {
                ((int64_t *) mem_26644)[lifted_lambda_res_26523] = defunc_0_op_res_26521;
            }
            ((int64_t *) mem_26648)[i_26441] = defunc_0_op_res_26521;
            
            int64_t inpacc_tmp_26787 = defunc_0_op_res_26521;
            
            inpacc_25219 = inpacc_tmp_26787;
        }
        inpacc_26010 = inpacc_25219;
        // src/bgpc_vv_coloring.fut:27:5-44
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_26656, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_26648, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {loop_dz2081Uz2083U_24132});
        // src/bgpc_vv_coloring.fut:23:10-29
        
        int64_t out_szz_f_res_24333;
        
        if (x_24326) {
            // src/bgpc_vv_coloring.fut:171:23-55
            
            int64_t x_26016 = ((int64_t *) mem_26656)[tmp_24327];
            
            out_szz_f_res_24333 = x_26016;
        } else {
            out_szz_f_res_24333 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        int64_t out_szz_24335;
        
        if (cond_24325) {
            out_szz_24335 = (int64_t) 0;
        } else {
            out_szz_24335 = out_szz_f_res_24333;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        
        int64_t bytes_26657 = (int64_t) 8 * out_szz_24335;
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        bool cond_24232 = slt64(defunc_0_reduce_res_26019, (int64_t) 0);
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        int64_t next_color_bound_24233;
        
        if (cond_24232) {
            next_color_bound_24233 = color_bound_24136;
        } else {
            // src/bgpc_vv_coloring.fut:110:40-46
            
            int64_t max_arg1_24234 = add64((int64_t) 2, defunc_0_reduce_res_26019);
            
            // src/bgpc_vv_coloring.fut:110:10-46
            
            int64_t max_res_24235 = smax64(color_bound_24136, max_arg1_24234);
            
            next_color_bound_24233 = max_res_24235;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        if (memblock_alloc(ctx, &mem_26658, bytes_26657, "mem_26658")) {
            err = 1;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        for (int64_t nest_i_26791 = 0; nest_i_26791 < out_szz_24335; nest_i_26791++) {
            ((int64_t *) mem_26658.mem)[nest_i_26791] = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:33:6-27
        
        bool acc_cert_24928;
        
        // src/bgpc_vv_coloring.fut:29:21-33:27
        for (int64_t i_26446 = 0; i_26446 < m_24346; i_26446++) {
            int64_t eta_p_24940 = ((int64_t *) mem_26644)[i_26446];
            int64_t v_24942 = ((int64_t *) mem_26646)[i_26446];
            
            // src/bgpc_vv_coloring.fut:29:44-50
            
            int64_t lifted_lambda_res_24943 = sub64(eta_p_24940, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:33:6-27
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_24943) && slt64(lifted_lambda_res_24943, out_szz_24335)) {
                ((int64_t *) mem_26658.mem)[lifted_lambda_res_24943] = v_24942;
            }
        }
        // src/bgpc_vv_coloring.fut:175:17-48
        
        bool acc_cert_24383;
        
        // src/bgpc_vv_coloring.fut:175:17-48
        for (int64_t i_26448 = 0; i_26448 < out_szz_24335; i_26448++) {
            int64_t v_24387 = ((int64_t *) mem_26658.mem)[i_26448];
            
            // src/bgpc_vv_coloring.fut:175:17-48
            // UpdateAcc
            if (sle64((int64_t) 0, v_24387) && slt64(v_24387, (int64_t) 4)) {
                ((int64_t *) mem_26588)[v_24387] = (int64_t) -1;
            }
        }
        // src/bgpc_vv_coloring.fut:198:20-23
        
        bool loop_cond_24389 = slt64((int64_t) 0, out_szz_24335);
        
        if (memblock_set(ctx, &mem_param_tmp_26763, &mem_26658, "mem_26658") != 0)
            return 1;
        
        int64_t loop_dz2081Uz2083U_tmp_26764 = out_szz_24335;
        bool loop_while_tmp_26765 = loop_cond_24389;
        int64_t color_bound_tmp_26768 = next_color_bound_24233;
        
        if (memblock_set(ctx, &mem_param_26598, &mem_param_tmp_26763, "mem_param_tmp_26763") != 0)
            return 1;
        loop_dz2081Uz2083U_24132 = loop_dz2081Uz2083U_tmp_26764;
        loop_while_24133 = loop_while_tmp_26765;
        color_bound_24136 = color_bound_tmp_26768;
    }
    if (memblock_set(ctx, &ext_mem_26663, &mem_param_26598, "mem_param_26598") != 0)
        return 1;
    vv_color_side_order_res_24127 = loop_dz2081Uz2083U_24132;
    vv_color_side_order_res_24128 = loop_while_24133;
    vv_color_side_order_res_24131 = color_bound_24136;
    if (memblock_unref(ctx, &mem_26587, "mem_26587") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26591, "ext_mem_26591") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26592, "ext_mem_26592") != 0)
        return 1;
    // src/sparse_jacobian_jvp.fut:107:8-112:40
    
    int64_t x_26031;
    int64_t redout_26449 = (int64_t) 0;
    
    for (int64_t i_26450 = 0; i_26450 < (int64_t) 4; i_26450++) {
        int64_t x_24394 = ((int64_t *) mem_26588)[i_26450];
        
        // src/sparse_jacobian_jvp.fut:14:22-29
        
        int64_t max_res_24397 = smax64(x_24394, redout_26449);
        int64_t redout_tmp_26794 = max_res_24397;
        
        redout_26449 = redout_tmp_26794;
    }
    x_26031 = redout_26449;
    // src/sparse_jacobian_jvp.fut:14:13-41
    
    int64_t num_colors_of_res_f_res_24398 = add64((int64_t) 1, x_26031);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool bounds_invalid_upwards_24400 = slt64(num_colors_of_res_f_res_24398, (int64_t) 0);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool valid_24401 = !bounds_invalid_upwards_24400;
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool range_valid_c_24402;
    
    if (!valid_24401) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Range ", (long long) (int64_t) 0, "..", (long long) (int64_t) 1, "..<", (long long) num_colors_of_res_f_res_24398, " is invalid.", "-> #0  src/sparse_jacobian_jvp.fut:29:11-18\n   #1  src/sparse_jacobian_jvp.fut:107:8-112:40\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    
    int64_t bytes_26666 = (int64_t) 16 * num_colors_of_res_f_res_24398;
    double zt_rhs_24493 = ((double *) x_mem_26563.mem)[(int64_t) 2];
    
    // test/test_sparse_jacobian_jvp.fut:43:19-25
    
    double binop_x_25872 = 0.0 * zt_rhs_24493;
    
    // src/dense_jacobian.fut:8:40-9:68
    if (mem_26564_cached_sizze_26942 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_26564, &mem_26564_cached_sizze_26942, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:5:3-21
    if (mem_26569_cached_sizze_26943 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_26569, &mem_26569_cached_sizze_26943, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:8:40-9:68
    for (int64_t i_26386 = 0; i_26386 < (int64_t) 4; i_26386++) {
        // src/dense_jacobian.fut:5:3-21
        for (int64_t nest_i_26796 = 0; nest_i_26796 < (int64_t) 4; nest_i_26796++) {
            ((double *) mem_26569)[nest_i_26796] = 0.0;
        }
        // src/dense_jacobian.fut:5:3-39
        ((double *) mem_26569)[i_26386] = 1.0;
        
        double zp_lhs_tan_25865 = ((double *) mem_26569)[(int64_t) 0];
        double zp_rhs_tan_25866 = ((double *) mem_26569)[(int64_t) 1];
        
        // test/test_sparse_jacobian_jvp.fut:42:17-23
        
        double y0_tan_25867 = zp_lhs_tan_25865 + zp_rhs_tan_25866;
        double zt_rhs_tan_25870 = ((double *) mem_26569)[(int64_t) 2];
        
        // test/test_sparse_jacobian_jvp.fut:43:19-25
        
        double binop_y_25873 = 7.0 * zt_rhs_tan_25870;
        
        // test/test_sparse_jacobian_jvp.fut:43:19-25
        
        double y1_tan_25871 = binop_x_25872 + binop_y_25873;
        
        // test/test_sparse_jacobian_jvp.fut:57:59-64
        ((double *) mem_26564)[i_26386 * (int64_t) 2] = y0_tan_25867;
        ((double *) mem_26564)[i_26386 * (int64_t) 2 + (int64_t) 1] = y1_tan_25871;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    if (mem_26578_cached_sizze_26944 < bytes_26577) {
        err = lexical_realloc(ctx, &mem_26578, &mem_26578_cached_sizze_26944, bytes_26577);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    for (int64_t nest_i_26797 = 0; nest_i_26797 < csr_bipartite_from_pattern_res_24087; nest_i_26797++) {
        ((double *) mem_26578)[nest_i_26797] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    if (mem_26667_cached_sizze_26956 < bytes_26666) {
        err = lexical_realloc(ctx, &mem_26667, &mem_26667_cached_sizze_26956, bytes_26666);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:17:3-61
    if (mem_26672_cached_sizze_26957 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_26672, &mem_26672_cached_sizze_26957, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    for (int64_t i_26457 = 0; i_26457 < num_colors_of_res_f_res_24398; i_26457++) {
        // src/sparse_jacobian_jvp.fut:17:3-61
        for (int64_t i_26453 = 0; i_26453 < (int64_t) 4; i_26453++) {
            int64_t eta_p_24411 = ((int64_t *) mem_26588)[i_26453];
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            bool cond_24412 = eta_p_24411 == i_26457;
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            double lifted_lambda_res_24413;
            
            if (cond_24412) {
                lifted_lambda_res_24413 = 1.0;
            } else {
                lifted_lambda_res_24413 = 0.0;
            }
            ((double *) mem_26672)[i_26453] = lifted_lambda_res_24413;
        }
        
        double zp_lhs_tan_25876 = ((double *) mem_26672)[(int64_t) 0];
        double zp_rhs_tan_25877 = ((double *) mem_26672)[(int64_t) 1];
        
        // test/test_sparse_jacobian_jvp.fut:42:17-23
        
        double y0_tan_25878 = zp_lhs_tan_25876 + zp_rhs_tan_25877;
        double zt_rhs_tan_25881 = ((double *) mem_26672)[(int64_t) 2];
        
        // test/test_sparse_jacobian_jvp.fut:43:19-25
        
        double binop_y_25884 = 7.0 * zt_rhs_tan_25881;
        
        // test/test_sparse_jacobian_jvp.fut:43:19-25
        
        double y1_tan_25882 = binop_x_25872 + binop_y_25884;
        
        // test/test_sparse_jacobian_jvp.fut:58:33-38
        ((double *) mem_26667)[i_26457 * (int64_t) 2] = y0_tan_25878;
        ((double *) mem_26667)[i_26457 * (int64_t) 2 + (int64_t) 1] = y1_tan_25882;
    }
    // src/sparse_jacobian_jvp.fut:45:5-52:27
    
    bool compressed_to_csr_vals_res_24422;
    int64_t compressed_to_csr_vals_res_24424;
    bool loop_while_24425;
    int64_t i_24427;
    
    loop_while_24425 = 1;
    i_24427 = (int64_t) 0;
    while (loop_while_24425) {
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool x_24428 = sle64((int64_t) 0, i_24427);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool y_24429 = slt64(i_24427, csr_bipartite_from_pattern_res_24086);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool bounds_check_24430 = x_24428 && y_24429;
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool index_certs_24431;
        
        if (!bounds_check_24430) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_24427, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/sparse_jacobian_jvp.fut:47:15-26\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        int64_t s_24432 = ((int64_t *) ext_mem_26576.mem)[i_24427];
        
        // src/sparse_jacobian_jvp.fut:48:25-27
        
        int64_t e_24433 = add64((int64_t) 1, i_24427);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool x_24434 = sle64((int64_t) 0, e_24433);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool y_24435 = slt64(e_24433, csr_bipartite_from_pattern_res_24086);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool bounds_check_24436 = x_24434 && y_24435;
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool index_certs_24437;
        
        if (!bounds_check_24436) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_24433, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/sparse_jacobian_jvp.fut:48:15-28\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        int64_t e_24438 = ((int64_t *) ext_mem_26576.mem)[e_24433];
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t j_m_i_24439 = sub64(e_24438, s_24432);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool empty_slice_24440 = j_m_i_24439 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t m_24441 = sub64(j_m_i_24439, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t i_p_m_t_s_24442 = add64(s_24432, m_24441);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_24443 = sle64((int64_t) 0, i_p_m_t_s_24442);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_24444 = slt64(i_p_m_t_s_24442, csr_bipartite_from_pattern_res_24087);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_lte_i_24445 = sle64((int64_t) 0, s_24432);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_lte_j_24446 = sle64(s_24432, e_24438);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_24447 = i_p_m_t_s_leq_w_24444 && zzero_lte_i_24445;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_24448 = zzero_leq_i_p_m_t_s_24443 && y_24447;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool forwards_ok_24449 = i_lte_j_24446 && y_24448;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool ok_or_empty_24450 = empty_slice_24440 || forwards_ok_24449;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool index_certs_24451;
        
        if (!ok_or_empty_24450) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_24432, ":", (long long) e_24438, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24087, "].", "-> #0  src/sparse_jacobian_jvp.fut:49:18-30\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool y_24453 = slt64(i_24427, (int64_t) 2);
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool bounds_check_24454 = x_24428 && y_24453;
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool index_certs_24455;
        
        if (!bounds_check_24454) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_24427, "] out of bounds for array of shape [", (long long) (int64_t) 2, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-45\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:18-51
        for (int64_t i_26461 = 0; i_26461 < j_m_i_24439; i_26461++) {
            int64_t index_primexp_26496 = s_24432 + i_26461;
            int64_t eta_p_24457 = ((int64_t *) ext_mem_26575.mem)[index_primexp_26496];
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool x_24458 = sle64((int64_t) 0, eta_p_24457);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool y_24459 = slt64(eta_p_24457, (int64_t) 4);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool bounds_check_24460 = x_24458 && y_24459;
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool index_certs_24461;
            
            if (!bounds_check_24460) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_24457, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:32-41\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            int64_t tmp_24462 = ((int64_t *) mem_26588)[eta_p_24457];
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool x_24463 = sle64((int64_t) 0, tmp_24462);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool y_24464 = slt64(tmp_24462, num_colors_of_res_f_res_24398);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool bounds_check_24465 = x_24463 && y_24464;
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool index_certs_24466;
            
            if (!bounds_check_24465) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_24462, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_24398, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-42\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:18-51
            
            double lifted_lambda_res_24467 = ((double *) mem_26667)[tmp_24462 * (int64_t) 2 + i_24427];
            
            ((double *) mem_26578)[s_24432 + i_26461] = lifted_lambda_res_24467;
        }
        // src/sparse_jacobian_jvp.fut:46:13-16
        
        bool loop_cond_24469 = slt64(e_24433, (int64_t) 2);
        bool loop_while_tmp_26800 = loop_cond_24469;
        int64_t i_tmp_26802 = e_24433;
        
        loop_while_24425 = loop_while_tmp_26800;
        i_24427 = i_tmp_26802;
    }
    compressed_to_csr_vals_res_24422 = loop_while_24425;
    compressed_to_csr_vals_res_24424 = i_24427;
    // src/sparse_jacobian_jvp.fut:67:29-47
    if (mem_26698_cached_sizze_26958 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_26698, &mem_26698_cached_sizze_26958, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:8:19-10:39
    
    bool defunc_0_reduce_res_26033;
    bool redout_26467 = 1;
    
    for (int64_t i_26468 = 0; i_26468 < (int64_t) 2; i_26468++) {
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool y_24851 = slt64(i_26468, csr_bipartite_from_pattern_res_24086);
        
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool index_certs_24853;
        
        if (!y_24851) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_26468, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/sparse_jacobian_jvp.fut:63:17-28\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #3  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        int64_t s_24854 = ((int64_t *) ext_mem_26576.mem)[i_26468];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_lte_i_24867 = sle64((int64_t) 0, s_24854);
        
        // src/sparse_jacobian_jvp.fut:64:27-29
        
        int64_t e_24855 = add64((int64_t) 1, i_26468);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool y_24857 = slt64(e_24855, csr_bipartite_from_pattern_res_24086);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool x_24856 = sle64((int64_t) 0, e_24855);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool bounds_check_24858 = x_24856 && y_24857;
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool index_certs_24859;
        
        if (!bounds_check_24858) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_24855, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/sparse_jacobian_jvp.fut:64:17-30\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #3  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        int64_t e_24860 = ((int64_t *) ext_mem_26576.mem)[e_24855];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t j_m_i_24861 = sub64(e_24860, s_24854);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t m_24863 = sub64(j_m_i_24861, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t i_p_m_t_s_24864 = add64(s_24854, m_24863);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_p_m_t_s_leq_w_24866 = slt64(i_p_m_t_s_24864, csr_bipartite_from_pattern_res_24087);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24875 = i_p_m_t_s_leq_w_24866 && zzero_lte_i_24867;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_leq_i_p_m_t_s_24865 = sle64((int64_t) 0, i_p_m_t_s_24864);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24876 = zzero_leq_i_p_m_t_s_24865 && y_24875;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_lte_j_24868 = sle64(s_24854, e_24860);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool forwards_ok_24877 = i_lte_j_24868 && y_24876;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool empty_slice_24862 = j_m_i_24861 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool ok_or_empty_24878 = empty_slice_24862 || forwards_ok_24877;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool index_certs_24879;
        
        if (!ok_or_empty_24878) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_24854, ":", (long long) e_24860, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24087, "].", "-> #0  src/sparse_jacobian_jvp.fut:66:20-29\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #3  test/test_sparse_jacobian_jvp.fut:41:15-58:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:67:29-47
        for (int64_t nest_i_26805 = 0; nest_i_26805 < (int64_t) 4; nest_i_26805++) {
            ((double *) mem_26698)[nest_i_26805] = 0.0;
        }
        // src/sparse_jacobian_jvp.fut:68:12-32
        
        bool acc_cert_24883;
        
        // src/sparse_jacobian_jvp.fut:68:12-32
        for (int64_t i_26464 = 0; i_26464 < j_m_i_24861; i_26464++) {
            int64_t index_primexp_26493 = s_24854 + i_26464;
            int64_t v_24887 = ((int64_t *) ext_mem_26575.mem)[index_primexp_26493];
            double v_24888 = ((double *) mem_26578)[index_primexp_26493];
            
            // src/sparse_jacobian_jvp.fut:68:12-32
            // UpdateAcc
            if (sle64((int64_t) 0, v_24887) && slt64(v_24887, (int64_t) 4)) {
                ((double *) mem_26698)[v_24887] = v_24888;
            }
        }
        // test/test_sparse_jacobian_jvp.fut:9:5-66
        
        bool defunc_0_reduce_res_26028;
        bool redout_26465 = 1;
        
        for (int64_t i_26466 = 0; i_26466 < (int64_t) 4; i_26466++) {
            bool eta_p_24911 = ((bool *) mem_26548.mem)[i_26468 * (int64_t) 4 + i_26466];
            double eta_p_24912 = ((double *) mem_26564)[i_26466 * (int64_t) 2 + i_26468];
            double eta_p_24913 = ((double *) mem_26698)[i_26466];
            
            // test/test_sparse_jacobian_jvp.fut:14:25-48
            
            double lifted_lambda_res_24914;
            
            if (eta_p_24911) {
                lifted_lambda_res_24914 = eta_p_24912;
            } else {
                lifted_lambda_res_24914 = 0.0;
            }
            // test/test_sparse_jacobian_jvp.fut:9:48-51
            
            double abs_arg0_24916 = eta_p_24913 - lifted_lambda_res_24914;
            
            // test/test_sparse_jacobian_jvp.fut:9:37-51
            
            double abs_res_24917 = fabs64(abs_arg0_24916);
            
            // test/test_sparse_jacobian_jvp.fut:9:53-59
            
            bool lifted_lambda_res_24918 = abs_res_24917 <= 1.0e-9;
            
            // test/test_sparse_jacobian_jvp.fut:9:5-66
            
            bool x_24901 = lifted_lambda_res_24918 && redout_26465;
            bool redout_tmp_26807 = x_24901;
            
            redout_26465 = redout_tmp_26807;
        }
        defunc_0_reduce_res_26028 = redout_26465;
        // test/test_sparse_jacobian_jvp.fut:10:6-39
        
        bool x_24592 = defunc_0_reduce_res_26028 && redout_26467;
        bool redout_tmp_26804 = x_24592;
        
        redout_26467 = redout_tmp_26804;
    }
    defunc_0_reduce_res_26033 = redout_26467;
    if (memblock_unref(ctx, &ext_mem_26575, "ext_mem_26575") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26576, "ext_mem_26576") != 0)
        return 1;
    prim_out_26758 = defunc_0_reduce_res_26033;
    *out_prim_out_26941 = prim_out_26758;
    
  cleanup:
    {
        free(mem_26564);
        free(mem_26569);
        free(mem_26578);
        free(mem_26579);
        free(mem_26588);
        free(mem_26600);
        free(mem_26604);
        free(mem_26621);
        free(mem_26623);
        free(mem_26624);
        free(mem_26644);
        free(mem_26646);
        free(mem_26648);
        free(mem_26656);
        free(mem_26667);
        free(mem_26672);
        free(mem_26698);
        if (memblock_unref(ctx, &mem_param_tmp_26763, "mem_param_tmp_26763") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26658, "mem_26658") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_26598, "mem_param_26598") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26663, "ext_mem_26663") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26591, "ext_mem_26591") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26592, "ext_mem_26592") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26589, "mem_26589") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26587, "mem_26587") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26575, "ext_mem_26575") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26576, "ext_mem_26576") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_jvp_ex4_matches_dense(struct futhark_context *ctx, bool *out_prim_out_26959, struct memblock x_mem_26563)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_26564_cached_sizze_26960 = 0;
    unsigned char *mem_26564 = NULL;
    int64_t mem_26569_cached_sizze_26961 = 0;
    unsigned char *mem_26569 = NULL;
    int64_t mem_26578_cached_sizze_26962 = 0;
    unsigned char *mem_26578 = NULL;
    int64_t mem_26579_cached_sizze_26963 = 0;
    unsigned char *mem_26579 = NULL;
    int64_t mem_26588_cached_sizze_26964 = 0;
    unsigned char *mem_26588 = NULL;
    int64_t mem_26600_cached_sizze_26965 = 0;
    unsigned char *mem_26600 = NULL;
    int64_t mem_26604_cached_sizze_26966 = 0;
    unsigned char *mem_26604 = NULL;
    int64_t mem_26621_cached_sizze_26967 = 0;
    unsigned char *mem_26621 = NULL;
    int64_t mem_26623_cached_sizze_26968 = 0;
    unsigned char *mem_26623 = NULL;
    int64_t mem_26624_cached_sizze_26969 = 0;
    unsigned char *mem_26624 = NULL;
    int64_t mem_26644_cached_sizze_26970 = 0;
    unsigned char *mem_26644 = NULL;
    int64_t mem_26646_cached_sizze_26971 = 0;
    unsigned char *mem_26646 = NULL;
    int64_t mem_26648_cached_sizze_26972 = 0;
    unsigned char *mem_26648 = NULL;
    int64_t mem_26656_cached_sizze_26973 = 0;
    unsigned char *mem_26656 = NULL;
    int64_t mem_26667_cached_sizze_26974 = 0;
    unsigned char *mem_26667 = NULL;
    int64_t mem_26672_cached_sizze_26975 = 0;
    unsigned char *mem_26672 = NULL;
    int64_t mem_26698_cached_sizze_26976 = 0;
    unsigned char *mem_26698 = NULL;
    struct memblock mem_param_tmp_26763;
    
    mem_param_tmp_26763.references = NULL;
    
    struct memblock mem_26658;
    
    mem_26658.references = NULL;
    
    struct memblock mem_param_26598;
    
    mem_param_26598.references = NULL;
    
    struct memblock ext_mem_26663;
    
    ext_mem_26663.references = NULL;
    
    struct memblock ext_mem_26591;
    
    ext_mem_26591.references = NULL;
    
    struct memblock ext_mem_26592;
    
    ext_mem_26592.references = NULL;
    
    struct memblock mem_26589;
    
    mem_26589.references = NULL;
    
    struct memblock mem_26587;
    
    mem_26587.references = NULL;
    
    struct memblock ext_mem_26575;
    
    ext_mem_26575.references = NULL;
    
    struct memblock ext_mem_26576;
    
    ext_mem_26576.references = NULL;
    
    struct memblock mem_26545 = ctx->constants->mem_26545;
    struct memblock mem_26548 = ctx->constants->mem_26548;
    struct memblock mem_26550 = ctx->constants->mem_26550;
    struct memblock mem_26555 = ctx->constants->mem_26555;
    struct memblock mem_26556 = ctx->constants->mem_26556;
    struct memblock mem_26562 = ctx->constants->mem_26562;
    bool prim_out_26758;
    
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_24086;
    int64_t csr_bipartite_from_pattern_res_24087;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26576, &ext_mem_26575, &csr_bipartite_from_pattern_res_24086, &csr_bipartite_from_pattern_res_24087, mem_26555, (int64_t) 4, (int64_t) 6) != 0) {
        err = 1;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    
    int64_t bytes_26577 = (int64_t) 8 * csr_bipartite_from_pattern_res_24087;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (mem_26579_cached_sizze_26963 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26579, &mem_26579_cached_sizze_26963, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t discard_26393;
    int64_t scanacc_26389 = (int64_t) 0;
    
    for (int64_t i_26391 = 0; i_26391 < (int64_t) 6; i_26391++) {
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t defunc_0_op_res_24102 = add64((int64_t) 1, scanacc_26389);
        
        ((int64_t *) mem_26579)[i_26391] = defunc_0_op_res_24102;
        
        int64_t scanacc_tmp_26759 = defunc_0_op_res_24102;
        
        scanacc_26389 = scanacc_tmp_26759;
    }
    discard_26393 = scanacc_26389;
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t x_24112 = ((int64_t *) mem_26579)[(int64_t) 5];
    
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    bool loop_cond_24126 = slt64((int64_t) 0, x_24112);
    
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t bytes_26586 = (int64_t) 8 * x_24112;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (memblock_alloc(ctx, &mem_26587, bytes_26586, "mem_26587")) {
        err = 1;
        goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    bool acc_cert_24821;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    for (int64_t i_26395 = 0; i_26395 < (int64_t) 6; i_26395++) {
        int64_t eta_p_24833 = ((int64_t *) mem_26579)[i_26395];
        
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t lifted_lambda_res_t_res_24836 = sub64(eta_p_24833, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:194:20-45
        // UpdateAcc
        if (sle64((int64_t) 0, lifted_lambda_res_t_res_24836) && slt64(lifted_lambda_res_t_res_24836, x_24112)) {
            ((int64_t *) mem_26587.mem)[lifted_lambda_res_t_res_24836] = i_26395;
        }
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    if (mem_26588_cached_sizze_26964 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26588, &mem_26588_cached_sizze_26964, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    for (int64_t nest_i_26762 = 0; nest_i_26762 < (int64_t) 6; nest_i_26762++) {
        ((int64_t *) mem_26588)[nest_i_26762] = (int64_t) -1;
    }
    // src/pattern_csr.fut:25:3-39
    if (memblock_alloc(ctx, &mem_26589, (int64_t) 24, "mem_26589")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:25:3-39
    // src/pattern_csr.fut:25:3-39
    lmad_copy_1b(ctx, 2, (uint8_t *) mem_26589.mem, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint8_t *) mem_26555.mem, (int64_t) 0, (int64_t []) {(int64_t) 1, (int64_t) 6}, (int64_t []) {(int64_t) 6, (int64_t) 4});
    // src/pattern_csr.fut:25:3-39
    
    int64_t csr_cols_from_pattern_res_24091;
    int64_t csr_cols_from_pattern_res_24092;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26592, &ext_mem_26591, &csr_cols_from_pattern_res_24091, &csr_cols_from_pattern_res_24092, mem_26589, (int64_t) 6, (int64_t) 4) != 0) {
        err = 1;
        goto cleanup;
    }
    if (memblock_unref(ctx, &mem_26589, "mem_26589") != 0)
        return 1;
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    int64_t vv_color_side_order_res_24127;
    bool vv_color_side_order_res_24128;
    int64_t vv_color_side_order_res_24131;
    int64_t loop_dz2081Uz2083U_24132;
    bool loop_while_24133;
    int64_t color_bound_24136;
    
    if (memblock_set(ctx, &mem_param_26598, &mem_26587, "mem_26587") != 0)
        return 1;
    loop_dz2081Uz2083U_24132 = x_24112;
    loop_while_24133 = loop_cond_24126;
    color_bound_24136 = (int64_t) 1;
    while (loop_while_24133) {
        // src/bgpc_vv_coloring.fut:62:5-64:15
        
        bool cond_24137 = slt64((int64_t) 0, color_bound_24136);
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t bytes_26599 = (int64_t) 8 * loop_dz2081Uz2083U_24132;
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        if (mem_26600_cached_sizze_26965 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26600, &mem_26600_cached_sizze_26965, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:77:24-51
        if (mem_26604_cached_sizze_26966 < color_bound_24136) {
            err = lexical_realloc(ctx, &mem_26604, &mem_26604_cached_sizze_26966, color_bound_24136);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t defunc_0_reduce_res_26019;
        int64_t redout_26397 = (int64_t) -1;
        
        for (int64_t i_26399 = 0; i_26399 < loop_dz2081Uz2083U_24132; i_26399++) {
            int64_t eta_p_25358 = ((int64_t *) mem_param_26598.mem)[i_26399];
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool x_25360 = sle64((int64_t) 0, eta_p_25358);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool y_25361 = slt64(eta_p_25358, csr_cols_from_pattern_res_24091);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool bounds_check_25362 = x_25360 && y_25361;
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool index_certs_25363;
            
            if (!bounds_check_25362) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25358, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24091, "].", "-> #0  src/bgpc_vv_coloring.fut:46:44-56\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:44-56
            
            int64_t seen_final_25364 = ((int64_t *) ext_mem_26592.mem)[eta_p_25358];
            
            // src/bgpc_vv_coloring.fut:46:70-72
            
            int64_t seen_final_25365 = add64((int64_t) 1, eta_p_25358);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool x_25366 = sle64((int64_t) 0, seen_final_25365);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool y_25367 = slt64(seen_final_25365, csr_cols_from_pattern_res_24091);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool bounds_check_25368 = x_25366 && y_25367;
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool index_certs_25369;
            
            if (!bounds_check_25368) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_25365, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24091, "].", "-> #0  src/bgpc_vv_coloring.fut:46:59-73\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:59-73
            
            int64_t seen_final_25370 = ((int64_t *) ext_mem_26592.mem)[seen_final_25365];
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t j_m_i_25371 = sub64(seen_final_25370, seen_final_25364);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool empty_slice_25372 = j_m_i_25371 == (int64_t) 0;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t m_25373 = sub64(j_m_i_25371, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t i_p_m_t_s_25374 = add64(seen_final_25364, m_25373);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_leq_i_p_m_t_s_25375 = sle64((int64_t) 0, i_p_m_t_s_25374);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_p_m_t_s_leq_w_25376 = slt64(i_p_m_t_s_25374, csr_cols_from_pattern_res_24092);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_lte_i_25377 = sle64((int64_t) 0, seen_final_25364);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_lte_j_25378 = sle64(seen_final_25364, seen_final_25370);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_25379 = i_p_m_t_s_leq_w_25376 && zzero_lte_i_25377;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_25380 = zzero_leq_i_p_m_t_s_25375 && y_25379;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool forwards_ok_25381 = i_lte_j_25378 && y_25380;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool ok_or_empty_25382 = empty_slice_25372 || forwards_ok_25381;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool index_certs_25383;
            
            if (!ok_or_empty_25382) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_25364, ":", (long long) seen_final_25370, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24092, "].", "-> #0  src/bgpc_vv_coloring.fut:46:5-54:19\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:77:24-51
            for (int64_t nest_i_26771 = 0; nest_i_26771 < color_bound_24136; nest_i_26771++) {
                ((bool *) mem_26604)[nest_i_26771] = 0;
            }
            // src/bgpc_vv_coloring.fut:46:5-54:19
            for (int64_t i_25385 = 0; i_25385 < j_m_i_25371; i_25385++) {
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t index_primexp_25387 = seen_final_25364 + i_25385;
                
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t v_25388 = ((int64_t *) ext_mem_26591.mem)[index_primexp_25387];
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool x_25389 = sle64((int64_t) 0, v_25388);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool y_25390 = slt64(v_25388, csr_bipartite_from_pattern_res_24086);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool bounds_check_25391 = x_25389 && y_25390;
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool index_certs_25392;
                
                if (!bounds_check_25391) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_25388, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/bgpc_vv_coloring.fut:49:26-37\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:26-37
                
                int64_t seen_acczq_25393 = ((int64_t *) ext_mem_26576.mem)[v_25388];
                
                // src/bgpc_vv_coloring.fut:49:50-52
                
                int64_t seen_acczq_25394 = add64((int64_t) 1, v_25388);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool x_25395 = sle64((int64_t) 0, seen_acczq_25394);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool y_25396 = slt64(seen_acczq_25394, csr_bipartite_from_pattern_res_24086);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool bounds_check_25397 = x_25395 && y_25396;
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool index_certs_25398;
                
                if (!bounds_check_25397) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_25394, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/bgpc_vv_coloring.fut:49:40-53\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:40-53
                
                int64_t seen_acczq_25399 = ((int64_t *) ext_mem_26576.mem)[seen_acczq_25394];
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t j_m_i_25400 = sub64(seen_acczq_25399, seen_acczq_25393);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool empty_slice_25401 = j_m_i_25400 == (int64_t) 0;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t m_25402 = sub64(j_m_i_25400, (int64_t) 1);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t i_p_m_t_s_25403 = add64(seen_acczq_25393, m_25402);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_leq_i_p_m_t_s_25404 = sle64((int64_t) 0, i_p_m_t_s_25403);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_p_m_t_s_leq_w_25405 = slt64(i_p_m_t_s_25403, csr_bipartite_from_pattern_res_24087);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_lte_i_25406 = sle64((int64_t) 0, seen_acczq_25393);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_lte_j_25407 = sle64(seen_acczq_25393, seen_acczq_25399);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_25408 = i_p_m_t_s_leq_w_25405 && zzero_lte_i_25406;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_25409 = zzero_leq_i_p_m_t_s_25404 && y_25408;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool forwards_ok_25410 = i_lte_j_25407 && y_25409;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool ok_or_empty_25411 = empty_slice_25401 || forwards_ok_25410;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool index_certs_25412;
                
                if (!ok_or_empty_25411) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_25393, ":", (long long) seen_acczq_25399, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24087, "].", "-> #0  src/bgpc_vv_coloring.fut:48:9-53:33\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:48:9-53:33
                for (int64_t i_25414 = 0; i_25414 < j_m_i_25400; i_25414++) {
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t index_primexp_25416 = seen_acczq_25393 + i_25414;
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t u_25417 = ((int64_t *) ext_mem_26575.mem)[index_primexp_25416];
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool x_25418 = sle64((int64_t) 0, u_25417);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool y_25419 = slt64(u_25417, (int64_t) 6);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool bounds_check_25420 = x_25418 && y_25419;
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool index_certs_25421;
                    
                    if (!bounds_check_25420) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_25417, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/bgpc_vv_coloring.fut:50:20-29\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    int64_t cu_25422 = ((int64_t *) mem_26588)[u_25417];
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_25423 = u_25417 == eta_p_25358;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_25424 = !cond_25423;
                    
                    // src/bgpc_vv_coloring.fut:51:30-37
                    
                    bool cond_t_res_25425 = sle64((int64_t) 0, cu_25422);
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    bool x_25426 = cond_25424 && cond_t_res_25425;
                    
                    // src/bgpc_vv_coloring.fut:51:44-67
                    
                    bool cond_t_res_25427 = slt64(cu_25422, color_bound_24136);
                    
                    // src/bgpc_vv_coloring.fut:94:14-103:69
                    
                    bool x_25428 = x_25426 && cond_t_res_25427;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    if (x_25428) {
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool bounds_check_25983 = cond_t_res_25425 && cond_t_res_25427;
                        
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool index_certs_25984;
                        
                        if (!bounds_check_25983) {
                            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) cu_25422, "] out of bounds for array of shape [", (long long) color_bound_24136, "].", "-> #0  src/bgpc_vv_coloring.fut:52:19-50\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                            err = FUTHARK_PROGRAM_ERROR;
                            goto cleanup;
                        }
                        // src/bgpc_vv_coloring.fut:52:19-50
                        ((bool *) mem_26604)[cu_25422] = 1;
                    }
                }
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool loop_cond_25433;
            
            if (cond_24137) {
                // src/bgpc_vv_coloring.fut:63:30-37
                
                bool loop_cond_t_res_25986 = ((bool *) mem_26604)[(int64_t) 0];
                
                loop_cond_25433 = loop_cond_t_res_25986;
            } else {
                loop_cond_25433 = 0;
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool c_final_25435;
            int64_t c_final_25436;
            bool loop_while_25437;
            int64_t c_25438;
            
            loop_while_25437 = loop_cond_25433;
            c_25438 = (int64_t) 0;
            while (loop_while_25437) {
                // src/bgpc_vv_coloring.fut:64:9-15
                
                int64_t loopres_25439 = add64((int64_t) 1, c_25438);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool cond_25440 = slt64(loopres_25439, color_bound_24136);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool loop_cond_25441;
                
                if (cond_25440) {
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool x_25987 = sle64((int64_t) 0, loopres_25439);
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool bounds_check_25988 = cond_25440 && x_25987;
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool index_certs_25989;
                    
                    if (!bounds_check_25988) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) loopres_25439, "] out of bounds for array of shape [", (long long) color_bound_24136, "].", "-> #0  src/bgpc_vv_coloring.fut:63:30-37\n   #1  src/bgpc_vv_coloring.fut:81:6-28\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool loop_cond_t_res_25990 = ((bool *) mem_26604)[loopres_25439];
                    
                    loop_cond_25441 = loop_cond_t_res_25990;
                } else {
                    loop_cond_25441 = 0;
                }
                
                bool loop_while_tmp_26774 = loop_cond_25441;
                int64_t c_tmp_26775 = loopres_25439;
                
                loop_while_25437 = loop_while_tmp_26774;
                c_25438 = c_tmp_26775;
            }
            c_final_25435 = loop_while_25437;
            c_final_25436 = c_25438;
            // src/bgpc_vv_coloring.fut:106:25-32
            
            int64_t max_res_24231 = smax64(c_final_25436, redout_26397);
            
            ((int64_t *) mem_26600)[i_26399] = c_final_25436;
            
            int64_t redout_tmp_26769 = max_res_24231;
            
            redout_26397 = redout_tmp_26769;
        }
        defunc_0_reduce_res_26019 = redout_26397;
        // src/bgpc_vv_coloring.fut:113:17-43
        
        bool acc_cert_24237;
        
        // src/bgpc_vv_coloring.fut:113:17-43
        for (int64_t i_26402 = 0; i_26402 < loop_dz2081Uz2083U_24132; i_26402++) {
            int64_t v_24241 = ((int64_t *) mem_param_26598.mem)[i_26402];
            int64_t v_24242 = ((int64_t *) mem_26600)[i_26402];
            
            // src/bgpc_vv_coloring.fut:113:17-43
            // UpdateAcc
            if (sle64((int64_t) 0, v_24241) && slt64(v_24241, (int64_t) 6)) {
                ((int64_t *) mem_26588)[v_24241] = v_24242;
            }
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26621_cached_sizze_26967 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26621, &mem_26621_cached_sizze_26967, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26623_cached_sizze_26968 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26623, &mem_26623_cached_sizze_26968, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26624_cached_sizze_26969 < loop_dz2081Uz2083U_24132) {
            err = lexical_realloc(ctx, &mem_26624, &mem_26624_cached_sizze_26969, loop_dz2081Uz2083U_24132);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t discard_26414;
        int64_t scanacc_26406 = (int64_t) 0;
        
        for (int64_t i_26410 = 0; i_26410 < loop_dz2081Uz2083U_24132; i_26410++) {
            int64_t eta_p_25282 = ((int64_t *) mem_param_26598.mem)[i_26410];
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool x_25283 = sle64((int64_t) 0, eta_p_25282);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool y_25284 = slt64(eta_p_25282, csr_cols_from_pattern_res_24091);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool bounds_check_25285 = x_25283 && y_25284;
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool index_certs_25286;
            
            if (!bounds_check_25285) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25282, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24091, "].", "-> #0  src/bgpc_vv_coloring.fut:126:15-27\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:127:26-28
            
            int64_t k_end_25288 = add64((int64_t) 1, eta_p_25282);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool x_25289 = sle64((int64_t) 0, k_end_25288);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool y_25290 = slt64(k_end_25288, csr_cols_from_pattern_res_24091);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool bounds_check_25291 = x_25289 && y_25290;
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool index_certs_25292;
            
            if (!bounds_check_25291) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_end_25288, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24091, "].", "-> #0  src/bgpc_vv_coloring.fut:127:15-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:126:15-27
            
            int64_t k0_25287 = ((int64_t *) ext_mem_26592.mem)[eta_p_25282];
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            int64_t k_end_25293 = ((int64_t *) ext_mem_26592.mem)[k_end_25288];
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool cond_25294 = slt64(k0_25287, k_end_25293);
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool loses_conflict_vertex_res_25295;
            bool loses_conflict_vertex_res_25296;
            int64_t loses_conflict_vertex_res_25297;
            bool loop_while_25298;
            bool lost_25299;
            int64_t k_25300;
            
            loop_while_25298 = cond_25294;
            lost_25299 = 0;
            k_25300 = k0_25287;
            while (loop_while_25298) {
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool x_25301 = sle64((int64_t) 0, k_25300);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool y_25302 = slt64(k_25300, csr_cols_from_pattern_res_24092);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool bounds_check_25303 = x_25301 && y_25302;
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool index_certs_25304;
                
                if (!bounds_check_25303) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_25300, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24092, "].", "-> #0  src/bgpc_vv_coloring.fut:132:15-26\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:132:15-26
                
                int64_t v_25305 = ((int64_t *) ext_mem_26591.mem)[k_25300];
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool x_25306 = sle64((int64_t) 0, v_25305);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool y_25307 = slt64(v_25305, csr_bipartite_from_pattern_res_24086);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool bounds_check_25308 = x_25306 && y_25307;
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool index_certs_25309;
                
                if (!bounds_check_25308) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_25305, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/bgpc_vv_coloring.fut:134:19-30\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:135:29-31
                
                int64_t t_end_25311 = add64((int64_t) 1, v_25305);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool x_25312 = sle64((int64_t) 0, t_end_25311);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool y_25313 = slt64(t_end_25311, csr_bipartite_from_pattern_res_24086);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool bounds_check_25314 = x_25312 && y_25313;
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool index_certs_25315;
                
                if (!bounds_check_25314) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_end_25311, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/bgpc_vv_coloring.fut:135:19-32\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:134:19-30
                
                int64_t t0_25310 = ((int64_t *) ext_mem_26576.mem)[v_25305];
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                int64_t t_end_25316 = ((int64_t *) ext_mem_26576.mem)[t_end_25311];
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool cond_25317 = slt64(t0_25310, t_end_25316);
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool loopres_25318;
                bool loopres_25319;
                int64_t loopres_25320;
                bool loop_while_25321;
                bool lost_in_net_25322;
                int64_t t_25323;
                
                loop_while_25321 = cond_25317;
                lost_in_net_25322 = 0;
                t_25323 = t0_25310;
                while (loop_while_25321) {
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool x_25324 = sle64((int64_t) 0, t_25323);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool y_25325 = slt64(t_25323, csr_bipartite_from_pattern_res_24087);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool bounds_check_25326 = x_25324 && y_25325;
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool index_certs_25327;
                    
                    if (!bounds_check_25326) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_25323, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24087, "].", "-> #0  src/bgpc_vv_coloring.fut:140:19-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    int64_t u_25328 = ((int64_t *) ext_mem_26575.mem)[t_25323];
                    
                    // src/bgpc_vv_coloring.fut:142:13-71
                    
                    bool lost_in_netzq_25329;
                    
                    if (lost_in_net_25322) {
                        lost_in_netzq_25329 = 1;
                    } else {
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25330 = u_25328 == eta_p_25282;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25331 = !cond_25330;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25332;
                        
                        if (cond_25331) {
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool x_25992 = sle64((int64_t) 0, u_25328);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool y_25993 = slt64(u_25328, (int64_t) 6);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool bounds_check_25994 = x_25992 && y_25993;
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool index_certs_25995;
                            
                            if (!bounds_check_25994) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_25328, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/bgpc_vv_coloring.fut:142:39-48\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool y_25997 = slt64(eta_p_25282, (int64_t) 6);
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool bounds_check_25998 = x_25283 && y_25997;
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool index_certs_25999;
                            
                            if (!bounds_check_25998) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25282, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/bgpc_vv_coloring.fut:142:52-61\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            int64_t zeze_lhs_25996 = ((int64_t *) mem_26588)[u_25328];
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            int64_t zeze_rhs_26000 = ((int64_t *) mem_26588)[eta_p_25282];
                            
                            // src/bgpc_vv_coloring.fut:142:49-61
                            
                            bool cond_t_res_26001 = zeze_lhs_25996 == zeze_rhs_26000;
                            
                            cond_25332 = cond_t_res_26001;
                        } else {
                            cond_25332 = 0;
                        }
                        // src/bgpc_vv_coloring.fut:142:67-70
                        
                        bool lost_in_netzq_f_res_t_res_25343 = slt64(u_25328, eta_p_25282);
                        
                        // src/bgpc_vv_coloring.fut:160:14-168:57
                        
                        bool x_25344 = cond_25332 && lost_in_netzq_f_res_t_res_25343;
                        
                        lost_in_netzq_25329 = x_25344;
                    }
                    // src/bgpc_vv_coloring.fut:143:31-37
                    
                    int64_t tmp_25345 = add64((int64_t) 1, t_25323);
                    
                    // src/bgpc_vv_coloring.fut:139:15-43
                    
                    bool cond_25346 = slt64(tmp_25345, t_end_25316);
                    
                    // src/bgpc_vv_coloring.fut:139:28-43
                    
                    bool not_res_25347 = !lost_in_netzq_25329;
                    
                    // src/bgpc_vv_coloring.fut:166:5-169:10
                    
                    bool x_25348 = cond_25346 && not_res_25347;
                    bool loop_while_tmp_26784 = x_25348;
                    bool lost_in_net_tmp_26785 = lost_in_netzq_25329;
                    int64_t t_tmp_26786 = tmp_25345;
                    
                    loop_while_25321 = loop_while_tmp_26784;
                    lost_in_net_25322 = lost_in_net_tmp_26785;
                    t_25323 = t_tmp_26786;
                }
                loopres_25318 = loop_while_25321;
                loopres_25319 = lost_in_net_25322;
                loopres_25320 = t_25323;
                // src/bgpc_vv_coloring.fut:160:14-168:57
                
                bool lostzq_25349 = lost_25299 || loopres_25319;
                
                // src/bgpc_vv_coloring.fut:146:20-26
                
                int64_t tmp_25350 = add64((int64_t) 1, k_25300);
                
                // src/bgpc_vv_coloring.fut:131:11-32
                
                bool cond_25351 = slt64(tmp_25350, k_end_25293);
                
                // src/bgpc_vv_coloring.fut:131:24-32
                
                bool not_res_25352 = !lostzq_25349;
                
                // src/bgpc_vv_coloring.fut:166:5-169:10
                
                bool x_25353 = cond_25351 && not_res_25352;
                bool loop_while_tmp_26781 = x_25353;
                bool lost_tmp_26782 = lostzq_25349;
                int64_t k_tmp_26783 = tmp_25350;
                
                loop_while_25298 = loop_while_tmp_26781;
                lost_25299 = lost_tmp_26782;
                k_25300 = k_tmp_26783;
            }
            loses_conflict_vertex_res_25295 = loop_while_25298;
            loses_conflict_vertex_res_25296 = lost_25299;
            loses_conflict_vertex_res_25297 = k_25300;
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_f_res_25355 = btoi_bool_i64(loses_conflict_vertex_res_25296);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_op_res_24343 = add64(defunc_0_f_res_25355, scanacc_26406);
            
            ((int64_t *) mem_26621)[i_26410] = defunc_0_op_res_24343;
            ((int64_t *) mem_26623)[i_26410] = defunc_0_f_res_25355;
            ((bool *) mem_26624)[i_26410] = loses_conflict_vertex_res_25296;
            
            int64_t scanacc_tmp_26777 = defunc_0_op_res_24343;
            
            scanacc_26406 = scanacc_tmp_26777;
        }
        discard_26414 = scanacc_26406;
        // src/bgpc_vv_coloring.fut:23:25-28
        
        int64_t tmp_24327 = sub64(loop_dz2081Uz2083U_24132, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool y_24329 = slt64(tmp_24327, loop_dz2081Uz2083U_24132);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool x_24328 = sle64((int64_t) 0, tmp_24327);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool bounds_check_24330 = x_24328 && y_24329;
        
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        bool cond_24325 = loop_dz2081Uz2083U_24132 == (int64_t) 0;
        
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool protect_assert_disj_24331 = cond_24325 || bounds_check_24330;
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool index_certs_24332;
        
        if (!protect_assert_disj_24331) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_24327, "] out of bounds for array of shape [", (long long) loop_dz2081Uz2083U_24132, "].", "-> #0  src/bgpc_vv_coloring.fut:23:10-29\n   #1  src/bgpc_vv_coloring.fut:171:23-55\n   #2  src/bgpc_vv_coloring.fut:186:14-205:56\n   #3  src/bgpc_vv_coloring.fut:213:14-216:64\n   #4  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #5  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #6  src/sparse_jacobian_jvp.fut:158:18-33\n   #7  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool x_24326 = !cond_24325;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_f_res_24344;
        
        if (x_24326) {
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t x_26002 = ((int64_t *) mem_26621)[tmp_24327];
            
            m_f_res_24344 = x_26002;
        } else {
            m_f_res_24344 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_24346;
        
        if (cond_24325) {
            m_24346 = (int64_t) 0;
        } else {
            m_24346 = m_f_res_24344;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t bytes_26643 = (int64_t) 8 * m_24346;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26644_cached_sizze_26970 < bytes_26643) {
            err = lexical_realloc(ctx, &mem_26644, &mem_26644_cached_sizze_26970, bytes_26643);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26646_cached_sizze_26971 < bytes_26643) {
            err = lexical_realloc(ctx, &mem_26646, &mem_26646_cached_sizze_26971, bytes_26643);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:18:24-27:44
        if (mem_26648_cached_sizze_26972 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26648, &mem_26648_cached_sizze_26972, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26656_cached_sizze_26973 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26656, &mem_26656_cached_sizze_26973, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        bool acc_cert_25155;
        bool acc_cert_25156;
        
        // src/bgpc_vv_coloring.fut:18:24-27:44
        
        int64_t inpacc_26010;
        int64_t inpacc_25219 = (int64_t) 0;
        
        for (int64_t i_26441 = 0; i_26441 < loop_dz2081Uz2083U_24132; i_26441++) {
            bool eta_p_26503 = ((bool *) mem_26624)[i_26441];
            
            // src/bgpc_vv_coloring.fut:18:28-39
            
            int64_t bool_to_i64_res_26504 = btoi_bool_i64(eta_p_26503);
            int64_t eta_p_26516 = ((int64_t *) mem_26623)[i_26441];
            int64_t eta_p_26517 = ((int64_t *) mem_26621)[i_26441];
            int64_t v_26520 = ((int64_t *) mem_param_26598.mem)[i_26441];
            
            // src/bgpc_vv_coloring.fut:19:29-32
            
            int64_t defunc_0_op_res_26521 = add64(inpacc_25219, bool_to_i64_res_26504);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            bool cond_26522 = eta_p_26516 == (int64_t) 1;
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t lifted_lambda_res_26523;
            
            if (cond_26522) {
                // src/bgpc_vv_coloring.fut:27:5-44
                
                int64_t lifted_lambda_res_t_res_26524 = sub64(eta_p_26517, (int64_t) 1);
                
                lifted_lambda_res_26523 = lifted_lambda_res_t_res_26524;
            } else {
                lifted_lambda_res_26523 = (int64_t) -1;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_26523) && slt64(lifted_lambda_res_26523, m_24346)) {
                ((int64_t *) mem_26646)[lifted_lambda_res_26523] = v_26520;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_26523) && slt64(lifted_lambda_res_26523, m_24346)) {
                ((int64_t *) mem_26644)[lifted_lambda_res_26523] = defunc_0_op_res_26521;
            }
            ((int64_t *) mem_26648)[i_26441] = defunc_0_op_res_26521;
            
            int64_t inpacc_tmp_26787 = defunc_0_op_res_26521;
            
            inpacc_25219 = inpacc_tmp_26787;
        }
        inpacc_26010 = inpacc_25219;
        // src/bgpc_vv_coloring.fut:27:5-44
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_26656, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_26648, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {loop_dz2081Uz2083U_24132});
        // src/bgpc_vv_coloring.fut:23:10-29
        
        int64_t out_szz_f_res_24333;
        
        if (x_24326) {
            // src/bgpc_vv_coloring.fut:171:23-55
            
            int64_t x_26016 = ((int64_t *) mem_26656)[tmp_24327];
            
            out_szz_f_res_24333 = x_26016;
        } else {
            out_szz_f_res_24333 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        int64_t out_szz_24335;
        
        if (cond_24325) {
            out_szz_24335 = (int64_t) 0;
        } else {
            out_szz_24335 = out_szz_f_res_24333;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        
        int64_t bytes_26657 = (int64_t) 8 * out_szz_24335;
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        bool cond_24232 = slt64(defunc_0_reduce_res_26019, (int64_t) 0);
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        int64_t next_color_bound_24233;
        
        if (cond_24232) {
            next_color_bound_24233 = color_bound_24136;
        } else {
            // src/bgpc_vv_coloring.fut:110:40-46
            
            int64_t max_arg1_24234 = add64((int64_t) 2, defunc_0_reduce_res_26019);
            
            // src/bgpc_vv_coloring.fut:110:10-46
            
            int64_t max_res_24235 = smax64(color_bound_24136, max_arg1_24234);
            
            next_color_bound_24233 = max_res_24235;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        if (memblock_alloc(ctx, &mem_26658, bytes_26657, "mem_26658")) {
            err = 1;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        for (int64_t nest_i_26791 = 0; nest_i_26791 < out_szz_24335; nest_i_26791++) {
            ((int64_t *) mem_26658.mem)[nest_i_26791] = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:33:6-27
        
        bool acc_cert_24928;
        
        // src/bgpc_vv_coloring.fut:29:21-33:27
        for (int64_t i_26446 = 0; i_26446 < m_24346; i_26446++) {
            int64_t eta_p_24940 = ((int64_t *) mem_26644)[i_26446];
            int64_t v_24942 = ((int64_t *) mem_26646)[i_26446];
            
            // src/bgpc_vv_coloring.fut:29:44-50
            
            int64_t lifted_lambda_res_24943 = sub64(eta_p_24940, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:33:6-27
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_24943) && slt64(lifted_lambda_res_24943, out_szz_24335)) {
                ((int64_t *) mem_26658.mem)[lifted_lambda_res_24943] = v_24942;
            }
        }
        // src/bgpc_vv_coloring.fut:175:17-48
        
        bool acc_cert_24383;
        
        // src/bgpc_vv_coloring.fut:175:17-48
        for (int64_t i_26448 = 0; i_26448 < out_szz_24335; i_26448++) {
            int64_t v_24387 = ((int64_t *) mem_26658.mem)[i_26448];
            
            // src/bgpc_vv_coloring.fut:175:17-48
            // UpdateAcc
            if (sle64((int64_t) 0, v_24387) && slt64(v_24387, (int64_t) 6)) {
                ((int64_t *) mem_26588)[v_24387] = (int64_t) -1;
            }
        }
        // src/bgpc_vv_coloring.fut:198:20-23
        
        bool loop_cond_24389 = slt64((int64_t) 0, out_szz_24335);
        
        if (memblock_set(ctx, &mem_param_tmp_26763, &mem_26658, "mem_26658") != 0)
            return 1;
        
        int64_t loop_dz2081Uz2083U_tmp_26764 = out_szz_24335;
        bool loop_while_tmp_26765 = loop_cond_24389;
        int64_t color_bound_tmp_26768 = next_color_bound_24233;
        
        if (memblock_set(ctx, &mem_param_26598, &mem_param_tmp_26763, "mem_param_tmp_26763") != 0)
            return 1;
        loop_dz2081Uz2083U_24132 = loop_dz2081Uz2083U_tmp_26764;
        loop_while_24133 = loop_while_tmp_26765;
        color_bound_24136 = color_bound_tmp_26768;
    }
    if (memblock_set(ctx, &ext_mem_26663, &mem_param_26598, "mem_param_26598") != 0)
        return 1;
    vv_color_side_order_res_24127 = loop_dz2081Uz2083U_24132;
    vv_color_side_order_res_24128 = loop_while_24133;
    vv_color_side_order_res_24131 = color_bound_24136;
    if (memblock_unref(ctx, &mem_26587, "mem_26587") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26591, "ext_mem_26591") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26592, "ext_mem_26592") != 0)
        return 1;
    // src/sparse_jacobian_jvp.fut:107:8-112:40
    
    int64_t x_26031;
    int64_t redout_26449 = (int64_t) 0;
    
    for (int64_t i_26450 = 0; i_26450 < (int64_t) 6; i_26450++) {
        int64_t x_24394 = ((int64_t *) mem_26588)[i_26450];
        
        // src/sparse_jacobian_jvp.fut:14:22-29
        
        int64_t max_res_24397 = smax64(x_24394, redout_26449);
        int64_t redout_tmp_26794 = max_res_24397;
        
        redout_26449 = redout_tmp_26794;
    }
    x_26031 = redout_26449;
    // src/sparse_jacobian_jvp.fut:14:13-41
    
    int64_t num_colors_of_res_f_res_24398 = add64((int64_t) 1, x_26031);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool bounds_invalid_upwards_24400 = slt64(num_colors_of_res_f_res_24398, (int64_t) 0);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool valid_24401 = !bounds_invalid_upwards_24400;
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool range_valid_c_24402;
    
    if (!valid_24401) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Range ", (long long) (int64_t) 0, "..", (long long) (int64_t) 1, "..<", (long long) num_colors_of_res_f_res_24398, " is invalid.", "-> #0  src/sparse_jacobian_jvp.fut:29:11-18\n   #1  src/sparse_jacobian_jvp.fut:107:8-112:40\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    
    int64_t bytes_26666 = (int64_t) 32 * num_colors_of_res_f_res_24398;
    double zt_lhs_24490 = ((double *) x_mem_26563.mem)[(int64_t) 0];
    double zt_rhs_24491 = ((double *) x_mem_26563.mem)[(int64_t) 1];
    double zt_rhs_24493 = ((double *) x_mem_26563.mem)[(int64_t) 4];
    
    // test/test_sparse_jacobian_jvp.fut:82:33-39
    
    double binop_x_25872 = 0.0 * zt_rhs_24493;
    double zt_lhs_24497 = ((double *) x_mem_26563.mem)[(int64_t) 5];
    
    // src/dense_jacobian.fut:8:40-9:68
    if (mem_26564_cached_sizze_26960 < (int64_t) 192) {
        err = lexical_realloc(ctx, &mem_26564, &mem_26564_cached_sizze_26960, (int64_t) 192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:5:3-21
    if (mem_26569_cached_sizze_26961 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26569, &mem_26569_cached_sizze_26961, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:8:40-9:68
    for (int64_t i_26386 = 0; i_26386 < (int64_t) 6; i_26386++) {
        // src/dense_jacobian.fut:5:3-21
        for (int64_t nest_i_26796 = 0; nest_i_26796 < (int64_t) 6; nest_i_26796++) {
            ((double *) mem_26569)[nest_i_26796] = 0.0;
        }
        // src/dense_jacobian.fut:5:3-39
        ((double *) mem_26569)[i_26386] = 1.0;
        
        double zt_lhs_tan_25865 = ((double *) mem_26569)[(int64_t) 0];
        double zt_rhs_tan_25866 = ((double *) mem_26569)[(int64_t) 1];
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_x_25868 = zt_rhs_24491 * zt_lhs_tan_25865;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_y_25869 = zt_lhs_24490 * zt_rhs_tan_25866;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double zp_lhs_tan_25867 = binop_x_25868 + binop_y_25869;
        double zt_rhs_tan_25870 = ((double *) mem_26569)[(int64_t) 4];
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double binop_y_25873 = 3.0 * zt_rhs_tan_25870;
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double zp_rhs_tan_25871 = binop_x_25872 + binop_y_25873;
        
        // test/test_sparse_jacobian_jvp.fut:82:24-39
        
        double y0_tan_25874 = zp_lhs_tan_25867 + zp_rhs_tan_25871;
        double zm_lhs_tan_25877 = ((double *) mem_26569)[(int64_t) 2];
        double zt_lhs_tan_25878 = ((double *) mem_26569)[(int64_t) 5];
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double binop_x_25880 = zt_lhs_24497 * zt_lhs_tan_25878;
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double zm_rhs_tan_25879 = binop_x_25880 + binop_x_25880;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double binop_y_25884 = -1.0 * zm_rhs_tan_25879;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double y1_tan_25882 = zm_lhs_tan_25877 + binop_y_25884;
        
        // test/test_sparse_jacobian_jvp.fut:84:17-23
        
        double zp_lhs_tan_25885 = zt_rhs_tan_25866 + zm_lhs_tan_25877;
        double zp_rhs_tan_25888 = ((double *) mem_26569)[(int64_t) 3];
        
        // test/test_sparse_jacobian_jvp.fut:84:24-30
        
        double y2_tan_25889 = zp_lhs_tan_25885 + zp_rhs_tan_25888;
        
        // test/test_sparse_jacobian_jvp.fut:103:59-64
        ((double *) mem_26564)[i_26386 * (int64_t) 4] = y0_tan_25874;
        ((double *) mem_26564)[i_26386 * (int64_t) 4 + (int64_t) 1] = y1_tan_25882;
        ((double *) mem_26564)[i_26386 * (int64_t) 4 + (int64_t) 2] = y2_tan_25889;
        ((double *) mem_26564)[i_26386 * (int64_t) 4 + (int64_t) 3] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    if (mem_26578_cached_sizze_26962 < bytes_26577) {
        err = lexical_realloc(ctx, &mem_26578, &mem_26578_cached_sizze_26962, bytes_26577);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    for (int64_t nest_i_26797 = 0; nest_i_26797 < csr_bipartite_from_pattern_res_24087; nest_i_26797++) {
        ((double *) mem_26578)[nest_i_26797] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    if (mem_26667_cached_sizze_26974 < bytes_26666) {
        err = lexical_realloc(ctx, &mem_26667, &mem_26667_cached_sizze_26974, bytes_26666);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:17:3-61
    if (mem_26672_cached_sizze_26975 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26672, &mem_26672_cached_sizze_26975, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    for (int64_t i_26457 = 0; i_26457 < num_colors_of_res_f_res_24398; i_26457++) {
        // src/sparse_jacobian_jvp.fut:17:3-61
        for (int64_t i_26453 = 0; i_26453 < (int64_t) 6; i_26453++) {
            int64_t eta_p_24411 = ((int64_t *) mem_26588)[i_26453];
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            bool cond_24412 = eta_p_24411 == i_26457;
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            double lifted_lambda_res_24413;
            
            if (cond_24412) {
                lifted_lambda_res_24413 = 1.0;
            } else {
                lifted_lambda_res_24413 = 0.0;
            }
            ((double *) mem_26672)[i_26453] = lifted_lambda_res_24413;
        }
        
        double zt_lhs_tan_25894 = ((double *) mem_26672)[(int64_t) 0];
        double zt_rhs_tan_25895 = ((double *) mem_26672)[(int64_t) 1];
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_x_25897 = zt_rhs_24491 * zt_lhs_tan_25894;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_y_25898 = zt_lhs_24490 * zt_rhs_tan_25895;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double zp_lhs_tan_25896 = binop_x_25897 + binop_y_25898;
        double zt_rhs_tan_25899 = ((double *) mem_26672)[(int64_t) 4];
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double binop_y_25902 = 3.0 * zt_rhs_tan_25899;
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double zp_rhs_tan_25900 = binop_x_25872 + binop_y_25902;
        
        // test/test_sparse_jacobian_jvp.fut:82:24-39
        
        double y0_tan_25903 = zp_lhs_tan_25896 + zp_rhs_tan_25900;
        double zm_lhs_tan_25906 = ((double *) mem_26672)[(int64_t) 2];
        double zt_lhs_tan_25907 = ((double *) mem_26672)[(int64_t) 5];
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double binop_x_25909 = zt_lhs_24497 * zt_lhs_tan_25907;
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double zm_rhs_tan_25908 = binop_x_25909 + binop_x_25909;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double binop_y_25913 = -1.0 * zm_rhs_tan_25908;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double y1_tan_25911 = zm_lhs_tan_25906 + binop_y_25913;
        
        // test/test_sparse_jacobian_jvp.fut:84:17-23
        
        double zp_lhs_tan_25914 = zt_rhs_tan_25895 + zm_lhs_tan_25906;
        double zp_rhs_tan_25917 = ((double *) mem_26672)[(int64_t) 3];
        
        // test/test_sparse_jacobian_jvp.fut:84:24-30
        
        double y2_tan_25918 = zp_lhs_tan_25914 + zp_rhs_tan_25917;
        
        // test/test_sparse_jacobian_jvp.fut:104:33-38
        ((double *) mem_26667)[i_26457 * (int64_t) 4] = y0_tan_25903;
        ((double *) mem_26667)[i_26457 * (int64_t) 4 + (int64_t) 1] = y1_tan_25911;
        ((double *) mem_26667)[i_26457 * (int64_t) 4 + (int64_t) 2] = y2_tan_25918;
        ((double *) mem_26667)[i_26457 * (int64_t) 4 + (int64_t) 3] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:45:5-52:27
    
    bool compressed_to_csr_vals_res_24422;
    int64_t compressed_to_csr_vals_res_24424;
    bool loop_while_24425;
    int64_t i_24427;
    
    loop_while_24425 = 1;
    i_24427 = (int64_t) 0;
    while (loop_while_24425) {
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool x_24428 = sle64((int64_t) 0, i_24427);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool y_24429 = slt64(i_24427, csr_bipartite_from_pattern_res_24086);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool bounds_check_24430 = x_24428 && y_24429;
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool index_certs_24431;
        
        if (!bounds_check_24430) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_24427, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/sparse_jacobian_jvp.fut:47:15-26\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        int64_t s_24432 = ((int64_t *) ext_mem_26576.mem)[i_24427];
        
        // src/sparse_jacobian_jvp.fut:48:25-27
        
        int64_t e_24433 = add64((int64_t) 1, i_24427);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool x_24434 = sle64((int64_t) 0, e_24433);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool y_24435 = slt64(e_24433, csr_bipartite_from_pattern_res_24086);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool bounds_check_24436 = x_24434 && y_24435;
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool index_certs_24437;
        
        if (!bounds_check_24436) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_24433, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/sparse_jacobian_jvp.fut:48:15-28\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        int64_t e_24438 = ((int64_t *) ext_mem_26576.mem)[e_24433];
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t j_m_i_24439 = sub64(e_24438, s_24432);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool empty_slice_24440 = j_m_i_24439 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t m_24441 = sub64(j_m_i_24439, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t i_p_m_t_s_24442 = add64(s_24432, m_24441);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_24443 = sle64((int64_t) 0, i_p_m_t_s_24442);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_24444 = slt64(i_p_m_t_s_24442, csr_bipartite_from_pattern_res_24087);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_lte_i_24445 = sle64((int64_t) 0, s_24432);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_lte_j_24446 = sle64(s_24432, e_24438);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_24447 = i_p_m_t_s_leq_w_24444 && zzero_lte_i_24445;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_24448 = zzero_leq_i_p_m_t_s_24443 && y_24447;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool forwards_ok_24449 = i_lte_j_24446 && y_24448;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool ok_or_empty_24450 = empty_slice_24440 || forwards_ok_24449;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool index_certs_24451;
        
        if (!ok_or_empty_24450) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_24432, ":", (long long) e_24438, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24087, "].", "-> #0  src/sparse_jacobian_jvp.fut:49:18-30\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool y_24453 = slt64(i_24427, (int64_t) 4);
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool bounds_check_24454 = x_24428 && y_24453;
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool index_certs_24455;
        
        if (!bounds_check_24454) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_24427, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-45\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:18-51
        for (int64_t i_26461 = 0; i_26461 < j_m_i_24439; i_26461++) {
            int64_t index_primexp_26496 = s_24432 + i_26461;
            int64_t eta_p_24457 = ((int64_t *) ext_mem_26575.mem)[index_primexp_26496];
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool x_24458 = sle64((int64_t) 0, eta_p_24457);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool y_24459 = slt64(eta_p_24457, (int64_t) 6);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool bounds_check_24460 = x_24458 && y_24459;
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool index_certs_24461;
            
            if (!bounds_check_24460) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_24457, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:32-41\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            int64_t tmp_24462 = ((int64_t *) mem_26588)[eta_p_24457];
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool x_24463 = sle64((int64_t) 0, tmp_24462);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool y_24464 = slt64(tmp_24462, num_colors_of_res_f_res_24398);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool bounds_check_24465 = x_24463 && y_24464;
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool index_certs_24466;
            
            if (!bounds_check_24465) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_24462, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_24398, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-42\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:18-51
            
            double lifted_lambda_res_24467 = ((double *) mem_26667)[tmp_24462 * (int64_t) 4 + i_24427];
            
            ((double *) mem_26578)[s_24432 + i_26461] = lifted_lambda_res_24467;
        }
        // src/sparse_jacobian_jvp.fut:46:13-16
        
        bool loop_cond_24469 = slt64(e_24433, (int64_t) 4);
        bool loop_while_tmp_26800 = loop_cond_24469;
        int64_t i_tmp_26802 = e_24433;
        
        loop_while_24425 = loop_while_tmp_26800;
        i_24427 = i_tmp_26802;
    }
    compressed_to_csr_vals_res_24422 = loop_while_24425;
    compressed_to_csr_vals_res_24424 = i_24427;
    // src/sparse_jacobian_jvp.fut:67:29-47
    if (mem_26698_cached_sizze_26976 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26698, &mem_26698_cached_sizze_26976, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:8:19-10:39
    
    bool defunc_0_reduce_res_26033;
    bool redout_26467 = 1;
    
    for (int64_t i_26468 = 0; i_26468 < (int64_t) 4; i_26468++) {
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool y_24851 = slt64(i_26468, csr_bipartite_from_pattern_res_24086);
        
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool index_certs_24853;
        
        if (!y_24851) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_26468, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/sparse_jacobian_jvp.fut:63:17-28\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        int64_t s_24854 = ((int64_t *) ext_mem_26576.mem)[i_26468];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_lte_i_24867 = sle64((int64_t) 0, s_24854);
        
        // src/sparse_jacobian_jvp.fut:64:27-29
        
        int64_t e_24855 = add64((int64_t) 1, i_26468);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool y_24857 = slt64(e_24855, csr_bipartite_from_pattern_res_24086);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool x_24856 = sle64((int64_t) 0, e_24855);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool bounds_check_24858 = x_24856 && y_24857;
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool index_certs_24859;
        
        if (!bounds_check_24858) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_24855, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24086, "].", "-> #0  src/sparse_jacobian_jvp.fut:64:17-30\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        int64_t e_24860 = ((int64_t *) ext_mem_26576.mem)[e_24855];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t j_m_i_24861 = sub64(e_24860, s_24854);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t m_24863 = sub64(j_m_i_24861, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t i_p_m_t_s_24864 = add64(s_24854, m_24863);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_p_m_t_s_leq_w_24866 = slt64(i_p_m_t_s_24864, csr_bipartite_from_pattern_res_24087);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24875 = i_p_m_t_s_leq_w_24866 && zzero_lte_i_24867;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_leq_i_p_m_t_s_24865 = sle64((int64_t) 0, i_p_m_t_s_24864);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24876 = zzero_leq_i_p_m_t_s_24865 && y_24875;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_lte_j_24868 = sle64(s_24854, e_24860);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool forwards_ok_24877 = i_lte_j_24868 && y_24876;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool empty_slice_24862 = j_m_i_24861 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool ok_or_empty_24878 = empty_slice_24862 || forwards_ok_24877;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool index_certs_24879;
        
        if (!ok_or_empty_24878) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_24854, ":", (long long) e_24860, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24087, "].", "-> #0  src/sparse_jacobian_jvp.fut:66:20-29\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-104:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:67:29-47
        for (int64_t nest_i_26805 = 0; nest_i_26805 < (int64_t) 6; nest_i_26805++) {
            ((double *) mem_26698)[nest_i_26805] = 0.0;
        }
        // src/sparse_jacobian_jvp.fut:68:12-32
        
        bool acc_cert_24883;
        
        // src/sparse_jacobian_jvp.fut:68:12-32
        for (int64_t i_26464 = 0; i_26464 < j_m_i_24861; i_26464++) {
            int64_t index_primexp_26493 = s_24854 + i_26464;
            int64_t v_24887 = ((int64_t *) ext_mem_26575.mem)[index_primexp_26493];
            double v_24888 = ((double *) mem_26578)[index_primexp_26493];
            
            // src/sparse_jacobian_jvp.fut:68:12-32
            // UpdateAcc
            if (sle64((int64_t) 0, v_24887) && slt64(v_24887, (int64_t) 6)) {
                ((double *) mem_26698)[v_24887] = v_24888;
            }
        }
        // test/test_sparse_jacobian_jvp.fut:9:5-66
        
        bool defunc_0_reduce_res_26028;
        bool redout_26465 = 1;
        
        for (int64_t i_26466 = 0; i_26466 < (int64_t) 6; i_26466++) {
            bool eta_p_24911 = ((bool *) mem_26555.mem)[i_26468 * (int64_t) 6 + i_26466];
            double eta_p_24912 = ((double *) mem_26564)[i_26466 * (int64_t) 4 + i_26468];
            double eta_p_24913 = ((double *) mem_26698)[i_26466];
            
            // test/test_sparse_jacobian_jvp.fut:14:25-48
            
            double lifted_lambda_res_24914;
            
            if (eta_p_24911) {
                lifted_lambda_res_24914 = eta_p_24912;
            } else {
                lifted_lambda_res_24914 = 0.0;
            }
            // test/test_sparse_jacobian_jvp.fut:9:48-51
            
            double abs_arg0_24916 = eta_p_24913 - lifted_lambda_res_24914;
            
            // test/test_sparse_jacobian_jvp.fut:9:37-51
            
            double abs_res_24917 = fabs64(abs_arg0_24916);
            
            // test/test_sparse_jacobian_jvp.fut:9:53-59
            
            bool lifted_lambda_res_24918 = abs_res_24917 <= 1.0e-9;
            
            // test/test_sparse_jacobian_jvp.fut:9:5-66
            
            bool x_24901 = lifted_lambda_res_24918 && redout_26465;
            bool redout_tmp_26807 = x_24901;
            
            redout_26465 = redout_tmp_26807;
        }
        defunc_0_reduce_res_26028 = redout_26465;
        // test/test_sparse_jacobian_jvp.fut:10:6-39
        
        bool x_24614 = defunc_0_reduce_res_26028 && redout_26467;
        bool redout_tmp_26804 = x_24614;
        
        redout_26467 = redout_tmp_26804;
    }
    defunc_0_reduce_res_26033 = redout_26467;
    if (memblock_unref(ctx, &ext_mem_26575, "ext_mem_26575") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26576, "ext_mem_26576") != 0)
        return 1;
    prim_out_26758 = defunc_0_reduce_res_26033;
    *out_prim_out_26959 = prim_out_26758;
    
  cleanup:
    {
        free(mem_26564);
        free(mem_26569);
        free(mem_26578);
        free(mem_26579);
        free(mem_26588);
        free(mem_26600);
        free(mem_26604);
        free(mem_26621);
        free(mem_26623);
        free(mem_26624);
        free(mem_26644);
        free(mem_26646);
        free(mem_26648);
        free(mem_26656);
        free(mem_26667);
        free(mem_26672);
        free(mem_26698);
        if (memblock_unref(ctx, &mem_param_tmp_26763, "mem_param_tmp_26763") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26658, "mem_26658") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_26598, "mem_param_26598") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26663, "ext_mem_26663") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26591, "ext_mem_26591") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26592, "ext_mem_26592") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26589, "mem_26589") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26587, "mem_26587") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26575, "ext_mem_26575") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26576, "ext_mem_26576") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_jvp_ex4_with_colors_matches_dense(struct futhark_context *ctx, bool *out_prim_out_26977, struct memblock x_mem_26563)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_26564_cached_sizze_26978 = 0;
    unsigned char *mem_26564 = NULL;
    int64_t mem_26569_cached_sizze_26979 = 0;
    unsigned char *mem_26569 = NULL;
    int64_t mem_26578_cached_sizze_26980 = 0;
    unsigned char *mem_26578 = NULL;
    int64_t mem_26581_cached_sizze_26981 = 0;
    unsigned char *mem_26581 = NULL;
    int64_t mem_26586_cached_sizze_26982 = 0;
    unsigned char *mem_26586 = NULL;
    int64_t mem_26612_cached_sizze_26983 = 0;
    unsigned char *mem_26612 = NULL;
    struct memblock ext_mem_26575;
    
    ext_mem_26575.references = NULL;
    
    struct memblock ext_mem_26576;
    
    ext_mem_26576.references = NULL;
    
    struct memblock mem_26545 = ctx->constants->mem_26545;
    struct memblock mem_26548 = ctx->constants->mem_26548;
    struct memblock mem_26550 = ctx->constants->mem_26550;
    struct memblock mem_26555 = ctx->constants->mem_26555;
    struct memblock mem_26556 = ctx->constants->mem_26556;
    struct memblock mem_26562 = ctx->constants->mem_26562;
    bool prim_out_26758;
    
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_22552;
    int64_t csr_bipartite_from_pattern_res_22553;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26576, &ext_mem_26575, &csr_bipartite_from_pattern_res_22552, &csr_bipartite_from_pattern_res_22553, mem_26555, (int64_t) 4, (int64_t) 6) != 0) {
        err = 1;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    
    int64_t bytes_26577 = (int64_t) 8 * csr_bipartite_from_pattern_res_22553;
    
    // src/sparse_jacobian_jvp.fut:182:8-189:40
    
    int64_t x_25986;
    int64_t redout_26388 = (int64_t) 0;
    
    for (int64_t i_26389 = 0; i_26389 < (int64_t) 6; i_26389++) {
        int64_t x_22560 = ((int64_t *) mem_26556.mem)[i_26389];
        
        // src/sparse_jacobian_jvp.fut:14:22-29
        
        int64_t max_res_22563 = smax64(x_22560, redout_26388);
        int64_t redout_tmp_26759 = max_res_22563;
        
        redout_26388 = redout_tmp_26759;
    }
    x_25986 = redout_26388;
    // src/sparse_jacobian_jvp.fut:14:13-41
    
    int64_t num_colors_of_res_f_res_22564 = add64((int64_t) 1, x_25986);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool bounds_invalid_upwards_22566 = slt64(num_colors_of_res_f_res_22564, (int64_t) 0);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool valid_22567 = !bounds_invalid_upwards_22566;
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool range_valid_c_22568;
    
    if (!valid_22567) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Range ", (long long) (int64_t) 0, "..", (long long) (int64_t) 1, "..<", (long long) num_colors_of_res_f_res_22564, " is invalid.", "-> #0  src/sparse_jacobian_jvp.fut:29:11-18\n   #1  src/sparse_jacobian_jvp.fut:182:8-189:40\n   #2  src/sparse_jacobian_jvp.fut:195:8-200:50\n   #3  src/sparse_jacobian_jvp.fut:207:8-213:43\n   #4  test/test_sparse_jacobian_jvp.fut:81:15-114:71\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    
    int64_t bytes_26580 = (int64_t) 32 * num_colors_of_res_f_res_22564;
    double zt_lhs_24490 = ((double *) x_mem_26563.mem)[(int64_t) 0];
    double zt_rhs_24491 = ((double *) x_mem_26563.mem)[(int64_t) 1];
    double zt_rhs_24493 = ((double *) x_mem_26563.mem)[(int64_t) 4];
    
    // test/test_sparse_jacobian_jvp.fut:82:33-39
    
    double binop_x_25872 = 0.0 * zt_rhs_24493;
    double zt_lhs_24497 = ((double *) x_mem_26563.mem)[(int64_t) 5];
    
    // src/dense_jacobian.fut:8:40-9:68
    if (mem_26564_cached_sizze_26978 < (int64_t) 192) {
        err = lexical_realloc(ctx, &mem_26564, &mem_26564_cached_sizze_26978, (int64_t) 192);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:5:3-21
    if (mem_26569_cached_sizze_26979 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26569, &mem_26569_cached_sizze_26979, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:8:40-9:68
    for (int64_t i_26386 = 0; i_26386 < (int64_t) 6; i_26386++) {
        // src/dense_jacobian.fut:5:3-21
        for (int64_t nest_i_26761 = 0; nest_i_26761 < (int64_t) 6; nest_i_26761++) {
            ((double *) mem_26569)[nest_i_26761] = 0.0;
        }
        // src/dense_jacobian.fut:5:3-39
        ((double *) mem_26569)[i_26386] = 1.0;
        
        double zt_lhs_tan_25865 = ((double *) mem_26569)[(int64_t) 0];
        double zt_rhs_tan_25866 = ((double *) mem_26569)[(int64_t) 1];
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_x_25868 = zt_rhs_24491 * zt_lhs_tan_25865;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_y_25869 = zt_lhs_24490 * zt_rhs_tan_25866;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double zp_lhs_tan_25867 = binop_x_25868 + binop_y_25869;
        double zt_rhs_tan_25870 = ((double *) mem_26569)[(int64_t) 4];
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double binop_y_25873 = 3.0 * zt_rhs_tan_25870;
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double zp_rhs_tan_25871 = binop_x_25872 + binop_y_25873;
        
        // test/test_sparse_jacobian_jvp.fut:82:24-39
        
        double y0_tan_25874 = zp_lhs_tan_25867 + zp_rhs_tan_25871;
        double zm_lhs_tan_25877 = ((double *) mem_26569)[(int64_t) 2];
        double zt_lhs_tan_25878 = ((double *) mem_26569)[(int64_t) 5];
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double binop_x_25880 = zt_lhs_24497 * zt_lhs_tan_25878;
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double zm_rhs_tan_25879 = binop_x_25880 + binop_x_25880;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double binop_y_25884 = -1.0 * zm_rhs_tan_25879;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double y1_tan_25882 = zm_lhs_tan_25877 + binop_y_25884;
        
        // test/test_sparse_jacobian_jvp.fut:84:17-23
        
        double zp_lhs_tan_25885 = zt_rhs_tan_25866 + zm_lhs_tan_25877;
        double zp_rhs_tan_25888 = ((double *) mem_26569)[(int64_t) 3];
        
        // test/test_sparse_jacobian_jvp.fut:84:24-30
        
        double y2_tan_25889 = zp_lhs_tan_25885 + zp_rhs_tan_25888;
        
        // test/test_sparse_jacobian_jvp.fut:113:59-64
        ((double *) mem_26564)[i_26386 * (int64_t) 4] = y0_tan_25874;
        ((double *) mem_26564)[i_26386 * (int64_t) 4 + (int64_t) 1] = y1_tan_25882;
        ((double *) mem_26564)[i_26386 * (int64_t) 4 + (int64_t) 2] = y2_tan_25889;
        ((double *) mem_26564)[i_26386 * (int64_t) 4 + (int64_t) 3] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    if (mem_26578_cached_sizze_26980 < bytes_26577) {
        err = lexical_realloc(ctx, &mem_26578, &mem_26578_cached_sizze_26980, bytes_26577);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    for (int64_t nest_i_26762 = 0; nest_i_26762 < csr_bipartite_from_pattern_res_22553; nest_i_26762++) {
        ((double *) mem_26578)[nest_i_26762] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    if (mem_26581_cached_sizze_26981 < bytes_26580) {
        err = lexical_realloc(ctx, &mem_26581, &mem_26581_cached_sizze_26981, bytes_26580);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:17:3-61
    if (mem_26586_cached_sizze_26982 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26586, &mem_26586_cached_sizze_26982, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    for (int64_t i_26396 = 0; i_26396 < num_colors_of_res_f_res_22564; i_26396++) {
        // src/sparse_jacobian_jvp.fut:17:3-61
        for (int64_t i_26392 = 0; i_26392 < (int64_t) 6; i_26392++) {
            int64_t eta_p_22577 = ((int64_t *) mem_26556.mem)[i_26392];
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            bool cond_22578 = eta_p_22577 == i_26396;
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            double lifted_lambda_res_22579;
            
            if (cond_22578) {
                lifted_lambda_res_22579 = 1.0;
            } else {
                lifted_lambda_res_22579 = 0.0;
            }
            ((double *) mem_26586)[i_26392] = lifted_lambda_res_22579;
        }
        
        double zt_lhs_tan_25894 = ((double *) mem_26586)[(int64_t) 0];
        double zt_rhs_tan_25895 = ((double *) mem_26586)[(int64_t) 1];
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_x_25897 = zt_rhs_24491 * zt_lhs_tan_25894;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double binop_y_25898 = zt_lhs_24490 * zt_rhs_tan_25895;
        
        // test/test_sparse_jacobian_jvp.fut:82:17-23
        
        double zp_lhs_tan_25896 = binop_x_25897 + binop_y_25898;
        double zt_rhs_tan_25899 = ((double *) mem_26586)[(int64_t) 4];
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double binop_y_25902 = 3.0 * zt_rhs_tan_25899;
        
        // test/test_sparse_jacobian_jvp.fut:82:33-39
        
        double zp_rhs_tan_25900 = binop_x_25872 + binop_y_25902;
        
        // test/test_sparse_jacobian_jvp.fut:82:24-39
        
        double y0_tan_25903 = zp_lhs_tan_25896 + zp_rhs_tan_25900;
        double zm_lhs_tan_25906 = ((double *) mem_26586)[(int64_t) 2];
        double zt_lhs_tan_25907 = ((double *) mem_26586)[(int64_t) 5];
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double binop_x_25909 = zt_lhs_24497 * zt_lhs_tan_25907;
        
        // test/test_sparse_jacobian_jvp.fut:83:24-30
        
        double zm_rhs_tan_25908 = binop_x_25909 + binop_x_25909;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double binop_y_25913 = -1.0 * zm_rhs_tan_25908;
        
        // test/test_sparse_jacobian_jvp.fut:83:17-30
        
        double y1_tan_25911 = zm_lhs_tan_25906 + binop_y_25913;
        
        // test/test_sparse_jacobian_jvp.fut:84:17-23
        
        double zp_lhs_tan_25914 = zt_rhs_tan_25895 + zm_lhs_tan_25906;
        double zp_rhs_tan_25917 = ((double *) mem_26586)[(int64_t) 3];
        
        // test/test_sparse_jacobian_jvp.fut:84:24-30
        
        double y2_tan_25918 = zp_lhs_tan_25914 + zp_rhs_tan_25917;
        
        // test/test_sparse_jacobian_jvp.fut:114:45-50
        ((double *) mem_26581)[i_26396 * (int64_t) 4] = y0_tan_25903;
        ((double *) mem_26581)[i_26396 * (int64_t) 4 + (int64_t) 1] = y1_tan_25911;
        ((double *) mem_26581)[i_26396 * (int64_t) 4 + (int64_t) 2] = y2_tan_25918;
        ((double *) mem_26581)[i_26396 * (int64_t) 4 + (int64_t) 3] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:45:5-52:27
    
    bool compressed_to_csr_vals_res_22588;
    int64_t compressed_to_csr_vals_res_22590;
    bool loop_while_22591;
    int64_t i_22593;
    
    loop_while_22591 = 1;
    i_22593 = (int64_t) 0;
    while (loop_while_22591) {
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool x_22594 = sle64((int64_t) 0, i_22593);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool y_22595 = slt64(i_22593, csr_bipartite_from_pattern_res_22552);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool bounds_check_22596 = x_22594 && y_22595;
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool index_certs_22597;
        
        if (!bounds_check_22596) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_22593, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_22552, "].", "-> #0  src/sparse_jacobian_jvp.fut:47:15-26\n   #1  src/sparse_jacobian_jvp.fut:195:18-202:63\n   #2  src/sparse_jacobian_jvp.fut:207:8-213:43\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-114:71\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        int64_t s_22598 = ((int64_t *) ext_mem_26576.mem)[i_22593];
        
        // src/sparse_jacobian_jvp.fut:48:25-27
        
        int64_t e_22599 = add64((int64_t) 1, i_22593);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool x_22600 = sle64((int64_t) 0, e_22599);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool y_22601 = slt64(e_22599, csr_bipartite_from_pattern_res_22552);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool bounds_check_22602 = x_22600 && y_22601;
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool index_certs_22603;
        
        if (!bounds_check_22602) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_22599, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_22552, "].", "-> #0  src/sparse_jacobian_jvp.fut:48:15-28\n   #1  src/sparse_jacobian_jvp.fut:195:18-202:63\n   #2  src/sparse_jacobian_jvp.fut:207:8-213:43\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-114:71\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        int64_t e_22604 = ((int64_t *) ext_mem_26576.mem)[e_22599];
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t j_m_i_22605 = sub64(e_22604, s_22598);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool empty_slice_22606 = j_m_i_22605 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t m_22607 = sub64(j_m_i_22605, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t i_p_m_t_s_22608 = add64(s_22598, m_22607);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_22609 = sle64((int64_t) 0, i_p_m_t_s_22608);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_22610 = slt64(i_p_m_t_s_22608, csr_bipartite_from_pattern_res_22553);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_lte_i_22611 = sle64((int64_t) 0, s_22598);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_lte_j_22612 = sle64(s_22598, e_22604);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_22613 = i_p_m_t_s_leq_w_22610 && zzero_lte_i_22611;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_22614 = zzero_leq_i_p_m_t_s_22609 && y_22613;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool forwards_ok_22615 = i_lte_j_22612 && y_22614;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool ok_or_empty_22616 = empty_slice_22606 || forwards_ok_22615;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool index_certs_22617;
        
        if (!ok_or_empty_22616) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_22598, ":", (long long) e_22604, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_22553, "].", "-> #0  src/sparse_jacobian_jvp.fut:49:18-30\n   #1  src/sparse_jacobian_jvp.fut:195:18-202:63\n   #2  src/sparse_jacobian_jvp.fut:207:8-213:43\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-114:71\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool y_22619 = slt64(i_22593, (int64_t) 4);
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool bounds_check_22620 = x_22594 && y_22619;
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool index_certs_22621;
        
        if (!bounds_check_22620) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_22593, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-45\n   #1  src/sparse_jacobian_jvp.fut:195:18-202:63\n   #2  src/sparse_jacobian_jvp.fut:207:8-213:43\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-114:71\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:18-51
        for (int64_t i_26400 = 0; i_26400 < j_m_i_22605; i_26400++) {
            int64_t index_primexp_26496 = s_22598 + i_26400;
            int64_t eta_p_22623 = ((int64_t *) ext_mem_26575.mem)[index_primexp_26496];
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool x_22624 = sle64((int64_t) 0, eta_p_22623);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool y_22625 = slt64(eta_p_22623, (int64_t) 6);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool bounds_check_22626 = x_22624 && y_22625;
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool index_certs_22627;
            
            if (!bounds_check_22626) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_22623, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:32-41\n   #1  src/sparse_jacobian_jvp.fut:195:18-202:63\n   #2  src/sparse_jacobian_jvp.fut:207:8-213:43\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-114:71\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            int64_t tmp_22628 = ((int64_t *) mem_26556.mem)[eta_p_22623];
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool x_22629 = sle64((int64_t) 0, tmp_22628);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool y_22630 = slt64(tmp_22628, num_colors_of_res_f_res_22564);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool bounds_check_22631 = x_22629 && y_22630;
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool index_certs_22632;
            
            if (!bounds_check_22631) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_22628, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_22564, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-42\n   #1  src/sparse_jacobian_jvp.fut:195:18-202:63\n   #2  src/sparse_jacobian_jvp.fut:207:8-213:43\n   #3  test/test_sparse_jacobian_jvp.fut:81:15-114:71\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:18-51
            
            double lifted_lambda_res_22633 = ((double *) mem_26581)[tmp_22628 * (int64_t) 4 + i_22593];
            
            ((double *) mem_26578)[s_22598 + i_26400] = lifted_lambda_res_22633;
        }
        // src/sparse_jacobian_jvp.fut:46:13-16
        
        bool loop_cond_22635 = slt64(e_22599, (int64_t) 4);
        bool loop_while_tmp_26765 = loop_cond_22635;
        int64_t i_tmp_26767 = e_22599;
        
        loop_while_22591 = loop_while_tmp_26765;
        i_22593 = i_tmp_26767;
    }
    compressed_to_csr_vals_res_22588 = loop_while_22591;
    compressed_to_csr_vals_res_22590 = i_22593;
    // src/sparse_jacobian_jvp.fut:67:29-47
    if (mem_26612_cached_sizze_26983 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_26612, &mem_26612_cached_sizze_26983, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:8:19-10:39
    
    bool defunc_0_reduce_res_25988;
    bool redout_26406 = 1;
    
    for (int64_t i_26407 = 0; i_26407 < (int64_t) 4; i_26407++) {
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool y_24830 = slt64(i_26407, csr_bipartite_from_pattern_res_22552);
        
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool index_certs_24832;
        
        if (!y_24830) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_26407, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_22552, "].", "-> #0  src/sparse_jacobian_jvp.fut:63:17-28\n   #1  src/sparse_jacobian_jvp.fut:207:18-214:40\n   #2  test/test_sparse_jacobian_jvp.fut:81:15-114:71\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        int64_t s_24833 = ((int64_t *) ext_mem_26576.mem)[i_26407];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_lte_i_24846 = sle64((int64_t) 0, s_24833);
        
        // src/sparse_jacobian_jvp.fut:64:27-29
        
        int64_t e_24834 = add64((int64_t) 1, i_26407);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool y_24836 = slt64(e_24834, csr_bipartite_from_pattern_res_22552);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool x_24835 = sle64((int64_t) 0, e_24834);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool bounds_check_24837 = x_24835 && y_24836;
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool index_certs_24838;
        
        if (!bounds_check_24837) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_24834, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_22552, "].", "-> #0  src/sparse_jacobian_jvp.fut:64:17-30\n   #1  src/sparse_jacobian_jvp.fut:207:18-214:40\n   #2  test/test_sparse_jacobian_jvp.fut:81:15-114:71\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        int64_t e_24839 = ((int64_t *) ext_mem_26576.mem)[e_24834];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t j_m_i_24840 = sub64(e_24839, s_24833);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t m_24842 = sub64(j_m_i_24840, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t i_p_m_t_s_24843 = add64(s_24833, m_24842);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_p_m_t_s_leq_w_24845 = slt64(i_p_m_t_s_24843, csr_bipartite_from_pattern_res_22553);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24854 = i_p_m_t_s_leq_w_24845 && zzero_lte_i_24846;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_leq_i_p_m_t_s_24844 = sle64((int64_t) 0, i_p_m_t_s_24843);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24855 = zzero_leq_i_p_m_t_s_24844 && y_24854;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_lte_j_24847 = sle64(s_24833, e_24839);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool forwards_ok_24856 = i_lte_j_24847 && y_24855;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool empty_slice_24841 = j_m_i_24840 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool ok_or_empty_24857 = empty_slice_24841 || forwards_ok_24856;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool index_certs_24858;
        
        if (!ok_or_empty_24857) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_24833, ":", (long long) e_24839, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_22553, "].", "-> #0  src/sparse_jacobian_jvp.fut:66:20-29\n   #1  src/sparse_jacobian_jvp.fut:207:18-214:40\n   #2  test/test_sparse_jacobian_jvp.fut:81:15-114:71\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:67:29-47
        for (int64_t nest_i_26770 = 0; nest_i_26770 < (int64_t) 6; nest_i_26770++) {
            ((double *) mem_26612)[nest_i_26770] = 0.0;
        }
        // src/sparse_jacobian_jvp.fut:68:12-32
        
        bool acc_cert_24862;
        
        // src/sparse_jacobian_jvp.fut:68:12-32
        for (int64_t i_26403 = 0; i_26403 < j_m_i_24840; i_26403++) {
            int64_t index_primexp_26493 = s_24833 + i_26403;
            int64_t v_24866 = ((int64_t *) ext_mem_26575.mem)[index_primexp_26493];
            double v_24867 = ((double *) mem_26578)[index_primexp_26493];
            
            // src/sparse_jacobian_jvp.fut:68:12-32
            // UpdateAcc
            if (sle64((int64_t) 0, v_24866) && slt64(v_24866, (int64_t) 6)) {
                ((double *) mem_26612)[v_24866] = v_24867;
            }
        }
        // test/test_sparse_jacobian_jvp.fut:9:5-66
        
        bool defunc_0_reduce_res_25984;
        bool redout_26404 = 1;
        
        for (int64_t i_26405 = 0; i_26405 < (int64_t) 6; i_26405++) {
            bool eta_p_24890 = ((bool *) mem_26555.mem)[i_26407 * (int64_t) 6 + i_26405];
            double eta_p_24891 = ((double *) mem_26564)[i_26405 * (int64_t) 4 + i_26407];
            double eta_p_24892 = ((double *) mem_26612)[i_26405];
            
            // test/test_sparse_jacobian_jvp.fut:14:25-48
            
            double lifted_lambda_res_24893;
            
            if (eta_p_24890) {
                lifted_lambda_res_24893 = eta_p_24891;
            } else {
                lifted_lambda_res_24893 = 0.0;
            }
            // test/test_sparse_jacobian_jvp.fut:9:48-51
            
            double abs_arg0_24895 = eta_p_24892 - lifted_lambda_res_24893;
            
            // test/test_sparse_jacobian_jvp.fut:9:37-51
            
            double abs_res_24896 = fabs64(abs_arg0_24895);
            
            // test/test_sparse_jacobian_jvp.fut:9:53-59
            
            bool lifted_lambda_res_24897 = abs_res_24896 <= 1.0e-9;
            
            // test/test_sparse_jacobian_jvp.fut:9:5-66
            
            bool x_24880 = lifted_lambda_res_24897 && redout_26404;
            bool redout_tmp_26772 = x_24880;
            
            redout_26404 = redout_tmp_26772;
        }
        defunc_0_reduce_res_25984 = redout_26404;
        // test/test_sparse_jacobian_jvp.fut:10:6-39
        
        bool x_24614 = defunc_0_reduce_res_25984 && redout_26406;
        bool redout_tmp_26769 = x_24614;
        
        redout_26406 = redout_tmp_26769;
    }
    defunc_0_reduce_res_25988 = redout_26406;
    if (memblock_unref(ctx, &ext_mem_26575, "ext_mem_26575") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26576, "ext_mem_26576") != 0)
        return 1;
    prim_out_26758 = defunc_0_reduce_res_25988;
    *out_prim_out_26977 = prim_out_26758;
    
  cleanup:
    {
        free(mem_26564);
        free(mem_26569);
        free(mem_26578);
        free(mem_26581);
        free(mem_26586);
        free(mem_26612);
        if (memblock_unref(ctx, &ext_mem_26575, "ext_mem_26575") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26576, "ext_mem_26576") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_jvp_ex5_matches_dense(struct futhark_context *ctx, bool *out_prim_out_26984, struct memblock x_mem_26563)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_26564_cached_sizze_26985 = 0;
    unsigned char *mem_26564 = NULL;
    int64_t mem_26569_cached_sizze_26986 = 0;
    unsigned char *mem_26569 = NULL;
    int64_t mem_26578_cached_sizze_26987 = 0;
    unsigned char *mem_26578 = NULL;
    int64_t mem_26579_cached_sizze_26988 = 0;
    unsigned char *mem_26579 = NULL;
    int64_t mem_26588_cached_sizze_26989 = 0;
    unsigned char *mem_26588 = NULL;
    int64_t mem_26600_cached_sizze_26990 = 0;
    unsigned char *mem_26600 = NULL;
    int64_t mem_26604_cached_sizze_26991 = 0;
    unsigned char *mem_26604 = NULL;
    int64_t mem_26621_cached_sizze_26992 = 0;
    unsigned char *mem_26621 = NULL;
    int64_t mem_26623_cached_sizze_26993 = 0;
    unsigned char *mem_26623 = NULL;
    int64_t mem_26624_cached_sizze_26994 = 0;
    unsigned char *mem_26624 = NULL;
    int64_t mem_26644_cached_sizze_26995 = 0;
    unsigned char *mem_26644 = NULL;
    int64_t mem_26646_cached_sizze_26996 = 0;
    unsigned char *mem_26646 = NULL;
    int64_t mem_26648_cached_sizze_26997 = 0;
    unsigned char *mem_26648 = NULL;
    int64_t mem_26656_cached_sizze_26998 = 0;
    unsigned char *mem_26656 = NULL;
    int64_t mem_26667_cached_sizze_26999 = 0;
    unsigned char *mem_26667 = NULL;
    int64_t mem_26672_cached_sizze_27000 = 0;
    unsigned char *mem_26672 = NULL;
    int64_t mem_26698_cached_sizze_27001 = 0;
    unsigned char *mem_26698 = NULL;
    struct memblock mem_param_tmp_26763;
    
    mem_param_tmp_26763.references = NULL;
    
    struct memblock mem_26658;
    
    mem_26658.references = NULL;
    
    struct memblock mem_param_26598;
    
    mem_param_26598.references = NULL;
    
    struct memblock ext_mem_26663;
    
    ext_mem_26663.references = NULL;
    
    struct memblock ext_mem_26591;
    
    ext_mem_26591.references = NULL;
    
    struct memblock ext_mem_26592;
    
    ext_mem_26592.references = NULL;
    
    struct memblock mem_26589;
    
    mem_26589.references = NULL;
    
    struct memblock mem_26587;
    
    mem_26587.references = NULL;
    
    struct memblock ext_mem_26575;
    
    ext_mem_26575.references = NULL;
    
    struct memblock ext_mem_26576;
    
    ext_mem_26576.references = NULL;
    
    struct memblock mem_26545 = ctx->constants->mem_26545;
    struct memblock mem_26548 = ctx->constants->mem_26548;
    struct memblock mem_26550 = ctx->constants->mem_26550;
    struct memblock mem_26555 = ctx->constants->mem_26555;
    struct memblock mem_26556 = ctx->constants->mem_26556;
    struct memblock mem_26562 = ctx->constants->mem_26562;
    bool prim_out_26758;
    
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_24074;
    int64_t csr_bipartite_from_pattern_res_24075;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26576, &ext_mem_26575, &csr_bipartite_from_pattern_res_24074, &csr_bipartite_from_pattern_res_24075, mem_26562, (int64_t) 5, (int64_t) 5) != 0) {
        err = 1;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    
    int64_t bytes_26577 = (int64_t) 8 * csr_bipartite_from_pattern_res_24075;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (mem_26579_cached_sizze_26988 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_26579, &mem_26579_cached_sizze_26988, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t discard_26393;
    int64_t scanacc_26389 = (int64_t) 0;
    
    for (int64_t i_26391 = 0; i_26391 < (int64_t) 5; i_26391++) {
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t defunc_0_op_res_24090 = add64((int64_t) 1, scanacc_26389);
        
        ((int64_t *) mem_26579)[i_26391] = defunc_0_op_res_24090;
        
        int64_t scanacc_tmp_26759 = defunc_0_op_res_24090;
        
        scanacc_26389 = scanacc_tmp_26759;
    }
    discard_26393 = scanacc_26389;
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t x_24100 = ((int64_t *) mem_26579)[(int64_t) 4];
    
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    bool loop_cond_24114 = slt64((int64_t) 0, x_24100);
    
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t bytes_26586 = (int64_t) 8 * x_24100;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (memblock_alloc(ctx, &mem_26587, bytes_26586, "mem_26587")) {
        err = 1;
        goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    bool acc_cert_24821;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    for (int64_t i_26395 = 0; i_26395 < (int64_t) 5; i_26395++) {
        int64_t eta_p_24833 = ((int64_t *) mem_26579)[i_26395];
        
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t lifted_lambda_res_t_res_24836 = sub64(eta_p_24833, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:194:20-45
        // UpdateAcc
        if (sle64((int64_t) 0, lifted_lambda_res_t_res_24836) && slt64(lifted_lambda_res_t_res_24836, x_24100)) {
            ((int64_t *) mem_26587.mem)[lifted_lambda_res_t_res_24836] = i_26395;
        }
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    if (mem_26588_cached_sizze_26989 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_26588, &mem_26588_cached_sizze_26989, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    for (int64_t nest_i_26762 = 0; nest_i_26762 < (int64_t) 5; nest_i_26762++) {
        ((int64_t *) mem_26588)[nest_i_26762] = (int64_t) -1;
    }
    // src/pattern_csr.fut:25:3-39
    if (memblock_alloc(ctx, &mem_26589, (int64_t) 25, "mem_26589")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:25:3-39
    // src/pattern_csr.fut:25:3-39
    lmad_copy_1b(ctx, 2, (uint8_t *) mem_26589.mem, (int64_t) 0, (int64_t []) {(int64_t) 5, (int64_t) 1}, (uint8_t *) mem_26562.mem, (int64_t) 0, (int64_t []) {(int64_t) 1, (int64_t) 5}, (int64_t []) {(int64_t) 5, (int64_t) 5});
    // src/pattern_csr.fut:25:3-39
    
    int64_t csr_cols_from_pattern_res_24079;
    int64_t csr_cols_from_pattern_res_24080;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26592, &ext_mem_26591, &csr_cols_from_pattern_res_24079, &csr_cols_from_pattern_res_24080, mem_26589, (int64_t) 5, (int64_t) 5) != 0) {
        err = 1;
        goto cleanup;
    }
    if (memblock_unref(ctx, &mem_26589, "mem_26589") != 0)
        return 1;
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    int64_t vv_color_side_order_res_24115;
    bool vv_color_side_order_res_24116;
    int64_t vv_color_side_order_res_24119;
    int64_t loop_dz2081Uz2083U_24120;
    bool loop_while_24121;
    int64_t color_bound_24124;
    
    if (memblock_set(ctx, &mem_param_26598, &mem_26587, "mem_26587") != 0)
        return 1;
    loop_dz2081Uz2083U_24120 = x_24100;
    loop_while_24121 = loop_cond_24114;
    color_bound_24124 = (int64_t) 1;
    while (loop_while_24121) {
        // src/bgpc_vv_coloring.fut:62:5-64:15
        
        bool cond_24125 = slt64((int64_t) 0, color_bound_24124);
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t bytes_26599 = (int64_t) 8 * loop_dz2081Uz2083U_24120;
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        if (mem_26600_cached_sizze_26990 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26600, &mem_26600_cached_sizze_26990, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:77:24-51
        if (mem_26604_cached_sizze_26991 < color_bound_24124) {
            err = lexical_realloc(ctx, &mem_26604, &mem_26604_cached_sizze_26991, color_bound_24124);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t defunc_0_reduce_res_26019;
        int64_t redout_26397 = (int64_t) -1;
        
        for (int64_t i_26399 = 0; i_26399 < loop_dz2081Uz2083U_24120; i_26399++) {
            int64_t eta_p_25358 = ((int64_t *) mem_param_26598.mem)[i_26399];
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool x_25360 = sle64((int64_t) 0, eta_p_25358);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool y_25361 = slt64(eta_p_25358, csr_cols_from_pattern_res_24079);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool bounds_check_25362 = x_25360 && y_25361;
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool index_certs_25363;
            
            if (!bounds_check_25362) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25358, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24079, "].", "-> #0  src/bgpc_vv_coloring.fut:46:44-56\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:44-56
            
            int64_t seen_final_25364 = ((int64_t *) ext_mem_26592.mem)[eta_p_25358];
            
            // src/bgpc_vv_coloring.fut:46:70-72
            
            int64_t seen_final_25365 = add64((int64_t) 1, eta_p_25358);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool x_25366 = sle64((int64_t) 0, seen_final_25365);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool y_25367 = slt64(seen_final_25365, csr_cols_from_pattern_res_24079);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool bounds_check_25368 = x_25366 && y_25367;
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool index_certs_25369;
            
            if (!bounds_check_25368) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_25365, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24079, "].", "-> #0  src/bgpc_vv_coloring.fut:46:59-73\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:59-73
            
            int64_t seen_final_25370 = ((int64_t *) ext_mem_26592.mem)[seen_final_25365];
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t j_m_i_25371 = sub64(seen_final_25370, seen_final_25364);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool empty_slice_25372 = j_m_i_25371 == (int64_t) 0;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t m_25373 = sub64(j_m_i_25371, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t i_p_m_t_s_25374 = add64(seen_final_25364, m_25373);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_leq_i_p_m_t_s_25375 = sle64((int64_t) 0, i_p_m_t_s_25374);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_p_m_t_s_leq_w_25376 = slt64(i_p_m_t_s_25374, csr_cols_from_pattern_res_24080);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_lte_i_25377 = sle64((int64_t) 0, seen_final_25364);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_lte_j_25378 = sle64(seen_final_25364, seen_final_25370);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_25379 = i_p_m_t_s_leq_w_25376 && zzero_lte_i_25377;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_25380 = zzero_leq_i_p_m_t_s_25375 && y_25379;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool forwards_ok_25381 = i_lte_j_25378 && y_25380;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool ok_or_empty_25382 = empty_slice_25372 || forwards_ok_25381;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool index_certs_25383;
            
            if (!ok_or_empty_25382) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_25364, ":", (long long) seen_final_25370, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24080, "].", "-> #0  src/bgpc_vv_coloring.fut:46:5-54:19\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:77:24-51
            for (int64_t nest_i_26771 = 0; nest_i_26771 < color_bound_24124; nest_i_26771++) {
                ((bool *) mem_26604)[nest_i_26771] = 0;
            }
            // src/bgpc_vv_coloring.fut:46:5-54:19
            for (int64_t i_25385 = 0; i_25385 < j_m_i_25371; i_25385++) {
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t index_primexp_25387 = seen_final_25364 + i_25385;
                
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t v_25388 = ((int64_t *) ext_mem_26591.mem)[index_primexp_25387];
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool x_25389 = sle64((int64_t) 0, v_25388);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool y_25390 = slt64(v_25388, csr_bipartite_from_pattern_res_24074);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool bounds_check_25391 = x_25389 && y_25390;
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool index_certs_25392;
                
                if (!bounds_check_25391) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_25388, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24074, "].", "-> #0  src/bgpc_vv_coloring.fut:49:26-37\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:26-37
                
                int64_t seen_acczq_25393 = ((int64_t *) ext_mem_26576.mem)[v_25388];
                
                // src/bgpc_vv_coloring.fut:49:50-52
                
                int64_t seen_acczq_25394 = add64((int64_t) 1, v_25388);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool x_25395 = sle64((int64_t) 0, seen_acczq_25394);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool y_25396 = slt64(seen_acczq_25394, csr_bipartite_from_pattern_res_24074);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool bounds_check_25397 = x_25395 && y_25396;
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool index_certs_25398;
                
                if (!bounds_check_25397) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_25394, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24074, "].", "-> #0  src/bgpc_vv_coloring.fut:49:40-53\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:40-53
                
                int64_t seen_acczq_25399 = ((int64_t *) ext_mem_26576.mem)[seen_acczq_25394];
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t j_m_i_25400 = sub64(seen_acczq_25399, seen_acczq_25393);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool empty_slice_25401 = j_m_i_25400 == (int64_t) 0;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t m_25402 = sub64(j_m_i_25400, (int64_t) 1);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t i_p_m_t_s_25403 = add64(seen_acczq_25393, m_25402);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_leq_i_p_m_t_s_25404 = sle64((int64_t) 0, i_p_m_t_s_25403);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_p_m_t_s_leq_w_25405 = slt64(i_p_m_t_s_25403, csr_bipartite_from_pattern_res_24075);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_lte_i_25406 = sle64((int64_t) 0, seen_acczq_25393);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_lte_j_25407 = sle64(seen_acczq_25393, seen_acczq_25399);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_25408 = i_p_m_t_s_leq_w_25405 && zzero_lte_i_25406;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_25409 = zzero_leq_i_p_m_t_s_25404 && y_25408;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool forwards_ok_25410 = i_lte_j_25407 && y_25409;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool ok_or_empty_25411 = empty_slice_25401 || forwards_ok_25410;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool index_certs_25412;
                
                if (!ok_or_empty_25411) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_25393, ":", (long long) seen_acczq_25399, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24075, "].", "-> #0  src/bgpc_vv_coloring.fut:48:9-53:33\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:48:9-53:33
                for (int64_t i_25414 = 0; i_25414 < j_m_i_25400; i_25414++) {
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t index_primexp_25416 = seen_acczq_25393 + i_25414;
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t u_25417 = ((int64_t *) ext_mem_26575.mem)[index_primexp_25416];
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool x_25418 = sle64((int64_t) 0, u_25417);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool y_25419 = slt64(u_25417, (int64_t) 5);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool bounds_check_25420 = x_25418 && y_25419;
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool index_certs_25421;
                    
                    if (!bounds_check_25420) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_25417, "] out of bounds for array of shape [", (long long) (int64_t) 5, "].", "-> #0  src/bgpc_vv_coloring.fut:50:20-29\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    int64_t cu_25422 = ((int64_t *) mem_26588)[u_25417];
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_25423 = u_25417 == eta_p_25358;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_25424 = !cond_25423;
                    
                    // src/bgpc_vv_coloring.fut:51:30-37
                    
                    bool cond_t_res_25425 = sle64((int64_t) 0, cu_25422);
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    bool x_25426 = cond_25424 && cond_t_res_25425;
                    
                    // src/bgpc_vv_coloring.fut:51:44-67
                    
                    bool cond_t_res_25427 = slt64(cu_25422, color_bound_24124);
                    
                    // src/bgpc_vv_coloring.fut:94:14-103:69
                    
                    bool x_25428 = x_25426 && cond_t_res_25427;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    if (x_25428) {
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool bounds_check_25983 = cond_t_res_25425 && cond_t_res_25427;
                        
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool index_certs_25984;
                        
                        if (!bounds_check_25983) {
                            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) cu_25422, "] out of bounds for array of shape [", (long long) color_bound_24124, "].", "-> #0  src/bgpc_vv_coloring.fut:52:19-50\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                            err = FUTHARK_PROGRAM_ERROR;
                            goto cleanup;
                        }
                        // src/bgpc_vv_coloring.fut:52:19-50
                        ((bool *) mem_26604)[cu_25422] = 1;
                    }
                }
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool loop_cond_25433;
            
            if (cond_24125) {
                // src/bgpc_vv_coloring.fut:63:30-37
                
                bool loop_cond_t_res_25986 = ((bool *) mem_26604)[(int64_t) 0];
                
                loop_cond_25433 = loop_cond_t_res_25986;
            } else {
                loop_cond_25433 = 0;
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool c_final_25435;
            int64_t c_final_25436;
            bool loop_while_25437;
            int64_t c_25438;
            
            loop_while_25437 = loop_cond_25433;
            c_25438 = (int64_t) 0;
            while (loop_while_25437) {
                // src/bgpc_vv_coloring.fut:64:9-15
                
                int64_t loopres_25439 = add64((int64_t) 1, c_25438);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool cond_25440 = slt64(loopres_25439, color_bound_24124);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool loop_cond_25441;
                
                if (cond_25440) {
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool x_25987 = sle64((int64_t) 0, loopres_25439);
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool bounds_check_25988 = cond_25440 && x_25987;
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool index_certs_25989;
                    
                    if (!bounds_check_25988) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) loopres_25439, "] out of bounds for array of shape [", (long long) color_bound_24124, "].", "-> #0  src/bgpc_vv_coloring.fut:63:30-37\n   #1  src/bgpc_vv_coloring.fut:81:6-28\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:213:14-216:64\n   #6  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #7  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #8  src/sparse_jacobian_jvp.fut:158:18-33\n   #9  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool loop_cond_t_res_25990 = ((bool *) mem_26604)[loopres_25439];
                    
                    loop_cond_25441 = loop_cond_t_res_25990;
                } else {
                    loop_cond_25441 = 0;
                }
                
                bool loop_while_tmp_26774 = loop_cond_25441;
                int64_t c_tmp_26775 = loopres_25439;
                
                loop_while_25437 = loop_while_tmp_26774;
                c_25438 = c_tmp_26775;
            }
            c_final_25435 = loop_while_25437;
            c_final_25436 = c_25438;
            // src/bgpc_vv_coloring.fut:106:25-32
            
            int64_t max_res_24219 = smax64(c_final_25436, redout_26397);
            
            ((int64_t *) mem_26600)[i_26399] = c_final_25436;
            
            int64_t redout_tmp_26769 = max_res_24219;
            
            redout_26397 = redout_tmp_26769;
        }
        defunc_0_reduce_res_26019 = redout_26397;
        // src/bgpc_vv_coloring.fut:113:17-43
        
        bool acc_cert_24225;
        
        // src/bgpc_vv_coloring.fut:113:17-43
        for (int64_t i_26402 = 0; i_26402 < loop_dz2081Uz2083U_24120; i_26402++) {
            int64_t v_24229 = ((int64_t *) mem_param_26598.mem)[i_26402];
            int64_t v_24230 = ((int64_t *) mem_26600)[i_26402];
            
            // src/bgpc_vv_coloring.fut:113:17-43
            // UpdateAcc
            if (sle64((int64_t) 0, v_24229) && slt64(v_24229, (int64_t) 5)) {
                ((int64_t *) mem_26588)[v_24229] = v_24230;
            }
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26621_cached_sizze_26992 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26621, &mem_26621_cached_sizze_26992, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26623_cached_sizze_26993 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26623, &mem_26623_cached_sizze_26993, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26624_cached_sizze_26994 < loop_dz2081Uz2083U_24120) {
            err = lexical_realloc(ctx, &mem_26624, &mem_26624_cached_sizze_26994, loop_dz2081Uz2083U_24120);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t discard_26414;
        int64_t scanacc_26406 = (int64_t) 0;
        
        for (int64_t i_26410 = 0; i_26410 < loop_dz2081Uz2083U_24120; i_26410++) {
            int64_t eta_p_25282 = ((int64_t *) mem_param_26598.mem)[i_26410];
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool x_25283 = sle64((int64_t) 0, eta_p_25282);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool y_25284 = slt64(eta_p_25282, csr_cols_from_pattern_res_24079);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool bounds_check_25285 = x_25283 && y_25284;
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool index_certs_25286;
            
            if (!bounds_check_25285) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25282, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24079, "].", "-> #0  src/bgpc_vv_coloring.fut:126:15-27\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:127:26-28
            
            int64_t k_end_25288 = add64((int64_t) 1, eta_p_25282);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool x_25289 = sle64((int64_t) 0, k_end_25288);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool y_25290 = slt64(k_end_25288, csr_cols_from_pattern_res_24079);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool bounds_check_25291 = x_25289 && y_25290;
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool index_certs_25292;
            
            if (!bounds_check_25291) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_end_25288, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24079, "].", "-> #0  src/bgpc_vv_coloring.fut:127:15-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:126:15-27
            
            int64_t k0_25287 = ((int64_t *) ext_mem_26592.mem)[eta_p_25282];
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            int64_t k_end_25293 = ((int64_t *) ext_mem_26592.mem)[k_end_25288];
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool cond_25294 = slt64(k0_25287, k_end_25293);
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool loses_conflict_vertex_res_25295;
            bool loses_conflict_vertex_res_25296;
            int64_t loses_conflict_vertex_res_25297;
            bool loop_while_25298;
            bool lost_25299;
            int64_t k_25300;
            
            loop_while_25298 = cond_25294;
            lost_25299 = 0;
            k_25300 = k0_25287;
            while (loop_while_25298) {
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool x_25301 = sle64((int64_t) 0, k_25300);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool y_25302 = slt64(k_25300, csr_cols_from_pattern_res_24080);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool bounds_check_25303 = x_25301 && y_25302;
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool index_certs_25304;
                
                if (!bounds_check_25303) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_25300, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_24080, "].", "-> #0  src/bgpc_vv_coloring.fut:132:15-26\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:132:15-26
                
                int64_t v_25305 = ((int64_t *) ext_mem_26591.mem)[k_25300];
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool x_25306 = sle64((int64_t) 0, v_25305);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool y_25307 = slt64(v_25305, csr_bipartite_from_pattern_res_24074);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool bounds_check_25308 = x_25306 && y_25307;
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool index_certs_25309;
                
                if (!bounds_check_25308) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_25305, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24074, "].", "-> #0  src/bgpc_vv_coloring.fut:134:19-30\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:135:29-31
                
                int64_t t_end_25311 = add64((int64_t) 1, v_25305);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool x_25312 = sle64((int64_t) 0, t_end_25311);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool y_25313 = slt64(t_end_25311, csr_bipartite_from_pattern_res_24074);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool bounds_check_25314 = x_25312 && y_25313;
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool index_certs_25315;
                
                if (!bounds_check_25314) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_end_25311, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24074, "].", "-> #0  src/bgpc_vv_coloring.fut:135:19-32\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:134:19-30
                
                int64_t t0_25310 = ((int64_t *) ext_mem_26576.mem)[v_25305];
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                int64_t t_end_25316 = ((int64_t *) ext_mem_26576.mem)[t_end_25311];
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool cond_25317 = slt64(t0_25310, t_end_25316);
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool loopres_25318;
                bool loopres_25319;
                int64_t loopres_25320;
                bool loop_while_25321;
                bool lost_in_net_25322;
                int64_t t_25323;
                
                loop_while_25321 = cond_25317;
                lost_in_net_25322 = 0;
                t_25323 = t0_25310;
                while (loop_while_25321) {
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool x_25324 = sle64((int64_t) 0, t_25323);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool y_25325 = slt64(t_25323, csr_bipartite_from_pattern_res_24075);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool bounds_check_25326 = x_25324 && y_25325;
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool index_certs_25327;
                    
                    if (!bounds_check_25326) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_25323, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24075, "].", "-> #0  src/bgpc_vv_coloring.fut:140:19-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    int64_t u_25328 = ((int64_t *) ext_mem_26575.mem)[t_25323];
                    
                    // src/bgpc_vv_coloring.fut:142:13-71
                    
                    bool lost_in_netzq_25329;
                    
                    if (lost_in_net_25322) {
                        lost_in_netzq_25329 = 1;
                    } else {
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25330 = u_25328 == eta_p_25282;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25331 = !cond_25330;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_25332;
                        
                        if (cond_25331) {
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool x_25992 = sle64((int64_t) 0, u_25328);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool y_25993 = slt64(u_25328, (int64_t) 5);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool bounds_check_25994 = x_25992 && y_25993;
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool index_certs_25995;
                            
                            if (!bounds_check_25994) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_25328, "] out of bounds for array of shape [", (long long) (int64_t) 5, "].", "-> #0  src/bgpc_vv_coloring.fut:142:39-48\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool y_25997 = slt64(eta_p_25282, (int64_t) 5);
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool bounds_check_25998 = x_25283 && y_25997;
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool index_certs_25999;
                            
                            if (!bounds_check_25998) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_25282, "] out of bounds for array of shape [", (long long) (int64_t) 5, "].", "-> #0  src/bgpc_vv_coloring.fut:142:52-61\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:213:14-216:64\n   #5  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #6  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #7  src/sparse_jacobian_jvp.fut:158:18-33\n   #8  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            int64_t zeze_lhs_25996 = ((int64_t *) mem_26588)[u_25328];
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            int64_t zeze_rhs_26000 = ((int64_t *) mem_26588)[eta_p_25282];
                            
                            // src/bgpc_vv_coloring.fut:142:49-61
                            
                            bool cond_t_res_26001 = zeze_lhs_25996 == zeze_rhs_26000;
                            
                            cond_25332 = cond_t_res_26001;
                        } else {
                            cond_25332 = 0;
                        }
                        // src/bgpc_vv_coloring.fut:142:67-70
                        
                        bool lost_in_netzq_f_res_t_res_25343 = slt64(u_25328, eta_p_25282);
                        
                        // src/bgpc_vv_coloring.fut:160:14-168:57
                        
                        bool x_25344 = cond_25332 && lost_in_netzq_f_res_t_res_25343;
                        
                        lost_in_netzq_25329 = x_25344;
                    }
                    // src/bgpc_vv_coloring.fut:143:31-37
                    
                    int64_t tmp_25345 = add64((int64_t) 1, t_25323);
                    
                    // src/bgpc_vv_coloring.fut:139:15-43
                    
                    bool cond_25346 = slt64(tmp_25345, t_end_25316);
                    
                    // src/bgpc_vv_coloring.fut:139:28-43
                    
                    bool not_res_25347 = !lost_in_netzq_25329;
                    
                    // src/bgpc_vv_coloring.fut:166:5-169:10
                    
                    bool x_25348 = cond_25346 && not_res_25347;
                    bool loop_while_tmp_26784 = x_25348;
                    bool lost_in_net_tmp_26785 = lost_in_netzq_25329;
                    int64_t t_tmp_26786 = tmp_25345;
                    
                    loop_while_25321 = loop_while_tmp_26784;
                    lost_in_net_25322 = lost_in_net_tmp_26785;
                    t_25323 = t_tmp_26786;
                }
                loopres_25318 = loop_while_25321;
                loopres_25319 = lost_in_net_25322;
                loopres_25320 = t_25323;
                // src/bgpc_vv_coloring.fut:160:14-168:57
                
                bool lostzq_25349 = lost_25299 || loopres_25319;
                
                // src/bgpc_vv_coloring.fut:146:20-26
                
                int64_t tmp_25350 = add64((int64_t) 1, k_25300);
                
                // src/bgpc_vv_coloring.fut:131:11-32
                
                bool cond_25351 = slt64(tmp_25350, k_end_25293);
                
                // src/bgpc_vv_coloring.fut:131:24-32
                
                bool not_res_25352 = !lostzq_25349;
                
                // src/bgpc_vv_coloring.fut:166:5-169:10
                
                bool x_25353 = cond_25351 && not_res_25352;
                bool loop_while_tmp_26781 = x_25353;
                bool lost_tmp_26782 = lostzq_25349;
                int64_t k_tmp_26783 = tmp_25350;
                
                loop_while_25298 = loop_while_tmp_26781;
                lost_25299 = lost_tmp_26782;
                k_25300 = k_tmp_26783;
            }
            loses_conflict_vertex_res_25295 = loop_while_25298;
            loses_conflict_vertex_res_25296 = lost_25299;
            loses_conflict_vertex_res_25297 = k_25300;
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_f_res_25355 = btoi_bool_i64(loses_conflict_vertex_res_25296);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_op_res_24331 = add64(defunc_0_f_res_25355, scanacc_26406);
            
            ((int64_t *) mem_26621)[i_26410] = defunc_0_op_res_24331;
            ((int64_t *) mem_26623)[i_26410] = defunc_0_f_res_25355;
            ((bool *) mem_26624)[i_26410] = loses_conflict_vertex_res_25296;
            
            int64_t scanacc_tmp_26777 = defunc_0_op_res_24331;
            
            scanacc_26406 = scanacc_tmp_26777;
        }
        discard_26414 = scanacc_26406;
        // src/bgpc_vv_coloring.fut:23:25-28
        
        int64_t tmp_24315 = sub64(loop_dz2081Uz2083U_24120, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool y_24317 = slt64(tmp_24315, loop_dz2081Uz2083U_24120);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool x_24316 = sle64((int64_t) 0, tmp_24315);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool bounds_check_24318 = x_24316 && y_24317;
        
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        bool cond_24313 = loop_dz2081Uz2083U_24120 == (int64_t) 0;
        
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool protect_assert_disj_24319 = cond_24313 || bounds_check_24318;
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool index_certs_24320;
        
        if (!protect_assert_disj_24319) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_24315, "] out of bounds for array of shape [", (long long) loop_dz2081Uz2083U_24120, "].", "-> #0  src/bgpc_vv_coloring.fut:23:10-29\n   #1  src/bgpc_vv_coloring.fut:171:23-55\n   #2  src/bgpc_vv_coloring.fut:186:14-205:56\n   #3  src/bgpc_vv_coloring.fut:213:14-216:64\n   #4  src/sparse_jacobian_jvp.fut:76:15-82:56\n   #5  src/sparse_jacobian_jvp.fut:88:10-93:60\n   #6  src/sparse_jacobian_jvp.fut:158:18-33\n   #7  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool x_24314 = !cond_24313;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_f_res_24332;
        
        if (x_24314) {
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t x_26002 = ((int64_t *) mem_26621)[tmp_24315];
            
            m_f_res_24332 = x_26002;
        } else {
            m_f_res_24332 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_24334;
        
        if (cond_24313) {
            m_24334 = (int64_t) 0;
        } else {
            m_24334 = m_f_res_24332;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t bytes_26643 = (int64_t) 8 * m_24334;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26644_cached_sizze_26995 < bytes_26643) {
            err = lexical_realloc(ctx, &mem_26644, &mem_26644_cached_sizze_26995, bytes_26643);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26646_cached_sizze_26996 < bytes_26643) {
            err = lexical_realloc(ctx, &mem_26646, &mem_26646_cached_sizze_26996, bytes_26643);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:18:24-27:44
        if (mem_26648_cached_sizze_26997 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26648, &mem_26648_cached_sizze_26997, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_26656_cached_sizze_26998 < bytes_26599) {
            err = lexical_realloc(ctx, &mem_26656, &mem_26656_cached_sizze_26998, bytes_26599);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        bool acc_cert_25155;
        bool acc_cert_25156;
        
        // src/bgpc_vv_coloring.fut:18:24-27:44
        
        int64_t inpacc_26010;
        int64_t inpacc_25219 = (int64_t) 0;
        
        for (int64_t i_26441 = 0; i_26441 < loop_dz2081Uz2083U_24120; i_26441++) {
            bool eta_p_26503 = ((bool *) mem_26624)[i_26441];
            
            // src/bgpc_vv_coloring.fut:18:28-39
            
            int64_t bool_to_i64_res_26504 = btoi_bool_i64(eta_p_26503);
            int64_t eta_p_26516 = ((int64_t *) mem_26623)[i_26441];
            int64_t eta_p_26517 = ((int64_t *) mem_26621)[i_26441];
            int64_t v_26520 = ((int64_t *) mem_param_26598.mem)[i_26441];
            
            // src/bgpc_vv_coloring.fut:19:29-32
            
            int64_t defunc_0_op_res_26521 = add64(inpacc_25219, bool_to_i64_res_26504);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            bool cond_26522 = eta_p_26516 == (int64_t) 1;
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t lifted_lambda_res_26523;
            
            if (cond_26522) {
                // src/bgpc_vv_coloring.fut:27:5-44
                
                int64_t lifted_lambda_res_t_res_26524 = sub64(eta_p_26517, (int64_t) 1);
                
                lifted_lambda_res_26523 = lifted_lambda_res_t_res_26524;
            } else {
                lifted_lambda_res_26523 = (int64_t) -1;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_26523) && slt64(lifted_lambda_res_26523, m_24334)) {
                ((int64_t *) mem_26646)[lifted_lambda_res_26523] = v_26520;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_26523) && slt64(lifted_lambda_res_26523, m_24334)) {
                ((int64_t *) mem_26644)[lifted_lambda_res_26523] = defunc_0_op_res_26521;
            }
            ((int64_t *) mem_26648)[i_26441] = defunc_0_op_res_26521;
            
            int64_t inpacc_tmp_26787 = defunc_0_op_res_26521;
            
            inpacc_25219 = inpacc_tmp_26787;
        }
        inpacc_26010 = inpacc_25219;
        // src/bgpc_vv_coloring.fut:27:5-44
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_26656, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_26648, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {loop_dz2081Uz2083U_24120});
        // src/bgpc_vv_coloring.fut:23:10-29
        
        int64_t out_szz_f_res_24321;
        
        if (x_24314) {
            // src/bgpc_vv_coloring.fut:171:23-55
            
            int64_t x_26016 = ((int64_t *) mem_26656)[tmp_24315];
            
            out_szz_f_res_24321 = x_26016;
        } else {
            out_szz_f_res_24321 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        int64_t out_szz_24323;
        
        if (cond_24313) {
            out_szz_24323 = (int64_t) 0;
        } else {
            out_szz_24323 = out_szz_f_res_24321;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        
        int64_t bytes_26657 = (int64_t) 8 * out_szz_24323;
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        bool cond_24220 = slt64(defunc_0_reduce_res_26019, (int64_t) 0);
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        int64_t next_color_bound_24221;
        
        if (cond_24220) {
            next_color_bound_24221 = color_bound_24124;
        } else {
            // src/bgpc_vv_coloring.fut:110:40-46
            
            int64_t max_arg1_24222 = add64((int64_t) 2, defunc_0_reduce_res_26019);
            
            // src/bgpc_vv_coloring.fut:110:10-46
            
            int64_t max_res_24223 = smax64(color_bound_24124, max_arg1_24222);
            
            next_color_bound_24221 = max_res_24223;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        if (memblock_alloc(ctx, &mem_26658, bytes_26657, "mem_26658")) {
            err = 1;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        for (int64_t nest_i_26791 = 0; nest_i_26791 < out_szz_24323; nest_i_26791++) {
            ((int64_t *) mem_26658.mem)[nest_i_26791] = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:33:6-27
        
        bool acc_cert_24928;
        
        // src/bgpc_vv_coloring.fut:29:21-33:27
        for (int64_t i_26446 = 0; i_26446 < m_24334; i_26446++) {
            int64_t eta_p_24940 = ((int64_t *) mem_26644)[i_26446];
            int64_t v_24942 = ((int64_t *) mem_26646)[i_26446];
            
            // src/bgpc_vv_coloring.fut:29:44-50
            
            int64_t lifted_lambda_res_24943 = sub64(eta_p_24940, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:33:6-27
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_24943) && slt64(lifted_lambda_res_24943, out_szz_24323)) {
                ((int64_t *) mem_26658.mem)[lifted_lambda_res_24943] = v_24942;
            }
        }
        // src/bgpc_vv_coloring.fut:175:17-48
        
        bool acc_cert_24371;
        
        // src/bgpc_vv_coloring.fut:175:17-48
        for (int64_t i_26448 = 0; i_26448 < out_szz_24323; i_26448++) {
            int64_t v_24375 = ((int64_t *) mem_26658.mem)[i_26448];
            
            // src/bgpc_vv_coloring.fut:175:17-48
            // UpdateAcc
            if (sle64((int64_t) 0, v_24375) && slt64(v_24375, (int64_t) 5)) {
                ((int64_t *) mem_26588)[v_24375] = (int64_t) -1;
            }
        }
        // src/bgpc_vv_coloring.fut:198:20-23
        
        bool loop_cond_24377 = slt64((int64_t) 0, out_szz_24323);
        
        if (memblock_set(ctx, &mem_param_tmp_26763, &mem_26658, "mem_26658") != 0)
            return 1;
        
        int64_t loop_dz2081Uz2083U_tmp_26764 = out_szz_24323;
        bool loop_while_tmp_26765 = loop_cond_24377;
        int64_t color_bound_tmp_26768 = next_color_bound_24221;
        
        if (memblock_set(ctx, &mem_param_26598, &mem_param_tmp_26763, "mem_param_tmp_26763") != 0)
            return 1;
        loop_dz2081Uz2083U_24120 = loop_dz2081Uz2083U_tmp_26764;
        loop_while_24121 = loop_while_tmp_26765;
        color_bound_24124 = color_bound_tmp_26768;
    }
    if (memblock_set(ctx, &ext_mem_26663, &mem_param_26598, "mem_param_26598") != 0)
        return 1;
    vv_color_side_order_res_24115 = loop_dz2081Uz2083U_24120;
    vv_color_side_order_res_24116 = loop_while_24121;
    vv_color_side_order_res_24119 = color_bound_24124;
    if (memblock_unref(ctx, &mem_26587, "mem_26587") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26591, "ext_mem_26591") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26592, "ext_mem_26592") != 0)
        return 1;
    // src/sparse_jacobian_jvp.fut:107:8-112:40
    
    int64_t x_26031;
    int64_t redout_26449 = (int64_t) 0;
    
    for (int64_t i_26450 = 0; i_26450 < (int64_t) 5; i_26450++) {
        int64_t x_24382 = ((int64_t *) mem_26588)[i_26450];
        
        // src/sparse_jacobian_jvp.fut:14:22-29
        
        int64_t max_res_24385 = smax64(x_24382, redout_26449);
        int64_t redout_tmp_26794 = max_res_24385;
        
        redout_26449 = redout_tmp_26794;
    }
    x_26031 = redout_26449;
    // src/sparse_jacobian_jvp.fut:14:13-41
    
    int64_t num_colors_of_res_f_res_24386 = add64((int64_t) 1, x_26031);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool bounds_invalid_upwards_24388 = slt64(num_colors_of_res_f_res_24386, (int64_t) 0);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool valid_24389 = !bounds_invalid_upwards_24388;
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool range_valid_c_24390;
    
    if (!valid_24389) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Range ", (long long) (int64_t) 0, "..", (long long) (int64_t) 1, "..<", (long long) num_colors_of_res_f_res_24386, " is invalid.", "-> #0  src/sparse_jacobian_jvp.fut:29:11-18\n   #1  src/sparse_jacobian_jvp.fut:107:8-112:40\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    
    int64_t bytes_26666 = (int64_t) 40 * num_colors_of_res_f_res_24386;
    double zt_lhs_24490 = ((double *) x_mem_26563.mem)[(int64_t) 0];
    double zp_rhs_24493 = ((double *) x_mem_26563.mem)[(int64_t) 4];
    double zt_lhs_24495 = ((double *) x_mem_26563.mem)[(int64_t) 1];
    double zt_rhs_24496 = ((double *) x_mem_26563.mem)[(int64_t) 3];
    double zp_lhs_24498 = ((double *) x_mem_26563.mem)[(int64_t) 2];
    
    // test/test_sparse_jacobian_jvp.fut:121:26-32
    
    double binop_x_25880 = 0.0 * zp_rhs_24493;
    
    // src/dense_jacobian.fut:8:40-9:68
    if (mem_26564_cached_sizze_26985 < (int64_t) 200) {
        err = lexical_realloc(ctx, &mem_26564, &mem_26564_cached_sizze_26985, (int64_t) 200);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:5:3-21
    if (mem_26569_cached_sizze_26986 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_26569, &mem_26569_cached_sizze_26986, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:8:40-9:68
    for (int64_t i_26386 = 0; i_26386 < (int64_t) 5; i_26386++) {
        // src/dense_jacobian.fut:5:3-21
        for (int64_t nest_i_26796 = 0; nest_i_26796 < (int64_t) 5; nest_i_26796++) {
            ((double *) mem_26569)[nest_i_26796] = 0.0;
        }
        // src/dense_jacobian.fut:5:3-39
        ((double *) mem_26569)[i_26386] = 1.0;
        
        double zt_lhs_tan_25865 = ((double *) mem_26569)[(int64_t) 0];
        
        // test/test_sparse_jacobian_jvp.fut:119:17-23
        
        double binop_x_25867 = zt_lhs_24490 * zt_lhs_tan_25865;
        
        // test/test_sparse_jacobian_jvp.fut:119:17-23
        
        double zp_lhs_tan_25866 = binop_x_25867 + binop_x_25867;
        double zp_rhs_tan_25869 = ((double *) mem_26569)[(int64_t) 4];
        
        // test/test_sparse_jacobian_jvp.fut:119:24-30
        
        double y0_tan_25870 = zp_lhs_tan_25866 + zp_rhs_tan_25869;
        double zt_lhs_tan_25873 = ((double *) mem_26569)[(int64_t) 1];
        double zt_rhs_tan_25874 = ((double *) mem_26569)[(int64_t) 3];
        
        // test/test_sparse_jacobian_jvp.fut:120:17-23
        
        double binop_x_25876 = zt_rhs_24496 * zt_lhs_tan_25873;
        
        // test/test_sparse_jacobian_jvp.fut:120:17-23
        
        double binop_y_25877 = zt_lhs_24495 * zt_rhs_tan_25874;
        
        // test/test_sparse_jacobian_jvp.fut:120:17-23
        
        double y1_tan_25875 = binop_x_25876 + binop_y_25877;
        double zp_lhs_tan_25878 = ((double *) mem_26569)[(int64_t) 2];
        
        // test/test_sparse_jacobian_jvp.fut:121:26-32
        
        double binop_y_25881 = 2.0 * zp_rhs_tan_25869;
        
        // test/test_sparse_jacobian_jvp.fut:121:26-32
        
        double zp_rhs_tan_25879 = binop_x_25880 + binop_y_25881;
        
        // test/test_sparse_jacobian_jvp.fut:121:17-32
        
        double y2_tan_25882 = zp_lhs_tan_25878 + zp_rhs_tan_25879;
        
        // test/test_sparse_jacobian_jvp.fut:122:24-30
        
        double binop_x_25886 = zp_lhs_24498 * zp_lhs_tan_25878;
        
        // test/test_sparse_jacobian_jvp.fut:122:24-30
        
        double zm_rhs_tan_25885 = binop_x_25886 + binop_x_25886;
        
        // test/test_sparse_jacobian_jvp.fut:122:17-30
        
        double binop_y_25890 = -1.0 * zm_rhs_tan_25885;
        
        // test/test_sparse_jacobian_jvp.fut:122:17-30
        
        double y3_tan_25888 = zt_lhs_tan_25865 + binop_y_25890;
        
        // test/test_sparse_jacobian_jvp.fut:140:59-64
        ((double *) mem_26564)[i_26386 * (int64_t) 5] = y0_tan_25870;
        ((double *) mem_26564)[i_26386 * (int64_t) 5 + (int64_t) 1] = y1_tan_25875;
        ((double *) mem_26564)[i_26386 * (int64_t) 5 + (int64_t) 2] = y2_tan_25882;
        ((double *) mem_26564)[i_26386 * (int64_t) 5 + (int64_t) 3] = y3_tan_25888;
        ((double *) mem_26564)[i_26386 * (int64_t) 5 + (int64_t) 4] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    if (mem_26578_cached_sizze_26987 < bytes_26577) {
        err = lexical_realloc(ctx, &mem_26578, &mem_26578_cached_sizze_26987, bytes_26577);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    for (int64_t nest_i_26797 = 0; nest_i_26797 < csr_bipartite_from_pattern_res_24075; nest_i_26797++) {
        ((double *) mem_26578)[nest_i_26797] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    if (mem_26667_cached_sizze_26999 < bytes_26666) {
        err = lexical_realloc(ctx, &mem_26667, &mem_26667_cached_sizze_26999, bytes_26666);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:17:3-61
    if (mem_26672_cached_sizze_27000 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_26672, &mem_26672_cached_sizze_27000, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    for (int64_t i_26457 = 0; i_26457 < num_colors_of_res_f_res_24386; i_26457++) {
        // src/sparse_jacobian_jvp.fut:17:3-61
        for (int64_t i_26453 = 0; i_26453 < (int64_t) 5; i_26453++) {
            int64_t eta_p_24399 = ((int64_t *) mem_26588)[i_26453];
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            bool cond_24400 = eta_p_24399 == i_26457;
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            double lifted_lambda_res_24401;
            
            if (cond_24400) {
                lifted_lambda_res_24401 = 1.0;
            } else {
                lifted_lambda_res_24401 = 0.0;
            }
            ((double *) mem_26672)[i_26453] = lifted_lambda_res_24401;
        }
        
        double zt_lhs_tan_25893 = ((double *) mem_26672)[(int64_t) 0];
        
        // test/test_sparse_jacobian_jvp.fut:119:17-23
        
        double binop_x_25895 = zt_lhs_24490 * zt_lhs_tan_25893;
        
        // test/test_sparse_jacobian_jvp.fut:119:17-23
        
        double zp_lhs_tan_25894 = binop_x_25895 + binop_x_25895;
        double zp_rhs_tan_25897 = ((double *) mem_26672)[(int64_t) 4];
        
        // test/test_sparse_jacobian_jvp.fut:119:24-30
        
        double y0_tan_25898 = zp_lhs_tan_25894 + zp_rhs_tan_25897;
        double zt_lhs_tan_25901 = ((double *) mem_26672)[(int64_t) 1];
        double zt_rhs_tan_25902 = ((double *) mem_26672)[(int64_t) 3];
        
        // test/test_sparse_jacobian_jvp.fut:120:17-23
        
        double binop_x_25904 = zt_rhs_24496 * zt_lhs_tan_25901;
        
        // test/test_sparse_jacobian_jvp.fut:120:17-23
        
        double binop_y_25905 = zt_lhs_24495 * zt_rhs_tan_25902;
        
        // test/test_sparse_jacobian_jvp.fut:120:17-23
        
        double y1_tan_25903 = binop_x_25904 + binop_y_25905;
        double zp_lhs_tan_25906 = ((double *) mem_26672)[(int64_t) 2];
        
        // test/test_sparse_jacobian_jvp.fut:121:26-32
        
        double binop_y_25909 = 2.0 * zp_rhs_tan_25897;
        
        // test/test_sparse_jacobian_jvp.fut:121:26-32
        
        double zp_rhs_tan_25907 = binop_x_25880 + binop_y_25909;
        
        // test/test_sparse_jacobian_jvp.fut:121:17-32
        
        double y2_tan_25910 = zp_lhs_tan_25906 + zp_rhs_tan_25907;
        
        // test/test_sparse_jacobian_jvp.fut:122:24-30
        
        double binop_x_25914 = zp_lhs_24498 * zp_lhs_tan_25906;
        
        // test/test_sparse_jacobian_jvp.fut:122:24-30
        
        double zm_rhs_tan_25913 = binop_x_25914 + binop_x_25914;
        
        // test/test_sparse_jacobian_jvp.fut:122:17-30
        
        double binop_y_25918 = -1.0 * zm_rhs_tan_25913;
        
        // test/test_sparse_jacobian_jvp.fut:122:17-30
        
        double y3_tan_25916 = zt_lhs_tan_25893 + binop_y_25918;
        
        // test/test_sparse_jacobian_jvp.fut:141:33-38
        ((double *) mem_26667)[i_26457 * (int64_t) 5] = y0_tan_25898;
        ((double *) mem_26667)[i_26457 * (int64_t) 5 + (int64_t) 1] = y1_tan_25903;
        ((double *) mem_26667)[i_26457 * (int64_t) 5 + (int64_t) 2] = y2_tan_25910;
        ((double *) mem_26667)[i_26457 * (int64_t) 5 + (int64_t) 3] = y3_tan_25916;
        ((double *) mem_26667)[i_26457 * (int64_t) 5 + (int64_t) 4] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:45:5-52:27
    
    bool compressed_to_csr_vals_res_24410;
    int64_t compressed_to_csr_vals_res_24412;
    bool loop_while_24413;
    int64_t i_24415;
    
    loop_while_24413 = 1;
    i_24415 = (int64_t) 0;
    while (loop_while_24413) {
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool x_24416 = sle64((int64_t) 0, i_24415);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool y_24417 = slt64(i_24415, csr_bipartite_from_pattern_res_24074);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool bounds_check_24418 = x_24416 && y_24417;
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool index_certs_24419;
        
        if (!bounds_check_24418) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_24415, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24074, "].", "-> #0  src/sparse_jacobian_jvp.fut:47:15-26\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        int64_t s_24420 = ((int64_t *) ext_mem_26576.mem)[i_24415];
        
        // src/sparse_jacobian_jvp.fut:48:25-27
        
        int64_t e_24421 = add64((int64_t) 1, i_24415);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool x_24422 = sle64((int64_t) 0, e_24421);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool y_24423 = slt64(e_24421, csr_bipartite_from_pattern_res_24074);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool bounds_check_24424 = x_24422 && y_24423;
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool index_certs_24425;
        
        if (!bounds_check_24424) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_24421, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24074, "].", "-> #0  src/sparse_jacobian_jvp.fut:48:15-28\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        int64_t e_24426 = ((int64_t *) ext_mem_26576.mem)[e_24421];
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t j_m_i_24427 = sub64(e_24426, s_24420);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool empty_slice_24428 = j_m_i_24427 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t m_24429 = sub64(j_m_i_24427, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t i_p_m_t_s_24430 = add64(s_24420, m_24429);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_24431 = sle64((int64_t) 0, i_p_m_t_s_24430);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_24432 = slt64(i_p_m_t_s_24430, csr_bipartite_from_pattern_res_24075);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_lte_i_24433 = sle64((int64_t) 0, s_24420);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_lte_j_24434 = sle64(s_24420, e_24426);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_24435 = i_p_m_t_s_leq_w_24432 && zzero_lte_i_24433;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_24436 = zzero_leq_i_p_m_t_s_24431 && y_24435;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool forwards_ok_24437 = i_lte_j_24434 && y_24436;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool ok_or_empty_24438 = empty_slice_24428 || forwards_ok_24437;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool index_certs_24439;
        
        if (!ok_or_empty_24438) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_24420, ":", (long long) e_24426, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24075, "].", "-> #0  src/sparse_jacobian_jvp.fut:49:18-30\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool y_24441 = slt64(i_24415, (int64_t) 5);
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool bounds_check_24442 = x_24416 && y_24441;
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool index_certs_24443;
        
        if (!bounds_check_24442) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_24415, "] out of bounds for array of shape [", (long long) (int64_t) 5, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-45\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:18-51
        for (int64_t i_26461 = 0; i_26461 < j_m_i_24427; i_26461++) {
            int64_t index_primexp_26496 = s_24420 + i_26461;
            int64_t eta_p_24445 = ((int64_t *) ext_mem_26575.mem)[index_primexp_26496];
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool x_24446 = sle64((int64_t) 0, eta_p_24445);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool y_24447 = slt64(eta_p_24445, (int64_t) 5);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool bounds_check_24448 = x_24446 && y_24447;
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool index_certs_24449;
            
            if (!bounds_check_24448) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_24445, "] out of bounds for array of shape [", (long long) (int64_t) 5, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:32-41\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            int64_t tmp_24450 = ((int64_t *) mem_26588)[eta_p_24445];
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool x_24451 = sle64((int64_t) 0, tmp_24450);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool y_24452 = slt64(tmp_24450, num_colors_of_res_f_res_24386);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool bounds_check_24453 = x_24451 && y_24452;
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool index_certs_24454;
            
            if (!bounds_check_24453) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_24450, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_24386, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-42\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:18-51
            
            double lifted_lambda_res_24455 = ((double *) mem_26667)[tmp_24450 * (int64_t) 5 + i_24415];
            
            ((double *) mem_26578)[s_24420 + i_26461] = lifted_lambda_res_24455;
        }
        // src/sparse_jacobian_jvp.fut:46:13-16
        
        bool loop_cond_24457 = slt64(e_24421, (int64_t) 5);
        bool loop_while_tmp_26800 = loop_cond_24457;
        int64_t i_tmp_26802 = e_24421;
        
        loop_while_24413 = loop_while_tmp_26800;
        i_24415 = i_tmp_26802;
    }
    compressed_to_csr_vals_res_24410 = loop_while_24413;
    compressed_to_csr_vals_res_24412 = i_24415;
    // src/sparse_jacobian_jvp.fut:67:29-47
    if (mem_26698_cached_sizze_27001 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_26698, &mem_26698_cached_sizze_27001, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:8:19-10:39
    
    bool defunc_0_reduce_res_26033;
    bool redout_26467 = 1;
    
    for (int64_t i_26468 = 0; i_26468 < (int64_t) 5; i_26468++) {
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool y_24851 = slt64(i_26468, csr_bipartite_from_pattern_res_24074);
        
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool index_certs_24853;
        
        if (!y_24851) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_26468, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24074, "].", "-> #0  src/sparse_jacobian_jvp.fut:63:17-28\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #3  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        int64_t s_24854 = ((int64_t *) ext_mem_26576.mem)[i_26468];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_lte_i_24867 = sle64((int64_t) 0, s_24854);
        
        // src/sparse_jacobian_jvp.fut:64:27-29
        
        int64_t e_24855 = add64((int64_t) 1, i_26468);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool y_24857 = slt64(e_24855, csr_bipartite_from_pattern_res_24074);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool x_24856 = sle64((int64_t) 0, e_24855);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool bounds_check_24858 = x_24856 && y_24857;
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool index_certs_24859;
        
        if (!bounds_check_24858) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_24855, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24074, "].", "-> #0  src/sparse_jacobian_jvp.fut:64:17-30\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #3  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        int64_t e_24860 = ((int64_t *) ext_mem_26576.mem)[e_24855];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t j_m_i_24861 = sub64(e_24860, s_24854);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t m_24863 = sub64(j_m_i_24861, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t i_p_m_t_s_24864 = add64(s_24854, m_24863);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_p_m_t_s_leq_w_24866 = slt64(i_p_m_t_s_24864, csr_bipartite_from_pattern_res_24075);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24875 = i_p_m_t_s_leq_w_24866 && zzero_lte_i_24867;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_leq_i_p_m_t_s_24865 = sle64((int64_t) 0, i_p_m_t_s_24864);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24876 = zzero_leq_i_p_m_t_s_24865 && y_24875;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_lte_j_24868 = sle64(s_24854, e_24860);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool forwards_ok_24877 = i_lte_j_24868 && y_24876;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool empty_slice_24862 = j_m_i_24861 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool ok_or_empty_24878 = empty_slice_24862 || forwards_ok_24877;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool index_certs_24879;
        
        if (!ok_or_empty_24878) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_24854, ":", (long long) e_24860, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24075, "].", "-> #0  src/sparse_jacobian_jvp.fut:66:20-29\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #3  test/test_sparse_jacobian_jvp.fut:118:15-141:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:67:29-47
        for (int64_t nest_i_26805 = 0; nest_i_26805 < (int64_t) 5; nest_i_26805++) {
            ((double *) mem_26698)[nest_i_26805] = 0.0;
        }
        // src/sparse_jacobian_jvp.fut:68:12-32
        
        bool acc_cert_24883;
        
        // src/sparse_jacobian_jvp.fut:68:12-32
        for (int64_t i_26464 = 0; i_26464 < j_m_i_24861; i_26464++) {
            int64_t index_primexp_26493 = s_24854 + i_26464;
            int64_t v_24887 = ((int64_t *) ext_mem_26575.mem)[index_primexp_26493];
            double v_24888 = ((double *) mem_26578)[index_primexp_26493];
            
            // src/sparse_jacobian_jvp.fut:68:12-32
            // UpdateAcc
            if (sle64((int64_t) 0, v_24887) && slt64(v_24887, (int64_t) 5)) {
                ((double *) mem_26698)[v_24887] = v_24888;
            }
        }
        // test/test_sparse_jacobian_jvp.fut:9:5-66
        
        bool defunc_0_reduce_res_26028;
        bool redout_26465 = 1;
        
        for (int64_t i_26466 = 0; i_26466 < (int64_t) 5; i_26466++) {
            bool eta_p_24911 = ((bool *) mem_26562.mem)[i_26468 * (int64_t) 5 + i_26466];
            double eta_p_24912 = ((double *) mem_26564)[i_26466 * (int64_t) 5 + i_26468];
            double eta_p_24913 = ((double *) mem_26698)[i_26466];
            
            // test/test_sparse_jacobian_jvp.fut:14:25-48
            
            double lifted_lambda_res_24914;
            
            if (eta_p_24911) {
                lifted_lambda_res_24914 = eta_p_24912;
            } else {
                lifted_lambda_res_24914 = 0.0;
            }
            // test/test_sparse_jacobian_jvp.fut:9:48-51
            
            double abs_arg0_24916 = eta_p_24913 - lifted_lambda_res_24914;
            
            // test/test_sparse_jacobian_jvp.fut:9:37-51
            
            double abs_res_24917 = fabs64(abs_arg0_24916);
            
            // test/test_sparse_jacobian_jvp.fut:9:53-59
            
            bool lifted_lambda_res_24918 = abs_res_24917 <= 1.0e-9;
            
            // test/test_sparse_jacobian_jvp.fut:9:5-66
            
            bool x_24901 = lifted_lambda_res_24918 && redout_26465;
            bool redout_tmp_26807 = x_24901;
            
            redout_26465 = redout_tmp_26807;
        }
        defunc_0_reduce_res_26028 = redout_26465;
        // test/test_sparse_jacobian_jvp.fut:10:6-39
        
        bool x_24480 = defunc_0_reduce_res_26028 && redout_26467;
        bool redout_tmp_26804 = x_24480;
        
        redout_26467 = redout_tmp_26804;
    }
    defunc_0_reduce_res_26033 = redout_26467;
    if (memblock_unref(ctx, &ext_mem_26575, "ext_mem_26575") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26576, "ext_mem_26576") != 0)
        return 1;
    prim_out_26758 = defunc_0_reduce_res_26033;
    *out_prim_out_26984 = prim_out_26758;
    
  cleanup:
    {
        free(mem_26564);
        free(mem_26569);
        free(mem_26578);
        free(mem_26579);
        free(mem_26588);
        free(mem_26600);
        free(mem_26604);
        free(mem_26621);
        free(mem_26623);
        free(mem_26624);
        free(mem_26644);
        free(mem_26646);
        free(mem_26648);
        free(mem_26656);
        free(mem_26667);
        free(mem_26672);
        free(mem_26698);
        if (memblock_unref(ctx, &mem_param_tmp_26763, "mem_param_tmp_26763") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26658, "mem_26658") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_26598, "mem_param_26598") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26663, "ext_mem_26663") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26591, "ext_mem_26591") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26592, "ext_mem_26592") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26589, "mem_26589") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_26587, "mem_26587") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26575, "ext_mem_26575") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26576, "ext_mem_26576") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_jvp_zzero_pattern_matches_dense(struct futhark_context *ctx, bool *out_prim_out_27002, struct memblock x_mem_26563)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_26567_cached_sizze_27003 = 0;
    unsigned char *mem_26567 = NULL;
    int64_t mem_26576_cached_sizze_27004 = 0;
    unsigned char *mem_26576 = NULL;
    struct memblock ext_mem_26564;
    
    ext_mem_26564.references = NULL;
    
    struct memblock ext_mem_26565;
    
    ext_mem_26565.references = NULL;
    
    struct memblock mem_26545 = ctx->constants->mem_26545;
    struct memblock mem_26548 = ctx->constants->mem_26548;
    struct memblock mem_26550 = ctx->constants->mem_26550;
    struct memblock mem_26555 = ctx->constants->mem_26555;
    struct memblock mem_26556 = ctx->constants->mem_26556;
    struct memblock mem_26562 = ctx->constants->mem_26562;
    bool prim_out_26758;
    
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_24083;
    int64_t csr_bipartite_from_pattern_res_24084;
    
    if (futrts_csr_rows_from_pattern_7271(ctx, &ext_mem_26565, &ext_mem_26564, &csr_bipartite_from_pattern_res_24083, &csr_bipartite_from_pattern_res_24084, mem_26550, (int64_t) 2, (int64_t) 3) != 0) {
        err = 1;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    
    int64_t bytes_26566 = (int64_t) 8 * csr_bipartite_from_pattern_res_24084;
    
    // src/sparse_jacobian_jvp.fut:42:15-35
    if (mem_26567_cached_sizze_27003 < bytes_26566) {
        err = lexical_realloc(ctx, &mem_26567, &mem_26567_cached_sizze_27003, bytes_26566);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    for (int64_t nest_i_26759 = 0; nest_i_26759 < csr_bipartite_from_pattern_res_24084; nest_i_26759++) {
        ((double *) mem_26567)[nest_i_26759] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:45:5-52:27
    
    bool compressed_to_csr_vals_res_24416;
    int64_t compressed_to_csr_vals_res_24418;
    bool loop_while_24419;
    int64_t i_24421;
    
    loop_while_24419 = 1;
    i_24421 = (int64_t) 0;
    while (loop_while_24419) {
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool x_24422 = sle64((int64_t) 0, i_24421);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool y_24423 = slt64(i_24421, csr_bipartite_from_pattern_res_24083);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool bounds_check_24424 = x_24422 && y_24423;
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool index_certs_24425;
        
        if (!bounds_check_24424) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_24421, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24083, "].", "-> #0  src/sparse_jacobian_jvp.fut:47:15-26\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:62:21-77:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        int64_t s_24426 = ((int64_t *) ext_mem_26565.mem)[i_24421];
        
        // src/sparse_jacobian_jvp.fut:48:25-27
        
        int64_t e_24427 = add64((int64_t) 1, i_24421);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool x_24428 = sle64((int64_t) 0, e_24427);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool y_24429 = slt64(e_24427, csr_bipartite_from_pattern_res_24083);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool bounds_check_24430 = x_24428 && y_24429;
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool index_certs_24431;
        
        if (!bounds_check_24430) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_24427, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24083, "].", "-> #0  src/sparse_jacobian_jvp.fut:48:15-28\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:62:21-77:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        int64_t e_24432 = ((int64_t *) ext_mem_26565.mem)[e_24427];
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t j_m_i_24433 = sub64(e_24432, s_24426);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool empty_slice_24434 = j_m_i_24433 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t m_24435 = sub64(j_m_i_24433, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t i_p_m_t_s_24436 = add64(s_24426, m_24435);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_24437 = sle64((int64_t) 0, i_p_m_t_s_24436);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_24438 = slt64(i_p_m_t_s_24436, csr_bipartite_from_pattern_res_24084);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_lte_i_24439 = sle64((int64_t) 0, s_24426);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_lte_j_24440 = sle64(s_24426, e_24432);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_24441 = i_p_m_t_s_leq_w_24438 && zzero_lte_i_24439;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_24442 = zzero_leq_i_p_m_t_s_24437 && y_24441;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool forwards_ok_24443 = i_lte_j_24440 && y_24442;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool ok_or_empty_24444 = empty_slice_24434 || forwards_ok_24443;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool index_certs_24445;
        
        if (!ok_or_empty_24444) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_24426, ":", (long long) e_24432, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24084, "].", "-> #0  src/sparse_jacobian_jvp.fut:49:18-30\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:62:21-77:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool y_24447 = slt64(i_24421, (int64_t) 2);
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool bounds_check_24448 = x_24422 && y_24447;
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool index_certs_24449;
        
        if (!bounds_check_24448) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_24421, "] out of bounds for array of shape [", (long long) (int64_t) 2, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-45\n   #1  src/sparse_jacobian_jvp.fut:108:16-113:63\n   #2  src/sparse_jacobian_jvp.fut:118:8-123:39\n   #3  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #4  test/test_sparse_jacobian_jvp.fut:62:21-77:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:18-51
        for (int64_t nest_i_26763 = 0; nest_i_26763 < j_m_i_24433; nest_i_26763++) {
            ((double *) mem_26567)[s_24426 + nest_i_26763] = 0.0;
        }
        // src/sparse_jacobian_jvp.fut:46:13-16
        
        bool loop_cond_24463 = slt64(e_24427, (int64_t) 2);
        bool loop_while_tmp_26760 = loop_cond_24463;
        int64_t i_tmp_26762 = e_24427;
        
        loop_while_24419 = loop_while_tmp_26760;
        i_24421 = i_tmp_26762;
    }
    compressed_to_csr_vals_res_24416 = loop_while_24419;
    compressed_to_csr_vals_res_24418 = i_24421;
    // src/sparse_jacobian_jvp.fut:67:29-47
    if (mem_26576_cached_sizze_27004 < (int64_t) 24) {
        err = lexical_realloc(ctx, &mem_26576, &mem_26576_cached_sizze_27004, (int64_t) 24);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // test/test_sparse_jacobian_jvp.fut:8:19-10:39
    
    bool defunc_0_reduce_res_25982;
    bool redout_26388 = 1;
    
    for (int64_t i_26389 = 0; i_26389 < (int64_t) 2; i_26389++) {
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool y_24850 = slt64(i_26389, csr_bipartite_from_pattern_res_24083);
        
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        bool index_certs_24852;
        
        if (!y_24850) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_26389, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24083, "].", "-> #0  src/sparse_jacobian_jvp.fut:63:17-28\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #3  test/test_sparse_jacobian_jvp.fut:62:21-77:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:63:17-28
        
        int64_t s_24853 = ((int64_t *) ext_mem_26565.mem)[i_26389];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_lte_i_24866 = sle64((int64_t) 0, s_24853);
        
        // src/sparse_jacobian_jvp.fut:64:27-29
        
        int64_t e_24854 = add64((int64_t) 1, i_26389);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool y_24856 = slt64(e_24854, csr_bipartite_from_pattern_res_24083);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool x_24855 = sle64((int64_t) 0, e_24854);
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool bounds_check_24857 = x_24855 && y_24856;
        
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        bool index_certs_24858;
        
        if (!bounds_check_24857) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_24854, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24083, "].", "-> #0  src/sparse_jacobian_jvp.fut:64:17-30\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #3  test/test_sparse_jacobian_jvp.fut:62:21-77:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:64:17-30
        
        int64_t e_24859 = ((int64_t *) ext_mem_26565.mem)[e_24854];
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t j_m_i_24860 = sub64(e_24859, s_24853);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t m_24862 = sub64(j_m_i_24860, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        int64_t i_p_m_t_s_24863 = add64(s_24853, m_24862);
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_p_m_t_s_leq_w_24865 = slt64(i_p_m_t_s_24863, csr_bipartite_from_pattern_res_24084);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24874 = i_p_m_t_s_leq_w_24865 && zzero_lte_i_24866;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool zzero_leq_i_p_m_t_s_24864 = sle64((int64_t) 0, i_p_m_t_s_24863);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool y_24875 = zzero_leq_i_p_m_t_s_24864 && y_24874;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool i_lte_j_24867 = sle64(s_24853, e_24859);
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool forwards_ok_24876 = i_lte_j_24867 && y_24875;
        
        // src/sparse_jacobian_jvp.fut:65:20-32
        
        bool empty_slice_24861 = j_m_i_24860 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool ok_or_empty_24877 = empty_slice_24861 || forwards_ok_24876;
        
        // src/sparse_jacobian_jvp.fut:66:20-29
        
        bool index_certs_24878;
        
        if (!ok_or_empty_24877) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_24853, ":", (long long) e_24859, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_24084, "].", "-> #0  src/sparse_jacobian_jvp.fut:66:20-29\n   #1  src/sparse_jacobian_jvp.fut:118:18-124:40\n   #2  src/sparse_jacobian_jvp.fut:154:8-159:42\n   #3  test/test_sparse_jacobian_jvp.fut:62:21-77:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:67:29-47
        for (int64_t nest_i_26765 = 0; nest_i_26765 < (int64_t) 3; nest_i_26765++) {
            ((double *) mem_26576)[nest_i_26765] = 0.0;
        }
        // src/sparse_jacobian_jvp.fut:68:12-32
        
        bool acc_cert_24882;
        
        // src/sparse_jacobian_jvp.fut:68:12-32
        for (int64_t i_26385 = 0; i_26385 < j_m_i_24860; i_26385++) {
            int64_t index_primexp_26493 = s_24853 + i_26385;
            int64_t v_24886 = ((int64_t *) ext_mem_26564.mem)[index_primexp_26493];
            double v_24887 = ((double *) mem_26567)[index_primexp_26493];
            
            // src/sparse_jacobian_jvp.fut:68:12-32
            // UpdateAcc
            if (sle64((int64_t) 0, v_24886) && slt64(v_24886, (int64_t) 3)) {
                ((double *) mem_26576)[v_24886] = v_24887;
            }
        }
        // test/test_sparse_jacobian_jvp.fut:9:5-66
        
        bool defunc_0_reduce_res_25981;
        bool redout_26386 = 1;
        
        for (int64_t i_26387 = 0; i_26387 < (int64_t) 3; i_26387++) {
            double eta_p_24912 = ((double *) mem_26576)[i_26387];
            
            // test/test_sparse_jacobian_jvp.fut:9:37-51
            
            double abs_res_24916 = fabs64(eta_p_24912);
            
            // test/test_sparse_jacobian_jvp.fut:9:53-59
            
            bool lifted_lambda_res_24917 = abs_res_24916 <= 1.0e-9;
            
            // test/test_sparse_jacobian_jvp.fut:9:5-66
            
            bool x_24900 = lifted_lambda_res_24917 && redout_26386;
            bool redout_tmp_26767 = x_24900;
            
            redout_26386 = redout_tmp_26767;
        }
        defunc_0_reduce_res_25981 = redout_26386;
        // test/test_sparse_jacobian_jvp.fut:10:6-39
        
        bool x_24578 = defunc_0_reduce_res_25981 && redout_26388;
        bool redout_tmp_26764 = x_24578;
        
        redout_26388 = redout_tmp_26764;
    }
    defunc_0_reduce_res_25982 = redout_26388;
    if (memblock_unref(ctx, &ext_mem_26564, "ext_mem_26564") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_26565, "ext_mem_26565") != 0)
        return 1;
    prim_out_26758 = defunc_0_reduce_res_25982;
    *out_prim_out_27002 = prim_out_26758;
    
  cleanup:
    {
        free(mem_26567);
        free(mem_26576);
        if (memblock_unref(ctx, &ext_mem_26564, "ext_mem_26564") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_26565, "ext_mem_26565") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_test_jvp_csr_from_csr_ex4_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0)
{
    bool prim_out_26758 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock x_mem_26563;
    
    x_mem_26563.references = NULL;
    x_mem_26563 = in0->mem;
    if (!((int64_t) 6 == in0->shape[0])) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_test_jvp_csr_from_csr_ex4_matches_dense(ctx, &prim_out_26758, x_mem_26563);
        if (ret == 0) {
            struct memblock mem_26545 = ctx->constants->mem_26545;
            struct memblock mem_26548 = ctx->constants->mem_26548;
            struct memblock mem_26550 = ctx->constants->mem_26550;
            struct memblock mem_26555 = ctx->constants->mem_26555;
            struct memblock mem_26556 = ctx->constants->mem_26556;
            struct memblock mem_26562 = ctx->constants->mem_26562;
            
            *out0 = prim_out_26758;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test_prepared_jvp_ex4_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0)
{
    bool prim_out_26758 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock x_mem_26563;
    
    x_mem_26563.references = NULL;
    x_mem_26563 = in0->mem;
    if (!((int64_t) 6 == in0->shape[0])) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_test_prepared_jvp_ex4_matches_dense(ctx, &prim_out_26758, x_mem_26563);
        if (ret == 0) {
            struct memblock mem_26545 = ctx->constants->mem_26545;
            struct memblock mem_26548 = ctx->constants->mem_26548;
            struct memblock mem_26550 = ctx->constants->mem_26550;
            struct memblock mem_26555 = ctx->constants->mem_26555;
            struct memblock mem_26556 = ctx->constants->mem_26556;
            struct memblock mem_26562 = ctx->constants->mem_26562;
            
            *out0 = prim_out_26758;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test_prepared_jvp_reuse_two_points(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0)
{
    bool prim_out_26758 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock x1_mem_26563;
    
    x1_mem_26563.references = NULL;
    x1_mem_26563 = in0->mem;
    if (!((int64_t) 6 == in0->shape[0])) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_test_prepared_jvp_reuse_two_points(ctx, &prim_out_26758, x1_mem_26563);
        if (ret == 0) {
            struct memblock mem_26545 = ctx->constants->mem_26545;
            struct memblock mem_26548 = ctx->constants->mem_26548;
            struct memblock mem_26550 = ctx->constants->mem_26550;
            struct memblock mem_26555 = ctx->constants->mem_26555;
            struct memblock mem_26556 = ctx->constants->mem_26556;
            struct memblock mem_26562 = ctx->constants->mem_26562;
            
            *out0 = prim_out_26758;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test_sparse_jvp_ex1_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0)
{
    bool prim_out_26758 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock x_mem_26563;
    
    x_mem_26563.references = NULL;
    x_mem_26563 = in0->mem;
    if (!((int64_t) 5 == in0->shape[0])) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_test_sparse_jvp_ex1_matches_dense(ctx, &prim_out_26758, x_mem_26563);
        if (ret == 0) {
            struct memblock mem_26545 = ctx->constants->mem_26545;
            struct memblock mem_26548 = ctx->constants->mem_26548;
            struct memblock mem_26550 = ctx->constants->mem_26550;
            struct memblock mem_26555 = ctx->constants->mem_26555;
            struct memblock mem_26556 = ctx->constants->mem_26556;
            struct memblock mem_26562 = ctx->constants->mem_26562;
            
            *out0 = prim_out_26758;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test_sparse_jvp_ex2_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0)
{
    bool prim_out_26758 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock x_mem_26563;
    
    x_mem_26563.references = NULL;
    x_mem_26563 = in0->mem;
    if (!((int64_t) 4 == in0->shape[0])) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_test_sparse_jvp_ex2_matches_dense(ctx, &prim_out_26758, x_mem_26563);
        if (ret == 0) {
            struct memblock mem_26545 = ctx->constants->mem_26545;
            struct memblock mem_26548 = ctx->constants->mem_26548;
            struct memblock mem_26550 = ctx->constants->mem_26550;
            struct memblock mem_26555 = ctx->constants->mem_26555;
            struct memblock mem_26556 = ctx->constants->mem_26556;
            struct memblock mem_26562 = ctx->constants->mem_26562;
            
            *out0 = prim_out_26758;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test_sparse_jvp_ex4_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0)
{
    bool prim_out_26758 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock x_mem_26563;
    
    x_mem_26563.references = NULL;
    x_mem_26563 = in0->mem;
    if (!((int64_t) 6 == in0->shape[0])) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_test_sparse_jvp_ex4_matches_dense(ctx, &prim_out_26758, x_mem_26563);
        if (ret == 0) {
            struct memblock mem_26545 = ctx->constants->mem_26545;
            struct memblock mem_26548 = ctx->constants->mem_26548;
            struct memblock mem_26550 = ctx->constants->mem_26550;
            struct memblock mem_26555 = ctx->constants->mem_26555;
            struct memblock mem_26556 = ctx->constants->mem_26556;
            struct memblock mem_26562 = ctx->constants->mem_26562;
            
            *out0 = prim_out_26758;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test_sparse_jvp_ex4_with_colors_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0)
{
    bool prim_out_26758 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock x_mem_26563;
    
    x_mem_26563.references = NULL;
    x_mem_26563 = in0->mem;
    if (!((int64_t) 6 == in0->shape[0])) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_test_sparse_jvp_ex4_with_colors_matches_dense(ctx, &prim_out_26758, x_mem_26563);
        if (ret == 0) {
            struct memblock mem_26545 = ctx->constants->mem_26545;
            struct memblock mem_26548 = ctx->constants->mem_26548;
            struct memblock mem_26550 = ctx->constants->mem_26550;
            struct memblock mem_26555 = ctx->constants->mem_26555;
            struct memblock mem_26556 = ctx->constants->mem_26556;
            struct memblock mem_26562 = ctx->constants->mem_26562;
            
            *out0 = prim_out_26758;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test_sparse_jvp_ex5_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0)
{
    bool prim_out_26758 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock x_mem_26563;
    
    x_mem_26563.references = NULL;
    x_mem_26563 = in0->mem;
    if (!((int64_t) 5 == in0->shape[0])) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_test_sparse_jvp_ex5_matches_dense(ctx, &prim_out_26758, x_mem_26563);
        if (ret == 0) {
            struct memblock mem_26545 = ctx->constants->mem_26545;
            struct memblock mem_26548 = ctx->constants->mem_26548;
            struct memblock mem_26550 = ctx->constants->mem_26550;
            struct memblock mem_26555 = ctx->constants->mem_26555;
            struct memblock mem_26556 = ctx->constants->mem_26556;
            struct memblock mem_26562 = ctx->constants->mem_26562;
            
            *out0 = prim_out_26758;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test_sparse_jvp_zero_pattern_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0)
{
    bool prim_out_26758 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock x_mem_26563;
    
    x_mem_26563.references = NULL;
    x_mem_26563 = in0->mem;
    if (!((int64_t) 3 == in0->shape[0])) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_test_sparse_jvp_zzero_pattern_matches_dense(ctx, &prim_out_26758, x_mem_26563);
        if (ret == 0) {
            struct memblock mem_26545 = ctx->constants->mem_26545;
            struct memblock mem_26548 = ctx->constants->mem_26548;
            struct memblock mem_26550 = ctx->constants->mem_26550;
            struct memblock mem_26555 = ctx->constants->mem_26555;
            struct memblock mem_26556 = ctx->constants->mem_26556;
            struct memblock mem_26562 = ctx->constants->mem_26562;
            
            *out0 = prim_out_26758;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
