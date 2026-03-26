
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
int futhark_entry_test_sparse_vjp_ex1_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0);
int futhark_entry_test_sparse_vjp_ex2_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0);
int futhark_entry_test_sparse_vjp_ex4_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0);
int futhark_entry_test_sparse_vjp_ex4_with_row_colors_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0);
int futhark_entry_test_sparse_vjp_ex5_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0);
int futhark_entry_test_sparse_vjp_zero_pattern_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0);

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
const struct type *test_sparse_vjp_ex1_matches_dense_out_types[] = {&type_bool, NULL};
bool test_sparse_vjp_ex1_matches_dense_out_unique[] = {false};
const struct type *test_sparse_vjp_ex1_matches_dense_in_types[] = {&type_ZMZNf64, NULL};
bool test_sparse_vjp_ex1_matches_dense_in_unique[] = {false};
const char *test_sparse_vjp_ex1_matches_dense_tuning_params[] = {NULL};
int call_test_sparse_vjp_ex1_matches_dense(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    struct futhark_f64_1d * in0 = *(struct futhark_f64_1d * *) ins[0];
    
    return futhark_entry_test_sparse_vjp_ex1_matches_dense(ctx, out0, in0);
}
const struct type *test_sparse_vjp_ex2_matches_dense_out_types[] = {&type_bool, NULL};
bool test_sparse_vjp_ex2_matches_dense_out_unique[] = {false};
const struct type *test_sparse_vjp_ex2_matches_dense_in_types[] = {&type_ZMZNf64, NULL};
bool test_sparse_vjp_ex2_matches_dense_in_unique[] = {false};
const char *test_sparse_vjp_ex2_matches_dense_tuning_params[] = {NULL};
int call_test_sparse_vjp_ex2_matches_dense(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    struct futhark_f64_1d * in0 = *(struct futhark_f64_1d * *) ins[0];
    
    return futhark_entry_test_sparse_vjp_ex2_matches_dense(ctx, out0, in0);
}
const struct type *test_sparse_vjp_ex4_matches_dense_out_types[] = {&type_bool, NULL};
bool test_sparse_vjp_ex4_matches_dense_out_unique[] = {false};
const struct type *test_sparse_vjp_ex4_matches_dense_in_types[] = {&type_ZMZNf64, NULL};
bool test_sparse_vjp_ex4_matches_dense_in_unique[] = {false};
const char *test_sparse_vjp_ex4_matches_dense_tuning_params[] = {NULL};
int call_test_sparse_vjp_ex4_matches_dense(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    struct futhark_f64_1d * in0 = *(struct futhark_f64_1d * *) ins[0];
    
    return futhark_entry_test_sparse_vjp_ex4_matches_dense(ctx, out0, in0);
}
const struct type *test_sparse_vjp_ex4_with_row_colors_matches_dense_out_types[] = {&type_bool, NULL};
bool test_sparse_vjp_ex4_with_row_colors_matches_dense_out_unique[] = {false};
const struct type *test_sparse_vjp_ex4_with_row_colors_matches_dense_in_types[] = {&type_ZMZNf64, NULL};
bool test_sparse_vjp_ex4_with_row_colors_matches_dense_in_unique[] = {false};
const char *test_sparse_vjp_ex4_with_row_colors_matches_dense_tuning_params[] = {NULL};
int call_test_sparse_vjp_ex4_with_row_colors_matches_dense(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    struct futhark_f64_1d * in0 = *(struct futhark_f64_1d * *) ins[0];
    
    return futhark_entry_test_sparse_vjp_ex4_with_row_colors_matches_dense(ctx, out0, in0);
}
const struct type *test_sparse_vjp_ex5_matches_dense_out_types[] = {&type_bool, NULL};
bool test_sparse_vjp_ex5_matches_dense_out_unique[] = {false};
const struct type *test_sparse_vjp_ex5_matches_dense_in_types[] = {&type_ZMZNf64, NULL};
bool test_sparse_vjp_ex5_matches_dense_in_unique[] = {false};
const char *test_sparse_vjp_ex5_matches_dense_tuning_params[] = {NULL};
int call_test_sparse_vjp_ex5_matches_dense(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    struct futhark_f64_1d * in0 = *(struct futhark_f64_1d * *) ins[0];
    
    return futhark_entry_test_sparse_vjp_ex5_matches_dense(ctx, out0, in0);
}
const struct type *test_sparse_vjp_zzero_pattern_matches_dense_out_types[] = {&type_bool, NULL};
bool test_sparse_vjp_zzero_pattern_matches_dense_out_unique[] = {false};
const struct type *test_sparse_vjp_zzero_pattern_matches_dense_in_types[] = {&type_ZMZNf64, NULL};
bool test_sparse_vjp_zzero_pattern_matches_dense_in_unique[] = {false};
const char *test_sparse_vjp_zzero_pattern_matches_dense_tuning_params[] = {NULL};
int call_test_sparse_vjp_zzero_pattern_matches_dense(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    struct futhark_f64_1d * in0 = *(struct futhark_f64_1d * *) ins[0];
    
    return futhark_entry_test_sparse_vjp_zero_pattern_matches_dense(ctx, out0, in0);
}
const struct type *types[] = {&type_i8, &type_i16, &type_i32, &type_i64, &type_u8, &type_u16, &type_u32, &type_u64, &type_f16, &type_f32, &type_f64, &type_bool, &type_ZMZNf64, NULL};
struct entry_point entry_points[] = {{.name ="test_sparse_vjp_ex1_matches_dense", .f =call_test_sparse_vjp_ex1_matches_dense, .tuning_params =test_sparse_vjp_ex1_matches_dense_tuning_params, .in_types =test_sparse_vjp_ex1_matches_dense_in_types, .out_types =test_sparse_vjp_ex1_matches_dense_out_types, .in_unique =test_sparse_vjp_ex1_matches_dense_in_unique, .out_unique =test_sparse_vjp_ex1_matches_dense_out_unique}, {.name ="test_sparse_vjp_ex2_matches_dense", .f =call_test_sparse_vjp_ex2_matches_dense, .tuning_params =test_sparse_vjp_ex2_matches_dense_tuning_params, .in_types =test_sparse_vjp_ex2_matches_dense_in_types, .out_types =test_sparse_vjp_ex2_matches_dense_out_types, .in_unique =test_sparse_vjp_ex2_matches_dense_in_unique, .out_unique =test_sparse_vjp_ex2_matches_dense_out_unique}, {.name ="test_sparse_vjp_ex4_matches_dense", .f =call_test_sparse_vjp_ex4_matches_dense, .tuning_params =test_sparse_vjp_ex4_matches_dense_tuning_params, .in_types =test_sparse_vjp_ex4_matches_dense_in_types, .out_types =test_sparse_vjp_ex4_matches_dense_out_types, .in_unique =test_sparse_vjp_ex4_matches_dense_in_unique, .out_unique =test_sparse_vjp_ex4_matches_dense_out_unique}, {.name ="test_sparse_vjp_ex4_with_row_colors_matches_dense", .f =call_test_sparse_vjp_ex4_with_row_colors_matches_dense, .tuning_params =test_sparse_vjp_ex4_with_row_colors_matches_dense_tuning_params, .in_types =test_sparse_vjp_ex4_with_row_colors_matches_dense_in_types, .out_types =test_sparse_vjp_ex4_with_row_colors_matches_dense_out_types, .in_unique =test_sparse_vjp_ex4_with_row_colors_matches_dense_in_unique, .out_unique =test_sparse_vjp_ex4_with_row_colors_matches_dense_out_unique}, {.name ="test_sparse_vjp_ex5_matches_dense", .f =call_test_sparse_vjp_ex5_matches_dense, .tuning_params =test_sparse_vjp_ex5_matches_dense_tuning_params, .in_types =test_sparse_vjp_ex5_matches_dense_in_types, .out_types =test_sparse_vjp_ex5_matches_dense_out_types, .in_unique =test_sparse_vjp_ex5_matches_dense_in_unique, .out_unique =test_sparse_vjp_ex5_matches_dense_out_unique}, {.name ="test_sparse_vjp_zero_pattern_matches_dense", .f =call_test_sparse_vjp_zzero_pattern_matches_dense, .tuning_params =test_sparse_vjp_zzero_pattern_matches_dense_tuning_params, .in_types =test_sparse_vjp_zzero_pattern_matches_dense_in_types, .out_types =test_sparse_vjp_zzero_pattern_matches_dense_out_types, .in_unique =test_sparse_vjp_zzero_pattern_matches_dense_in_unique, .out_unique =test_sparse_vjp_zzero_pattern_matches_dense_out_unique}, {.name =NULL}};
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
    struct memblock mem_21809;
    struct memblock mem_21812;
    struct memblock mem_21814;
    struct memblock mem_21819;
    struct memblock mem_21820;
    struct memblock mem_21826;
};
static bool static_array_realtype_22030[5] = { 1,0,0,1,0};
static bool static_array_realtype_22031[5] = { 0,1,0,0,0};
static bool static_array_realtype_22032[5] = { 0,0,1,0,0};
static bool static_array_realtype_22033[4] = { 1,1,0,0};
static bool static_array_realtype_22034[4] = { 0,0,1,0};
static bool static_array_realtype_22035[3] = { 0,0,0};
static bool static_array_realtype_22036[6] = { 1,1,0,0,1,0};
static bool static_array_realtype_22037[6] = { 0,0,1,0,0,1};
static bool static_array_realtype_22038[6] = { 0,1,1,1,0,0};
static bool static_array_realtype_22039[6] = { 0,0,0,0,0,0};
static int64_t static_array_realtype_22040[4] = { (int64_t) 0,(int64_t) 0,(int64_t) 1,(int64_t) 0};
static bool static_array_realtype_22041[5] = { 1,0,0,0,1};
static bool static_array_realtype_22042[5] = { 0,1,0,1,0};
static bool static_array_realtype_22043[5] = { 0,0,1,0,1};
static bool static_array_realtype_22044[5] = { 1,0,1,0,0};
static bool static_array_realtype_22045[5] = { 0,0,0,0,0};
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

FUTHARK_FUN_ATTR int futrts_csr_rows_from_pattern_7010(struct futhark_context *ctx, struct memblock *mem_out_p_22046, struct memblock *mem_out_p_22047, int64_t *out_prim_out_22048, int64_t *out_prim_out_22049, struct memblock pat_mem_21827, int64_t m_12184, int64_t n_12185);
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_vjp_ex1_matches_dense(struct futhark_context *ctx, bool *out_prim_out_22053, struct memblock x_mem_21827);
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_vjp_ex2_matches_dense(struct futhark_context *ctx, bool *out_prim_out_22071, struct memblock x_mem_21827);
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_vjp_ex4_matches_dense(struct futhark_context *ctx, bool *out_prim_out_22089, struct memblock x_mem_21827);
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_vjp_ex4_with_row_colors_matches_dense(struct futhark_context *ctx, bool *out_prim_out_22107, struct memblock x_mem_21827);
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_vjp_ex5_matches_dense(struct futhark_context *ctx, bool *out_prim_out_22114, struct memblock x_mem_21827);
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_vjp_zzero_pattern_matches_dense(struct futhark_context *ctx, bool *out_prim_out_22132, struct memblock x_mem_21827);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    #define mem_21809 (ctx->constants->mem_21809)
    #define mem_21812 (ctx->constants->mem_21812)
    #define mem_21814 (ctx->constants->mem_21814)
    #define mem_21819 (ctx->constants->mem_21819)
    #define mem_21820 (ctx->constants->mem_21820)
    #define mem_21826 (ctx->constants->mem_21826)
    
    struct memblock mem_21825;
    
    mem_21825.references = NULL;
    
    struct memblock mem_21824;
    
    mem_21824.references = NULL;
    
    struct memblock mem_21823;
    
    mem_21823.references = NULL;
    
    struct memblock mem_21822;
    
    mem_21822.references = NULL;
    
    struct memblock mem_21821;
    
    mem_21821.references = NULL;
    
    struct memblock mem_21818;
    
    mem_21818.references = NULL;
    
    struct memblock mem_21817;
    
    mem_21817.references = NULL;
    
    struct memblock mem_21816;
    
    mem_21816.references = NULL;
    
    struct memblock mem_21815;
    
    mem_21815.references = NULL;
    
    struct memblock mem_21813;
    
    mem_21813.references = NULL;
    
    struct memblock mem_21811;
    
    mem_21811.references = NULL;
    
    struct memblock mem_21810;
    
    mem_21810.references = NULL;
    
    struct memblock mem_21808;
    
    mem_21808.references = NULL;
    
    struct memblock mem_21807;
    
    mem_21807.references = NULL;
    
    struct memblock mem_21806;
    
    mem_21806.references = NULL;
    mem_21809.references = NULL;
    mem_21812.references = NULL;
    mem_21814.references = NULL;
    mem_21819.references = NULL;
    mem_21820.references = NULL;
    mem_21826.references = NULL;
    // test/test_sparse_jacobian_vjp.fut:24:5-40
    if (memblock_alloc(ctx, &mem_21806, (int64_t) 5, "mem_21806")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:24:5-40
    
    struct memblock static_array_22013 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22030, 0, "static_array_22013"};
    
    // test/test_sparse_jacobian_vjp.fut:24:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21806.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_22013.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    // test/test_sparse_jacobian_vjp.fut:25:5-40
    if (memblock_alloc(ctx, &mem_21807, (int64_t) 5, "mem_21807")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:25:5-40
    
    struct memblock static_array_22014 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22031, 0, "static_array_22014"};
    
    // test/test_sparse_jacobian_vjp.fut:25:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21807.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_22014.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    // test/test_sparse_jacobian_vjp.fut:26:5-40
    if (memblock_alloc(ctx, &mem_21808, (int64_t) 5, "mem_21808")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:26:5-40
    
    struct memblock static_array_22015 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22032, 0, "static_array_22015"};
    
    // test/test_sparse_jacobian_vjp.fut:26:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21808.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_22015.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    if (memblock_alloc(ctx, &mem_21809, (int64_t) 15, "mem_21809")) {
        err = 1;
        goto cleanup;
    }
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21809.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_21806.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21809.mem, (int64_t) 5, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_21807.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21809.mem, (int64_t) 10, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_21808.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    if (memblock_unref(ctx, &mem_21806, "mem_21806") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21807, "mem_21807") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21808, "mem_21808") != 0)
        return 1;
    // test/test_sparse_jacobian_vjp.fut:46:5-33
    if (memblock_alloc(ctx, &mem_21810, (int64_t) 4, "mem_21810")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:46:5-33
    
    struct memblock static_array_22016 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22033, 0, "static_array_22016"};
    
    // test/test_sparse_jacobian_vjp.fut:46:5-33
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21810.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_22016.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
    // test/test_sparse_jacobian_vjp.fut:47:5-33
    if (memblock_alloc(ctx, &mem_21811, (int64_t) 4, "mem_21811")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:47:5-33
    
    struct memblock static_array_22017 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22034, 0, "static_array_22017"};
    
    // test/test_sparse_jacobian_vjp.fut:47:5-33
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21811.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_22017.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
    if (memblock_alloc(ctx, &mem_21812, (int64_t) 8, "mem_21812")) {
        err = 1;
        goto cleanup;
    }
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21812.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_21810.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21812.mem, (int64_t) 4, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_21811.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
    if (memblock_unref(ctx, &mem_21810, "mem_21810") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21811, "mem_21811") != 0)
        return 1;
    // test/test_sparse_jacobian_vjp.fut:65:5-26
    if (memblock_alloc(ctx, &mem_21813, (int64_t) 3, "mem_21813")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:65:5-26
    
    struct memblock static_array_22018 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22035, 0, "static_array_22018"};
    
    // test/test_sparse_jacobian_vjp.fut:65:5-26
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21813.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_22018.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 3});
    if (memblock_alloc(ctx, &mem_21814, (int64_t) 6, "mem_21814")) {
        err = 1;
        goto cleanup;
    }
    for (int64_t nest_i_22019 = 0; nest_i_22019 < (int64_t) 2; nest_i_22019++) {
        lmad_copy_1b(ctx, 1, (uint8_t *) mem_21814.mem, nest_i_22019 * (int64_t) 3, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_21813.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 3});
    }
    if (memblock_unref(ctx, &mem_21813, "mem_21813") != 0)
        return 1;
    // test/test_sparse_jacobian_vjp.fut:88:5-47
    if (memblock_alloc(ctx, &mem_21815, (int64_t) 6, "mem_21815")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:88:5-47
    
    struct memblock static_array_22020 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22036, 0, "static_array_22020"};
    
    // test/test_sparse_jacobian_vjp.fut:88:5-47
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21815.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_22020.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    // test/test_sparse_jacobian_vjp.fut:89:5-47
    if (memblock_alloc(ctx, &mem_21816, (int64_t) 6, "mem_21816")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:89:5-47
    
    struct memblock static_array_22021 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22037, 0, "static_array_22021"};
    
    // test/test_sparse_jacobian_vjp.fut:89:5-47
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21816.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_22021.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    // test/test_sparse_jacobian_vjp.fut:90:5-47
    if (memblock_alloc(ctx, &mem_21817, (int64_t) 6, "mem_21817")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:90:5-47
    
    struct memblock static_array_22022 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22038, 0, "static_array_22022"};
    
    // test/test_sparse_jacobian_vjp.fut:90:5-47
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21817.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_22022.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    // test/test_sparse_jacobian_vjp.fut:91:5-47
    if (memblock_alloc(ctx, &mem_21818, (int64_t) 6, "mem_21818")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:91:5-47
    
    struct memblock static_array_22023 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22039, 0, "static_array_22023"};
    
    // test/test_sparse_jacobian_vjp.fut:91:5-47
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21818.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_22023.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    if (memblock_alloc(ctx, &mem_21819, (int64_t) 24, "mem_21819")) {
        err = 1;
        goto cleanup;
    }
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21819.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_21815.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21819.mem, (int64_t) 6, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_21816.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21819.mem, (int64_t) 12, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_21817.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21819.mem, (int64_t) 18, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_21818.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 6});
    if (memblock_unref(ctx, &mem_21815, "mem_21815") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21816, "mem_21816") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21817, "mem_21817") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21818, "mem_21818") != 0)
        return 1;
    // test/test_sparse_jacobian_vjp.fut:94:31-55
    if (memblock_alloc(ctx, &mem_21820, (int64_t) 32, "mem_21820")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:94:31-55
    
    struct memblock static_array_22024 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22040, 0, "static_array_22024"};
    
    // test/test_sparse_jacobian_vjp.fut:94:31-55
    lmad_copy_8b(ctx, 1, (uint64_t *) mem_21820.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) static_array_22024.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 4});
    // test/test_sparse_jacobian_vjp.fut:125:5-40
    if (memblock_alloc(ctx, &mem_21821, (int64_t) 5, "mem_21821")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:125:5-40
    
    struct memblock static_array_22025 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22041, 0, "static_array_22025"};
    
    // test/test_sparse_jacobian_vjp.fut:125:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21821.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_22025.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    // test/test_sparse_jacobian_vjp.fut:126:5-40
    if (memblock_alloc(ctx, &mem_21822, (int64_t) 5, "mem_21822")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:126:5-40
    
    struct memblock static_array_22026 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22042, 0, "static_array_22026"};
    
    // test/test_sparse_jacobian_vjp.fut:126:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21822.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_22026.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    // test/test_sparse_jacobian_vjp.fut:127:5-40
    if (memblock_alloc(ctx, &mem_21823, (int64_t) 5, "mem_21823")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:127:5-40
    
    struct memblock static_array_22027 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22043, 0, "static_array_22027"};
    
    // test/test_sparse_jacobian_vjp.fut:127:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21823.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_22027.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    // test/test_sparse_jacobian_vjp.fut:128:5-40
    if (memblock_alloc(ctx, &mem_21824, (int64_t) 5, "mem_21824")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:128:5-40
    
    struct memblock static_array_22028 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22044, 0, "static_array_22028"};
    
    // test/test_sparse_jacobian_vjp.fut:128:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21824.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_22028.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    // test/test_sparse_jacobian_vjp.fut:129:5-40
    if (memblock_alloc(ctx, &mem_21825, (int64_t) 5, "mem_21825")) {
        err = 1;
        goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:129:5-40
    
    struct memblock static_array_22029 = (struct memblock) {NULL, (unsigned char *) static_array_realtype_22045, 0, "static_array_22029"};
    
    // test/test_sparse_jacobian_vjp.fut:129:5-40
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21825.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) static_array_22029.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    if (memblock_alloc(ctx, &mem_21826, (int64_t) 25, "mem_21826")) {
        err = 1;
        goto cleanup;
    }
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21826.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_21821.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21826.mem, (int64_t) 5, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_21822.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21826.mem, (int64_t) 10, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_21823.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21826.mem, (int64_t) 15, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_21824.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    lmad_copy_1b(ctx, 1, (uint8_t *) mem_21826.mem, (int64_t) 20, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_21825.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 5});
    if (memblock_unref(ctx, &mem_21821, "mem_21821") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21822, "mem_21822") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21823, "mem_21823") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21824, "mem_21824") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21825, "mem_21825") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21825, "mem_21825") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21824, "mem_21824") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21823, "mem_21823") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21822, "mem_21822") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21821, "mem_21821") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21818, "mem_21818") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21817, "mem_21817") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21816, "mem_21816") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21815, "mem_21815") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21813, "mem_21813") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21811, "mem_21811") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21810, "mem_21810") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21808, "mem_21808") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21807, "mem_21807") != 0)
        return 1;
    if (memblock_unref(ctx, &mem_21806, "mem_21806") != 0)
        return 1;
    #undef mem_21809
    #undef mem_21812
    #undef mem_21814
    #undef mem_21819
    #undef mem_21820
    #undef mem_21826
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    if (memblock_unref(ctx, &ctx->constants->mem_21809, "ctx->constants->mem_21809") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_21812, "ctx->constants->mem_21812") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_21814, "ctx->constants->mem_21814") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_21819, "ctx->constants->mem_21819") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_21820, "ctx->constants->mem_21820") != 0)
        return 1;
    if (memblock_unref(ctx, &ctx->constants->mem_21826, "ctx->constants->mem_21826") != 0)
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

FUTHARK_FUN_ATTR int futrts_csr_rows_from_pattern_7010(struct futhark_context *ctx, struct memblock *mem_out_p_22046, struct memblock *mem_out_p_22047, int64_t *out_prim_out_22048, int64_t *out_prim_out_22049, struct memblock pat_mem_21827, int64_t m_12184, int64_t n_12185)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_21829_cached_sizze_22050 = 0;
    unsigned char *mem_21829 = NULL;
    int64_t mem_21839_cached_sizze_22051 = 0;
    unsigned char *mem_21839 = NULL;
    int64_t mem_21841_cached_sizze_22052 = 0;
    unsigned char *mem_21841 = NULL;
    struct memblock mem_21855;
    
    mem_21855.references = NULL;
    
    struct memblock mem_21837;
    
    mem_21837.references = NULL;
    
    struct memblock mem_out_21963;
    
    mem_out_21963.references = NULL;
    
    struct memblock mem_out_21962;
    
    mem_out_21962.references = NULL;
    
    struct memblock mem_21809 = ctx->constants->mem_21809;
    struct memblock mem_21812 = ctx->constants->mem_21812;
    struct memblock mem_21814 = ctx->constants->mem_21814;
    struct memblock mem_21819 = ctx->constants->mem_21819;
    struct memblock mem_21820 = ctx->constants->mem_21820;
    struct memblock mem_21826 = ctx->constants->mem_21826;
    int64_t prim_out_21964;
    int64_t prim_out_21965;
    int64_t dzlz7bUZLztZRz20Umz20Unz7dUzg_12187 = mul64(m_12184, n_12185);
    
    // src/pattern_csr.fut:7:23-40
    
    int64_t bytes_21828 = (int64_t) 8 * m_12184;
    
    // src/pattern_csr.fut:8:17-20
    
    int64_t dzlz7bUZLzpZRz20Umz20U1z7dUzg_17759 = add64((int64_t) 1, m_12184);
    
    // src/pattern_csr.fut:8:6-26
    
    int64_t bytes_21836 = (int64_t) 8 * dzlz7bUZLzpZRz20Umz20U1z7dUzg_17759;
    
    // src/pattern_csr.fut:8:6-46
    
    bool empty_slice_17766 = m_12184 == (int64_t) 0;
    
    // src/pattern_csr.fut:8:6-46
    
    bool i_p_m_t_s_leq_w_17767 = slt64(m_12184, dzlz7bUZLzpZRz20Umz20U1z7dUzg_17759);
    
    // src/pattern_csr.fut:8:6-46
    
    bool i_lte_j_17768 = sle64((int64_t) 1, dzlz7bUZLzpZRz20Umz20U1z7dUzg_17759);
    
    // src/pattern_csr.fut:8:6-46
    
    bool forwards_ok_17769 = i_p_m_t_s_leq_w_17767 && i_lte_j_17768;
    
    // src/pattern_csr.fut:8:6-46
    
    bool ok_or_empty_17770 = empty_slice_17766 || forwards_ok_17769;
    
    // src/pattern_csr.fut:8:6-46
    
    bool index_certs_17771;
    
    if (!ok_or_empty_17770) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) (int64_t) 1, ":", (long long) dzlz7bUZLzpZRz20Umz20U1z7dUzg_17759, "] out of bounds for array of shape [", (long long) dzlz7bUZLzpZRz20Umz20U1z7dUzg_17759, "].", "-> #0  src/pattern_csr.fut:8:6-46\n   #1  src/pattern_csr.fut:12:27-48\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    
    int64_t bytes_21838 = (int64_t) 8 * dzlz7bUZLztZRz20Umz20Unz7dUzg_12187;
    
    // src/pattern_csr.fut:19:34-61
    if (mem_21839_cached_sizze_22051 < bytes_21838) {
        err = lexical_realloc(ctx, &mem_21839, &mem_21839_cached_sizze_22051, bytes_21838);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    if (mem_21841_cached_sizze_22052 < bytes_21838) {
        err = lexical_realloc(ctx, &mem_21841, &mem_21841_cached_sizze_22052, bytes_21838);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    
    int64_t discard_21702;
    int64_t scanacc_21696 = (int64_t) 0;
    
    for (int64_t i_21699 = 0; i_21699 < dzlz7bUZLztZRz20Umz20Unz7dUzg_12187; i_21699++) {
        int64_t new_index_21769 = squot64(i_21699, n_12185);
        int64_t binop_y_21771 = n_12185 * new_index_21769;
        int64_t new_index_21772 = i_21699 - binop_y_21771;
        bool eta_p_20676 = ((bool *) pat_mem_21827.mem)[new_index_21769 * n_12185 + new_index_21772];
        
        // src/pattern_csr.fut:19:34-61
        
        int64_t defunc_0_f_res_20677 = btoi_bool_i64(eta_p_20676);
        
        // src/pattern_csr.fut:19:34-61
        
        int64_t defunc_0_op_res_17884 = add64(defunc_0_f_res_20677, scanacc_21696);
        
        ((int64_t *) mem_21839)[i_21699] = defunc_0_op_res_17884;
        ((int64_t *) mem_21841)[i_21699] = defunc_0_f_res_20677;
        
        int64_t scanacc_tmp_21966 = defunc_0_op_res_17884;
        
        scanacc_21696 = scanacc_tmp_21966;
    }
    discard_21702 = scanacc_21696;
    // src/pattern_csr.fut:16:43-46
    
    bool zzero_17574 = n_12185 == (int64_t) 0;
    
    // src/pattern_csr.fut:16:43-46
    
    bool nonzzero_17575 = !zzero_17574;
    
    // src/pattern_csr.fut:16:43-46
    
    bool nonzzero_cert_17576;
    
    if (!nonzzero_17575) {
        set_error(ctx, msgprintf("Error: %s\n\nBacktrace:\n%s", "division by zero", "-> #0  src/pattern_csr.fut:16:43-46\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    
    int64_t tmp_17887 = sub64(dzlz7bUZLztZRz20Umz20Unz7dUzg_12187, (int64_t) 1);
    
    // src/pattern_csr.fut:19:34-61
    
    bool y_17889 = slt64(tmp_17887, dzlz7bUZLztZRz20Umz20Unz7dUzg_12187);
    
    // src/pattern_csr.fut:19:34-61
    
    bool x_17888 = sle64((int64_t) 0, tmp_17887);
    
    // src/pattern_csr.fut:19:34-61
    
    bool bounds_check_17890 = x_17888 && y_17889;
    
    // src/pattern_csr.fut:19:34-61
    
    bool cond_17885 = dzlz7bUZLztZRz20Umz20Unz7dUzg_12187 == (int64_t) 0;
    
    // src/pattern_csr.fut:19:34-61
    
    bool protect_assert_disj_17891 = cond_17885 || bounds_check_17890;
    
    // src/pattern_csr.fut:19:34-61
    
    bool index_certs_17892;
    
    if (!protect_assert_disj_17891) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_17887, "] out of bounds for array of shape [", (long long) dzlz7bUZLztZRz20Umz20Unz7dUzg_12187, "].", "-> #0  src/pattern_csr.fut:19:34-61\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    
    bool x_17886 = !cond_17885;
    
    // src/pattern_csr.fut:19:34-61
    
    int64_t m_f_res_17893;
    
    if (x_17886) {
        // src/pattern_csr.fut:19:34-61
        
        int64_t x_21634 = ((int64_t *) mem_21839)[tmp_17887];
        
        m_f_res_17893 = x_21634;
    } else {
        m_f_res_17893 = (int64_t) 0;
    }
    // src/pattern_csr.fut:19:34-61
    
    int64_t m_17895;
    
    if (cond_17885) {
        m_17895 = (int64_t) 0;
    } else {
        m_17895 = m_f_res_17893;
    }
    // src/pattern_csr.fut:19:34-61
    
    int64_t bytes_21854 = (int64_t) 8 * m_17895;
    
    // src/pattern_csr.fut:7:23-40
    if (mem_21829_cached_sizze_22050 < bytes_21828) {
        err = lexical_realloc(ctx, &mem_21829, &mem_21829_cached_sizze_22050, bytes_21828);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/pattern_csr.fut:7:23-40
    
    int64_t discard_21693;
    int64_t scanacc_21689 = (int64_t) 0;
    
    for (int64_t i_21691 = 0; i_21691 < m_12184; i_21691++) {
        // src/pattern_csr.fut:4:3-57
        
        int64_t defunc_0_reduce_res_21633;
        int64_t redout_21686 = (int64_t) 0;
        
        for (int64_t i_21687 = 0; i_21687 < n_12185; i_21687++) {
            bool eta_p_20727 = ((bool *) pat_mem_21827.mem)[i_21691 * n_12185 + i_21687];
            
            // src/pattern_csr.fut:4:17-57
            
            int64_t lifted_lambda_res_20728 = btoi_bool_i64(eta_p_20727);
            
            // src/pattern_csr.fut:4:10-13
            
            int64_t defunc_0_op_res_20724 = add64(lifted_lambda_res_20728, redout_21686);
            int64_t redout_tmp_21971 = defunc_0_op_res_20724;
            
            redout_21686 = redout_tmp_21971;
        }
        defunc_0_reduce_res_21633 = redout_21686;
        // src/pattern_csr.fut:7:28-31
        
        int64_t defunc_0_op_res_17764 = add64(defunc_0_reduce_res_21633, scanacc_21689);
        
        ((int64_t *) mem_21829)[i_21691] = defunc_0_op_res_17764;
        
        int64_t scanacc_tmp_21969 = defunc_0_op_res_17764;
        
        scanacc_21689 = scanacc_tmp_21969;
    }
    discard_21693 = scanacc_21689;
    // src/pattern_csr.fut:8:6-26
    if (memblock_alloc(ctx, &mem_21837, bytes_21836, "mem_21837")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:8:6-26
    for (int64_t nest_i_21972 = 0; nest_i_21972 < dzlz7bUZLzpZRz20Umz20U1z7dUzg_17759; nest_i_21972++) {
        ((int64_t *) mem_21837.mem)[nest_i_21972] = (int64_t) 0;
    }
    // src/pattern_csr.fut:8:6-46
    // src/pattern_csr.fut:8:6-46
    lmad_copy_8b(ctx, 1, (uint64_t *) mem_21837.mem, (int64_t) 1, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_21829, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {m_12184});
    // src/pattern_csr.fut:19:34-61
    if (memblock_alloc(ctx, &mem_21855, bytes_21854, "mem_21855")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    
    bool acc_cert_20682;
    
    // src/pattern_csr.fut:16:30-19:61
    for (int64_t i_21704 = 0; i_21704 < dzlz7bUZLztZRz20Umz20Unz7dUzg_12187; i_21704++) {
        int64_t eta_p_20701 = ((int64_t *) mem_21841)[i_21704];
        int64_t eta_p_20702 = ((int64_t *) mem_21839)[i_21704];
        
        // src/pattern_csr.fut:16:43-46
        
        int64_t lifted_lambda_res_20704 = smod64(i_21704, n_12185);
        
        // src/pattern_csr.fut:19:34-61
        
        bool cond_20706 = eta_p_20701 == (int64_t) 1;
        
        // src/pattern_csr.fut:19:34-61
        
        int64_t lifted_lambda_res_20707;
        
        if (cond_20706) {
            // src/pattern_csr.fut:19:34-61
            
            int64_t lifted_lambda_res_t_res_21635 = sub64(eta_p_20702, (int64_t) 1);
            
            lifted_lambda_res_20707 = lifted_lambda_res_t_res_21635;
        } else {
            lifted_lambda_res_20707 = (int64_t) -1;
        }
        // src/pattern_csr.fut:19:34-61
        // UpdateAcc
        if (sle64((int64_t) 0, lifted_lambda_res_20707) && slt64(lifted_lambda_res_20707, m_17895)) {
            ((int64_t *) mem_21855.mem)[lifted_lambda_res_20707] = lifted_lambda_res_20704;
        }
    }
    if (memblock_set(ctx, &mem_out_21962, &mem_21837, "mem_21837") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_21963, &mem_21855, "mem_21855") != 0)
        return 1;
    prim_out_21964 = dzlz7bUZLzpZRz20Umz20U1z7dUzg_17759;
    prim_out_21965 = m_17895;
    if (memblock_set(ctx, &*mem_out_p_22046, &mem_out_21962, "mem_out_21962") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_22047, &mem_out_21963, "mem_out_21963") != 0)
        return 1;
    *out_prim_out_22048 = prim_out_21964;
    *out_prim_out_22049 = prim_out_21965;
    
  cleanup:
    {
        free(mem_21829);
        free(mem_21839);
        free(mem_21841);
        if (memblock_unref(ctx, &mem_21855, "mem_21855") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21837, "mem_21837") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_21963, "mem_out_21963") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_21962, "mem_out_21962") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_vjp_ex1_matches_dense(struct futhark_context *ctx, bool *out_prim_out_22053, struct memblock x_mem_21827)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_21831_cached_sizze_22054 = 0;
    unsigned char *mem_21831 = NULL;
    int64_t mem_21832_cached_sizze_22055 = 0;
    unsigned char *mem_21832 = NULL;
    int64_t mem_21841_cached_sizze_22056 = 0;
    unsigned char *mem_21841 = NULL;
    int64_t mem_21853_cached_sizze_22057 = 0;
    unsigned char *mem_21853 = NULL;
    int64_t mem_21857_cached_sizze_22058 = 0;
    unsigned char *mem_21857 = NULL;
    int64_t mem_21874_cached_sizze_22059 = 0;
    unsigned char *mem_21874 = NULL;
    int64_t mem_21876_cached_sizze_22060 = 0;
    unsigned char *mem_21876 = NULL;
    int64_t mem_21877_cached_sizze_22061 = 0;
    unsigned char *mem_21877 = NULL;
    int64_t mem_21897_cached_sizze_22062 = 0;
    unsigned char *mem_21897 = NULL;
    int64_t mem_21899_cached_sizze_22063 = 0;
    unsigned char *mem_21899 = NULL;
    int64_t mem_21901_cached_sizze_22064 = 0;
    unsigned char *mem_21901 = NULL;
    int64_t mem_21909_cached_sizze_22065 = 0;
    unsigned char *mem_21909 = NULL;
    int64_t mem_21920_cached_sizze_22066 = 0;
    unsigned char *mem_21920 = NULL;
    int64_t mem_21925_cached_sizze_22067 = 0;
    unsigned char *mem_21925 = NULL;
    int64_t mem_21951_cached_sizze_22068 = 0;
    unsigned char *mem_21951 = NULL;
    int64_t mem_21952_cached_sizze_22069 = 0;
    unsigned char *mem_21952 = NULL;
    int64_t mem_21953_cached_sizze_22070 = 0;
    unsigned char *mem_21953 = NULL;
    struct memblock mem_param_tmp_21967;
    
    mem_param_tmp_21967.references = NULL;
    
    struct memblock mem_21911;
    
    mem_21911.references = NULL;
    
    struct memblock mem_param_21851;
    
    mem_param_21851.references = NULL;
    
    struct memblock ext_mem_21916;
    
    ext_mem_21916.references = NULL;
    
    struct memblock ext_mem_21844;
    
    ext_mem_21844.references = NULL;
    
    struct memblock ext_mem_21845;
    
    ext_mem_21845.references = NULL;
    
    struct memblock mem_21842;
    
    mem_21842.references = NULL;
    
    struct memblock mem_21840;
    
    mem_21840.references = NULL;
    
    struct memblock ext_mem_21828;
    
    ext_mem_21828.references = NULL;
    
    struct memblock ext_mem_21829;
    
    ext_mem_21829.references = NULL;
    
    struct memblock mem_21809 = ctx->constants->mem_21809;
    struct memblock mem_21812 = ctx->constants->mem_21812;
    struct memblock mem_21814 = ctx->constants->mem_21814;
    struct memblock mem_21819 = ctx->constants->mem_21819;
    struct memblock mem_21820 = ctx->constants->mem_21820;
    struct memblock mem_21826 = ctx->constants->mem_21826;
    bool prim_out_21962;
    
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_20122;
    int64_t csr_bipartite_from_pattern_res_20123;
    
    if (futrts_csr_rows_from_pattern_7010(ctx, &ext_mem_21829, &ext_mem_21828, &csr_bipartite_from_pattern_res_20122, &csr_bipartite_from_pattern_res_20123, mem_21809, (int64_t) 3, (int64_t) 5) != 0) {
        err = 1;
        goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:42:15-35
    
    int64_t bytes_21830 = (int64_t) 8 * csr_bipartite_from_pattern_res_20123;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (mem_21832_cached_sizze_22055 < (int64_t) 24) {
        err = lexical_realloc(ctx, &mem_21832, &mem_21832_cached_sizze_22055, (int64_t) 24);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t discard_21691;
    int64_t scanacc_21687 = (int64_t) 0;
    
    for (int64_t i_21689 = 0; i_21689 < (int64_t) 3; i_21689++) {
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t defunc_0_op_res_20138 = add64((int64_t) 1, scanacc_21687);
        
        ((int64_t *) mem_21832)[i_21689] = defunc_0_op_res_20138;
        
        int64_t scanacc_tmp_21963 = defunc_0_op_res_20138;
        
        scanacc_21687 = scanacc_tmp_21963;
    }
    discard_21691 = scanacc_21687;
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t x_20148 = ((int64_t *) mem_21832)[(int64_t) 2];
    
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    bool loop_cond_20162 = slt64((int64_t) 0, x_20148);
    
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t bytes_21839 = (int64_t) 8 * x_20148;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (memblock_alloc(ctx, &mem_21840, bytes_21839, "mem_21840")) {
        err = 1;
        goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    bool acc_cert_20718;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    for (int64_t i_21693 = 0; i_21693 < (int64_t) 3; i_21693++) {
        int64_t eta_p_20730 = ((int64_t *) mem_21832)[i_21693];
        
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t lifted_lambda_res_t_res_20733 = sub64(eta_p_20730, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:194:20-45
        // UpdateAcc
        if (sle64((int64_t) 0, lifted_lambda_res_t_res_20733) && slt64(lifted_lambda_res_t_res_20733, x_20148)) {
            ((int64_t *) mem_21840.mem)[lifted_lambda_res_t_res_20733] = i_21693;
        }
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    if (mem_21841_cached_sizze_22056 < (int64_t) 24) {
        err = lexical_realloc(ctx, &mem_21841, &mem_21841_cached_sizze_22056, (int64_t) 24);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    for (int64_t nest_i_21966 = 0; nest_i_21966 < (int64_t) 3; nest_i_21966++) {
        ((int64_t *) mem_21841)[nest_i_21966] = (int64_t) -1;
    }
    // src/pattern_csr.fut:25:3-39
    if (memblock_alloc(ctx, &mem_21842, (int64_t) 15, "mem_21842")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:25:3-39
    // src/pattern_csr.fut:25:3-39
    lmad_copy_1b(ctx, 2, (uint8_t *) mem_21842.mem, (int64_t) 0, (int64_t []) {(int64_t) 3, (int64_t) 1}, (uint8_t *) mem_21809.mem, (int64_t) 0, (int64_t []) {(int64_t) 1, (int64_t) 5}, (int64_t []) {(int64_t) 5, (int64_t) 3});
    // src/pattern_csr.fut:25:3-39
    
    int64_t csr_cols_from_pattern_res_20127;
    int64_t csr_cols_from_pattern_res_20128;
    
    if (futrts_csr_rows_from_pattern_7010(ctx, &ext_mem_21845, &ext_mem_21844, &csr_cols_from_pattern_res_20127, &csr_cols_from_pattern_res_20128, mem_21842, (int64_t) 5, (int64_t) 3) != 0) {
        err = 1;
        goto cleanup;
    }
    if (memblock_unref(ctx, &mem_21842, "mem_21842") != 0)
        return 1;
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    int64_t vv_color_side_order_res_20163;
    bool vv_color_side_order_res_20164;
    int64_t vv_color_side_order_res_20167;
    int64_t loop_dz2081Uz2083U_20168;
    bool loop_while_20169;
    int64_t color_bound_20172;
    
    if (memblock_set(ctx, &mem_param_21851, &mem_21840, "mem_21840") != 0)
        return 1;
    loop_dz2081Uz2083U_20168 = x_20148;
    loop_while_20169 = loop_cond_20162;
    color_bound_20172 = (int64_t) 1;
    while (loop_while_20169) {
        // src/bgpc_vv_coloring.fut:62:5-64:15
        
        bool cond_20173 = slt64((int64_t) 0, color_bound_20172);
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t bytes_21852 = (int64_t) 8 * loop_dz2081Uz2083U_20168;
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        if (mem_21853_cached_sizze_22057 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21853, &mem_21853_cached_sizze_22057, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:77:24-51
        if (mem_21857_cached_sizze_22058 < color_bound_20172) {
            err = lexical_realloc(ctx, &mem_21857, &mem_21857_cached_sizze_22058, color_bound_20172);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t defunc_0_reduce_res_21672;
        int64_t redout_21695 = (int64_t) -1;
        
        for (int64_t i_21697 = 0; i_21697 < loop_dz2081Uz2083U_20168; i_21697++) {
            int64_t eta_p_21336 = ((int64_t *) mem_param_21851.mem)[i_21697];
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool x_21338 = sle64((int64_t) 0, eta_p_21336);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool y_21339 = slt64(eta_p_21336, csr_bipartite_from_pattern_res_20122);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool bounds_check_21340 = x_21338 && y_21339;
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool index_certs_21341;
            
            if (!bounds_check_21340) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_21336, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/bgpc_vv_coloring.fut:46:44-56\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:44-56
            
            int64_t seen_final_21342 = ((int64_t *) ext_mem_21829.mem)[eta_p_21336];
            
            // src/bgpc_vv_coloring.fut:46:70-72
            
            int64_t seen_final_21343 = add64((int64_t) 1, eta_p_21336);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool x_21344 = sle64((int64_t) 0, seen_final_21343);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool y_21345 = slt64(seen_final_21343, csr_bipartite_from_pattern_res_20122);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool bounds_check_21346 = x_21344 && y_21345;
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool index_certs_21347;
            
            if (!bounds_check_21346) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_21343, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/bgpc_vv_coloring.fut:46:59-73\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:59-73
            
            int64_t seen_final_21348 = ((int64_t *) ext_mem_21829.mem)[seen_final_21343];
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t j_m_i_21349 = sub64(seen_final_21348, seen_final_21342);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool empty_slice_21350 = j_m_i_21349 == (int64_t) 0;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t m_21351 = sub64(j_m_i_21349, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t i_p_m_t_s_21352 = add64(seen_final_21342, m_21351);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_leq_i_p_m_t_s_21353 = sle64((int64_t) 0, i_p_m_t_s_21352);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_p_m_t_s_leq_w_21354 = slt64(i_p_m_t_s_21352, csr_bipartite_from_pattern_res_20123);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_lte_i_21355 = sle64((int64_t) 0, seen_final_21342);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_lte_j_21356 = sle64(seen_final_21342, seen_final_21348);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_21357 = i_p_m_t_s_leq_w_21354 && zzero_lte_i_21355;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_21358 = zzero_leq_i_p_m_t_s_21353 && y_21357;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool forwards_ok_21359 = i_lte_j_21356 && y_21358;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool ok_or_empty_21360 = empty_slice_21350 || forwards_ok_21359;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool index_certs_21361;
            
            if (!ok_or_empty_21360) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_21342, ":", (long long) seen_final_21348, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20123, "].", "-> #0  src/bgpc_vv_coloring.fut:46:5-54:19\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:77:24-51
            for (int64_t nest_i_21975 = 0; nest_i_21975 < color_bound_20172; nest_i_21975++) {
                ((bool *) mem_21857)[nest_i_21975] = 0;
            }
            // src/bgpc_vv_coloring.fut:46:5-54:19
            for (int64_t i_21363 = 0; i_21363 < j_m_i_21349; i_21363++) {
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t index_primexp_21365 = seen_final_21342 + i_21363;
                
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t v_21366 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21365];
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool x_21367 = sle64((int64_t) 0, v_21366);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool y_21368 = slt64(v_21366, csr_cols_from_pattern_res_20127);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool bounds_check_21369 = x_21367 && y_21368;
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool index_certs_21370;
                
                if (!bounds_check_21369) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_21366, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20127, "].", "-> #0  src/bgpc_vv_coloring.fut:49:26-37\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:26-37
                
                int64_t seen_acczq_21371 = ((int64_t *) ext_mem_21845.mem)[v_21366];
                
                // src/bgpc_vv_coloring.fut:49:50-52
                
                int64_t seen_acczq_21372 = add64((int64_t) 1, v_21366);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool x_21373 = sle64((int64_t) 0, seen_acczq_21372);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool y_21374 = slt64(seen_acczq_21372, csr_cols_from_pattern_res_20127);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool bounds_check_21375 = x_21373 && y_21374;
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool index_certs_21376;
                
                if (!bounds_check_21375) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_21372, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20127, "].", "-> #0  src/bgpc_vv_coloring.fut:49:40-53\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:40-53
                
                int64_t seen_acczq_21377 = ((int64_t *) ext_mem_21845.mem)[seen_acczq_21372];
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t j_m_i_21378 = sub64(seen_acczq_21377, seen_acczq_21371);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool empty_slice_21379 = j_m_i_21378 == (int64_t) 0;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t m_21380 = sub64(j_m_i_21378, (int64_t) 1);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t i_p_m_t_s_21381 = add64(seen_acczq_21371, m_21380);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_leq_i_p_m_t_s_21382 = sle64((int64_t) 0, i_p_m_t_s_21381);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_p_m_t_s_leq_w_21383 = slt64(i_p_m_t_s_21381, csr_cols_from_pattern_res_20128);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_lte_i_21384 = sle64((int64_t) 0, seen_acczq_21371);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_lte_j_21385 = sle64(seen_acczq_21371, seen_acczq_21377);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_21386 = i_p_m_t_s_leq_w_21383 && zzero_lte_i_21384;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_21387 = zzero_leq_i_p_m_t_s_21382 && y_21386;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool forwards_ok_21388 = i_lte_j_21385 && y_21387;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool ok_or_empty_21389 = empty_slice_21379 || forwards_ok_21388;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool index_certs_21390;
                
                if (!ok_or_empty_21389) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_21371, ":", (long long) seen_acczq_21377, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20128, "].", "-> #0  src/bgpc_vv_coloring.fut:48:9-53:33\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:48:9-53:33
                for (int64_t i_21392 = 0; i_21392 < j_m_i_21378; i_21392++) {
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t index_primexp_21394 = seen_acczq_21371 + i_21392;
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t u_21395 = ((int64_t *) ext_mem_21844.mem)[index_primexp_21394];
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool x_21396 = sle64((int64_t) 0, u_21395);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool y_21397 = slt64(u_21395, (int64_t) 3);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool bounds_check_21398 = x_21396 && y_21397;
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool index_certs_21399;
                    
                    if (!bounds_check_21398) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_21395, "] out of bounds for array of shape [", (long long) (int64_t) 3, "].", "-> #0  src/bgpc_vv_coloring.fut:50:20-29\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    int64_t cu_21400 = ((int64_t *) mem_21841)[u_21395];
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_21401 = u_21395 == eta_p_21336;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_21402 = !cond_21401;
                    
                    // src/bgpc_vv_coloring.fut:51:30-37
                    
                    bool cond_t_res_21403 = sle64((int64_t) 0, cu_21400);
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    bool x_21404 = cond_21402 && cond_t_res_21403;
                    
                    // src/bgpc_vv_coloring.fut:51:44-67
                    
                    bool cond_t_res_21405 = slt64(cu_21400, color_bound_20172);
                    
                    // src/bgpc_vv_coloring.fut:94:14-103:69
                    
                    bool x_21406 = x_21404 && cond_t_res_21405;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    if (x_21406) {
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool bounds_check_21636 = cond_t_res_21403 && cond_t_res_21405;
                        
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool index_certs_21637;
                        
                        if (!bounds_check_21636) {
                            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) cu_21400, "] out of bounds for array of shape [", (long long) color_bound_20172, "].", "-> #0  src/bgpc_vv_coloring.fut:52:19-50\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                            err = FUTHARK_PROGRAM_ERROR;
                            goto cleanup;
                        }
                        // src/bgpc_vv_coloring.fut:52:19-50
                        ((bool *) mem_21857)[cu_21400] = 1;
                    }
                }
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool loop_cond_21411;
            
            if (cond_20173) {
                // src/bgpc_vv_coloring.fut:63:30-37
                
                bool loop_cond_t_res_21639 = ((bool *) mem_21857)[(int64_t) 0];
                
                loop_cond_21411 = loop_cond_t_res_21639;
            } else {
                loop_cond_21411 = 0;
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool c_final_21413;
            int64_t c_final_21414;
            bool loop_while_21415;
            int64_t c_21416;
            
            loop_while_21415 = loop_cond_21411;
            c_21416 = (int64_t) 0;
            while (loop_while_21415) {
                // src/bgpc_vv_coloring.fut:64:9-15
                
                int64_t loopres_21417 = add64((int64_t) 1, c_21416);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool cond_21418 = slt64(loopres_21417, color_bound_20172);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool loop_cond_21419;
                
                if (cond_21418) {
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool x_21640 = sle64((int64_t) 0, loopres_21417);
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool bounds_check_21641 = cond_21418 && x_21640;
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool index_certs_21642;
                    
                    if (!bounds_check_21641) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) loopres_21417, "] out of bounds for array of shape [", (long long) color_bound_20172, "].", "-> #0  src/bgpc_vv_coloring.fut:63:30-37\n   #1  src/bgpc_vv_coloring.fut:81:6-28\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool loop_cond_t_res_21643 = ((bool *) mem_21857)[loopres_21417];
                    
                    loop_cond_21419 = loop_cond_t_res_21643;
                } else {
                    loop_cond_21419 = 0;
                }
                
                bool loop_while_tmp_21978 = loop_cond_21419;
                int64_t c_tmp_21979 = loopres_21417;
                
                loop_while_21415 = loop_while_tmp_21978;
                c_21416 = c_tmp_21979;
            }
            c_final_21413 = loop_while_21415;
            c_final_21414 = c_21416;
            // src/bgpc_vv_coloring.fut:106:25-32
            
            int64_t max_res_20267 = smax64(c_final_21414, redout_21695);
            
            ((int64_t *) mem_21853)[i_21697] = c_final_21414;
            
            int64_t redout_tmp_21973 = max_res_20267;
            
            redout_21695 = redout_tmp_21973;
        }
        defunc_0_reduce_res_21672 = redout_21695;
        // src/bgpc_vv_coloring.fut:113:17-43
        
        bool acc_cert_20273;
        
        // src/bgpc_vv_coloring.fut:113:17-43
        for (int64_t i_21700 = 0; i_21700 < loop_dz2081Uz2083U_20168; i_21700++) {
            int64_t v_20277 = ((int64_t *) mem_param_21851.mem)[i_21700];
            int64_t v_20278 = ((int64_t *) mem_21853)[i_21700];
            
            // src/bgpc_vv_coloring.fut:113:17-43
            // UpdateAcc
            if (sle64((int64_t) 0, v_20277) && slt64(v_20277, (int64_t) 3)) {
                ((int64_t *) mem_21841)[v_20277] = v_20278;
            }
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21874_cached_sizze_22059 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21874, &mem_21874_cached_sizze_22059, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21876_cached_sizze_22060 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21876, &mem_21876_cached_sizze_22060, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21877_cached_sizze_22061 < loop_dz2081Uz2083U_20168) {
            err = lexical_realloc(ctx, &mem_21877, &mem_21877_cached_sizze_22061, loop_dz2081Uz2083U_20168);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t discard_21712;
        int64_t scanacc_21704 = (int64_t) 0;
        
        for (int64_t i_21708 = 0; i_21708 < loop_dz2081Uz2083U_20168; i_21708++) {
            int64_t eta_p_21260 = ((int64_t *) mem_param_21851.mem)[i_21708];
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool x_21261 = sle64((int64_t) 0, eta_p_21260);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool y_21262 = slt64(eta_p_21260, csr_bipartite_from_pattern_res_20122);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool bounds_check_21263 = x_21261 && y_21262;
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool index_certs_21264;
            
            if (!bounds_check_21263) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_21260, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/bgpc_vv_coloring.fut:126:15-27\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:127:26-28
            
            int64_t k_end_21266 = add64((int64_t) 1, eta_p_21260);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool x_21267 = sle64((int64_t) 0, k_end_21266);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool y_21268 = slt64(k_end_21266, csr_bipartite_from_pattern_res_20122);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool bounds_check_21269 = x_21267 && y_21268;
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool index_certs_21270;
            
            if (!bounds_check_21269) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_end_21266, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/bgpc_vv_coloring.fut:127:15-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:126:15-27
            
            int64_t k0_21265 = ((int64_t *) ext_mem_21829.mem)[eta_p_21260];
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            int64_t k_end_21271 = ((int64_t *) ext_mem_21829.mem)[k_end_21266];
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool cond_21272 = slt64(k0_21265, k_end_21271);
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool loses_conflict_vertex_res_21273;
            bool loses_conflict_vertex_res_21274;
            int64_t loses_conflict_vertex_res_21275;
            bool loop_while_21276;
            bool lost_21277;
            int64_t k_21278;
            
            loop_while_21276 = cond_21272;
            lost_21277 = 0;
            k_21278 = k0_21265;
            while (loop_while_21276) {
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool x_21279 = sle64((int64_t) 0, k_21278);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool y_21280 = slt64(k_21278, csr_bipartite_from_pattern_res_20123);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool bounds_check_21281 = x_21279 && y_21280;
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool index_certs_21282;
                
                if (!bounds_check_21281) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_21278, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20123, "].", "-> #0  src/bgpc_vv_coloring.fut:132:15-26\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:132:15-26
                
                int64_t v_21283 = ((int64_t *) ext_mem_21828.mem)[k_21278];
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool x_21284 = sle64((int64_t) 0, v_21283);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool y_21285 = slt64(v_21283, csr_cols_from_pattern_res_20127);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool bounds_check_21286 = x_21284 && y_21285;
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool index_certs_21287;
                
                if (!bounds_check_21286) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_21283, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20127, "].", "-> #0  src/bgpc_vv_coloring.fut:134:19-30\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:135:29-31
                
                int64_t t_end_21289 = add64((int64_t) 1, v_21283);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool x_21290 = sle64((int64_t) 0, t_end_21289);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool y_21291 = slt64(t_end_21289, csr_cols_from_pattern_res_20127);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool bounds_check_21292 = x_21290 && y_21291;
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool index_certs_21293;
                
                if (!bounds_check_21292) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_end_21289, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20127, "].", "-> #0  src/bgpc_vv_coloring.fut:135:19-32\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:134:19-30
                
                int64_t t0_21288 = ((int64_t *) ext_mem_21845.mem)[v_21283];
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                int64_t t_end_21294 = ((int64_t *) ext_mem_21845.mem)[t_end_21289];
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool cond_21295 = slt64(t0_21288, t_end_21294);
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool loopres_21296;
                bool loopres_21297;
                int64_t loopres_21298;
                bool loop_while_21299;
                bool lost_in_net_21300;
                int64_t t_21301;
                
                loop_while_21299 = cond_21295;
                lost_in_net_21300 = 0;
                t_21301 = t0_21288;
                while (loop_while_21299) {
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool x_21302 = sle64((int64_t) 0, t_21301);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool y_21303 = slt64(t_21301, csr_cols_from_pattern_res_20128);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool bounds_check_21304 = x_21302 && y_21303;
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool index_certs_21305;
                    
                    if (!bounds_check_21304) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_21301, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20128, "].", "-> #0  src/bgpc_vv_coloring.fut:140:19-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    int64_t u_21306 = ((int64_t *) ext_mem_21844.mem)[t_21301];
                    
                    // src/bgpc_vv_coloring.fut:142:13-71
                    
                    bool lost_in_netzq_21307;
                    
                    if (lost_in_net_21300) {
                        lost_in_netzq_21307 = 1;
                    } else {
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_21308 = u_21306 == eta_p_21260;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_21309 = !cond_21308;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_21310;
                        
                        if (cond_21309) {
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool x_21645 = sle64((int64_t) 0, u_21306);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool y_21646 = slt64(u_21306, (int64_t) 3);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool bounds_check_21647 = x_21645 && y_21646;
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool index_certs_21648;
                            
                            if (!bounds_check_21647) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_21306, "] out of bounds for array of shape [", (long long) (int64_t) 3, "].", "-> #0  src/bgpc_vv_coloring.fut:142:39-48\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool y_21650 = slt64(eta_p_21260, (int64_t) 3);
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool bounds_check_21651 = x_21261 && y_21650;
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool index_certs_21652;
                            
                            if (!bounds_check_21651) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_21260, "] out of bounds for array of shape [", (long long) (int64_t) 3, "].", "-> #0  src/bgpc_vv_coloring.fut:142:52-61\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            int64_t zeze_lhs_21649 = ((int64_t *) mem_21841)[u_21306];
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            int64_t zeze_rhs_21653 = ((int64_t *) mem_21841)[eta_p_21260];
                            
                            // src/bgpc_vv_coloring.fut:142:49-61
                            
                            bool cond_t_res_21654 = zeze_lhs_21649 == zeze_rhs_21653;
                            
                            cond_21310 = cond_t_res_21654;
                        } else {
                            cond_21310 = 0;
                        }
                        // src/bgpc_vv_coloring.fut:142:67-70
                        
                        bool lost_in_netzq_f_res_t_res_21321 = slt64(u_21306, eta_p_21260);
                        
                        // src/bgpc_vv_coloring.fut:160:14-168:57
                        
                        bool x_21322 = cond_21310 && lost_in_netzq_f_res_t_res_21321;
                        
                        lost_in_netzq_21307 = x_21322;
                    }
                    // src/bgpc_vv_coloring.fut:143:31-37
                    
                    int64_t tmp_21323 = add64((int64_t) 1, t_21301);
                    
                    // src/bgpc_vv_coloring.fut:139:15-43
                    
                    bool cond_21324 = slt64(tmp_21323, t_end_21294);
                    
                    // src/bgpc_vv_coloring.fut:139:28-43
                    
                    bool not_res_21325 = !lost_in_netzq_21307;
                    
                    // src/bgpc_vv_coloring.fut:166:5-169:10
                    
                    bool x_21326 = cond_21324 && not_res_21325;
                    bool loop_while_tmp_21988 = x_21326;
                    bool lost_in_net_tmp_21989 = lost_in_netzq_21307;
                    int64_t t_tmp_21990 = tmp_21323;
                    
                    loop_while_21299 = loop_while_tmp_21988;
                    lost_in_net_21300 = lost_in_net_tmp_21989;
                    t_21301 = t_tmp_21990;
                }
                loopres_21296 = loop_while_21299;
                loopres_21297 = lost_in_net_21300;
                loopres_21298 = t_21301;
                // src/bgpc_vv_coloring.fut:160:14-168:57
                
                bool lostzq_21327 = lost_21277 || loopres_21297;
                
                // src/bgpc_vv_coloring.fut:146:20-26
                
                int64_t tmp_21328 = add64((int64_t) 1, k_21278);
                
                // src/bgpc_vv_coloring.fut:131:11-32
                
                bool cond_21329 = slt64(tmp_21328, k_end_21271);
                
                // src/bgpc_vv_coloring.fut:131:24-32
                
                bool not_res_21330 = !lostzq_21327;
                
                // src/bgpc_vv_coloring.fut:166:5-169:10
                
                bool x_21331 = cond_21329 && not_res_21330;
                bool loop_while_tmp_21985 = x_21331;
                bool lost_tmp_21986 = lostzq_21327;
                int64_t k_tmp_21987 = tmp_21328;
                
                loop_while_21276 = loop_while_tmp_21985;
                lost_21277 = lost_tmp_21986;
                k_21278 = k_tmp_21987;
            }
            loses_conflict_vertex_res_21273 = loop_while_21276;
            loses_conflict_vertex_res_21274 = lost_21277;
            loses_conflict_vertex_res_21275 = k_21278;
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_f_res_21333 = btoi_bool_i64(loses_conflict_vertex_res_21274);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_op_res_20379 = add64(defunc_0_f_res_21333, scanacc_21704);
            
            ((int64_t *) mem_21874)[i_21708] = defunc_0_op_res_20379;
            ((int64_t *) mem_21876)[i_21708] = defunc_0_f_res_21333;
            ((bool *) mem_21877)[i_21708] = loses_conflict_vertex_res_21274;
            
            int64_t scanacc_tmp_21981 = defunc_0_op_res_20379;
            
            scanacc_21704 = scanacc_tmp_21981;
        }
        discard_21712 = scanacc_21704;
        // src/bgpc_vv_coloring.fut:23:25-28
        
        int64_t tmp_20363 = sub64(loop_dz2081Uz2083U_20168, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool y_20365 = slt64(tmp_20363, loop_dz2081Uz2083U_20168);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool x_20364 = sle64((int64_t) 0, tmp_20363);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool bounds_check_20366 = x_20364 && y_20365;
        
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        bool cond_20361 = loop_dz2081Uz2083U_20168 == (int64_t) 0;
        
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool protect_assert_disj_20367 = cond_20361 || bounds_check_20366;
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool index_certs_20368;
        
        if (!protect_assert_disj_20367) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_20363, "] out of bounds for array of shape [", (long long) loop_dz2081Uz2083U_20168, "].", "-> #0  src/bgpc_vv_coloring.fut:23:10-29\n   #1  src/bgpc_vv_coloring.fut:171:23-55\n   #2  src/bgpc_vv_coloring.fut:186:14-205:56\n   #3  src/bgpc_vv_coloring.fut:220:14-223:64\n   #4  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #5  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #6  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #7  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool x_20362 = !cond_20361;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_f_res_20380;
        
        if (x_20362) {
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t x_21655 = ((int64_t *) mem_21874)[tmp_20363];
            
            m_f_res_20380 = x_21655;
        } else {
            m_f_res_20380 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_20382;
        
        if (cond_20361) {
            m_20382 = (int64_t) 0;
        } else {
            m_20382 = m_f_res_20380;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t bytes_21896 = (int64_t) 8 * m_20382;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21897_cached_sizze_22062 < bytes_21896) {
            err = lexical_realloc(ctx, &mem_21897, &mem_21897_cached_sizze_22062, bytes_21896);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21899_cached_sizze_22063 < bytes_21896) {
            err = lexical_realloc(ctx, &mem_21899, &mem_21899_cached_sizze_22063, bytes_21896);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:18:24-27:44
        if (mem_21901_cached_sizze_22064 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21901, &mem_21901_cached_sizze_22064, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21909_cached_sizze_22065 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21909, &mem_21909_cached_sizze_22065, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        bool acc_cert_21133;
        bool acc_cert_21134;
        
        // src/bgpc_vv_coloring.fut:18:24-27:44
        
        int64_t inpacc_21663;
        int64_t inpacc_21197 = (int64_t) 0;
        
        for (int64_t i_21739 = 0; i_21739 < loop_dz2081Uz2083U_20168; i_21739++) {
            bool eta_p_21778 = ((bool *) mem_21877)[i_21739];
            
            // src/bgpc_vv_coloring.fut:18:28-39
            
            int64_t bool_to_i64_res_21779 = btoi_bool_i64(eta_p_21778);
            int64_t eta_p_21791 = ((int64_t *) mem_21876)[i_21739];
            int64_t eta_p_21792 = ((int64_t *) mem_21874)[i_21739];
            int64_t v_21795 = ((int64_t *) mem_param_21851.mem)[i_21739];
            
            // src/bgpc_vv_coloring.fut:19:29-32
            
            int64_t defunc_0_op_res_21796 = add64(inpacc_21197, bool_to_i64_res_21779);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            bool cond_21797 = eta_p_21791 == (int64_t) 1;
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t lifted_lambda_res_21798;
            
            if (cond_21797) {
                // src/bgpc_vv_coloring.fut:27:5-44
                
                int64_t lifted_lambda_res_t_res_21799 = sub64(eta_p_21792, (int64_t) 1);
                
                lifted_lambda_res_21798 = lifted_lambda_res_t_res_21799;
            } else {
                lifted_lambda_res_21798 = (int64_t) -1;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_21798) && slt64(lifted_lambda_res_21798, m_20382)) {
                ((int64_t *) mem_21899)[lifted_lambda_res_21798] = v_21795;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_21798) && slt64(lifted_lambda_res_21798, m_20382)) {
                ((int64_t *) mem_21897)[lifted_lambda_res_21798] = defunc_0_op_res_21796;
            }
            ((int64_t *) mem_21901)[i_21739] = defunc_0_op_res_21796;
            
            int64_t inpacc_tmp_21991 = defunc_0_op_res_21796;
            
            inpacc_21197 = inpacc_tmp_21991;
        }
        inpacc_21663 = inpacc_21197;
        // src/bgpc_vv_coloring.fut:27:5-44
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_21909, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_21901, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {loop_dz2081Uz2083U_20168});
        // src/bgpc_vv_coloring.fut:23:10-29
        
        int64_t out_szz_f_res_20369;
        
        if (x_20362) {
            // src/bgpc_vv_coloring.fut:171:23-55
            
            int64_t x_21669 = ((int64_t *) mem_21909)[tmp_20363];
            
            out_szz_f_res_20369 = x_21669;
        } else {
            out_szz_f_res_20369 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        int64_t out_szz_20371;
        
        if (cond_20361) {
            out_szz_20371 = (int64_t) 0;
        } else {
            out_szz_20371 = out_szz_f_res_20369;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        
        int64_t bytes_21910 = (int64_t) 8 * out_szz_20371;
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        bool cond_20268 = slt64(defunc_0_reduce_res_21672, (int64_t) 0);
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        int64_t next_color_bound_20269;
        
        if (cond_20268) {
            next_color_bound_20269 = color_bound_20172;
        } else {
            // src/bgpc_vv_coloring.fut:110:40-46
            
            int64_t max_arg1_20270 = add64((int64_t) 2, defunc_0_reduce_res_21672);
            
            // src/bgpc_vv_coloring.fut:110:10-46
            
            int64_t max_res_20271 = smax64(color_bound_20172, max_arg1_20270);
            
            next_color_bound_20269 = max_res_20271;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        if (memblock_alloc(ctx, &mem_21911, bytes_21910, "mem_21911")) {
            err = 1;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        for (int64_t nest_i_21995 = 0; nest_i_21995 < out_szz_20371; nest_i_21995++) {
            ((int64_t *) mem_21911.mem)[nest_i_21995] = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:33:6-27
        
        bool acc_cert_20906;
        
        // src/bgpc_vv_coloring.fut:29:21-33:27
        for (int64_t i_21744 = 0; i_21744 < m_20382; i_21744++) {
            int64_t eta_p_20918 = ((int64_t *) mem_21897)[i_21744];
            int64_t v_20920 = ((int64_t *) mem_21899)[i_21744];
            
            // src/bgpc_vv_coloring.fut:29:44-50
            
            int64_t lifted_lambda_res_20921 = sub64(eta_p_20918, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:33:6-27
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_20921) && slt64(lifted_lambda_res_20921, out_szz_20371)) {
                ((int64_t *) mem_21911.mem)[lifted_lambda_res_20921] = v_20920;
            }
        }
        // src/bgpc_vv_coloring.fut:175:17-48
        
        bool acc_cert_20419;
        
        // src/bgpc_vv_coloring.fut:175:17-48
        for (int64_t i_21746 = 0; i_21746 < out_szz_20371; i_21746++) {
            int64_t v_20423 = ((int64_t *) mem_21911.mem)[i_21746];
            
            // src/bgpc_vv_coloring.fut:175:17-48
            // UpdateAcc
            if (sle64((int64_t) 0, v_20423) && slt64(v_20423, (int64_t) 3)) {
                ((int64_t *) mem_21841)[v_20423] = (int64_t) -1;
            }
        }
        // src/bgpc_vv_coloring.fut:198:20-23
        
        bool loop_cond_20425 = slt64((int64_t) 0, out_szz_20371);
        
        if (memblock_set(ctx, &mem_param_tmp_21967, &mem_21911, "mem_21911") != 0)
            return 1;
        
        int64_t loop_dz2081Uz2083U_tmp_21968 = out_szz_20371;
        bool loop_while_tmp_21969 = loop_cond_20425;
        int64_t color_bound_tmp_21972 = next_color_bound_20269;
        
        if (memblock_set(ctx, &mem_param_21851, &mem_param_tmp_21967, "mem_param_tmp_21967") != 0)
            return 1;
        loop_dz2081Uz2083U_20168 = loop_dz2081Uz2083U_tmp_21968;
        loop_while_20169 = loop_while_tmp_21969;
        color_bound_20172 = color_bound_tmp_21972;
    }
    if (memblock_set(ctx, &ext_mem_21916, &mem_param_21851, "mem_param_21851") != 0)
        return 1;
    vv_color_side_order_res_20163 = loop_dz2081Uz2083U_20168;
    vv_color_side_order_res_20164 = loop_while_20169;
    vv_color_side_order_res_20167 = color_bound_20172;
    if (memblock_unref(ctx, &mem_21840, "mem_21840") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21844, "ext_mem_21844") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21845, "ext_mem_21845") != 0)
        return 1;
    // src/sparse_jacobian_vjp.fut:78:8-87:44
    
    int64_t x_21683;
    int64_t redout_21747 = (int64_t) 0;
    
    for (int64_t i_21748 = 0; i_21748 < (int64_t) 3; i_21748++) {
        int64_t x_20428 = ((int64_t *) mem_21841)[i_21748];
        
        // src/sparse_jacobian_vjp.fut:14:22-29
        
        int64_t max_res_20431 = smax64(x_20428, redout_21747);
        int64_t redout_tmp_21998 = max_res_20431;
        
        redout_21747 = redout_tmp_21998;
    }
    x_21683 = redout_21747;
    // src/sparse_jacobian_vjp.fut:14:13-45
    
    int64_t num_colors_of_res_f_res_20432 = add64((int64_t) 1, x_21683);
    
    // src/sparse_jacobian_vjp.fut:29:11-18
    
    bool bounds_invalid_upwards_20434 = slt64(num_colors_of_res_f_res_20432, (int64_t) 0);
    
    // src/sparse_jacobian_vjp.fut:29:11-18
    
    bool valid_20435 = !bounds_invalid_upwards_20434;
    
    // src/sparse_jacobian_vjp.fut:29:11-18
    
    bool range_valid_c_20436;
    
    if (!valid_20435) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Range ", (long long) (int64_t) 0, "..", (long long) (int64_t) 1, "..<", (long long) num_colors_of_res_f_res_20432, " is invalid.", "-> #0  src/sparse_jacobian_vjp.fut:29:11-18\n   #1  src/sparse_jacobian_vjp.fut:78:8-87:44\n   #2  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #3  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #4  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:21:8-29:18
    
    int64_t bytes_21919 = (int64_t) 40 * num_colors_of_res_f_res_20432;
    
    // src/sparse_jacobian_vjp.fut:42:15-35
    if (mem_21831_cached_sizze_22054 < bytes_21830) {
        err = lexical_realloc(ctx, &mem_21831, &mem_21831_cached_sizze_22054, bytes_21830);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:42:15-35
    for (int64_t nest_i_21999 = 0; nest_i_21999 < csr_bipartite_from_pattern_res_20123; nest_i_21999++) {
        ((double *) mem_21831)[nest_i_21999] = 0.0;
    }
    
    double zt_lhs_20549 = ((double *) x_mem_21827.mem)[(int64_t) 2];
    
    // src/sparse_jacobian_vjp.fut:21:8-29:18
    if (mem_21920_cached_sizze_22066 < bytes_21919) {
        err = lexical_realloc(ctx, &mem_21920, &mem_21920_cached_sizze_22066, bytes_21919);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:17:3-61
    if (mem_21925_cached_sizze_22067 < (int64_t) 24) {
        err = lexical_realloc(ctx, &mem_21925, &mem_21925_cached_sizze_22067, (int64_t) 24);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:21:8-29:18
    for (int64_t i_21755 = 0; i_21755 < num_colors_of_res_f_res_20432; i_21755++) {
        // src/sparse_jacobian_vjp.fut:17:3-61
        for (int64_t i_21751 = 0; i_21751 < (int64_t) 3; i_21751++) {
            int64_t eta_p_20445 = ((int64_t *) mem_21841)[i_21751];
            
            // src/sparse_jacobian_vjp.fut:17:15-49
            
            bool cond_20446 = eta_p_20445 == i_21755;
            
            // src/sparse_jacobian_vjp.fut:17:15-49
            
            double lifted_lambda_res_20447;
            
            if (cond_20446) {
                lifted_lambda_res_20447 = 1.0;
            } else {
                lifted_lambda_res_20447 = 0.0;
            }
            ((double *) mem_21925)[i_21751] = lifted_lambda_res_20447;
        }
        
        double elem_adj_21572 = ((double *) mem_21925)[(int64_t) 0];
        double elem_adj_21573 = ((double *) mem_21925)[(int64_t) 1];
        double elem_adj_21574 = ((double *) mem_21925)[(int64_t) 2];
        double binop_x_adj_21575 = zt_lhs_20549 * elem_adj_21574;
        double zt_lhs_adj_21577 = binop_x_adj_21575 + binop_x_adj_21575;
        double binop_y_adj_21579 = 5.0 * elem_adj_21573;
        double binop_y_adj_21583 = 2.0 * elem_adj_21572;
        
        for (int64_t nest_i_22002 = 0; nest_i_22002 < (int64_t) 5; nest_i_22002++) {
            ((double *) mem_21920)[i_21755 * (int64_t) 5 + nest_i_22002] = 0.0;
        }
        ((double *) mem_21920)[i_21755 * (int64_t) 5] = elem_adj_21572;
        ((double *) mem_21920)[i_21755 * (int64_t) 5 + (int64_t) 3] = binop_y_adj_21583;
        ((double *) mem_21920)[i_21755 * (int64_t) 5 + (int64_t) 1] = binop_y_adj_21579;
        ((double *) mem_21920)[i_21755 * (int64_t) 5 + (int64_t) 2] = zt_lhs_adj_21577;
    }
    // src/sparse_jacobian_vjp.fut:45:5-53:27
    
    bool compressed_to_csr_vals_res_20456;
    int64_t compressed_to_csr_vals_res_20458;
    bool loop_while_20459;
    int64_t i_20461;
    
    loop_while_20459 = 1;
    i_20461 = (int64_t) 0;
    while (loop_while_20459) {
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool x_20462 = sle64((int64_t) 0, i_20461);
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool y_20463 = slt64(i_20461, csr_bipartite_from_pattern_res_20122);
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool bounds_check_20464 = x_20462 && y_20463;
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool index_certs_20465;
        
        if (!bounds_check_20464) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_20461, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/sparse_jacobian_vjp.fut:47:15-26\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        int64_t s_20466 = ((int64_t *) ext_mem_21829.mem)[i_20461];
        
        // src/sparse_jacobian_vjp.fut:48:25-27
        
        int64_t e_20467 = add64((int64_t) 1, i_20461);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool x_20468 = sle64((int64_t) 0, e_20467);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool y_20469 = slt64(e_20467, csr_bipartite_from_pattern_res_20122);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool bounds_check_20470 = x_20468 && y_20469;
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool index_certs_20471;
        
        if (!bounds_check_20470) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_20467, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/sparse_jacobian_vjp.fut:48:15-28\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        int64_t e_20472 = ((int64_t *) ext_mem_21829.mem)[e_20467];
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t j_m_i_20473 = sub64(e_20472, s_20466);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool empty_slice_20474 = j_m_i_20473 == (int64_t) 0;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t m_20475 = sub64(j_m_i_20473, (int64_t) 1);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t i_p_m_t_s_20476 = add64(s_20466, m_20475);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_20477 = sle64((int64_t) 0, i_p_m_t_s_20476);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_20478 = slt64(i_p_m_t_s_20476, csr_bipartite_from_pattern_res_20123);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool zzero_lte_i_20479 = sle64((int64_t) 0, s_20466);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool i_lte_j_20480 = sle64(s_20466, e_20472);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool y_20481 = i_p_m_t_s_leq_w_20478 && zzero_lte_i_20479;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool y_20482 = zzero_leq_i_p_m_t_s_20477 && y_20481;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool forwards_ok_20483 = i_lte_j_20480 && y_20482;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool ok_or_empty_20484 = empty_slice_20474 || forwards_ok_20483;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool index_certs_20485;
        
        if (!ok_or_empty_20484) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_20466, ":", (long long) e_20472, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20123, "].", "-> #0  src/sparse_jacobian_vjp.fut:49:18-30\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool y_20487 = slt64(i_20461, (int64_t) 3);
        
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool bounds_check_20488 = x_20462 && y_20487;
        
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool index_certs_20489;
        
        if (!bounds_check_20488) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_20461, "] out of bounds for array of shape [", (long long) (int64_t) 3, "].", "-> #0  src/sparse_jacobian_vjp.fut:50:16-29\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        int64_t rc_20490 = ((int64_t *) mem_21841)[i_20461];
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool y_20492 = slt64(rc_20490, num_colors_of_res_f_res_20432);
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool x_20491 = sle64((int64_t) 0, rc_20490);
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool bounds_check_20493 = x_20491 && y_20492;
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool index_certs_20494;
        
        if (!bounds_check_20493) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) rc_20490, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_20432, "].", "-> #0  src/sparse_jacobian_vjp.fut:51:28-34\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:51:17-43
        for (int64_t i_21759 = 0; i_21759 < j_m_i_20473; i_21759++) {
            int64_t index_primexp_21771 = s_20466 + i_21759;
            int64_t eta_p_20496 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21771];
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool x_20497 = sle64((int64_t) 0, eta_p_20496);
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool y_20498 = slt64(eta_p_20496, (int64_t) 5);
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool bounds_check_20499 = x_20497 && y_20498;
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool index_certs_20500;
            
            if (!bounds_check_20499) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_20496, "] out of bounds for array of shape [", (long long) (int64_t) 5, "].", "-> #0  src/sparse_jacobian_vjp.fut:51:28-37\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_vjp.fut:51:17-43
            
            double lifted_lambda_res_20501 = ((double *) mem_21920)[rc_20490 * (int64_t) 5 + eta_p_20496];
            
            ((double *) mem_21831)[s_20466 + i_21759] = lifted_lambda_res_20501;
        }
        // src/sparse_jacobian_vjp.fut:46:13-16
        
        bool loop_cond_20503 = slt64(e_20467, (int64_t) 3);
        bool loop_while_tmp_22003 = loop_cond_20503;
        int64_t i_tmp_22005 = e_20467;
        
        loop_while_20459 = loop_while_tmp_22003;
        i_20461 = i_tmp_22005;
    }
    compressed_to_csr_vals_res_20456 = loop_while_20459;
    compressed_to_csr_vals_res_20458 = i_20461;
    // src/dense_jacobian.fut:5:3-21
    if (mem_21951_cached_sizze_22068 < (int64_t) 24) {
        err = lexical_realloc(ctx, &mem_21951, &mem_21951_cached_sizze_22068, (int64_t) 24);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    if (mem_21952_cached_sizze_22069 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_21952, &mem_21952_cached_sizze_22069, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:68:29-47
    if (mem_21953_cached_sizze_22070 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_21953, &mem_21953_cached_sizze_22070, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:7:19-9:39
    
    bool defunc_0_reduce_res_21685;
    bool redout_21765 = 1;
    
    for (int64_t i_21766 = 0; i_21766 < (int64_t) 3; i_21766++) {
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        bool y_20829 = slt64(i_21766, csr_bipartite_from_pattern_res_20122);
        
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        bool index_certs_20831;
        
        if (!y_20829) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_21766, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/sparse_jacobian_vjp.fut:64:17-28\n   #1  src/sparse_jacobian_vjp.fut:132:18-138:40\n   #2  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        int64_t s_20832 = ((int64_t *) ext_mem_21829.mem)[i_21766];
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool zzero_lte_i_20845 = sle64((int64_t) 0, s_20832);
        
        // src/sparse_jacobian_vjp.fut:65:27-29
        
        int64_t e_20833 = add64((int64_t) 1, i_21766);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool y_20835 = slt64(e_20833, csr_bipartite_from_pattern_res_20122);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool x_20834 = sle64((int64_t) 0, e_20833);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool bounds_check_20836 = x_20834 && y_20835;
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool index_certs_20837;
        
        if (!bounds_check_20836) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_20833, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/sparse_jacobian_vjp.fut:65:17-30\n   #1  src/sparse_jacobian_vjp.fut:132:18-138:40\n   #2  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        int64_t e_20838 = ((int64_t *) ext_mem_21829.mem)[e_20833];
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t j_m_i_20839 = sub64(e_20838, s_20832);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t m_20841 = sub64(j_m_i_20839, (int64_t) 1);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t i_p_m_t_s_20842 = add64(s_20832, m_20841);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool i_p_m_t_s_leq_w_20844 = slt64(i_p_m_t_s_20842, csr_bipartite_from_pattern_res_20123);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool y_20853 = i_p_m_t_s_leq_w_20844 && zzero_lte_i_20845;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool zzero_leq_i_p_m_t_s_20843 = sle64((int64_t) 0, i_p_m_t_s_20842);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool y_20854 = zzero_leq_i_p_m_t_s_20843 && y_20853;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool i_lte_j_20846 = sle64(s_20832, e_20838);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool forwards_ok_20855 = i_lte_j_20846 && y_20854;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool empty_slice_20840 = j_m_i_20839 == (int64_t) 0;
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool ok_or_empty_20856 = empty_slice_20840 || forwards_ok_20855;
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool index_certs_20857;
        
        if (!ok_or_empty_20856) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_20832, ":", (long long) e_20838, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20123, "].", "-> #0  src/sparse_jacobian_vjp.fut:67:20-29\n   #1  src/sparse_jacobian_vjp.fut:132:18-138:40\n   #2  test/test_sparse_jacobian_vjp.fut:17:15-36:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/dense_jacobian.fut:5:3-21
        for (int64_t nest_i_22008 = 0; nest_i_22008 < (int64_t) 3; nest_i_22008++) {
            ((double *) mem_21951)[nest_i_22008] = 0.0;
        }
        // src/dense_jacobian.fut:5:3-39
        ((double *) mem_21951)[i_21766] = 1.0;
        
        double elem_adj_21590 = ((double *) mem_21951)[(int64_t) 0];
        double elem_adj_21591 = ((double *) mem_21951)[(int64_t) 1];
        double elem_adj_21592 = ((double *) mem_21951)[(int64_t) 2];
        double binop_x_adj_21593 = zt_lhs_20549 * elem_adj_21592;
        double zt_lhs_adj_21595 = binop_x_adj_21593 + binop_x_adj_21593;
        double binop_y_adj_21597 = 5.0 * elem_adj_21591;
        double binop_y_adj_21601 = 2.0 * elem_adj_21590;
        
        for (int64_t nest_i_22009 = 0; nest_i_22009 < (int64_t) 5; nest_i_22009++) {
            ((double *) mem_21952)[nest_i_22009] = 0.0;
        }
        ((double *) mem_21952)[(int64_t) 0] = elem_adj_21590;
        ((double *) mem_21952)[(int64_t) 3] = binop_y_adj_21601;
        ((double *) mem_21952)[(int64_t) 1] = binop_y_adj_21597;
        ((double *) mem_21952)[(int64_t) 2] = zt_lhs_adj_21595;
        // src/sparse_jacobian_vjp.fut:68:29-47
        for (int64_t nest_i_22010 = 0; nest_i_22010 < (int64_t) 5; nest_i_22010++) {
            ((double *) mem_21953)[nest_i_22010] = 0.0;
        }
        // src/sparse_jacobian_vjp.fut:69:12-32
        
        bool acc_cert_20861;
        
        // src/sparse_jacobian_vjp.fut:69:12-32
        for (int64_t i_21762 = 0; i_21762 < j_m_i_20839; i_21762++) {
            int64_t index_primexp_21768 = s_20832 + i_21762;
            int64_t v_20865 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21768];
            double v_20866 = ((double *) mem_21831)[index_primexp_21768];
            
            // src/sparse_jacobian_vjp.fut:69:12-32
            // UpdateAcc
            if (sle64((int64_t) 0, v_20865) && slt64(v_20865, (int64_t) 5)) {
                ((double *) mem_21953)[v_20865] = v_20866;
            }
        }
        // test/test_sparse_jacobian_vjp.fut:8:5-66
        
        bool defunc_0_reduce_res_21681;
        bool redout_21763 = 1;
        
        for (int64_t i_21764 = 0; i_21764 < (int64_t) 5; i_21764++) {
            bool eta_p_20889 = ((bool *) mem_21809.mem)[i_21766 * (int64_t) 5 + i_21764];
            double eta_p_20890 = ((double *) mem_21952)[i_21764];
            double eta_p_20891 = ((double *) mem_21953)[i_21764];
            
            // test/test_sparse_jacobian_vjp.fut:13:25-48
            
            double lifted_lambda_res_20892;
            
            if (eta_p_20889) {
                lifted_lambda_res_20892 = eta_p_20890;
            } else {
                lifted_lambda_res_20892 = 0.0;
            }
            // test/test_sparse_jacobian_vjp.fut:8:48-51
            
            double abs_arg0_20894 = eta_p_20891 - lifted_lambda_res_20892;
            
            // test/test_sparse_jacobian_vjp.fut:8:37-51
            
            double abs_res_20895 = fabs64(abs_arg0_20894);
            
            // test/test_sparse_jacobian_vjp.fut:8:53-59
            
            bool lifted_lambda_res_20896 = abs_res_20895 <= 1.0e-9;
            
            // test/test_sparse_jacobian_vjp.fut:8:5-66
            
            bool x_20879 = lifted_lambda_res_20896 && redout_21763;
            bool redout_tmp_22012 = x_20879;
            
            redout_21763 = redout_tmp_22012;
        }
        defunc_0_reduce_res_21681 = redout_21763;
        // test/test_sparse_jacobian_vjp.fut:9:6-39
        
        bool x_20631 = defunc_0_reduce_res_21681 && redout_21765;
        bool redout_tmp_22007 = x_20631;
        
        redout_21765 = redout_tmp_22007;
    }
    defunc_0_reduce_res_21685 = redout_21765;
    if (memblock_unref(ctx, &ext_mem_21828, "ext_mem_21828") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21829, "ext_mem_21829") != 0)
        return 1;
    prim_out_21962 = defunc_0_reduce_res_21685;
    *out_prim_out_22053 = prim_out_21962;
    
  cleanup:
    {
        free(mem_21831);
        free(mem_21832);
        free(mem_21841);
        free(mem_21853);
        free(mem_21857);
        free(mem_21874);
        free(mem_21876);
        free(mem_21877);
        free(mem_21897);
        free(mem_21899);
        free(mem_21901);
        free(mem_21909);
        free(mem_21920);
        free(mem_21925);
        free(mem_21951);
        free(mem_21952);
        free(mem_21953);
        if (memblock_unref(ctx, &mem_param_tmp_21967, "mem_param_tmp_21967") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21911, "mem_21911") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_21851, "mem_param_21851") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21916, "ext_mem_21916") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21844, "ext_mem_21844") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21845, "ext_mem_21845") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21842, "mem_21842") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21840, "mem_21840") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21828, "ext_mem_21828") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21829, "ext_mem_21829") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_vjp_ex2_matches_dense(struct futhark_context *ctx, bool *out_prim_out_22071, struct memblock x_mem_21827)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_21831_cached_sizze_22072 = 0;
    unsigned char *mem_21831 = NULL;
    int64_t mem_21832_cached_sizze_22073 = 0;
    unsigned char *mem_21832 = NULL;
    int64_t mem_21841_cached_sizze_22074 = 0;
    unsigned char *mem_21841 = NULL;
    int64_t mem_21853_cached_sizze_22075 = 0;
    unsigned char *mem_21853 = NULL;
    int64_t mem_21857_cached_sizze_22076 = 0;
    unsigned char *mem_21857 = NULL;
    int64_t mem_21874_cached_sizze_22077 = 0;
    unsigned char *mem_21874 = NULL;
    int64_t mem_21876_cached_sizze_22078 = 0;
    unsigned char *mem_21876 = NULL;
    int64_t mem_21877_cached_sizze_22079 = 0;
    unsigned char *mem_21877 = NULL;
    int64_t mem_21897_cached_sizze_22080 = 0;
    unsigned char *mem_21897 = NULL;
    int64_t mem_21899_cached_sizze_22081 = 0;
    unsigned char *mem_21899 = NULL;
    int64_t mem_21901_cached_sizze_22082 = 0;
    unsigned char *mem_21901 = NULL;
    int64_t mem_21909_cached_sizze_22083 = 0;
    unsigned char *mem_21909 = NULL;
    int64_t mem_21920_cached_sizze_22084 = 0;
    unsigned char *mem_21920 = NULL;
    int64_t mem_21925_cached_sizze_22085 = 0;
    unsigned char *mem_21925 = NULL;
    int64_t mem_21951_cached_sizze_22086 = 0;
    unsigned char *mem_21951 = NULL;
    int64_t mem_21952_cached_sizze_22087 = 0;
    unsigned char *mem_21952 = NULL;
    int64_t mem_21953_cached_sizze_22088 = 0;
    unsigned char *mem_21953 = NULL;
    struct memblock mem_param_tmp_21967;
    
    mem_param_tmp_21967.references = NULL;
    
    struct memblock mem_21911;
    
    mem_21911.references = NULL;
    
    struct memblock mem_param_21851;
    
    mem_param_21851.references = NULL;
    
    struct memblock ext_mem_21916;
    
    ext_mem_21916.references = NULL;
    
    struct memblock ext_mem_21844;
    
    ext_mem_21844.references = NULL;
    
    struct memblock ext_mem_21845;
    
    ext_mem_21845.references = NULL;
    
    struct memblock mem_21842;
    
    mem_21842.references = NULL;
    
    struct memblock mem_21840;
    
    mem_21840.references = NULL;
    
    struct memblock ext_mem_21828;
    
    ext_mem_21828.references = NULL;
    
    struct memblock ext_mem_21829;
    
    ext_mem_21829.references = NULL;
    
    struct memblock mem_21809 = ctx->constants->mem_21809;
    struct memblock mem_21812 = ctx->constants->mem_21812;
    struct memblock mem_21814 = ctx->constants->mem_21814;
    struct memblock mem_21819 = ctx->constants->mem_21819;
    struct memblock mem_21820 = ctx->constants->mem_21820;
    struct memblock mem_21826 = ctx->constants->mem_21826;
    bool prim_out_21962;
    
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_20122;
    int64_t csr_bipartite_from_pattern_res_20123;
    
    if (futrts_csr_rows_from_pattern_7010(ctx, &ext_mem_21829, &ext_mem_21828, &csr_bipartite_from_pattern_res_20122, &csr_bipartite_from_pattern_res_20123, mem_21812, (int64_t) 2, (int64_t) 4) != 0) {
        err = 1;
        goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:42:15-35
    
    int64_t bytes_21830 = (int64_t) 8 * csr_bipartite_from_pattern_res_20123;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (mem_21832_cached_sizze_22073 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_21832, &mem_21832_cached_sizze_22073, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t discard_21691;
    int64_t scanacc_21687 = (int64_t) 0;
    
    for (int64_t i_21689 = 0; i_21689 < (int64_t) 2; i_21689++) {
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t defunc_0_op_res_20138 = add64((int64_t) 1, scanacc_21687);
        
        ((int64_t *) mem_21832)[i_21689] = defunc_0_op_res_20138;
        
        int64_t scanacc_tmp_21963 = defunc_0_op_res_20138;
        
        scanacc_21687 = scanacc_tmp_21963;
    }
    discard_21691 = scanacc_21687;
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t x_20148 = ((int64_t *) mem_21832)[(int64_t) 1];
    
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    bool loop_cond_20162 = slt64((int64_t) 0, x_20148);
    
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t bytes_21839 = (int64_t) 8 * x_20148;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (memblock_alloc(ctx, &mem_21840, bytes_21839, "mem_21840")) {
        err = 1;
        goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    bool acc_cert_20718;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    for (int64_t i_21693 = 0; i_21693 < (int64_t) 2; i_21693++) {
        int64_t eta_p_20730 = ((int64_t *) mem_21832)[i_21693];
        
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t lifted_lambda_res_t_res_20733 = sub64(eta_p_20730, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:194:20-45
        // UpdateAcc
        if (sle64((int64_t) 0, lifted_lambda_res_t_res_20733) && slt64(lifted_lambda_res_t_res_20733, x_20148)) {
            ((int64_t *) mem_21840.mem)[lifted_lambda_res_t_res_20733] = i_21693;
        }
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    if (mem_21841_cached_sizze_22074 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_21841, &mem_21841_cached_sizze_22074, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    for (int64_t nest_i_21966 = 0; nest_i_21966 < (int64_t) 2; nest_i_21966++) {
        ((int64_t *) mem_21841)[nest_i_21966] = (int64_t) -1;
    }
    // src/pattern_csr.fut:25:3-39
    if (memblock_alloc(ctx, &mem_21842, (int64_t) 8, "mem_21842")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:25:3-39
    // src/pattern_csr.fut:25:3-39
    lmad_copy_1b(ctx, 2, (uint8_t *) mem_21842.mem, (int64_t) 0, (int64_t []) {(int64_t) 2, (int64_t) 1}, (uint8_t *) mem_21812.mem, (int64_t) 0, (int64_t []) {(int64_t) 1, (int64_t) 4}, (int64_t []) {(int64_t) 4, (int64_t) 2});
    // src/pattern_csr.fut:25:3-39
    
    int64_t csr_cols_from_pattern_res_20127;
    int64_t csr_cols_from_pattern_res_20128;
    
    if (futrts_csr_rows_from_pattern_7010(ctx, &ext_mem_21845, &ext_mem_21844, &csr_cols_from_pattern_res_20127, &csr_cols_from_pattern_res_20128, mem_21842, (int64_t) 4, (int64_t) 2) != 0) {
        err = 1;
        goto cleanup;
    }
    if (memblock_unref(ctx, &mem_21842, "mem_21842") != 0)
        return 1;
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    int64_t vv_color_side_order_res_20163;
    bool vv_color_side_order_res_20164;
    int64_t vv_color_side_order_res_20167;
    int64_t loop_dz2081Uz2083U_20168;
    bool loop_while_20169;
    int64_t color_bound_20172;
    
    if (memblock_set(ctx, &mem_param_21851, &mem_21840, "mem_21840") != 0)
        return 1;
    loop_dz2081Uz2083U_20168 = x_20148;
    loop_while_20169 = loop_cond_20162;
    color_bound_20172 = (int64_t) 1;
    while (loop_while_20169) {
        // src/bgpc_vv_coloring.fut:62:5-64:15
        
        bool cond_20173 = slt64((int64_t) 0, color_bound_20172);
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t bytes_21852 = (int64_t) 8 * loop_dz2081Uz2083U_20168;
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        if (mem_21853_cached_sizze_22075 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21853, &mem_21853_cached_sizze_22075, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:77:24-51
        if (mem_21857_cached_sizze_22076 < color_bound_20172) {
            err = lexical_realloc(ctx, &mem_21857, &mem_21857_cached_sizze_22076, color_bound_20172);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t defunc_0_reduce_res_21672;
        int64_t redout_21695 = (int64_t) -1;
        
        for (int64_t i_21697 = 0; i_21697 < loop_dz2081Uz2083U_20168; i_21697++) {
            int64_t eta_p_21333 = ((int64_t *) mem_param_21851.mem)[i_21697];
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool x_21335 = sle64((int64_t) 0, eta_p_21333);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool y_21336 = slt64(eta_p_21333, csr_bipartite_from_pattern_res_20122);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool bounds_check_21337 = x_21335 && y_21336;
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool index_certs_21338;
            
            if (!bounds_check_21337) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_21333, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/bgpc_vv_coloring.fut:46:44-56\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:44-56
            
            int64_t seen_final_21339 = ((int64_t *) ext_mem_21829.mem)[eta_p_21333];
            
            // src/bgpc_vv_coloring.fut:46:70-72
            
            int64_t seen_final_21340 = add64((int64_t) 1, eta_p_21333);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool x_21341 = sle64((int64_t) 0, seen_final_21340);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool y_21342 = slt64(seen_final_21340, csr_bipartite_from_pattern_res_20122);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool bounds_check_21343 = x_21341 && y_21342;
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool index_certs_21344;
            
            if (!bounds_check_21343) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_21340, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/bgpc_vv_coloring.fut:46:59-73\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:59-73
            
            int64_t seen_final_21345 = ((int64_t *) ext_mem_21829.mem)[seen_final_21340];
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t j_m_i_21346 = sub64(seen_final_21345, seen_final_21339);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool empty_slice_21347 = j_m_i_21346 == (int64_t) 0;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t m_21348 = sub64(j_m_i_21346, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t i_p_m_t_s_21349 = add64(seen_final_21339, m_21348);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_leq_i_p_m_t_s_21350 = sle64((int64_t) 0, i_p_m_t_s_21349);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_p_m_t_s_leq_w_21351 = slt64(i_p_m_t_s_21349, csr_bipartite_from_pattern_res_20123);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_lte_i_21352 = sle64((int64_t) 0, seen_final_21339);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_lte_j_21353 = sle64(seen_final_21339, seen_final_21345);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_21354 = i_p_m_t_s_leq_w_21351 && zzero_lte_i_21352;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_21355 = zzero_leq_i_p_m_t_s_21350 && y_21354;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool forwards_ok_21356 = i_lte_j_21353 && y_21355;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool ok_or_empty_21357 = empty_slice_21347 || forwards_ok_21356;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool index_certs_21358;
            
            if (!ok_or_empty_21357) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_21339, ":", (long long) seen_final_21345, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20123, "].", "-> #0  src/bgpc_vv_coloring.fut:46:5-54:19\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:77:24-51
            for (int64_t nest_i_21975 = 0; nest_i_21975 < color_bound_20172; nest_i_21975++) {
                ((bool *) mem_21857)[nest_i_21975] = 0;
            }
            // src/bgpc_vv_coloring.fut:46:5-54:19
            for (int64_t i_21360 = 0; i_21360 < j_m_i_21346; i_21360++) {
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t index_primexp_21362 = seen_final_21339 + i_21360;
                
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t v_21363 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21362];
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool x_21364 = sle64((int64_t) 0, v_21363);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool y_21365 = slt64(v_21363, csr_cols_from_pattern_res_20127);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool bounds_check_21366 = x_21364 && y_21365;
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool index_certs_21367;
                
                if (!bounds_check_21366) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_21363, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20127, "].", "-> #0  src/bgpc_vv_coloring.fut:49:26-37\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:26-37
                
                int64_t seen_acczq_21368 = ((int64_t *) ext_mem_21845.mem)[v_21363];
                
                // src/bgpc_vv_coloring.fut:49:50-52
                
                int64_t seen_acczq_21369 = add64((int64_t) 1, v_21363);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool x_21370 = sle64((int64_t) 0, seen_acczq_21369);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool y_21371 = slt64(seen_acczq_21369, csr_cols_from_pattern_res_20127);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool bounds_check_21372 = x_21370 && y_21371;
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool index_certs_21373;
                
                if (!bounds_check_21372) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_21369, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20127, "].", "-> #0  src/bgpc_vv_coloring.fut:49:40-53\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:40-53
                
                int64_t seen_acczq_21374 = ((int64_t *) ext_mem_21845.mem)[seen_acczq_21369];
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t j_m_i_21375 = sub64(seen_acczq_21374, seen_acczq_21368);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool empty_slice_21376 = j_m_i_21375 == (int64_t) 0;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t m_21377 = sub64(j_m_i_21375, (int64_t) 1);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t i_p_m_t_s_21378 = add64(seen_acczq_21368, m_21377);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_leq_i_p_m_t_s_21379 = sle64((int64_t) 0, i_p_m_t_s_21378);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_p_m_t_s_leq_w_21380 = slt64(i_p_m_t_s_21378, csr_cols_from_pattern_res_20128);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_lte_i_21381 = sle64((int64_t) 0, seen_acczq_21368);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_lte_j_21382 = sle64(seen_acczq_21368, seen_acczq_21374);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_21383 = i_p_m_t_s_leq_w_21380 && zzero_lte_i_21381;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_21384 = zzero_leq_i_p_m_t_s_21379 && y_21383;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool forwards_ok_21385 = i_lte_j_21382 && y_21384;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool ok_or_empty_21386 = empty_slice_21376 || forwards_ok_21385;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool index_certs_21387;
                
                if (!ok_or_empty_21386) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_21368, ":", (long long) seen_acczq_21374, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20128, "].", "-> #0  src/bgpc_vv_coloring.fut:48:9-53:33\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:48:9-53:33
                for (int64_t i_21389 = 0; i_21389 < j_m_i_21375; i_21389++) {
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t index_primexp_21391 = seen_acczq_21368 + i_21389;
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t u_21392 = ((int64_t *) ext_mem_21844.mem)[index_primexp_21391];
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool x_21393 = sle64((int64_t) 0, u_21392);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool y_21394 = slt64(u_21392, (int64_t) 2);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool bounds_check_21395 = x_21393 && y_21394;
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool index_certs_21396;
                    
                    if (!bounds_check_21395) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_21392, "] out of bounds for array of shape [", (long long) (int64_t) 2, "].", "-> #0  src/bgpc_vv_coloring.fut:50:20-29\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    int64_t cu_21397 = ((int64_t *) mem_21841)[u_21392];
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_21398 = u_21392 == eta_p_21333;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_21399 = !cond_21398;
                    
                    // src/bgpc_vv_coloring.fut:51:30-37
                    
                    bool cond_t_res_21400 = sle64((int64_t) 0, cu_21397);
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    bool x_21401 = cond_21399 && cond_t_res_21400;
                    
                    // src/bgpc_vv_coloring.fut:51:44-67
                    
                    bool cond_t_res_21402 = slt64(cu_21397, color_bound_20172);
                    
                    // src/bgpc_vv_coloring.fut:94:14-103:69
                    
                    bool x_21403 = x_21401 && cond_t_res_21402;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    if (x_21403) {
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool bounds_check_21636 = cond_t_res_21400 && cond_t_res_21402;
                        
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool index_certs_21637;
                        
                        if (!bounds_check_21636) {
                            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) cu_21397, "] out of bounds for array of shape [", (long long) color_bound_20172, "].", "-> #0  src/bgpc_vv_coloring.fut:52:19-50\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                            err = FUTHARK_PROGRAM_ERROR;
                            goto cleanup;
                        }
                        // src/bgpc_vv_coloring.fut:52:19-50
                        ((bool *) mem_21857)[cu_21397] = 1;
                    }
                }
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool loop_cond_21408;
            
            if (cond_20173) {
                // src/bgpc_vv_coloring.fut:63:30-37
                
                bool loop_cond_t_res_21639 = ((bool *) mem_21857)[(int64_t) 0];
                
                loop_cond_21408 = loop_cond_t_res_21639;
            } else {
                loop_cond_21408 = 0;
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool c_final_21410;
            int64_t c_final_21411;
            bool loop_while_21412;
            int64_t c_21413;
            
            loop_while_21412 = loop_cond_21408;
            c_21413 = (int64_t) 0;
            while (loop_while_21412) {
                // src/bgpc_vv_coloring.fut:64:9-15
                
                int64_t loopres_21414 = add64((int64_t) 1, c_21413);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool cond_21415 = slt64(loopres_21414, color_bound_20172);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool loop_cond_21416;
                
                if (cond_21415) {
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool x_21640 = sle64((int64_t) 0, loopres_21414);
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool bounds_check_21641 = cond_21415 && x_21640;
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool index_certs_21642;
                    
                    if (!bounds_check_21641) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) loopres_21414, "] out of bounds for array of shape [", (long long) color_bound_20172, "].", "-> #0  src/bgpc_vv_coloring.fut:63:30-37\n   #1  src/bgpc_vv_coloring.fut:81:6-28\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool loop_cond_t_res_21643 = ((bool *) mem_21857)[loopres_21414];
                    
                    loop_cond_21416 = loop_cond_t_res_21643;
                } else {
                    loop_cond_21416 = 0;
                }
                
                bool loop_while_tmp_21978 = loop_cond_21416;
                int64_t c_tmp_21979 = loopres_21414;
                
                loop_while_21412 = loop_while_tmp_21978;
                c_21413 = c_tmp_21979;
            }
            c_final_21410 = loop_while_21412;
            c_final_21411 = c_21413;
            // src/bgpc_vv_coloring.fut:106:25-32
            
            int64_t max_res_20267 = smax64(c_final_21411, redout_21695);
            
            ((int64_t *) mem_21853)[i_21697] = c_final_21411;
            
            int64_t redout_tmp_21973 = max_res_20267;
            
            redout_21695 = redout_tmp_21973;
        }
        defunc_0_reduce_res_21672 = redout_21695;
        // src/bgpc_vv_coloring.fut:113:17-43
        
        bool acc_cert_20273;
        
        // src/bgpc_vv_coloring.fut:113:17-43
        for (int64_t i_21700 = 0; i_21700 < loop_dz2081Uz2083U_20168; i_21700++) {
            int64_t v_20277 = ((int64_t *) mem_param_21851.mem)[i_21700];
            int64_t v_20278 = ((int64_t *) mem_21853)[i_21700];
            
            // src/bgpc_vv_coloring.fut:113:17-43
            // UpdateAcc
            if (sle64((int64_t) 0, v_20277) && slt64(v_20277, (int64_t) 2)) {
                ((int64_t *) mem_21841)[v_20277] = v_20278;
            }
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21874_cached_sizze_22077 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21874, &mem_21874_cached_sizze_22077, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21876_cached_sizze_22078 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21876, &mem_21876_cached_sizze_22078, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21877_cached_sizze_22079 < loop_dz2081Uz2083U_20168) {
            err = lexical_realloc(ctx, &mem_21877, &mem_21877_cached_sizze_22079, loop_dz2081Uz2083U_20168);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t discard_21712;
        int64_t scanacc_21704 = (int64_t) 0;
        
        for (int64_t i_21708 = 0; i_21708 < loop_dz2081Uz2083U_20168; i_21708++) {
            int64_t eta_p_21257 = ((int64_t *) mem_param_21851.mem)[i_21708];
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool x_21258 = sle64((int64_t) 0, eta_p_21257);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool y_21259 = slt64(eta_p_21257, csr_bipartite_from_pattern_res_20122);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool bounds_check_21260 = x_21258 && y_21259;
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool index_certs_21261;
            
            if (!bounds_check_21260) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_21257, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/bgpc_vv_coloring.fut:126:15-27\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:127:26-28
            
            int64_t k_end_21263 = add64((int64_t) 1, eta_p_21257);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool x_21264 = sle64((int64_t) 0, k_end_21263);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool y_21265 = slt64(k_end_21263, csr_bipartite_from_pattern_res_20122);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool bounds_check_21266 = x_21264 && y_21265;
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool index_certs_21267;
            
            if (!bounds_check_21266) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_end_21263, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/bgpc_vv_coloring.fut:127:15-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:126:15-27
            
            int64_t k0_21262 = ((int64_t *) ext_mem_21829.mem)[eta_p_21257];
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            int64_t k_end_21268 = ((int64_t *) ext_mem_21829.mem)[k_end_21263];
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool cond_21269 = slt64(k0_21262, k_end_21268);
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool loses_conflict_vertex_res_21270;
            bool loses_conflict_vertex_res_21271;
            int64_t loses_conflict_vertex_res_21272;
            bool loop_while_21273;
            bool lost_21274;
            int64_t k_21275;
            
            loop_while_21273 = cond_21269;
            lost_21274 = 0;
            k_21275 = k0_21262;
            while (loop_while_21273) {
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool x_21276 = sle64((int64_t) 0, k_21275);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool y_21277 = slt64(k_21275, csr_bipartite_from_pattern_res_20123);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool bounds_check_21278 = x_21276 && y_21277;
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool index_certs_21279;
                
                if (!bounds_check_21278) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_21275, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20123, "].", "-> #0  src/bgpc_vv_coloring.fut:132:15-26\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:132:15-26
                
                int64_t v_21280 = ((int64_t *) ext_mem_21828.mem)[k_21275];
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool x_21281 = sle64((int64_t) 0, v_21280);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool y_21282 = slt64(v_21280, csr_cols_from_pattern_res_20127);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool bounds_check_21283 = x_21281 && y_21282;
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool index_certs_21284;
                
                if (!bounds_check_21283) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_21280, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20127, "].", "-> #0  src/bgpc_vv_coloring.fut:134:19-30\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:135:29-31
                
                int64_t t_end_21286 = add64((int64_t) 1, v_21280);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool x_21287 = sle64((int64_t) 0, t_end_21286);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool y_21288 = slt64(t_end_21286, csr_cols_from_pattern_res_20127);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool bounds_check_21289 = x_21287 && y_21288;
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool index_certs_21290;
                
                if (!bounds_check_21289) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_end_21286, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20127, "].", "-> #0  src/bgpc_vv_coloring.fut:135:19-32\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:134:19-30
                
                int64_t t0_21285 = ((int64_t *) ext_mem_21845.mem)[v_21280];
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                int64_t t_end_21291 = ((int64_t *) ext_mem_21845.mem)[t_end_21286];
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool cond_21292 = slt64(t0_21285, t_end_21291);
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool loopres_21293;
                bool loopres_21294;
                int64_t loopres_21295;
                bool loop_while_21296;
                bool lost_in_net_21297;
                int64_t t_21298;
                
                loop_while_21296 = cond_21292;
                lost_in_net_21297 = 0;
                t_21298 = t0_21285;
                while (loop_while_21296) {
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool x_21299 = sle64((int64_t) 0, t_21298);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool y_21300 = slt64(t_21298, csr_cols_from_pattern_res_20128);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool bounds_check_21301 = x_21299 && y_21300;
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool index_certs_21302;
                    
                    if (!bounds_check_21301) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_21298, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20128, "].", "-> #0  src/bgpc_vv_coloring.fut:140:19-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    int64_t u_21303 = ((int64_t *) ext_mem_21844.mem)[t_21298];
                    
                    // src/bgpc_vv_coloring.fut:142:13-71
                    
                    bool lost_in_netzq_21304;
                    
                    if (lost_in_net_21297) {
                        lost_in_netzq_21304 = 1;
                    } else {
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_21305 = u_21303 == eta_p_21257;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_21306 = !cond_21305;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_21307;
                        
                        if (cond_21306) {
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool x_21645 = sle64((int64_t) 0, u_21303);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool y_21646 = slt64(u_21303, (int64_t) 2);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool bounds_check_21647 = x_21645 && y_21646;
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool index_certs_21648;
                            
                            if (!bounds_check_21647) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_21303, "] out of bounds for array of shape [", (long long) (int64_t) 2, "].", "-> #0  src/bgpc_vv_coloring.fut:142:39-48\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool y_21650 = slt64(eta_p_21257, (int64_t) 2);
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool bounds_check_21651 = x_21258 && y_21650;
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool index_certs_21652;
                            
                            if (!bounds_check_21651) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_21257, "] out of bounds for array of shape [", (long long) (int64_t) 2, "].", "-> #0  src/bgpc_vv_coloring.fut:142:52-61\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            int64_t zeze_lhs_21649 = ((int64_t *) mem_21841)[u_21303];
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            int64_t zeze_rhs_21653 = ((int64_t *) mem_21841)[eta_p_21257];
                            
                            // src/bgpc_vv_coloring.fut:142:49-61
                            
                            bool cond_t_res_21654 = zeze_lhs_21649 == zeze_rhs_21653;
                            
                            cond_21307 = cond_t_res_21654;
                        } else {
                            cond_21307 = 0;
                        }
                        // src/bgpc_vv_coloring.fut:142:67-70
                        
                        bool lost_in_netzq_f_res_t_res_21318 = slt64(u_21303, eta_p_21257);
                        
                        // src/bgpc_vv_coloring.fut:160:14-168:57
                        
                        bool x_21319 = cond_21307 && lost_in_netzq_f_res_t_res_21318;
                        
                        lost_in_netzq_21304 = x_21319;
                    }
                    // src/bgpc_vv_coloring.fut:143:31-37
                    
                    int64_t tmp_21320 = add64((int64_t) 1, t_21298);
                    
                    // src/bgpc_vv_coloring.fut:139:15-43
                    
                    bool cond_21321 = slt64(tmp_21320, t_end_21291);
                    
                    // src/bgpc_vv_coloring.fut:139:28-43
                    
                    bool not_res_21322 = !lost_in_netzq_21304;
                    
                    // src/bgpc_vv_coloring.fut:166:5-169:10
                    
                    bool x_21323 = cond_21321 && not_res_21322;
                    bool loop_while_tmp_21988 = x_21323;
                    bool lost_in_net_tmp_21989 = lost_in_netzq_21304;
                    int64_t t_tmp_21990 = tmp_21320;
                    
                    loop_while_21296 = loop_while_tmp_21988;
                    lost_in_net_21297 = lost_in_net_tmp_21989;
                    t_21298 = t_tmp_21990;
                }
                loopres_21293 = loop_while_21296;
                loopres_21294 = lost_in_net_21297;
                loopres_21295 = t_21298;
                // src/bgpc_vv_coloring.fut:160:14-168:57
                
                bool lostzq_21324 = lost_21274 || loopres_21294;
                
                // src/bgpc_vv_coloring.fut:146:20-26
                
                int64_t tmp_21325 = add64((int64_t) 1, k_21275);
                
                // src/bgpc_vv_coloring.fut:131:11-32
                
                bool cond_21326 = slt64(tmp_21325, k_end_21268);
                
                // src/bgpc_vv_coloring.fut:131:24-32
                
                bool not_res_21327 = !lostzq_21324;
                
                // src/bgpc_vv_coloring.fut:166:5-169:10
                
                bool x_21328 = cond_21326 && not_res_21327;
                bool loop_while_tmp_21985 = x_21328;
                bool lost_tmp_21986 = lostzq_21324;
                int64_t k_tmp_21987 = tmp_21325;
                
                loop_while_21273 = loop_while_tmp_21985;
                lost_21274 = lost_tmp_21986;
                k_21275 = k_tmp_21987;
            }
            loses_conflict_vertex_res_21270 = loop_while_21273;
            loses_conflict_vertex_res_21271 = lost_21274;
            loses_conflict_vertex_res_21272 = k_21275;
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_f_res_21330 = btoi_bool_i64(loses_conflict_vertex_res_21271);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_op_res_20379 = add64(defunc_0_f_res_21330, scanacc_21704);
            
            ((int64_t *) mem_21874)[i_21708] = defunc_0_op_res_20379;
            ((int64_t *) mem_21876)[i_21708] = defunc_0_f_res_21330;
            ((bool *) mem_21877)[i_21708] = loses_conflict_vertex_res_21271;
            
            int64_t scanacc_tmp_21981 = defunc_0_op_res_20379;
            
            scanacc_21704 = scanacc_tmp_21981;
        }
        discard_21712 = scanacc_21704;
        // src/bgpc_vv_coloring.fut:23:25-28
        
        int64_t tmp_20363 = sub64(loop_dz2081Uz2083U_20168, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool y_20365 = slt64(tmp_20363, loop_dz2081Uz2083U_20168);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool x_20364 = sle64((int64_t) 0, tmp_20363);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool bounds_check_20366 = x_20364 && y_20365;
        
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        bool cond_20361 = loop_dz2081Uz2083U_20168 == (int64_t) 0;
        
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool protect_assert_disj_20367 = cond_20361 || bounds_check_20366;
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool index_certs_20368;
        
        if (!protect_assert_disj_20367) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_20363, "] out of bounds for array of shape [", (long long) loop_dz2081Uz2083U_20168, "].", "-> #0  src/bgpc_vv_coloring.fut:23:10-29\n   #1  src/bgpc_vv_coloring.fut:171:23-55\n   #2  src/bgpc_vv_coloring.fut:186:14-205:56\n   #3  src/bgpc_vv_coloring.fut:220:14-223:64\n   #4  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #5  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #6  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #7  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool x_20362 = !cond_20361;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_f_res_20380;
        
        if (x_20362) {
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t x_21655 = ((int64_t *) mem_21874)[tmp_20363];
            
            m_f_res_20380 = x_21655;
        } else {
            m_f_res_20380 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_20382;
        
        if (cond_20361) {
            m_20382 = (int64_t) 0;
        } else {
            m_20382 = m_f_res_20380;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t bytes_21896 = (int64_t) 8 * m_20382;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21897_cached_sizze_22080 < bytes_21896) {
            err = lexical_realloc(ctx, &mem_21897, &mem_21897_cached_sizze_22080, bytes_21896);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21899_cached_sizze_22081 < bytes_21896) {
            err = lexical_realloc(ctx, &mem_21899, &mem_21899_cached_sizze_22081, bytes_21896);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:18:24-27:44
        if (mem_21901_cached_sizze_22082 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21901, &mem_21901_cached_sizze_22082, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21909_cached_sizze_22083 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21909, &mem_21909_cached_sizze_22083, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        bool acc_cert_21130;
        bool acc_cert_21131;
        
        // src/bgpc_vv_coloring.fut:18:24-27:44
        
        int64_t inpacc_21663;
        int64_t inpacc_21194 = (int64_t) 0;
        
        for (int64_t i_21739 = 0; i_21739 < loop_dz2081Uz2083U_20168; i_21739++) {
            bool eta_p_21778 = ((bool *) mem_21877)[i_21739];
            
            // src/bgpc_vv_coloring.fut:18:28-39
            
            int64_t bool_to_i64_res_21779 = btoi_bool_i64(eta_p_21778);
            int64_t eta_p_21791 = ((int64_t *) mem_21876)[i_21739];
            int64_t eta_p_21792 = ((int64_t *) mem_21874)[i_21739];
            int64_t v_21795 = ((int64_t *) mem_param_21851.mem)[i_21739];
            
            // src/bgpc_vv_coloring.fut:19:29-32
            
            int64_t defunc_0_op_res_21796 = add64(inpacc_21194, bool_to_i64_res_21779);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            bool cond_21797 = eta_p_21791 == (int64_t) 1;
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t lifted_lambda_res_21798;
            
            if (cond_21797) {
                // src/bgpc_vv_coloring.fut:27:5-44
                
                int64_t lifted_lambda_res_t_res_21799 = sub64(eta_p_21792, (int64_t) 1);
                
                lifted_lambda_res_21798 = lifted_lambda_res_t_res_21799;
            } else {
                lifted_lambda_res_21798 = (int64_t) -1;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_21798) && slt64(lifted_lambda_res_21798, m_20382)) {
                ((int64_t *) mem_21899)[lifted_lambda_res_21798] = v_21795;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_21798) && slt64(lifted_lambda_res_21798, m_20382)) {
                ((int64_t *) mem_21897)[lifted_lambda_res_21798] = defunc_0_op_res_21796;
            }
            ((int64_t *) mem_21901)[i_21739] = defunc_0_op_res_21796;
            
            int64_t inpacc_tmp_21991 = defunc_0_op_res_21796;
            
            inpacc_21194 = inpacc_tmp_21991;
        }
        inpacc_21663 = inpacc_21194;
        // src/bgpc_vv_coloring.fut:27:5-44
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_21909, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_21901, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {loop_dz2081Uz2083U_20168});
        // src/bgpc_vv_coloring.fut:23:10-29
        
        int64_t out_szz_f_res_20369;
        
        if (x_20362) {
            // src/bgpc_vv_coloring.fut:171:23-55
            
            int64_t x_21669 = ((int64_t *) mem_21909)[tmp_20363];
            
            out_szz_f_res_20369 = x_21669;
        } else {
            out_szz_f_res_20369 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        int64_t out_szz_20371;
        
        if (cond_20361) {
            out_szz_20371 = (int64_t) 0;
        } else {
            out_szz_20371 = out_szz_f_res_20369;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        
        int64_t bytes_21910 = (int64_t) 8 * out_szz_20371;
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        bool cond_20268 = slt64(defunc_0_reduce_res_21672, (int64_t) 0);
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        int64_t next_color_bound_20269;
        
        if (cond_20268) {
            next_color_bound_20269 = color_bound_20172;
        } else {
            // src/bgpc_vv_coloring.fut:110:40-46
            
            int64_t max_arg1_20270 = add64((int64_t) 2, defunc_0_reduce_res_21672);
            
            // src/bgpc_vv_coloring.fut:110:10-46
            
            int64_t max_res_20271 = smax64(color_bound_20172, max_arg1_20270);
            
            next_color_bound_20269 = max_res_20271;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        if (memblock_alloc(ctx, &mem_21911, bytes_21910, "mem_21911")) {
            err = 1;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        for (int64_t nest_i_21995 = 0; nest_i_21995 < out_szz_20371; nest_i_21995++) {
            ((int64_t *) mem_21911.mem)[nest_i_21995] = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:33:6-27
        
        bool acc_cert_20903;
        
        // src/bgpc_vv_coloring.fut:29:21-33:27
        for (int64_t i_21744 = 0; i_21744 < m_20382; i_21744++) {
            int64_t eta_p_20915 = ((int64_t *) mem_21897)[i_21744];
            int64_t v_20917 = ((int64_t *) mem_21899)[i_21744];
            
            // src/bgpc_vv_coloring.fut:29:44-50
            
            int64_t lifted_lambda_res_20918 = sub64(eta_p_20915, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:33:6-27
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_20918) && slt64(lifted_lambda_res_20918, out_szz_20371)) {
                ((int64_t *) mem_21911.mem)[lifted_lambda_res_20918] = v_20917;
            }
        }
        // src/bgpc_vv_coloring.fut:175:17-48
        
        bool acc_cert_20419;
        
        // src/bgpc_vv_coloring.fut:175:17-48
        for (int64_t i_21746 = 0; i_21746 < out_szz_20371; i_21746++) {
            int64_t v_20423 = ((int64_t *) mem_21911.mem)[i_21746];
            
            // src/bgpc_vv_coloring.fut:175:17-48
            // UpdateAcc
            if (sle64((int64_t) 0, v_20423) && slt64(v_20423, (int64_t) 2)) {
                ((int64_t *) mem_21841)[v_20423] = (int64_t) -1;
            }
        }
        // src/bgpc_vv_coloring.fut:198:20-23
        
        bool loop_cond_20425 = slt64((int64_t) 0, out_szz_20371);
        
        if (memblock_set(ctx, &mem_param_tmp_21967, &mem_21911, "mem_21911") != 0)
            return 1;
        
        int64_t loop_dz2081Uz2083U_tmp_21968 = out_szz_20371;
        bool loop_while_tmp_21969 = loop_cond_20425;
        int64_t color_bound_tmp_21972 = next_color_bound_20269;
        
        if (memblock_set(ctx, &mem_param_21851, &mem_param_tmp_21967, "mem_param_tmp_21967") != 0)
            return 1;
        loop_dz2081Uz2083U_20168 = loop_dz2081Uz2083U_tmp_21968;
        loop_while_20169 = loop_while_tmp_21969;
        color_bound_20172 = color_bound_tmp_21972;
    }
    if (memblock_set(ctx, &ext_mem_21916, &mem_param_21851, "mem_param_21851") != 0)
        return 1;
    vv_color_side_order_res_20163 = loop_dz2081Uz2083U_20168;
    vv_color_side_order_res_20164 = loop_while_20169;
    vv_color_side_order_res_20167 = color_bound_20172;
    if (memblock_unref(ctx, &mem_21840, "mem_21840") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21844, "ext_mem_21844") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21845, "ext_mem_21845") != 0)
        return 1;
    // src/sparse_jacobian_vjp.fut:78:8-87:44
    
    int64_t x_21683;
    int64_t redout_21747 = (int64_t) 0;
    
    for (int64_t i_21748 = 0; i_21748 < (int64_t) 2; i_21748++) {
        int64_t x_20428 = ((int64_t *) mem_21841)[i_21748];
        
        // src/sparse_jacobian_vjp.fut:14:22-29
        
        int64_t max_res_20431 = smax64(x_20428, redout_21747);
        int64_t redout_tmp_21998 = max_res_20431;
        
        redout_21747 = redout_tmp_21998;
    }
    x_21683 = redout_21747;
    // src/sparse_jacobian_vjp.fut:14:13-45
    
    int64_t num_colors_of_res_f_res_20432 = add64((int64_t) 1, x_21683);
    
    // src/sparse_jacobian_vjp.fut:29:11-18
    
    bool bounds_invalid_upwards_20434 = slt64(num_colors_of_res_f_res_20432, (int64_t) 0);
    
    // src/sparse_jacobian_vjp.fut:29:11-18
    
    bool valid_20435 = !bounds_invalid_upwards_20434;
    
    // src/sparse_jacobian_vjp.fut:29:11-18
    
    bool range_valid_c_20436;
    
    if (!valid_20435) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Range ", (long long) (int64_t) 0, "..", (long long) (int64_t) 1, "..<", (long long) num_colors_of_res_f_res_20432, " is invalid.", "-> #0  src/sparse_jacobian_vjp.fut:29:11-18\n   #1  src/sparse_jacobian_vjp.fut:78:8-87:44\n   #2  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #3  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #4  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:21:8-29:18
    
    int64_t bytes_21919 = (int64_t) 32 * num_colors_of_res_f_res_20432;
    
    // src/sparse_jacobian_vjp.fut:42:15-35
    if (mem_21831_cached_sizze_22072 < bytes_21830) {
        err = lexical_realloc(ctx, &mem_21831, &mem_21831_cached_sizze_22072, bytes_21830);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:42:15-35
    for (int64_t nest_i_21999 = 0; nest_i_21999 < csr_bipartite_from_pattern_res_20123; nest_i_21999++) {
        ((double *) mem_21831)[nest_i_21999] = 0.0;
    }
    // src/sparse_jacobian_vjp.fut:21:8-29:18
    if (mem_21920_cached_sizze_22084 < bytes_21919) {
        err = lexical_realloc(ctx, &mem_21920, &mem_21920_cached_sizze_22084, bytes_21919);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:17:3-61
    if (mem_21925_cached_sizze_22085 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_21925, &mem_21925_cached_sizze_22085, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:21:8-29:18
    for (int64_t i_21755 = 0; i_21755 < num_colors_of_res_f_res_20432; i_21755++) {
        // src/sparse_jacobian_vjp.fut:17:3-61
        for (int64_t i_21751 = 0; i_21751 < (int64_t) 2; i_21751++) {
            int64_t eta_p_20445 = ((int64_t *) mem_21841)[i_21751];
            
            // src/sparse_jacobian_vjp.fut:17:15-49
            
            bool cond_20446 = eta_p_20445 == i_21755;
            
            // src/sparse_jacobian_vjp.fut:17:15-49
            
            double lifted_lambda_res_20447;
            
            if (cond_20446) {
                lifted_lambda_res_20447 = 1.0;
            } else {
                lifted_lambda_res_20447 = 0.0;
            }
            ((double *) mem_21925)[i_21751] = lifted_lambda_res_20447;
        }
        
        double elem_adj_21572 = ((double *) mem_21925)[(int64_t) 0];
        double elem_adj_21573 = ((double *) mem_21925)[(int64_t) 1];
        double binop_y_adj_21575 = 7.0 * elem_adj_21573;
        
        for (int64_t nest_i_22002 = 0; nest_i_22002 < (int64_t) 4; nest_i_22002++) {
            ((double *) mem_21920)[i_21755 * (int64_t) 4 + nest_i_22002] = 0.0;
        }
        ((double *) mem_21920)[i_21755 * (int64_t) 4] = elem_adj_21572;
        ((double *) mem_21920)[i_21755 * (int64_t) 4 + (int64_t) 1] = elem_adj_21572;
        ((double *) mem_21920)[i_21755 * (int64_t) 4 + (int64_t) 2] = binop_y_adj_21575;
    }
    // src/sparse_jacobian_vjp.fut:45:5-53:27
    
    bool compressed_to_csr_vals_res_20456;
    int64_t compressed_to_csr_vals_res_20458;
    bool loop_while_20459;
    int64_t i_20461;
    
    loop_while_20459 = 1;
    i_20461 = (int64_t) 0;
    while (loop_while_20459) {
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool x_20462 = sle64((int64_t) 0, i_20461);
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool y_20463 = slt64(i_20461, csr_bipartite_from_pattern_res_20122);
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool bounds_check_20464 = x_20462 && y_20463;
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool index_certs_20465;
        
        if (!bounds_check_20464) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_20461, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/sparse_jacobian_vjp.fut:47:15-26\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        int64_t s_20466 = ((int64_t *) ext_mem_21829.mem)[i_20461];
        
        // src/sparse_jacobian_vjp.fut:48:25-27
        
        int64_t e_20467 = add64((int64_t) 1, i_20461);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool x_20468 = sle64((int64_t) 0, e_20467);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool y_20469 = slt64(e_20467, csr_bipartite_from_pattern_res_20122);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool bounds_check_20470 = x_20468 && y_20469;
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool index_certs_20471;
        
        if (!bounds_check_20470) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_20467, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/sparse_jacobian_vjp.fut:48:15-28\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        int64_t e_20472 = ((int64_t *) ext_mem_21829.mem)[e_20467];
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t j_m_i_20473 = sub64(e_20472, s_20466);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool empty_slice_20474 = j_m_i_20473 == (int64_t) 0;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t m_20475 = sub64(j_m_i_20473, (int64_t) 1);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t i_p_m_t_s_20476 = add64(s_20466, m_20475);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_20477 = sle64((int64_t) 0, i_p_m_t_s_20476);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_20478 = slt64(i_p_m_t_s_20476, csr_bipartite_from_pattern_res_20123);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool zzero_lte_i_20479 = sle64((int64_t) 0, s_20466);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool i_lte_j_20480 = sle64(s_20466, e_20472);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool y_20481 = i_p_m_t_s_leq_w_20478 && zzero_lte_i_20479;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool y_20482 = zzero_leq_i_p_m_t_s_20477 && y_20481;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool forwards_ok_20483 = i_lte_j_20480 && y_20482;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool ok_or_empty_20484 = empty_slice_20474 || forwards_ok_20483;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool index_certs_20485;
        
        if (!ok_or_empty_20484) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_20466, ":", (long long) e_20472, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20123, "].", "-> #0  src/sparse_jacobian_vjp.fut:49:18-30\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool y_20487 = slt64(i_20461, (int64_t) 2);
        
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool bounds_check_20488 = x_20462 && y_20487;
        
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool index_certs_20489;
        
        if (!bounds_check_20488) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_20461, "] out of bounds for array of shape [", (long long) (int64_t) 2, "].", "-> #0  src/sparse_jacobian_vjp.fut:50:16-29\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        int64_t rc_20490 = ((int64_t *) mem_21841)[i_20461];
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool y_20492 = slt64(rc_20490, num_colors_of_res_f_res_20432);
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool x_20491 = sle64((int64_t) 0, rc_20490);
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool bounds_check_20493 = x_20491 && y_20492;
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool index_certs_20494;
        
        if (!bounds_check_20493) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) rc_20490, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_20432, "].", "-> #0  src/sparse_jacobian_vjp.fut:51:28-34\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:51:17-43
        for (int64_t i_21759 = 0; i_21759 < j_m_i_20473; i_21759++) {
            int64_t index_primexp_21771 = s_20466 + i_21759;
            int64_t eta_p_20496 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21771];
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool x_20497 = sle64((int64_t) 0, eta_p_20496);
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool y_20498 = slt64(eta_p_20496, (int64_t) 4);
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool bounds_check_20499 = x_20497 && y_20498;
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool index_certs_20500;
            
            if (!bounds_check_20499) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_20496, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/sparse_jacobian_vjp.fut:51:28-37\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_vjp.fut:51:17-43
            
            double lifted_lambda_res_20501 = ((double *) mem_21920)[rc_20490 * (int64_t) 4 + eta_p_20496];
            
            ((double *) mem_21831)[s_20466 + i_21759] = lifted_lambda_res_20501;
        }
        // src/sparse_jacobian_vjp.fut:46:13-16
        
        bool loop_cond_20503 = slt64(e_20467, (int64_t) 2);
        bool loop_while_tmp_22003 = loop_cond_20503;
        int64_t i_tmp_22005 = e_20467;
        
        loop_while_20459 = loop_while_tmp_22003;
        i_20461 = i_tmp_22005;
    }
    compressed_to_csr_vals_res_20456 = loop_while_20459;
    compressed_to_csr_vals_res_20458 = i_20461;
    // src/dense_jacobian.fut:5:3-21
    if (mem_21951_cached_sizze_22086 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_21951, &mem_21951_cached_sizze_22086, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    if (mem_21952_cached_sizze_22087 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_21952, &mem_21952_cached_sizze_22087, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:68:29-47
    if (mem_21953_cached_sizze_22088 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_21953, &mem_21953_cached_sizze_22088, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:7:19-9:39
    
    bool defunc_0_reduce_res_21685;
    bool redout_21765 = 1;
    
    for (int64_t i_21766 = 0; i_21766 < (int64_t) 2; i_21766++) {
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        bool y_20826 = slt64(i_21766, csr_bipartite_from_pattern_res_20122);
        
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        bool index_certs_20828;
        
        if (!y_20826) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_21766, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/sparse_jacobian_vjp.fut:64:17-28\n   #1  src/sparse_jacobian_vjp.fut:132:18-138:40\n   #2  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        int64_t s_20829 = ((int64_t *) ext_mem_21829.mem)[i_21766];
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool zzero_lte_i_20842 = sle64((int64_t) 0, s_20829);
        
        // src/sparse_jacobian_vjp.fut:65:27-29
        
        int64_t e_20830 = add64((int64_t) 1, i_21766);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool y_20832 = slt64(e_20830, csr_bipartite_from_pattern_res_20122);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool x_20831 = sle64((int64_t) 0, e_20830);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool bounds_check_20833 = x_20831 && y_20832;
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool index_certs_20834;
        
        if (!bounds_check_20833) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_20830, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/sparse_jacobian_vjp.fut:65:17-30\n   #1  src/sparse_jacobian_vjp.fut:132:18-138:40\n   #2  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        int64_t e_20835 = ((int64_t *) ext_mem_21829.mem)[e_20830];
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t j_m_i_20836 = sub64(e_20835, s_20829);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t m_20838 = sub64(j_m_i_20836, (int64_t) 1);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t i_p_m_t_s_20839 = add64(s_20829, m_20838);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool i_p_m_t_s_leq_w_20841 = slt64(i_p_m_t_s_20839, csr_bipartite_from_pattern_res_20123);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool y_20850 = i_p_m_t_s_leq_w_20841 && zzero_lte_i_20842;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool zzero_leq_i_p_m_t_s_20840 = sle64((int64_t) 0, i_p_m_t_s_20839);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool y_20851 = zzero_leq_i_p_m_t_s_20840 && y_20850;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool i_lte_j_20843 = sle64(s_20829, e_20835);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool forwards_ok_20852 = i_lte_j_20843 && y_20851;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool empty_slice_20837 = j_m_i_20836 == (int64_t) 0;
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool ok_or_empty_20853 = empty_slice_20837 || forwards_ok_20852;
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool index_certs_20854;
        
        if (!ok_or_empty_20853) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_20829, ":", (long long) e_20835, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20123, "].", "-> #0  src/sparse_jacobian_vjp.fut:67:20-29\n   #1  src/sparse_jacobian_vjp.fut:132:18-138:40\n   #2  test/test_sparse_jacobian_vjp.fut:40:15-57:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/dense_jacobian.fut:5:3-21
        for (int64_t nest_i_22008 = 0; nest_i_22008 < (int64_t) 2; nest_i_22008++) {
            ((double *) mem_21951)[nest_i_22008] = 0.0;
        }
        // src/dense_jacobian.fut:5:3-39
        ((double *) mem_21951)[i_21766] = 1.0;
        
        double elem_adj_21583 = ((double *) mem_21951)[(int64_t) 0];
        double elem_adj_21584 = ((double *) mem_21951)[(int64_t) 1];
        double binop_y_adj_21586 = 7.0 * elem_adj_21584;
        
        for (int64_t nest_i_22009 = 0; nest_i_22009 < (int64_t) 4; nest_i_22009++) {
            ((double *) mem_21952)[nest_i_22009] = 0.0;
        }
        ((double *) mem_21952)[(int64_t) 0] = elem_adj_21583;
        ((double *) mem_21952)[(int64_t) 1] = elem_adj_21583;
        ((double *) mem_21952)[(int64_t) 2] = binop_y_adj_21586;
        // src/sparse_jacobian_vjp.fut:68:29-47
        for (int64_t nest_i_22010 = 0; nest_i_22010 < (int64_t) 4; nest_i_22010++) {
            ((double *) mem_21953)[nest_i_22010] = 0.0;
        }
        // src/sparse_jacobian_vjp.fut:69:12-32
        
        bool acc_cert_20858;
        
        // src/sparse_jacobian_vjp.fut:69:12-32
        for (int64_t i_21762 = 0; i_21762 < j_m_i_20836; i_21762++) {
            int64_t index_primexp_21768 = s_20829 + i_21762;
            int64_t v_20862 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21768];
            double v_20863 = ((double *) mem_21831)[index_primexp_21768];
            
            // src/sparse_jacobian_vjp.fut:69:12-32
            // UpdateAcc
            if (sle64((int64_t) 0, v_20862) && slt64(v_20862, (int64_t) 4)) {
                ((double *) mem_21953)[v_20862] = v_20863;
            }
        }
        // test/test_sparse_jacobian_vjp.fut:8:5-66
        
        bool defunc_0_reduce_res_21681;
        bool redout_21763 = 1;
        
        for (int64_t i_21764 = 0; i_21764 < (int64_t) 4; i_21764++) {
            bool eta_p_20886 = ((bool *) mem_21812.mem)[i_21766 * (int64_t) 4 + i_21764];
            double eta_p_20887 = ((double *) mem_21952)[i_21764];
            double eta_p_20888 = ((double *) mem_21953)[i_21764];
            
            // test/test_sparse_jacobian_vjp.fut:13:25-48
            
            double lifted_lambda_res_20889;
            
            if (eta_p_20886) {
                lifted_lambda_res_20889 = eta_p_20887;
            } else {
                lifted_lambda_res_20889 = 0.0;
            }
            // test/test_sparse_jacobian_vjp.fut:8:48-51
            
            double abs_arg0_20891 = eta_p_20888 - lifted_lambda_res_20889;
            
            // test/test_sparse_jacobian_vjp.fut:8:37-51
            
            double abs_res_20892 = fabs64(abs_arg0_20891);
            
            // test/test_sparse_jacobian_vjp.fut:8:53-59
            
            bool lifted_lambda_res_20893 = abs_res_20892 <= 1.0e-9;
            
            // test/test_sparse_jacobian_vjp.fut:8:5-66
            
            bool x_20876 = lifted_lambda_res_20893 && redout_21763;
            bool redout_tmp_22012 = x_20876;
            
            redout_21763 = redout_tmp_22012;
        }
        defunc_0_reduce_res_21681 = redout_21763;
        // test/test_sparse_jacobian_vjp.fut:9:6-39
        
        bool x_20623 = defunc_0_reduce_res_21681 && redout_21765;
        bool redout_tmp_22007 = x_20623;
        
        redout_21765 = redout_tmp_22007;
    }
    defunc_0_reduce_res_21685 = redout_21765;
    if (memblock_unref(ctx, &ext_mem_21828, "ext_mem_21828") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21829, "ext_mem_21829") != 0)
        return 1;
    prim_out_21962 = defunc_0_reduce_res_21685;
    *out_prim_out_22071 = prim_out_21962;
    
  cleanup:
    {
        free(mem_21831);
        free(mem_21832);
        free(mem_21841);
        free(mem_21853);
        free(mem_21857);
        free(mem_21874);
        free(mem_21876);
        free(mem_21877);
        free(mem_21897);
        free(mem_21899);
        free(mem_21901);
        free(mem_21909);
        free(mem_21920);
        free(mem_21925);
        free(mem_21951);
        free(mem_21952);
        free(mem_21953);
        if (memblock_unref(ctx, &mem_param_tmp_21967, "mem_param_tmp_21967") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21911, "mem_21911") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_21851, "mem_param_21851") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21916, "ext_mem_21916") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21844, "ext_mem_21844") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21845, "ext_mem_21845") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21842, "mem_21842") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21840, "mem_21840") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21828, "ext_mem_21828") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21829, "ext_mem_21829") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_vjp_ex4_matches_dense(struct futhark_context *ctx, bool *out_prim_out_22089, struct memblock x_mem_21827)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_21831_cached_sizze_22090 = 0;
    unsigned char *mem_21831 = NULL;
    int64_t mem_21832_cached_sizze_22091 = 0;
    unsigned char *mem_21832 = NULL;
    int64_t mem_21841_cached_sizze_22092 = 0;
    unsigned char *mem_21841 = NULL;
    int64_t mem_21853_cached_sizze_22093 = 0;
    unsigned char *mem_21853 = NULL;
    int64_t mem_21857_cached_sizze_22094 = 0;
    unsigned char *mem_21857 = NULL;
    int64_t mem_21874_cached_sizze_22095 = 0;
    unsigned char *mem_21874 = NULL;
    int64_t mem_21876_cached_sizze_22096 = 0;
    unsigned char *mem_21876 = NULL;
    int64_t mem_21877_cached_sizze_22097 = 0;
    unsigned char *mem_21877 = NULL;
    int64_t mem_21897_cached_sizze_22098 = 0;
    unsigned char *mem_21897 = NULL;
    int64_t mem_21899_cached_sizze_22099 = 0;
    unsigned char *mem_21899 = NULL;
    int64_t mem_21901_cached_sizze_22100 = 0;
    unsigned char *mem_21901 = NULL;
    int64_t mem_21909_cached_sizze_22101 = 0;
    unsigned char *mem_21909 = NULL;
    int64_t mem_21920_cached_sizze_22102 = 0;
    unsigned char *mem_21920 = NULL;
    int64_t mem_21925_cached_sizze_22103 = 0;
    unsigned char *mem_21925 = NULL;
    int64_t mem_21951_cached_sizze_22104 = 0;
    unsigned char *mem_21951 = NULL;
    int64_t mem_21952_cached_sizze_22105 = 0;
    unsigned char *mem_21952 = NULL;
    int64_t mem_21953_cached_sizze_22106 = 0;
    unsigned char *mem_21953 = NULL;
    struct memblock mem_param_tmp_21967;
    
    mem_param_tmp_21967.references = NULL;
    
    struct memblock mem_21911;
    
    mem_21911.references = NULL;
    
    struct memblock mem_param_21851;
    
    mem_param_21851.references = NULL;
    
    struct memblock ext_mem_21916;
    
    ext_mem_21916.references = NULL;
    
    struct memblock ext_mem_21844;
    
    ext_mem_21844.references = NULL;
    
    struct memblock ext_mem_21845;
    
    ext_mem_21845.references = NULL;
    
    struct memblock mem_21842;
    
    mem_21842.references = NULL;
    
    struct memblock mem_21840;
    
    mem_21840.references = NULL;
    
    struct memblock ext_mem_21828;
    
    ext_mem_21828.references = NULL;
    
    struct memblock ext_mem_21829;
    
    ext_mem_21829.references = NULL;
    
    struct memblock mem_21809 = ctx->constants->mem_21809;
    struct memblock mem_21812 = ctx->constants->mem_21812;
    struct memblock mem_21814 = ctx->constants->mem_21814;
    struct memblock mem_21819 = ctx->constants->mem_21819;
    struct memblock mem_21820 = ctx->constants->mem_21820;
    struct memblock mem_21826 = ctx->constants->mem_21826;
    bool prim_out_21962;
    
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_20122;
    int64_t csr_bipartite_from_pattern_res_20123;
    
    if (futrts_csr_rows_from_pattern_7010(ctx, &ext_mem_21829, &ext_mem_21828, &csr_bipartite_from_pattern_res_20122, &csr_bipartite_from_pattern_res_20123, mem_21819, (int64_t) 4, (int64_t) 6) != 0) {
        err = 1;
        goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:42:15-35
    
    int64_t bytes_21830 = (int64_t) 8 * csr_bipartite_from_pattern_res_20123;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (mem_21832_cached_sizze_22091 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_21832, &mem_21832_cached_sizze_22091, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t discard_21691;
    int64_t scanacc_21687 = (int64_t) 0;
    
    for (int64_t i_21689 = 0; i_21689 < (int64_t) 4; i_21689++) {
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t defunc_0_op_res_20138 = add64((int64_t) 1, scanacc_21687);
        
        ((int64_t *) mem_21832)[i_21689] = defunc_0_op_res_20138;
        
        int64_t scanacc_tmp_21963 = defunc_0_op_res_20138;
        
        scanacc_21687 = scanacc_tmp_21963;
    }
    discard_21691 = scanacc_21687;
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t x_20148 = ((int64_t *) mem_21832)[(int64_t) 3];
    
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    bool loop_cond_20162 = slt64((int64_t) 0, x_20148);
    
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t bytes_21839 = (int64_t) 8 * x_20148;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (memblock_alloc(ctx, &mem_21840, bytes_21839, "mem_21840")) {
        err = 1;
        goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    bool acc_cert_20718;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    for (int64_t i_21693 = 0; i_21693 < (int64_t) 4; i_21693++) {
        int64_t eta_p_20730 = ((int64_t *) mem_21832)[i_21693];
        
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t lifted_lambda_res_t_res_20733 = sub64(eta_p_20730, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:194:20-45
        // UpdateAcc
        if (sle64((int64_t) 0, lifted_lambda_res_t_res_20733) && slt64(lifted_lambda_res_t_res_20733, x_20148)) {
            ((int64_t *) mem_21840.mem)[lifted_lambda_res_t_res_20733] = i_21693;
        }
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    if (mem_21841_cached_sizze_22092 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_21841, &mem_21841_cached_sizze_22092, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    for (int64_t nest_i_21966 = 0; nest_i_21966 < (int64_t) 4; nest_i_21966++) {
        ((int64_t *) mem_21841)[nest_i_21966] = (int64_t) -1;
    }
    // src/pattern_csr.fut:25:3-39
    if (memblock_alloc(ctx, &mem_21842, (int64_t) 24, "mem_21842")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:25:3-39
    // src/pattern_csr.fut:25:3-39
    lmad_copy_1b(ctx, 2, (uint8_t *) mem_21842.mem, (int64_t) 0, (int64_t []) {(int64_t) 4, (int64_t) 1}, (uint8_t *) mem_21819.mem, (int64_t) 0, (int64_t []) {(int64_t) 1, (int64_t) 6}, (int64_t []) {(int64_t) 6, (int64_t) 4});
    // src/pattern_csr.fut:25:3-39
    
    int64_t csr_cols_from_pattern_res_20127;
    int64_t csr_cols_from_pattern_res_20128;
    
    if (futrts_csr_rows_from_pattern_7010(ctx, &ext_mem_21845, &ext_mem_21844, &csr_cols_from_pattern_res_20127, &csr_cols_from_pattern_res_20128, mem_21842, (int64_t) 6, (int64_t) 4) != 0) {
        err = 1;
        goto cleanup;
    }
    if (memblock_unref(ctx, &mem_21842, "mem_21842") != 0)
        return 1;
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    int64_t vv_color_side_order_res_20163;
    bool vv_color_side_order_res_20164;
    int64_t vv_color_side_order_res_20167;
    int64_t loop_dz2081Uz2083U_20168;
    bool loop_while_20169;
    int64_t color_bound_20172;
    
    if (memblock_set(ctx, &mem_param_21851, &mem_21840, "mem_21840") != 0)
        return 1;
    loop_dz2081Uz2083U_20168 = x_20148;
    loop_while_20169 = loop_cond_20162;
    color_bound_20172 = (int64_t) 1;
    while (loop_while_20169) {
        // src/bgpc_vv_coloring.fut:62:5-64:15
        
        bool cond_20173 = slt64((int64_t) 0, color_bound_20172);
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t bytes_21852 = (int64_t) 8 * loop_dz2081Uz2083U_20168;
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        if (mem_21853_cached_sizze_22093 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21853, &mem_21853_cached_sizze_22093, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:77:24-51
        if (mem_21857_cached_sizze_22094 < color_bound_20172) {
            err = lexical_realloc(ctx, &mem_21857, &mem_21857_cached_sizze_22094, color_bound_20172);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t defunc_0_reduce_res_21672;
        int64_t redout_21695 = (int64_t) -1;
        
        for (int64_t i_21697 = 0; i_21697 < loop_dz2081Uz2083U_20168; i_21697++) {
            int64_t eta_p_21341 = ((int64_t *) mem_param_21851.mem)[i_21697];
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool x_21343 = sle64((int64_t) 0, eta_p_21341);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool y_21344 = slt64(eta_p_21341, csr_bipartite_from_pattern_res_20122);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool bounds_check_21345 = x_21343 && y_21344;
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool index_certs_21346;
            
            if (!bounds_check_21345) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_21341, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/bgpc_vv_coloring.fut:46:44-56\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:44-56
            
            int64_t seen_final_21347 = ((int64_t *) ext_mem_21829.mem)[eta_p_21341];
            
            // src/bgpc_vv_coloring.fut:46:70-72
            
            int64_t seen_final_21348 = add64((int64_t) 1, eta_p_21341);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool x_21349 = sle64((int64_t) 0, seen_final_21348);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool y_21350 = slt64(seen_final_21348, csr_bipartite_from_pattern_res_20122);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool bounds_check_21351 = x_21349 && y_21350;
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool index_certs_21352;
            
            if (!bounds_check_21351) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_21348, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/bgpc_vv_coloring.fut:46:59-73\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:59-73
            
            int64_t seen_final_21353 = ((int64_t *) ext_mem_21829.mem)[seen_final_21348];
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t j_m_i_21354 = sub64(seen_final_21353, seen_final_21347);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool empty_slice_21355 = j_m_i_21354 == (int64_t) 0;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t m_21356 = sub64(j_m_i_21354, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t i_p_m_t_s_21357 = add64(seen_final_21347, m_21356);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_leq_i_p_m_t_s_21358 = sle64((int64_t) 0, i_p_m_t_s_21357);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_p_m_t_s_leq_w_21359 = slt64(i_p_m_t_s_21357, csr_bipartite_from_pattern_res_20123);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_lte_i_21360 = sle64((int64_t) 0, seen_final_21347);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_lte_j_21361 = sle64(seen_final_21347, seen_final_21353);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_21362 = i_p_m_t_s_leq_w_21359 && zzero_lte_i_21360;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_21363 = zzero_leq_i_p_m_t_s_21358 && y_21362;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool forwards_ok_21364 = i_lte_j_21361 && y_21363;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool ok_or_empty_21365 = empty_slice_21355 || forwards_ok_21364;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool index_certs_21366;
            
            if (!ok_or_empty_21365) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_21347, ":", (long long) seen_final_21353, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20123, "].", "-> #0  src/bgpc_vv_coloring.fut:46:5-54:19\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:77:24-51
            for (int64_t nest_i_21975 = 0; nest_i_21975 < color_bound_20172; nest_i_21975++) {
                ((bool *) mem_21857)[nest_i_21975] = 0;
            }
            // src/bgpc_vv_coloring.fut:46:5-54:19
            for (int64_t i_21368 = 0; i_21368 < j_m_i_21354; i_21368++) {
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t index_primexp_21370 = seen_final_21347 + i_21368;
                
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t v_21371 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21370];
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool x_21372 = sle64((int64_t) 0, v_21371);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool y_21373 = slt64(v_21371, csr_cols_from_pattern_res_20127);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool bounds_check_21374 = x_21372 && y_21373;
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool index_certs_21375;
                
                if (!bounds_check_21374) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_21371, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20127, "].", "-> #0  src/bgpc_vv_coloring.fut:49:26-37\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:26-37
                
                int64_t seen_acczq_21376 = ((int64_t *) ext_mem_21845.mem)[v_21371];
                
                // src/bgpc_vv_coloring.fut:49:50-52
                
                int64_t seen_acczq_21377 = add64((int64_t) 1, v_21371);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool x_21378 = sle64((int64_t) 0, seen_acczq_21377);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool y_21379 = slt64(seen_acczq_21377, csr_cols_from_pattern_res_20127);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool bounds_check_21380 = x_21378 && y_21379;
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool index_certs_21381;
                
                if (!bounds_check_21380) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_21377, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20127, "].", "-> #0  src/bgpc_vv_coloring.fut:49:40-53\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:40-53
                
                int64_t seen_acczq_21382 = ((int64_t *) ext_mem_21845.mem)[seen_acczq_21377];
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t j_m_i_21383 = sub64(seen_acczq_21382, seen_acczq_21376);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool empty_slice_21384 = j_m_i_21383 == (int64_t) 0;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t m_21385 = sub64(j_m_i_21383, (int64_t) 1);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t i_p_m_t_s_21386 = add64(seen_acczq_21376, m_21385);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_leq_i_p_m_t_s_21387 = sle64((int64_t) 0, i_p_m_t_s_21386);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_p_m_t_s_leq_w_21388 = slt64(i_p_m_t_s_21386, csr_cols_from_pattern_res_20128);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_lte_i_21389 = sle64((int64_t) 0, seen_acczq_21376);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_lte_j_21390 = sle64(seen_acczq_21376, seen_acczq_21382);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_21391 = i_p_m_t_s_leq_w_21388 && zzero_lte_i_21389;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_21392 = zzero_leq_i_p_m_t_s_21387 && y_21391;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool forwards_ok_21393 = i_lte_j_21390 && y_21392;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool ok_or_empty_21394 = empty_slice_21384 || forwards_ok_21393;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool index_certs_21395;
                
                if (!ok_or_empty_21394) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_21376, ":", (long long) seen_acczq_21382, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20128, "].", "-> #0  src/bgpc_vv_coloring.fut:48:9-53:33\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:48:9-53:33
                for (int64_t i_21397 = 0; i_21397 < j_m_i_21383; i_21397++) {
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t index_primexp_21399 = seen_acczq_21376 + i_21397;
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t u_21400 = ((int64_t *) ext_mem_21844.mem)[index_primexp_21399];
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool x_21401 = sle64((int64_t) 0, u_21400);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool y_21402 = slt64(u_21400, (int64_t) 4);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool bounds_check_21403 = x_21401 && y_21402;
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool index_certs_21404;
                    
                    if (!bounds_check_21403) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_21400, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/bgpc_vv_coloring.fut:50:20-29\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    int64_t cu_21405 = ((int64_t *) mem_21841)[u_21400];
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_21406 = u_21400 == eta_p_21341;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_21407 = !cond_21406;
                    
                    // src/bgpc_vv_coloring.fut:51:30-37
                    
                    bool cond_t_res_21408 = sle64((int64_t) 0, cu_21405);
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    bool x_21409 = cond_21407 && cond_t_res_21408;
                    
                    // src/bgpc_vv_coloring.fut:51:44-67
                    
                    bool cond_t_res_21410 = slt64(cu_21405, color_bound_20172);
                    
                    // src/bgpc_vv_coloring.fut:94:14-103:69
                    
                    bool x_21411 = x_21409 && cond_t_res_21410;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    if (x_21411) {
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool bounds_check_21636 = cond_t_res_21408 && cond_t_res_21410;
                        
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool index_certs_21637;
                        
                        if (!bounds_check_21636) {
                            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) cu_21405, "] out of bounds for array of shape [", (long long) color_bound_20172, "].", "-> #0  src/bgpc_vv_coloring.fut:52:19-50\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                            err = FUTHARK_PROGRAM_ERROR;
                            goto cleanup;
                        }
                        // src/bgpc_vv_coloring.fut:52:19-50
                        ((bool *) mem_21857)[cu_21405] = 1;
                    }
                }
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool loop_cond_21416;
            
            if (cond_20173) {
                // src/bgpc_vv_coloring.fut:63:30-37
                
                bool loop_cond_t_res_21639 = ((bool *) mem_21857)[(int64_t) 0];
                
                loop_cond_21416 = loop_cond_t_res_21639;
            } else {
                loop_cond_21416 = 0;
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool c_final_21418;
            int64_t c_final_21419;
            bool loop_while_21420;
            int64_t c_21421;
            
            loop_while_21420 = loop_cond_21416;
            c_21421 = (int64_t) 0;
            while (loop_while_21420) {
                // src/bgpc_vv_coloring.fut:64:9-15
                
                int64_t loopres_21422 = add64((int64_t) 1, c_21421);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool cond_21423 = slt64(loopres_21422, color_bound_20172);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool loop_cond_21424;
                
                if (cond_21423) {
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool x_21640 = sle64((int64_t) 0, loopres_21422);
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool bounds_check_21641 = cond_21423 && x_21640;
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool index_certs_21642;
                    
                    if (!bounds_check_21641) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) loopres_21422, "] out of bounds for array of shape [", (long long) color_bound_20172, "].", "-> #0  src/bgpc_vv_coloring.fut:63:30-37\n   #1  src/bgpc_vv_coloring.fut:81:6-28\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool loop_cond_t_res_21643 = ((bool *) mem_21857)[loopres_21422];
                    
                    loop_cond_21424 = loop_cond_t_res_21643;
                } else {
                    loop_cond_21424 = 0;
                }
                
                bool loop_while_tmp_21978 = loop_cond_21424;
                int64_t c_tmp_21979 = loopres_21422;
                
                loop_while_21420 = loop_while_tmp_21978;
                c_21421 = c_tmp_21979;
            }
            c_final_21418 = loop_while_21420;
            c_final_21419 = c_21421;
            // src/bgpc_vv_coloring.fut:106:25-32
            
            int64_t max_res_20267 = smax64(c_final_21419, redout_21695);
            
            ((int64_t *) mem_21853)[i_21697] = c_final_21419;
            
            int64_t redout_tmp_21973 = max_res_20267;
            
            redout_21695 = redout_tmp_21973;
        }
        defunc_0_reduce_res_21672 = redout_21695;
        // src/bgpc_vv_coloring.fut:113:17-43
        
        bool acc_cert_20273;
        
        // src/bgpc_vv_coloring.fut:113:17-43
        for (int64_t i_21700 = 0; i_21700 < loop_dz2081Uz2083U_20168; i_21700++) {
            int64_t v_20277 = ((int64_t *) mem_param_21851.mem)[i_21700];
            int64_t v_20278 = ((int64_t *) mem_21853)[i_21700];
            
            // src/bgpc_vv_coloring.fut:113:17-43
            // UpdateAcc
            if (sle64((int64_t) 0, v_20277) && slt64(v_20277, (int64_t) 4)) {
                ((int64_t *) mem_21841)[v_20277] = v_20278;
            }
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21874_cached_sizze_22095 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21874, &mem_21874_cached_sizze_22095, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21876_cached_sizze_22096 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21876, &mem_21876_cached_sizze_22096, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21877_cached_sizze_22097 < loop_dz2081Uz2083U_20168) {
            err = lexical_realloc(ctx, &mem_21877, &mem_21877_cached_sizze_22097, loop_dz2081Uz2083U_20168);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t discard_21712;
        int64_t scanacc_21704 = (int64_t) 0;
        
        for (int64_t i_21708 = 0; i_21708 < loop_dz2081Uz2083U_20168; i_21708++) {
            int64_t eta_p_21265 = ((int64_t *) mem_param_21851.mem)[i_21708];
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool x_21266 = sle64((int64_t) 0, eta_p_21265);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool y_21267 = slt64(eta_p_21265, csr_bipartite_from_pattern_res_20122);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool bounds_check_21268 = x_21266 && y_21267;
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool index_certs_21269;
            
            if (!bounds_check_21268) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_21265, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/bgpc_vv_coloring.fut:126:15-27\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:127:26-28
            
            int64_t k_end_21271 = add64((int64_t) 1, eta_p_21265);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool x_21272 = sle64((int64_t) 0, k_end_21271);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool y_21273 = slt64(k_end_21271, csr_bipartite_from_pattern_res_20122);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool bounds_check_21274 = x_21272 && y_21273;
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool index_certs_21275;
            
            if (!bounds_check_21274) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_end_21271, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/bgpc_vv_coloring.fut:127:15-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:126:15-27
            
            int64_t k0_21270 = ((int64_t *) ext_mem_21829.mem)[eta_p_21265];
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            int64_t k_end_21276 = ((int64_t *) ext_mem_21829.mem)[k_end_21271];
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool cond_21277 = slt64(k0_21270, k_end_21276);
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool loses_conflict_vertex_res_21278;
            bool loses_conflict_vertex_res_21279;
            int64_t loses_conflict_vertex_res_21280;
            bool loop_while_21281;
            bool lost_21282;
            int64_t k_21283;
            
            loop_while_21281 = cond_21277;
            lost_21282 = 0;
            k_21283 = k0_21270;
            while (loop_while_21281) {
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool x_21284 = sle64((int64_t) 0, k_21283);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool y_21285 = slt64(k_21283, csr_bipartite_from_pattern_res_20123);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool bounds_check_21286 = x_21284 && y_21285;
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool index_certs_21287;
                
                if (!bounds_check_21286) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_21283, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20123, "].", "-> #0  src/bgpc_vv_coloring.fut:132:15-26\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:132:15-26
                
                int64_t v_21288 = ((int64_t *) ext_mem_21828.mem)[k_21283];
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool x_21289 = sle64((int64_t) 0, v_21288);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool y_21290 = slt64(v_21288, csr_cols_from_pattern_res_20127);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool bounds_check_21291 = x_21289 && y_21290;
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool index_certs_21292;
                
                if (!bounds_check_21291) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_21288, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20127, "].", "-> #0  src/bgpc_vv_coloring.fut:134:19-30\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:135:29-31
                
                int64_t t_end_21294 = add64((int64_t) 1, v_21288);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool x_21295 = sle64((int64_t) 0, t_end_21294);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool y_21296 = slt64(t_end_21294, csr_cols_from_pattern_res_20127);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool bounds_check_21297 = x_21295 && y_21296;
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool index_certs_21298;
                
                if (!bounds_check_21297) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_end_21294, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20127, "].", "-> #0  src/bgpc_vv_coloring.fut:135:19-32\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:134:19-30
                
                int64_t t0_21293 = ((int64_t *) ext_mem_21845.mem)[v_21288];
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                int64_t t_end_21299 = ((int64_t *) ext_mem_21845.mem)[t_end_21294];
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool cond_21300 = slt64(t0_21293, t_end_21299);
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool loopres_21301;
                bool loopres_21302;
                int64_t loopres_21303;
                bool loop_while_21304;
                bool lost_in_net_21305;
                int64_t t_21306;
                
                loop_while_21304 = cond_21300;
                lost_in_net_21305 = 0;
                t_21306 = t0_21293;
                while (loop_while_21304) {
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool x_21307 = sle64((int64_t) 0, t_21306);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool y_21308 = slt64(t_21306, csr_cols_from_pattern_res_20128);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool bounds_check_21309 = x_21307 && y_21308;
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool index_certs_21310;
                    
                    if (!bounds_check_21309) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_21306, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20128, "].", "-> #0  src/bgpc_vv_coloring.fut:140:19-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    int64_t u_21311 = ((int64_t *) ext_mem_21844.mem)[t_21306];
                    
                    // src/bgpc_vv_coloring.fut:142:13-71
                    
                    bool lost_in_netzq_21312;
                    
                    if (lost_in_net_21305) {
                        lost_in_netzq_21312 = 1;
                    } else {
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_21313 = u_21311 == eta_p_21265;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_21314 = !cond_21313;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_21315;
                        
                        if (cond_21314) {
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool x_21645 = sle64((int64_t) 0, u_21311);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool y_21646 = slt64(u_21311, (int64_t) 4);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool bounds_check_21647 = x_21645 && y_21646;
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool index_certs_21648;
                            
                            if (!bounds_check_21647) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_21311, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/bgpc_vv_coloring.fut:142:39-48\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool y_21650 = slt64(eta_p_21265, (int64_t) 4);
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool bounds_check_21651 = x_21266 && y_21650;
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool index_certs_21652;
                            
                            if (!bounds_check_21651) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_21265, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/bgpc_vv_coloring.fut:142:52-61\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            int64_t zeze_lhs_21649 = ((int64_t *) mem_21841)[u_21311];
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            int64_t zeze_rhs_21653 = ((int64_t *) mem_21841)[eta_p_21265];
                            
                            // src/bgpc_vv_coloring.fut:142:49-61
                            
                            bool cond_t_res_21654 = zeze_lhs_21649 == zeze_rhs_21653;
                            
                            cond_21315 = cond_t_res_21654;
                        } else {
                            cond_21315 = 0;
                        }
                        // src/bgpc_vv_coloring.fut:142:67-70
                        
                        bool lost_in_netzq_f_res_t_res_21326 = slt64(u_21311, eta_p_21265);
                        
                        // src/bgpc_vv_coloring.fut:160:14-168:57
                        
                        bool x_21327 = cond_21315 && lost_in_netzq_f_res_t_res_21326;
                        
                        lost_in_netzq_21312 = x_21327;
                    }
                    // src/bgpc_vv_coloring.fut:143:31-37
                    
                    int64_t tmp_21328 = add64((int64_t) 1, t_21306);
                    
                    // src/bgpc_vv_coloring.fut:139:15-43
                    
                    bool cond_21329 = slt64(tmp_21328, t_end_21299);
                    
                    // src/bgpc_vv_coloring.fut:139:28-43
                    
                    bool not_res_21330 = !lost_in_netzq_21312;
                    
                    // src/bgpc_vv_coloring.fut:166:5-169:10
                    
                    bool x_21331 = cond_21329 && not_res_21330;
                    bool loop_while_tmp_21988 = x_21331;
                    bool lost_in_net_tmp_21989 = lost_in_netzq_21312;
                    int64_t t_tmp_21990 = tmp_21328;
                    
                    loop_while_21304 = loop_while_tmp_21988;
                    lost_in_net_21305 = lost_in_net_tmp_21989;
                    t_21306 = t_tmp_21990;
                }
                loopres_21301 = loop_while_21304;
                loopres_21302 = lost_in_net_21305;
                loopres_21303 = t_21306;
                // src/bgpc_vv_coloring.fut:160:14-168:57
                
                bool lostzq_21332 = lost_21282 || loopres_21302;
                
                // src/bgpc_vv_coloring.fut:146:20-26
                
                int64_t tmp_21333 = add64((int64_t) 1, k_21283);
                
                // src/bgpc_vv_coloring.fut:131:11-32
                
                bool cond_21334 = slt64(tmp_21333, k_end_21276);
                
                // src/bgpc_vv_coloring.fut:131:24-32
                
                bool not_res_21335 = !lostzq_21332;
                
                // src/bgpc_vv_coloring.fut:166:5-169:10
                
                bool x_21336 = cond_21334 && not_res_21335;
                bool loop_while_tmp_21985 = x_21336;
                bool lost_tmp_21986 = lostzq_21332;
                int64_t k_tmp_21987 = tmp_21333;
                
                loop_while_21281 = loop_while_tmp_21985;
                lost_21282 = lost_tmp_21986;
                k_21283 = k_tmp_21987;
            }
            loses_conflict_vertex_res_21278 = loop_while_21281;
            loses_conflict_vertex_res_21279 = lost_21282;
            loses_conflict_vertex_res_21280 = k_21283;
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_f_res_21338 = btoi_bool_i64(loses_conflict_vertex_res_21279);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_op_res_20379 = add64(defunc_0_f_res_21338, scanacc_21704);
            
            ((int64_t *) mem_21874)[i_21708] = defunc_0_op_res_20379;
            ((int64_t *) mem_21876)[i_21708] = defunc_0_f_res_21338;
            ((bool *) mem_21877)[i_21708] = loses_conflict_vertex_res_21279;
            
            int64_t scanacc_tmp_21981 = defunc_0_op_res_20379;
            
            scanacc_21704 = scanacc_tmp_21981;
        }
        discard_21712 = scanacc_21704;
        // src/bgpc_vv_coloring.fut:23:25-28
        
        int64_t tmp_20363 = sub64(loop_dz2081Uz2083U_20168, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool y_20365 = slt64(tmp_20363, loop_dz2081Uz2083U_20168);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool x_20364 = sle64((int64_t) 0, tmp_20363);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool bounds_check_20366 = x_20364 && y_20365;
        
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        bool cond_20361 = loop_dz2081Uz2083U_20168 == (int64_t) 0;
        
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool protect_assert_disj_20367 = cond_20361 || bounds_check_20366;
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool index_certs_20368;
        
        if (!protect_assert_disj_20367) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_20363, "] out of bounds for array of shape [", (long long) loop_dz2081Uz2083U_20168, "].", "-> #0  src/bgpc_vv_coloring.fut:23:10-29\n   #1  src/bgpc_vv_coloring.fut:171:23-55\n   #2  src/bgpc_vv_coloring.fut:186:14-205:56\n   #3  src/bgpc_vv_coloring.fut:220:14-223:64\n   #4  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #5  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #6  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #7  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool x_20362 = !cond_20361;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_f_res_20380;
        
        if (x_20362) {
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t x_21655 = ((int64_t *) mem_21874)[tmp_20363];
            
            m_f_res_20380 = x_21655;
        } else {
            m_f_res_20380 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_20382;
        
        if (cond_20361) {
            m_20382 = (int64_t) 0;
        } else {
            m_20382 = m_f_res_20380;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t bytes_21896 = (int64_t) 8 * m_20382;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21897_cached_sizze_22098 < bytes_21896) {
            err = lexical_realloc(ctx, &mem_21897, &mem_21897_cached_sizze_22098, bytes_21896);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21899_cached_sizze_22099 < bytes_21896) {
            err = lexical_realloc(ctx, &mem_21899, &mem_21899_cached_sizze_22099, bytes_21896);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:18:24-27:44
        if (mem_21901_cached_sizze_22100 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21901, &mem_21901_cached_sizze_22100, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21909_cached_sizze_22101 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21909, &mem_21909_cached_sizze_22101, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        bool acc_cert_21138;
        bool acc_cert_21139;
        
        // src/bgpc_vv_coloring.fut:18:24-27:44
        
        int64_t inpacc_21663;
        int64_t inpacc_21202 = (int64_t) 0;
        
        for (int64_t i_21739 = 0; i_21739 < loop_dz2081Uz2083U_20168; i_21739++) {
            bool eta_p_21778 = ((bool *) mem_21877)[i_21739];
            
            // src/bgpc_vv_coloring.fut:18:28-39
            
            int64_t bool_to_i64_res_21779 = btoi_bool_i64(eta_p_21778);
            int64_t eta_p_21791 = ((int64_t *) mem_21876)[i_21739];
            int64_t eta_p_21792 = ((int64_t *) mem_21874)[i_21739];
            int64_t v_21795 = ((int64_t *) mem_param_21851.mem)[i_21739];
            
            // src/bgpc_vv_coloring.fut:19:29-32
            
            int64_t defunc_0_op_res_21796 = add64(inpacc_21202, bool_to_i64_res_21779);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            bool cond_21797 = eta_p_21791 == (int64_t) 1;
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t lifted_lambda_res_21798;
            
            if (cond_21797) {
                // src/bgpc_vv_coloring.fut:27:5-44
                
                int64_t lifted_lambda_res_t_res_21799 = sub64(eta_p_21792, (int64_t) 1);
                
                lifted_lambda_res_21798 = lifted_lambda_res_t_res_21799;
            } else {
                lifted_lambda_res_21798 = (int64_t) -1;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_21798) && slt64(lifted_lambda_res_21798, m_20382)) {
                ((int64_t *) mem_21899)[lifted_lambda_res_21798] = v_21795;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_21798) && slt64(lifted_lambda_res_21798, m_20382)) {
                ((int64_t *) mem_21897)[lifted_lambda_res_21798] = defunc_0_op_res_21796;
            }
            ((int64_t *) mem_21901)[i_21739] = defunc_0_op_res_21796;
            
            int64_t inpacc_tmp_21991 = defunc_0_op_res_21796;
            
            inpacc_21202 = inpacc_tmp_21991;
        }
        inpacc_21663 = inpacc_21202;
        // src/bgpc_vv_coloring.fut:27:5-44
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_21909, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_21901, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {loop_dz2081Uz2083U_20168});
        // src/bgpc_vv_coloring.fut:23:10-29
        
        int64_t out_szz_f_res_20369;
        
        if (x_20362) {
            // src/bgpc_vv_coloring.fut:171:23-55
            
            int64_t x_21669 = ((int64_t *) mem_21909)[tmp_20363];
            
            out_szz_f_res_20369 = x_21669;
        } else {
            out_szz_f_res_20369 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        int64_t out_szz_20371;
        
        if (cond_20361) {
            out_szz_20371 = (int64_t) 0;
        } else {
            out_szz_20371 = out_szz_f_res_20369;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        
        int64_t bytes_21910 = (int64_t) 8 * out_szz_20371;
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        bool cond_20268 = slt64(defunc_0_reduce_res_21672, (int64_t) 0);
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        int64_t next_color_bound_20269;
        
        if (cond_20268) {
            next_color_bound_20269 = color_bound_20172;
        } else {
            // src/bgpc_vv_coloring.fut:110:40-46
            
            int64_t max_arg1_20270 = add64((int64_t) 2, defunc_0_reduce_res_21672);
            
            // src/bgpc_vv_coloring.fut:110:10-46
            
            int64_t max_res_20271 = smax64(color_bound_20172, max_arg1_20270);
            
            next_color_bound_20269 = max_res_20271;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        if (memblock_alloc(ctx, &mem_21911, bytes_21910, "mem_21911")) {
            err = 1;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        for (int64_t nest_i_21995 = 0; nest_i_21995 < out_szz_20371; nest_i_21995++) {
            ((int64_t *) mem_21911.mem)[nest_i_21995] = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:33:6-27
        
        bool acc_cert_20911;
        
        // src/bgpc_vv_coloring.fut:29:21-33:27
        for (int64_t i_21744 = 0; i_21744 < m_20382; i_21744++) {
            int64_t eta_p_20923 = ((int64_t *) mem_21897)[i_21744];
            int64_t v_20925 = ((int64_t *) mem_21899)[i_21744];
            
            // src/bgpc_vv_coloring.fut:29:44-50
            
            int64_t lifted_lambda_res_20926 = sub64(eta_p_20923, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:33:6-27
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_20926) && slt64(lifted_lambda_res_20926, out_szz_20371)) {
                ((int64_t *) mem_21911.mem)[lifted_lambda_res_20926] = v_20925;
            }
        }
        // src/bgpc_vv_coloring.fut:175:17-48
        
        bool acc_cert_20419;
        
        // src/bgpc_vv_coloring.fut:175:17-48
        for (int64_t i_21746 = 0; i_21746 < out_szz_20371; i_21746++) {
            int64_t v_20423 = ((int64_t *) mem_21911.mem)[i_21746];
            
            // src/bgpc_vv_coloring.fut:175:17-48
            // UpdateAcc
            if (sle64((int64_t) 0, v_20423) && slt64(v_20423, (int64_t) 4)) {
                ((int64_t *) mem_21841)[v_20423] = (int64_t) -1;
            }
        }
        // src/bgpc_vv_coloring.fut:198:20-23
        
        bool loop_cond_20425 = slt64((int64_t) 0, out_szz_20371);
        
        if (memblock_set(ctx, &mem_param_tmp_21967, &mem_21911, "mem_21911") != 0)
            return 1;
        
        int64_t loop_dz2081Uz2083U_tmp_21968 = out_szz_20371;
        bool loop_while_tmp_21969 = loop_cond_20425;
        int64_t color_bound_tmp_21972 = next_color_bound_20269;
        
        if (memblock_set(ctx, &mem_param_21851, &mem_param_tmp_21967, "mem_param_tmp_21967") != 0)
            return 1;
        loop_dz2081Uz2083U_20168 = loop_dz2081Uz2083U_tmp_21968;
        loop_while_20169 = loop_while_tmp_21969;
        color_bound_20172 = color_bound_tmp_21972;
    }
    if (memblock_set(ctx, &ext_mem_21916, &mem_param_21851, "mem_param_21851") != 0)
        return 1;
    vv_color_side_order_res_20163 = loop_dz2081Uz2083U_20168;
    vv_color_side_order_res_20164 = loop_while_20169;
    vv_color_side_order_res_20167 = color_bound_20172;
    if (memblock_unref(ctx, &mem_21840, "mem_21840") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21844, "ext_mem_21844") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21845, "ext_mem_21845") != 0)
        return 1;
    // src/sparse_jacobian_vjp.fut:78:8-87:44
    
    int64_t x_21683;
    int64_t redout_21747 = (int64_t) 0;
    
    for (int64_t i_21748 = 0; i_21748 < (int64_t) 4; i_21748++) {
        int64_t x_20428 = ((int64_t *) mem_21841)[i_21748];
        
        // src/sparse_jacobian_vjp.fut:14:22-29
        
        int64_t max_res_20431 = smax64(x_20428, redout_21747);
        int64_t redout_tmp_21998 = max_res_20431;
        
        redout_21747 = redout_tmp_21998;
    }
    x_21683 = redout_21747;
    // src/sparse_jacobian_vjp.fut:14:13-45
    
    int64_t num_colors_of_res_f_res_20432 = add64((int64_t) 1, x_21683);
    
    // src/sparse_jacobian_vjp.fut:29:11-18
    
    bool bounds_invalid_upwards_20434 = slt64(num_colors_of_res_f_res_20432, (int64_t) 0);
    
    // src/sparse_jacobian_vjp.fut:29:11-18
    
    bool valid_20435 = !bounds_invalid_upwards_20434;
    
    // src/sparse_jacobian_vjp.fut:29:11-18
    
    bool range_valid_c_20436;
    
    if (!valid_20435) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Range ", (long long) (int64_t) 0, "..", (long long) (int64_t) 1, "..<", (long long) num_colors_of_res_f_res_20432, " is invalid.", "-> #0  src/sparse_jacobian_vjp.fut:29:11-18\n   #1  src/sparse_jacobian_vjp.fut:78:8-87:44\n   #2  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #3  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #4  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:21:8-29:18
    
    int64_t bytes_21919 = (int64_t) 48 * num_colors_of_res_f_res_20432;
    
    // src/sparse_jacobian_vjp.fut:42:15-35
    if (mem_21831_cached_sizze_22090 < bytes_21830) {
        err = lexical_realloc(ctx, &mem_21831, &mem_21831_cached_sizze_22090, bytes_21830);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:42:15-35
    for (int64_t nest_i_21999 = 0; nest_i_21999 < csr_bipartite_from_pattern_res_20123; nest_i_21999++) {
        ((double *) mem_21831)[nest_i_21999] = 0.0;
    }
    
    double zt_lhs_20550 = ((double *) x_mem_21827.mem)[(int64_t) 0];
    double zt_rhs_20551 = ((double *) x_mem_21827.mem)[(int64_t) 1];
    double zt_lhs_20557 = ((double *) x_mem_21827.mem)[(int64_t) 5];
    
    // src/sparse_jacobian_vjp.fut:21:8-29:18
    if (mem_21920_cached_sizze_22102 < bytes_21919) {
        err = lexical_realloc(ctx, &mem_21920, &mem_21920_cached_sizze_22102, bytes_21919);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:17:3-61
    if (mem_21925_cached_sizze_22103 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_21925, &mem_21925_cached_sizze_22103, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:21:8-29:18
    for (int64_t i_21755 = 0; i_21755 < num_colors_of_res_f_res_20432; i_21755++) {
        // src/sparse_jacobian_vjp.fut:17:3-61
        for (int64_t i_21751 = 0; i_21751 < (int64_t) 4; i_21751++) {
            int64_t eta_p_20445 = ((int64_t *) mem_21841)[i_21751];
            
            // src/sparse_jacobian_vjp.fut:17:15-49
            
            bool cond_20446 = eta_p_20445 == i_21755;
            
            // src/sparse_jacobian_vjp.fut:17:15-49
            
            double lifted_lambda_res_20447;
            
            if (cond_20446) {
                lifted_lambda_res_20447 = 1.0;
            } else {
                lifted_lambda_res_20447 = 0.0;
            }
            ((double *) mem_21925)[i_21751] = lifted_lambda_res_20447;
        }
        
        double elem_adj_21572 = ((double *) mem_21925)[(int64_t) 0];
        double elem_adj_21573 = ((double *) mem_21925)[(int64_t) 1];
        double elem_adj_21574 = ((double *) mem_21925)[(int64_t) 2];
        double binop_y_adj_21581 = -1.0 * elem_adj_21573;
        double zm_lhs_adj_21582 = elem_adj_21573 + elem_adj_21574;
        double binop_x_adj_21583 = zt_lhs_20557 * binop_y_adj_21581;
        double zt_lhs_adj_21585 = binop_x_adj_21583 + binop_x_adj_21583;
        double binop_y_adj_21589 = 3.0 * elem_adj_21572;
        double binop_x_adj_21590 = zt_rhs_20551 * elem_adj_21572;
        double binop_y_adj_21591 = zt_lhs_20550 * elem_adj_21572;
        double zt_rhs_adj_21592 = elem_adj_21574 + binop_y_adj_21591;
        
        for (int64_t nest_i_22002 = 0; nest_i_22002 < (int64_t) 6; nest_i_22002++) {
            ((double *) mem_21920)[i_21755 * (int64_t) 6 + nest_i_22002] = 0.0;
        }
        ((double *) mem_21920)[i_21755 * (int64_t) 6] = binop_x_adj_21590;
        ((double *) mem_21920)[i_21755 * (int64_t) 6 + (int64_t) 1] = zt_rhs_adj_21592;
        ((double *) mem_21920)[i_21755 * (int64_t) 6 + (int64_t) 4] = binop_y_adj_21589;
        ((double *) mem_21920)[i_21755 * (int64_t) 6 + (int64_t) 2] = zm_lhs_adj_21582;
        ((double *) mem_21920)[i_21755 * (int64_t) 6 + (int64_t) 5] = zt_lhs_adj_21585;
        ((double *) mem_21920)[i_21755 * (int64_t) 6 + (int64_t) 3] = elem_adj_21574;
    }
    // src/sparse_jacobian_vjp.fut:45:5-53:27
    
    bool compressed_to_csr_vals_res_20456;
    int64_t compressed_to_csr_vals_res_20458;
    bool loop_while_20459;
    int64_t i_20461;
    
    loop_while_20459 = 1;
    i_20461 = (int64_t) 0;
    while (loop_while_20459) {
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool x_20462 = sle64((int64_t) 0, i_20461);
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool y_20463 = slt64(i_20461, csr_bipartite_from_pattern_res_20122);
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool bounds_check_20464 = x_20462 && y_20463;
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool index_certs_20465;
        
        if (!bounds_check_20464) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_20461, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/sparse_jacobian_vjp.fut:47:15-26\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        int64_t s_20466 = ((int64_t *) ext_mem_21829.mem)[i_20461];
        
        // src/sparse_jacobian_vjp.fut:48:25-27
        
        int64_t e_20467 = add64((int64_t) 1, i_20461);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool x_20468 = sle64((int64_t) 0, e_20467);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool y_20469 = slt64(e_20467, csr_bipartite_from_pattern_res_20122);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool bounds_check_20470 = x_20468 && y_20469;
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool index_certs_20471;
        
        if (!bounds_check_20470) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_20467, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/sparse_jacobian_vjp.fut:48:15-28\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        int64_t e_20472 = ((int64_t *) ext_mem_21829.mem)[e_20467];
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t j_m_i_20473 = sub64(e_20472, s_20466);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool empty_slice_20474 = j_m_i_20473 == (int64_t) 0;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t m_20475 = sub64(j_m_i_20473, (int64_t) 1);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t i_p_m_t_s_20476 = add64(s_20466, m_20475);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_20477 = sle64((int64_t) 0, i_p_m_t_s_20476);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_20478 = slt64(i_p_m_t_s_20476, csr_bipartite_from_pattern_res_20123);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool zzero_lte_i_20479 = sle64((int64_t) 0, s_20466);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool i_lte_j_20480 = sle64(s_20466, e_20472);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool y_20481 = i_p_m_t_s_leq_w_20478 && zzero_lte_i_20479;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool y_20482 = zzero_leq_i_p_m_t_s_20477 && y_20481;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool forwards_ok_20483 = i_lte_j_20480 && y_20482;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool ok_or_empty_20484 = empty_slice_20474 || forwards_ok_20483;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool index_certs_20485;
        
        if (!ok_or_empty_20484) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_20466, ":", (long long) e_20472, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20123, "].", "-> #0  src/sparse_jacobian_vjp.fut:49:18-30\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool y_20487 = slt64(i_20461, (int64_t) 4);
        
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool bounds_check_20488 = x_20462 && y_20487;
        
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool index_certs_20489;
        
        if (!bounds_check_20488) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_20461, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/sparse_jacobian_vjp.fut:50:16-29\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        int64_t rc_20490 = ((int64_t *) mem_21841)[i_20461];
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool y_20492 = slt64(rc_20490, num_colors_of_res_f_res_20432);
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool x_20491 = sle64((int64_t) 0, rc_20490);
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool bounds_check_20493 = x_20491 && y_20492;
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool index_certs_20494;
        
        if (!bounds_check_20493) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) rc_20490, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_20432, "].", "-> #0  src/sparse_jacobian_vjp.fut:51:28-34\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:51:17-43
        for (int64_t i_21759 = 0; i_21759 < j_m_i_20473; i_21759++) {
            int64_t index_primexp_21771 = s_20466 + i_21759;
            int64_t eta_p_20496 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21771];
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool x_20497 = sle64((int64_t) 0, eta_p_20496);
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool y_20498 = slt64(eta_p_20496, (int64_t) 6);
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool bounds_check_20499 = x_20497 && y_20498;
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool index_certs_20500;
            
            if (!bounds_check_20499) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_20496, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/sparse_jacobian_vjp.fut:51:28-37\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_vjp.fut:51:17-43
            
            double lifted_lambda_res_20501 = ((double *) mem_21920)[rc_20490 * (int64_t) 6 + eta_p_20496];
            
            ((double *) mem_21831)[s_20466 + i_21759] = lifted_lambda_res_20501;
        }
        // src/sparse_jacobian_vjp.fut:46:13-16
        
        bool loop_cond_20503 = slt64(e_20467, (int64_t) 4);
        bool loop_while_tmp_22003 = loop_cond_20503;
        int64_t i_tmp_22005 = e_20467;
        
        loop_while_20459 = loop_while_tmp_22003;
        i_20461 = i_tmp_22005;
    }
    compressed_to_csr_vals_res_20456 = loop_while_20459;
    compressed_to_csr_vals_res_20458 = i_20461;
    // src/dense_jacobian.fut:5:3-21
    if (mem_21951_cached_sizze_22104 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_21951, &mem_21951_cached_sizze_22104, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    if (mem_21952_cached_sizze_22105 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_21952, &mem_21952_cached_sizze_22105, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:68:29-47
    if (mem_21953_cached_sizze_22106 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_21953, &mem_21953_cached_sizze_22106, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:7:19-9:39
    
    bool defunc_0_reduce_res_21685;
    bool redout_21765 = 1;
    
    for (int64_t i_21766 = 0; i_21766 < (int64_t) 4; i_21766++) {
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        bool y_20834 = slt64(i_21766, csr_bipartite_from_pattern_res_20122);
        
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        bool index_certs_20836;
        
        if (!y_20834) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_21766, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/sparse_jacobian_vjp.fut:64:17-28\n   #1  src/sparse_jacobian_vjp.fut:132:18-138:40\n   #2  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        int64_t s_20837 = ((int64_t *) ext_mem_21829.mem)[i_21766];
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool zzero_lte_i_20850 = sle64((int64_t) 0, s_20837);
        
        // src/sparse_jacobian_vjp.fut:65:27-29
        
        int64_t e_20838 = add64((int64_t) 1, i_21766);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool y_20840 = slt64(e_20838, csr_bipartite_from_pattern_res_20122);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool x_20839 = sle64((int64_t) 0, e_20838);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool bounds_check_20841 = x_20839 && y_20840;
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool index_certs_20842;
        
        if (!bounds_check_20841) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_20838, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20122, "].", "-> #0  src/sparse_jacobian_vjp.fut:65:17-30\n   #1  src/sparse_jacobian_vjp.fut:132:18-138:40\n   #2  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        int64_t e_20843 = ((int64_t *) ext_mem_21829.mem)[e_20838];
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t j_m_i_20844 = sub64(e_20843, s_20837);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t m_20846 = sub64(j_m_i_20844, (int64_t) 1);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t i_p_m_t_s_20847 = add64(s_20837, m_20846);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool i_p_m_t_s_leq_w_20849 = slt64(i_p_m_t_s_20847, csr_bipartite_from_pattern_res_20123);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool y_20858 = i_p_m_t_s_leq_w_20849 && zzero_lte_i_20850;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool zzero_leq_i_p_m_t_s_20848 = sle64((int64_t) 0, i_p_m_t_s_20847);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool y_20859 = zzero_leq_i_p_m_t_s_20848 && y_20858;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool i_lte_j_20851 = sle64(s_20837, e_20843);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool forwards_ok_20860 = i_lte_j_20851 && y_20859;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool empty_slice_20845 = j_m_i_20844 == (int64_t) 0;
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool ok_or_empty_20861 = empty_slice_20845 || forwards_ok_20860;
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool index_certs_20862;
        
        if (!ok_or_empty_20861) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_20837, ":", (long long) e_20843, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20123, "].", "-> #0  src/sparse_jacobian_vjp.fut:67:20-29\n   #1  src/sparse_jacobian_vjp.fut:132:18-138:40\n   #2  test/test_sparse_jacobian_vjp.fut:80:15-103:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/dense_jacobian.fut:5:3-21
        for (int64_t nest_i_22008 = 0; nest_i_22008 < (int64_t) 4; nest_i_22008++) {
            ((double *) mem_21951)[nest_i_22008] = 0.0;
        }
        // src/dense_jacobian.fut:5:3-39
        ((double *) mem_21951)[i_21766] = 1.0;
        
        double elem_adj_21601 = ((double *) mem_21951)[(int64_t) 0];
        double elem_adj_21602 = ((double *) mem_21951)[(int64_t) 1];
        double elem_adj_21603 = ((double *) mem_21951)[(int64_t) 2];
        double binop_y_adj_21610 = -1.0 * elem_adj_21602;
        double zm_lhs_adj_21611 = elem_adj_21602 + elem_adj_21603;
        double binop_x_adj_21612 = zt_lhs_20557 * binop_y_adj_21610;
        double zt_lhs_adj_21614 = binop_x_adj_21612 + binop_x_adj_21612;
        double binop_y_adj_21618 = 3.0 * elem_adj_21601;
        double binop_x_adj_21619 = zt_rhs_20551 * elem_adj_21601;
        double binop_y_adj_21620 = zt_lhs_20550 * elem_adj_21601;
        double zt_rhs_adj_21621 = elem_adj_21603 + binop_y_adj_21620;
        
        for (int64_t nest_i_22009 = 0; nest_i_22009 < (int64_t) 6; nest_i_22009++) {
            ((double *) mem_21952)[nest_i_22009] = 0.0;
        }
        ((double *) mem_21952)[(int64_t) 0] = binop_x_adj_21619;
        ((double *) mem_21952)[(int64_t) 1] = zt_rhs_adj_21621;
        ((double *) mem_21952)[(int64_t) 4] = binop_y_adj_21618;
        ((double *) mem_21952)[(int64_t) 2] = zm_lhs_adj_21611;
        ((double *) mem_21952)[(int64_t) 5] = zt_lhs_adj_21614;
        ((double *) mem_21952)[(int64_t) 3] = elem_adj_21603;
        // src/sparse_jacobian_vjp.fut:68:29-47
        for (int64_t nest_i_22010 = 0; nest_i_22010 < (int64_t) 6; nest_i_22010++) {
            ((double *) mem_21953)[nest_i_22010] = 0.0;
        }
        // src/sparse_jacobian_vjp.fut:69:12-32
        
        bool acc_cert_20866;
        
        // src/sparse_jacobian_vjp.fut:69:12-32
        for (int64_t i_21762 = 0; i_21762 < j_m_i_20844; i_21762++) {
            int64_t index_primexp_21768 = s_20837 + i_21762;
            int64_t v_20870 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21768];
            double v_20871 = ((double *) mem_21831)[index_primexp_21768];
            
            // src/sparse_jacobian_vjp.fut:69:12-32
            // UpdateAcc
            if (sle64((int64_t) 0, v_20870) && slt64(v_20870, (int64_t) 6)) {
                ((double *) mem_21953)[v_20870] = v_20871;
            }
        }
        // test/test_sparse_jacobian_vjp.fut:8:5-66
        
        bool defunc_0_reduce_res_21681;
        bool redout_21763 = 1;
        
        for (int64_t i_21764 = 0; i_21764 < (int64_t) 6; i_21764++) {
            bool eta_p_20894 = ((bool *) mem_21819.mem)[i_21766 * (int64_t) 6 + i_21764];
            double eta_p_20895 = ((double *) mem_21952)[i_21764];
            double eta_p_20896 = ((double *) mem_21953)[i_21764];
            
            // test/test_sparse_jacobian_vjp.fut:13:25-48
            
            double lifted_lambda_res_20897;
            
            if (eta_p_20894) {
                lifted_lambda_res_20897 = eta_p_20895;
            } else {
                lifted_lambda_res_20897 = 0.0;
            }
            // test/test_sparse_jacobian_vjp.fut:8:48-51
            
            double abs_arg0_20899 = eta_p_20896 - lifted_lambda_res_20897;
            
            // test/test_sparse_jacobian_vjp.fut:8:37-51
            
            double abs_res_20900 = fabs64(abs_arg0_20899);
            
            // test/test_sparse_jacobian_vjp.fut:8:53-59
            
            bool lifted_lambda_res_20901 = abs_res_20900 <= 1.0e-9;
            
            // test/test_sparse_jacobian_vjp.fut:8:5-66
            
            bool x_20884 = lifted_lambda_res_20901 && redout_21763;
            bool redout_tmp_22012 = x_20884;
            
            redout_21763 = redout_tmp_22012;
        }
        defunc_0_reduce_res_21681 = redout_21763;
        // test/test_sparse_jacobian_vjp.fut:9:6-39
        
        bool x_20645 = defunc_0_reduce_res_21681 && redout_21765;
        bool redout_tmp_22007 = x_20645;
        
        redout_21765 = redout_tmp_22007;
    }
    defunc_0_reduce_res_21685 = redout_21765;
    if (memblock_unref(ctx, &ext_mem_21828, "ext_mem_21828") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21829, "ext_mem_21829") != 0)
        return 1;
    prim_out_21962 = defunc_0_reduce_res_21685;
    *out_prim_out_22089 = prim_out_21962;
    
  cleanup:
    {
        free(mem_21831);
        free(mem_21832);
        free(mem_21841);
        free(mem_21853);
        free(mem_21857);
        free(mem_21874);
        free(mem_21876);
        free(mem_21877);
        free(mem_21897);
        free(mem_21899);
        free(mem_21901);
        free(mem_21909);
        free(mem_21920);
        free(mem_21925);
        free(mem_21951);
        free(mem_21952);
        free(mem_21953);
        if (memblock_unref(ctx, &mem_param_tmp_21967, "mem_param_tmp_21967") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21911, "mem_21911") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_21851, "mem_param_21851") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21916, "ext_mem_21916") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21844, "ext_mem_21844") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21845, "ext_mem_21845") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21842, "mem_21842") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21840, "mem_21840") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21828, "ext_mem_21828") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21829, "ext_mem_21829") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_vjp_ex4_with_row_colors_matches_dense(struct futhark_context *ctx, bool *out_prim_out_22107, struct memblock x_mem_21827)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_21831_cached_sizze_22108 = 0;
    unsigned char *mem_21831 = NULL;
    int64_t mem_21834_cached_sizze_22109 = 0;
    unsigned char *mem_21834 = NULL;
    int64_t mem_21839_cached_sizze_22110 = 0;
    unsigned char *mem_21839 = NULL;
    int64_t mem_21865_cached_sizze_22111 = 0;
    unsigned char *mem_21865 = NULL;
    int64_t mem_21866_cached_sizze_22112 = 0;
    unsigned char *mem_21866 = NULL;
    int64_t mem_21867_cached_sizze_22113 = 0;
    unsigned char *mem_21867 = NULL;
    struct memblock ext_mem_21828;
    
    ext_mem_21828.references = NULL;
    
    struct memblock ext_mem_21829;
    
    ext_mem_21829.references = NULL;
    
    struct memblock mem_21809 = ctx->constants->mem_21809;
    struct memblock mem_21812 = ctx->constants->mem_21812;
    struct memblock mem_21814 = ctx->constants->mem_21814;
    struct memblock mem_21819 = ctx->constants->mem_21819;
    struct memblock mem_21820 = ctx->constants->mem_21820;
    struct memblock mem_21826 = ctx->constants->mem_21826;
    bool prim_out_21962;
    
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_18694;
    int64_t csr_bipartite_from_pattern_res_18695;
    
    if (futrts_csr_rows_from_pattern_7010(ctx, &ext_mem_21829, &ext_mem_21828, &csr_bipartite_from_pattern_res_18694, &csr_bipartite_from_pattern_res_18695, mem_21819, (int64_t) 4, (int64_t) 6) != 0) {
        err = 1;
        goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:42:15-35
    
    int64_t bytes_21830 = (int64_t) 8 * csr_bipartite_from_pattern_res_18695;
    
    // src/sparse_jacobian_vjp.fut:93:8-100:44
    
    int64_t x_21638;
    int64_t redout_21686 = (int64_t) 0;
    
    for (int64_t i_21687 = 0; i_21687 < (int64_t) 4; i_21687++) {
        int64_t x_18702 = ((int64_t *) mem_21820.mem)[i_21687];
        
        // src/sparse_jacobian_vjp.fut:14:22-29
        
        int64_t max_res_18705 = smax64(x_18702, redout_21686);
        int64_t redout_tmp_21963 = max_res_18705;
        
        redout_21686 = redout_tmp_21963;
    }
    x_21638 = redout_21686;
    // src/sparse_jacobian_vjp.fut:14:13-45
    
    int64_t num_colors_of_res_f_res_18706 = add64((int64_t) 1, x_21638);
    
    // src/sparse_jacobian_vjp.fut:29:11-18
    
    bool bounds_invalid_upwards_18708 = slt64(num_colors_of_res_f_res_18706, (int64_t) 0);
    
    // src/sparse_jacobian_vjp.fut:29:11-18
    
    bool valid_18709 = !bounds_invalid_upwards_18708;
    
    // src/sparse_jacobian_vjp.fut:29:11-18
    
    bool range_valid_c_18710;
    
    if (!valid_18709) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Range ", (long long) (int64_t) 0, "..", (long long) (int64_t) 1, "..<", (long long) num_colors_of_res_f_res_18706, " is invalid.", "-> #0  src/sparse_jacobian_vjp.fut:29:11-18\n   #1  src/sparse_jacobian_vjp.fut:93:8-100:44\n   #2  src/sparse_jacobian_vjp.fut:119:8-124:58\n   #3  src/sparse_jacobian_vjp.fut:142:8-148:51\n   #4  test/test_sparse_jacobian_vjp.fut:80:15-113:79\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:21:8-29:18
    
    int64_t bytes_21833 = (int64_t) 48 * num_colors_of_res_f_res_18706;
    
    // src/sparse_jacobian_vjp.fut:42:15-35
    if (mem_21831_cached_sizze_22108 < bytes_21830) {
        err = lexical_realloc(ctx, &mem_21831, &mem_21831_cached_sizze_22108, bytes_21830);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:42:15-35
    for (int64_t nest_i_21964 = 0; nest_i_21964 < csr_bipartite_from_pattern_res_18695; nest_i_21964++) {
        ((double *) mem_21831)[nest_i_21964] = 0.0;
    }
    
    double zt_lhs_20550 = ((double *) x_mem_21827.mem)[(int64_t) 0];
    double zt_rhs_20551 = ((double *) x_mem_21827.mem)[(int64_t) 1];
    double zt_lhs_20557 = ((double *) x_mem_21827.mem)[(int64_t) 5];
    
    // src/sparse_jacobian_vjp.fut:21:8-29:18
    if (mem_21834_cached_sizze_22109 < bytes_21833) {
        err = lexical_realloc(ctx, &mem_21834, &mem_21834_cached_sizze_22109, bytes_21833);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:17:3-61
    if (mem_21839_cached_sizze_22110 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_21839, &mem_21839_cached_sizze_22110, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:21:8-29:18
    for (int64_t i_21694 = 0; i_21694 < num_colors_of_res_f_res_18706; i_21694++) {
        // src/sparse_jacobian_vjp.fut:17:3-61
        for (int64_t i_21690 = 0; i_21690 < (int64_t) 4; i_21690++) {
            int64_t eta_p_18719 = ((int64_t *) mem_21820.mem)[i_21690];
            
            // src/sparse_jacobian_vjp.fut:17:15-49
            
            bool cond_18720 = eta_p_18719 == i_21694;
            
            // src/sparse_jacobian_vjp.fut:17:15-49
            
            double lifted_lambda_res_18721;
            
            if (cond_18720) {
                lifted_lambda_res_18721 = 1.0;
            } else {
                lifted_lambda_res_18721 = 0.0;
            }
            ((double *) mem_21839)[i_21690] = lifted_lambda_res_18721;
        }
        
        double elem_adj_21572 = ((double *) mem_21839)[(int64_t) 0];
        double elem_adj_21573 = ((double *) mem_21839)[(int64_t) 1];
        double elem_adj_21574 = ((double *) mem_21839)[(int64_t) 2];
        double binop_y_adj_21581 = -1.0 * elem_adj_21573;
        double zm_lhs_adj_21582 = elem_adj_21573 + elem_adj_21574;
        double binop_x_adj_21583 = zt_lhs_20557 * binop_y_adj_21581;
        double zt_lhs_adj_21585 = binop_x_adj_21583 + binop_x_adj_21583;
        double binop_y_adj_21589 = 3.0 * elem_adj_21572;
        double binop_x_adj_21590 = zt_rhs_20551 * elem_adj_21572;
        double binop_y_adj_21591 = zt_lhs_20550 * elem_adj_21572;
        double zt_rhs_adj_21592 = elem_adj_21574 + binop_y_adj_21591;
        
        for (int64_t nest_i_21967 = 0; nest_i_21967 < (int64_t) 6; nest_i_21967++) {
            ((double *) mem_21834)[i_21694 * (int64_t) 6 + nest_i_21967] = 0.0;
        }
        ((double *) mem_21834)[i_21694 * (int64_t) 6] = binop_x_adj_21590;
        ((double *) mem_21834)[i_21694 * (int64_t) 6 + (int64_t) 1] = zt_rhs_adj_21592;
        ((double *) mem_21834)[i_21694 * (int64_t) 6 + (int64_t) 4] = binop_y_adj_21589;
        ((double *) mem_21834)[i_21694 * (int64_t) 6 + (int64_t) 2] = zm_lhs_adj_21582;
        ((double *) mem_21834)[i_21694 * (int64_t) 6 + (int64_t) 5] = zt_lhs_adj_21585;
        ((double *) mem_21834)[i_21694 * (int64_t) 6 + (int64_t) 3] = elem_adj_21574;
    }
    // src/sparse_jacobian_vjp.fut:45:5-53:27
    
    bool compressed_to_csr_vals_res_18730;
    int64_t compressed_to_csr_vals_res_18732;
    bool loop_while_18733;
    int64_t i_18735;
    
    loop_while_18733 = 1;
    i_18735 = (int64_t) 0;
    while (loop_while_18733) {
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool x_18736 = sle64((int64_t) 0, i_18735);
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool y_18737 = slt64(i_18735, csr_bipartite_from_pattern_res_18694);
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool bounds_check_18738 = x_18736 && y_18737;
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool index_certs_18739;
        
        if (!bounds_check_18738) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_18735, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_18694, "].", "-> #0  src/sparse_jacobian_vjp.fut:47:15-26\n   #1  src/sparse_jacobian_vjp.fut:119:18-126:67\n   #2  src/sparse_jacobian_vjp.fut:142:8-148:51\n   #3  test/test_sparse_jacobian_vjp.fut:80:15-113:79\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        int64_t s_18740 = ((int64_t *) ext_mem_21829.mem)[i_18735];
        
        // src/sparse_jacobian_vjp.fut:48:25-27
        
        int64_t e_18741 = add64((int64_t) 1, i_18735);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool x_18742 = sle64((int64_t) 0, e_18741);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool y_18743 = slt64(e_18741, csr_bipartite_from_pattern_res_18694);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool bounds_check_18744 = x_18742 && y_18743;
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool index_certs_18745;
        
        if (!bounds_check_18744) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_18741, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_18694, "].", "-> #0  src/sparse_jacobian_vjp.fut:48:15-28\n   #1  src/sparse_jacobian_vjp.fut:119:18-126:67\n   #2  src/sparse_jacobian_vjp.fut:142:8-148:51\n   #3  test/test_sparse_jacobian_vjp.fut:80:15-113:79\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        int64_t e_18746 = ((int64_t *) ext_mem_21829.mem)[e_18741];
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t j_m_i_18747 = sub64(e_18746, s_18740);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool empty_slice_18748 = j_m_i_18747 == (int64_t) 0;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t m_18749 = sub64(j_m_i_18747, (int64_t) 1);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t i_p_m_t_s_18750 = add64(s_18740, m_18749);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_18751 = sle64((int64_t) 0, i_p_m_t_s_18750);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_18752 = slt64(i_p_m_t_s_18750, csr_bipartite_from_pattern_res_18695);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool zzero_lte_i_18753 = sle64((int64_t) 0, s_18740);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool i_lte_j_18754 = sle64(s_18740, e_18746);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool y_18755 = i_p_m_t_s_leq_w_18752 && zzero_lte_i_18753;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool y_18756 = zzero_leq_i_p_m_t_s_18751 && y_18755;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool forwards_ok_18757 = i_lte_j_18754 && y_18756;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool ok_or_empty_18758 = empty_slice_18748 || forwards_ok_18757;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool index_certs_18759;
        
        if (!ok_or_empty_18758) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_18740, ":", (long long) e_18746, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_18695, "].", "-> #0  src/sparse_jacobian_vjp.fut:49:18-30\n   #1  src/sparse_jacobian_vjp.fut:119:18-126:67\n   #2  src/sparse_jacobian_vjp.fut:142:8-148:51\n   #3  test/test_sparse_jacobian_vjp.fut:80:15-113:79\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool y_18761 = slt64(i_18735, (int64_t) 4);
        
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool bounds_check_18762 = x_18736 && y_18761;
        
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool index_certs_18763;
        
        if (!bounds_check_18762) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_18735, "] out of bounds for array of shape [", (long long) (int64_t) 4, "].", "-> #0  src/sparse_jacobian_vjp.fut:50:16-29\n   #1  src/sparse_jacobian_vjp.fut:119:18-126:67\n   #2  src/sparse_jacobian_vjp.fut:142:8-148:51\n   #3  test/test_sparse_jacobian_vjp.fut:80:15-113:79\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        int64_t rc_18764 = ((int64_t *) mem_21820.mem)[i_18735];
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool y_18766 = slt64(rc_18764, num_colors_of_res_f_res_18706);
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool x_18765 = sle64((int64_t) 0, rc_18764);
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool bounds_check_18767 = x_18765 && y_18766;
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool index_certs_18768;
        
        if (!bounds_check_18767) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) rc_18764, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_18706, "].", "-> #0  src/sparse_jacobian_vjp.fut:51:28-34\n   #1  src/sparse_jacobian_vjp.fut:119:18-126:67\n   #2  src/sparse_jacobian_vjp.fut:142:8-148:51\n   #3  test/test_sparse_jacobian_vjp.fut:80:15-113:79\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:51:17-43
        for (int64_t i_21698 = 0; i_21698 < j_m_i_18747; i_21698++) {
            int64_t index_primexp_21771 = s_18740 + i_21698;
            int64_t eta_p_18770 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21771];
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool x_18771 = sle64((int64_t) 0, eta_p_18770);
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool y_18772 = slt64(eta_p_18770, (int64_t) 6);
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool bounds_check_18773 = x_18771 && y_18772;
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool index_certs_18774;
            
            if (!bounds_check_18773) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_18770, "] out of bounds for array of shape [", (long long) (int64_t) 6, "].", "-> #0  src/sparse_jacobian_vjp.fut:51:28-37\n   #1  src/sparse_jacobian_vjp.fut:119:18-126:67\n   #2  src/sparse_jacobian_vjp.fut:142:8-148:51\n   #3  test/test_sparse_jacobian_vjp.fut:80:15-113:79\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_vjp.fut:51:17-43
            
            double lifted_lambda_res_18775 = ((double *) mem_21834)[rc_18764 * (int64_t) 6 + eta_p_18770];
            
            ((double *) mem_21831)[s_18740 + i_21698] = lifted_lambda_res_18775;
        }
        // src/sparse_jacobian_vjp.fut:46:13-16
        
        bool loop_cond_18777 = slt64(e_18741, (int64_t) 4);
        bool loop_while_tmp_21968 = loop_cond_18777;
        int64_t i_tmp_21970 = e_18741;
        
        loop_while_18733 = loop_while_tmp_21968;
        i_18735 = i_tmp_21970;
    }
    compressed_to_csr_vals_res_18730 = loop_while_18733;
    compressed_to_csr_vals_res_18732 = i_18735;
    // src/dense_jacobian.fut:5:3-21
    if (mem_21865_cached_sizze_22111 < (int64_t) 32) {
        err = lexical_realloc(ctx, &mem_21865, &mem_21865_cached_sizze_22111, (int64_t) 32);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    if (mem_21866_cached_sizze_22112 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_21866, &mem_21866_cached_sizze_22112, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:68:29-47
    if (mem_21867_cached_sizze_22113 < (int64_t) 48) {
        err = lexical_realloc(ctx, &mem_21867, &mem_21867_cached_sizze_22113, (int64_t) 48);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:7:19-9:39
    
    bool defunc_0_reduce_res_21640;
    bool redout_21704 = 1;
    
    for (int64_t i_21705 = 0; i_21705 < (int64_t) 4; i_21705++) {
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        bool y_20813 = slt64(i_21705, csr_bipartite_from_pattern_res_18694);
        
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        bool index_certs_20815;
        
        if (!y_20813) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_21705, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_18694, "].", "-> #0  src/sparse_jacobian_vjp.fut:64:17-28\n   #1  src/sparse_jacobian_vjp.fut:142:18-149:40\n   #2  test/test_sparse_jacobian_vjp.fut:80:15-113:79\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        int64_t s_20816 = ((int64_t *) ext_mem_21829.mem)[i_21705];
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool zzero_lte_i_20829 = sle64((int64_t) 0, s_20816);
        
        // src/sparse_jacobian_vjp.fut:65:27-29
        
        int64_t e_20817 = add64((int64_t) 1, i_21705);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool y_20819 = slt64(e_20817, csr_bipartite_from_pattern_res_18694);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool x_20818 = sle64((int64_t) 0, e_20817);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool bounds_check_20820 = x_20818 && y_20819;
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool index_certs_20821;
        
        if (!bounds_check_20820) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_20817, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_18694, "].", "-> #0  src/sparse_jacobian_vjp.fut:65:17-30\n   #1  src/sparse_jacobian_vjp.fut:142:18-149:40\n   #2  test/test_sparse_jacobian_vjp.fut:80:15-113:79\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        int64_t e_20822 = ((int64_t *) ext_mem_21829.mem)[e_20817];
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t j_m_i_20823 = sub64(e_20822, s_20816);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t m_20825 = sub64(j_m_i_20823, (int64_t) 1);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t i_p_m_t_s_20826 = add64(s_20816, m_20825);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool i_p_m_t_s_leq_w_20828 = slt64(i_p_m_t_s_20826, csr_bipartite_from_pattern_res_18695);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool y_20837 = i_p_m_t_s_leq_w_20828 && zzero_lte_i_20829;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool zzero_leq_i_p_m_t_s_20827 = sle64((int64_t) 0, i_p_m_t_s_20826);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool y_20838 = zzero_leq_i_p_m_t_s_20827 && y_20837;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool i_lte_j_20830 = sle64(s_20816, e_20822);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool forwards_ok_20839 = i_lte_j_20830 && y_20838;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool empty_slice_20824 = j_m_i_20823 == (int64_t) 0;
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool ok_or_empty_20840 = empty_slice_20824 || forwards_ok_20839;
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool index_certs_20841;
        
        if (!ok_or_empty_20840) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_20816, ":", (long long) e_20822, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_18695, "].", "-> #0  src/sparse_jacobian_vjp.fut:67:20-29\n   #1  src/sparse_jacobian_vjp.fut:142:18-149:40\n   #2  test/test_sparse_jacobian_vjp.fut:80:15-113:79\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/dense_jacobian.fut:5:3-21
        for (int64_t nest_i_21973 = 0; nest_i_21973 < (int64_t) 4; nest_i_21973++) {
            ((double *) mem_21865)[nest_i_21973] = 0.0;
        }
        // src/dense_jacobian.fut:5:3-39
        ((double *) mem_21865)[i_21705] = 1.0;
        
        double elem_adj_21601 = ((double *) mem_21865)[(int64_t) 0];
        double elem_adj_21602 = ((double *) mem_21865)[(int64_t) 1];
        double elem_adj_21603 = ((double *) mem_21865)[(int64_t) 2];
        double binop_y_adj_21610 = -1.0 * elem_adj_21602;
        double zm_lhs_adj_21611 = elem_adj_21602 + elem_adj_21603;
        double binop_x_adj_21612 = zt_lhs_20557 * binop_y_adj_21610;
        double zt_lhs_adj_21614 = binop_x_adj_21612 + binop_x_adj_21612;
        double binop_y_adj_21618 = 3.0 * elem_adj_21601;
        double binop_x_adj_21619 = zt_rhs_20551 * elem_adj_21601;
        double binop_y_adj_21620 = zt_lhs_20550 * elem_adj_21601;
        double zt_rhs_adj_21621 = elem_adj_21603 + binop_y_adj_21620;
        
        for (int64_t nest_i_21974 = 0; nest_i_21974 < (int64_t) 6; nest_i_21974++) {
            ((double *) mem_21866)[nest_i_21974] = 0.0;
        }
        ((double *) mem_21866)[(int64_t) 0] = binop_x_adj_21619;
        ((double *) mem_21866)[(int64_t) 1] = zt_rhs_adj_21621;
        ((double *) mem_21866)[(int64_t) 4] = binop_y_adj_21618;
        ((double *) mem_21866)[(int64_t) 2] = zm_lhs_adj_21611;
        ((double *) mem_21866)[(int64_t) 5] = zt_lhs_adj_21614;
        ((double *) mem_21866)[(int64_t) 3] = elem_adj_21603;
        // src/sparse_jacobian_vjp.fut:68:29-47
        for (int64_t nest_i_21975 = 0; nest_i_21975 < (int64_t) 6; nest_i_21975++) {
            ((double *) mem_21867)[nest_i_21975] = 0.0;
        }
        // src/sparse_jacobian_vjp.fut:69:12-32
        
        bool acc_cert_20845;
        
        // src/sparse_jacobian_vjp.fut:69:12-32
        for (int64_t i_21701 = 0; i_21701 < j_m_i_20823; i_21701++) {
            int64_t index_primexp_21768 = s_20816 + i_21701;
            int64_t v_20849 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21768];
            double v_20850 = ((double *) mem_21831)[index_primexp_21768];
            
            // src/sparse_jacobian_vjp.fut:69:12-32
            // UpdateAcc
            if (sle64((int64_t) 0, v_20849) && slt64(v_20849, (int64_t) 6)) {
                ((double *) mem_21867)[v_20849] = v_20850;
            }
        }
        // test/test_sparse_jacobian_vjp.fut:8:5-66
        
        bool defunc_0_reduce_res_21637;
        bool redout_21702 = 1;
        
        for (int64_t i_21703 = 0; i_21703 < (int64_t) 6; i_21703++) {
            bool eta_p_20873 = ((bool *) mem_21819.mem)[i_21705 * (int64_t) 6 + i_21703];
            double eta_p_20874 = ((double *) mem_21866)[i_21703];
            double eta_p_20875 = ((double *) mem_21867)[i_21703];
            
            // test/test_sparse_jacobian_vjp.fut:13:25-48
            
            double lifted_lambda_res_20876;
            
            if (eta_p_20873) {
                lifted_lambda_res_20876 = eta_p_20874;
            } else {
                lifted_lambda_res_20876 = 0.0;
            }
            // test/test_sparse_jacobian_vjp.fut:8:48-51
            
            double abs_arg0_20878 = eta_p_20875 - lifted_lambda_res_20876;
            
            // test/test_sparse_jacobian_vjp.fut:8:37-51
            
            double abs_res_20879 = fabs64(abs_arg0_20878);
            
            // test/test_sparse_jacobian_vjp.fut:8:53-59
            
            bool lifted_lambda_res_20880 = abs_res_20879 <= 1.0e-9;
            
            // test/test_sparse_jacobian_vjp.fut:8:5-66
            
            bool x_20863 = lifted_lambda_res_20880 && redout_21702;
            bool redout_tmp_21977 = x_20863;
            
            redout_21702 = redout_tmp_21977;
        }
        defunc_0_reduce_res_21637 = redout_21702;
        // test/test_sparse_jacobian_vjp.fut:9:6-39
        
        bool x_20645 = defunc_0_reduce_res_21637 && redout_21704;
        bool redout_tmp_21972 = x_20645;
        
        redout_21704 = redout_tmp_21972;
    }
    defunc_0_reduce_res_21640 = redout_21704;
    if (memblock_unref(ctx, &ext_mem_21828, "ext_mem_21828") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21829, "ext_mem_21829") != 0)
        return 1;
    prim_out_21962 = defunc_0_reduce_res_21640;
    *out_prim_out_22107 = prim_out_21962;
    
  cleanup:
    {
        free(mem_21831);
        free(mem_21834);
        free(mem_21839);
        free(mem_21865);
        free(mem_21866);
        free(mem_21867);
        if (memblock_unref(ctx, &ext_mem_21828, "ext_mem_21828") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21829, "ext_mem_21829") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_vjp_ex5_matches_dense(struct futhark_context *ctx, bool *out_prim_out_22114, struct memblock x_mem_21827)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_21831_cached_sizze_22115 = 0;
    unsigned char *mem_21831 = NULL;
    int64_t mem_21832_cached_sizze_22116 = 0;
    unsigned char *mem_21832 = NULL;
    int64_t mem_21841_cached_sizze_22117 = 0;
    unsigned char *mem_21841 = NULL;
    int64_t mem_21853_cached_sizze_22118 = 0;
    unsigned char *mem_21853 = NULL;
    int64_t mem_21857_cached_sizze_22119 = 0;
    unsigned char *mem_21857 = NULL;
    int64_t mem_21874_cached_sizze_22120 = 0;
    unsigned char *mem_21874 = NULL;
    int64_t mem_21876_cached_sizze_22121 = 0;
    unsigned char *mem_21876 = NULL;
    int64_t mem_21877_cached_sizze_22122 = 0;
    unsigned char *mem_21877 = NULL;
    int64_t mem_21897_cached_sizze_22123 = 0;
    unsigned char *mem_21897 = NULL;
    int64_t mem_21899_cached_sizze_22124 = 0;
    unsigned char *mem_21899 = NULL;
    int64_t mem_21901_cached_sizze_22125 = 0;
    unsigned char *mem_21901 = NULL;
    int64_t mem_21909_cached_sizze_22126 = 0;
    unsigned char *mem_21909 = NULL;
    int64_t mem_21920_cached_sizze_22127 = 0;
    unsigned char *mem_21920 = NULL;
    int64_t mem_21925_cached_sizze_22128 = 0;
    unsigned char *mem_21925 = NULL;
    int64_t mem_21951_cached_sizze_22129 = 0;
    unsigned char *mem_21951 = NULL;
    int64_t mem_21952_cached_sizze_22130 = 0;
    unsigned char *mem_21952 = NULL;
    int64_t mem_21953_cached_sizze_22131 = 0;
    unsigned char *mem_21953 = NULL;
    struct memblock mem_param_tmp_21967;
    
    mem_param_tmp_21967.references = NULL;
    
    struct memblock mem_21911;
    
    mem_21911.references = NULL;
    
    struct memblock mem_param_21851;
    
    mem_param_21851.references = NULL;
    
    struct memblock ext_mem_21916;
    
    ext_mem_21916.references = NULL;
    
    struct memblock ext_mem_21844;
    
    ext_mem_21844.references = NULL;
    
    struct memblock ext_mem_21845;
    
    ext_mem_21845.references = NULL;
    
    struct memblock mem_21842;
    
    mem_21842.references = NULL;
    
    struct memblock mem_21840;
    
    mem_21840.references = NULL;
    
    struct memblock ext_mem_21828;
    
    ext_mem_21828.references = NULL;
    
    struct memblock ext_mem_21829;
    
    ext_mem_21829.references = NULL;
    
    struct memblock mem_21809 = ctx->constants->mem_21809;
    struct memblock mem_21812 = ctx->constants->mem_21812;
    struct memblock mem_21814 = ctx->constants->mem_21814;
    struct memblock mem_21819 = ctx->constants->mem_21819;
    struct memblock mem_21820 = ctx->constants->mem_21820;
    struct memblock mem_21826 = ctx->constants->mem_21826;
    bool prim_out_21962;
    
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_20110;
    int64_t csr_bipartite_from_pattern_res_20111;
    
    if (futrts_csr_rows_from_pattern_7010(ctx, &ext_mem_21829, &ext_mem_21828, &csr_bipartite_from_pattern_res_20110, &csr_bipartite_from_pattern_res_20111, mem_21826, (int64_t) 5, (int64_t) 5) != 0) {
        err = 1;
        goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:42:15-35
    
    int64_t bytes_21830 = (int64_t) 8 * csr_bipartite_from_pattern_res_20111;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (mem_21832_cached_sizze_22116 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_21832, &mem_21832_cached_sizze_22116, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t discard_21691;
    int64_t scanacc_21687 = (int64_t) 0;
    
    for (int64_t i_21689 = 0; i_21689 < (int64_t) 5; i_21689++) {
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t defunc_0_op_res_20126 = add64((int64_t) 1, scanacc_21687);
        
        ((int64_t *) mem_21832)[i_21689] = defunc_0_op_res_20126;
        
        int64_t scanacc_tmp_21963 = defunc_0_op_res_20126;
        
        scanacc_21687 = scanacc_tmp_21963;
    }
    discard_21691 = scanacc_21687;
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t x_20136 = ((int64_t *) mem_21832)[(int64_t) 4];
    
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    bool loop_cond_20150 = slt64((int64_t) 0, x_20136);
    
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t bytes_21839 = (int64_t) 8 * x_20136;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (memblock_alloc(ctx, &mem_21840, bytes_21839, "mem_21840")) {
        err = 1;
        goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    bool acc_cert_20718;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    for (int64_t i_21693 = 0; i_21693 < (int64_t) 5; i_21693++) {
        int64_t eta_p_20730 = ((int64_t *) mem_21832)[i_21693];
        
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t lifted_lambda_res_t_res_20733 = sub64(eta_p_20730, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:194:20-45
        // UpdateAcc
        if (sle64((int64_t) 0, lifted_lambda_res_t_res_20733) && slt64(lifted_lambda_res_t_res_20733, x_20136)) {
            ((int64_t *) mem_21840.mem)[lifted_lambda_res_t_res_20733] = i_21693;
        }
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    if (mem_21841_cached_sizze_22117 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_21841, &mem_21841_cached_sizze_22117, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    for (int64_t nest_i_21966 = 0; nest_i_21966 < (int64_t) 5; nest_i_21966++) {
        ((int64_t *) mem_21841)[nest_i_21966] = (int64_t) -1;
    }
    // src/pattern_csr.fut:25:3-39
    if (memblock_alloc(ctx, &mem_21842, (int64_t) 25, "mem_21842")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:25:3-39
    // src/pattern_csr.fut:25:3-39
    lmad_copy_1b(ctx, 2, (uint8_t *) mem_21842.mem, (int64_t) 0, (int64_t []) {(int64_t) 5, (int64_t) 1}, (uint8_t *) mem_21826.mem, (int64_t) 0, (int64_t []) {(int64_t) 1, (int64_t) 5}, (int64_t []) {(int64_t) 5, (int64_t) 5});
    // src/pattern_csr.fut:25:3-39
    
    int64_t csr_cols_from_pattern_res_20115;
    int64_t csr_cols_from_pattern_res_20116;
    
    if (futrts_csr_rows_from_pattern_7010(ctx, &ext_mem_21845, &ext_mem_21844, &csr_cols_from_pattern_res_20115, &csr_cols_from_pattern_res_20116, mem_21842, (int64_t) 5, (int64_t) 5) != 0) {
        err = 1;
        goto cleanup;
    }
    if (memblock_unref(ctx, &mem_21842, "mem_21842") != 0)
        return 1;
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    int64_t vv_color_side_order_res_20151;
    bool vv_color_side_order_res_20152;
    int64_t vv_color_side_order_res_20155;
    int64_t loop_dz2081Uz2083U_20156;
    bool loop_while_20157;
    int64_t color_bound_20160;
    
    if (memblock_set(ctx, &mem_param_21851, &mem_21840, "mem_21840") != 0)
        return 1;
    loop_dz2081Uz2083U_20156 = x_20136;
    loop_while_20157 = loop_cond_20150;
    color_bound_20160 = (int64_t) 1;
    while (loop_while_20157) {
        // src/bgpc_vv_coloring.fut:62:5-64:15
        
        bool cond_20161 = slt64((int64_t) 0, color_bound_20160);
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t bytes_21852 = (int64_t) 8 * loop_dz2081Uz2083U_20156;
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        if (mem_21853_cached_sizze_22118 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21853, &mem_21853_cached_sizze_22118, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:77:24-51
        if (mem_21857_cached_sizze_22119 < color_bound_20160) {
            err = lexical_realloc(ctx, &mem_21857, &mem_21857_cached_sizze_22119, color_bound_20160);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t defunc_0_reduce_res_21672;
        int64_t redout_21695 = (int64_t) -1;
        
        for (int64_t i_21697 = 0; i_21697 < loop_dz2081Uz2083U_20156; i_21697++) {
            int64_t eta_p_21340 = ((int64_t *) mem_param_21851.mem)[i_21697];
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool x_21342 = sle64((int64_t) 0, eta_p_21340);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool y_21343 = slt64(eta_p_21340, csr_bipartite_from_pattern_res_20110);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool bounds_check_21344 = x_21342 && y_21343;
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool index_certs_21345;
            
            if (!bounds_check_21344) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_21340, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20110, "].", "-> #0  src/bgpc_vv_coloring.fut:46:44-56\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:44-56
            
            int64_t seen_final_21346 = ((int64_t *) ext_mem_21829.mem)[eta_p_21340];
            
            // src/bgpc_vv_coloring.fut:46:70-72
            
            int64_t seen_final_21347 = add64((int64_t) 1, eta_p_21340);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool x_21348 = sle64((int64_t) 0, seen_final_21347);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool y_21349 = slt64(seen_final_21347, csr_bipartite_from_pattern_res_20110);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool bounds_check_21350 = x_21348 && y_21349;
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool index_certs_21351;
            
            if (!bounds_check_21350) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_21347, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20110, "].", "-> #0  src/bgpc_vv_coloring.fut:46:59-73\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:59-73
            
            int64_t seen_final_21352 = ((int64_t *) ext_mem_21829.mem)[seen_final_21347];
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t j_m_i_21353 = sub64(seen_final_21352, seen_final_21346);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool empty_slice_21354 = j_m_i_21353 == (int64_t) 0;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t m_21355 = sub64(j_m_i_21353, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t i_p_m_t_s_21356 = add64(seen_final_21346, m_21355);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_leq_i_p_m_t_s_21357 = sle64((int64_t) 0, i_p_m_t_s_21356);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_p_m_t_s_leq_w_21358 = slt64(i_p_m_t_s_21356, csr_bipartite_from_pattern_res_20111);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_lte_i_21359 = sle64((int64_t) 0, seen_final_21346);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_lte_j_21360 = sle64(seen_final_21346, seen_final_21352);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_21361 = i_p_m_t_s_leq_w_21358 && zzero_lte_i_21359;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_21362 = zzero_leq_i_p_m_t_s_21357 && y_21361;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool forwards_ok_21363 = i_lte_j_21360 && y_21362;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool ok_or_empty_21364 = empty_slice_21354 || forwards_ok_21363;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool index_certs_21365;
            
            if (!ok_or_empty_21364) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_21346, ":", (long long) seen_final_21352, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20111, "].", "-> #0  src/bgpc_vv_coloring.fut:46:5-54:19\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:77:24-51
            for (int64_t nest_i_21975 = 0; nest_i_21975 < color_bound_20160; nest_i_21975++) {
                ((bool *) mem_21857)[nest_i_21975] = 0;
            }
            // src/bgpc_vv_coloring.fut:46:5-54:19
            for (int64_t i_21367 = 0; i_21367 < j_m_i_21353; i_21367++) {
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t index_primexp_21369 = seen_final_21346 + i_21367;
                
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t v_21370 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21369];
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool x_21371 = sle64((int64_t) 0, v_21370);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool y_21372 = slt64(v_21370, csr_cols_from_pattern_res_20115);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool bounds_check_21373 = x_21371 && y_21372;
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool index_certs_21374;
                
                if (!bounds_check_21373) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_21370, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20115, "].", "-> #0  src/bgpc_vv_coloring.fut:49:26-37\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:26-37
                
                int64_t seen_acczq_21375 = ((int64_t *) ext_mem_21845.mem)[v_21370];
                
                // src/bgpc_vv_coloring.fut:49:50-52
                
                int64_t seen_acczq_21376 = add64((int64_t) 1, v_21370);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool x_21377 = sle64((int64_t) 0, seen_acczq_21376);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool y_21378 = slt64(seen_acczq_21376, csr_cols_from_pattern_res_20115);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool bounds_check_21379 = x_21377 && y_21378;
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool index_certs_21380;
                
                if (!bounds_check_21379) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_21376, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20115, "].", "-> #0  src/bgpc_vv_coloring.fut:49:40-53\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:40-53
                
                int64_t seen_acczq_21381 = ((int64_t *) ext_mem_21845.mem)[seen_acczq_21376];
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t j_m_i_21382 = sub64(seen_acczq_21381, seen_acczq_21375);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool empty_slice_21383 = j_m_i_21382 == (int64_t) 0;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t m_21384 = sub64(j_m_i_21382, (int64_t) 1);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t i_p_m_t_s_21385 = add64(seen_acczq_21375, m_21384);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_leq_i_p_m_t_s_21386 = sle64((int64_t) 0, i_p_m_t_s_21385);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_p_m_t_s_leq_w_21387 = slt64(i_p_m_t_s_21385, csr_cols_from_pattern_res_20116);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_lte_i_21388 = sle64((int64_t) 0, seen_acczq_21375);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_lte_j_21389 = sle64(seen_acczq_21375, seen_acczq_21381);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_21390 = i_p_m_t_s_leq_w_21387 && zzero_lte_i_21388;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_21391 = zzero_leq_i_p_m_t_s_21386 && y_21390;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool forwards_ok_21392 = i_lte_j_21389 && y_21391;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool ok_or_empty_21393 = empty_slice_21383 || forwards_ok_21392;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool index_certs_21394;
                
                if (!ok_or_empty_21393) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_21375, ":", (long long) seen_acczq_21381, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20116, "].", "-> #0  src/bgpc_vv_coloring.fut:48:9-53:33\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:48:9-53:33
                for (int64_t i_21396 = 0; i_21396 < j_m_i_21382; i_21396++) {
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t index_primexp_21398 = seen_acczq_21375 + i_21396;
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t u_21399 = ((int64_t *) ext_mem_21844.mem)[index_primexp_21398];
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool x_21400 = sle64((int64_t) 0, u_21399);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool y_21401 = slt64(u_21399, (int64_t) 5);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool bounds_check_21402 = x_21400 && y_21401;
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool index_certs_21403;
                    
                    if (!bounds_check_21402) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_21399, "] out of bounds for array of shape [", (long long) (int64_t) 5, "].", "-> #0  src/bgpc_vv_coloring.fut:50:20-29\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    int64_t cu_21404 = ((int64_t *) mem_21841)[u_21399];
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_21405 = u_21399 == eta_p_21340;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_21406 = !cond_21405;
                    
                    // src/bgpc_vv_coloring.fut:51:30-37
                    
                    bool cond_t_res_21407 = sle64((int64_t) 0, cu_21404);
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    bool x_21408 = cond_21406 && cond_t_res_21407;
                    
                    // src/bgpc_vv_coloring.fut:51:44-67
                    
                    bool cond_t_res_21409 = slt64(cu_21404, color_bound_20160);
                    
                    // src/bgpc_vv_coloring.fut:94:14-103:69
                    
                    bool x_21410 = x_21408 && cond_t_res_21409;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    if (x_21410) {
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool bounds_check_21636 = cond_t_res_21407 && cond_t_res_21409;
                        
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool index_certs_21637;
                        
                        if (!bounds_check_21636) {
                            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) cu_21404, "] out of bounds for array of shape [", (long long) color_bound_20160, "].", "-> #0  src/bgpc_vv_coloring.fut:52:19-50\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                            err = FUTHARK_PROGRAM_ERROR;
                            goto cleanup;
                        }
                        // src/bgpc_vv_coloring.fut:52:19-50
                        ((bool *) mem_21857)[cu_21404] = 1;
                    }
                }
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool loop_cond_21415;
            
            if (cond_20161) {
                // src/bgpc_vv_coloring.fut:63:30-37
                
                bool loop_cond_t_res_21639 = ((bool *) mem_21857)[(int64_t) 0];
                
                loop_cond_21415 = loop_cond_t_res_21639;
            } else {
                loop_cond_21415 = 0;
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool c_final_21417;
            int64_t c_final_21418;
            bool loop_while_21419;
            int64_t c_21420;
            
            loop_while_21419 = loop_cond_21415;
            c_21420 = (int64_t) 0;
            while (loop_while_21419) {
                // src/bgpc_vv_coloring.fut:64:9-15
                
                int64_t loopres_21421 = add64((int64_t) 1, c_21420);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool cond_21422 = slt64(loopres_21421, color_bound_20160);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool loop_cond_21423;
                
                if (cond_21422) {
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool x_21640 = sle64((int64_t) 0, loopres_21421);
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool bounds_check_21641 = cond_21422 && x_21640;
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool index_certs_21642;
                    
                    if (!bounds_check_21641) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) loopres_21421, "] out of bounds for array of shape [", (long long) color_bound_20160, "].", "-> #0  src/bgpc_vv_coloring.fut:63:30-37\n   #1  src/bgpc_vv_coloring.fut:81:6-28\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool loop_cond_t_res_21643 = ((bool *) mem_21857)[loopres_21421];
                    
                    loop_cond_21423 = loop_cond_t_res_21643;
                } else {
                    loop_cond_21423 = 0;
                }
                
                bool loop_while_tmp_21978 = loop_cond_21423;
                int64_t c_tmp_21979 = loopres_21421;
                
                loop_while_21419 = loop_while_tmp_21978;
                c_21420 = c_tmp_21979;
            }
            c_final_21417 = loop_while_21419;
            c_final_21418 = c_21420;
            // src/bgpc_vv_coloring.fut:106:25-32
            
            int64_t max_res_20255 = smax64(c_final_21418, redout_21695);
            
            ((int64_t *) mem_21853)[i_21697] = c_final_21418;
            
            int64_t redout_tmp_21973 = max_res_20255;
            
            redout_21695 = redout_tmp_21973;
        }
        defunc_0_reduce_res_21672 = redout_21695;
        // src/bgpc_vv_coloring.fut:113:17-43
        
        bool acc_cert_20261;
        
        // src/bgpc_vv_coloring.fut:113:17-43
        for (int64_t i_21700 = 0; i_21700 < loop_dz2081Uz2083U_20156; i_21700++) {
            int64_t v_20265 = ((int64_t *) mem_param_21851.mem)[i_21700];
            int64_t v_20266 = ((int64_t *) mem_21853)[i_21700];
            
            // src/bgpc_vv_coloring.fut:113:17-43
            // UpdateAcc
            if (sle64((int64_t) 0, v_20265) && slt64(v_20265, (int64_t) 5)) {
                ((int64_t *) mem_21841)[v_20265] = v_20266;
            }
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21874_cached_sizze_22120 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21874, &mem_21874_cached_sizze_22120, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21876_cached_sizze_22121 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21876, &mem_21876_cached_sizze_22121, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21877_cached_sizze_22122 < loop_dz2081Uz2083U_20156) {
            err = lexical_realloc(ctx, &mem_21877, &mem_21877_cached_sizze_22122, loop_dz2081Uz2083U_20156);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t discard_21712;
        int64_t scanacc_21704 = (int64_t) 0;
        
        for (int64_t i_21708 = 0; i_21708 < loop_dz2081Uz2083U_20156; i_21708++) {
            int64_t eta_p_21264 = ((int64_t *) mem_param_21851.mem)[i_21708];
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool x_21265 = sle64((int64_t) 0, eta_p_21264);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool y_21266 = slt64(eta_p_21264, csr_bipartite_from_pattern_res_20110);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool bounds_check_21267 = x_21265 && y_21266;
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool index_certs_21268;
            
            if (!bounds_check_21267) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_21264, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20110, "].", "-> #0  src/bgpc_vv_coloring.fut:126:15-27\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:127:26-28
            
            int64_t k_end_21270 = add64((int64_t) 1, eta_p_21264);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool x_21271 = sle64((int64_t) 0, k_end_21270);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool y_21272 = slt64(k_end_21270, csr_bipartite_from_pattern_res_20110);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool bounds_check_21273 = x_21271 && y_21272;
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool index_certs_21274;
            
            if (!bounds_check_21273) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_end_21270, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20110, "].", "-> #0  src/bgpc_vv_coloring.fut:127:15-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:126:15-27
            
            int64_t k0_21269 = ((int64_t *) ext_mem_21829.mem)[eta_p_21264];
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            int64_t k_end_21275 = ((int64_t *) ext_mem_21829.mem)[k_end_21270];
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool cond_21276 = slt64(k0_21269, k_end_21275);
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool loses_conflict_vertex_res_21277;
            bool loses_conflict_vertex_res_21278;
            int64_t loses_conflict_vertex_res_21279;
            bool loop_while_21280;
            bool lost_21281;
            int64_t k_21282;
            
            loop_while_21280 = cond_21276;
            lost_21281 = 0;
            k_21282 = k0_21269;
            while (loop_while_21280) {
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool x_21283 = sle64((int64_t) 0, k_21282);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool y_21284 = slt64(k_21282, csr_bipartite_from_pattern_res_20111);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool bounds_check_21285 = x_21283 && y_21284;
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool index_certs_21286;
                
                if (!bounds_check_21285) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_21282, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20111, "].", "-> #0  src/bgpc_vv_coloring.fut:132:15-26\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:132:15-26
                
                int64_t v_21287 = ((int64_t *) ext_mem_21828.mem)[k_21282];
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool x_21288 = sle64((int64_t) 0, v_21287);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool y_21289 = slt64(v_21287, csr_cols_from_pattern_res_20115);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool bounds_check_21290 = x_21288 && y_21289;
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool index_certs_21291;
                
                if (!bounds_check_21290) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_21287, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20115, "].", "-> #0  src/bgpc_vv_coloring.fut:134:19-30\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:135:29-31
                
                int64_t t_end_21293 = add64((int64_t) 1, v_21287);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool x_21294 = sle64((int64_t) 0, t_end_21293);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool y_21295 = slt64(t_end_21293, csr_cols_from_pattern_res_20115);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool bounds_check_21296 = x_21294 && y_21295;
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool index_certs_21297;
                
                if (!bounds_check_21296) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_end_21293, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20115, "].", "-> #0  src/bgpc_vv_coloring.fut:135:19-32\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:134:19-30
                
                int64_t t0_21292 = ((int64_t *) ext_mem_21845.mem)[v_21287];
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                int64_t t_end_21298 = ((int64_t *) ext_mem_21845.mem)[t_end_21293];
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool cond_21299 = slt64(t0_21292, t_end_21298);
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool loopres_21300;
                bool loopres_21301;
                int64_t loopres_21302;
                bool loop_while_21303;
                bool lost_in_net_21304;
                int64_t t_21305;
                
                loop_while_21303 = cond_21299;
                lost_in_net_21304 = 0;
                t_21305 = t0_21292;
                while (loop_while_21303) {
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool x_21306 = sle64((int64_t) 0, t_21305);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool y_21307 = slt64(t_21305, csr_cols_from_pattern_res_20116);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool bounds_check_21308 = x_21306 && y_21307;
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool index_certs_21309;
                    
                    if (!bounds_check_21308) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_21305, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20116, "].", "-> #0  src/bgpc_vv_coloring.fut:140:19-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    int64_t u_21310 = ((int64_t *) ext_mem_21844.mem)[t_21305];
                    
                    // src/bgpc_vv_coloring.fut:142:13-71
                    
                    bool lost_in_netzq_21311;
                    
                    if (lost_in_net_21304) {
                        lost_in_netzq_21311 = 1;
                    } else {
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_21312 = u_21310 == eta_p_21264;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_21313 = !cond_21312;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_21314;
                        
                        if (cond_21313) {
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool x_21645 = sle64((int64_t) 0, u_21310);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool y_21646 = slt64(u_21310, (int64_t) 5);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool bounds_check_21647 = x_21645 && y_21646;
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool index_certs_21648;
                            
                            if (!bounds_check_21647) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_21310, "] out of bounds for array of shape [", (long long) (int64_t) 5, "].", "-> #0  src/bgpc_vv_coloring.fut:142:39-48\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool y_21650 = slt64(eta_p_21264, (int64_t) 5);
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool bounds_check_21651 = x_21265 && y_21650;
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool index_certs_21652;
                            
                            if (!bounds_check_21651) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_21264, "] out of bounds for array of shape [", (long long) (int64_t) 5, "].", "-> #0  src/bgpc_vv_coloring.fut:142:52-61\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            int64_t zeze_lhs_21649 = ((int64_t *) mem_21841)[u_21310];
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            int64_t zeze_rhs_21653 = ((int64_t *) mem_21841)[eta_p_21264];
                            
                            // src/bgpc_vv_coloring.fut:142:49-61
                            
                            bool cond_t_res_21654 = zeze_lhs_21649 == zeze_rhs_21653;
                            
                            cond_21314 = cond_t_res_21654;
                        } else {
                            cond_21314 = 0;
                        }
                        // src/bgpc_vv_coloring.fut:142:67-70
                        
                        bool lost_in_netzq_f_res_t_res_21325 = slt64(u_21310, eta_p_21264);
                        
                        // src/bgpc_vv_coloring.fut:160:14-168:57
                        
                        bool x_21326 = cond_21314 && lost_in_netzq_f_res_t_res_21325;
                        
                        lost_in_netzq_21311 = x_21326;
                    }
                    // src/bgpc_vv_coloring.fut:143:31-37
                    
                    int64_t tmp_21327 = add64((int64_t) 1, t_21305);
                    
                    // src/bgpc_vv_coloring.fut:139:15-43
                    
                    bool cond_21328 = slt64(tmp_21327, t_end_21298);
                    
                    // src/bgpc_vv_coloring.fut:139:28-43
                    
                    bool not_res_21329 = !lost_in_netzq_21311;
                    
                    // src/bgpc_vv_coloring.fut:166:5-169:10
                    
                    bool x_21330 = cond_21328 && not_res_21329;
                    bool loop_while_tmp_21988 = x_21330;
                    bool lost_in_net_tmp_21989 = lost_in_netzq_21311;
                    int64_t t_tmp_21990 = tmp_21327;
                    
                    loop_while_21303 = loop_while_tmp_21988;
                    lost_in_net_21304 = lost_in_net_tmp_21989;
                    t_21305 = t_tmp_21990;
                }
                loopres_21300 = loop_while_21303;
                loopres_21301 = lost_in_net_21304;
                loopres_21302 = t_21305;
                // src/bgpc_vv_coloring.fut:160:14-168:57
                
                bool lostzq_21331 = lost_21281 || loopres_21301;
                
                // src/bgpc_vv_coloring.fut:146:20-26
                
                int64_t tmp_21332 = add64((int64_t) 1, k_21282);
                
                // src/bgpc_vv_coloring.fut:131:11-32
                
                bool cond_21333 = slt64(tmp_21332, k_end_21275);
                
                // src/bgpc_vv_coloring.fut:131:24-32
                
                bool not_res_21334 = !lostzq_21331;
                
                // src/bgpc_vv_coloring.fut:166:5-169:10
                
                bool x_21335 = cond_21333 && not_res_21334;
                bool loop_while_tmp_21985 = x_21335;
                bool lost_tmp_21986 = lostzq_21331;
                int64_t k_tmp_21987 = tmp_21332;
                
                loop_while_21280 = loop_while_tmp_21985;
                lost_21281 = lost_tmp_21986;
                k_21282 = k_tmp_21987;
            }
            loses_conflict_vertex_res_21277 = loop_while_21280;
            loses_conflict_vertex_res_21278 = lost_21281;
            loses_conflict_vertex_res_21279 = k_21282;
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_f_res_21337 = btoi_bool_i64(loses_conflict_vertex_res_21278);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_op_res_20367 = add64(defunc_0_f_res_21337, scanacc_21704);
            
            ((int64_t *) mem_21874)[i_21708] = defunc_0_op_res_20367;
            ((int64_t *) mem_21876)[i_21708] = defunc_0_f_res_21337;
            ((bool *) mem_21877)[i_21708] = loses_conflict_vertex_res_21278;
            
            int64_t scanacc_tmp_21981 = defunc_0_op_res_20367;
            
            scanacc_21704 = scanacc_tmp_21981;
        }
        discard_21712 = scanacc_21704;
        // src/bgpc_vv_coloring.fut:23:25-28
        
        int64_t tmp_20351 = sub64(loop_dz2081Uz2083U_20156, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool y_20353 = slt64(tmp_20351, loop_dz2081Uz2083U_20156);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool x_20352 = sle64((int64_t) 0, tmp_20351);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool bounds_check_20354 = x_20352 && y_20353;
        
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        bool cond_20349 = loop_dz2081Uz2083U_20156 == (int64_t) 0;
        
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool protect_assert_disj_20355 = cond_20349 || bounds_check_20354;
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool index_certs_20356;
        
        if (!protect_assert_disj_20355) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_20351, "] out of bounds for array of shape [", (long long) loop_dz2081Uz2083U_20156, "].", "-> #0  src/bgpc_vv_coloring.fut:23:10-29\n   #1  src/bgpc_vv_coloring.fut:171:23-55\n   #2  src/bgpc_vv_coloring.fut:186:14-205:56\n   #3  src/bgpc_vv_coloring.fut:220:14-223:64\n   #4  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #5  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #6  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #7  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool x_20350 = !cond_20349;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_f_res_20368;
        
        if (x_20350) {
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t x_21655 = ((int64_t *) mem_21874)[tmp_20351];
            
            m_f_res_20368 = x_21655;
        } else {
            m_f_res_20368 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_20370;
        
        if (cond_20349) {
            m_20370 = (int64_t) 0;
        } else {
            m_20370 = m_f_res_20368;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t bytes_21896 = (int64_t) 8 * m_20370;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21897_cached_sizze_22123 < bytes_21896) {
            err = lexical_realloc(ctx, &mem_21897, &mem_21897_cached_sizze_22123, bytes_21896);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21899_cached_sizze_22124 < bytes_21896) {
            err = lexical_realloc(ctx, &mem_21899, &mem_21899_cached_sizze_22124, bytes_21896);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:18:24-27:44
        if (mem_21901_cached_sizze_22125 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21901, &mem_21901_cached_sizze_22125, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21909_cached_sizze_22126 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21909, &mem_21909_cached_sizze_22126, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        bool acc_cert_21137;
        bool acc_cert_21138;
        
        // src/bgpc_vv_coloring.fut:18:24-27:44
        
        int64_t inpacc_21663;
        int64_t inpacc_21201 = (int64_t) 0;
        
        for (int64_t i_21739 = 0; i_21739 < loop_dz2081Uz2083U_20156; i_21739++) {
            bool eta_p_21778 = ((bool *) mem_21877)[i_21739];
            
            // src/bgpc_vv_coloring.fut:18:28-39
            
            int64_t bool_to_i64_res_21779 = btoi_bool_i64(eta_p_21778);
            int64_t eta_p_21791 = ((int64_t *) mem_21876)[i_21739];
            int64_t eta_p_21792 = ((int64_t *) mem_21874)[i_21739];
            int64_t v_21795 = ((int64_t *) mem_param_21851.mem)[i_21739];
            
            // src/bgpc_vv_coloring.fut:19:29-32
            
            int64_t defunc_0_op_res_21796 = add64(inpacc_21201, bool_to_i64_res_21779);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            bool cond_21797 = eta_p_21791 == (int64_t) 1;
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t lifted_lambda_res_21798;
            
            if (cond_21797) {
                // src/bgpc_vv_coloring.fut:27:5-44
                
                int64_t lifted_lambda_res_t_res_21799 = sub64(eta_p_21792, (int64_t) 1);
                
                lifted_lambda_res_21798 = lifted_lambda_res_t_res_21799;
            } else {
                lifted_lambda_res_21798 = (int64_t) -1;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_21798) && slt64(lifted_lambda_res_21798, m_20370)) {
                ((int64_t *) mem_21899)[lifted_lambda_res_21798] = v_21795;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_21798) && slt64(lifted_lambda_res_21798, m_20370)) {
                ((int64_t *) mem_21897)[lifted_lambda_res_21798] = defunc_0_op_res_21796;
            }
            ((int64_t *) mem_21901)[i_21739] = defunc_0_op_res_21796;
            
            int64_t inpacc_tmp_21991 = defunc_0_op_res_21796;
            
            inpacc_21201 = inpacc_tmp_21991;
        }
        inpacc_21663 = inpacc_21201;
        // src/bgpc_vv_coloring.fut:27:5-44
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_21909, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_21901, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {loop_dz2081Uz2083U_20156});
        // src/bgpc_vv_coloring.fut:23:10-29
        
        int64_t out_szz_f_res_20357;
        
        if (x_20350) {
            // src/bgpc_vv_coloring.fut:171:23-55
            
            int64_t x_21669 = ((int64_t *) mem_21909)[tmp_20351];
            
            out_szz_f_res_20357 = x_21669;
        } else {
            out_szz_f_res_20357 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        int64_t out_szz_20359;
        
        if (cond_20349) {
            out_szz_20359 = (int64_t) 0;
        } else {
            out_szz_20359 = out_szz_f_res_20357;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        
        int64_t bytes_21910 = (int64_t) 8 * out_szz_20359;
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        bool cond_20256 = slt64(defunc_0_reduce_res_21672, (int64_t) 0);
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        int64_t next_color_bound_20257;
        
        if (cond_20256) {
            next_color_bound_20257 = color_bound_20160;
        } else {
            // src/bgpc_vv_coloring.fut:110:40-46
            
            int64_t max_arg1_20258 = add64((int64_t) 2, defunc_0_reduce_res_21672);
            
            // src/bgpc_vv_coloring.fut:110:10-46
            
            int64_t max_res_20259 = smax64(color_bound_20160, max_arg1_20258);
            
            next_color_bound_20257 = max_res_20259;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        if (memblock_alloc(ctx, &mem_21911, bytes_21910, "mem_21911")) {
            err = 1;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        for (int64_t nest_i_21995 = 0; nest_i_21995 < out_szz_20359; nest_i_21995++) {
            ((int64_t *) mem_21911.mem)[nest_i_21995] = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:33:6-27
        
        bool acc_cert_20910;
        
        // src/bgpc_vv_coloring.fut:29:21-33:27
        for (int64_t i_21744 = 0; i_21744 < m_20370; i_21744++) {
            int64_t eta_p_20922 = ((int64_t *) mem_21897)[i_21744];
            int64_t v_20924 = ((int64_t *) mem_21899)[i_21744];
            
            // src/bgpc_vv_coloring.fut:29:44-50
            
            int64_t lifted_lambda_res_20925 = sub64(eta_p_20922, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:33:6-27
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_20925) && slt64(lifted_lambda_res_20925, out_szz_20359)) {
                ((int64_t *) mem_21911.mem)[lifted_lambda_res_20925] = v_20924;
            }
        }
        // src/bgpc_vv_coloring.fut:175:17-48
        
        bool acc_cert_20407;
        
        // src/bgpc_vv_coloring.fut:175:17-48
        for (int64_t i_21746 = 0; i_21746 < out_szz_20359; i_21746++) {
            int64_t v_20411 = ((int64_t *) mem_21911.mem)[i_21746];
            
            // src/bgpc_vv_coloring.fut:175:17-48
            // UpdateAcc
            if (sle64((int64_t) 0, v_20411) && slt64(v_20411, (int64_t) 5)) {
                ((int64_t *) mem_21841)[v_20411] = (int64_t) -1;
            }
        }
        // src/bgpc_vv_coloring.fut:198:20-23
        
        bool loop_cond_20413 = slt64((int64_t) 0, out_szz_20359);
        
        if (memblock_set(ctx, &mem_param_tmp_21967, &mem_21911, "mem_21911") != 0)
            return 1;
        
        int64_t loop_dz2081Uz2083U_tmp_21968 = out_szz_20359;
        bool loop_while_tmp_21969 = loop_cond_20413;
        int64_t color_bound_tmp_21972 = next_color_bound_20257;
        
        if (memblock_set(ctx, &mem_param_21851, &mem_param_tmp_21967, "mem_param_tmp_21967") != 0)
            return 1;
        loop_dz2081Uz2083U_20156 = loop_dz2081Uz2083U_tmp_21968;
        loop_while_20157 = loop_while_tmp_21969;
        color_bound_20160 = color_bound_tmp_21972;
    }
    if (memblock_set(ctx, &ext_mem_21916, &mem_param_21851, "mem_param_21851") != 0)
        return 1;
    vv_color_side_order_res_20151 = loop_dz2081Uz2083U_20156;
    vv_color_side_order_res_20152 = loop_while_20157;
    vv_color_side_order_res_20155 = color_bound_20160;
    if (memblock_unref(ctx, &mem_21840, "mem_21840") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21844, "ext_mem_21844") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21845, "ext_mem_21845") != 0)
        return 1;
    // src/sparse_jacobian_vjp.fut:78:8-87:44
    
    int64_t x_21683;
    int64_t redout_21747 = (int64_t) 0;
    
    for (int64_t i_21748 = 0; i_21748 < (int64_t) 5; i_21748++) {
        int64_t x_20416 = ((int64_t *) mem_21841)[i_21748];
        
        // src/sparse_jacobian_vjp.fut:14:22-29
        
        int64_t max_res_20419 = smax64(x_20416, redout_21747);
        int64_t redout_tmp_21998 = max_res_20419;
        
        redout_21747 = redout_tmp_21998;
    }
    x_21683 = redout_21747;
    // src/sparse_jacobian_vjp.fut:14:13-45
    
    int64_t num_colors_of_res_f_res_20420 = add64((int64_t) 1, x_21683);
    
    // src/sparse_jacobian_vjp.fut:29:11-18
    
    bool bounds_invalid_upwards_20422 = slt64(num_colors_of_res_f_res_20420, (int64_t) 0);
    
    // src/sparse_jacobian_vjp.fut:29:11-18
    
    bool valid_20423 = !bounds_invalid_upwards_20422;
    
    // src/sparse_jacobian_vjp.fut:29:11-18
    
    bool range_valid_c_20424;
    
    if (!valid_20423) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Range ", (long long) (int64_t) 0, "..", (long long) (int64_t) 1, "..<", (long long) num_colors_of_res_f_res_20420, " is invalid.", "-> #0  src/sparse_jacobian_vjp.fut:29:11-18\n   #1  src/sparse_jacobian_vjp.fut:78:8-87:44\n   #2  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #3  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #4  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:21:8-29:18
    
    int64_t bytes_21919 = (int64_t) 40 * num_colors_of_res_f_res_20420;
    
    // src/sparse_jacobian_vjp.fut:42:15-35
    if (mem_21831_cached_sizze_22115 < bytes_21830) {
        err = lexical_realloc(ctx, &mem_21831, &mem_21831_cached_sizze_22115, bytes_21830);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:42:15-35
    for (int64_t nest_i_21999 = 0; nest_i_21999 < csr_bipartite_from_pattern_res_20111; nest_i_21999++) {
        ((double *) mem_21831)[nest_i_21999] = 0.0;
    }
    
    double zt_lhs_20540 = ((double *) x_mem_21827.mem)[(int64_t) 0];
    double zt_lhs_20545 = ((double *) x_mem_21827.mem)[(int64_t) 1];
    double zt_rhs_20546 = ((double *) x_mem_21827.mem)[(int64_t) 3];
    double zp_lhs_20548 = ((double *) x_mem_21827.mem)[(int64_t) 2];
    
    // src/sparse_jacobian_vjp.fut:21:8-29:18
    if (mem_21920_cached_sizze_22127 < bytes_21919) {
        err = lexical_realloc(ctx, &mem_21920, &mem_21920_cached_sizze_22127, bytes_21919);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:17:3-61
    if (mem_21925_cached_sizze_22128 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_21925, &mem_21925_cached_sizze_22128, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:21:8-29:18
    for (int64_t i_21755 = 0; i_21755 < num_colors_of_res_f_res_20420; i_21755++) {
        // src/sparse_jacobian_vjp.fut:17:3-61
        for (int64_t i_21751 = 0; i_21751 < (int64_t) 5; i_21751++) {
            int64_t eta_p_20433 = ((int64_t *) mem_21841)[i_21751];
            
            // src/sparse_jacobian_vjp.fut:17:15-49
            
            bool cond_20434 = eta_p_20433 == i_21755;
            
            // src/sparse_jacobian_vjp.fut:17:15-49
            
            double lifted_lambda_res_20435;
            
            if (cond_20434) {
                lifted_lambda_res_20435 = 1.0;
            } else {
                lifted_lambda_res_20435 = 0.0;
            }
            ((double *) mem_21925)[i_21751] = lifted_lambda_res_20435;
        }
        
        double elem_adj_21572 = ((double *) mem_21925)[(int64_t) 0];
        double elem_adj_21573 = ((double *) mem_21925)[(int64_t) 1];
        double elem_adj_21574 = ((double *) mem_21925)[(int64_t) 2];
        double elem_adj_21575 = ((double *) mem_21925)[(int64_t) 3];
        double binop_y_adj_21578 = -1.0 * elem_adj_21575;
        double binop_x_adj_21579 = zp_lhs_20548 * binop_y_adj_21578;
        double zp_lhs_adj_21581 = binop_x_adj_21579 + binop_x_adj_21579;
        double zp_lhs_adj_21584 = elem_adj_21574 + zp_lhs_adj_21581;
        double binop_y_adj_21586 = 2.0 * elem_adj_21574;
        double binop_x_adj_21587 = zt_rhs_20546 * elem_adj_21573;
        double binop_y_adj_21588 = zt_lhs_20545 * elem_adj_21573;
        double zp_rhs_adj_21591 = elem_adj_21572 + binop_y_adj_21586;
        double binop_x_adj_21592 = zt_lhs_20540 * elem_adj_21572;
        double zt_lhs_adj_21594 = elem_adj_21575 + binop_x_adj_21592;
        double zt_lhs_adj_21595 = binop_x_adj_21592 + zt_lhs_adj_21594;
        
        for (int64_t nest_i_22002 = 0; nest_i_22002 < (int64_t) 5; nest_i_22002++) {
            ((double *) mem_21920)[i_21755 * (int64_t) 5 + nest_i_22002] = 0.0;
        }
        ((double *) mem_21920)[i_21755 * (int64_t) 5] = zt_lhs_adj_21595;
        ((double *) mem_21920)[i_21755 * (int64_t) 5 + (int64_t) 4] = zp_rhs_adj_21591;
        ((double *) mem_21920)[i_21755 * (int64_t) 5 + (int64_t) 1] = binop_x_adj_21587;
        ((double *) mem_21920)[i_21755 * (int64_t) 5 + (int64_t) 3] = binop_y_adj_21588;
        ((double *) mem_21920)[i_21755 * (int64_t) 5 + (int64_t) 2] = zp_lhs_adj_21584;
    }
    // src/sparse_jacobian_vjp.fut:45:5-53:27
    
    bool compressed_to_csr_vals_res_20444;
    int64_t compressed_to_csr_vals_res_20446;
    bool loop_while_20447;
    int64_t i_20449;
    
    loop_while_20447 = 1;
    i_20449 = (int64_t) 0;
    while (loop_while_20447) {
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool x_20450 = sle64((int64_t) 0, i_20449);
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool y_20451 = slt64(i_20449, csr_bipartite_from_pattern_res_20110);
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool bounds_check_20452 = x_20450 && y_20451;
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool index_certs_20453;
        
        if (!bounds_check_20452) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_20449, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20110, "].", "-> #0  src/sparse_jacobian_vjp.fut:47:15-26\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        int64_t s_20454 = ((int64_t *) ext_mem_21829.mem)[i_20449];
        
        // src/sparse_jacobian_vjp.fut:48:25-27
        
        int64_t e_20455 = add64((int64_t) 1, i_20449);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool x_20456 = sle64((int64_t) 0, e_20455);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool y_20457 = slt64(e_20455, csr_bipartite_from_pattern_res_20110);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool bounds_check_20458 = x_20456 && y_20457;
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool index_certs_20459;
        
        if (!bounds_check_20458) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_20455, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20110, "].", "-> #0  src/sparse_jacobian_vjp.fut:48:15-28\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        int64_t e_20460 = ((int64_t *) ext_mem_21829.mem)[e_20455];
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t j_m_i_20461 = sub64(e_20460, s_20454);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool empty_slice_20462 = j_m_i_20461 == (int64_t) 0;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t m_20463 = sub64(j_m_i_20461, (int64_t) 1);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t i_p_m_t_s_20464 = add64(s_20454, m_20463);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_20465 = sle64((int64_t) 0, i_p_m_t_s_20464);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_20466 = slt64(i_p_m_t_s_20464, csr_bipartite_from_pattern_res_20111);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool zzero_lte_i_20467 = sle64((int64_t) 0, s_20454);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool i_lte_j_20468 = sle64(s_20454, e_20460);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool y_20469 = i_p_m_t_s_leq_w_20466 && zzero_lte_i_20467;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool y_20470 = zzero_leq_i_p_m_t_s_20465 && y_20469;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool forwards_ok_20471 = i_lte_j_20468 && y_20470;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool ok_or_empty_20472 = empty_slice_20462 || forwards_ok_20471;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool index_certs_20473;
        
        if (!ok_or_empty_20472) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_20454, ":", (long long) e_20460, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20111, "].", "-> #0  src/sparse_jacobian_vjp.fut:49:18-30\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool y_20475 = slt64(i_20449, (int64_t) 5);
        
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool bounds_check_20476 = x_20450 && y_20475;
        
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool index_certs_20477;
        
        if (!bounds_check_20476) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_20449, "] out of bounds for array of shape [", (long long) (int64_t) 5, "].", "-> #0  src/sparse_jacobian_vjp.fut:50:16-29\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        int64_t rc_20478 = ((int64_t *) mem_21841)[i_20449];
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool y_20480 = slt64(rc_20478, num_colors_of_res_f_res_20420);
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool x_20479 = sle64((int64_t) 0, rc_20478);
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool bounds_check_20481 = x_20479 && y_20480;
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool index_certs_20482;
        
        if (!bounds_check_20481) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) rc_20478, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_20420, "].", "-> #0  src/sparse_jacobian_vjp.fut:51:28-34\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:51:17-43
        for (int64_t i_21759 = 0; i_21759 < j_m_i_20461; i_21759++) {
            int64_t index_primexp_21771 = s_20454 + i_21759;
            int64_t eta_p_20484 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21771];
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool x_20485 = sle64((int64_t) 0, eta_p_20484);
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool y_20486 = slt64(eta_p_20484, (int64_t) 5);
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool bounds_check_20487 = x_20485 && y_20486;
            
            // src/sparse_jacobian_vjp.fut:51:28-37
            
            bool index_certs_20488;
            
            if (!bounds_check_20487) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_20484, "] out of bounds for array of shape [", (long long) (int64_t) 5, "].", "-> #0  src/sparse_jacobian_vjp.fut:51:28-37\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_vjp.fut:51:17-43
            
            double lifted_lambda_res_20489 = ((double *) mem_21920)[rc_20478 * (int64_t) 5 + eta_p_20484];
            
            ((double *) mem_21831)[s_20454 + i_21759] = lifted_lambda_res_20489;
        }
        // src/sparse_jacobian_vjp.fut:46:13-16
        
        bool loop_cond_20491 = slt64(e_20455, (int64_t) 5);
        bool loop_while_tmp_22003 = loop_cond_20491;
        int64_t i_tmp_22005 = e_20455;
        
        loop_while_20447 = loop_while_tmp_22003;
        i_20449 = i_tmp_22005;
    }
    compressed_to_csr_vals_res_20444 = loop_while_20447;
    compressed_to_csr_vals_res_20446 = i_20449;
    // src/dense_jacobian.fut:5:3-21
    if (mem_21951_cached_sizze_22129 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_21951, &mem_21951_cached_sizze_22129, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    if (mem_21952_cached_sizze_22130 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_21952, &mem_21952_cached_sizze_22130, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:68:29-47
    if (mem_21953_cached_sizze_22131 < (int64_t) 40) {
        err = lexical_realloc(ctx, &mem_21953, &mem_21953_cached_sizze_22131, (int64_t) 40);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:7:19-9:39
    
    bool defunc_0_reduce_res_21685;
    bool redout_21765 = 1;
    
    for (int64_t i_21766 = 0; i_21766 < (int64_t) 5; i_21766++) {
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        bool y_20833 = slt64(i_21766, csr_bipartite_from_pattern_res_20110);
        
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        bool index_certs_20835;
        
        if (!y_20833) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_21766, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20110, "].", "-> #0  src/sparse_jacobian_vjp.fut:64:17-28\n   #1  src/sparse_jacobian_vjp.fut:132:18-138:40\n   #2  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        int64_t s_20836 = ((int64_t *) ext_mem_21829.mem)[i_21766];
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool zzero_lte_i_20849 = sle64((int64_t) 0, s_20836);
        
        // src/sparse_jacobian_vjp.fut:65:27-29
        
        int64_t e_20837 = add64((int64_t) 1, i_21766);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool y_20839 = slt64(e_20837, csr_bipartite_from_pattern_res_20110);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool x_20838 = sle64((int64_t) 0, e_20837);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool bounds_check_20840 = x_20838 && y_20839;
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool index_certs_20841;
        
        if (!bounds_check_20840) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_20837, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20110, "].", "-> #0  src/sparse_jacobian_vjp.fut:65:17-30\n   #1  src/sparse_jacobian_vjp.fut:132:18-138:40\n   #2  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        int64_t e_20842 = ((int64_t *) ext_mem_21829.mem)[e_20837];
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t j_m_i_20843 = sub64(e_20842, s_20836);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t m_20845 = sub64(j_m_i_20843, (int64_t) 1);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t i_p_m_t_s_20846 = add64(s_20836, m_20845);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool i_p_m_t_s_leq_w_20848 = slt64(i_p_m_t_s_20846, csr_bipartite_from_pattern_res_20111);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool y_20857 = i_p_m_t_s_leq_w_20848 && zzero_lte_i_20849;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool zzero_leq_i_p_m_t_s_20847 = sle64((int64_t) 0, i_p_m_t_s_20846);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool y_20858 = zzero_leq_i_p_m_t_s_20847 && y_20857;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool i_lte_j_20850 = sle64(s_20836, e_20842);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool forwards_ok_20859 = i_lte_j_20850 && y_20858;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool empty_slice_20844 = j_m_i_20843 == (int64_t) 0;
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool ok_or_empty_20860 = empty_slice_20844 || forwards_ok_20859;
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool index_certs_20861;
        
        if (!ok_or_empty_20860) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_20836, ":", (long long) e_20842, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20111, "].", "-> #0  src/sparse_jacobian_vjp.fut:67:20-29\n   #1  src/sparse_jacobian_vjp.fut:132:18-138:40\n   #2  test/test_sparse_jacobian_vjp.fut:116:15-139:48\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/dense_jacobian.fut:5:3-21
        for (int64_t nest_i_22008 = 0; nest_i_22008 < (int64_t) 5; nest_i_22008++) {
            ((double *) mem_21951)[nest_i_22008] = 0.0;
        }
        // src/dense_jacobian.fut:5:3-39
        ((double *) mem_21951)[i_21766] = 1.0;
        
        double elem_adj_21603 = ((double *) mem_21951)[(int64_t) 0];
        double elem_adj_21604 = ((double *) mem_21951)[(int64_t) 1];
        double elem_adj_21605 = ((double *) mem_21951)[(int64_t) 2];
        double elem_adj_21606 = ((double *) mem_21951)[(int64_t) 3];
        double binop_y_adj_21609 = -1.0 * elem_adj_21606;
        double binop_x_adj_21610 = zp_lhs_20548 * binop_y_adj_21609;
        double zp_lhs_adj_21612 = binop_x_adj_21610 + binop_x_adj_21610;
        double zp_lhs_adj_21615 = elem_adj_21605 + zp_lhs_adj_21612;
        double binop_y_adj_21617 = 2.0 * elem_adj_21605;
        double binop_x_adj_21618 = zt_rhs_20546 * elem_adj_21604;
        double binop_y_adj_21619 = zt_lhs_20545 * elem_adj_21604;
        double zp_rhs_adj_21622 = elem_adj_21603 + binop_y_adj_21617;
        double binop_x_adj_21623 = zt_lhs_20540 * elem_adj_21603;
        double zt_lhs_adj_21625 = elem_adj_21606 + binop_x_adj_21623;
        double zt_lhs_adj_21626 = binop_x_adj_21623 + zt_lhs_adj_21625;
        
        for (int64_t nest_i_22009 = 0; nest_i_22009 < (int64_t) 5; nest_i_22009++) {
            ((double *) mem_21952)[nest_i_22009] = 0.0;
        }
        ((double *) mem_21952)[(int64_t) 0] = zt_lhs_adj_21626;
        ((double *) mem_21952)[(int64_t) 4] = zp_rhs_adj_21622;
        ((double *) mem_21952)[(int64_t) 1] = binop_x_adj_21618;
        ((double *) mem_21952)[(int64_t) 3] = binop_y_adj_21619;
        ((double *) mem_21952)[(int64_t) 2] = zp_lhs_adj_21615;
        // src/sparse_jacobian_vjp.fut:68:29-47
        for (int64_t nest_i_22010 = 0; nest_i_22010 < (int64_t) 5; nest_i_22010++) {
            ((double *) mem_21953)[nest_i_22010] = 0.0;
        }
        // src/sparse_jacobian_vjp.fut:69:12-32
        
        bool acc_cert_20865;
        
        // src/sparse_jacobian_vjp.fut:69:12-32
        for (int64_t i_21762 = 0; i_21762 < j_m_i_20843; i_21762++) {
            int64_t index_primexp_21768 = s_20836 + i_21762;
            int64_t v_20869 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21768];
            double v_20870 = ((double *) mem_21831)[index_primexp_21768];
            
            // src/sparse_jacobian_vjp.fut:69:12-32
            // UpdateAcc
            if (sle64((int64_t) 0, v_20869) && slt64(v_20869, (int64_t) 5)) {
                ((double *) mem_21953)[v_20869] = v_20870;
            }
        }
        // test/test_sparse_jacobian_vjp.fut:8:5-66
        
        bool defunc_0_reduce_res_21681;
        bool redout_21763 = 1;
        
        for (int64_t i_21764 = 0; i_21764 < (int64_t) 5; i_21764++) {
            bool eta_p_20893 = ((bool *) mem_21826.mem)[i_21766 * (int64_t) 5 + i_21764];
            double eta_p_20894 = ((double *) mem_21952)[i_21764];
            double eta_p_20895 = ((double *) mem_21953)[i_21764];
            
            // test/test_sparse_jacobian_vjp.fut:13:25-48
            
            double lifted_lambda_res_20896;
            
            if (eta_p_20893) {
                lifted_lambda_res_20896 = eta_p_20894;
            } else {
                lifted_lambda_res_20896 = 0.0;
            }
            // test/test_sparse_jacobian_vjp.fut:8:48-51
            
            double abs_arg0_20898 = eta_p_20895 - lifted_lambda_res_20896;
            
            // test/test_sparse_jacobian_vjp.fut:8:37-51
            
            double abs_res_20899 = fabs64(abs_arg0_20898);
            
            // test/test_sparse_jacobian_vjp.fut:8:53-59
            
            bool lifted_lambda_res_20900 = abs_res_20899 <= 1.0e-9;
            
            // test/test_sparse_jacobian_vjp.fut:8:5-66
            
            bool x_20883 = lifted_lambda_res_20900 && redout_21763;
            bool redout_tmp_22012 = x_20883;
            
            redout_21763 = redout_tmp_22012;
        }
        defunc_0_reduce_res_21681 = redout_21763;
        // test/test_sparse_jacobian_vjp.fut:9:6-39
        
        bool x_20511 = defunc_0_reduce_res_21681 && redout_21765;
        bool redout_tmp_22007 = x_20511;
        
        redout_21765 = redout_tmp_22007;
    }
    defunc_0_reduce_res_21685 = redout_21765;
    if (memblock_unref(ctx, &ext_mem_21828, "ext_mem_21828") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21829, "ext_mem_21829") != 0)
        return 1;
    prim_out_21962 = defunc_0_reduce_res_21685;
    *out_prim_out_22114 = prim_out_21962;
    
  cleanup:
    {
        free(mem_21831);
        free(mem_21832);
        free(mem_21841);
        free(mem_21853);
        free(mem_21857);
        free(mem_21874);
        free(mem_21876);
        free(mem_21877);
        free(mem_21897);
        free(mem_21899);
        free(mem_21901);
        free(mem_21909);
        free(mem_21920);
        free(mem_21925);
        free(mem_21951);
        free(mem_21952);
        free(mem_21953);
        if (memblock_unref(ctx, &mem_param_tmp_21967, "mem_param_tmp_21967") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21911, "mem_21911") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_21851, "mem_param_21851") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21916, "ext_mem_21916") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21844, "ext_mem_21844") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21845, "ext_mem_21845") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21842, "mem_21842") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21840, "mem_21840") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21828, "ext_mem_21828") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21829, "ext_mem_21829") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_sparse_vjp_zzero_pattern_matches_dense(struct futhark_context *ctx, bool *out_prim_out_22132, struct memblock x_mem_21827)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_21831_cached_sizze_22133 = 0;
    unsigned char *mem_21831 = NULL;
    int64_t mem_21832_cached_sizze_22134 = 0;
    unsigned char *mem_21832 = NULL;
    int64_t mem_21841_cached_sizze_22135 = 0;
    unsigned char *mem_21841 = NULL;
    int64_t mem_21853_cached_sizze_22136 = 0;
    unsigned char *mem_21853 = NULL;
    int64_t mem_21857_cached_sizze_22137 = 0;
    unsigned char *mem_21857 = NULL;
    int64_t mem_21874_cached_sizze_22138 = 0;
    unsigned char *mem_21874 = NULL;
    int64_t mem_21876_cached_sizze_22139 = 0;
    unsigned char *mem_21876 = NULL;
    int64_t mem_21877_cached_sizze_22140 = 0;
    unsigned char *mem_21877 = NULL;
    int64_t mem_21897_cached_sizze_22141 = 0;
    unsigned char *mem_21897 = NULL;
    int64_t mem_21899_cached_sizze_22142 = 0;
    unsigned char *mem_21899 = NULL;
    int64_t mem_21901_cached_sizze_22143 = 0;
    unsigned char *mem_21901 = NULL;
    int64_t mem_21909_cached_sizze_22144 = 0;
    unsigned char *mem_21909 = NULL;
    int64_t mem_21926_cached_sizze_22145 = 0;
    unsigned char *mem_21926 = NULL;
    struct memblock mem_param_tmp_21968;
    
    mem_param_tmp_21968.references = NULL;
    
    struct memblock mem_21911;
    
    mem_21911.references = NULL;
    
    struct memblock mem_param_21851;
    
    mem_param_21851.references = NULL;
    
    struct memblock ext_mem_21916;
    
    ext_mem_21916.references = NULL;
    
    struct memblock ext_mem_21844;
    
    ext_mem_21844.references = NULL;
    
    struct memblock ext_mem_21845;
    
    ext_mem_21845.references = NULL;
    
    struct memblock mem_21842;
    
    mem_21842.references = NULL;
    
    struct memblock mem_21840;
    
    mem_21840.references = NULL;
    
    struct memblock ext_mem_21828;
    
    ext_mem_21828.references = NULL;
    
    struct memblock ext_mem_21829;
    
    ext_mem_21829.references = NULL;
    
    struct memblock mem_21809 = ctx->constants->mem_21809;
    struct memblock mem_21812 = ctx->constants->mem_21812;
    struct memblock mem_21814 = ctx->constants->mem_21814;
    struct memblock mem_21819 = ctx->constants->mem_21819;
    struct memblock mem_21820 = ctx->constants->mem_21820;
    struct memblock mem_21826 = ctx->constants->mem_21826;
    bool prim_out_21962;
    
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_20119;
    int64_t csr_bipartite_from_pattern_res_20120;
    
    if (futrts_csr_rows_from_pattern_7010(ctx, &ext_mem_21829, &ext_mem_21828, &csr_bipartite_from_pattern_res_20119, &csr_bipartite_from_pattern_res_20120, mem_21814, (int64_t) 2, (int64_t) 3) != 0) {
        err = 1;
        goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:42:15-35
    
    int64_t bytes_21830 = (int64_t) 8 * csr_bipartite_from_pattern_res_20120;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (mem_21832_cached_sizze_22134 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_21832, &mem_21832_cached_sizze_22134, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t discard_21691;
    int64_t scanacc_21687 = (int64_t) 0;
    
    for (int64_t i_21689 = 0; i_21689 < (int64_t) 2; i_21689++) {
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t defunc_0_op_res_20135 = add64((int64_t) 1, scanacc_21687);
        
        ((int64_t *) mem_21832)[i_21689] = defunc_0_op_res_20135;
        
        int64_t scanacc_tmp_21963 = defunc_0_op_res_20135;
        
        scanacc_21687 = scanacc_tmp_21963;
    }
    discard_21691 = scanacc_21687;
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t x_20145 = ((int64_t *) mem_21832)[(int64_t) 1];
    
    // src/bgpc_vv_coloring.fut:194:20-45
    
    int64_t bytes_21839 = (int64_t) 8 * x_20145;
    
    // src/sparse_jacobian_vjp.fut:42:15-35
    if (mem_21831_cached_sizze_22133 < bytes_21830) {
        err = lexical_realloc(ctx, &mem_21831, &mem_21831_cached_sizze_22133, bytes_21830);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_vjp.fut:42:15-35
    for (int64_t nest_i_21965 = 0; nest_i_21965 < csr_bipartite_from_pattern_res_20120; nest_i_21965++) {
        ((double *) mem_21831)[nest_i_21965] = 0.0;
    }
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    bool loop_cond_20159 = slt64((int64_t) 0, x_20145);
    
    // src/bgpc_vv_coloring.fut:194:20-45
    if (memblock_alloc(ctx, &mem_21840, bytes_21839, "mem_21840")) {
        err = 1;
        goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:194:20-45
    
    bool acc_cert_20718;
    
    // src/bgpc_vv_coloring.fut:194:20-45
    for (int64_t i_21693 = 0; i_21693 < (int64_t) 2; i_21693++) {
        int64_t eta_p_20730 = ((int64_t *) mem_21832)[i_21693];
        
        // src/bgpc_vv_coloring.fut:194:20-45
        
        int64_t lifted_lambda_res_t_res_20733 = sub64(eta_p_20730, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:194:20-45
        // UpdateAcc
        if (sle64((int64_t) 0, lifted_lambda_res_t_res_20733) && slt64(lifted_lambda_res_t_res_20733, x_20145)) {
            ((int64_t *) mem_21840.mem)[lifted_lambda_res_t_res_20733] = i_21693;
        }
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    if (mem_21841_cached_sizze_22135 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_21841, &mem_21841_cached_sizze_22135, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/bgpc_vv_coloring.fut:190:35-57
    for (int64_t nest_i_21967 = 0; nest_i_21967 < (int64_t) 2; nest_i_21967++) {
        ((int64_t *) mem_21841)[nest_i_21967] = (int64_t) -1;
    }
    // src/pattern_csr.fut:25:3-39
    if (memblock_alloc(ctx, &mem_21842, (int64_t) 6, "mem_21842")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:25:3-39
    // src/pattern_csr.fut:25:3-39
    lmad_copy_1b(ctx, 2, (uint8_t *) mem_21842.mem, (int64_t) 0, (int64_t []) {(int64_t) 2, (int64_t) 1}, (uint8_t *) mem_21814.mem, (int64_t) 0, (int64_t []) {(int64_t) 1, (int64_t) 3}, (int64_t []) {(int64_t) 3, (int64_t) 2});
    // src/pattern_csr.fut:25:3-39
    
    int64_t csr_cols_from_pattern_res_20124;
    int64_t csr_cols_from_pattern_res_20125;
    
    if (futrts_csr_rows_from_pattern_7010(ctx, &ext_mem_21845, &ext_mem_21844, &csr_cols_from_pattern_res_20124, &csr_cols_from_pattern_res_20125, mem_21842, (int64_t) 3, (int64_t) 2) != 0) {
        err = 1;
        goto cleanup;
    }
    if (memblock_unref(ctx, &mem_21842, "mem_21842") != 0)
        return 1;
    // src/bgpc_vv_coloring.fut:197:5-207:37
    
    int64_t vv_color_side_order_res_20160;
    bool vv_color_side_order_res_20161;
    int64_t vv_color_side_order_res_20164;
    int64_t loop_dz2081Uz2083U_20165;
    bool loop_while_20166;
    int64_t color_bound_20169;
    
    if (memblock_set(ctx, &mem_param_21851, &mem_21840, "mem_21840") != 0)
        return 1;
    loop_dz2081Uz2083U_20165 = x_20145;
    loop_while_20166 = loop_cond_20159;
    color_bound_20169 = (int64_t) 1;
    while (loop_while_20166) {
        // src/bgpc_vv_coloring.fut:62:5-64:15
        
        bool cond_20170 = slt64((int64_t) 0, color_bound_20169);
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t bytes_21852 = (int64_t) 8 * loop_dz2081Uz2083U_20165;
        
        // src/bgpc_vv_coloring.fut:101:5-106:50
        if (mem_21853_cached_sizze_22136 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21853, &mem_21853_cached_sizze_22136, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:77:24-51
        if (mem_21857_cached_sizze_22137 < color_bound_20169) {
            err = lexical_realloc(ctx, &mem_21857, &mem_21857_cached_sizze_22137, color_bound_20169);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:101:5-106:50
        
        int64_t defunc_0_reduce_res_21672;
        int64_t redout_21695 = (int64_t) -1;
        
        for (int64_t i_21697 = 0; i_21697 < loop_dz2081Uz2083U_20165; i_21697++) {
            int64_t eta_p_21325 = ((int64_t *) mem_param_21851.mem)[i_21697];
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool x_21327 = sle64((int64_t) 0, eta_p_21325);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool y_21328 = slt64(eta_p_21325, csr_bipartite_from_pattern_res_20119);
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool bounds_check_21329 = x_21327 && y_21328;
            
            // src/bgpc_vv_coloring.fut:46:44-56
            
            bool index_certs_21330;
            
            if (!bounds_check_21329) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_21325, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20119, "].", "-> #0  src/bgpc_vv_coloring.fut:46:44-56\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:44-56
            
            int64_t seen_final_21331 = ((int64_t *) ext_mem_21829.mem)[eta_p_21325];
            
            // src/bgpc_vv_coloring.fut:46:70-72
            
            int64_t seen_final_21332 = add64((int64_t) 1, eta_p_21325);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool x_21333 = sle64((int64_t) 0, seen_final_21332);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool y_21334 = slt64(seen_final_21332, csr_bipartite_from_pattern_res_20119);
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool bounds_check_21335 = x_21333 && y_21334;
            
            // src/bgpc_vv_coloring.fut:46:59-73
            
            bool index_certs_21336;
            
            if (!bounds_check_21335) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_21332, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20119, "].", "-> #0  src/bgpc_vv_coloring.fut:46:59-73\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:46:59-73
            
            int64_t seen_final_21337 = ((int64_t *) ext_mem_21829.mem)[seen_final_21332];
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t j_m_i_21338 = sub64(seen_final_21337, seen_final_21331);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool empty_slice_21339 = j_m_i_21338 == (int64_t) 0;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t m_21340 = sub64(j_m_i_21338, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            int64_t i_p_m_t_s_21341 = add64(seen_final_21331, m_21340);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_leq_i_p_m_t_s_21342 = sle64((int64_t) 0, i_p_m_t_s_21341);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_p_m_t_s_leq_w_21343 = slt64(i_p_m_t_s_21341, csr_bipartite_from_pattern_res_20120);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool zzero_lte_i_21344 = sle64((int64_t) 0, seen_final_21331);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool i_lte_j_21345 = sle64(seen_final_21331, seen_final_21337);
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_21346 = i_p_m_t_s_leq_w_21343 && zzero_lte_i_21344;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool y_21347 = zzero_leq_i_p_m_t_s_21342 && y_21346;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool forwards_ok_21348 = i_lte_j_21345 && y_21347;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool ok_or_empty_21349 = empty_slice_21339 || forwards_ok_21348;
            
            // src/bgpc_vv_coloring.fut:46:5-54:19
            
            bool index_certs_21350;
            
            if (!ok_or_empty_21349) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_final_21331, ":", (long long) seen_final_21337, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20120, "].", "-> #0  src/bgpc_vv_coloring.fut:46:5-54:19\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:77:24-51
            for (int64_t nest_i_21976 = 0; nest_i_21976 < color_bound_20169; nest_i_21976++) {
                ((bool *) mem_21857)[nest_i_21976] = 0;
            }
            // src/bgpc_vv_coloring.fut:46:5-54:19
            for (int64_t i_21352 = 0; i_21352 < j_m_i_21338; i_21352++) {
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t index_primexp_21354 = seen_final_21331 + i_21352;
                
                // src/bgpc_vv_coloring.fut:71:14-80:57
                
                int64_t v_21355 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21354];
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool x_21356 = sle64((int64_t) 0, v_21355);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool y_21357 = slt64(v_21355, csr_cols_from_pattern_res_20124);
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool bounds_check_21358 = x_21356 && y_21357;
                
                // src/bgpc_vv_coloring.fut:49:26-37
                
                bool index_certs_21359;
                
                if (!bounds_check_21358) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_21355, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20124, "].", "-> #0  src/bgpc_vv_coloring.fut:49:26-37\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:26-37
                
                int64_t seen_acczq_21360 = ((int64_t *) ext_mem_21845.mem)[v_21355];
                
                // src/bgpc_vv_coloring.fut:49:50-52
                
                int64_t seen_acczq_21361 = add64((int64_t) 1, v_21355);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool x_21362 = sle64((int64_t) 0, seen_acczq_21361);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool y_21363 = slt64(seen_acczq_21361, csr_cols_from_pattern_res_20124);
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool bounds_check_21364 = x_21362 && y_21363;
                
                // src/bgpc_vv_coloring.fut:49:40-53
                
                bool index_certs_21365;
                
                if (!bounds_check_21364) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_21361, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20124, "].", "-> #0  src/bgpc_vv_coloring.fut:49:40-53\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:49:40-53
                
                int64_t seen_acczq_21366 = ((int64_t *) ext_mem_21845.mem)[seen_acczq_21361];
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t j_m_i_21367 = sub64(seen_acczq_21366, seen_acczq_21360);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool empty_slice_21368 = j_m_i_21367 == (int64_t) 0;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t m_21369 = sub64(j_m_i_21367, (int64_t) 1);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                int64_t i_p_m_t_s_21370 = add64(seen_acczq_21360, m_21369);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_leq_i_p_m_t_s_21371 = sle64((int64_t) 0, i_p_m_t_s_21370);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_p_m_t_s_leq_w_21372 = slt64(i_p_m_t_s_21370, csr_cols_from_pattern_res_20125);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool zzero_lte_i_21373 = sle64((int64_t) 0, seen_acczq_21360);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool i_lte_j_21374 = sle64(seen_acczq_21360, seen_acczq_21366);
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_21375 = i_p_m_t_s_leq_w_21372 && zzero_lte_i_21373;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool y_21376 = zzero_leq_i_p_m_t_s_21371 && y_21375;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool forwards_ok_21377 = i_lte_j_21374 && y_21376;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool ok_or_empty_21378 = empty_slice_21368 || forwards_ok_21377;
                
                // src/bgpc_vv_coloring.fut:48:9-53:33
                
                bool index_certs_21379;
                
                if (!ok_or_empty_21378) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) seen_acczq_21360, ":", (long long) seen_acczq_21366, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20125, "].", "-> #0  src/bgpc_vv_coloring.fut:48:9-53:33\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:48:9-53:33
                for (int64_t i_21381 = 0; i_21381 < j_m_i_21367; i_21381++) {
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t index_primexp_21383 = seen_acczq_21360 + i_21381;
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    int64_t u_21384 = ((int64_t *) ext_mem_21844.mem)[index_primexp_21383];
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool x_21385 = sle64((int64_t) 0, u_21384);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool y_21386 = slt64(u_21384, (int64_t) 2);
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool bounds_check_21387 = x_21385 && y_21386;
                    
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    bool index_certs_21388;
                    
                    if (!bounds_check_21387) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_21384, "] out of bounds for array of shape [", (long long) (int64_t) 2, "].", "-> #0  src/bgpc_vv_coloring.fut:50:20-29\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:50:20-29
                    
                    int64_t cu_21389 = ((int64_t *) mem_21841)[u_21384];
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_21390 = u_21384 == eta_p_21325;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    
                    bool cond_21391 = !cond_21390;
                    
                    // src/bgpc_vv_coloring.fut:51:30-37
                    
                    bool cond_t_res_21392 = sle64((int64_t) 0, cu_21389);
                    
                    // src/bgpc_vv_coloring.fut:71:14-80:57
                    
                    bool x_21393 = cond_21391 && cond_t_res_21392;
                    
                    // src/bgpc_vv_coloring.fut:51:44-67
                    
                    bool cond_t_res_21394 = slt64(cu_21389, color_bound_20169);
                    
                    // src/bgpc_vv_coloring.fut:94:14-103:69
                    
                    bool x_21395 = x_21393 && cond_t_res_21394;
                    
                    // src/bgpc_vv_coloring.fut:51:14-53:33
                    if (x_21395) {
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool bounds_check_21636 = cond_t_res_21392 && cond_t_res_21394;
                        
                        // src/bgpc_vv_coloring.fut:52:19-50
                        
                        bool index_certs_21637;
                        
                        if (!bounds_check_21636) {
                            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) cu_21389, "] out of bounds for array of shape [", (long long) color_bound_20169, "].", "-> #0  src/bgpc_vv_coloring.fut:52:19-50\n   #1  src/bgpc_vv_coloring.fut:71:14-80:57\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                            err = FUTHARK_PROGRAM_ERROR;
                            goto cleanup;
                        }
                        // src/bgpc_vv_coloring.fut:52:19-50
                        ((bool *) mem_21857)[cu_21389] = 1;
                    }
                }
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool loop_cond_21400;
            
            if (cond_20170) {
                // src/bgpc_vv_coloring.fut:63:30-37
                
                bool loop_cond_t_res_21639 = ((bool *) mem_21857)[(int64_t) 0];
                
                loop_cond_21400 = loop_cond_t_res_21639;
            } else {
                loop_cond_21400 = 0;
            }
            // src/bgpc_vv_coloring.fut:62:5-64:15
            
            bool c_final_21402;
            int64_t c_final_21403;
            bool loop_while_21404;
            int64_t c_21405;
            
            loop_while_21404 = loop_cond_21400;
            c_21405 = (int64_t) 0;
            while (loop_while_21404) {
                // src/bgpc_vv_coloring.fut:64:9-15
                
                int64_t loopres_21406 = add64((int64_t) 1, c_21405);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool cond_21407 = slt64(loopres_21406, color_bound_20169);
                
                // src/bgpc_vv_coloring.fut:63:11-37
                
                bool loop_cond_21408;
                
                if (cond_21407) {
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool x_21640 = sle64((int64_t) 0, loopres_21406);
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool bounds_check_21641 = cond_21407 && x_21640;
                    
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool index_certs_21642;
                    
                    if (!bounds_check_21641) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) loopres_21406, "] out of bounds for array of shape [", (long long) color_bound_20169, "].", "-> #0  src/bgpc_vv_coloring.fut:63:30-37\n   #1  src/bgpc_vv_coloring.fut:81:6-28\n   #2  src/bgpc_vv_coloring.fut:94:14-103:69\n   #3  src/bgpc_vv_coloring.fut:101:5-104:10\n   #4  src/bgpc_vv_coloring.fut:186:14-201:67\n   #5  src/bgpc_vv_coloring.fut:220:14-223:64\n   #6  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #7  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #8  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #9  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:63:30-37
                    
                    bool loop_cond_t_res_21643 = ((bool *) mem_21857)[loopres_21406];
                    
                    loop_cond_21408 = loop_cond_t_res_21643;
                } else {
                    loop_cond_21408 = 0;
                }
                
                bool loop_while_tmp_21979 = loop_cond_21408;
                int64_t c_tmp_21980 = loopres_21406;
                
                loop_while_21404 = loop_while_tmp_21979;
                c_21405 = c_tmp_21980;
            }
            c_final_21402 = loop_while_21404;
            c_final_21403 = c_21405;
            // src/bgpc_vv_coloring.fut:106:25-32
            
            int64_t max_res_20264 = smax64(c_final_21403, redout_21695);
            
            ((int64_t *) mem_21853)[i_21697] = c_final_21403;
            
            int64_t redout_tmp_21974 = max_res_20264;
            
            redout_21695 = redout_tmp_21974;
        }
        defunc_0_reduce_res_21672 = redout_21695;
        // src/bgpc_vv_coloring.fut:113:17-43
        
        bool acc_cert_20270;
        
        // src/bgpc_vv_coloring.fut:113:17-43
        for (int64_t i_21700 = 0; i_21700 < loop_dz2081Uz2083U_20165; i_21700++) {
            int64_t v_20274 = ((int64_t *) mem_param_21851.mem)[i_21700];
            int64_t v_20275 = ((int64_t *) mem_21853)[i_21700];
            
            // src/bgpc_vv_coloring.fut:113:17-43
            // UpdateAcc
            if (sle64((int64_t) 0, v_20274) && slt64(v_20274, (int64_t) 2)) {
                ((int64_t *) mem_21841)[v_20274] = v_20275;
            }
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21874_cached_sizze_22138 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21874, &mem_21874_cached_sizze_22138, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21876_cached_sizze_22139 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21876, &mem_21876_cached_sizze_22139, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21877_cached_sizze_22140 < loop_dz2081Uz2083U_20165) {
            err = lexical_realloc(ctx, &mem_21877, &mem_21877_cached_sizze_22140, loop_dz2081Uz2083U_20165);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t discard_21712;
        int64_t scanacc_21704 = (int64_t) 0;
        
        for (int64_t i_21708 = 0; i_21708 < loop_dz2081Uz2083U_20165; i_21708++) {
            int64_t eta_p_21249 = ((int64_t *) mem_param_21851.mem)[i_21708];
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool x_21250 = sle64((int64_t) 0, eta_p_21249);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool y_21251 = slt64(eta_p_21249, csr_bipartite_from_pattern_res_20119);
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool bounds_check_21252 = x_21250 && y_21251;
            
            // src/bgpc_vv_coloring.fut:126:15-27
            
            bool index_certs_21253;
            
            if (!bounds_check_21252) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_21249, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20119, "].", "-> #0  src/bgpc_vv_coloring.fut:126:15-27\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:127:26-28
            
            int64_t k_end_21255 = add64((int64_t) 1, eta_p_21249);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool x_21256 = sle64((int64_t) 0, k_end_21255);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool y_21257 = slt64(k_end_21255, csr_bipartite_from_pattern_res_20119);
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool bounds_check_21258 = x_21256 && y_21257;
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            bool index_certs_21259;
            
            if (!bounds_check_21258) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_end_21255, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20119, "].", "-> #0  src/bgpc_vv_coloring.fut:127:15-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/bgpc_vv_coloring.fut:126:15-27
            
            int64_t k0_21254 = ((int64_t *) ext_mem_21829.mem)[eta_p_21249];
            
            // src/bgpc_vv_coloring.fut:127:15-29
            
            int64_t k_end_21260 = ((int64_t *) ext_mem_21829.mem)[k_end_21255];
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool cond_21261 = slt64(k0_21254, k_end_21260);
            
            // src/bgpc_vv_coloring.fut:130:5-146:27
            
            bool loses_conflict_vertex_res_21262;
            bool loses_conflict_vertex_res_21263;
            int64_t loses_conflict_vertex_res_21264;
            bool loop_while_21265;
            bool lost_21266;
            int64_t k_21267;
            
            loop_while_21265 = cond_21261;
            lost_21266 = 0;
            k_21267 = k0_21254;
            while (loop_while_21265) {
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool x_21268 = sle64((int64_t) 0, k_21267);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool y_21269 = slt64(k_21267, csr_bipartite_from_pattern_res_20120);
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool bounds_check_21270 = x_21268 && y_21269;
                
                // src/bgpc_vv_coloring.fut:132:15-26
                
                bool index_certs_21271;
                
                if (!bounds_check_21270) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_21267, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20120, "].", "-> #0  src/bgpc_vv_coloring.fut:132:15-26\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:132:15-26
                
                int64_t v_21272 = ((int64_t *) ext_mem_21828.mem)[k_21267];
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool x_21273 = sle64((int64_t) 0, v_21272);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool y_21274 = slt64(v_21272, csr_cols_from_pattern_res_20124);
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool bounds_check_21275 = x_21273 && y_21274;
                
                // src/bgpc_vv_coloring.fut:134:19-30
                
                bool index_certs_21276;
                
                if (!bounds_check_21275) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) v_21272, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20124, "].", "-> #0  src/bgpc_vv_coloring.fut:134:19-30\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:135:29-31
                
                int64_t t_end_21278 = add64((int64_t) 1, v_21272);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool x_21279 = sle64((int64_t) 0, t_end_21278);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool y_21280 = slt64(t_end_21278, csr_cols_from_pattern_res_20124);
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool bounds_check_21281 = x_21279 && y_21280;
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                bool index_certs_21282;
                
                if (!bounds_check_21281) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_end_21278, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20124, "].", "-> #0  src/bgpc_vv_coloring.fut:135:19-32\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/bgpc_vv_coloring.fut:134:19-30
                
                int64_t t0_21277 = ((int64_t *) ext_mem_21845.mem)[v_21272];
                
                // src/bgpc_vv_coloring.fut:135:19-32
                
                int64_t t_end_21283 = ((int64_t *) ext_mem_21845.mem)[t_end_21278];
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool cond_21284 = slt64(t0_21277, t_end_21283);
                
                // src/bgpc_vv_coloring.fut:138:9-143:38
                
                bool loopres_21285;
                bool loopres_21286;
                int64_t loopres_21287;
                bool loop_while_21288;
                bool lost_in_net_21289;
                int64_t t_21290;
                
                loop_while_21288 = cond_21284;
                lost_in_net_21289 = 0;
                t_21290 = t0_21277;
                while (loop_while_21288) {
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool x_21291 = sle64((int64_t) 0, t_21290);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool y_21292 = slt64(t_21290, csr_cols_from_pattern_res_20125);
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool bounds_check_21293 = x_21291 && y_21292;
                    
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    bool index_certs_21294;
                    
                    if (!bounds_check_21293) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_21290, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20125, "].", "-> #0  src/bgpc_vv_coloring.fut:140:19-29\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/bgpc_vv_coloring.fut:140:19-29
                    
                    int64_t u_21295 = ((int64_t *) ext_mem_21844.mem)[t_21290];
                    
                    // src/bgpc_vv_coloring.fut:142:13-71
                    
                    bool lost_in_netzq_21296;
                    
                    if (lost_in_net_21289) {
                        lost_in_netzq_21296 = 1;
                    } else {
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_21297 = u_21295 == eta_p_21249;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_21298 = !cond_21297;
                        
                        // src/bgpc_vv_coloring.fut:142:29-70
                        
                        bool cond_21299;
                        
                        if (cond_21298) {
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool x_21645 = sle64((int64_t) 0, u_21295);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool y_21646 = slt64(u_21295, (int64_t) 2);
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool bounds_check_21647 = x_21645 && y_21646;
                            
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            bool index_certs_21648;
                            
                            if (!bounds_check_21647) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) u_21295, "] out of bounds for array of shape [", (long long) (int64_t) 2, "].", "-> #0  src/bgpc_vv_coloring.fut:142:39-48\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool y_21650 = slt64(eta_p_21249, (int64_t) 2);
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool bounds_check_21651 = x_21250 && y_21650;
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            bool index_certs_21652;
                            
                            if (!bounds_check_21651) {
                                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_21249, "] out of bounds for array of shape [", (long long) (int64_t) 2, "].", "-> #0  src/bgpc_vv_coloring.fut:142:52-61\n   #1  src/bgpc_vv_coloring.fut:160:14-168:57\n   #2  src/bgpc_vv_coloring.fut:166:5-169:10\n   #3  src/bgpc_vv_coloring.fut:186:14-205:56\n   #4  src/bgpc_vv_coloring.fut:220:14-223:64\n   #5  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #6  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #7  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #8  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
                                err = FUTHARK_PROGRAM_ERROR;
                                goto cleanup;
                            }
                            // src/bgpc_vv_coloring.fut:142:39-48
                            
                            int64_t zeze_lhs_21649 = ((int64_t *) mem_21841)[u_21295];
                            
                            // src/bgpc_vv_coloring.fut:142:52-61
                            
                            int64_t zeze_rhs_21653 = ((int64_t *) mem_21841)[eta_p_21249];
                            
                            // src/bgpc_vv_coloring.fut:142:49-61
                            
                            bool cond_t_res_21654 = zeze_lhs_21649 == zeze_rhs_21653;
                            
                            cond_21299 = cond_t_res_21654;
                        } else {
                            cond_21299 = 0;
                        }
                        // src/bgpc_vv_coloring.fut:142:67-70
                        
                        bool lost_in_netzq_f_res_t_res_21310 = slt64(u_21295, eta_p_21249);
                        
                        // src/bgpc_vv_coloring.fut:160:14-168:57
                        
                        bool x_21311 = cond_21299 && lost_in_netzq_f_res_t_res_21310;
                        
                        lost_in_netzq_21296 = x_21311;
                    }
                    // src/bgpc_vv_coloring.fut:143:31-37
                    
                    int64_t tmp_21312 = add64((int64_t) 1, t_21290);
                    
                    // src/bgpc_vv_coloring.fut:139:15-43
                    
                    bool cond_21313 = slt64(tmp_21312, t_end_21283);
                    
                    // src/bgpc_vv_coloring.fut:139:28-43
                    
                    bool not_res_21314 = !lost_in_netzq_21296;
                    
                    // src/bgpc_vv_coloring.fut:166:5-169:10
                    
                    bool x_21315 = cond_21313 && not_res_21314;
                    bool loop_while_tmp_21989 = x_21315;
                    bool lost_in_net_tmp_21990 = lost_in_netzq_21296;
                    int64_t t_tmp_21991 = tmp_21312;
                    
                    loop_while_21288 = loop_while_tmp_21989;
                    lost_in_net_21289 = lost_in_net_tmp_21990;
                    t_21290 = t_tmp_21991;
                }
                loopres_21285 = loop_while_21288;
                loopres_21286 = lost_in_net_21289;
                loopres_21287 = t_21290;
                // src/bgpc_vv_coloring.fut:160:14-168:57
                
                bool lostzq_21316 = lost_21266 || loopres_21286;
                
                // src/bgpc_vv_coloring.fut:146:20-26
                
                int64_t tmp_21317 = add64((int64_t) 1, k_21267);
                
                // src/bgpc_vv_coloring.fut:131:11-32
                
                bool cond_21318 = slt64(tmp_21317, k_end_21260);
                
                // src/bgpc_vv_coloring.fut:131:24-32
                
                bool not_res_21319 = !lostzq_21316;
                
                // src/bgpc_vv_coloring.fut:166:5-169:10
                
                bool x_21320 = cond_21318 && not_res_21319;
                bool loop_while_tmp_21986 = x_21320;
                bool lost_tmp_21987 = lostzq_21316;
                int64_t k_tmp_21988 = tmp_21317;
                
                loop_while_21265 = loop_while_tmp_21986;
                lost_21266 = lost_tmp_21987;
                k_21267 = k_tmp_21988;
            }
            loses_conflict_vertex_res_21262 = loop_while_21265;
            loses_conflict_vertex_res_21263 = lost_21266;
            loses_conflict_vertex_res_21264 = k_21267;
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_f_res_21322 = btoi_bool_i64(loses_conflict_vertex_res_21263);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t defunc_0_op_res_20376 = add64(defunc_0_f_res_21322, scanacc_21704);
            
            ((int64_t *) mem_21874)[i_21708] = defunc_0_op_res_20376;
            ((int64_t *) mem_21876)[i_21708] = defunc_0_f_res_21322;
            ((bool *) mem_21877)[i_21708] = loses_conflict_vertex_res_21263;
            
            int64_t scanacc_tmp_21982 = defunc_0_op_res_20376;
            
            scanacc_21704 = scanacc_tmp_21982;
        }
        discard_21712 = scanacc_21704;
        // src/bgpc_vv_coloring.fut:23:25-28
        
        int64_t tmp_20360 = sub64(loop_dz2081Uz2083U_20165, (int64_t) 1);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool y_20362 = slt64(tmp_20360, loop_dz2081Uz2083U_20165);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool x_20361 = sle64((int64_t) 0, tmp_20360);
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool bounds_check_20363 = x_20361 && y_20362;
        
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        bool cond_20358 = loop_dz2081Uz2083U_20165 == (int64_t) 0;
        
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool protect_assert_disj_20364 = cond_20358 || bounds_check_20363;
        
        // src/bgpc_vv_coloring.fut:23:10-29
        
        bool index_certs_20365;
        
        if (!protect_assert_disj_20364) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_20360, "] out of bounds for array of shape [", (long long) loop_dz2081Uz2083U_20165, "].", "-> #0  src/bgpc_vv_coloring.fut:23:10-29\n   #1  src/bgpc_vv_coloring.fut:171:23-55\n   #2  src/bgpc_vv_coloring.fut:186:14-205:56\n   #3  src/bgpc_vv_coloring.fut:220:14-223:64\n   #4  src/sparse_jacobian_vjp.fut:79:10-85:56\n   #5  src/sparse_jacobian_vjp.fut:108:8-112:31\n   #6  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #7  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:171:23-55
        
        bool x_20359 = !cond_20358;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_f_res_20377;
        
        if (x_20359) {
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t x_21655 = ((int64_t *) mem_21874)[tmp_20360];
            
            m_f_res_20377 = x_21655;
        } else {
            m_f_res_20377 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t m_20379;
        
        if (cond_20358) {
            m_20379 = (int64_t) 0;
        } else {
            m_20379 = m_f_res_20377;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        int64_t bytes_21896 = (int64_t) 8 * m_20379;
        
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21897_cached_sizze_22141 < bytes_21896) {
            err = lexical_realloc(ctx, &mem_21897, &mem_21897_cached_sizze_22141, bytes_21896);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21899_cached_sizze_22142 < bytes_21896) {
            err = lexical_realloc(ctx, &mem_21899, &mem_21899_cached_sizze_22142, bytes_21896);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:18:24-27:44
        if (mem_21901_cached_sizze_22143 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21901, &mem_21901_cached_sizze_22143, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        if (mem_21909_cached_sizze_22144 < bytes_21852) {
            err = lexical_realloc(ctx, &mem_21909, &mem_21909_cached_sizze_22144, bytes_21852);
            if (err != FUTHARK_SUCCESS)
                goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:27:5-44
        
        bool acc_cert_21122;
        bool acc_cert_21123;
        
        // src/bgpc_vv_coloring.fut:18:24-27:44
        
        int64_t inpacc_21663;
        int64_t inpacc_21186 = (int64_t) 0;
        
        for (int64_t i_21739 = 0; i_21739 < loop_dz2081Uz2083U_20165; i_21739++) {
            bool eta_p_21775 = ((bool *) mem_21877)[i_21739];
            
            // src/bgpc_vv_coloring.fut:18:28-39
            
            int64_t bool_to_i64_res_21776 = btoi_bool_i64(eta_p_21775);
            int64_t eta_p_21788 = ((int64_t *) mem_21876)[i_21739];
            int64_t eta_p_21789 = ((int64_t *) mem_21874)[i_21739];
            int64_t v_21792 = ((int64_t *) mem_param_21851.mem)[i_21739];
            
            // src/bgpc_vv_coloring.fut:19:29-32
            
            int64_t defunc_0_op_res_21793 = add64(inpacc_21186, bool_to_i64_res_21776);
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            bool cond_21794 = eta_p_21788 == (int64_t) 1;
            
            // src/bgpc_vv_coloring.fut:27:5-44
            
            int64_t lifted_lambda_res_21795;
            
            if (cond_21794) {
                // src/bgpc_vv_coloring.fut:27:5-44
                
                int64_t lifted_lambda_res_t_res_21796 = sub64(eta_p_21789, (int64_t) 1);
                
                lifted_lambda_res_21795 = lifted_lambda_res_t_res_21796;
            } else {
                lifted_lambda_res_21795 = (int64_t) -1;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_21795) && slt64(lifted_lambda_res_21795, m_20379)) {
                ((int64_t *) mem_21899)[lifted_lambda_res_21795] = v_21792;
            }
            // src/bgpc_vv_coloring.fut:27:5-44
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_21795) && slt64(lifted_lambda_res_21795, m_20379)) {
                ((int64_t *) mem_21897)[lifted_lambda_res_21795] = defunc_0_op_res_21793;
            }
            ((int64_t *) mem_21901)[i_21739] = defunc_0_op_res_21793;
            
            int64_t inpacc_tmp_21992 = defunc_0_op_res_21793;
            
            inpacc_21186 = inpacc_tmp_21992;
        }
        inpacc_21663 = inpacc_21186;
        // src/bgpc_vv_coloring.fut:27:5-44
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_21909, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_21901, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {loop_dz2081Uz2083U_20165});
        // src/bgpc_vv_coloring.fut:23:10-29
        
        int64_t out_szz_f_res_20366;
        
        if (x_20359) {
            // src/bgpc_vv_coloring.fut:171:23-55
            
            int64_t x_21669 = ((int64_t *) mem_21909)[tmp_20360];
            
            out_szz_f_res_20366 = x_21669;
        } else {
            out_szz_f_res_20366 = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:21:5-23:29
        
        int64_t out_szz_20368;
        
        if (cond_20358) {
            out_szz_20368 = (int64_t) 0;
        } else {
            out_szz_20368 = out_szz_f_res_20366;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        
        int64_t bytes_21910 = (int64_t) 8 * out_szz_20368;
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        bool cond_20265 = slt64(defunc_0_reduce_res_21672, (int64_t) 0);
        
        // src/bgpc_vv_coloring.fut:108:5-110:47
        
        int64_t next_color_bound_20266;
        
        if (cond_20265) {
            next_color_bound_20266 = color_bound_20169;
        } else {
            // src/bgpc_vv_coloring.fut:110:40-46
            
            int64_t max_arg1_20267 = add64((int64_t) 2, defunc_0_reduce_res_21672);
            
            // src/bgpc_vv_coloring.fut:110:10-46
            
            int64_t max_res_20268 = smax64(color_bound_20169, max_arg1_20267);
            
            next_color_bound_20266 = max_res_20268;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        if (memblock_alloc(ctx, &mem_21911, bytes_21910, "mem_21911")) {
            err = 1;
            goto cleanup;
        }
        // src/bgpc_vv_coloring.fut:32:22-43
        for (int64_t nest_i_21996 = 0; nest_i_21996 < out_szz_20368; nest_i_21996++) {
            ((int64_t *) mem_21911.mem)[nest_i_21996] = (int64_t) 0;
        }
        // src/bgpc_vv_coloring.fut:33:6-27
        
        bool acc_cert_20895;
        
        // src/bgpc_vv_coloring.fut:29:21-33:27
        for (int64_t i_21744 = 0; i_21744 < m_20379; i_21744++) {
            int64_t eta_p_20907 = ((int64_t *) mem_21897)[i_21744];
            int64_t v_20909 = ((int64_t *) mem_21899)[i_21744];
            
            // src/bgpc_vv_coloring.fut:29:44-50
            
            int64_t lifted_lambda_res_20910 = sub64(eta_p_20907, (int64_t) 1);
            
            // src/bgpc_vv_coloring.fut:33:6-27
            // UpdateAcc
            if (sle64((int64_t) 0, lifted_lambda_res_20910) && slt64(lifted_lambda_res_20910, out_szz_20368)) {
                ((int64_t *) mem_21911.mem)[lifted_lambda_res_20910] = v_20909;
            }
        }
        // src/bgpc_vv_coloring.fut:175:17-48
        
        bool acc_cert_20416;
        
        // src/bgpc_vv_coloring.fut:175:17-48
        for (int64_t i_21746 = 0; i_21746 < out_szz_20368; i_21746++) {
            int64_t v_20420 = ((int64_t *) mem_21911.mem)[i_21746];
            
            // src/bgpc_vv_coloring.fut:175:17-48
            // UpdateAcc
            if (sle64((int64_t) 0, v_20420) && slt64(v_20420, (int64_t) 2)) {
                ((int64_t *) mem_21841)[v_20420] = (int64_t) -1;
            }
        }
        // src/bgpc_vv_coloring.fut:198:20-23
        
        bool loop_cond_20422 = slt64((int64_t) 0, out_szz_20368);
        
        if (memblock_set(ctx, &mem_param_tmp_21968, &mem_21911, "mem_21911") != 0)
            return 1;
        
        int64_t loop_dz2081Uz2083U_tmp_21969 = out_szz_20368;
        bool loop_while_tmp_21970 = loop_cond_20422;
        int64_t color_bound_tmp_21973 = next_color_bound_20266;
        
        if (memblock_set(ctx, &mem_param_21851, &mem_param_tmp_21968, "mem_param_tmp_21968") != 0)
            return 1;
        loop_dz2081Uz2083U_20165 = loop_dz2081Uz2083U_tmp_21969;
        loop_while_20166 = loop_while_tmp_21970;
        color_bound_20169 = color_bound_tmp_21973;
    }
    if (memblock_set(ctx, &ext_mem_21916, &mem_param_21851, "mem_param_21851") != 0)
        return 1;
    vv_color_side_order_res_20160 = loop_dz2081Uz2083U_20165;
    vv_color_side_order_res_20161 = loop_while_20166;
    vv_color_side_order_res_20164 = color_bound_20169;
    if (memblock_unref(ctx, &mem_21840, "mem_21840") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21844, "ext_mem_21844") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21845, "ext_mem_21845") != 0)
        return 1;
    // src/sparse_jacobian_vjp.fut:78:8-87:44
    
    int64_t x_21680;
    int64_t redout_21747 = (int64_t) 0;
    
    for (int64_t i_21748 = 0; i_21748 < (int64_t) 2; i_21748++) {
        int64_t x_20425 = ((int64_t *) mem_21841)[i_21748];
        
        // src/sparse_jacobian_vjp.fut:14:22-29
        
        int64_t max_res_20428 = smax64(x_20425, redout_21747);
        int64_t redout_tmp_21999 = max_res_20428;
        
        redout_21747 = redout_tmp_21999;
    }
    x_21680 = redout_21747;
    // src/sparse_jacobian_vjp.fut:14:13-45
    
    int64_t num_colors_of_res_f_res_20429 = add64((int64_t) 1, x_21680);
    
    // src/sparse_jacobian_vjp.fut:45:5-53:27
    
    bool compressed_to_csr_vals_res_20450;
    int64_t compressed_to_csr_vals_res_20452;
    bool loop_while_20453;
    int64_t i_20455;
    
    loop_while_20453 = 1;
    i_20455 = (int64_t) 0;
    while (loop_while_20453) {
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool x_20456 = sle64((int64_t) 0, i_20455);
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool y_20457 = slt64(i_20455, csr_bipartite_from_pattern_res_20119);
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool bounds_check_20458 = x_20456 && y_20457;
        
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        bool index_certs_20459;
        
        if (!bounds_check_20458) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_20455, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20119, "].", "-> #0  src/sparse_jacobian_vjp.fut:47:15-26\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:47:15-26
        
        int64_t s_20460 = ((int64_t *) ext_mem_21829.mem)[i_20455];
        
        // src/sparse_jacobian_vjp.fut:48:25-27
        
        int64_t e_20461 = add64((int64_t) 1, i_20455);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool x_20462 = sle64((int64_t) 0, e_20461);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool y_20463 = slt64(e_20461, csr_bipartite_from_pattern_res_20119);
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool bounds_check_20464 = x_20462 && y_20463;
        
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        bool index_certs_20465;
        
        if (!bounds_check_20464) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_20461, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20119, "].", "-> #0  src/sparse_jacobian_vjp.fut:48:15-28\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:48:15-28
        
        int64_t e_20466 = ((int64_t *) ext_mem_21829.mem)[e_20461];
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t j_m_i_20467 = sub64(e_20466, s_20460);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool empty_slice_20468 = j_m_i_20467 == (int64_t) 0;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t m_20469 = sub64(j_m_i_20467, (int64_t) 1);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        int64_t i_p_m_t_s_20470 = add64(s_20460, m_20469);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_20471 = sle64((int64_t) 0, i_p_m_t_s_20470);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_20472 = slt64(i_p_m_t_s_20470, csr_bipartite_from_pattern_res_20120);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool zzero_lte_i_20473 = sle64((int64_t) 0, s_20460);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool i_lte_j_20474 = sle64(s_20460, e_20466);
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool y_20475 = i_p_m_t_s_leq_w_20472 && zzero_lte_i_20473;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool y_20476 = zzero_leq_i_p_m_t_s_20471 && y_20475;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool forwards_ok_20477 = i_lte_j_20474 && y_20476;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool ok_or_empty_20478 = empty_slice_20468 || forwards_ok_20477;
        
        // src/sparse_jacobian_vjp.fut:49:18-30
        
        bool index_certs_20479;
        
        if (!ok_or_empty_20478) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_20460, ":", (long long) e_20466, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20120, "].", "-> #0  src/sparse_jacobian_vjp.fut:49:18-30\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool y_20481 = slt64(i_20455, (int64_t) 2);
        
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool bounds_check_20482 = x_20456 && y_20481;
        
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        bool index_certs_20483;
        
        if (!bounds_check_20482) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_20455, "] out of bounds for array of shape [", (long long) (int64_t) 2, "].", "-> #0  src/sparse_jacobian_vjp.fut:50:16-29\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:50:16-29
        
        int64_t rc_20484 = ((int64_t *) mem_21841)[i_20455];
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool y_20486 = slt64(rc_20484, num_colors_of_res_f_res_20429);
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool x_20485 = sle64((int64_t) 0, rc_20484);
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool bounds_check_20487 = x_20485 && y_20486;
        
        // src/sparse_jacobian_vjp.fut:51:28-34
        
        bool index_certs_20488;
        
        if (!bounds_check_20487) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) rc_20484, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_20429, "].", "-> #0  src/sparse_jacobian_vjp.fut:51:28-34\n   #1  src/sparse_jacobian_vjp.fut:108:18-114:67\n   #2  src/sparse_jacobian_vjp.fut:132:8-137:24\n   #3  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:51:17-43
        for (int64_t nest_i_22003 = 0; nest_i_22003 < j_m_i_20467; nest_i_22003++) {
            ((double *) mem_21831)[s_20460 + nest_i_22003] = 0.0;
        }
        // src/sparse_jacobian_vjp.fut:46:13-16
        
        bool loop_cond_20497 = slt64(e_20461, (int64_t) 2);
        bool loop_while_tmp_22000 = loop_cond_20497;
        int64_t i_tmp_22002 = e_20461;
        
        loop_while_20453 = loop_while_tmp_22000;
        i_20455 = i_tmp_22002;
    }
    compressed_to_csr_vals_res_20450 = loop_while_20453;
    compressed_to_csr_vals_res_20452 = i_20455;
    // src/sparse_jacobian_vjp.fut:68:29-47
    if (mem_21926_cached_sizze_22145 < (int64_t) 24) {
        err = lexical_realloc(ctx, &mem_21926, &mem_21926_cached_sizze_22145, (int64_t) 24);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // test/test_sparse_jacobian_vjp.fut:7:19-9:39
    
    bool defunc_0_reduce_res_21681;
    bool redout_21753 = 1;
    
    for (int64_t i_21754 = 0; i_21754 < (int64_t) 2; i_21754++) {
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        bool y_20818 = slt64(i_21754, csr_bipartite_from_pattern_res_20119);
        
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        bool index_certs_20820;
        
        if (!y_20818) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_21754, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20119, "].", "-> #0  src/sparse_jacobian_vjp.fut:64:17-28\n   #1  src/sparse_jacobian_vjp.fut:132:18-138:40\n   #2  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:64:17-28
        
        int64_t s_20821 = ((int64_t *) ext_mem_21829.mem)[i_21754];
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool zzero_lte_i_20834 = sle64((int64_t) 0, s_20821);
        
        // src/sparse_jacobian_vjp.fut:65:27-29
        
        int64_t e_20822 = add64((int64_t) 1, i_21754);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool y_20824 = slt64(e_20822, csr_bipartite_from_pattern_res_20119);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool x_20823 = sle64((int64_t) 0, e_20822);
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool bounds_check_20825 = x_20823 && y_20824;
        
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        bool index_certs_20826;
        
        if (!bounds_check_20825) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_20822, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20119, "].", "-> #0  src/sparse_jacobian_vjp.fut:65:17-30\n   #1  src/sparse_jacobian_vjp.fut:132:18-138:40\n   #2  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:65:17-30
        
        int64_t e_20827 = ((int64_t *) ext_mem_21829.mem)[e_20822];
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t j_m_i_20828 = sub64(e_20827, s_20821);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t m_20830 = sub64(j_m_i_20828, (int64_t) 1);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        int64_t i_p_m_t_s_20831 = add64(s_20821, m_20830);
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool i_p_m_t_s_leq_w_20833 = slt64(i_p_m_t_s_20831, csr_bipartite_from_pattern_res_20120);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool y_20842 = i_p_m_t_s_leq_w_20833 && zzero_lte_i_20834;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool zzero_leq_i_p_m_t_s_20832 = sle64((int64_t) 0, i_p_m_t_s_20831);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool y_20843 = zzero_leq_i_p_m_t_s_20832 && y_20842;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool i_lte_j_20835 = sle64(s_20821, e_20827);
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool forwards_ok_20844 = i_lte_j_20835 && y_20843;
        
        // src/sparse_jacobian_vjp.fut:66:20-32
        
        bool empty_slice_20829 = j_m_i_20828 == (int64_t) 0;
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool ok_or_empty_20845 = empty_slice_20829 || forwards_ok_20844;
        
        // src/sparse_jacobian_vjp.fut:67:20-29
        
        bool index_certs_20846;
        
        if (!ok_or_empty_20845) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_20821, ":", (long long) e_20827, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_20120, "].", "-> #0  src/sparse_jacobian_vjp.fut:67:20-29\n   #1  src/sparse_jacobian_vjp.fut:132:18-138:40\n   #2  test/test_sparse_jacobian_vjp.fut:61:21-76:58\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_vjp.fut:68:29-47
        for (int64_t nest_i_22005 = 0; nest_i_22005 < (int64_t) 3; nest_i_22005++) {
            ((double *) mem_21926)[nest_i_22005] = 0.0;
        }
        // src/sparse_jacobian_vjp.fut:69:12-32
        
        bool acc_cert_20850;
        
        // src/sparse_jacobian_vjp.fut:69:12-32
        for (int64_t i_21750 = 0; i_21750 < j_m_i_20828; i_21750++) {
            int64_t index_primexp_21768 = s_20821 + i_21750;
            int64_t v_20854 = ((int64_t *) ext_mem_21828.mem)[index_primexp_21768];
            double v_20855 = ((double *) mem_21831)[index_primexp_21768];
            
            // src/sparse_jacobian_vjp.fut:69:12-32
            // UpdateAcc
            if (sle64((int64_t) 0, v_20854) && slt64(v_20854, (int64_t) 3)) {
                ((double *) mem_21926)[v_20854] = v_20855;
            }
        }
        // test/test_sparse_jacobian_vjp.fut:8:5-66
        
        bool defunc_0_reduce_res_21678;
        bool redout_21751 = 1;
        
        for (int64_t i_21752 = 0; i_21752 < (int64_t) 3; i_21752++) {
            double eta_p_20880 = ((double *) mem_21926)[i_21752];
            
            // test/test_sparse_jacobian_vjp.fut:8:37-51
            
            double abs_res_20884 = fabs64(eta_p_20880);
            
            // test/test_sparse_jacobian_vjp.fut:8:53-59
            
            bool lifted_lambda_res_20885 = abs_res_20884 <= 1.0e-9;
            
            // test/test_sparse_jacobian_vjp.fut:8:5-66
            
            bool x_20868 = lifted_lambda_res_20885 && redout_21751;
            bool redout_tmp_22007 = x_20868;
            
            redout_21751 = redout_tmp_22007;
        }
        defunc_0_reduce_res_21678 = redout_21751;
        // test/test_sparse_jacobian_vjp.fut:9:6-39
        
        bool x_20609 = defunc_0_reduce_res_21678 && redout_21753;
        bool redout_tmp_22004 = x_20609;
        
        redout_21753 = redout_tmp_22004;
    }
    defunc_0_reduce_res_21681 = redout_21753;
    if (memblock_unref(ctx, &ext_mem_21828, "ext_mem_21828") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_21829, "ext_mem_21829") != 0)
        return 1;
    prim_out_21962 = defunc_0_reduce_res_21681;
    *out_prim_out_22132 = prim_out_21962;
    
  cleanup:
    {
        free(mem_21831);
        free(mem_21832);
        free(mem_21841);
        free(mem_21853);
        free(mem_21857);
        free(mem_21874);
        free(mem_21876);
        free(mem_21877);
        free(mem_21897);
        free(mem_21899);
        free(mem_21901);
        free(mem_21909);
        free(mem_21926);
        if (memblock_unref(ctx, &mem_param_tmp_21968, "mem_param_tmp_21968") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21911, "mem_21911") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_param_21851, "mem_param_21851") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21916, "ext_mem_21916") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21844, "ext_mem_21844") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21845, "ext_mem_21845") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21842, "mem_21842") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_21840, "mem_21840") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21828, "ext_mem_21828") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_21829, "ext_mem_21829") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_test_sparse_vjp_ex1_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0)
{
    bool prim_out_21962 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock x_mem_21827;
    
    x_mem_21827.references = NULL;
    x_mem_21827 = in0->mem;
    if (!((int64_t) 5 == in0->shape[0])) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_test_sparse_vjp_ex1_matches_dense(ctx, &prim_out_21962, x_mem_21827);
        if (ret == 0) {
            struct memblock mem_21809 = ctx->constants->mem_21809;
            struct memblock mem_21812 = ctx->constants->mem_21812;
            struct memblock mem_21814 = ctx->constants->mem_21814;
            struct memblock mem_21819 = ctx->constants->mem_21819;
            struct memblock mem_21820 = ctx->constants->mem_21820;
            struct memblock mem_21826 = ctx->constants->mem_21826;
            
            *out0 = prim_out_21962;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test_sparse_vjp_ex2_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0)
{
    bool prim_out_21962 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock x_mem_21827;
    
    x_mem_21827.references = NULL;
    x_mem_21827 = in0->mem;
    if (!((int64_t) 4 == in0->shape[0])) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_test_sparse_vjp_ex2_matches_dense(ctx, &prim_out_21962, x_mem_21827);
        if (ret == 0) {
            struct memblock mem_21809 = ctx->constants->mem_21809;
            struct memblock mem_21812 = ctx->constants->mem_21812;
            struct memblock mem_21814 = ctx->constants->mem_21814;
            struct memblock mem_21819 = ctx->constants->mem_21819;
            struct memblock mem_21820 = ctx->constants->mem_21820;
            struct memblock mem_21826 = ctx->constants->mem_21826;
            
            *out0 = prim_out_21962;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test_sparse_vjp_ex4_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0)
{
    bool prim_out_21962 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock x_mem_21827;
    
    x_mem_21827.references = NULL;
    x_mem_21827 = in0->mem;
    if (!((int64_t) 6 == in0->shape[0])) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_test_sparse_vjp_ex4_matches_dense(ctx, &prim_out_21962, x_mem_21827);
        if (ret == 0) {
            struct memblock mem_21809 = ctx->constants->mem_21809;
            struct memblock mem_21812 = ctx->constants->mem_21812;
            struct memblock mem_21814 = ctx->constants->mem_21814;
            struct memblock mem_21819 = ctx->constants->mem_21819;
            struct memblock mem_21820 = ctx->constants->mem_21820;
            struct memblock mem_21826 = ctx->constants->mem_21826;
            
            *out0 = prim_out_21962;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test_sparse_vjp_ex4_with_row_colors_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0)
{
    bool prim_out_21962 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock x_mem_21827;
    
    x_mem_21827.references = NULL;
    x_mem_21827 = in0->mem;
    if (!((int64_t) 6 == in0->shape[0])) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_test_sparse_vjp_ex4_with_row_colors_matches_dense(ctx, &prim_out_21962, x_mem_21827);
        if (ret == 0) {
            struct memblock mem_21809 = ctx->constants->mem_21809;
            struct memblock mem_21812 = ctx->constants->mem_21812;
            struct memblock mem_21814 = ctx->constants->mem_21814;
            struct memblock mem_21819 = ctx->constants->mem_21819;
            struct memblock mem_21820 = ctx->constants->mem_21820;
            struct memblock mem_21826 = ctx->constants->mem_21826;
            
            *out0 = prim_out_21962;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test_sparse_vjp_ex5_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0)
{
    bool prim_out_21962 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock x_mem_21827;
    
    x_mem_21827.references = NULL;
    x_mem_21827 = in0->mem;
    if (!((int64_t) 5 == in0->shape[0])) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_test_sparse_vjp_ex5_matches_dense(ctx, &prim_out_21962, x_mem_21827);
        if (ret == 0) {
            struct memblock mem_21809 = ctx->constants->mem_21809;
            struct memblock mem_21812 = ctx->constants->mem_21812;
            struct memblock mem_21814 = ctx->constants->mem_21814;
            struct memblock mem_21819 = ctx->constants->mem_21819;
            struct memblock mem_21820 = ctx->constants->mem_21820;
            struct memblock mem_21826 = ctx->constants->mem_21826;
            
            *out0 = prim_out_21962;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_test_sparse_vjp_zero_pattern_matches_dense(struct futhark_context *ctx, bool *out0, const struct futhark_f64_1d *in0)
{
    bool prim_out_21962 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    
    struct memblock x_mem_21827;
    
    x_mem_21827.references = NULL;
    x_mem_21827 = in0->mem;
    if (!((int64_t) 3 == in0->shape[0])) {
        ret = 1;
        set_error(ctx, msgprintf("Error: entry point arguments have invalid sizes.\n"));
    }
    if (ret == 0) {
        ret = futrts_entry_test_sparse_vjp_zzero_pattern_matches_dense(ctx, &prim_out_21962, x_mem_21827);
        if (ret == 0) {
            struct memblock mem_21809 = ctx->constants->mem_21809;
            struct memblock mem_21812 = ctx->constants->mem_21812;
            struct memblock mem_21814 = ctx->constants->mem_21814;
            struct memblock mem_21819 = ctx->constants->mem_21819;
            struct memblock mem_21820 = ctx->constants->mem_21820;
            struct memblock mem_21826 = ctx->constants->mem_21826;
            
            *out0 = prim_out_21962;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
