
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


// Opaque values



// Entry points
int futhark_entry_test_ba_d2_matches_dense_tiny(struct futhark_context *ctx, bool *out0, const int32_t in0);

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

const struct type *test_ba_d2_matches_dense_tiny_out_types[] = {&type_bool, NULL};
bool test_ba_d2_matches_dense_tiny_out_unique[] = {false};
const struct type *test_ba_d2_matches_dense_tiny_in_types[] = {&type_i32, NULL};
bool test_ba_d2_matches_dense_tiny_in_unique[] = {false};
const char *test_ba_d2_matches_dense_tiny_tuning_params[] = {NULL};
int call_test_ba_d2_matches_dense_tiny(struct futhark_context *ctx, void **outs, void **ins)
{
    bool *out0 = outs[0];
    int32_t in0 = *(int32_t *) ins[0];
    
    return futhark_entry_test_ba_d2_matches_dense_tiny(ctx, out0, in0);
}
const struct type *types[] = {&type_i8, &type_i16, &type_i32, &type_i64, &type_u8, &type_u16, &type_u32, &type_u64, &type_f16, &type_f32, &type_f64, &type_bool, NULL};
struct entry_point entry_points[] = {{.name ="test_ba_d2_matches_dense_tiny", .f =call_test_ba_d2_matches_dense_tiny, .tuning_params =test_ba_d2_matches_dense_tiny_tuning_params, .in_types =test_ba_d2_matches_dense_tiny_in_types, .out_types =test_ba_d2_matches_dense_tiny_out_types, .in_unique =test_ba_d2_matches_dense_tiny_in_unique, .out_unique =test_ba_d2_matches_dense_tiny_out_unique}, {.name =NULL}};
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
};
static double test_ba_d2_matches_dense_tinyzistatic_array_realtype_24066[2] = { 0.0,0.0};
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

FUTHARK_FUN_ATTR int futrts_csr_rows_from_pattern_9525(struct futhark_context *ctx, struct memblock *mem_out_p_23969, struct memblock *mem_out_p_23970, int64_t *out_prim_out_23971, int64_t *out_prim_out_23972, struct memblock pat_mem_23183, int64_t m_14508, int64_t n_14509);
FUTHARK_FUN_ATTR int futrts_entry_test_ba_d2_matches_dense_tiny(struct futhark_context *ctx, bool *out_prim_out_23976, int32_t _dummy_18303);

static int init_constants(struct futhark_context *ctx)
{
    (void) ctx;
    
    int err = 0;
    
    
  cleanup:
    return err;
}
static int free_constants(struct futhark_context *ctx)
{
    (void) ctx;
    return 0;
}

FUTHARK_FUN_ATTR int futrts_csr_rows_from_pattern_9525(struct futhark_context *ctx, struct memblock *mem_out_p_23969, struct memblock *mem_out_p_23970, int64_t *out_prim_out_23971, int64_t *out_prim_out_23972, struct memblock pat_mem_23183, int64_t m_14508, int64_t n_14509)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_23185_cached_sizze_23973 = 0;
    unsigned char *mem_23185 = NULL;
    int64_t mem_23195_cached_sizze_23974 = 0;
    unsigned char *mem_23195 = NULL;
    int64_t mem_23197_cached_sizze_23975 = 0;
    unsigned char *mem_23197 = NULL;
    struct memblock mem_23211;
    
    mem_23211.references = NULL;
    
    struct memblock mem_23193;
    
    mem_23193.references = NULL;
    
    struct memblock mem_out_23857;
    
    mem_out_23857.references = NULL;
    
    struct memblock mem_out_23856;
    
    mem_out_23856.references = NULL;
    
    int64_t prim_out_23858;
    int64_t prim_out_23859;
    int64_t dzlz7bUZLztZRz20Umz20Unz7dUzg_14511 = mul64(m_14508, n_14509);
    
    // src/pattern_csr.fut:7:23-40
    
    int64_t bytes_23184 = (int64_t) 8 * m_14508;
    
    // src/pattern_csr.fut:8:17-20
    
    int64_t dzlz7bUZLzpZRz20Umz20U1z7dUzg_18601 = add64((int64_t) 1, m_14508);
    
    // src/pattern_csr.fut:8:6-26
    
    int64_t bytes_23192 = (int64_t) 8 * dzlz7bUZLzpZRz20Umz20U1z7dUzg_18601;
    
    // src/pattern_csr.fut:8:6-46
    
    bool empty_slice_18608 = m_14508 == (int64_t) 0;
    
    // src/pattern_csr.fut:8:6-46
    
    bool i_p_m_t_s_leq_w_18609 = slt64(m_14508, dzlz7bUZLzpZRz20Umz20U1z7dUzg_18601);
    
    // src/pattern_csr.fut:8:6-46
    
    bool i_lte_j_18610 = sle64((int64_t) 1, dzlz7bUZLzpZRz20Umz20U1z7dUzg_18601);
    
    // src/pattern_csr.fut:8:6-46
    
    bool forwards_ok_18611 = i_p_m_t_s_leq_w_18609 && i_lte_j_18610;
    
    // src/pattern_csr.fut:8:6-46
    
    bool ok_or_empty_18612 = empty_slice_18608 || forwards_ok_18611;
    
    // src/pattern_csr.fut:8:6-46
    
    bool index_certs_18613;
    
    if (!ok_or_empty_18612) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) (int64_t) 1, ":", (long long) dzlz7bUZLzpZRz20Umz20U1z7dUzg_18601, "] out of bounds for array of shape [", (long long) dzlz7bUZLzpZRz20Umz20U1z7dUzg_18601, "].", "-> #0  src/pattern_csr.fut:8:6-46\n   #1  src/pattern_csr.fut:12:27-48\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    
    int64_t bytes_23194 = (int64_t) 8 * dzlz7bUZLztZRz20Umz20Unz7dUzg_14511;
    
    // src/pattern_csr.fut:19:34-61
    if (mem_23195_cached_sizze_23974 < bytes_23194) {
        err = lexical_realloc(ctx, &mem_23195, &mem_23195_cached_sizze_23974, bytes_23194);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    if (mem_23197_cached_sizze_23975 < bytes_23194) {
        err = lexical_realloc(ctx, &mem_23197, &mem_23197_cached_sizze_23975, bytes_23194);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    
    int64_t discard_22873;
    int64_t scanacc_22867 = (int64_t) 0;
    
    for (int64_t i_22870 = 0; i_22870 < dzlz7bUZLztZRz20Umz20Unz7dUzg_14511; i_22870++) {
        int64_t new_index_23143 = squot64(i_22870, n_14509);
        int64_t binop_y_23145 = n_14509 * new_index_23143;
        int64_t new_index_23146 = i_22870 - binop_y_23145;
        bool eta_p_20913 = ((bool *) pat_mem_23183.mem)[new_index_23143 * n_14509 + new_index_23146];
        
        // src/pattern_csr.fut:19:34-61
        
        int64_t defunc_0_f_res_20914 = btoi_bool_i64(eta_p_20913);
        
        // src/pattern_csr.fut:19:34-61
        
        int64_t defunc_0_op_res_18740 = add64(defunc_0_f_res_20914, scanacc_22867);
        
        ((int64_t *) mem_23195)[i_22870] = defunc_0_op_res_18740;
        ((int64_t *) mem_23197)[i_22870] = defunc_0_f_res_20914;
        
        int64_t scanacc_tmp_23860 = defunc_0_op_res_18740;
        
        scanacc_22867 = scanacc_tmp_23860;
    }
    discard_22873 = scanacc_22867;
    // src/pattern_csr.fut:16:43-46
    
    bool zzero_18482 = n_14509 == (int64_t) 0;
    
    // src/pattern_csr.fut:16:43-46
    
    bool nonzzero_18483 = !zzero_18482;
    
    // src/pattern_csr.fut:16:43-46
    
    bool nonzzero_cert_18484;
    
    if (!nonzzero_18483) {
        set_error(ctx, msgprintf("Error: %s\n\nBacktrace:\n%s", "division by zero", "-> #0  src/pattern_csr.fut:16:43-46\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    
    int64_t tmp_18743 = sub64(dzlz7bUZLztZRz20Umz20Unz7dUzg_14511, (int64_t) 1);
    
    // src/pattern_csr.fut:19:34-61
    
    bool y_18745 = slt64(tmp_18743, dzlz7bUZLztZRz20Umz20Unz7dUzg_14511);
    
    // src/pattern_csr.fut:19:34-61
    
    bool x_18744 = sle64((int64_t) 0, tmp_18743);
    
    // src/pattern_csr.fut:19:34-61
    
    bool bounds_check_18746 = x_18744 && y_18745;
    
    // src/pattern_csr.fut:19:34-61
    
    bool cond_18741 = dzlz7bUZLztZRz20Umz20Unz7dUzg_14511 == (int64_t) 0;
    
    // src/pattern_csr.fut:19:34-61
    
    bool protect_assert_disj_18747 = cond_18741 || bounds_check_18746;
    
    // src/pattern_csr.fut:19:34-61
    
    bool index_certs_18748;
    
    if (!protect_assert_disj_18747) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_18743, "] out of bounds for array of shape [", (long long) dzlz7bUZLztZRz20Umz20Unz7dUzg_14511, "].", "-> #0  src/pattern_csr.fut:19:34-61\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    
    bool x_18742 = !cond_18741;
    
    // src/pattern_csr.fut:19:34-61
    
    int64_t m_f_res_18749;
    
    if (x_18742) {
        // src/pattern_csr.fut:19:34-61
        
        int64_t x_22430 = ((int64_t *) mem_23195)[tmp_18743];
        
        m_f_res_18749 = x_22430;
    } else {
        m_f_res_18749 = (int64_t) 0;
    }
    // src/pattern_csr.fut:19:34-61
    
    int64_t m_18751;
    
    if (cond_18741) {
        m_18751 = (int64_t) 0;
    } else {
        m_18751 = m_f_res_18749;
    }
    // src/pattern_csr.fut:19:34-61
    
    int64_t bytes_23210 = (int64_t) 8 * m_18751;
    
    // src/pattern_csr.fut:7:23-40
    if (mem_23185_cached_sizze_23973 < bytes_23184) {
        err = lexical_realloc(ctx, &mem_23185, &mem_23185_cached_sizze_23973, bytes_23184);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/pattern_csr.fut:7:23-40
    
    int64_t discard_22864;
    int64_t scanacc_22860 = (int64_t) 0;
    
    for (int64_t i_22862 = 0; i_22862 < m_14508; i_22862++) {
        // src/pattern_csr.fut:4:3-57
        
        int64_t defunc_0_reduce_res_22429;
        int64_t redout_22857 = (int64_t) 0;
        
        for (int64_t i_22858 = 0; i_22858 < n_14509; i_22858++) {
            bool eta_p_20964 = ((bool *) pat_mem_23183.mem)[i_22862 * n_14509 + i_22858];
            
            // src/pattern_csr.fut:4:17-57
            
            int64_t lifted_lambda_res_20965 = btoi_bool_i64(eta_p_20964);
            
            // src/pattern_csr.fut:4:10-13
            
            int64_t defunc_0_op_res_20961 = add64(lifted_lambda_res_20965, redout_22857);
            int64_t redout_tmp_23865 = defunc_0_op_res_20961;
            
            redout_22857 = redout_tmp_23865;
        }
        defunc_0_reduce_res_22429 = redout_22857;
        // src/pattern_csr.fut:7:28-31
        
        int64_t defunc_0_op_res_18606 = add64(defunc_0_reduce_res_22429, scanacc_22860);
        
        ((int64_t *) mem_23185)[i_22862] = defunc_0_op_res_18606;
        
        int64_t scanacc_tmp_23863 = defunc_0_op_res_18606;
        
        scanacc_22860 = scanacc_tmp_23863;
    }
    discard_22864 = scanacc_22860;
    // src/pattern_csr.fut:8:6-26
    if (memblock_alloc(ctx, &mem_23193, bytes_23192, "mem_23193")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:8:6-26
    for (int64_t nest_i_23866 = 0; nest_i_23866 < dzlz7bUZLzpZRz20Umz20U1z7dUzg_18601; nest_i_23866++) {
        ((int64_t *) mem_23193.mem)[nest_i_23866] = (int64_t) 0;
    }
    // src/pattern_csr.fut:8:6-46
    // src/pattern_csr.fut:8:6-46
    lmad_copy_8b(ctx, 1, (uint64_t *) mem_23193.mem, (int64_t) 1, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_23185, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {m_14508});
    // src/pattern_csr.fut:19:34-61
    if (memblock_alloc(ctx, &mem_23211, bytes_23210, "mem_23211")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:19:34-61
    
    bool acc_cert_20919;
    
    // src/pattern_csr.fut:16:30-19:61
    for (int64_t i_22875 = 0; i_22875 < dzlz7bUZLztZRz20Umz20Unz7dUzg_14511; i_22875++) {
        int64_t eta_p_20938 = ((int64_t *) mem_23197)[i_22875];
        int64_t eta_p_20939 = ((int64_t *) mem_23195)[i_22875];
        
        // src/pattern_csr.fut:16:43-46
        
        int64_t lifted_lambda_res_20941 = smod64(i_22875, n_14509);
        
        // src/pattern_csr.fut:19:34-61
        
        bool cond_20943 = eta_p_20938 == (int64_t) 1;
        
        // src/pattern_csr.fut:19:34-61
        
        int64_t lifted_lambda_res_20944;
        
        if (cond_20943) {
            // src/pattern_csr.fut:19:34-61
            
            int64_t lifted_lambda_res_t_res_22431 = sub64(eta_p_20939, (int64_t) 1);
            
            lifted_lambda_res_20944 = lifted_lambda_res_t_res_22431;
        } else {
            lifted_lambda_res_20944 = (int64_t) -1;
        }
        // src/pattern_csr.fut:19:34-61
        // UpdateAcc
        if (sle64((int64_t) 0, lifted_lambda_res_20944) && slt64(lifted_lambda_res_20944, m_18751)) {
            ((int64_t *) mem_23211.mem)[lifted_lambda_res_20944] = lifted_lambda_res_20941;
        }
    }
    if (memblock_set(ctx, &mem_out_23856, &mem_23193, "mem_23193") != 0)
        return 1;
    if (memblock_set(ctx, &mem_out_23857, &mem_23211, "mem_23211") != 0)
        return 1;
    prim_out_23858 = dzlz7bUZLzpZRz20Umz20U1z7dUzg_18601;
    prim_out_23859 = m_18751;
    if (memblock_set(ctx, &*mem_out_p_23969, &mem_out_23856, "mem_out_23856") != 0)
        return 1;
    if (memblock_set(ctx, &*mem_out_p_23970, &mem_out_23857, "mem_out_23857") != 0)
        return 1;
    *out_prim_out_23971 = prim_out_23858;
    *out_prim_out_23972 = prim_out_23859;
    
  cleanup:
    {
        free(mem_23185);
        free(mem_23195);
        free(mem_23197);
        if (memblock_unref(ctx, &mem_23211, "mem_23211") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_23193, "mem_23193") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_23857, "mem_out_23857") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_out_23856, "mem_out_23856") != 0)
            return 1;
    }
    return err;
}
FUTHARK_FUN_ATTR int futrts_entry_test_ba_d2_matches_dense_tiny(struct futhark_context *ctx, bool *out_prim_out_23976, int32_t _dummy_18303)
{
    (void) ctx;
    
    int err = 0;
    int64_t mem_23183_cached_sizze_23977 = 0;
    unsigned char *mem_23183 = NULL;
    int64_t mem_23193_cached_sizze_23978 = 0;
    unsigned char *mem_23193 = NULL;
    int64_t mem_23194_cached_sizze_23979 = 0;
    unsigned char *mem_23194 = NULL;
    int64_t mem_23201_cached_sizze_23980 = 0;
    unsigned char *mem_23201 = NULL;
    int64_t mem_23206_cached_sizze_23981 = 0;
    unsigned char *mem_23206 = NULL;
    int64_t mem_23225_cached_sizze_23982 = 0;
    unsigned char *mem_23225 = NULL;
    int64_t mem_23230_cached_sizze_23983 = 0;
    unsigned char *mem_23230 = NULL;
    int64_t mem_23231_cached_sizze_23984 = 0;
    unsigned char *mem_23231 = NULL;
    int64_t mem_23232_cached_sizze_23985 = 0;
    unsigned char *mem_23232 = NULL;
    int64_t mem_23241_cached_sizze_23986 = 0;
    unsigned char *mem_23241 = NULL;
    int64_t mem_23242_cached_sizze_23987 = 0;
    unsigned char *mem_23242 = NULL;
    int64_t mem_23263_cached_sizze_23988 = 0;
    unsigned char *mem_23263 = NULL;
    int64_t mem_23264_cached_sizze_23989 = 0;
    unsigned char *mem_23264 = NULL;
    int64_t mem_23265_cached_sizze_23990 = 0;
    unsigned char *mem_23265 = NULL;
    int64_t mem_23266_cached_sizze_23991 = 0;
    unsigned char *mem_23266 = NULL;
    int64_t mem_23267_cached_sizze_23992 = 0;
    unsigned char *mem_23267 = NULL;
    int64_t mem_23268_cached_sizze_23993 = 0;
    unsigned char *mem_23268 = NULL;
    int64_t mem_23287_cached_sizze_23994 = 0;
    unsigned char *mem_23287 = NULL;
    int64_t mem_23288_cached_sizze_23995 = 0;
    unsigned char *mem_23288 = NULL;
    int64_t mem_23319_cached_sizze_23996 = 0;
    unsigned char *mem_23319 = NULL;
    int64_t mem_23320_cached_sizze_23997 = 0;
    unsigned char *mem_23320 = NULL;
    int64_t mem_23321_cached_sizze_23998 = 0;
    unsigned char *mem_23321 = NULL;
    int64_t mem_23340_cached_sizze_23999 = 0;
    unsigned char *mem_23340 = NULL;
    int64_t mem_23341_cached_sizze_24000 = 0;
    unsigned char *mem_23341 = NULL;
    int64_t mem_23342_cached_sizze_24001 = 0;
    unsigned char *mem_23342 = NULL;
    int64_t mem_23343_cached_sizze_24002 = 0;
    unsigned char *mem_23343 = NULL;
    int64_t mem_23344_cached_sizze_24003 = 0;
    unsigned char *mem_23344 = NULL;
    int64_t mem_23345_cached_sizze_24004 = 0;
    unsigned char *mem_23345 = NULL;
    int64_t mem_23346_cached_sizze_24005 = 0;
    unsigned char *mem_23346 = NULL;
    int64_t mem_23347_cached_sizze_24006 = 0;
    unsigned char *mem_23347 = NULL;
    int64_t mem_23348_cached_sizze_24007 = 0;
    unsigned char *mem_23348 = NULL;
    int64_t mem_23349_cached_sizze_24008 = 0;
    unsigned char *mem_23349 = NULL;
    int64_t mem_23350_cached_sizze_24009 = 0;
    unsigned char *mem_23350 = NULL;
    int64_t mem_23351_cached_sizze_24010 = 0;
    unsigned char *mem_23351 = NULL;
    int64_t mem_23352_cached_sizze_24011 = 0;
    unsigned char *mem_23352 = NULL;
    int64_t mem_23353_cached_sizze_24012 = 0;
    unsigned char *mem_23353 = NULL;
    int64_t mem_23354_cached_sizze_24013 = 0;
    unsigned char *mem_23354 = NULL;
    int64_t mem_23355_cached_sizze_24014 = 0;
    unsigned char *mem_23355 = NULL;
    int64_t mem_23356_cached_sizze_24015 = 0;
    unsigned char *mem_23356 = NULL;
    int64_t mem_23357_cached_sizze_24016 = 0;
    unsigned char *mem_23357 = NULL;
    int64_t mem_23358_cached_sizze_24017 = 0;
    unsigned char *mem_23358 = NULL;
    int64_t mem_23359_cached_sizze_24018 = 0;
    unsigned char *mem_23359 = NULL;
    int64_t mem_23480_cached_sizze_24019 = 0;
    unsigned char *mem_23480 = NULL;
    int64_t mem_23481_cached_sizze_24020 = 0;
    unsigned char *mem_23481 = NULL;
    int64_t mem_23494_cached_sizze_24021 = 0;
    unsigned char *mem_23494 = NULL;
    int64_t mem_23506_cached_sizze_24022 = 0;
    unsigned char *mem_23506 = NULL;
    int64_t mem_23521_cached_sizze_24023 = 0;
    unsigned char *mem_23521 = NULL;
    int64_t mem_23522_cached_sizze_24024 = 0;
    unsigned char *mem_23522 = NULL;
    int64_t mem_23549_cached_sizze_24025 = 0;
    unsigned char *mem_23549 = NULL;
    int64_t mem_23554_cached_sizze_24026 = 0;
    unsigned char *mem_23554 = NULL;
    int64_t mem_23561_cached_sizze_24027 = 0;
    unsigned char *mem_23561 = NULL;
    int64_t mem_23562_cached_sizze_24028 = 0;
    unsigned char *mem_23562 = NULL;
    int64_t mem_23571_cached_sizze_24029 = 0;
    unsigned char *mem_23571 = NULL;
    int64_t mem_23572_cached_sizze_24030 = 0;
    unsigned char *mem_23572 = NULL;
    int64_t mem_23593_cached_sizze_24031 = 0;
    unsigned char *mem_23593 = NULL;
    int64_t mem_23594_cached_sizze_24032 = 0;
    unsigned char *mem_23594 = NULL;
    int64_t mem_23595_cached_sizze_24033 = 0;
    unsigned char *mem_23595 = NULL;
    int64_t mem_23596_cached_sizze_24034 = 0;
    unsigned char *mem_23596 = NULL;
    int64_t mem_23597_cached_sizze_24035 = 0;
    unsigned char *mem_23597 = NULL;
    int64_t mem_23598_cached_sizze_24036 = 0;
    unsigned char *mem_23598 = NULL;
    int64_t mem_23617_cached_sizze_24037 = 0;
    unsigned char *mem_23617 = NULL;
    int64_t mem_23618_cached_sizze_24038 = 0;
    unsigned char *mem_23618 = NULL;
    int64_t mem_23649_cached_sizze_24039 = 0;
    unsigned char *mem_23649 = NULL;
    int64_t mem_23650_cached_sizze_24040 = 0;
    unsigned char *mem_23650 = NULL;
    int64_t mem_23651_cached_sizze_24041 = 0;
    unsigned char *mem_23651 = NULL;
    int64_t mem_23670_cached_sizze_24042 = 0;
    unsigned char *mem_23670 = NULL;
    int64_t mem_23671_cached_sizze_24043 = 0;
    unsigned char *mem_23671 = NULL;
    int64_t mem_23672_cached_sizze_24044 = 0;
    unsigned char *mem_23672 = NULL;
    int64_t mem_23673_cached_sizze_24045 = 0;
    unsigned char *mem_23673 = NULL;
    int64_t mem_23674_cached_sizze_24046 = 0;
    unsigned char *mem_23674 = NULL;
    int64_t mem_23675_cached_sizze_24047 = 0;
    unsigned char *mem_23675 = NULL;
    int64_t mem_23676_cached_sizze_24048 = 0;
    unsigned char *mem_23676 = NULL;
    int64_t mem_23677_cached_sizze_24049 = 0;
    unsigned char *mem_23677 = NULL;
    int64_t mem_23678_cached_sizze_24050 = 0;
    unsigned char *mem_23678 = NULL;
    int64_t mem_23679_cached_sizze_24051 = 0;
    unsigned char *mem_23679 = NULL;
    int64_t mem_23680_cached_sizze_24052 = 0;
    unsigned char *mem_23680 = NULL;
    int64_t mem_23681_cached_sizze_24053 = 0;
    unsigned char *mem_23681 = NULL;
    int64_t mem_23682_cached_sizze_24054 = 0;
    unsigned char *mem_23682 = NULL;
    int64_t mem_23683_cached_sizze_24055 = 0;
    unsigned char *mem_23683 = NULL;
    int64_t mem_23684_cached_sizze_24056 = 0;
    unsigned char *mem_23684 = NULL;
    int64_t mem_23685_cached_sizze_24057 = 0;
    unsigned char *mem_23685 = NULL;
    int64_t mem_23686_cached_sizze_24058 = 0;
    unsigned char *mem_23686 = NULL;
    int64_t mem_23687_cached_sizze_24059 = 0;
    unsigned char *mem_23687 = NULL;
    int64_t mem_23688_cached_sizze_24060 = 0;
    unsigned char *mem_23688 = NULL;
    int64_t mem_23689_cached_sizze_24061 = 0;
    unsigned char *mem_23689 = NULL;
    int64_t mem_23810_cached_sizze_24062 = 0;
    unsigned char *mem_23810 = NULL;
    int64_t mem_23811_cached_sizze_24063 = 0;
    unsigned char *mem_23811 = NULL;
    int64_t mem_23824_cached_sizze_24064 = 0;
    unsigned char *mem_23824 = NULL;
    int64_t mem_23836_cached_sizze_24065 = 0;
    unsigned char *mem_23836 = NULL;
    struct memblock ext_mem_23223;
    
    ext_mem_23223.references = NULL;
    
    struct memblock ext_mem_23224;
    
    ext_mem_23224.references = NULL;
    
    struct memblock mem_23221;
    
    mem_23221.references = NULL;
    
    struct memblock ext_mem_23219;
    
    ext_mem_23219.references = NULL;
    
    struct memblock ext_mem_23220;
    
    ext_mem_23220.references = NULL;
    
    struct memblock mem_23217;
    
    mem_23217.references = NULL;
    
    bool prim_out_23856;
    
    // benchmark/ba/bench_jvp_ba_simple.fut:83:3-85:45
    if (mem_23183_cached_sizze_23977 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_23183, &mem_23183_cached_sizze_23977, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:83:3-85:45
    for (int64_t i_22859 = 0; i_22859 < (int64_t) 16; i_22859++) {
        // benchmark/ba/bench_jvp_ba_simple.fut:84:17-27
        
        int64_t i64_arg0_19896 = smod64(i_22859, (int64_t) 2);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:84:6-27
        
        int32_t i64_res_19897 = sext_i64_i32(i64_arg0_19896);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:85:20-22
        
        int64_t zp_lhs_19898 = mul64((int64_t) 7, i_22859);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:85:23-29
        
        int64_t zv_lhs_19899 = add64((int64_t) 3, zp_lhs_19898);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:85:31-43
        
        int64_t i64_arg0_19900 = smod64(zv_lhs_19899, (int64_t) 8);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:85:6-43
        
        int32_t i64_res_19901 = sext_i64_i32(i64_arg0_19900);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:83:3-85:45
        ((int32_t *) mem_23183)[i_22859 * (int64_t) 2] = i64_res_19897;
        ((int32_t *) mem_23183)[i_22859 * (int64_t) 2 + (int64_t) 1] = i64_res_19901;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:113:3-130:28
    if (mem_23201_cached_sizze_23980 < (int64_t) 2976) {
        err = lexical_realloc(ctx, &mem_23201, &mem_23201_cached_sizze_23980, (int64_t) 2976);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
    if (mem_23206_cached_sizze_23981 < (int64_t) 62) {
        err = lexical_realloc(ctx, &mem_23206, &mem_23206_cached_sizze_23981, (int64_t) 62);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:113:3-130:28
    for (int64_t i_22871 = 0; i_22871 < (int64_t) 48; i_22871++) {
        // benchmark/ba/bench_jvp_ba_simple.fut:115:7-130:27
        
        bool cond_19953 = slt64(i_22871, (int64_t) 32);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:116:20-26
        
        int64_t ob_19954 = sdiv64(i_22871, (int64_t) 2);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:117:30-40
        
        bool x_19956 = sle64((int64_t) 0, ob_19954);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:117:30-40
        
        bool y_19957 = slt64(ob_19954, (int64_t) 16);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:117:30-40
        
        bool bounds_check_19959 = x_19956 && y_19957;
        
        // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
        
        bool loop_not_taken_19961 = !cond_19953;
        
        // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
        
        bool protect_assert_disj_19962 = bounds_check_19959 || loop_not_taken_19961;
        
        // benchmark/ba/bench_jvp_ba_simple.fut:117:30-40
        
        bool index_certs_19963;
        
        if (!protect_assert_disj_19962) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) ob_19954, ", ", (long long) (int64_t) 0, "] out of bounds for array of shape [", (long long) (int64_t) 16, "][", (long long) (int64_t) 2, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:117:30-40\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:143:5-35\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:193:5-35\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // benchmark/ba/bench_jvp_ba_simple.fut:118:32-42
        
        bool index_certs_19964;
        
        if (!protect_assert_disj_19962) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) ob_19954, ", ", (long long) (int64_t) 1, "] out of bounds for array of shape [", (long long) (int64_t) 16, "][", (long long) (int64_t) 2, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:118:32-42\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:143:5-35\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:193:5-35\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // benchmark/ba/bench_jvp_ba_simple.fut:128:20-31
        
        int64_t ob_19955 = sub64(i_22871, (int64_t) 32);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:129:59-63
        
        int64_t weight_col_19958 = add64((int64_t) 46, ob_19955);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:122:59-63
        
        int64_t weight_col_19960 = add64((int64_t) 46, ob_19954);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:117:30-40
        
        int32_t i32_arg0_19965;
        
        if (cond_19953) {
            // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
            
            int32_t x_22457 = ((int32_t *) mem_23183)[ob_19954 * (int64_t) 2];
            
            i32_arg0_19965 = x_22457;
        } else {
            i32_arg0_19965 = 0;
        }
        // benchmark/ba/bench_jvp_ba_simple.fut:118:32-42
        
        int32_t i32_arg0_19967;
        
        if (cond_19953) {
            // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
            
            int32_t x_22458 = ((int32_t *) mem_23183)[ob_19954 * (int64_t) 2 + (int64_t) 1];
            
            i32_arg0_19967 = x_22458;
        } else {
            i32_arg0_19967 = 0;
        }
        // benchmark/ba/bench_jvp_ba_simple.fut:117:22-40
        
        int64_t i32_res_19969 = sext_i32_i64(i32_arg0_19965);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:118:24-42
        
        int64_t i32_res_19970 = sext_i32_i64(i32_arg0_19967);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:120:31-39
        
        int64_t cam_start_19971 = mul64((int64_t) 11, i32_res_19969);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:121:48-57
        
        int64_t zp_rhs_19972 = mul64((int64_t) 3, i32_res_19970);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:121:42-57
        
        int64_t point_start_19973 = add64((int64_t) 22, zp_rhs_19972);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:124:45-52
        
        int64_t zl_rhs_19974 = add64((int64_t) 11, cam_start_19971);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:125:49-55
        
        int64_t zl_rhs_19975 = add64((int64_t) 25, zp_rhs_19972);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
        for (int64_t i_22867 = 0; i_22867 < (int64_t) 62; i_22867++) {
            // benchmark/ba/bench_jvp_ba_simple.fut:124:12-126:27
            
            bool cond_19978 = sle64(cam_start_19971, i_22867);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:124:33-52
            
            bool cond_t_res_19979 = slt64(i_22867, zl_rhs_19974);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
            
            bool x_19980 = cond_19978 && cond_t_res_19979;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:125:13-55
            
            bool cond_19981 = sle64(point_start_19973, i_22867);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:125:35-55
            
            bool cond_f_res_t_res_19982 = slt64(i_22867, zl_rhs_19975);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
            
            bool x_19983 = cond_19981 && cond_f_res_t_res_19982;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
            
            bool x_19984 = !x_19980;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
            
            bool y_19985 = x_19983 && x_19984;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
            
            bool cond_19986 = x_19980 || y_19985;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:126:14-27
            
            bool lifted_lambda_res_t_res_f_res_19987 = i_22867 == weight_col_19960;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
            
            bool x_19988 = !cond_19986;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
            
            bool y_19989 = lifted_lambda_res_t_res_f_res_19987 && x_19988;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
            
            bool lifted_lambda_res_t_res_19990 = cond_19986 || y_19989;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:130:14-27
            
            bool lifted_lambda_res_f_res_19991 = i_22867 == weight_col_19958;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
            
            bool x_19992 = cond_19953 && lifted_lambda_res_t_res_19990;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
            
            bool y_19993 = loop_not_taken_19961 && lifted_lambda_res_f_res_19991;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:114:5-130:27
            
            bool lifted_lambda_res_19994 = x_19992 || y_19993;
            
            ((bool *) mem_23206)[i_22867] = lifted_lambda_res_19994;
        }
        lmad_copy_1b(ctx, 1, (uint8_t *) mem_23201, i_22871 * (int64_t) 62, (int64_t []) {(int64_t) 1}, (uint8_t *) mem_23206, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 62});
    }
    // src/pattern_csr.fut:29:14-39
    if (memblock_alloc(ctx, &mem_23217, (int64_t) 2976, "mem_23217")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:29:14-39
    // src/pattern_csr.fut:29:14-39
    lmad_copy_1b(ctx, 2, (uint8_t *) mem_23217.mem, (int64_t) 0, (int64_t []) {(int64_t) 62, (int64_t) 1}, (uint8_t *) mem_23201, (int64_t) 0, (int64_t []) {(int64_t) 62, (int64_t) 1}, (int64_t []) {(int64_t) 48, (int64_t) 62});
    // src/pattern_csr.fut:29:14-39
    
    int64_t csr_bipartite_from_pattern_res_19996;
    int64_t csr_bipartite_from_pattern_res_19997;
    
    if (futrts_csr_rows_from_pattern_9525(ctx, &ext_mem_23220, &ext_mem_23219, &csr_bipartite_from_pattern_res_19996, &csr_bipartite_from_pattern_res_19997, mem_23217, (int64_t) 48, (int64_t) 62) != 0) {
        err = 1;
        goto cleanup;
    }
    if (memblock_unref(ctx, &mem_23217, "mem_23217") != 0)
        return 1;
    // src/pattern_csr.fut:25:3-39
    if (memblock_alloc(ctx, &mem_23221, (int64_t) 2976, "mem_23221")) {
        err = 1;
        goto cleanup;
    }
    // src/pattern_csr.fut:25:3-39
    // src/pattern_csr.fut:25:3-39
    lmad_copy_1b(ctx, 2, (uint8_t *) mem_23221.mem, (int64_t) 0, (int64_t []) {(int64_t) 48, (int64_t) 1}, (uint8_t *) mem_23201, (int64_t) 0, (int64_t []) {(int64_t) 1, (int64_t) 62}, (int64_t []) {(int64_t) 62, (int64_t) 48});
    // src/pattern_csr.fut:25:3-39
    
    int64_t csr_cols_from_pattern_res_20001;
    int64_t csr_cols_from_pattern_res_20002;
    
    if (futrts_csr_rows_from_pattern_9525(ctx, &ext_mem_23224, &ext_mem_23223, &csr_cols_from_pattern_res_20001, &csr_cols_from_pattern_res_20002, mem_23221, (int64_t) 62, (int64_t) 48) != 0) {
        err = 1;
        goto cleanup;
    }
    if (memblock_unref(ctx, &mem_23221, "mem_23221") != 0)
        return 1;
    // benchmark/ba/bench_jvp_ba_simple.fut:16:15-35
    
    int64_t bytes_23505 = (int64_t) 8 * csr_bipartite_from_pattern_res_19997;
    
    // src/partial_d2_coloring.fut:50:26-44
    if (mem_23521_cached_sizze_24023 < (int64_t) 496) {
        err = lexical_realloc(ctx, &mem_23521, &mem_23521_cached_sizze_24023, (int64_t) 496);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/partial_d2_coloring.fut:50:26-44
    for (int64_t nest_i_23860 = 0; nest_i_23860 < (int64_t) 62; nest_i_23860++) {
        ((int64_t *) mem_23521)[nest_i_23860] = (int64_t) -1;
    }
    // src/partial_d2_coloring.fut:51:26-44
    if (mem_23522_cached_sizze_24024 < (int64_t) 496) {
        err = lexical_realloc(ctx, &mem_23522, &mem_23522_cached_sizze_24024, (int64_t) 496);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/partial_d2_coloring.fut:51:26-44
    for (int64_t nest_i_23861 = 0; nest_i_23861 < (int64_t) 62; nest_i_23861++) {
        ((int64_t *) mem_23522)[nest_i_23861] = (int64_t) -1;
    }
    // src/partial_d2_coloring.fut:56:5-65:38
    
    bool partial_d2_color_cols_order_res_20071;
    int64_t partial_d2_color_cols_order_res_20074;
    int64_t partial_d2_color_cols_order_res_20075;
    bool loop_while_20076;
    int64_t stamp0_20079;
    int64_t k0_20080;
    
    loop_while_20076 = 1;
    stamp0_20079 = (int64_t) 0;
    k0_20080 = (int64_t) 0;
    while (loop_while_20076) {
        // src/partial_d2_coloring.fut:58:15-24
        
        bool x_20081 = sle64((int64_t) 0, k0_20080);
        
        // src/partial_d2_coloring.fut:58:15-24
        
        bool y_20082 = slt64(k0_20080, (int64_t) 62);
        
        // src/partial_d2_coloring.fut:58:15-24
        
        bool bounds_check_20083 = x_20081 && y_20082;
        
        // src/partial_d2_coloring.fut:58:15-24
        
        bool index_certs_20084;
        
        if (!bounds_check_20083) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k0_20080, "] out of bounds for array of shape [", (long long) (int64_t) 62, "].", "-> #0  src/partial_d2_coloring.fut:58:15-24\n   #1  src/partial_d2_coloring.fut:70:14-73:72\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:166:14-175:63\n   #3  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/partial_d2_coloring.fut:15:11-22
        
        bool y_20085 = slt64(k0_20080, csr_cols_from_pattern_res_20001);
        
        // src/partial_d2_coloring.fut:15:11-22
        
        bool bounds_check_20086 = x_20081 && y_20085;
        
        // src/partial_d2_coloring.fut:15:11-22
        
        bool index_certs_20087;
        
        if (!bounds_check_20086) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k0_20080, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20001, "].", "-> #0  src/partial_d2_coloring.fut:15:11-22\n   #1  src/partial_d2_coloring.fut:46:14-60:97\n   #2  src/partial_d2_coloring.fut:70:14-73:72\n   #3  benchmark/ba/bench_jvp_ba_simple.fut:166:14-175:63\n   #4  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/partial_d2_coloring.fut:16:25-27
        
        int64_t k_end_20089 = add64((int64_t) 1, k0_20080);
        
        // src/partial_d2_coloring.fut:16:15-28
        
        bool x_20090 = sle64((int64_t) 0, k_end_20089);
        
        // src/partial_d2_coloring.fut:16:15-28
        
        bool y_20091 = slt64(k_end_20089, csr_cols_from_pattern_res_20001);
        
        // src/partial_d2_coloring.fut:16:15-28
        
        bool bounds_check_20092 = x_20090 && y_20091;
        
        // src/partial_d2_coloring.fut:16:15-28
        
        bool index_certs_20093;
        
        if (!bounds_check_20092) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_end_20089, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20001, "].", "-> #0  src/partial_d2_coloring.fut:16:15-28\n   #1  src/partial_d2_coloring.fut:46:14-60:97\n   #2  src/partial_d2_coloring.fut:70:14-73:72\n   #3  benchmark/ba/bench_jvp_ba_simple.fut:166:14-175:63\n   #4  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/partial_d2_coloring.fut:15:11-22
        
        int64_t k_20088 = ((int64_t *) ext_mem_23224.mem)[k0_20080];
        
        // src/partial_d2_coloring.fut:16:15-28
        
        int64_t k_end_20094 = ((int64_t *) ext_mem_23224.mem)[k_end_20089];
        
        // src/partial_d2_coloring.fut:18:5-33:33
        
        bool loop_cond_20095 = slt64(k_20088, k_end_20094);
        
        // src/partial_d2_coloring.fut:18:5-33:33
        
        bool mark_forbidden_colors_res_20096;
        int64_t mark_forbidden_colors_res_20098;
        bool loop_while_20099;
        int64_t k_20101;
        
        loop_while_20099 = loop_cond_20095;
        k_20101 = k_20088;
        while (loop_while_20099) {
            // src/partial_d2_coloring.fut:20:15-25
            
            bool x_20102 = sle64((int64_t) 0, k_20101);
            
            // src/partial_d2_coloring.fut:20:15-25
            
            bool y_20103 = slt64(k_20101, csr_cols_from_pattern_res_20002);
            
            // src/partial_d2_coloring.fut:20:15-25
            
            bool bounds_check_20104 = x_20102 && y_20103;
            
            // src/partial_d2_coloring.fut:20:15-25
            
            bool index_certs_20105;
            
            if (!bounds_check_20104) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) k_20101, "] out of bounds for array of shape [", (long long) csr_cols_from_pattern_res_20002, "].", "-> #0  src/partial_d2_coloring.fut:20:15-25\n   #1  src/partial_d2_coloring.fut:46:14-60:97\n   #2  src/partial_d2_coloring.fut:70:14-73:72\n   #3  benchmark/ba/bench_jvp_ba_simple.fut:166:14-175:63\n   #4  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/partial_d2_coloring.fut:20:15-25
            
            int64_t w_20106 = ((int64_t *) ext_mem_23223.mem)[k_20101];
            
            // src/partial_d2_coloring.fut:22:15-26
            
            bool x_20107 = sle64((int64_t) 0, w_20106);
            
            // src/partial_d2_coloring.fut:22:15-26
            
            bool y_20108 = slt64(w_20106, csr_bipartite_from_pattern_res_19996);
            
            // src/partial_d2_coloring.fut:22:15-26
            
            bool bounds_check_20109 = x_20107 && y_20108;
            
            // src/partial_d2_coloring.fut:22:15-26
            
            bool index_certs_20110;
            
            if (!bounds_check_20109) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) w_20106, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_19996, "].", "-> #0  src/partial_d2_coloring.fut:22:15-26\n   #1  src/partial_d2_coloring.fut:46:14-60:97\n   #2  src/partial_d2_coloring.fut:70:14-73:72\n   #3  benchmark/ba/bench_jvp_ba_simple.fut:166:14-175:63\n   #4  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/partial_d2_coloring.fut:23:29-31
            
            int64_t t_end_20112 = add64((int64_t) 1, w_20106);
            
            // src/partial_d2_coloring.fut:23:19-32
            
            bool x_20113 = sle64((int64_t) 0, t_end_20112);
            
            // src/partial_d2_coloring.fut:23:19-32
            
            bool y_20114 = slt64(t_end_20112, csr_bipartite_from_pattern_res_19996);
            
            // src/partial_d2_coloring.fut:23:19-32
            
            bool bounds_check_20115 = x_20113 && y_20114;
            
            // src/partial_d2_coloring.fut:23:19-32
            
            bool index_certs_20116;
            
            if (!bounds_check_20115) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_end_20112, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_19996, "].", "-> #0  src/partial_d2_coloring.fut:23:19-32\n   #1  src/partial_d2_coloring.fut:46:14-60:97\n   #2  src/partial_d2_coloring.fut:70:14-73:72\n   #3  benchmark/ba/bench_jvp_ba_simple.fut:166:14-175:63\n   #4  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/partial_d2_coloring.fut:22:15-26
            
            int64_t t_20111 = ((int64_t *) ext_mem_23220.mem)[w_20106];
            
            // src/partial_d2_coloring.fut:23:19-32
            
            int64_t t_end_20117 = ((int64_t *) ext_mem_23220.mem)[t_end_20112];
            
            // src/partial_d2_coloring.fut:25:9-31:31
            
            bool loop_cond_20118 = slt64(t_20111, t_end_20117);
            
            // src/partial_d2_coloring.fut:25:9-31:31
            
            bool loopres_20119;
            int64_t loopres_20121;
            bool loop_while_20122;
            int64_t t_20124;
            
            loop_while_20122 = loop_cond_20118;
            t_20124 = t_20111;
            while (loop_while_20122) {
                // src/partial_d2_coloring.fut:27:19-29
                
                bool x_20125 = sle64((int64_t) 0, t_20124);
                
                // src/partial_d2_coloring.fut:27:19-29
                
                bool y_20126 = slt64(t_20124, csr_bipartite_from_pattern_res_19997);
                
                // src/partial_d2_coloring.fut:27:19-29
                
                bool bounds_check_20127 = x_20125 && y_20126;
                
                // src/partial_d2_coloring.fut:27:19-29
                
                bool index_certs_20128;
                
                if (!bounds_check_20127) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) t_20124, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_19997, "].", "-> #0  src/partial_d2_coloring.fut:27:19-29\n   #1  src/partial_d2_coloring.fut:46:14-60:97\n   #2  src/partial_d2_coloring.fut:70:14-73:72\n   #3  benchmark/ba/bench_jvp_ba_simple.fut:166:14-175:63\n   #4  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/partial_d2_coloring.fut:27:19-29
                
                int64_t x_20129 = ((int64_t *) ext_mem_23219.mem)[t_20124];
                
                // src/partial_d2_coloring.fut:28:19-28
                
                bool x_20130 = sle64((int64_t) 0, x_20129);
                
                // src/partial_d2_coloring.fut:28:19-28
                
                bool y_20131 = slt64(x_20129, (int64_t) 62);
                
                // src/partial_d2_coloring.fut:28:19-28
                
                bool bounds_check_20132 = x_20130 && y_20131;
                
                // src/partial_d2_coloring.fut:28:19-28
                
                bool index_certs_20133;
                
                if (!bounds_check_20132) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) x_20129, "] out of bounds for array of shape [", (long long) (int64_t) 62, "].", "-> #0  src/partial_d2_coloring.fut:28:19-28\n   #1  src/partial_d2_coloring.fut:46:14-60:97\n   #2  src/partial_d2_coloring.fut:70:14-73:72\n   #3  benchmark/ba/bench_jvp_ba_simple.fut:166:14-175:63\n   #4  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/partial_d2_coloring.fut:28:19-28
                
                int64_t c_20134 = ((int64_t *) mem_23521)[x_20129];
                
                // src/partial_d2_coloring.fut:30:13-62
                
                bool cond_20135 = sle64((int64_t) 0, c_20134);
                
                // src/partial_d2_coloring.fut:30:13-62
                if (cond_20135) {
                    // src/partial_d2_coloring.fut:30:31-52
                    
                    bool y_22650 = slt64(c_20134, (int64_t) 62);
                    
                    // src/partial_d2_coloring.fut:30:31-52
                    
                    bool bounds_check_22651 = cond_20135 && y_22650;
                    
                    // src/partial_d2_coloring.fut:30:31-52
                    
                    bool index_certs_22652;
                    
                    if (!bounds_check_22651) {
                        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) c_20134, "] out of bounds for array of shape [", (long long) (int64_t) 62, "].", "-> #0  src/partial_d2_coloring.fut:30:31-52\n   #1  src/partial_d2_coloring.fut:46:14-60:97\n   #2  src/partial_d2_coloring.fut:70:14-73:72\n   #3  benchmark/ba/bench_jvp_ba_simple.fut:166:14-175:63\n   #4  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                        err = FUTHARK_PROGRAM_ERROR;
                        goto cleanup;
                    }
                    // src/partial_d2_coloring.fut:30:31-52
                    ((int64_t *) mem_23522)[c_20134] = stamp0_20079;
                }
                // src/partial_d2_coloring.fut:31:24-30
                
                int64_t tmp_20141 = add64((int64_t) 1, t_20124);
                
                // src/partial_d2_coloring.fut:26:17-24
                
                bool loop_cond_20142 = slt64(tmp_20141, t_end_20117);
                bool loop_while_tmp_23870 = loop_cond_20142;
                int64_t t_tmp_23872 = tmp_20141;
                
                loop_while_20122 = loop_while_tmp_23870;
                t_20124 = t_tmp_23872;
            }
            loopres_20119 = loop_while_20122;
            loopres_20121 = t_20124;
            // src/partial_d2_coloring.fut:33:26-32
            
            int64_t tmp_20143 = add64((int64_t) 1, k_20101);
            
            // src/partial_d2_coloring.fut:19:13-20
            
            bool loop_cond_20144 = slt64(tmp_20143, k_end_20094);
            bool loop_while_tmp_23867 = loop_cond_20144;
            int64_t k_tmp_23869 = tmp_20143;
            
            loop_while_20099 = loop_while_tmp_23867;
            k_20101 = k_tmp_23869;
        }
        mark_forbidden_colors_res_20096 = loop_while_20099;
        mark_forbidden_colors_res_20098 = k_20101;
        // src/partial_d2_coloring.fut:40:21-29
        
        int64_t zeze_lhs_20146 = ((int64_t *) mem_23522)[(int64_t) 0];
        
        // src/partial_d2_coloring.fut:40:30-38
        
        bool loop_cond_t_res_20147 = zeze_lhs_20146 == stamp0_20079;
        
        // src/partial_d2_coloring.fut:39:5-42:12
        
        bool c_final_20148;
        int64_t c_final_20149;
        bool loop_while_20150;
        int64_t c0_20151;
        
        loop_while_20150 = loop_cond_t_res_20147;
        c0_20151 = (int64_t) 0;
        while (loop_while_20150) {
            // src/partial_d2_coloring.fut:41:19-25
            
            int64_t c1_20152 = add64((int64_t) 1, c0_20151);
            
            // src/partial_d2_coloring.fut:40:11-38
            
            bool cond_20153 = slt64(c1_20152, (int64_t) 62);
            
            // src/partial_d2_coloring.fut:40:11-38
            
            bool loop_cond_20154;
            
            if (cond_20153) {
                // src/partial_d2_coloring.fut:40:21-29
                
                bool x_22654 = sle64((int64_t) 0, c1_20152);
                
                // src/partial_d2_coloring.fut:40:21-29
                
                bool bounds_check_22655 = cond_20153 && x_22654;
                
                // src/partial_d2_coloring.fut:40:21-29
                
                bool index_certs_22656;
                
                if (!bounds_check_22655) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) c1_20152, "] out of bounds for array of shape [", (long long) (int64_t) 62, "].", "-> #0  src/partial_d2_coloring.fut:40:21-29\n   #1  src/partial_d2_coloring.fut:61:15-44\n   #2  src/partial_d2_coloring.fut:70:14-73:72\n   #3  benchmark/ba/bench_jvp_ba_simple.fut:166:14-175:63\n   #4  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // src/partial_d2_coloring.fut:40:21-29
                
                int64_t zeze_lhs_22657 = ((int64_t *) mem_23522)[c1_20152];
                
                // src/partial_d2_coloring.fut:40:30-38
                
                bool loop_cond_t_res_22658 = zeze_lhs_22657 == stamp0_20079;
                
                loop_cond_20154 = loop_cond_t_res_22658;
            } else {
                loop_cond_20154 = 0;
            }
            
            bool loop_while_tmp_23873 = loop_cond_20154;
            int64_t c0_tmp_23874 = c1_20152;
            
            loop_while_20150 = loop_while_tmp_23873;
            c0_20151 = c0_tmp_23874;
        }
        c_final_20148 = loop_while_20150;
        c_final_20149 = c0_20151;
        // src/partial_d2_coloring.fut:62:21-41
        ((int64_t *) mem_23521)[k0_20080] = c_final_20149;
        // src/partial_d2_coloring.fut:63:27-33
        
        int64_t stamp1_20163 = add64((int64_t) 1, stamp0_20079);
        
        // src/partial_d2_coloring.fut:57:14-17
        
        bool loop_cond_20164 = slt64(k_end_20089, (int64_t) 62);
        bool loop_while_tmp_23862 = loop_cond_20164;
        int64_t stamp0_tmp_23865 = stamp1_20163;
        int64_t k0_tmp_23866 = k_end_20089;
        
        loop_while_20076 = loop_while_tmp_23862;
        stamp0_20079 = stamp0_tmp_23865;
        k0_20080 = k0_tmp_23866;
    }
    partial_d2_color_cols_order_res_20071 = loop_while_20076;
    partial_d2_color_cols_order_res_20074 = stamp0_20079;
    partial_d2_color_cols_order_res_20075 = k0_20080;
    if (memblock_unref(ctx, &ext_mem_23223, "ext_mem_23223") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_23224, "ext_mem_23224") != 0)
        return 1;
    // benchmark/ba/bench_jvp_ba_simple.fut:56:8-181:8
    
    int64_t x_22854;
    int64_t redout_23003 = (int64_t) 0;
    
    for (int64_t i_23004 = 0; i_23004 < (int64_t) 62; i_23004++) {
        int64_t x_20169 = ((int64_t *) mem_23521)[i_23004];
        
        // src/sparse_jacobian_jvp.fut:14:22-29
        
        int64_t max_res_20172 = smax64(x_20169, redout_23003);
        int64_t redout_tmp_23875 = max_res_20172;
        
        redout_23003 = redout_tmp_23875;
    }
    x_22854 = redout_23003;
    // src/sparse_jacobian_jvp.fut:14:13-41
    
    int64_t num_colors_of_res_f_res_20173 = add64((int64_t) 1, x_22854);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool bounds_invalid_upwards_20175 = slt64(num_colors_of_res_f_res_20173, (int64_t) 0);
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool valid_20176 = !bounds_invalid_upwards_20175;
    
    // src/sparse_jacobian_jvp.fut:29:11-18
    
    bool range_valid_c_20177;
    
    if (!valid_20176) {
        set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Range ", (long long) (int64_t) 0, "..", (long long) (int64_t) 1, "..<", (long long) num_colors_of_res_f_res_20173, " is invalid.", "-> #0  src/sparse_jacobian_jvp.fut:29:11-18\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:56:8-181:8\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
        err = FUTHARK_PROGRAM_ERROR;
        goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    
    int64_t bytes_23548 = (int64_t) 384 * num_colors_of_res_f_res_20173;
    
    // benchmark/ba/bench_jvp_ba_simple.fut:88:27-43
    if (mem_23193_cached_sizze_23978 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23193, &mem_23193_cached_sizze_23978, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:88:27-43
    
    struct memblock test_ba_d2_matches_dense_tinyzistatic_array_23876 = (struct memblock) {NULL, (unsigned char *) test_ba_d2_matches_dense_tinyzistatic_array_realtype_24066, 0, "test_ba_d2_matches_dense_tiny.static_array_23876"};
    
    // benchmark/ba/bench_jvp_ba_simple.fut:88:27-43
    lmad_copy_8b(ctx, 1, (uint64_t *) mem_23193, (int64_t) 0, (int64_t []) {(int64_t) 1}, (uint64_t *) test_ba_d2_matches_dense_tinyzistatic_array_23876.mem, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 2});
    // benchmark/ba/bench_jvp_ba_simple.fut:92:3-107:13
    if (mem_23194_cached_sizze_23979 < (int64_t) 496) {
        err = lexical_realloc(ctx, &mem_23194, &mem_23194_cached_sizze_23979, (int64_t) 496);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:92:3-107:13
    for (int64_t i_22863 = 0; i_22863 < (int64_t) 62; i_22863++) {
        // benchmark/ba/bench_jvp_ba_simple.fut:93:5-107:13
        
        bool cond_19915 = slt64(i_22863, (int64_t) 22);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:93:5-107:13
        
        double lifted_lambda_res_19916;
        
        if (cond_19915) {
            // benchmark/ba/bench_jvp_ba_simple.fut:94:19-26
            
            int64_t k_22429 = smod64(i_22863, (int64_t) 11);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:95:10-46
            
            bool cond_22430 = k_22429 == (int64_t) 6;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:95:10-46
            
            double lifted_lambda_res_t_res_22431;
            
            if (cond_22430) {
                lifted_lambda_res_t_res_22431 = 1.0;
            } else {
                lifted_lambda_res_t_res_22431 = 0.0;
            }
            lifted_lambda_res_19916 = lifted_lambda_res_t_res_22431;
        } else {
            // benchmark/ba/bench_jvp_ba_simple.fut:96:10-107:13
            
            bool cond_19923 = slt64(i_22863, (int64_t) 46);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:96:10-107:13
            
            double lifted_lambda_res_f_res_19924;
            
            if (cond_19923) {
                // benchmark/ba/bench_jvp_ba_simple.fut:97:21-34
                
                int64_t rel_22439 = sub64(i_22863, (int64_t) 22);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:98:26-32
                
                int64_t point_id_22440 = sdiv64(rel_22439, (int64_t) 3);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:99:19-25
                
                int64_t k_22441 = smod64(rel_22439, (int64_t) 3);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:100:10-105:57
                
                bool cond_22442 = k_22441 == (int64_t) 0;
                
                // benchmark/ba/bench_jvp_ba_simple.fut:100:10-105:57
                
                double lifted_lambda_res_f_res_t_res_22443;
                
                if (cond_22442) {
                    // benchmark/ba/bench_jvp_ba_simple.fut:101:40-47
                    
                    int64_t i64_arg0_22444 = smod64(point_id_22440, (int64_t) 17);
                    
                    // benchmark/ba/bench_jvp_ba_simple.fut:101:22-47
                    
                    double i64_res_22445 = sitofp_i64_f64(i64_arg0_22444);
                    
                    // benchmark/ba/bench_jvp_ba_simple.fut:101:20-48
                    
                    double lifted_lambda_res_f_res_t_res_t_res_22446 = 1.0e-2 * i64_res_22445;
                    
                    lifted_lambda_res_f_res_t_res_22443 = lifted_lambda_res_f_res_t_res_t_res_22446;
                } else {
                    // benchmark/ba/bench_jvp_ba_simple.fut:102:15-105:57
                    
                    bool cond_22447 = k_22441 == (int64_t) 1;
                    
                    // benchmark/ba/bench_jvp_ba_simple.fut:102:15-105:57
                    
                    double lifted_lambda_res_f_res_t_res_f_res_22448;
                    
                    if (cond_22447) {
                        // benchmark/ba/bench_jvp_ba_simple.fut:103:37-47
                        
                        int64_t zv_lhs_22449 = mul64((int64_t) 3, point_id_22440);
                        
                        // benchmark/ba/bench_jvp_ba_simple.fut:103:49-56
                        
                        int64_t i64_arg0_22450 = smod64(zv_lhs_22449, (int64_t) 17);
                        
                        // benchmark/ba/bench_jvp_ba_simple.fut:103:22-56
                        
                        double i64_res_22451 = sitofp_i64_f64(i64_arg0_22450);
                        
                        // benchmark/ba/bench_jvp_ba_simple.fut:103:20-57
                        
                        double lifted_lambda_res_f_res_t_res_f_res_t_res_22452 = 1.0e-2 * i64_res_22451;
                        
                        lifted_lambda_res_f_res_t_res_f_res_22448 = lifted_lambda_res_f_res_t_res_f_res_t_res_22452;
                    } else {
                        // benchmark/ba/bench_jvp_ba_simple.fut:105:49-56
                        
                        int64_t i64_arg0_22453 = smod64(point_id_22440, (int64_t) 13);
                        
                        // benchmark/ba/bench_jvp_ba_simple.fut:105:31-56
                        
                        double i64_res_22454 = sitofp_i64_f64(i64_arg0_22453);
                        
                        // benchmark/ba/bench_jvp_ba_simple.fut:105:29-57
                        
                        double zp_rhs_22455 = 1.0e-2 * i64_res_22454;
                        
                        // benchmark/ba/bench_jvp_ba_simple.fut:105:19-57
                        
                        double lifted_lambda_res_f_res_t_res_f_res_f_res_22456 = 4.0 + zp_rhs_22455;
                        
                        lifted_lambda_res_f_res_t_res_f_res_22448 = lifted_lambda_res_f_res_t_res_f_res_f_res_22456;
                    }
                    lifted_lambda_res_f_res_t_res_22443 = lifted_lambda_res_f_res_t_res_f_res_22448;
                }
                lifted_lambda_res_f_res_19924 = lifted_lambda_res_f_res_t_res_22443;
            } else {
                lifted_lambda_res_f_res_19924 = 1.0;
            }
            lifted_lambda_res_19916 = lifted_lambda_res_f_res_19924;
        }
        ((double *) mem_23194)[i_22863] = lifted_lambda_res_19916;
    }
    // benchmark/ba/ba_gradbench_original.fut:9:31-35
    
    double tmp_20443 = ((double *) mem_23193)[(int64_t) 0];
    
    // benchmark/ba/ba_gradbench_original.fut:9:39-43
    
    double tmp_20446 = ((double *) mem_23193)[(int64_t) 1];
    
    // src/dense_jacobian.fut:8:40-9:68
    if (mem_23225_cached_sizze_23982 < (int64_t) 23808) {
        err = lexical_realloc(ctx, &mem_23225, &mem_23225_cached_sizze_23982, (int64_t) 23808);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:5:3-21
    if (mem_23230_cached_sizze_23983 < (int64_t) 496) {
        err = lexical_realloc(ctx, &mem_23230, &mem_23230_cached_sizze_23983, (int64_t) 496);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:36:5-38:24
    if (mem_23231_cached_sizze_23984 < (int64_t) 176) {
        err = lexical_realloc(ctx, &mem_23231, &mem_23231_cached_sizze_23984, (int64_t) 176);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:36:5-38:24
    if (mem_23232_cached_sizze_23985 < (int64_t) 176) {
        err = lexical_realloc(ctx, &mem_23232, &mem_23232_cached_sizze_23985, (int64_t) 176);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:37:7-38:23
    if (mem_23241_cached_sizze_23986 < (int64_t) 88) {
        err = lexical_realloc(ctx, &mem_23241, &mem_23241_cached_sizze_23986, (int64_t) 88);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:37:7-38:23
    if (mem_23242_cached_sizze_23987 < (int64_t) 88) {
        err = lexical_realloc(ctx, &mem_23242, &mem_23242_cached_sizze_23987, (int64_t) 88);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:34:38-65:31
    if (mem_23263_cached_sizze_23988 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_23263, &mem_23263_cached_sizze_23988, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:34:38-65:31
    if (mem_23264_cached_sizze_23989 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_23264, &mem_23264_cached_sizze_23989, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:34:38-65:31
    if (mem_23265_cached_sizze_23990 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_23265, &mem_23265_cached_sizze_23990, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:34:38-65:31
    if (mem_23266_cached_sizze_23991 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_23266, &mem_23266_cached_sizze_23991, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:34:38-65:31
    if (mem_23267_cached_sizze_23992 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_23267, &mem_23267_cached_sizze_23992, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:34:38-65:31
    if (mem_23268_cached_sizze_23993 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_23268, &mem_23268_cached_sizze_23993, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:42:7-43:39
    if (mem_23287_cached_sizze_23994 < (int64_t) 24) {
        err = lexical_realloc(ctx, &mem_23287, &mem_23287_cached_sizze_23994, (int64_t) 24);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:42:7-43:39
    if (mem_23288_cached_sizze_23995 < (int64_t) 24) {
        err = lexical_realloc(ctx, &mem_23288, &mem_23288_cached_sizze_23995, (int64_t) 24);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:52:15-46
    if (mem_23319_cached_sizze_23996 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_23319, &mem_23319_cached_sizze_23996, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:52:15-46
    if (mem_23320_cached_sizze_23997 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_23320, &mem_23320_cached_sizze_23997, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:52:15-46
    if (mem_23321_cached_sizze_23998 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_23321, &mem_23321_cached_sizze_23998, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23340_cached_sizze_23999 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23340, &mem_23340_cached_sizze_23999, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23341_cached_sizze_24000 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23341, &mem_23341_cached_sizze_24000, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23342_cached_sizze_24001 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23342, &mem_23342_cached_sizze_24001, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23343_cached_sizze_24002 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23343, &mem_23343_cached_sizze_24002, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23344_cached_sizze_24003 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23344, &mem_23344_cached_sizze_24003, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23345_cached_sizze_24004 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23345, &mem_23345_cached_sizze_24004, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23346_cached_sizze_24005 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23346, &mem_23346_cached_sizze_24005, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23347_cached_sizze_24006 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23347, &mem_23347_cached_sizze_24006, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23348_cached_sizze_24007 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23348, &mem_23348_cached_sizze_24007, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23349_cached_sizze_24008 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23349, &mem_23349_cached_sizze_24008, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23350_cached_sizze_24009 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23350, &mem_23350_cached_sizze_24009, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23351_cached_sizze_24010 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23351, &mem_23351_cached_sizze_24010, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23352_cached_sizze_24011 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23352, &mem_23352_cached_sizze_24011, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23353_cached_sizze_24012 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23353, &mem_23353_cached_sizze_24012, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23354_cached_sizze_24013 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23354, &mem_23354_cached_sizze_24013, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23355_cached_sizze_24014 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23355, &mem_23355_cached_sizze_24014, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23356_cached_sizze_24015 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23356, &mem_23356_cached_sizze_24015, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23357_cached_sizze_24016 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23357, &mem_23357_cached_sizze_24016, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23358_cached_sizze_24017 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23358, &mem_23358_cached_sizze_24017, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23359_cached_sizze_24018 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23359, &mem_23359_cached_sizze_24018, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:48:5-51:49
    if (mem_23480_cached_sizze_24019 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_23480, &mem_23480_cached_sizze_24019, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:48:5-51:49
    if (mem_23481_cached_sizze_24020 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_23481, &mem_23481_cached_sizze_24020, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:73:6-79:35
    if (mem_23494_cached_sizze_24021 < (int64_t) 384) {
        err = lexical_realloc(ctx, &mem_23494, &mem_23494_cached_sizze_24021, (int64_t) 384);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/dense_jacobian.fut:8:40-9:68
    for (int64_t i_22997 = 0; i_22997 < (int64_t) 62; i_22997++) {
        // src/dense_jacobian.fut:5:3-21
        for (int64_t nest_i_23879 = 0; nest_i_23879 < (int64_t) 62; nest_i_23879++) {
            ((double *) mem_23230)[nest_i_23879] = 0.0;
        }
        // src/dense_jacobian.fut:5:3-39
        ((double *) mem_23230)[i_22997] = 1.0;
        // benchmark/ba/bench_jvp_ba_simple.fut:36:5-38:24
        for (int64_t i_22884 = 0; i_22884 < (int64_t) 2; i_22884++) {
            // benchmark/ba/bench_jvp_ba_simple.fut:38:16-18
            
            int64_t zp_lhs_20338 = mul64((int64_t) 11, i_22884);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:37:7-38:23
            for (int64_t i_22877 = 0; i_22877 < (int64_t) 11; i_22877++) {
                // benchmark/ba/bench_jvp_ba_simple.fut:38:19-22
                
                int64_t tmp_20341 = add64(zp_lhs_20338, i_22877);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:38:9-23
                
                bool x_20342 = sle64((int64_t) 0, tmp_20341);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:38:9-23
                
                bool y_20343 = slt64(tmp_20341, (int64_t) 62);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:38:9-23
                
                bool bounds_check_20344 = x_20342 && y_20343;
                
                // benchmark/ba/bench_jvp_ba_simple.fut:38:9-23
                
                bool index_certs_20345;
                
                if (!bounds_check_20344) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_20341, "] out of bounds for array of shape [", (long long) (int64_t) 62, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:38:9-23\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:59:5-46\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:161:33-81\n   #3  src/dense_jacobian.fut:8:30-9:58\n   #4  src/dense_jacobian.fut:8:40-9:68\n   #5  benchmark/ba/bench_jvp_ba_simple.fut:56:8-161:84\n   #6  benchmark/ba/bench_jvp_ba_simple.fut:193:30-196:60\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                
                double lifted_lambda_res_20346 = ((double *) mem_23194)[tmp_20341];
                double lifted_lambda_res_tan_21598 = ((double *) mem_23230)[tmp_20341];
                
                ((double *) mem_23241)[i_22877] = lifted_lambda_res_20346;
                ((double *) mem_23242)[i_22877] = lifted_lambda_res_tan_21598;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_23231, i_22884 * (int64_t) 11, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_23241, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 11});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_23232, i_22884 * (int64_t) 11, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_23242, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 11});
        }
        // benchmark/ba/bench_jvp_ba_simple.fut:34:38-65:31
        for (int64_t i_22906 = 0; i_22906 < (int64_t) 8; i_22906++) {
            // benchmark/ba/bench_jvp_ba_simple.fut:43:32-34
            
            int64_t zp_rhs_20934 = mul64((int64_t) 3, i_22906);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:43:26-34
            
            int64_t zp_lhs_20935 = add64((int64_t) 22, zp_rhs_20934);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:42:7-43:39
            for (int64_t i_22891 = 0; i_22891 < (int64_t) 3; i_22891++) {
                // benchmark/ba/bench_jvp_ba_simple.fut:43:35-38
                
                int64_t tmp_20938 = add64(zp_lhs_20935, i_22891);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:43:9-39
                
                bool x_20939 = sle64((int64_t) 0, tmp_20938);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:43:9-39
                
                bool y_20940 = slt64(tmp_20938, (int64_t) 62);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:43:9-39
                
                bool bounds_check_20941 = x_20939 && y_20940;
                
                // benchmark/ba/bench_jvp_ba_simple.fut:43:9-39
                
                bool index_certs_20942;
                
                if (!bounds_check_20941) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_20938, "] out of bounds for array of shape [", (long long) (int64_t) 62, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:43:9-39\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:59:5-46\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:161:33-81\n   #3  src/dense_jacobian.fut:8:30-9:58\n   #4  src/dense_jacobian.fut:8:40-9:68\n   #5  benchmark/ba/bench_jvp_ba_simple.fut:56:8-161:84\n   #6  benchmark/ba/bench_jvp_ba_simple.fut:193:30-196:60\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                
                double lifted_lambda_res_20943 = ((double *) mem_23194)[tmp_20938];
                double lifted_lambda_res_tan_21622 = ((double *) mem_23230)[tmp_20938];
                
                ((double *) mem_23287)[i_22891] = lifted_lambda_res_20943;
                ((double *) mem_23288)[i_22891] = lifted_lambda_res_tan_21622;
            }
            // benchmark/ba/ba_gradbench_original.fut:5:31-35
            
            double tmp_20945 = ((double *) mem_23287)[(int64_t) 0];
            
            // benchmark/ba/ba_gradbench_original.fut:5:31-35
            
            double tmp_tan_21623 = ((double *) mem_23288)[(int64_t) 0];
            
            // benchmark/ba/ba_gradbench_original.fut:5:39-43
            
            double tmp_20946 = ((double *) mem_23287)[(int64_t) 1];
            
            // benchmark/ba/ba_gradbench_original.fut:5:39-43
            
            double tmp_tan_21624 = ((double *) mem_23288)[(int64_t) 1];
            
            // benchmark/ba/ba_gradbench_original.fut:5:47-51
            
            double tmp_20947 = ((double *) mem_23287)[(int64_t) 2];
            
            // benchmark/ba/ba_gradbench_original.fut:5:47-51
            
            double tmp_tan_21625 = ((double *) mem_23288)[(int64_t) 2];
            
            ((double *) mem_23263)[i_22906] = tmp_20945;
            ((double *) mem_23264)[i_22906] = tmp_tan_21623;
            ((double *) mem_23265)[i_22906] = tmp_20946;
            ((double *) mem_23266)[i_22906] = tmp_tan_21624;
            ((double *) mem_23267)[i_22906] = tmp_20947;
            ((double *) mem_23268)[i_22906] = tmp_tan_21625;
        }
        // benchmark/ba/ba_gradbench_original.fut:52:15-46
        for (int64_t i_22919 = 0; i_22919 < (int64_t) 16; i_22919++) {
            // benchmark/ba/bench_jvp_ba_simple.fut:47:42-45
            
            int64_t tmp_20923 = add64((int64_t) 46, i_22919);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:47:7-46
            
            bool x_20924 = sle64((int64_t) 0, tmp_20923);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:47:7-46
            
            bool y_20925 = slt64(tmp_20923, (int64_t) 62);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:47:7-46
            
            bool bounds_check_20926 = x_20924 && y_20925;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:47:7-46
            
            bool index_certs_20927;
            
            if (!bounds_check_20926) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_20923, "] out of bounds for array of shape [", (long long) (int64_t) 62, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:47:7-46\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:59:5-46\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:161:33-81\n   #3  src/dense_jacobian.fut:8:30-9:58\n   #4  src/dense_jacobian.fut:8:40-9:68\n   #5  benchmark/ba/bench_jvp_ba_simple.fut:56:8-161:84\n   #6  benchmark/ba/bench_jvp_ba_simple.fut:193:30-196:60\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            double lifted_lambda_res_20928 = ((double *) mem_23194)[tmp_20923];
            double lifted_lambda_res_tan_21639 = ((double *) mem_23230)[tmp_20923];
            
            // benchmark/ba/ba_gradbench_original.fut:44:8-10
            
            double binop_x_21641 = lifted_lambda_res_20928 * lifted_lambda_res_tan_21639;
            
            // benchmark/ba/ba_gradbench_original.fut:44:8-10
            
            double zm_rhs_tan_21640 = binop_x_21641 + binop_x_21641;
            
            // benchmark/ba/ba_gradbench_original.fut:44:5-10
            
            double binop_y_21645 = -1.0 * zm_rhs_tan_21640;
            
            ((double *) mem_23319)[i_22919] = binop_y_21645;
            ((double *) mem_23320)[i_22919] = lifted_lambda_res_20928;
            ((double *) mem_23321)[i_22919] = lifted_lambda_res_tan_21639;
        }
        // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
        for (int64_t i_22963 = 0; i_22963 < (int64_t) 2; i_22963++) {
            double tmp_20397 = ((double *) mem_23231)[i_22963 * (int64_t) 11];
            double tmp_tan_21664 = ((double *) mem_23232)[i_22963 * (int64_t) 11];
            double tmp_20400 = ((double *) mem_23231)[i_22963 * (int64_t) 11 + (int64_t) 1];
            double tmp_tan_21665 = ((double *) mem_23232)[i_22963 * (int64_t) 11 + (int64_t) 1];
            double tmp_20403 = ((double *) mem_23231)[i_22963 * (int64_t) 11 + (int64_t) 2];
            double tmp_tan_21666 = ((double *) mem_23232)[i_22963 * (int64_t) 11 + (int64_t) 2];
            double tmp_20406 = ((double *) mem_23231)[i_22963 * (int64_t) 11 + (int64_t) 3];
            double tmp_tan_21667 = ((double *) mem_23232)[i_22963 * (int64_t) 11 + (int64_t) 3];
            double tmp_20409 = ((double *) mem_23231)[i_22963 * (int64_t) 11 + (int64_t) 4];
            double tmp_tan_21668 = ((double *) mem_23232)[i_22963 * (int64_t) 11 + (int64_t) 4];
            double tmp_20412 = ((double *) mem_23231)[i_22963 * (int64_t) 11 + (int64_t) 5];
            double tmp_tan_21669 = ((double *) mem_23232)[i_22963 * (int64_t) 11 + (int64_t) 5];
            double tmp_20415 = ((double *) mem_23231)[i_22963 * (int64_t) 11 + (int64_t) 7];
            double tmp_tan_21670 = ((double *) mem_23232)[i_22963 * (int64_t) 11 + (int64_t) 7];
            double tmp_20418 = ((double *) mem_23231)[i_22963 * (int64_t) 11 + (int64_t) 8];
            double tmp_tan_21671 = ((double *) mem_23232)[i_22963 * (int64_t) 11 + (int64_t) 8];
            double tmp_20421 = ((double *) mem_23231)[i_22963 * (int64_t) 11 + (int64_t) 9];
            double tmp_tan_21672 = ((double *) mem_23232)[i_22963 * (int64_t) 11 + (int64_t) 9];
            double tmp_20424 = ((double *) mem_23231)[i_22963 * (int64_t) 11 + (int64_t) 10];
            double tmp_tan_21673 = ((double *) mem_23232)[i_22963 * (int64_t) 11 + (int64_t) 10];
            
            ((double *) mem_23340)[i_22963] = tmp_20406;
            ((double *) mem_23341)[i_22963] = tmp_tan_21667;
            ((double *) mem_23342)[i_22963] = tmp_20409;
            ((double *) mem_23343)[i_22963] = tmp_tan_21668;
            ((double *) mem_23344)[i_22963] = tmp_20412;
            ((double *) mem_23345)[i_22963] = tmp_tan_21669;
            ((double *) mem_23346)[i_22963] = tmp_20421;
            ((double *) mem_23347)[i_22963] = tmp_tan_21672;
            ((double *) mem_23348)[i_22963] = tmp_20424;
            ((double *) mem_23349)[i_22963] = tmp_tan_21673;
            ((double *) mem_23350)[i_22963] = tmp_20397;
            ((double *) mem_23351)[i_22963] = tmp_tan_21664;
            ((double *) mem_23352)[i_22963] = tmp_20400;
            ((double *) mem_23353)[i_22963] = tmp_tan_21665;
            ((double *) mem_23354)[i_22963] = tmp_20403;
            ((double *) mem_23355)[i_22963] = tmp_tan_21666;
            ((double *) mem_23356)[i_22963] = tmp_20415;
            ((double *) mem_23357)[i_22963] = tmp_tan_21670;
            ((double *) mem_23358)[i_22963] = tmp_20418;
            ((double *) mem_23359)[i_22963] = tmp_tan_21671;
        }
        // benchmark/ba/ba_gradbench_original.fut:48:5-51:49
        for (int64_t i_22988 = 0; i_22988 < (int64_t) 16; i_22988++) {
            // benchmark/ba/ba_gradbench_original.fut:49:44-52
            
            int32_t compute_reproj_err_arg1_20462 = ((int32_t *) mem_23183)[i_22988 * (int64_t) 2 + (int64_t) 1];
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            int64_t compute_reproj_err_arg1_20463 = sext_i32_i64(compute_reproj_err_arg1_20462);
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            bool x_20464 = sle64((int64_t) 0, compute_reproj_err_arg1_20463);
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            bool y_20465 = slt64(compute_reproj_err_arg1_20463, (int64_t) 8);
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            bool bounds_check_20466 = x_20464 && y_20465;
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            bool index_certs_20467;
            
            if (!bounds_check_20466) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) compute_reproj_err_arg1_20463, "] out of bounds for array of shape [", (long long) (int64_t) 8, "].", "-> #0  benchmark/ba/ba_gradbench_original.fut:49:42-53\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:71:5-50\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:161:33-81\n   #3  src/dense_jacobian.fut:8:30-9:58\n   #4  src/dense_jacobian.fut:8:40-9:68\n   #5  benchmark/ba/bench_jvp_ba_simple.fut:56:8-161:84\n   #6  benchmark/ba/bench_jvp_ba_simple.fut:193:30-196:60\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // benchmark/ba/ba_gradbench_original.fut:48:47-55
            
            int32_t compute_reproj_err_arg0_20472 = ((int32_t *) mem_23183)[i_22988 * (int64_t) 2];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            int64_t compute_reproj_err_arg0_20473 = sext_i32_i64(compute_reproj_err_arg0_20472);
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            bool x_20474 = sle64((int64_t) 0, compute_reproj_err_arg0_20473);
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            bool y_20475 = slt64(compute_reproj_err_arg0_20473, (int64_t) 2);
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            bool bounds_check_20476 = x_20474 && y_20475;
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            bool index_certs_20477;
            
            if (!bounds_check_20476) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) compute_reproj_err_arg0_20473, "] out of bounds for array of shape [", (long long) (int64_t) 2, "].", "-> #0  benchmark/ba/ba_gradbench_original.fut:48:42-56\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:71:5-50\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:161:33-81\n   #3  src/dense_jacobian.fut:8:30-9:58\n   #4  src/dense_jacobian.fut:8:40-9:68\n   #5  benchmark/ba/bench_jvp_ba_simple.fut:56:8-161:84\n   #6  benchmark/ba/bench_jvp_ba_simple.fut:193:30-196:60\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // benchmark/ba/ba_gradbench_original.fut:50:42-46
            
            double compute_reproj_err_arg2_20460 = ((double *) mem_23320)[i_22988];
            
            // benchmark/ba/ba_gradbench_original.fut:50:42-46
            
            double compute_reproj_err_arg2_tan_21684 = ((double *) mem_23321)[i_22988];
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            double lifted_lambda_res_20468 = ((double *) mem_23263)[compute_reproj_err_arg1_20463];
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            double lifted_lambda_res_tan_21695 = ((double *) mem_23264)[compute_reproj_err_arg1_20463];
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            double lifted_lambda_res_20469 = ((double *) mem_23265)[compute_reproj_err_arg1_20463];
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            double lifted_lambda_res_tan_21696 = ((double *) mem_23266)[compute_reproj_err_arg1_20463];
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            double lifted_lambda_res_20470 = ((double *) mem_23267)[compute_reproj_err_arg1_20463];
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            double lifted_lambda_res_tan_21697 = ((double *) mem_23268)[compute_reproj_err_arg1_20463];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20478 = ((double *) mem_23340)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_21708 = ((double *) mem_23341)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20479 = ((double *) mem_23342)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_21709 = ((double *) mem_23343)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20480 = ((double *) mem_23344)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_21710 = ((double *) mem_23345)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/bench_jvp_ba_simple.fut:161:33-81
            
            double lifted_lambda_res_20481 = ((double *) mem_23231)[compute_reproj_err_arg0_20473 * (int64_t) 11 + (int64_t) 6];
            
            // benchmark/ba/bench_jvp_ba_simple.fut:161:33-81
            
            double lifted_lambda_res_tan_21711 = ((double *) mem_23232)[compute_reproj_err_arg0_20473 * (int64_t) 11 + (int64_t) 6];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20482 = ((double *) mem_23346)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_21712 = ((double *) mem_23347)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20483 = ((double *) mem_23348)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_21713 = ((double *) mem_23349)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20484 = ((double *) mem_23350)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_21714 = ((double *) mem_23351)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20485 = ((double *) mem_23352)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_21715 = ((double *) mem_23353)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20486 = ((double *) mem_23354)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_21716 = ((double *) mem_23355)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20487 = ((double *) mem_23356)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_21717 = ((double *) mem_23357)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20488 = ((double *) mem_23358)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_21718 = ((double *) mem_23359)[compute_reproj_err_arg0_20473];
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double zm_res_20489 = lifted_lambda_res_20468 - lifted_lambda_res_20478;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double binop_y_21721 = -1.0 * lifted_lambda_res_tan_21708;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double zm_res_tan_21719 = lifted_lambda_res_tan_21695 + binop_y_21721;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double zm_res_20490 = lifted_lambda_res_20469 - lifted_lambda_res_20479;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double binop_y_21724 = -1.0 * lifted_lambda_res_tan_21709;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double zm_res_tan_21722 = lifted_lambda_res_tan_21696 + binop_y_21724;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double zm_res_20491 = lifted_lambda_res_20470 - lifted_lambda_res_20480;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double binop_y_21727 = -1.0 * lifted_lambda_res_tan_21710;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double zm_res_tan_21725 = lifted_lambda_res_tan_21697 + binop_y_21727;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:11-18
            
            double zt_res_20492 = lifted_lambda_res_20484 * lifted_lambda_res_20484;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:11-18
            
            double binop_x_21729 = lifted_lambda_res_20484 * lifted_lambda_res_tan_21714;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:11-18
            
            double zt_res_tan_21728 = binop_x_21729 + binop_x_21729;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:21-28
            
            double zt_res_20493 = lifted_lambda_res_20485 * lifted_lambda_res_20485;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:21-28
            
            double binop_x_21732 = lifted_lambda_res_20485 * lifted_lambda_res_tan_21715;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:21-28
            
            double zt_res_tan_21731 = binop_x_21732 + binop_x_21732;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:14-28
            
            double zp_res_20494 = zt_res_20492 + zt_res_20493;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:14-28
            
            double zp_res_tan_21734 = zt_res_tan_21728 + zt_res_tan_21731;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:31-38
            
            double zt_res_20495 = lifted_lambda_res_20486 * lifted_lambda_res_20486;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:31-38
            
            double binop_x_21738 = lifted_lambda_res_20486 * lifted_lambda_res_tan_21716;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:31-38
            
            double zt_res_tan_21737 = binop_x_21738 + binop_x_21738;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:19-38
            
            double zp_res_20496 = zp_res_20494 + zt_res_20495;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:19-38
            
            double zp_res_tan_21740 = zp_res_tan_21734 + zt_res_tan_21737;
            
            // benchmark/ba/ba_gradbench_original.fut:13:5-26:31
            
            bool cond_20497 = zp_res_20496 == 0.0;
            
            // benchmark/ba/ba_gradbench_original.fut:13:5-26:31
            
            bool cond_20498 = !cond_20497;
            
            // benchmark/ba/ba_gradbench_original.fut:13:5-26:31
            
            double rodrigues_rotate_point_res_20499;
            double rodrigues_rotate_point_res_tan_21899;
            double rodrigues_rotate_point_res_20500;
            double rodrigues_rotate_point_res_tan_21900;
            double rodrigues_rotate_point_res_20501;
            double rodrigues_rotate_point_res_tan_21901;
            
            if (cond_20498) {
                // benchmark/ba/ba_gradbench_original.fut:14:21-37
                
                double sqrt_res_22464 = futrts_sqrt64(zp_res_20496);
                double binop_y_22465 = fpow64(zp_res_20496, 0.5);
                double binop_y_22466 = 2.0 * binop_y_22465;
                double binop_y_22467 = 1.0 / binop_y_22466;
                double sqrt_res_tan_22468 = zp_res_tan_21740 * binop_y_22467;
                
                // benchmark/ba/ba_gradbench_original.fut:15:24-37
                
                double cos_res_22469 = futrts_cos64(sqrt_res_22464);
                double binop_y_22470 = futrts_sin64(sqrt_res_22464);
                double binop_y_22471 = 0.0 - binop_y_22470;
                double cos_res_tan_22472 = sqrt_res_tan_22468 * binop_y_22471;
                double sin_res_tan_22473 = sqrt_res_tan_22468 * cos_res_22469;
                
                // benchmark/ba/ba_gradbench_original.fut:17:27-34
                
                double theta_inv_22474 = 1.0 / sqrt_res_22464;
                
                // benchmark/ba/ba_gradbench_original.fut:17:27-34
                
                double binop_x_22475 = 0.0 * theta_inv_22474;
                
                // benchmark/ba/ba_gradbench_original.fut:17:27-34
                
                double binop_y_22476 = sqrt_res_22464 * sqrt_res_22464;
                
                // benchmark/ba/ba_gradbench_original.fut:17:27-34
                
                double binop_y_22477 = 1.0 / binop_y_22476;
                
                // benchmark/ba/ba_gradbench_original.fut:17:27-34
                
                double binop_y_22478 = 0.0 - binop_y_22477;
                
                // benchmark/ba/ba_gradbench_original.fut:17:27-34
                
                double binop_y_22479 = sqrt_res_tan_22468 * binop_y_22478;
                
                // benchmark/ba/ba_gradbench_original.fut:17:27-34
                
                double theta_inv_tan_22480 = binop_x_22475 + binop_y_22479;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22481 = lifted_lambda_res_20484 * theta_inv_22474;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22482 = lifted_lambda_res_tan_21714 * theta_inv_22474;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22483 = lifted_lambda_res_20484 * theta_inv_tan_22480;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22484 = binop_x_22482 + binop_y_22483;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22485 = lifted_lambda_res_20485 * theta_inv_22474;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22486 = lifted_lambda_res_tan_21715 * theta_inv_22474;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22487 = lifted_lambda_res_20485 * theta_inv_tan_22480;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22488 = binop_x_22486 + binop_y_22487;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22489 = lifted_lambda_res_20486 * theta_inv_22474;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22490 = lifted_lambda_res_tan_21716 * theta_inv_22474;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22491 = lifted_lambda_res_20486 * theta_inv_tan_22480;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22492 = binop_x_22490 + binop_y_22491;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double zt_res_22493 = zm_res_20491 * zt_res_22485;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double binop_x_22494 = zm_res_tan_21725 * zt_res_22485;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double binop_y_22495 = zm_res_20491 * zt_res_tan_22488;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double zt_res_tan_22496 = binop_x_22494 + binop_y_22495;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double zt_res_22497 = zm_res_20490 * zt_res_22489;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double binop_x_22498 = zm_res_tan_21722 * zt_res_22489;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double binop_y_22499 = zm_res_20490 * zt_res_tan_22492;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double zt_res_tan_22500 = binop_x_22498 + binop_y_22499;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:16-25
                
                double zm_res_22501 = zt_res_22493 - zt_res_22497;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:16-25
                
                double binop_y_22502 = -1.0 * zt_res_tan_22500;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:16-25
                
                double zm_res_tan_22503 = zt_res_tan_22496 + binop_y_22502;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double zt_res_22504 = zm_res_20489 * zt_res_22489;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double binop_x_22505 = zm_res_tan_21719 * zt_res_22489;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double binop_y_22506 = zm_res_20489 * zt_res_tan_22492;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double zt_res_tan_22507 = binop_x_22505 + binop_y_22506;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double zt_res_22508 = zm_res_20491 * zt_res_22481;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double binop_x_22509 = zm_res_tan_21725 * zt_res_22481;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double binop_y_22510 = zm_res_20491 * zt_res_tan_22484;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double zt_res_tan_22511 = binop_x_22509 + binop_y_22510;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:31-40
                
                double zm_res_22512 = zt_res_22504 - zt_res_22508;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:31-40
                
                double binop_y_22513 = -1.0 * zt_res_tan_22511;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:31-40
                
                double zm_res_tan_22514 = zt_res_tan_22507 + binop_y_22513;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double zt_res_22515 = zm_res_20490 * zt_res_22481;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double binop_x_22516 = zm_res_tan_21722 * zt_res_22481;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double binop_y_22517 = zm_res_20490 * zt_res_tan_22484;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double zt_res_tan_22518 = binop_x_22516 + binop_y_22517;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double zt_res_22519 = zm_res_20489 * zt_res_22485;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double binop_x_22520 = zm_res_tan_21719 * zt_res_22485;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double binop_y_22521 = zm_res_20489 * zt_res_tan_22488;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double zt_res_tan_22522 = binop_x_22520 + binop_y_22521;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:46-55
                
                double zm_res_22523 = zt_res_22515 - zt_res_22519;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:46-55
                
                double binop_y_22524 = -1.0 * zt_res_tan_22522;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:46-55
                
                double zm_res_tan_22525 = zt_res_tan_22518 + binop_y_22524;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:11-18
                
                double zt_res_22526 = zm_res_20489 * zt_res_22481;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:11-18
                
                double binop_x_22527 = zm_res_tan_21719 * zt_res_22481;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:11-18
                
                double binop_y_22528 = zm_res_20489 * zt_res_tan_22484;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:11-18
                
                double zt_res_tan_22529 = binop_x_22527 + binop_y_22528;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:21-28
                
                double zt_res_22530 = zm_res_20490 * zt_res_22485;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:21-28
                
                double binop_x_22531 = zm_res_tan_21722 * zt_res_22485;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:21-28
                
                double binop_y_22532 = zm_res_20490 * zt_res_tan_22488;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:21-28
                
                double zt_res_tan_22533 = binop_x_22531 + binop_y_22532;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:14-28
                
                double zp_res_22534 = zt_res_22526 + zt_res_22530;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:14-28
                
                double zp_res_tan_22535 = zt_res_tan_22529 + zt_res_tan_22533;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:31-38
                
                double zt_res_22536 = zm_res_20491 * zt_res_22489;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:31-38
                
                double binop_x_22537 = zm_res_tan_21725 * zt_res_22489;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:31-38
                
                double binop_y_22538 = zm_res_20491 * zt_res_tan_22492;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:31-38
                
                double zt_res_tan_22539 = binop_x_22537 + binop_y_22538;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:19-38
                
                double zp_res_22540 = zp_res_22534 + zt_res_22536;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:19-38
                
                double zp_res_tan_22541 = zp_res_tan_22535 + zt_res_tan_22539;
                
                // benchmark/ba/ba_gradbench_original.fut:21:36-46
                
                double zt_rhs_22542 = 1.0 - cos_res_22469;
                
                // benchmark/ba/ba_gradbench_original.fut:21:36-46
                
                double binop_y_22543 = -1.0 * cos_res_tan_22472;
                
                // benchmark/ba/ba_gradbench_original.fut:21:31-46
                
                double tmp_22544 = zp_res_22540 * zt_rhs_22542;
                
                // benchmark/ba/ba_gradbench_original.fut:21:31-46
                
                double binop_x_22545 = zp_res_tan_22541 * zt_rhs_22542;
                
                // benchmark/ba/ba_gradbench_original.fut:21:31-46
                
                double binop_y_22546 = zp_res_22540 * binop_y_22543;
                
                // benchmark/ba/ba_gradbench_original.fut:21:31-46
                
                double tmp_tan_22547 = binop_x_22545 + binop_y_22546;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22548 = zm_res_20489 * cos_res_22469;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22549 = zm_res_tan_21719 * cos_res_22469;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22550 = zm_res_20489 * cos_res_tan_22472;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22551 = binop_x_22549 + binop_y_22550;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22552 = zm_res_20490 * cos_res_22469;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22553 = zm_res_tan_21722 * cos_res_22469;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22554 = zm_res_20490 * cos_res_tan_22472;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22555 = binop_x_22553 + binop_y_22554;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22556 = zm_res_20491 * cos_res_22469;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22557 = zm_res_tan_21725 * cos_res_22469;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22558 = zm_res_20491 * cos_res_tan_22472;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22559 = binop_x_22557 + binop_y_22558;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22560 = binop_y_22470 * zm_res_22501;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22561 = sin_res_tan_22473 * zm_res_22501;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22562 = binop_y_22470 * zm_res_tan_22503;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22563 = binop_x_22561 + binop_y_22562;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22564 = binop_y_22470 * zm_res_22512;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22565 = sin_res_tan_22473 * zm_res_22512;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22566 = binop_y_22470 * zm_res_tan_22514;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22567 = binop_x_22565 + binop_y_22566;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22568 = binop_y_22470 * zm_res_22523;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22569 = sin_res_tan_22473 * zm_res_22523;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22570 = binop_y_22470 * zm_res_tan_22525;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22571 = binop_x_22569 + binop_y_22570;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_22572 = zt_res_22548 + zt_res_22560;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_22573 = zt_res_tan_22551 + zt_res_tan_22563;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_22574 = zt_res_22552 + zt_res_22564;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_22575 = zt_res_tan_22555 + zt_res_tan_22567;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_22576 = zt_res_22556 + zt_res_22568;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_22577 = zt_res_tan_22559 + zt_res_tan_22571;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22578 = zt_res_22481 * tmp_22544;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22579 = zt_res_tan_22484 * tmp_22544;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22580 = zt_res_22481 * tmp_tan_22547;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22581 = binop_x_22579 + binop_y_22580;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22582 = zt_res_22485 * tmp_22544;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22583 = zt_res_tan_22488 * tmp_22544;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22584 = zt_res_22485 * tmp_tan_22547;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22585 = binop_x_22583 + binop_y_22584;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22586 = zt_res_22489 * tmp_22544;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22587 = zt_res_tan_22492 * tmp_22544;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22588 = zt_res_22489 * tmp_tan_22547;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22589 = binop_x_22587 + binop_y_22588;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_22590 = zp_res_22572 + zt_res_22578;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_22591 = zp_res_tan_22573 + zt_res_tan_22581;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_22592 = zp_res_22574 + zt_res_22582;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_22593 = zp_res_tan_22575 + zt_res_tan_22585;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_22594 = zp_res_22576 + zt_res_22586;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_22595 = zp_res_tan_22577 + zt_res_tan_22589;
                
                rodrigues_rotate_point_res_20499 = zp_res_22590;
                rodrigues_rotate_point_res_tan_21899 = zp_res_tan_22591;
                rodrigues_rotate_point_res_20500 = zp_res_22592;
                rodrigues_rotate_point_res_tan_21900 = zp_res_tan_22593;
                rodrigues_rotate_point_res_20501 = zp_res_22594;
                rodrigues_rotate_point_res_tan_21901 = zp_res_tan_22595;
            } else {
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double zt_res_20540 = lifted_lambda_res_20485 * zm_res_20491;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double binop_x_21864 = zm_res_20491 * lifted_lambda_res_tan_21715;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double binop_y_21865 = lifted_lambda_res_20485 * zm_res_tan_21725;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double zt_res_tan_21863 = binop_x_21864 + binop_y_21865;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double zt_res_20541 = lifted_lambda_res_20486 * zm_res_20490;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double binop_x_21867 = zm_res_20490 * lifted_lambda_res_tan_21716;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double binop_y_21868 = lifted_lambda_res_20486 * zm_res_tan_21722;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double zt_res_tan_21866 = binop_x_21867 + binop_y_21868;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:16-25
                
                double zm_res_20542 = zt_res_20540 - zt_res_20541;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:16-25
                
                double binop_y_21871 = -1.0 * zt_res_tan_21866;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:16-25
                
                double zm_res_tan_21869 = zt_res_tan_21863 + binop_y_21871;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double zt_res_20543 = lifted_lambda_res_20486 * zm_res_20489;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double binop_x_21873 = zm_res_20489 * lifted_lambda_res_tan_21716;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double binop_y_21874 = lifted_lambda_res_20486 * zm_res_tan_21719;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double zt_res_tan_21872 = binop_x_21873 + binop_y_21874;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double zt_res_20544 = lifted_lambda_res_20484 * zm_res_20491;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double binop_x_21876 = zm_res_20491 * lifted_lambda_res_tan_21714;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double binop_y_21877 = lifted_lambda_res_20484 * zm_res_tan_21725;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double zt_res_tan_21875 = binop_x_21876 + binop_y_21877;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:31-40
                
                double zm_res_20545 = zt_res_20543 - zt_res_20544;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:31-40
                
                double binop_y_21880 = -1.0 * zt_res_tan_21875;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:31-40
                
                double zm_res_tan_21878 = zt_res_tan_21872 + binop_y_21880;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double zt_res_20546 = lifted_lambda_res_20484 * zm_res_20490;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double binop_x_21882 = zm_res_20490 * lifted_lambda_res_tan_21714;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double binop_y_21883 = lifted_lambda_res_20484 * zm_res_tan_21722;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double zt_res_tan_21881 = binop_x_21882 + binop_y_21883;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double zt_res_20547 = lifted_lambda_res_20485 * zm_res_20489;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double binop_x_21885 = zm_res_20489 * lifted_lambda_res_tan_21715;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double binop_y_21886 = lifted_lambda_res_20485 * zm_res_tan_21719;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double zt_res_tan_21884 = binop_x_21885 + binop_y_21886;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:46-55
                
                double zm_res_20548 = zt_res_20546 - zt_res_20547;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:46-55
                
                double binop_y_21889 = -1.0 * zt_res_tan_21884;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:46-55
                
                double zm_res_tan_21887 = zt_res_tan_21881 + binop_y_21889;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_20549 = zm_res_20489 + zm_res_20542;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_21890 = zm_res_tan_21719 + zm_res_tan_21869;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_20550 = zm_res_20490 + zm_res_20545;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_21893 = zm_res_tan_21722 + zm_res_tan_21878;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_20551 = zm_res_20491 + zm_res_20548;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_21896 = zm_res_tan_21725 + zm_res_tan_21887;
                
                rodrigues_rotate_point_res_20499 = zp_res_20549;
                rodrigues_rotate_point_res_tan_21899 = zp_res_tan_21890;
                rodrigues_rotate_point_res_20500 = zp_res_20550;
                rodrigues_rotate_point_res_tan_21900 = zp_res_tan_21893;
                rodrigues_rotate_point_res_20501 = zp_res_20551;
                rodrigues_rotate_point_res_tan_21901 = zp_res_tan_21896;
            }
            // benchmark/ba/ba_gradbench_original.fut:37:55-62
            
            double scale_arg0_20552 = 1.0 / rodrigues_rotate_point_res_20501;
            
            // benchmark/ba/ba_gradbench_original.fut:37:55-62
            
            double binop_x_21904 = 0.0 * scale_arg0_20552;
            
            // benchmark/ba/ba_gradbench_original.fut:37:55-62
            
            double binop_y_21905 = rodrigues_rotate_point_res_20501 * rodrigues_rotate_point_res_20501;
            
            // benchmark/ba/ba_gradbench_original.fut:37:55-62
            
            double binop_y_21906 = 1.0 / binop_y_21905;
            
            // benchmark/ba/ba_gradbench_original.fut:37:55-62
            
            double binop_y_21907 = 0.0 - binop_y_21906;
            
            // benchmark/ba/ba_gradbench_original.fut:37:55-62
            
            double binop_y_21908 = rodrigues_rotate_point_res_tan_21901 * binop_y_21907;
            
            // benchmark/ba/ba_gradbench_original.fut:37:55-62
            
            double scale_arg0_tan_21902 = binop_x_21904 + binop_y_21908;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_20553 = rodrigues_rotate_point_res_20499 * scale_arg0_20552;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_21910 = scale_arg0_20552 * rodrigues_rotate_point_res_tan_21899;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_21911 = rodrigues_rotate_point_res_20499 * scale_arg0_tan_21902;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_21909 = binop_x_21910 + binop_y_21911;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_20554 = rodrigues_rotate_point_res_20500 * scale_arg0_20552;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_21913 = scale_arg0_20552 * rodrigues_rotate_point_res_tan_21900;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_21914 = rodrigues_rotate_point_res_20500 * scale_arg0_tan_21902;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_21912 = binop_x_21913 + binop_y_21914;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:11-18
            
            double zt_res_20555 = zt_res_20553 * zt_res_20553;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:11-18
            
            double binop_x_21916 = zt_res_20553 * zt_res_tan_21909;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:11-18
            
            double zt_res_tan_21915 = binop_x_21916 + binop_x_21916;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:21-28
            
            double zt_res_20556 = zt_res_20554 * zt_res_20554;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:21-28
            
            double binop_x_21919 = zt_res_20554 * zt_res_tan_21912;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:21-28
            
            double zt_res_tan_21918 = binop_x_21919 + binop_x_21919;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:14-28
            
            double zp_res_20557 = zt_res_20555 + zt_res_20556;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:14-28
            
            double zp_res_tan_21921 = zt_res_tan_21915 + zt_res_tan_21918;
            
            // benchmark/ba/ba_gradbench_original.fut:30:28-33
            
            double zp_rhs_20558 = lifted_lambda_res_20482 * zp_res_20557;
            
            // benchmark/ba/ba_gradbench_original.fut:30:28-33
            
            double binop_x_21925 = zp_res_20557 * lifted_lambda_res_tan_21712;
            
            // benchmark/ba/ba_gradbench_original.fut:30:28-33
            
            double binop_y_21926 = lifted_lambda_res_20482 * zp_res_tan_21921;
            
            // benchmark/ba/ba_gradbench_original.fut:30:28-33
            
            double zp_rhs_tan_21924 = binop_x_21925 + binop_y_21926;
            
            // benchmark/ba/ba_gradbench_original.fut:30:13-33
            
            double zp_lhs_20559 = 1.0 + zp_rhs_20558;
            
            // benchmark/ba/ba_gradbench_original.fut:30:49-54
            
            double zt_lhs_20560 = lifted_lambda_res_20483 * zp_res_20557;
            
            // benchmark/ba/ba_gradbench_original.fut:30:49-54
            
            double binop_x_21931 = zp_res_20557 * lifted_lambda_res_tan_21713;
            
            // benchmark/ba/ba_gradbench_original.fut:30:49-54
            
            double binop_y_21932 = lifted_lambda_res_20483 * zp_res_tan_21921;
            
            // benchmark/ba/ba_gradbench_original.fut:30:49-54
            
            double zt_lhs_tan_21930 = binop_x_21931 + binop_y_21932;
            
            // benchmark/ba/ba_gradbench_original.fut:30:55-60
            
            double zp_rhs_20561 = zp_res_20557 * zt_lhs_20560;
            
            // benchmark/ba/ba_gradbench_original.fut:30:55-60
            
            double binop_x_21934 = zt_lhs_20560 * zp_res_tan_21921;
            
            // benchmark/ba/ba_gradbench_original.fut:30:55-60
            
            double binop_y_21935 = zp_res_20557 * zt_lhs_tan_21930;
            
            // benchmark/ba/ba_gradbench_original.fut:30:55-60
            
            double zp_rhs_tan_21933 = binop_x_21934 + binop_y_21935;
            
            // benchmark/ba/ba_gradbench_original.fut:30:34-60
            
            double L_20562 = zp_lhs_20559 + zp_rhs_20561;
            
            // benchmark/ba/ba_gradbench_original.fut:30:34-60
            
            double L_tan_21936 = zp_rhs_tan_21924 + zp_rhs_tan_21933;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_20563 = zt_res_20553 * L_20562;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_21940 = L_20562 * zt_res_tan_21909;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_21941 = zt_res_20553 * L_tan_21936;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_21939 = binop_x_21940 + binop_y_21941;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_20564 = zt_res_20554 * L_20562;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_21943 = L_20562 * zt_res_tan_21912;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_21944 = zt_res_20554 * L_tan_21936;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_21942 = binop_x_21943 + binop_y_21944;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_20565 = lifted_lambda_res_20481 * zt_res_20563;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_21946 = zt_res_20563 * lifted_lambda_res_tan_21711;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_21947 = lifted_lambda_res_20481 * zt_res_tan_21939;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_21945 = binop_x_21946 + binop_y_21947;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_20566 = lifted_lambda_res_20481 * zt_res_20564;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_21949 = zt_res_20564 * lifted_lambda_res_tan_21711;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_21950 = lifted_lambda_res_20481 * zt_res_tan_21942;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_21948 = binop_x_21949 + binop_y_21950;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:87:18-26
            
            double zp_res_20567 = lifted_lambda_res_20487 + zt_res_20565;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:87:18-26
            
            double zp_res_tan_21951 = lifted_lambda_res_tan_21717 + zt_res_tan_21945;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:87:18-26
            
            double zp_res_20568 = lifted_lambda_res_20488 + zt_res_20566;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:87:18-26
            
            double zp_res_tan_21954 = lifted_lambda_res_tan_21718 + zt_res_tan_21948;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:88:18-26
            
            double zm_res_20569 = zp_res_20567 - tmp_20443;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:88:18-26
            
            double zm_res_20570 = zp_res_20568 - tmp_20446;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_21966 = zm_res_20569 * compute_reproj_err_arg2_tan_21684;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_21967 = compute_reproj_err_arg2_20460 * zp_res_tan_21951;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_21965 = binop_x_21966 + binop_y_21967;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_21969 = zm_res_20570 * compute_reproj_err_arg2_tan_21684;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_21970 = compute_reproj_err_arg2_20460 * zp_res_tan_21954;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_21968 = binop_x_21969 + binop_y_21970;
            
            ((double *) mem_23480)[i_22988] = zt_res_tan_21965;
            ((double *) mem_23481)[i_22988] = zt_res_tan_21968;
        }
        // benchmark/ba/bench_jvp_ba_simple.fut:73:6-79:35
        for (int64_t i_22993 = 0; i_22993 < (int64_t) 48; i_22993++) {
            // benchmark/ba/bench_jvp_ba_simple.fut:74:8-79:35
            
            bool cond_20584 = slt64(i_22993, (int64_t) 32);
            double lifted_lambda_res_tan_22003;
            
            if (cond_20584) {
                // benchmark/ba/bench_jvp_ba_simple.fut:75:21-27
                
                int64_t ob_22596 = sdiv64(i_22993, (int64_t) 2);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:76:18-32
                
                bool x_22597 = sle64((int64_t) 0, ob_22596);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:76:18-32
                
                bool y_22598 = slt64(ob_22596, (int64_t) 16);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:76:18-32
                
                bool bounds_check_22599 = x_22597 && y_22598;
                
                // benchmark/ba/bench_jvp_ba_simple.fut:76:18-32
                
                bool index_certs_22600;
                
                if (!bounds_check_22599) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) ob_22596, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:76:18-32\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:161:33-81\n   #2  src/dense_jacobian.fut:8:30-9:58\n   #3  src/dense_jacobian.fut:8:40-9:68\n   #4  benchmark/ba/bench_jvp_ba_simple.fut:56:8-161:84\n   #5  benchmark/ba/bench_jvp_ba_simple.fut:193:30-196:60\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // benchmark/ba/bench_jvp_ba_simple.fut:76:18-32
                
                double lifted_lambda_res_t_res_tan_22601 = ((double *) mem_23480)[ob_22596];
                
                // benchmark/ba/bench_jvp_ba_simple.fut:76:18-32
                
                double lifted_lambda_res_t_res_tan_22602 = ((double *) mem_23481)[ob_22596];
                
                // benchmark/ba/bench_jvp_ba_simple.fut:77:18-24
                
                int64_t zeze_lhs_22603 = smod64(i_22993, (int64_t) 2);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:77:13-50
                
                bool cond_22604 = zeze_lhs_22603 == (int64_t) 0;
                double lifted_lambda_res_t_res_tan_22605;
                
                if (cond_22604) {
                    lifted_lambda_res_t_res_tan_22605 = lifted_lambda_res_t_res_tan_22601;
                } else {
                    lifted_lambda_res_t_res_tan_22605 = lifted_lambda_res_t_res_tan_22602;
                }
                lifted_lambda_res_tan_22003 = lifted_lambda_res_t_res_tan_22605;
            } else {
                // benchmark/ba/bench_jvp_ba_simple.fut:79:23-34
                
                int64_t tmp_20597 = sub64(i_22993, (int64_t) 32);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:79:10-35
                
                bool x_20598 = sle64((int64_t) 0, tmp_20597);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:79:10-35
                
                bool y_20599 = slt64(tmp_20597, (int64_t) 16);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:79:10-35
                
                bool bounds_check_20600 = x_20598 && y_20599;
                
                // benchmark/ba/bench_jvp_ba_simple.fut:79:10-35
                
                bool index_certs_20601;
                
                if (!bounds_check_20600) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_20597, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:79:10-35\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:161:33-81\n   #2  src/dense_jacobian.fut:8:30-9:58\n   #3  src/dense_jacobian.fut:8:40-9:68\n   #4  benchmark/ba/bench_jvp_ba_simple.fut:56:8-161:84\n   #5  benchmark/ba/bench_jvp_ba_simple.fut:193:30-196:60\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // benchmark/ba/bench_jvp_ba_simple.fut:79:10-35
                
                double lifted_lambda_res_f_res_tan_22002 = ((double *) mem_23319)[tmp_20597];
                
                lifted_lambda_res_tan_22003 = lifted_lambda_res_f_res_tan_22002;
            }
            ((double *) mem_23494)[i_22993] = lifted_lambda_res_tan_22003;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_23225, i_22997 * (int64_t) 48, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_23494, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 48});
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:16:15-35
    if (mem_23506_cached_sizze_24022 < bytes_23505) {
        err = lexical_realloc(ctx, &mem_23506, &mem_23506_cached_sizze_24022, bytes_23505);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:16:15-35
    for (int64_t nest_i_23918 = 0; nest_i_23918 < csr_bipartite_from_pattern_res_19997; nest_i_23918++) {
        ((double *) mem_23506)[nest_i_23918] = 0.0;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:19:5-27:27
    
    bool dense_to_csr_vals_res_20025;
    int64_t dense_to_csr_vals_res_20027;
    bool loop_while_20028;
    int64_t i_20030;
    
    loop_while_20028 = 1;
    i_20030 = (int64_t) 0;
    while (loop_while_20028) {
        // benchmark/ba/bench_jvp_ba_simple.fut:21:15-26
        
        bool x_20031 = sle64((int64_t) 0, i_20030);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:21:15-26
        
        bool y_20032 = slt64(i_20030, csr_bipartite_from_pattern_res_19996);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:21:15-26
        
        bool bounds_check_20033 = x_20031 && y_20032;
        
        // benchmark/ba/bench_jvp_ba_simple.fut:21:15-26
        
        bool index_certs_20034;
        
        if (!bounds_check_20033) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_20030, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_19996, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:21:15-26\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:154:14-162:42\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:193:30-196:60\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // benchmark/ba/bench_jvp_ba_simple.fut:21:15-26
        
        int64_t s_20035 = ((int64_t *) ext_mem_23220.mem)[i_20030];
        
        // benchmark/ba/bench_jvp_ba_simple.fut:22:25-27
        
        int64_t e_20036 = add64((int64_t) 1, i_20030);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:22:15-28
        
        bool x_20037 = sle64((int64_t) 0, e_20036);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:22:15-28
        
        bool y_20038 = slt64(e_20036, csr_bipartite_from_pattern_res_19996);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:22:15-28
        
        bool bounds_check_20039 = x_20037 && y_20038;
        
        // benchmark/ba/bench_jvp_ba_simple.fut:22:15-28
        
        bool index_certs_20040;
        
        if (!bounds_check_20039) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_20036, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_19996, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:22:15-28\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:154:14-162:42\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:193:30-196:60\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // benchmark/ba/bench_jvp_ba_simple.fut:22:15-28
        
        int64_t e_20041 = ((int64_t *) ext_mem_23220.mem)[e_20036];
        
        // benchmark/ba/bench_jvp_ba_simple.fut:23:18-30
        
        int64_t j_m_i_20042 = sub64(e_20041, s_20035);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:23:18-30
        
        bool empty_slice_20043 = j_m_i_20042 == (int64_t) 0;
        
        // benchmark/ba/bench_jvp_ba_simple.fut:23:18-30
        
        int64_t m_20044 = sub64(j_m_i_20042, (int64_t) 1);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:23:18-30
        
        int64_t i_p_m_t_s_20045 = add64(s_20035, m_20044);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:23:18-30
        
        bool zzero_leq_i_p_m_t_s_20046 = sle64((int64_t) 0, i_p_m_t_s_20045);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:23:18-30
        
        bool i_p_m_t_s_leq_w_20047 = slt64(i_p_m_t_s_20045, csr_bipartite_from_pattern_res_19997);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:23:18-30
        
        bool zzero_lte_i_20048 = sle64((int64_t) 0, s_20035);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:23:18-30
        
        bool i_lte_j_20049 = sle64(s_20035, e_20041);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:23:18-30
        
        bool y_20050 = i_p_m_t_s_leq_w_20047 && zzero_lte_i_20048;
        
        // benchmark/ba/bench_jvp_ba_simple.fut:23:18-30
        
        bool y_20051 = zzero_leq_i_p_m_t_s_20046 && y_20050;
        
        // benchmark/ba/bench_jvp_ba_simple.fut:23:18-30
        
        bool forwards_ok_20052 = i_lte_j_20049 && y_20051;
        
        // benchmark/ba/bench_jvp_ba_simple.fut:23:18-30
        
        bool ok_or_empty_20053 = empty_slice_20043 || forwards_ok_20052;
        
        // benchmark/ba/bench_jvp_ba_simple.fut:23:18-30
        
        bool index_certs_20054;
        
        if (!ok_or_empty_20053) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_20035, ":", (long long) e_20041, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_19997, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:23:18-30\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:154:14-162:42\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:193:30-196:60\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // benchmark/ba/bench_jvp_ba_simple.fut:24:18-22
        
        bool y_20056 = slt64(i_20030, (int64_t) 48);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:24:18-22
        
        bool bounds_check_20057 = x_20031 && y_20056;
        
        // benchmark/ba/bench_jvp_ba_simple.fut:24:18-22
        
        bool index_certs_20058;
        
        if (!bounds_check_20057) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_20030, "] out of bounds for array of shape [", (long long) (int64_t) 48, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:24:18-22\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:154:14-162:42\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:193:30-196:60\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // benchmark/ba/bench_jvp_ba_simple.fut:25:17-45
        for (int64_t i_23001 = 0; i_23001 < j_m_i_20042; i_23001++) {
            int64_t index_primexp_23158 = s_20035 + i_23001;
            int64_t eta_p_20060 = ((int64_t *) ext_mem_23219.mem)[index_primexp_23158];
            
            // benchmark/ba/bench_jvp_ba_simple.fut:25:30-39
            
            bool x_20061 = sle64((int64_t) 0, eta_p_20060);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:25:30-39
            
            bool y_20062 = slt64(eta_p_20060, (int64_t) 62);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:25:30-39
            
            bool bounds_check_20063 = x_20061 && y_20062;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:25:30-39
            
            bool index_certs_20064;
            
            if (!bounds_check_20063) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_20060, "] out of bounds for array of shape [", (long long) (int64_t) 62, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:25:30-39\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:154:14-162:42\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:193:30-196:60\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // benchmark/ba/bench_jvp_ba_simple.fut:193:30-196:60
            
            double lifted_lambda_res_20065 = ((double *) mem_23225)[eta_p_20060 * (int64_t) 48 + i_20030];
            
            ((double *) mem_23506)[s_20035 + i_23001] = lifted_lambda_res_20065;
        }
        // benchmark/ba/bench_jvp_ba_simple.fut:20:13-16
        
        bool loop_cond_20067 = slt64(e_20036, (int64_t) 48);
        bool loop_while_tmp_23919 = loop_cond_20067;
        int64_t i_tmp_23921 = e_20036;
        
        loop_while_20028 = loop_while_tmp_23919;
        i_20030 = i_tmp_23921;
    }
    dense_to_csr_vals_res_20025 = loop_while_20028;
    dense_to_csr_vals_res_20027 = i_20030;
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    if (mem_23549_cached_sizze_24025 < bytes_23548) {
        err = lexical_realloc(ctx, &mem_23549, &mem_23549_cached_sizze_24025, bytes_23548);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:17:3-61
    if (mem_23554_cached_sizze_24026 < (int64_t) 496) {
        err = lexical_realloc(ctx, &mem_23554, &mem_23554_cached_sizze_24026, (int64_t) 496);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:36:5-38:24
    if (mem_23561_cached_sizze_24027 < (int64_t) 176) {
        err = lexical_realloc(ctx, &mem_23561, &mem_23561_cached_sizze_24027, (int64_t) 176);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:36:5-38:24
    if (mem_23562_cached_sizze_24028 < (int64_t) 176) {
        err = lexical_realloc(ctx, &mem_23562, &mem_23562_cached_sizze_24028, (int64_t) 176);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:37:7-38:23
    if (mem_23571_cached_sizze_24029 < (int64_t) 88) {
        err = lexical_realloc(ctx, &mem_23571, &mem_23571_cached_sizze_24029, (int64_t) 88);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:37:7-38:23
    if (mem_23572_cached_sizze_24030 < (int64_t) 88) {
        err = lexical_realloc(ctx, &mem_23572, &mem_23572_cached_sizze_24030, (int64_t) 88);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:34:38-65:31
    if (mem_23593_cached_sizze_24031 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_23593, &mem_23593_cached_sizze_24031, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:34:38-65:31
    if (mem_23594_cached_sizze_24032 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_23594, &mem_23594_cached_sizze_24032, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:34:38-65:31
    if (mem_23595_cached_sizze_24033 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_23595, &mem_23595_cached_sizze_24033, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:34:38-65:31
    if (mem_23596_cached_sizze_24034 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_23596, &mem_23596_cached_sizze_24034, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:34:38-65:31
    if (mem_23597_cached_sizze_24035 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_23597, &mem_23597_cached_sizze_24035, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:34:38-65:31
    if (mem_23598_cached_sizze_24036 < (int64_t) 64) {
        err = lexical_realloc(ctx, &mem_23598, &mem_23598_cached_sizze_24036, (int64_t) 64);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:42:7-43:39
    if (mem_23617_cached_sizze_24037 < (int64_t) 24) {
        err = lexical_realloc(ctx, &mem_23617, &mem_23617_cached_sizze_24037, (int64_t) 24);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:42:7-43:39
    if (mem_23618_cached_sizze_24038 < (int64_t) 24) {
        err = lexical_realloc(ctx, &mem_23618, &mem_23618_cached_sizze_24038, (int64_t) 24);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:52:15-46
    if (mem_23649_cached_sizze_24039 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_23649, &mem_23649_cached_sizze_24039, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:52:15-46
    if (mem_23650_cached_sizze_24040 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_23650, &mem_23650_cached_sizze_24040, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:52:15-46
    if (mem_23651_cached_sizze_24041 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_23651, &mem_23651_cached_sizze_24041, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23670_cached_sizze_24042 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23670, &mem_23670_cached_sizze_24042, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23671_cached_sizze_24043 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23671, &mem_23671_cached_sizze_24043, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23672_cached_sizze_24044 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23672, &mem_23672_cached_sizze_24044, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23673_cached_sizze_24045 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23673, &mem_23673_cached_sizze_24045, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23674_cached_sizze_24046 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23674, &mem_23674_cached_sizze_24046, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23675_cached_sizze_24047 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23675, &mem_23675_cached_sizze_24047, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23676_cached_sizze_24048 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23676, &mem_23676_cached_sizze_24048, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23677_cached_sizze_24049 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23677, &mem_23677_cached_sizze_24049, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23678_cached_sizze_24050 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23678, &mem_23678_cached_sizze_24050, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23679_cached_sizze_24051 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23679, &mem_23679_cached_sizze_24051, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23680_cached_sizze_24052 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23680, &mem_23680_cached_sizze_24052, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23681_cached_sizze_24053 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23681, &mem_23681_cached_sizze_24053, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23682_cached_sizze_24054 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23682, &mem_23682_cached_sizze_24054, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23683_cached_sizze_24055 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23683, &mem_23683_cached_sizze_24055, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23684_cached_sizze_24056 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23684, &mem_23684_cached_sizze_24056, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23685_cached_sizze_24057 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23685, &mem_23685_cached_sizze_24057, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23686_cached_sizze_24058 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23686, &mem_23686_cached_sizze_24058, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23687_cached_sizze_24059 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23687, &mem_23687_cached_sizze_24059, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23688_cached_sizze_24060 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23688, &mem_23688_cached_sizze_24060, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
    if (mem_23689_cached_sizze_24061 < (int64_t) 16) {
        err = lexical_realloc(ctx, &mem_23689, &mem_23689_cached_sizze_24061, (int64_t) 16);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:48:5-51:49
    if (mem_23810_cached_sizze_24062 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_23810, &mem_23810_cached_sizze_24062, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/ba_gradbench_original.fut:48:5-51:49
    if (mem_23811_cached_sizze_24063 < (int64_t) 128) {
        err = lexical_realloc(ctx, &mem_23811, &mem_23811_cached_sizze_24063, (int64_t) 128);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // benchmark/ba/bench_jvp_ba_simple.fut:73:6-79:35
    if (mem_23824_cached_sizze_24064 < (int64_t) 384) {
        err = lexical_realloc(ctx, &mem_23824, &mem_23824_cached_sizze_24064, (int64_t) 384);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:21:18-29:18
    for (int64_t i_23133 = 0; i_23133 < num_colors_of_res_f_res_20173; i_23133++) {
        // src/sparse_jacobian_jvp.fut:17:3-61
        for (int64_t i_23007 = 0; i_23007 < (int64_t) 62; i_23007++) {
            int64_t eta_p_20182 = ((int64_t *) mem_23521)[i_23007];
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            bool cond_20183 = eta_p_20182 == i_23133;
            
            // src/sparse_jacobian_jvp.fut:17:17-53
            
            double lifted_lambda_res_20184;
            
            if (cond_20183) {
                lifted_lambda_res_20184 = 1.0;
            } else {
                lifted_lambda_res_20184 = 0.0;
            }
            ((double *) mem_23554)[i_23007] = lifted_lambda_res_20184;
        }
        // benchmark/ba/bench_jvp_ba_simple.fut:36:5-38:24
        for (int64_t i_23020 = 0; i_23020 < (int64_t) 2; i_23020++) {
            // benchmark/ba/bench_jvp_ba_simple.fut:38:16-18
            
            int64_t zp_lhs_20618 = mul64((int64_t) 11, i_23020);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:37:7-38:23
            for (int64_t i_23013 = 0; i_23013 < (int64_t) 11; i_23013++) {
                // benchmark/ba/bench_jvp_ba_simple.fut:38:19-22
                
                int64_t tmp_20621 = add64(zp_lhs_20618, i_23013);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:38:9-23
                
                bool x_20622 = sle64((int64_t) 0, tmp_20621);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:38:9-23
                
                bool y_20623 = slt64(tmp_20621, (int64_t) 62);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:38:9-23
                
                bool bounds_check_20624 = x_20622 && y_20623;
                
                // benchmark/ba/bench_jvp_ba_simple.fut:38:9-23
                
                bool index_certs_20625;
                
                if (!bounds_check_20624) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_20621, "] out of bounds for array of shape [", (long long) (int64_t) 62, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:38:9-23\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:59:5-46\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:179:15-63\n   #3  src/sparse_jacobian_jvp.fut:21:8-28:28\n   #4  src/sparse_jacobian_jvp.fut:21:18-29:18\n   #5  benchmark/ba/bench_jvp_ba_simple.fut:56:8-181:8\n   #6  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                
                double lifted_lambda_res_20626 = ((double *) mem_23194)[tmp_20621];
                double lifted_lambda_res_tan_22023 = ((double *) mem_23554)[tmp_20621];
                
                ((double *) mem_23571)[i_23013] = lifted_lambda_res_20626;
                ((double *) mem_23572)[i_23013] = lifted_lambda_res_tan_22023;
            }
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_23561, i_23020 * (int64_t) 11, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_23571, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 11});
            lmad_copy_8b(ctx, 1, (uint64_t *) mem_23562, i_23020 * (int64_t) 11, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_23572, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 11});
        }
        // benchmark/ba/bench_jvp_ba_simple.fut:34:38-65:31
        for (int64_t i_23042 = 0; i_23042 < (int64_t) 8; i_23042++) {
            // benchmark/ba/bench_jvp_ba_simple.fut:43:32-34
            
            int64_t zp_rhs_21168 = mul64((int64_t) 3, i_23042);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:43:26-34
            
            int64_t zp_lhs_21169 = add64((int64_t) 22, zp_rhs_21168);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:42:7-43:39
            for (int64_t i_23027 = 0; i_23027 < (int64_t) 3; i_23027++) {
                // benchmark/ba/bench_jvp_ba_simple.fut:43:35-38
                
                int64_t tmp_21172 = add64(zp_lhs_21169, i_23027);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:43:9-39
                
                bool x_21173 = sle64((int64_t) 0, tmp_21172);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:43:9-39
                
                bool y_21174 = slt64(tmp_21172, (int64_t) 62);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:43:9-39
                
                bool bounds_check_21175 = x_21173 && y_21174;
                
                // benchmark/ba/bench_jvp_ba_simple.fut:43:9-39
                
                bool index_certs_21176;
                
                if (!bounds_check_21175) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_21172, "] out of bounds for array of shape [", (long long) (int64_t) 62, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:43:9-39\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:59:5-46\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:179:15-63\n   #3  src/sparse_jacobian_jvp.fut:21:8-28:28\n   #4  src/sparse_jacobian_jvp.fut:21:18-29:18\n   #5  benchmark/ba/bench_jvp_ba_simple.fut:56:8-181:8\n   #6  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                
                double lifted_lambda_res_21177 = ((double *) mem_23194)[tmp_21172];
                double lifted_lambda_res_tan_22047 = ((double *) mem_23554)[tmp_21172];
                
                ((double *) mem_23617)[i_23027] = lifted_lambda_res_21177;
                ((double *) mem_23618)[i_23027] = lifted_lambda_res_tan_22047;
            }
            // benchmark/ba/ba_gradbench_original.fut:5:31-35
            
            double tmp_21179 = ((double *) mem_23617)[(int64_t) 0];
            
            // benchmark/ba/ba_gradbench_original.fut:5:31-35
            
            double tmp_tan_22048 = ((double *) mem_23618)[(int64_t) 0];
            
            // benchmark/ba/ba_gradbench_original.fut:5:39-43
            
            double tmp_21180 = ((double *) mem_23617)[(int64_t) 1];
            
            // benchmark/ba/ba_gradbench_original.fut:5:39-43
            
            double tmp_tan_22049 = ((double *) mem_23618)[(int64_t) 1];
            
            // benchmark/ba/ba_gradbench_original.fut:5:47-51
            
            double tmp_21181 = ((double *) mem_23617)[(int64_t) 2];
            
            // benchmark/ba/ba_gradbench_original.fut:5:47-51
            
            double tmp_tan_22050 = ((double *) mem_23618)[(int64_t) 2];
            
            ((double *) mem_23593)[i_23042] = tmp_21179;
            ((double *) mem_23594)[i_23042] = tmp_tan_22048;
            ((double *) mem_23595)[i_23042] = tmp_21180;
            ((double *) mem_23596)[i_23042] = tmp_tan_22049;
            ((double *) mem_23597)[i_23042] = tmp_21181;
            ((double *) mem_23598)[i_23042] = tmp_tan_22050;
        }
        // benchmark/ba/ba_gradbench_original.fut:52:15-46
        for (int64_t i_23055 = 0; i_23055 < (int64_t) 16; i_23055++) {
            // benchmark/ba/bench_jvp_ba_simple.fut:47:42-45
            
            int64_t tmp_21157 = add64((int64_t) 46, i_23055);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:47:7-46
            
            bool x_21158 = sle64((int64_t) 0, tmp_21157);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:47:7-46
            
            bool y_21159 = slt64(tmp_21157, (int64_t) 62);
            
            // benchmark/ba/bench_jvp_ba_simple.fut:47:7-46
            
            bool bounds_check_21160 = x_21158 && y_21159;
            
            // benchmark/ba/bench_jvp_ba_simple.fut:47:7-46
            
            bool index_certs_21161;
            
            if (!bounds_check_21160) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_21157, "] out of bounds for array of shape [", (long long) (int64_t) 62, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:47:7-46\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:59:5-46\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:179:15-63\n   #3  src/sparse_jacobian_jvp.fut:21:8-28:28\n   #4  src/sparse_jacobian_jvp.fut:21:18-29:18\n   #5  benchmark/ba/bench_jvp_ba_simple.fut:56:8-181:8\n   #6  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            
            double lifted_lambda_res_21162 = ((double *) mem_23194)[tmp_21157];
            double lifted_lambda_res_tan_22064 = ((double *) mem_23554)[tmp_21157];
            
            // benchmark/ba/ba_gradbench_original.fut:44:8-10
            
            double binop_x_22066 = lifted_lambda_res_21162 * lifted_lambda_res_tan_22064;
            
            // benchmark/ba/ba_gradbench_original.fut:44:8-10
            
            double zm_rhs_tan_22065 = binop_x_22066 + binop_x_22066;
            
            // benchmark/ba/ba_gradbench_original.fut:44:5-10
            
            double binop_y_22070 = -1.0 * zm_rhs_tan_22065;
            
            ((double *) mem_23649)[i_23055] = binop_y_22070;
            ((double *) mem_23650)[i_23055] = lifted_lambda_res_21162;
            ((double *) mem_23651)[i_23055] = lifted_lambda_res_tan_22064;
        }
        // benchmark/ba/ba_gradbench_original.fut:87:23-62:31
        for (int64_t i_23099 = 0; i_23099 < (int64_t) 2; i_23099++) {
            double tmp_20677 = ((double *) mem_23561)[i_23099 * (int64_t) 11];
            double tmp_tan_22089 = ((double *) mem_23562)[i_23099 * (int64_t) 11];
            double tmp_20680 = ((double *) mem_23561)[i_23099 * (int64_t) 11 + (int64_t) 1];
            double tmp_tan_22090 = ((double *) mem_23562)[i_23099 * (int64_t) 11 + (int64_t) 1];
            double tmp_20683 = ((double *) mem_23561)[i_23099 * (int64_t) 11 + (int64_t) 2];
            double tmp_tan_22091 = ((double *) mem_23562)[i_23099 * (int64_t) 11 + (int64_t) 2];
            double tmp_20686 = ((double *) mem_23561)[i_23099 * (int64_t) 11 + (int64_t) 3];
            double tmp_tan_22092 = ((double *) mem_23562)[i_23099 * (int64_t) 11 + (int64_t) 3];
            double tmp_20689 = ((double *) mem_23561)[i_23099 * (int64_t) 11 + (int64_t) 4];
            double tmp_tan_22093 = ((double *) mem_23562)[i_23099 * (int64_t) 11 + (int64_t) 4];
            double tmp_20692 = ((double *) mem_23561)[i_23099 * (int64_t) 11 + (int64_t) 5];
            double tmp_tan_22094 = ((double *) mem_23562)[i_23099 * (int64_t) 11 + (int64_t) 5];
            double tmp_20695 = ((double *) mem_23561)[i_23099 * (int64_t) 11 + (int64_t) 7];
            double tmp_tan_22095 = ((double *) mem_23562)[i_23099 * (int64_t) 11 + (int64_t) 7];
            double tmp_20698 = ((double *) mem_23561)[i_23099 * (int64_t) 11 + (int64_t) 8];
            double tmp_tan_22096 = ((double *) mem_23562)[i_23099 * (int64_t) 11 + (int64_t) 8];
            double tmp_20701 = ((double *) mem_23561)[i_23099 * (int64_t) 11 + (int64_t) 9];
            double tmp_tan_22097 = ((double *) mem_23562)[i_23099 * (int64_t) 11 + (int64_t) 9];
            double tmp_20704 = ((double *) mem_23561)[i_23099 * (int64_t) 11 + (int64_t) 10];
            double tmp_tan_22098 = ((double *) mem_23562)[i_23099 * (int64_t) 11 + (int64_t) 10];
            
            ((double *) mem_23670)[i_23099] = tmp_20686;
            ((double *) mem_23671)[i_23099] = tmp_tan_22092;
            ((double *) mem_23672)[i_23099] = tmp_20689;
            ((double *) mem_23673)[i_23099] = tmp_tan_22093;
            ((double *) mem_23674)[i_23099] = tmp_20692;
            ((double *) mem_23675)[i_23099] = tmp_tan_22094;
            ((double *) mem_23676)[i_23099] = tmp_20701;
            ((double *) mem_23677)[i_23099] = tmp_tan_22097;
            ((double *) mem_23678)[i_23099] = tmp_20704;
            ((double *) mem_23679)[i_23099] = tmp_tan_22098;
            ((double *) mem_23680)[i_23099] = tmp_20677;
            ((double *) mem_23681)[i_23099] = tmp_tan_22089;
            ((double *) mem_23682)[i_23099] = tmp_20680;
            ((double *) mem_23683)[i_23099] = tmp_tan_22090;
            ((double *) mem_23684)[i_23099] = tmp_20683;
            ((double *) mem_23685)[i_23099] = tmp_tan_22091;
            ((double *) mem_23686)[i_23099] = tmp_20695;
            ((double *) mem_23687)[i_23099] = tmp_tan_22095;
            ((double *) mem_23688)[i_23099] = tmp_20698;
            ((double *) mem_23689)[i_23099] = tmp_tan_22096;
        }
        // benchmark/ba/ba_gradbench_original.fut:48:5-51:49
        for (int64_t i_23124 = 0; i_23124 < (int64_t) 16; i_23124++) {
            // benchmark/ba/ba_gradbench_original.fut:49:44-52
            
            int32_t compute_reproj_err_arg1_20742 = ((int32_t *) mem_23183)[i_23124 * (int64_t) 2 + (int64_t) 1];
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            int64_t compute_reproj_err_arg1_20743 = sext_i32_i64(compute_reproj_err_arg1_20742);
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            bool x_20744 = sle64((int64_t) 0, compute_reproj_err_arg1_20743);
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            bool y_20745 = slt64(compute_reproj_err_arg1_20743, (int64_t) 8);
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            bool bounds_check_20746 = x_20744 && y_20745;
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            bool index_certs_20747;
            
            if (!bounds_check_20746) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) compute_reproj_err_arg1_20743, "] out of bounds for array of shape [", (long long) (int64_t) 8, "].", "-> #0  benchmark/ba/ba_gradbench_original.fut:49:42-53\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:71:5-50\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:179:15-63\n   #3  src/sparse_jacobian_jvp.fut:21:8-28:28\n   #4  src/sparse_jacobian_jvp.fut:21:18-29:18\n   #5  benchmark/ba/bench_jvp_ba_simple.fut:56:8-181:8\n   #6  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // benchmark/ba/ba_gradbench_original.fut:48:47-55
            
            int32_t compute_reproj_err_arg0_20752 = ((int32_t *) mem_23183)[i_23124 * (int64_t) 2];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            int64_t compute_reproj_err_arg0_20753 = sext_i32_i64(compute_reproj_err_arg0_20752);
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            bool x_20754 = sle64((int64_t) 0, compute_reproj_err_arg0_20753);
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            bool y_20755 = slt64(compute_reproj_err_arg0_20753, (int64_t) 2);
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            bool bounds_check_20756 = x_20754 && y_20755;
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            bool index_certs_20757;
            
            if (!bounds_check_20756) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) compute_reproj_err_arg0_20753, "] out of bounds for array of shape [", (long long) (int64_t) 2, "].", "-> #0  benchmark/ba/ba_gradbench_original.fut:48:42-56\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:71:5-50\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:179:15-63\n   #3  src/sparse_jacobian_jvp.fut:21:8-28:28\n   #4  src/sparse_jacobian_jvp.fut:21:18-29:18\n   #5  benchmark/ba/bench_jvp_ba_simple.fut:56:8-181:8\n   #6  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // benchmark/ba/ba_gradbench_original.fut:50:42-46
            
            double compute_reproj_err_arg2_20740 = ((double *) mem_23650)[i_23124];
            
            // benchmark/ba/ba_gradbench_original.fut:50:42-46
            
            double compute_reproj_err_arg2_tan_22109 = ((double *) mem_23651)[i_23124];
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            double lifted_lambda_res_20748 = ((double *) mem_23593)[compute_reproj_err_arg1_20743];
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            double lifted_lambda_res_tan_22120 = ((double *) mem_23594)[compute_reproj_err_arg1_20743];
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            double lifted_lambda_res_20749 = ((double *) mem_23595)[compute_reproj_err_arg1_20743];
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            double lifted_lambda_res_tan_22121 = ((double *) mem_23596)[compute_reproj_err_arg1_20743];
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            double lifted_lambda_res_20750 = ((double *) mem_23597)[compute_reproj_err_arg1_20743];
            
            // benchmark/ba/ba_gradbench_original.fut:49:42-53
            
            double lifted_lambda_res_tan_22122 = ((double *) mem_23598)[compute_reproj_err_arg1_20743];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20758 = ((double *) mem_23670)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_22133 = ((double *) mem_23671)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20759 = ((double *) mem_23672)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_22134 = ((double *) mem_23673)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20760 = ((double *) mem_23674)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_22135 = ((double *) mem_23675)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/bench_jvp_ba_simple.fut:179:15-63
            
            double lifted_lambda_res_20761 = ((double *) mem_23561)[compute_reproj_err_arg0_20753 * (int64_t) 11 + (int64_t) 6];
            
            // benchmark/ba/bench_jvp_ba_simple.fut:179:15-63
            
            double lifted_lambda_res_tan_22136 = ((double *) mem_23562)[compute_reproj_err_arg0_20753 * (int64_t) 11 + (int64_t) 6];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20762 = ((double *) mem_23676)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_22137 = ((double *) mem_23677)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20763 = ((double *) mem_23678)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_22138 = ((double *) mem_23679)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20764 = ((double *) mem_23680)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_22139 = ((double *) mem_23681)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20765 = ((double *) mem_23682)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_22140 = ((double *) mem_23683)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20766 = ((double *) mem_23684)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_22141 = ((double *) mem_23685)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20767 = ((double *) mem_23686)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_22142 = ((double *) mem_23687)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_20768 = ((double *) mem_23688)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/ba_gradbench_original.fut:48:42-56
            
            double lifted_lambda_res_tan_22143 = ((double *) mem_23689)[compute_reproj_err_arg0_20753];
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double zm_res_20769 = lifted_lambda_res_20748 - lifted_lambda_res_20758;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double binop_y_22146 = -1.0 * lifted_lambda_res_tan_22133;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double zm_res_tan_22144 = lifted_lambda_res_tan_22120 + binop_y_22146;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double zm_res_20770 = lifted_lambda_res_20749 - lifted_lambda_res_20759;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double binop_y_22149 = -1.0 * lifted_lambda_res_tan_22134;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double zm_res_tan_22147 = lifted_lambda_res_tan_22121 + binop_y_22149;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double zm_res_20771 = lifted_lambda_res_20750 - lifted_lambda_res_20760;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double binop_y_22152 = -1.0 * lifted_lambda_res_tan_22135;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:146:18-26
            
            double zm_res_tan_22150 = lifted_lambda_res_tan_22122 + binop_y_22152;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:11-18
            
            double zt_res_20772 = lifted_lambda_res_20764 * lifted_lambda_res_20764;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:11-18
            
            double binop_x_22154 = lifted_lambda_res_20764 * lifted_lambda_res_tan_22139;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:11-18
            
            double zt_res_tan_22153 = binop_x_22154 + binop_x_22154;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:21-28
            
            double zt_res_20773 = lifted_lambda_res_20765 * lifted_lambda_res_20765;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:21-28
            
            double binop_x_22157 = lifted_lambda_res_20765 * lifted_lambda_res_tan_22140;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:21-28
            
            double zt_res_tan_22156 = binop_x_22157 + binop_x_22157;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:14-28
            
            double zp_res_20774 = zt_res_20772 + zt_res_20773;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:14-28
            
            double zp_res_tan_22159 = zt_res_tan_22153 + zt_res_tan_22156;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:31-38
            
            double zt_res_20775 = lifted_lambda_res_20766 * lifted_lambda_res_20766;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:31-38
            
            double binop_x_22163 = lifted_lambda_res_20766 * lifted_lambda_res_tan_22141;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:31-38
            
            double zt_res_tan_22162 = binop_x_22163 + binop_x_22163;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:19-38
            
            double zp_res_20776 = zp_res_20774 + zt_res_20775;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:19-38
            
            double zp_res_tan_22165 = zp_res_tan_22159 + zt_res_tan_22162;
            
            // benchmark/ba/ba_gradbench_original.fut:13:5-26:31
            
            bool cond_20777 = zp_res_20776 == 0.0;
            
            // benchmark/ba/ba_gradbench_original.fut:13:5-26:31
            
            bool cond_20778 = !cond_20777;
            
            // benchmark/ba/ba_gradbench_original.fut:13:5-26:31
            
            double rodrigues_rotate_point_res_20779;
            double rodrigues_rotate_point_res_tan_22324;
            double rodrigues_rotate_point_res_20780;
            double rodrigues_rotate_point_res_tan_22325;
            double rodrigues_rotate_point_res_20781;
            double rodrigues_rotate_point_res_tan_22326;
            
            if (cond_20778) {
                // benchmark/ba/ba_gradbench_original.fut:14:21-37
                
                double sqrt_res_22663 = futrts_sqrt64(zp_res_20776);
                double binop_y_22664 = fpow64(zp_res_20776, 0.5);
                double binop_y_22665 = 2.0 * binop_y_22664;
                double binop_y_22666 = 1.0 / binop_y_22665;
                double sqrt_res_tan_22667 = zp_res_tan_22165 * binop_y_22666;
                
                // benchmark/ba/ba_gradbench_original.fut:15:24-37
                
                double cos_res_22668 = futrts_cos64(sqrt_res_22663);
                double binop_y_22669 = futrts_sin64(sqrt_res_22663);
                double binop_y_22670 = 0.0 - binop_y_22669;
                double cos_res_tan_22671 = sqrt_res_tan_22667 * binop_y_22670;
                double sin_res_tan_22672 = sqrt_res_tan_22667 * cos_res_22668;
                
                // benchmark/ba/ba_gradbench_original.fut:17:27-34
                
                double theta_inv_22673 = 1.0 / sqrt_res_22663;
                
                // benchmark/ba/ba_gradbench_original.fut:17:27-34
                
                double binop_x_22674 = 0.0 * theta_inv_22673;
                
                // benchmark/ba/ba_gradbench_original.fut:17:27-34
                
                double binop_y_22675 = sqrt_res_22663 * sqrt_res_22663;
                
                // benchmark/ba/ba_gradbench_original.fut:17:27-34
                
                double binop_y_22676 = 1.0 / binop_y_22675;
                
                // benchmark/ba/ba_gradbench_original.fut:17:27-34
                
                double binop_y_22677 = 0.0 - binop_y_22676;
                
                // benchmark/ba/ba_gradbench_original.fut:17:27-34
                
                double binop_y_22678 = sqrt_res_tan_22667 * binop_y_22677;
                
                // benchmark/ba/ba_gradbench_original.fut:17:27-34
                
                double theta_inv_tan_22679 = binop_x_22674 + binop_y_22678;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22680 = lifted_lambda_res_20764 * theta_inv_22673;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22681 = lifted_lambda_res_tan_22139 * theta_inv_22673;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22682 = lifted_lambda_res_20764 * theta_inv_tan_22679;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22683 = binop_x_22681 + binop_y_22682;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22684 = lifted_lambda_res_20765 * theta_inv_22673;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22685 = lifted_lambda_res_tan_22140 * theta_inv_22673;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22686 = lifted_lambda_res_20765 * theta_inv_tan_22679;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22687 = binop_x_22685 + binop_y_22686;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22688 = lifted_lambda_res_20766 * theta_inv_22673;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22689 = lifted_lambda_res_tan_22141 * theta_inv_22673;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22690 = lifted_lambda_res_20766 * theta_inv_tan_22679;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22691 = binop_x_22689 + binop_y_22690;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double zt_res_22692 = zm_res_20771 * zt_res_22684;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double binop_x_22693 = zm_res_tan_22150 * zt_res_22684;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double binop_y_22694 = zm_res_20771 * zt_res_tan_22687;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double zt_res_tan_22695 = binop_x_22693 + binop_y_22694;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double zt_res_22696 = zm_res_20770 * zt_res_22688;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double binop_x_22697 = zm_res_tan_22147 * zt_res_22688;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double binop_y_22698 = zm_res_20770 * zt_res_tan_22691;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double zt_res_tan_22699 = binop_x_22697 + binop_y_22698;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:16-25
                
                double zm_res_22700 = zt_res_22692 - zt_res_22696;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:16-25
                
                double binop_y_22701 = -1.0 * zt_res_tan_22699;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:16-25
                
                double zm_res_tan_22702 = zt_res_tan_22695 + binop_y_22701;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double zt_res_22703 = zm_res_20769 * zt_res_22688;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double binop_x_22704 = zm_res_tan_22144 * zt_res_22688;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double binop_y_22705 = zm_res_20769 * zt_res_tan_22691;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double zt_res_tan_22706 = binop_x_22704 + binop_y_22705;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double zt_res_22707 = zm_res_20771 * zt_res_22680;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double binop_x_22708 = zm_res_tan_22150 * zt_res_22680;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double binop_y_22709 = zm_res_20771 * zt_res_tan_22683;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double zt_res_tan_22710 = binop_x_22708 + binop_y_22709;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:31-40
                
                double zm_res_22711 = zt_res_22703 - zt_res_22707;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:31-40
                
                double binop_y_22712 = -1.0 * zt_res_tan_22710;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:31-40
                
                double zm_res_tan_22713 = zt_res_tan_22706 + binop_y_22712;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double zt_res_22714 = zm_res_20770 * zt_res_22680;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double binop_x_22715 = zm_res_tan_22147 * zt_res_22680;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double binop_y_22716 = zm_res_20770 * zt_res_tan_22683;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double zt_res_tan_22717 = binop_x_22715 + binop_y_22716;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double zt_res_22718 = zm_res_20769 * zt_res_22684;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double binop_x_22719 = zm_res_tan_22144 * zt_res_22684;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double binop_y_22720 = zm_res_20769 * zt_res_tan_22687;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double zt_res_tan_22721 = binop_x_22719 + binop_y_22720;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:46-55
                
                double zm_res_22722 = zt_res_22714 - zt_res_22718;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:46-55
                
                double binop_y_22723 = -1.0 * zt_res_tan_22721;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:46-55
                
                double zm_res_tan_22724 = zt_res_tan_22717 + binop_y_22723;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:11-18
                
                double zt_res_22725 = zm_res_20769 * zt_res_22680;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:11-18
                
                double binop_x_22726 = zm_res_tan_22144 * zt_res_22680;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:11-18
                
                double binop_y_22727 = zm_res_20769 * zt_res_tan_22683;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:11-18
                
                double zt_res_tan_22728 = binop_x_22726 + binop_y_22727;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:21-28
                
                double zt_res_22729 = zm_res_20770 * zt_res_22684;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:21-28
                
                double binop_x_22730 = zm_res_tan_22147 * zt_res_22684;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:21-28
                
                double binop_y_22731 = zm_res_20770 * zt_res_tan_22687;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:21-28
                
                double zt_res_tan_22732 = binop_x_22730 + binop_y_22731;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:14-28
                
                double zp_res_22733 = zt_res_22725 + zt_res_22729;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:14-28
                
                double zp_res_tan_22734 = zt_res_tan_22728 + zt_res_tan_22732;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:31-38
                
                double zt_res_22735 = zm_res_20771 * zt_res_22688;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:31-38
                
                double binop_x_22736 = zm_res_tan_22150 * zt_res_22688;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:31-38
                
                double binop_y_22737 = zm_res_20771 * zt_res_tan_22691;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:31-38
                
                double zt_res_tan_22738 = binop_x_22736 + binop_y_22737;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:19-38
                
                double zp_res_22739 = zp_res_22733 + zt_res_22735;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:151:19-38
                
                double zp_res_tan_22740 = zp_res_tan_22734 + zt_res_tan_22738;
                
                // benchmark/ba/ba_gradbench_original.fut:21:36-46
                
                double zt_rhs_22741 = 1.0 - cos_res_22668;
                
                // benchmark/ba/ba_gradbench_original.fut:21:36-46
                
                double binop_y_22742 = -1.0 * cos_res_tan_22671;
                
                // benchmark/ba/ba_gradbench_original.fut:21:31-46
                
                double tmp_22743 = zp_res_22739 * zt_rhs_22741;
                
                // benchmark/ba/ba_gradbench_original.fut:21:31-46
                
                double binop_x_22744 = zp_res_tan_22740 * zt_rhs_22741;
                
                // benchmark/ba/ba_gradbench_original.fut:21:31-46
                
                double binop_y_22745 = zp_res_22739 * binop_y_22742;
                
                // benchmark/ba/ba_gradbench_original.fut:21:31-46
                
                double tmp_tan_22746 = binop_x_22744 + binop_y_22745;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22747 = zm_res_20769 * cos_res_22668;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22748 = zm_res_tan_22144 * cos_res_22668;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22749 = zm_res_20769 * cos_res_tan_22671;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22750 = binop_x_22748 + binop_y_22749;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22751 = zm_res_20770 * cos_res_22668;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22752 = zm_res_tan_22147 * cos_res_22668;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22753 = zm_res_20770 * cos_res_tan_22671;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22754 = binop_x_22752 + binop_y_22753;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22755 = zm_res_20771 * cos_res_22668;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22756 = zm_res_tan_22150 * cos_res_22668;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22757 = zm_res_20771 * cos_res_tan_22671;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22758 = binop_x_22756 + binop_y_22757;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22759 = binop_y_22669 * zm_res_22700;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22760 = sin_res_tan_22672 * zm_res_22700;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22761 = binop_y_22669 * zm_res_tan_22702;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22762 = binop_x_22760 + binop_y_22761;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22763 = binop_y_22669 * zm_res_22711;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22764 = sin_res_tan_22672 * zm_res_22711;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22765 = binop_y_22669 * zm_res_tan_22713;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22766 = binop_x_22764 + binop_y_22765;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22767 = binop_y_22669 * zm_res_22722;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22768 = sin_res_tan_22672 * zm_res_22722;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22769 = binop_y_22669 * zm_res_tan_22724;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22770 = binop_x_22768 + binop_y_22769;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_22771 = zt_res_22747 + zt_res_22759;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_22772 = zt_res_tan_22750 + zt_res_tan_22762;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_22773 = zt_res_22751 + zt_res_22763;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_22774 = zt_res_tan_22754 + zt_res_tan_22766;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_22775 = zt_res_22755 + zt_res_22767;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_22776 = zt_res_tan_22758 + zt_res_tan_22770;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22777 = zt_res_22680 * tmp_22743;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22778 = zt_res_tan_22683 * tmp_22743;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22779 = zt_res_22680 * tmp_tan_22746;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22780 = binop_x_22778 + binop_y_22779;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22781 = zt_res_22684 * tmp_22743;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22782 = zt_res_tan_22687 * tmp_22743;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22783 = zt_res_22684 * tmp_tan_22746;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22784 = binop_x_22782 + binop_y_22783;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_22785 = zt_res_22688 * tmp_22743;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_x_22786 = zt_res_tan_22691 * tmp_22743;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double binop_y_22787 = zt_res_22688 * tmp_tan_22746;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:159:29-39
                
                double zt_res_tan_22788 = binop_x_22786 + binop_y_22787;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_22789 = zp_res_22771 + zt_res_22777;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_22790 = zp_res_tan_22772 + zt_res_tan_22780;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_22791 = zp_res_22773 + zt_res_22781;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_22792 = zp_res_tan_22774 + zt_res_tan_22784;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_22793 = zp_res_22775 + zt_res_22785;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_22794 = zp_res_tan_22776 + zt_res_tan_22788;
                
                rodrigues_rotate_point_res_20779 = zp_res_22789;
                rodrigues_rotate_point_res_tan_22324 = zp_res_tan_22790;
                rodrigues_rotate_point_res_20780 = zp_res_22791;
                rodrigues_rotate_point_res_tan_22325 = zp_res_tan_22792;
                rodrigues_rotate_point_res_20781 = zp_res_22793;
                rodrigues_rotate_point_res_tan_22326 = zp_res_tan_22794;
            } else {
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double zt_res_20820 = lifted_lambda_res_20765 * zm_res_20771;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double binop_x_22289 = zm_res_20771 * lifted_lambda_res_tan_22140;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double binop_y_22290 = lifted_lambda_res_20765 * zm_res_tan_22150;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:14-19
                
                double zt_res_tan_22288 = binop_x_22289 + binop_y_22290;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double zt_res_20821 = lifted_lambda_res_20766 * zm_res_20770;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double binop_x_22292 = zm_res_20770 * lifted_lambda_res_tan_22141;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double binop_y_22293 = lifted_lambda_res_20766 * zm_res_tan_22147;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:20-25
                
                double zt_res_tan_22291 = binop_x_22292 + binop_y_22293;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:16-25
                
                double zm_res_20822 = zt_res_20820 - zt_res_20821;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:16-25
                
                double binop_y_22296 = -1.0 * zt_res_tan_22291;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:16-25
                
                double zm_res_tan_22294 = zt_res_tan_22288 + binop_y_22296;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double zt_res_20823 = lifted_lambda_res_20766 * zm_res_20769;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double binop_x_22298 = zm_res_20769 * lifted_lambda_res_tan_22141;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double binop_y_22299 = lifted_lambda_res_20766 * zm_res_tan_22144;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:29-34
                
                double zt_res_tan_22297 = binop_x_22298 + binop_y_22299;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double zt_res_20824 = lifted_lambda_res_20764 * zm_res_20771;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double binop_x_22301 = zm_res_20771 * lifted_lambda_res_tan_22139;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double binop_y_22302 = lifted_lambda_res_20764 * zm_res_tan_22150;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:35-40
                
                double zt_res_tan_22300 = binop_x_22301 + binop_y_22302;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:31-40
                
                double zm_res_20825 = zt_res_20823 - zt_res_20824;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:31-40
                
                double binop_y_22305 = -1.0 * zt_res_tan_22300;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:31-40
                
                double zm_res_tan_22303 = zt_res_tan_22297 + binop_y_22305;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double zt_res_20826 = lifted_lambda_res_20764 * zm_res_20770;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double binop_x_22307 = zm_res_20770 * lifted_lambda_res_tan_22139;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double binop_y_22308 = lifted_lambda_res_20764 * zm_res_tan_22147;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:44-49
                
                double zt_res_tan_22306 = binop_x_22307 + binop_y_22308;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double zt_res_20827 = lifted_lambda_res_20765 * zm_res_20769;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double binop_x_22310 = zm_res_20769 * lifted_lambda_res_tan_22140;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double binop_y_22311 = lifted_lambda_res_20765 * zm_res_tan_22144;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:50-55
                
                double zt_res_tan_22309 = binop_x_22310 + binop_y_22311;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:46-55
                
                double zm_res_20828 = zt_res_20826 - zt_res_20827;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:46-55
                
                double binop_y_22314 = -1.0 * zt_res_tan_22309;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:155:46-55
                
                double zm_res_tan_22312 = zt_res_tan_22306 + binop_y_22314;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_20829 = zm_res_20769 + zm_res_20822;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_22315 = zm_res_tan_22144 + zm_res_tan_22294;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_20830 = zm_res_20770 + zm_res_20825;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_22318 = zm_res_tan_22147 + zm_res_tan_22303;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_20831 = zm_res_20771 + zm_res_20828;
                
                // benchmark/ba/lib/github.com/athas/vector/vspace.fut:145:18-26
                
                double zp_res_tan_22321 = zm_res_tan_22150 + zm_res_tan_22312;
                
                rodrigues_rotate_point_res_20779 = zp_res_20829;
                rodrigues_rotate_point_res_tan_22324 = zp_res_tan_22315;
                rodrigues_rotate_point_res_20780 = zp_res_20830;
                rodrigues_rotate_point_res_tan_22325 = zp_res_tan_22318;
                rodrigues_rotate_point_res_20781 = zp_res_20831;
                rodrigues_rotate_point_res_tan_22326 = zp_res_tan_22321;
            }
            // benchmark/ba/ba_gradbench_original.fut:37:55-62
            
            double scale_arg0_20832 = 1.0 / rodrigues_rotate_point_res_20781;
            
            // benchmark/ba/ba_gradbench_original.fut:37:55-62
            
            double binop_x_22329 = 0.0 * scale_arg0_20832;
            
            // benchmark/ba/ba_gradbench_original.fut:37:55-62
            
            double binop_y_22330 = rodrigues_rotate_point_res_20781 * rodrigues_rotate_point_res_20781;
            
            // benchmark/ba/ba_gradbench_original.fut:37:55-62
            
            double binop_y_22331 = 1.0 / binop_y_22330;
            
            // benchmark/ba/ba_gradbench_original.fut:37:55-62
            
            double binop_y_22332 = 0.0 - binop_y_22331;
            
            // benchmark/ba/ba_gradbench_original.fut:37:55-62
            
            double binop_y_22333 = rodrigues_rotate_point_res_tan_22326 * binop_y_22332;
            
            // benchmark/ba/ba_gradbench_original.fut:37:55-62
            
            double scale_arg0_tan_22327 = binop_x_22329 + binop_y_22333;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_20833 = rodrigues_rotate_point_res_20779 * scale_arg0_20832;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_22335 = scale_arg0_20832 * rodrigues_rotate_point_res_tan_22324;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_22336 = rodrigues_rotate_point_res_20779 * scale_arg0_tan_22327;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_22334 = binop_x_22335 + binop_y_22336;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_20834 = rodrigues_rotate_point_res_20780 * scale_arg0_20832;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_22338 = scale_arg0_20832 * rodrigues_rotate_point_res_tan_22325;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_22339 = rodrigues_rotate_point_res_20780 * scale_arg0_tan_22327;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_22337 = binop_x_22338 + binop_y_22339;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:11-18
            
            double zt_res_20835 = zt_res_20833 * zt_res_20833;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:11-18
            
            double binop_x_22341 = zt_res_20833 * zt_res_tan_22334;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:11-18
            
            double zt_res_tan_22340 = binop_x_22341 + binop_x_22341;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:21-28
            
            double zt_res_20836 = zt_res_20834 * zt_res_20834;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:21-28
            
            double binop_x_22344 = zt_res_20834 * zt_res_tan_22337;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:21-28
            
            double zt_res_tan_22343 = binop_x_22344 + binop_x_22344;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:14-28
            
            double zp_res_20837 = zt_res_20835 + zt_res_20836;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:93:14-28
            
            double zp_res_tan_22346 = zt_res_tan_22340 + zt_res_tan_22343;
            
            // benchmark/ba/ba_gradbench_original.fut:30:28-33
            
            double zp_rhs_20838 = lifted_lambda_res_20762 * zp_res_20837;
            
            // benchmark/ba/ba_gradbench_original.fut:30:28-33
            
            double binop_x_22350 = zp_res_20837 * lifted_lambda_res_tan_22137;
            
            // benchmark/ba/ba_gradbench_original.fut:30:28-33
            
            double binop_y_22351 = lifted_lambda_res_20762 * zp_res_tan_22346;
            
            // benchmark/ba/ba_gradbench_original.fut:30:28-33
            
            double zp_rhs_tan_22349 = binop_x_22350 + binop_y_22351;
            
            // benchmark/ba/ba_gradbench_original.fut:30:13-33
            
            double zp_lhs_20839 = 1.0 + zp_rhs_20838;
            
            // benchmark/ba/ba_gradbench_original.fut:30:49-54
            
            double zt_lhs_20840 = lifted_lambda_res_20763 * zp_res_20837;
            
            // benchmark/ba/ba_gradbench_original.fut:30:49-54
            
            double binop_x_22356 = zp_res_20837 * lifted_lambda_res_tan_22138;
            
            // benchmark/ba/ba_gradbench_original.fut:30:49-54
            
            double binop_y_22357 = lifted_lambda_res_20763 * zp_res_tan_22346;
            
            // benchmark/ba/ba_gradbench_original.fut:30:49-54
            
            double zt_lhs_tan_22355 = binop_x_22356 + binop_y_22357;
            
            // benchmark/ba/ba_gradbench_original.fut:30:55-60
            
            double zp_rhs_20841 = zp_res_20837 * zt_lhs_20840;
            
            // benchmark/ba/ba_gradbench_original.fut:30:55-60
            
            double binop_x_22359 = zt_lhs_20840 * zp_res_tan_22346;
            
            // benchmark/ba/ba_gradbench_original.fut:30:55-60
            
            double binop_y_22360 = zp_res_20837 * zt_lhs_tan_22355;
            
            // benchmark/ba/ba_gradbench_original.fut:30:55-60
            
            double zp_rhs_tan_22358 = binop_x_22359 + binop_y_22360;
            
            // benchmark/ba/ba_gradbench_original.fut:30:34-60
            
            double L_20842 = zp_lhs_20839 + zp_rhs_20841;
            
            // benchmark/ba/ba_gradbench_original.fut:30:34-60
            
            double L_tan_22361 = zp_rhs_tan_22349 + zp_rhs_tan_22358;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_20843 = zt_res_20833 * L_20842;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_22365 = L_20842 * zt_res_tan_22334;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_22366 = zt_res_20833 * L_tan_22361;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_22364 = binop_x_22365 + binop_y_22366;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_20844 = zt_res_20834 * L_20842;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_22368 = L_20842 * zt_res_tan_22337;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_22369 = zt_res_20834 * L_tan_22361;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_22367 = binop_x_22368 + binop_y_22369;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_20845 = lifted_lambda_res_20761 * zt_res_20843;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_22371 = zt_res_20843 * lifted_lambda_res_tan_22136;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_22372 = lifted_lambda_res_20761 * zt_res_tan_22364;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_22370 = binop_x_22371 + binop_y_22372;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_20846 = lifted_lambda_res_20761 * zt_res_20844;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_22374 = zt_res_20844 * lifted_lambda_res_tan_22136;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_22375 = lifted_lambda_res_20761 * zt_res_tan_22367;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_22373 = binop_x_22374 + binop_y_22375;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:87:18-26
            
            double zp_res_20847 = lifted_lambda_res_20767 + zt_res_20845;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:87:18-26
            
            double zp_res_tan_22376 = lifted_lambda_res_tan_22142 + zt_res_tan_22370;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:87:18-26
            
            double zp_res_20848 = lifted_lambda_res_20768 + zt_res_20846;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:87:18-26
            
            double zp_res_tan_22379 = lifted_lambda_res_tan_22143 + zt_res_tan_22373;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:88:18-26
            
            double zm_res_20849 = zp_res_20847 - tmp_20443;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:88:18-26
            
            double zm_res_20850 = zp_res_20848 - tmp_20446;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_22391 = zm_res_20849 * compute_reproj_err_arg2_tan_22109;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_22392 = compute_reproj_err_arg2_20740 * zp_res_tan_22376;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_22390 = binop_x_22391 + binop_y_22392;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_x_22394 = zm_res_20850 * compute_reproj_err_arg2_tan_22109;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double binop_y_22395 = compute_reproj_err_arg2_20740 * zp_res_tan_22379;
            
            // benchmark/ba/lib/github.com/athas/vector/vspace.fut:97:29-39
            
            double zt_res_tan_22393 = binop_x_22394 + binop_y_22395;
            
            ((double *) mem_23810)[i_23124] = zt_res_tan_22390;
            ((double *) mem_23811)[i_23124] = zt_res_tan_22393;
        }
        // benchmark/ba/bench_jvp_ba_simple.fut:73:6-79:35
        for (int64_t i_23129 = 0; i_23129 < (int64_t) 48; i_23129++) {
            // benchmark/ba/bench_jvp_ba_simple.fut:74:8-79:35
            
            bool cond_20864 = slt64(i_23129, (int64_t) 32);
            double lifted_lambda_res_tan_22428;
            
            if (cond_20864) {
                // benchmark/ba/bench_jvp_ba_simple.fut:75:21-27
                
                int64_t ob_22795 = sdiv64(i_23129, (int64_t) 2);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:76:18-32
                
                bool x_22796 = sle64((int64_t) 0, ob_22795);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:76:18-32
                
                bool y_22797 = slt64(ob_22795, (int64_t) 16);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:76:18-32
                
                bool bounds_check_22798 = x_22796 && y_22797;
                
                // benchmark/ba/bench_jvp_ba_simple.fut:76:18-32
                
                bool index_certs_22799;
                
                if (!bounds_check_22798) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) ob_22795, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:76:18-32\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:179:15-63\n   #2  src/sparse_jacobian_jvp.fut:21:8-28:28\n   #3  src/sparse_jacobian_jvp.fut:21:18-29:18\n   #4  benchmark/ba/bench_jvp_ba_simple.fut:56:8-181:8\n   #5  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // benchmark/ba/bench_jvp_ba_simple.fut:76:18-32
                
                double lifted_lambda_res_t_res_tan_22800 = ((double *) mem_23810)[ob_22795];
                
                // benchmark/ba/bench_jvp_ba_simple.fut:76:18-32
                
                double lifted_lambda_res_t_res_tan_22801 = ((double *) mem_23811)[ob_22795];
                
                // benchmark/ba/bench_jvp_ba_simple.fut:77:18-24
                
                int64_t zeze_lhs_22802 = smod64(i_23129, (int64_t) 2);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:77:13-50
                
                bool cond_22803 = zeze_lhs_22802 == (int64_t) 0;
                double lifted_lambda_res_t_res_tan_22804;
                
                if (cond_22803) {
                    lifted_lambda_res_t_res_tan_22804 = lifted_lambda_res_t_res_tan_22800;
                } else {
                    lifted_lambda_res_t_res_tan_22804 = lifted_lambda_res_t_res_tan_22801;
                }
                lifted_lambda_res_tan_22428 = lifted_lambda_res_t_res_tan_22804;
            } else {
                // benchmark/ba/bench_jvp_ba_simple.fut:79:23-34
                
                int64_t tmp_20877 = sub64(i_23129, (int64_t) 32);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:79:10-35
                
                bool x_20878 = sle64((int64_t) 0, tmp_20877);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:79:10-35
                
                bool y_20879 = slt64(tmp_20877, (int64_t) 16);
                
                // benchmark/ba/bench_jvp_ba_simple.fut:79:10-35
                
                bool bounds_check_20880 = x_20878 && y_20879;
                
                // benchmark/ba/bench_jvp_ba_simple.fut:79:10-35
                
                bool index_certs_20881;
                
                if (!bounds_check_20880) {
                    set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_20877, "] out of bounds for array of shape [", (long long) (int64_t) 16, "].", "-> #0  benchmark/ba/bench_jvp_ba_simple.fut:79:10-35\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:179:15-63\n   #2  src/sparse_jacobian_jvp.fut:21:8-28:28\n   #3  src/sparse_jacobian_jvp.fut:21:18-29:18\n   #4  benchmark/ba/bench_jvp_ba_simple.fut:56:8-181:8\n   #5  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                    err = FUTHARK_PROGRAM_ERROR;
                    goto cleanup;
                }
                // benchmark/ba/bench_jvp_ba_simple.fut:79:10-35
                
                double lifted_lambda_res_f_res_tan_22427 = ((double *) mem_23649)[tmp_20877];
                
                lifted_lambda_res_tan_22428 = lifted_lambda_res_f_res_tan_22427;
            }
            ((double *) mem_23824)[i_23129] = lifted_lambda_res_tan_22428;
        }
        lmad_copy_8b(ctx, 1, (uint64_t *) mem_23549, i_23133 * (int64_t) 48, (int64_t []) {(int64_t) 1}, (uint64_t *) mem_23824, (int64_t) 0, (int64_t []) {(int64_t) 1}, (int64_t []) {(int64_t) 48});
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    if (mem_23836_cached_sizze_24065 < bytes_23505) {
        err = lexical_realloc(ctx, &mem_23836, &mem_23836_cached_sizze_24065, bytes_23505);
        if (err != FUTHARK_SUCCESS)
            goto cleanup;
    }
    // src/sparse_jacobian_jvp.fut:42:15-35
    for (int64_t nest_i_23963 = 0; nest_i_23963 < csr_bipartite_from_pattern_res_19997; nest_i_23963++) {
        ((double *) mem_23836)[nest_i_23963] = 0.0;
    }
    // src/sparse_jacobian_jvp.fut:45:5-52:27
    
    bool compressed_to_csr_vals_res_20194;
    int64_t compressed_to_csr_vals_res_20196;
    bool loop_while_20197;
    int64_t i_20199;
    
    loop_while_20197 = 1;
    i_20199 = (int64_t) 0;
    while (loop_while_20197) {
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool x_20200 = sle64((int64_t) 0, i_20199);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool y_20201 = slt64(i_20199, csr_bipartite_from_pattern_res_19996);
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool bounds_check_20202 = x_20200 && y_20201;
        
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        bool index_certs_20203;
        
        if (!bounds_check_20202) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_20199, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_19996, "].", "-> #0  src/sparse_jacobian_jvp.fut:47:15-26\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:166:14-183:62\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:47:15-26
        
        int64_t s_20204 = ((int64_t *) ext_mem_23220.mem)[i_20199];
        
        // src/sparse_jacobian_jvp.fut:48:25-27
        
        int64_t e_20205 = add64((int64_t) 1, i_20199);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool x_20206 = sle64((int64_t) 0, e_20205);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool y_20207 = slt64(e_20205, csr_bipartite_from_pattern_res_19996);
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool bounds_check_20208 = x_20206 && y_20207;
        
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        bool index_certs_20209;
        
        if (!bounds_check_20208) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) e_20205, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_19996, "].", "-> #0  src/sparse_jacobian_jvp.fut:48:15-28\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:166:14-183:62\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:48:15-28
        
        int64_t e_20210 = ((int64_t *) ext_mem_23220.mem)[e_20205];
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t j_m_i_20211 = sub64(e_20210, s_20204);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool empty_slice_20212 = j_m_i_20211 == (int64_t) 0;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t m_20213 = sub64(j_m_i_20211, (int64_t) 1);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        int64_t i_p_m_t_s_20214 = add64(s_20204, m_20213);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_leq_i_p_m_t_s_20215 = sle64((int64_t) 0, i_p_m_t_s_20214);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_p_m_t_s_leq_w_20216 = slt64(i_p_m_t_s_20214, csr_bipartite_from_pattern_res_19997);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool zzero_lte_i_20217 = sle64((int64_t) 0, s_20204);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool i_lte_j_20218 = sle64(s_20204, e_20210);
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_20219 = i_p_m_t_s_leq_w_20216 && zzero_lte_i_20217;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool y_20220 = zzero_leq_i_p_m_t_s_20215 && y_20219;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool forwards_ok_20221 = i_lte_j_20218 && y_20220;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool ok_or_empty_20222 = empty_slice_20212 || forwards_ok_20221;
        
        // src/sparse_jacobian_jvp.fut:49:18-30
        
        bool index_certs_20223;
        
        if (!ok_or_empty_20222) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) s_20204, ":", (long long) e_20210, "] out of bounds for array of shape [", (long long) csr_bipartite_from_pattern_res_19997, "].", "-> #0  src/sparse_jacobian_jvp.fut:49:18-30\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:166:14-183:62\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool y_20225 = slt64(i_20199, (int64_t) 48);
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool bounds_check_20226 = x_20200 && y_20225;
        
        // src/sparse_jacobian_jvp.fut:50:29-45
        
        bool index_certs_20227;
        
        if (!bounds_check_20226) {
            set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) i_20199, "] out of bounds for array of shape [", (long long) (int64_t) 48, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-45\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:166:14-183:62\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
            err = FUTHARK_PROGRAM_ERROR;
            goto cleanup;
        }
        // src/sparse_jacobian_jvp.fut:50:18-51
        for (int64_t i_23137 = 0; i_23137 < j_m_i_20211; i_23137++) {
            int64_t index_primexp_23141 = s_20204 + i_23137;
            int64_t eta_p_20229 = ((int64_t *) ext_mem_23219.mem)[index_primexp_23141];
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool x_20230 = sle64((int64_t) 0, eta_p_20229);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool y_20231 = slt64(eta_p_20229, (int64_t) 62);
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool bounds_check_20232 = x_20230 && y_20231;
            
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            bool index_certs_20233;
            
            if (!bounds_check_20232) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) eta_p_20229, "] out of bounds for array of shape [", (long long) (int64_t) 62, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:32-41\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:166:14-183:62\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:32-41
            
            int64_t tmp_20234 = ((int64_t *) mem_23521)[eta_p_20229];
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool x_20235 = sle64((int64_t) 0, tmp_20234);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool y_20236 = slt64(tmp_20234, num_colors_of_res_f_res_20173);
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool bounds_check_20237 = x_20235 && y_20236;
            
            // src/sparse_jacobian_jvp.fut:50:29-42
            
            bool index_certs_20238;
            
            if (!bounds_check_20237) {
                set_error(ctx, msgprintf("Error: %s%lld%s%lld%s\n\nBacktrace:\n%s", "Index [", (long long) tmp_20234, "] out of bounds for array of shape [", (long long) num_colors_of_res_f_res_20173, "].", "-> #0  src/sparse_jacobian_jvp.fut:50:29-42\n   #1  benchmark/ba/bench_jvp_ba_simple.fut:166:14-183:62\n   #2  benchmark/ba/bench_jvp_ba_simple.fut:193:30-199:81\n"));
                err = FUTHARK_PROGRAM_ERROR;
                goto cleanup;
            }
            // src/sparse_jacobian_jvp.fut:50:18-51
            
            double lifted_lambda_res_20239 = ((double *) mem_23549)[tmp_20234 * (int64_t) 48 + i_20199];
            
            ((double *) mem_23836)[s_20204 + i_23137] = lifted_lambda_res_20239;
        }
        // src/sparse_jacobian_jvp.fut:46:13-16
        
        bool loop_cond_20241 = slt64(e_20205, (int64_t) 48);
        bool loop_while_tmp_23964 = loop_cond_20241;
        int64_t i_tmp_23966 = e_20205;
        
        loop_while_20197 = loop_while_tmp_23964;
        i_20199 = i_tmp_23966;
    }
    compressed_to_csr_vals_res_20194 = loop_while_20197;
    compressed_to_csr_vals_res_20196 = i_20199;
    if (memblock_unref(ctx, &ext_mem_23219, "ext_mem_23219") != 0)
        return 1;
    if (memblock_unref(ctx, &ext_mem_23220, "ext_mem_23220") != 0)
        return 1;
    // benchmark/ba/bench_jvp_ba_simple.fut:8:3-62
    
    bool defunc_0_reduce_res_22856;
    bool redout_23139 = 1;
    
    for (int64_t i_23140 = 0; i_23140 < csr_bipartite_from_pattern_res_19997; i_23140++) {
        double eta_p_20884 = ((double *) mem_23506)[i_23140];
        double eta_p_20885 = ((double *) mem_23836)[i_23140];
        
        // benchmark/ba/bench_jvp_ba_simple.fut:8:46-49
        
        double abs_arg0_20886 = eta_p_20884 - eta_p_20885;
        
        // benchmark/ba/bench_jvp_ba_simple.fut:8:35-49
        
        double abs_res_20887 = fabs64(abs_arg0_20886);
        
        // benchmark/ba/bench_jvp_ba_simple.fut:8:51-57
        
        bool lifted_lambda_res_20888 = abs_res_20887 <= 1.0e-7;
        
        // benchmark/ba/bench_jvp_ba_simple.fut:8:3-62
        
        bool x_20255 = lifted_lambda_res_20888 && redout_23139;
        bool redout_tmp_23968 = x_20255;
        
        redout_23139 = redout_tmp_23968;
    }
    defunc_0_reduce_res_22856 = redout_23139;
    prim_out_23856 = defunc_0_reduce_res_22856;
    *out_prim_out_23976 = prim_out_23856;
    
  cleanup:
    {
        free(mem_23183);
        free(mem_23193);
        free(mem_23194);
        free(mem_23201);
        free(mem_23206);
        free(mem_23225);
        free(mem_23230);
        free(mem_23231);
        free(mem_23232);
        free(mem_23241);
        free(mem_23242);
        free(mem_23263);
        free(mem_23264);
        free(mem_23265);
        free(mem_23266);
        free(mem_23267);
        free(mem_23268);
        free(mem_23287);
        free(mem_23288);
        free(mem_23319);
        free(mem_23320);
        free(mem_23321);
        free(mem_23340);
        free(mem_23341);
        free(mem_23342);
        free(mem_23343);
        free(mem_23344);
        free(mem_23345);
        free(mem_23346);
        free(mem_23347);
        free(mem_23348);
        free(mem_23349);
        free(mem_23350);
        free(mem_23351);
        free(mem_23352);
        free(mem_23353);
        free(mem_23354);
        free(mem_23355);
        free(mem_23356);
        free(mem_23357);
        free(mem_23358);
        free(mem_23359);
        free(mem_23480);
        free(mem_23481);
        free(mem_23494);
        free(mem_23506);
        free(mem_23521);
        free(mem_23522);
        free(mem_23549);
        free(mem_23554);
        free(mem_23561);
        free(mem_23562);
        free(mem_23571);
        free(mem_23572);
        free(mem_23593);
        free(mem_23594);
        free(mem_23595);
        free(mem_23596);
        free(mem_23597);
        free(mem_23598);
        free(mem_23617);
        free(mem_23618);
        free(mem_23649);
        free(mem_23650);
        free(mem_23651);
        free(mem_23670);
        free(mem_23671);
        free(mem_23672);
        free(mem_23673);
        free(mem_23674);
        free(mem_23675);
        free(mem_23676);
        free(mem_23677);
        free(mem_23678);
        free(mem_23679);
        free(mem_23680);
        free(mem_23681);
        free(mem_23682);
        free(mem_23683);
        free(mem_23684);
        free(mem_23685);
        free(mem_23686);
        free(mem_23687);
        free(mem_23688);
        free(mem_23689);
        free(mem_23810);
        free(mem_23811);
        free(mem_23824);
        free(mem_23836);
        if (memblock_unref(ctx, &ext_mem_23223, "ext_mem_23223") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_23224, "ext_mem_23224") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_23221, "mem_23221") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_23219, "ext_mem_23219") != 0)
            return 1;
        if (memblock_unref(ctx, &ext_mem_23220, "ext_mem_23220") != 0)
            return 1;
        if (memblock_unref(ctx, &mem_23217, "mem_23217") != 0)
            return 1;
    }
    return err;
}

int futhark_entry_test_ba_d2_matches_dense_tiny(struct futhark_context *ctx, bool *out0, const int32_t in0)
{
    int32_t _dummy_18303 = 0;
    bool prim_out_23856 = 0;
    int ret = 0;
    
    lock_lock(&ctx->lock);
    _dummy_18303 = in0;
    if (ret == 0) {
        ret = futrts_entry_test_ba_d2_matches_dense_tiny(ctx, &prim_out_23856, _dummy_18303);
        if (ret == 0) {
            *out0 = prim_out_23856;
        }
    }
    lock_unlock(&ctx->lock);
    return ret;
}
  
