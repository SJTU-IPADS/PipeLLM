#include <dlfcn.h>
#include <openssl/ssl.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define HOOK_C_API extern "C"
#define HOOK_DECL_EXPORT __attribute__((visibility("default")))

static inline void *get_symbol_cuda(const char *name)
{
    auto handle = dlopen("/usr/local/cuda/lib64/libcudart.so.12", RTLD_NOW | RTLD_LOCAL);
    return dlsym(handle, name);
}

static inline cudaError_t real_cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                                        enum cudaMemcpyKind kind, cudaStream_t stream)
{
    using func_ptr = cudaError_t (*)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_cuda("cudaMemcpyAsync"));
    return func_entry(dst, src, count, kind, stream);
}

static inline cudaError_t real_cudaMemcpy(void *dst, const void *src, size_t count,
                                                        enum cudaMemcpyKind kind)
{
    using func_ptr = cudaError_t (*)(void *, const void *, size_t, enum cudaMemcpyKind);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_cuda("cudaMemcpy"));
    return func_entry(dst, src, count, kind);
}

static inline cudaError_t real_cudaGetLastError()
{
    using func_ptr = cudaError_t (*)(void);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_cuda("cudaGetLastError"));
    return func_entry();
}

static inline cudaError_t real_cudaMalloc(void **devPtr, size_t size)
{
    using func_ptr = cudaError_t (*)(void **devPtr, size_t size);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_cuda("cudaMalloc"));
    return func_entry(devPtr, size);
}

static inline cudaError_t real_cudaMallocHost(void **devPtr, size_t size)
{
    using func_ptr = cudaError_t (*)(void **devPtr, size_t size);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_cuda("cudaMallocHost"));
    return func_entry(devPtr, size);
}

static inline cudaError_t real_cudaFree(void *devPtr)
{
    using func_ptr = cudaError_t (*)(void *devPtr);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_cuda("cudaFree"));
    return func_entry(devPtr);
}

static inline cudaError_t real_cudaFreeHost(void *ptr)
{
    using func_ptr = cudaError_t (*)(void *ptr);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_cuda("cudaFreeHost"));
    return func_entry(ptr);
}

static inline cudaError_t real_cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream)
{
    using func_ptr = cudaError_t (*)(void *, int, size_t, cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_cuda("cudaMemsetAsync"));
    return func_entry(devPtr, value, count, stream);
}

static inline cudaError_t real_cudaStreamSynchronize(cudaStream_t stream)
{
    using func_ptr = cudaError_t (*)(cudaStream_t);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_cuda("cudaStreamSynchronize"));
    return func_entry(stream);
}

static inline cudaError_t real_cudaStreamCreate(cudaStream_t* pstream)
{
    using func_ptr = cudaError_t (*)(cudaStream_t*);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_cuda("cudaStreamCreate"));
    return func_entry(pstream);
}

static inline cudaError_t real_cudaStreamCreateWithFlags(cudaStream_t* pstream, unsigned int flags)
{
    using func_ptr = cudaError_t (*)(cudaStream_t*, unsigned int);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_cuda("cudaStreamCreateWithFlags"));
    return func_entry(pstream, flags);
}

static inline cudaError_t real_cudaDeviceSynchronize()
{
    using func_ptr = cudaError_t (*)(void);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_cuda("cudaDeviceSynchronize"));
    return func_entry();
}

static inline cudaError_t real_cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    using func_ptr = cudaError_t (*)(cudaStream_t stream, cudaEvent_t event, unsigned int flags);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_cuda("cudaStreamWaitEvent"));
    return func_entry(stream, event, flags);
}

static inline void *get_symbol_openssl(const char *name)
{
    auto handle = dlopen("/root/anaconda3/envs/vllm/lib/libcrypto.so.4", RTLD_NOW | RTLD_LOCAL);
    return dlsym(handle, name);
}

static inline EVP_CIPHER_CTX *real_EVP_CIPHER_CTX_new()
{
    using func_ptr = EVP_CIPHER_CTX *(*)(void);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_CIPHER_CTX_new"));
    return func_entry();
}

static inline int real_EVP_EncryptUpdate(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl,
                      const unsigned char *in, int inl)
{
    using func_ptr = int (*)(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl,
                      const unsigned char *in, int inl);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_EncryptUpdate"));
    return func_entry(ctx, out, outl, in, inl);
}

static inline int real_EVP_EncryptInit_ex(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *cipher,
                       ENGINE *impl, const unsigned char *key,
                       const unsigned char *iv)
{
    using func_ptr = int (*)(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *cipher,
                       ENGINE *impl, const unsigned char *key,
                       const unsigned char *iv);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_EncryptInit_ex"));
    return func_entry(ctx, cipher, impl, key, iv);
}

static inline int real_EVP_EncryptFinal_ex(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl)
{
    using func_ptr = int (*)(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_EncryptFinal_ex"));
    return func_entry(ctx, out, outl);
}

static inline int real_EVP_CIPHER_CTX_ctrl(EVP_CIPHER_CTX *ctx, int type, int arg, void *ptr)
{
    using func_ptr = int (*)(EVP_CIPHER_CTX *ctx, int type, int arg, void *ptr);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_CIPHER_CTX_ctrl"));
    return func_entry(ctx, type, arg, ptr);
}

static inline int real_EVP_DecryptUpdate(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl,
                      const unsigned char *in, int inl)
{
    using func_ptr = int (*)(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl,
                      const unsigned char *in, int inl);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_DecryptUpdate"));
    return func_entry(ctx, out, outl, in, inl);
}

static inline int real_EVP_DecryptInit_ex(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *cipher,
                       ENGINE *impl, const unsigned char *key,
                       const unsigned char *iv)
{
    using func_ptr = int (*)(EVP_CIPHER_CTX *ctx, const EVP_CIPHER *cipher,
                       ENGINE *impl, const unsigned char *key,
                       const unsigned char *iv);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_DecryptInit_ex"));
    return func_entry(ctx, cipher, impl, key, iv);
}

static inline int real_EVP_DecryptFinal_ex(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl)
{
    using func_ptr = int (*)(EVP_CIPHER_CTX *ctx, unsigned char *out, int *outl);
    static auto func_entry = reinterpret_cast<func_ptr>(get_symbol_openssl("EVP_DecryptFinal_ex"));
    return func_entry(ctx, out, outl);
}