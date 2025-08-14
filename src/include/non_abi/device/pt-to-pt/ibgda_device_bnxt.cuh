/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _NVSHMEMI_IBGDA_DEVICE_H_
#define _NVSHMEMI_IBGDA_DEVICE_H_

#include <cuda_runtime.h>
#if !defined __CUDACC_RTC__
#include <limits.h>
#else
#include <cuda/std/climits>
#endif

#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "device_host_transport/nvshmem_common_ibgda.h"
#include "non_abi/nvshmem_build_options.h"
#include "utils_device.h"
#include "infiniband/bnxt_re_hsi.h"

#include <algorithm>

#undef HTOLE16
#define HTOLE16(x)        (x)

#undef HTOLE32
#define HTOLE32(x)        (x)

#undef HTOLE64
#define HTOLE64(x)        (x)

#undef LE32TOH
#define LE32TOH(x)        (x)

//#define NVSHMEM_IBGDA_DEBUG
//#define NVSHMEM_TIMEOUT_DEVICE_POLLING

#define NVSHMEMI_MIN(x, y) ((x) < (y) ? (x) : (y))
#define NVSHMEMI_MAX(x, y) ((x) > (y) ? (x) : (y))

#define NVSHMEMI_IBGDA_PTX_OPTIMIZATION_MFENCE

#ifdef NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY
// These PTX optimizations are for GPU memory access only.
// Both data and NIC control objects must be in GPU memory.
#define NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
#define NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
#endif

#define IBGDA_FULL_WARP 0xffffffffU
#define IBGDA_POLL_TIMEOUT 4000000000LLU

/* When we exceed a specific number of threads doing quiet
 * we end up with cache thrashing which causes a significant
 * perf hit. TODO: Tune this number for each supported arch.
 */
#define IBGDA_MAX_THREADS_PER_QUIET 32

// BNXT accepts up to 2 GiB per command
// TBD - Double check
#define IBGDA_MAX_TRANSFER_SIZE 2147483648LLU

#ifndef likely
#define likely(x) (__builtin_expect(!!(x), 1))
#endif

#ifndef unlikely
#define unlikely(x) (__builtin_expect(!!(x), 0))
#endif

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

/**
 * DO NOT use BSWAP(READ_ONCE(x)) as it could create a bug.
 * BSWAP is a pre-processor function. It will be unrolled to many READ_ONCE.
 */
#ifndef READ_ONCE
#define READ_ONCE(x) ACCESS_ONCE(x)
#endif

#ifndef WRITE_ONCE
#define WRITE_ONCE(x, v) (ACCESS_ONCE(x) = (v))
#endif

#define IBGDA_4_BYTE_EXT_AMO_OPMOD 0x08000000
#define IBGDA_8_BYTE_EXT_AMO_OPMOD 0x09000000

typedef struct bnxt_re_bsqe __attribute__((__aligned__(8))) ibgda_bnxt_ctrl_seg_t;

typedef struct {
    uint32_t add_data;
    uint32_t field_boundary;
    uint64_t reserved;
} __attribute__((__packed__)) ibgda_atomic_32_masked_fa_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(ibgda_atomic_32_masked_fa_seg_t) == 16,
              "sizeof(ibgda_atomic_32_masked_fa_seg_t) == 16 failed.");
#endif

typedef struct {
    uint64_t add_data;
    uint64_t field_boundary;
} __attribute__((__packed__)) ibgda_atomic_64_masked_fa_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(ibgda_atomic_64_masked_fa_seg_t) == 16,
              "sizeof(ibgda_atomic_64_masked_fa_seg_t) == 16 failed.");
#endif

typedef struct {
    uint32_t swap_data;
    uint32_t compare_data;
    uint32_t swap_mask;
    uint32_t compare_mask;
} __attribute__((__packed__)) ibgda_atomic_32_masked_cs_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(ibgda_atomic_32_masked_cs_seg_t) == 16,
              "sizeof(ibgda_atomic_32_masked_cs_seg_t) == 16 failed.");
#endif

typedef struct {
    uint64_t swap;
    uint64_t compare;
} __attribute__((__packed__)) ibgda_atomic_64_masked_cs_seg_t;
#if __cplusplus >= 201103L
static_assert(sizeof(ibgda_atomic_64_masked_cs_seg_t) == 16,
              "sizeof(ibgda_atomic_64_masked_cs_seg_t) == 16 failed.");
#endif

#ifdef __CUDA_ARCH__

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t ibgda_query_globaltimer() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret)::"memory");
    return ret;
}
#endif /* NVSHMEM_TIMEOUT_DEVICE_POLLING */

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE nvshmemi_ibgda_device_state_t *
ibgda_get_state() {
    return &nvshmemi_ibgda_device_state_d;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE bool ibgda_is_rc_enabled() {
    return ibgda_get_state()->num_rc_per_pe > 0;
}

// Prevent code reordering from both compiler and GPU
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void IBGDA_MFENCE() {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_MFENCE
    asm volatile("fence.acq_rel.cta;" ::: "memory");
#else
    __threadfence_block();
#endif /* NVSHMEMI_IBGDA_PTX_OPTIMIZATION_MFENCE */
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void IBGDA_MEMBAR_NO_OPTIMIZATION() {
#ifdef NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY
    __threadfence();
#else
    if (likely(ibgda_get_state()->nic_buf_on_gpumem))
        __threadfence();
    else
        __threadfence_system();
#endif /* NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY */
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void IBGDA_MEMBAR() {
// st.release automatically adds membar in SASS.
#ifndef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE

#ifdef NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY
    __threadfence();
#else
    if (likely(ibgda_get_state()->nic_buf_on_gpumem))
        __threadfence();
    else
        __threadfence_system();
#endif /* NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY */

#endif /* NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE */
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_thread_id_in_warp() {
    int myIdx;
    asm volatile("mov.u32  %0,  %%laneid;" : "=r"(myIdx));
    return myIdx;
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_warp_size() {
    return ((blockDim.x * blockDim.y * blockDim.z) < warpSize)
               ? (blockDim.x * blockDim.y * blockDim.z)
               : warpSize;
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_warp_sync() { __syncwarp(); }

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_thread_id_in_block() {
    return (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y);
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE int nvshmemi_block_size() {
    return (blockDim.x * blockDim.y * blockDim.z);
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint32_t ibgda_get_smid() {
    uint32_t smid;
    asm("mov.u32  %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint32_t ibgda_get_ctaid() {
    return (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y);
}

template <typename T>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_store_relaxed(T *ptr, T val) {
    WRITE_ONCE(*ptr, val);
}

template <>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_store_relaxed(uint8_t *ptr, uint8_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    uint16_t _val = val;
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b8 [%0], %1;" : : "l"(ptr), "h"(_val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

template <>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_store_relaxed(uint16_t *ptr, uint16_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b16 [%0], %1;" : : "l"(ptr), "h"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

template <>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_store_relaxed(uint32_t *ptr, uint32_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

template <>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_store_relaxed(uint64_t *ptr, uint64_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_store_release(uint32_t *ptr,
                                                                                  uint32_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_store_release(uint64_t *ptr,
                                                                                  uint64_t val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_STORE_RELEASE
    asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

/**
 * DO NOT use BSWAP(ibgda_atomic_read(x)) as it could create a bug.
 * See the comment near READ_ONCE.
 */
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint8_t ibgda_atomic_read(uint8_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return (uint8_t)ret;
#else
    return READ_ONCE(*ptr);
#endif
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint16_t ibgda_atomic_read(uint16_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
    uint16_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return ret;
#else
    return READ_ONCE(*ptr);
#endif
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint32_t ibgda_atomic_read(uint32_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
    uint32_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
#else
    return READ_ONCE(*ptr);
#endif
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t ibgda_atomic_read(uint64_t *ptr) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
    uint64_t ret;
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
#else
    return READ_ONCE(*ptr);
#endif
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_atomic_set(int *ptr, int val) {
#ifdef NVSHMEMI_IBGDA_PTX_OPTIMIZATION_ATOMIC_READ_SET
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#else
    WRITE_ONCE(*ptr, val);
#endif
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE size_t
ibgda_cal_transfer_size(size_t req_size, size_t lchunk_size, size_t rchunk_size) {
    return NVSHMEMI_MIN(IBGDA_MAX_TRANSFER_SIZE,
                        NVSHMEMI_MIN(req_size, NVSHMEMI_MIN(rchunk_size, lchunk_size)));
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void global_lock_acquire(int *lock) {
    // Attempt to acquire lock by swapping 0 → 1
    while (atomicCAS(lock, 0, 1) != 0) {
        // Optional: backoff or __nanosleep to reduce contention
    }

    // Ensure all memory operations after the lock are not reordered before it
    __threadfence();
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void global_lock_release(int *lock) {
    // Ensure all memory operations are complete before releasing
    __threadfence();
    atomicExch(lock, 0);
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_lock_acquire(int *lock) {
    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0)
        while (atomicCAS(lock, 0, 1) == 1)
            ;  // Wait until we get the lock.

    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD)
        IBGDA_MFENCE();  // Prevent reordering before lock is acquired.

    // For other scopes, __syncwarp / __syncthreads guarantee the ordering
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_lock_release(int *lock) {
    // For other scopes, __syncwarp / __syncthreads guarantee the ordering
    nvshmemi_threadgroup_sync<SCOPE>();

    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD)
        IBGDA_MFENCE();  // Prevent reordering before lock is released.

    if (nvshmemi_thread_id_in_threadgroup<SCOPE>() == 0) ibgda_atomic_set(lock, 0);
}

// Multiple threads may update get_head concurrently.
// Only the latest one w.r.t. wqe_idx is important.
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_update_get_head(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t new_get_head) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    atomicMax((unsigned long long int *)&mvars->tx_wq.get_head,
              (unsigned long long int)new_get_head);
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_update_get_tail(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t new_get_tail) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    atomicMax((unsigned long long int *)&mvars->tx_wq.get_tail,
              (unsigned long long int)new_get_tail);
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void *ibgda_get_wqe_slot_ptr(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t slot_idx) {
    slot_idx = slot_idx % qp->tx_wq.sq_depth;
    return (void *)((uintptr_t)qp->tx_wq.wqe + slot_idx * BNXT_RE_SLOT_SIZE_BB);
}

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE int ibgda_check_poll_timeout(
    nvshmemi_ibgda_device_cq_t *cq, uint64_t now, uint64_t start, uint64_t idx, int *error) {
    int status = 0;

    if (unlikely(now - start > IBGDA_POLL_TIMEOUT)) {
        *error = -ETIME;
        /*
        printf(
            "[%d] ibgda_poll_cq timeout:\n"
            "    cons_idx=%#lx, prod_idx=%#lx, cqn=%#x, qpn=%#x\n"
            "    resv_head=%#lx, ready_head=%#lx\n"
            "    while waiting for idx=%#lx\n",
            nvshmemi_device_state_d.mype, ibgda_atomic_read(cq->cons_idx),
            ibgda_atomic_read(cq->prod_idx), cq->cqn, cq->qpn,
            ibgda_atomic_read(cq->resv_head),
            ibgda_atomic_read(cq->ready_head), idx);
        */
        status = -1;
    }
    return status;
}
#endif

#define bnxt_re_get_cqe_sz()    (sizeof(struct bnxt_re_req_cqe) +   \
                 sizeof(struct bnxt_re_bcqe))

#define bnxt_re_is_cqe_valid(valid, phase)              \
                (((valid) & BNXT_RE_BCQE_PH_MASK) == (phase))

#if __cplusplus >= 201103L
static_assert(NVSHMEMI_IBGDA_MAX_QP_DEPTH <= 32768,
              "static_assert(NVSHMEMI_IBGDA_MAX_QP_DEPTH <= 32768) failed");
#endif

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE int ibgda_poll_cq(
    nvshmemi_ibgda_device_cq_t *cq, uint64_t idx, int *error) {
    int status = 0;

    assert(likely(cq->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC));

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    uint64_t start = ibgda_query_globaltimer();
    uint64_t now;
#endif
    bnxt_re_bcqe *hdr;
    uint32_t flg_val;
    struct bnxt_re_req_cqe *hwcqe = (struct bnxt_re_req_cqe *)cq->cqe;
#ifdef NVSHMEM_IBGDA_DEBUG
    bool valid_comp = false;
#endif
    uint8_t cqe_status = 0;

    // If idx is a lot greater than cons_idx, we might get incorrect result due
    // to wqe_counter wraparound. We need to check prod_idx to be sure that idx
    // has already been submitted.
    while (unlikely(ibgda_atomic_read(cq->prod_idx) < idx))
        ;
    IBGDA_MFENCE();

    global_lock_acquire<NVSHMEMI_THREADGROUP_THREAD>(cq->poll_cq_lock);

    auto cons_idx = ibgda_atomic_read(cq->cons_idx);
    auto prod_idx = ibgda_atomic_read(cq->prod_idx);
    auto cqe_idx = ibgda_atomic_read(cq->cqe_idx);
    if (idx <= cons_idx)
        goto poll_done;

    // Handle some CQ polling TBD. CQ poll might be called for periodic
    // check and it is possible to have no potential completion on that CQ.
    // Currently it is handled through timeout.
    do {
        cons_idx = ibgda_atomic_read(cq->cons_idx);
        prod_idx = ibgda_atomic_read(cq->prod_idx);
        cqe_idx = ibgda_atomic_read(cq->cqe_idx);

        hwcqe = (struct bnxt_re_req_cqe *)((unsigned long)cq->cqe +
                 (cqe_idx * bnxt_re_get_cqe_sz()));
        hdr = (bnxt_re_bcqe*)((unsigned long)hwcqe + sizeof(bnxt_re_req_cqe));
        flg_val = LE32TOH(ibgda_atomic_read(&hdr->flg_st_typ_ph));

#ifdef NVSHMEM_IBGDA_DEBUG
        uint32_t *cqe_slot;
        int i;

        cqe_slot = (uint32_t *)(uint32_t*)hwcqe;
        for (i = 0; i < 4; i++) {
            printf("hwcqe 0x%lx : %08x %08x %08x %08x (cqn %d slot %ld)\n",
                 &cqe_slot[0], (cqe_slot[1]), (cqe_slot[0]),
                 (cqe_slot[3]), (cqe_slot[2]), cq->cqn, cons_idx + i);
            cqe_slot = cqe_slot + 4;
        }
        printf("qpn 0x%x cqn 0x%x flg_val %08x  resv_head 0x%lx ready_head 0x%lx "
            "idx from caller 0x%lx tx prod 0x%lx tx cons 0x%lx/0x%lx "
            "(hw sq_cons 0x%x) phase 0x%lx\n",
            cq->qpn, cq->cqn, flg_val, ibgda_atomic_read(cq->resv_head),
            ibgda_atomic_read(cq->ready_head), idx,
            prod_idx, cons_idx, cqe_idx, hwcqe->con_indx, ibgda_atomic_read(cq->cq_phase));
#endif

        if (bnxt_re_is_cqe_valid(flg_val, (uint32_t)ibgda_atomic_read(cq->cq_phase))) {
#ifdef NVSHMEM_IBGDA_DEBUG
            valid_comp = true;
#endif
            cqe_idx = (cqe_idx + 1) % cq->ncqes;
            if (cqe_idx == 0)
                atomicXor((unsigned long long int *)cq->cq_phase, 0x1);
            atomicExch((unsigned long long int *)cq->cqe_idx, cqe_idx);

            int wqe_cnt = hwcqe->con_indx - (int)ibgda_atomic_read(cq->sq_cons_idx);
            if (wqe_cnt < 0)
                wqe_cnt += cq->sq_size;
            cons_idx += wqe_cnt;
            atomicMax((unsigned long long int *)cq->cons_idx, cons_idx);
            atomicExch((unsigned long long int *)cq->sq_cons_idx, (unsigned long long int)hwcqe->con_indx);

            cqe_status = (flg_val >> BNXT_RE_BCQE_STATUS_SHIFT) &
                          BNXT_RE_BCQE_STATUS_MASK;
            if (cqe_status) {
                status = -1;
                goto check_opcode;
            }
        }
#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
        // TODO: Integrate timeout handler with the core NVSHMEM
        now = ibgda_query_globaltimer();
        status = ibgda_check_poll_timeout(cq, now, start, idx, error);
        if (status != 0) goto check_opcode;
#endif
        // TBD - We need proper handling here.
        // Poll might be called for those CQs as well, which has never
        // done any posting.
    } while (cons_idx < idx);

    // Prevent reordering of the opcode wait above
    IBGDA_MFENCE();

#ifdef NVSHMEM_TIMEOUT_DEVICE_POLLING
    start = ibgda_query_globaltimer();
#endif

check_opcode:
    /* TBD CQE_REQ_ERR Case handling*/

    // Prevent reordering of this function and subsequent instructions
    IBGDA_MFENCE();
#ifdef NVSHMEM_IBGDA_DEBUG
    printf(
        "[%d] ibgda_poll_cq %s: \n"
        "    cons_idx=%#lx, prod_idx=%#lx, cqn=%#x, qpn=%#x \n"
        "    resv_head=%#lx, ready_head=%#lx\n"
        "    while waiting for idx=%#lx. cqe_status 0x%x\n",
        nvshmemi_device_state_d.mype, valid_comp ? "SUCCESS" : "TIMEOUT",
        cons_idx, prod_idx, cq->cqn, cq->qpn,
        ibgda_atomic_read(cq->resv_head), ibgda_atomic_read(cq->ready_head), idx,
        cqe_status);
#endif

poll_done:
    global_lock_release<NVSHMEMI_THREADGROUP_THREAD>(cq->poll_cq_lock);
    return status;
}

// Updates the last slot idx unconditionally with wrap consideration
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t ibgda_reserve_slot_idx(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t num_slots) {
    unsigned long long int *addr, oldval, assumed, newval;

    addr = (unsigned long long int *)&qp->mvars.tx_wq.resv_prod_slot_idx;
    if (!num_slots)
        return atomicAdd(addr, 0);  //Safe read

    oldval = atomicAdd(addr, 0);  //First read
    do {
        assumed = oldval;
        newval = (assumed + num_slots) % qp->tx_wq.sq_depth;
        oldval = atomicCAS(addr, assumed, newval);
    } while (oldval != assumed);

#ifdef NVSHMEM_IBGDA_DEBUG
    printf("[%d] update slot idx from %lld to %lld\n",
           nvshmemi_device_state_d.mype, oldval, newval);
#endif

    return oldval;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t bnxt_re_get_pkts_per_wqe(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t data_bytes) {
    if (!data_bytes)
        data_bytes = 1;
    return (data_bytes + qp->mtu - 1) / qp->mtu;
}

// These MSN table routines need to change to cp the entire 64b into the GPU memory instead
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void *bnxt_re_pull_psn_buff(
    nvshmemi_ibgda_device_qp_t *qp, uint32_t msn_idx) {
   // MSN entries are 64b wide << 3
   return (void *)(((char *) qp->pad) + (msn_idx << 3));
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t bnxt_re_update_msn_tbl(
    uint32_t st_idx, uint32_t npsn, uint32_t start_psn) {
   /* Adjust the field values to their respective ofsets */
   /* In LE64 */
   return ((((uint64_t)(st_idx) << BNXT_RE_SQ_MSN_SEARCH_START_IDX_SHIFT) &
                       BNXT_RE_SQ_MSN_SEARCH_START_IDX_MASK) |
                       (((uint64_t)(npsn) << BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_SHIFT) &
                       BNXT_RE_SQ_MSN_SEARCH_NEXT_PSN_MASK) |
                       (((start_psn) << BNXT_RE_SQ_MSN_SEARCH_START_PSN_SHIFT) &
                       BNXT_RE_SQ_MSN_SEARCH_START_PSN_MASK));
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void bnxt_re_fill_psns_for_msntbl(
    nvshmemi_ibgda_device_qp_t *qp, uint16_t slot_idx, uint32_t msn_idx, uint32_t start_psn,
    uint32_t pkt_cnt) {
    struct bnxt_re_msns msns;
    uint64_t *msns_ptr;

    msns_ptr = (uint64_t *)bnxt_re_pull_psn_buff(qp, msn_idx);
    msns.start_idx_next_psn_start_psn =
                        bnxt_re_update_msn_tbl(slot_idx, start_psn + pkt_cnt, start_psn);

    ibgda_store_release(msns_ptr, *((uint64_t *)&msns));
}

template <bool support_half_av_seg>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_write_rdma_write_wqe(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t laddr, __be32 lkey, uint64_t raddr, __be32 rkey,
    uint32_t bytes, uint64_t wqe_slot_idx, uint64_t msn_idx, uint64_t psn, uint32_t npkts,
    uint8_t fm_ce_se) {
    ibgda_bnxt_ctrl_seg_t ctrl_seg;
    struct bnxt_re_rdma raddr_seg;
    struct bnxt_re_sge data_seg;
    uint32_t slots = 3;

    assert(likely(qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC));

    ibgda_bnxt_ctrl_seg_t *ctrl_seg_ptr = (ibgda_bnxt_ctrl_seg_t *)ibgda_get_wqe_slot_ptr(qp, wqe_slot_idx);
    struct bnxt_re_rdma *raddr_seg_ptr = (struct bnxt_re_rdma *)ibgda_get_wqe_slot_ptr(qp, wqe_slot_idx + 1);
    struct bnxt_re_sge *data_seg_ptr = (struct bnxt_re_sge *)ibgda_get_wqe_slot_ptr(qp, wqe_slot_idx + 2);

    ctrl_seg = { 0 };
    ctrl_seg.rsv_ws_fl_wt = HTOLE32((slots << BNXT_RE_HDR_WS_SHIFT) |
                                    (BNXT_RE_WR_FLAGS_SIGNALED << BNXT_RE_HDR_FLAGS_SHIFT) |
                                    BNXT_RE_WR_OPCD_RDMA_WRITE);
    ctrl_seg.key_immd = 0;
    ctrl_seg.lhdr.qkey_len = HTOLE32(bytes);

    raddr_seg.rva = HTOLE64(raddr);
    raddr_seg.rkey = HTOBE32(rkey);
    raddr_seg.ts = 0;

    data_seg.length = HTOLE32(bytes);
    data_seg.lkey = HTOBE32(lkey);
    data_seg.pa = HTOLE64(laddr);

    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

#ifdef NVSHMEM_IBGDA_DEBUG
    printf("slot 0: %08x %08x (ctrl_seg_ptr 0x%lx)\n", (dst[1]), (dst[0]),
                    (unsigned long) ctrl_seg_ptr);
    printf("      : %08x %08x\n", (dst[3]), (dst[2]));
#endif

    dst = (uint32_t *)raddr_seg_ptr;
    src = (uint32_t *)&raddr_seg;
    for (int i = 0; i < sizeof(*raddr_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

#ifdef NVSHMEM_IBGDA_DEBUG
    printf("slot 1: %08x %08x\n", (dst[1]), (dst[0]));
    printf("      : %08x %08x\n", (dst[3]), (dst[2]));
#endif
    dst = (uint32_t *)data_seg_ptr;
    src = (uint32_t *)&data_seg;
    for (int i = 0; i < sizeof(*data_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

    // Calculate and fill start and end PSN of the WQE
    bnxt_re_fill_psns_for_msntbl(qp, wqe_slot_idx, msn_idx, psn, npkts);
}

template <bool support_half_av_seg>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_write_rdma_write_inl_wqe(
    nvshmemi_ibgda_device_qp_t *qp, const void *val, uint64_t raddr, __be32 rkey,
    uint32_t bytes, uint64_t wqe_slot_idx, uint64_t msn_idx, uint64_t psn, uint8_t fm_ce_se) {
    ibgda_bnxt_ctrl_seg_t ctrl_seg;
    struct bnxt_re_rdma raddr_seg;
    uint32_t slots = 2;

    ibgda_bnxt_ctrl_seg_t *ctrl_seg_ptr = (ibgda_bnxt_ctrl_seg_t *)ibgda_get_wqe_slot_ptr(qp, wqe_slot_idx);
    struct bnxt_re_rdma *raddr_seg_ptr = (struct bnxt_re_rdma *)ibgda_get_wqe_slot_ptr(qp, wqe_slot_idx + 1);
    struct bnxt_re_sge *wqe_data_seg_ptr = (struct bnxt_re_sge *)ibgda_get_wqe_slot_ptr(qp, wqe_slot_idx + 2);

    // Allow up to 12 bytes
    //assert(likely(bytes <= 12));

    slots += (bytes + (BNXT_RE_SLOT_SIZE_BB - 1)) / BNXT_RE_SLOT_SIZE_BB;
    ctrl_seg = { 0 };
    ctrl_seg.rsv_ws_fl_wt = HTOLE32((slots << BNXT_RE_HDR_WS_SHIFT) |
                                (BNXT_RE_WR_FLAGS_SIGNALED << BNXT_RE_HDR_FLAGS_SHIFT) |
                                (BNXT_RE_WR_FLAGS_INLINE << BNXT_RE_HDR_FLAGS_SHIFT) |
                                BNXT_RE_WR_OPCD_RDMA_WRITE);
    ctrl_seg.key_immd = HTOLE32(*(uint32_t *)val);
    ctrl_seg.lhdr.qkey_len = HTOLE32(bytes);

    raddr_seg.rva = HTOLE64(raddr);
    raddr_seg.rkey = HTOBE32(rkey);
    raddr_seg.ts = 0;

    uint32_t *dst = (uint32_t *)ctrl_seg_ptr;
    uint32_t *src = (uint32_t *)&ctrl_seg;
    for (int i = 0; i < sizeof(*ctrl_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);

#ifdef NVSHMEM_IBGDA_DEBUG
    printf("slot 0: %08x %08x (ctrl_seg_ptr 0x%lx) bytes %d\n",
            (dst[1]), (dst[0]),
            (unsigned long) ctrl_seg_ptr, bytes);
    printf("      : %08x %08x\n", (dst[3]), (dst[2]));
#endif

    dst = (uint32_t *)raddr_seg_ptr;
    src = (uint32_t *)&raddr_seg;
    for (int i = 0; i < sizeof(*raddr_seg_ptr) / sizeof(uint32_t); ++i)
        ibgda_store_relaxed(&dst[i], src[i]);
#ifdef NVSHMEM_IBGDA_DEBUG
    printf("slot 1: %08x %08x\n", (dst[1]), (dst[0]));
    printf("      : %08x %08x\n", (dst[3]), (dst[2]));
#endif

    switch (bytes) {
        case 1:
            ibgda_store_relaxed((uint8_t *)wqe_data_seg_ptr, *((uint8_t *)val));
            break;
        case 2:
            ibgda_store_relaxed((uint16_t *)wqe_data_seg_ptr, *((uint16_t *)val));
            break;
        case 4:
            ibgda_store_relaxed((uint32_t *)wqe_data_seg_ptr, *((uint32_t *)val));
            break;
        case 8:
            // wqe_data_ptr is aligned at 4B. We cannot use uint64_t here.
            ibgda_store_relaxed(&(((uint32_t *)wqe_data_seg_ptr)[0]), ((uint32_t *)val)[0]);
            ibgda_store_relaxed(&(((uint32_t *)wqe_data_seg_ptr)[1]), ((uint32_t *)val)[1]);
            break;
        default:
            memcpy(wqe_data_seg_ptr, val, bytes);
    }

#ifdef NVSHMEM_IBGDA_DEBUG
    dst = (uint32_t *)wqe_data_seg_ptr;
    printf("slot 2: %08x %08x\n", (dst[1]), (dst[0]));
    printf("      : %08x %08x\n", (dst[3]), (dst[2]));
#endif

    // Calculate and fill start and end PSN of the WQE
    bnxt_re_fill_psns_for_msntbl(qp, wqe_slot_idx, msn_idx, psn, 1);
}

/**
 * For RC
 * The header already consumes 1 wqebb and leaves 12 + 16 bytes for NVSHMEMI_DEVICE_ALWAYS_INLINE
 * data. The last wqebb is no-op. One wqebb is 64 bytes. Pre-calculate as it is faster to do lookup.
 * Formula: ceil(((sizeof(T) * 32) - (12 + 16)) / 64) + 2
 */
template <typename T, nvshmemi_ibgda_device_qp_type_t qp_type>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint32_t
ibgda_get_num_wqes_in_inl_combine_warp() {
        // RC supports up to 64 DS WQE
        switch (sizeof(T)) {
            case 1:
            case 2:
                return 3;
            case 4:
                return 4;
            case 8:
                return 6;
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unsupported type.\n");
#endif
                assert(0);
                return 0;
        }
}

template <typename T>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void
ibgda_write_rdma_write_inl_wqe_combine_warp(nvshmemi_ibgda_device_qp_t *qp, const T val,
                                            uint64_t _raddr, __be32 rkey,
                                            uint16_t wqe_slot_idx, int my_tid) {
    // TODO - The write inline data is based on the thread ID
    assert(0);
}

template <bool support_half_av_seg>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_write_rdma_read_wqe(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t laddr, __be32 lkey,
    uint64_t raddr, __be32 rkey, uint32_t bytes, uint16_t wqe_slot_idx, uint8_t fm_ce_se) {
    assert(0);
    // TODO - This can be simple same as ibgda_write_rdma_write_wqe.
}

template <typename T>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint32_t
ibgda_get_num_wqes_in_atomic(nvshmemi_amo_t amo_op, nvshmemi_ibgda_device_qp_type_t qp_type) {
    if (sizeof(T) == 8) {
        // RC
        switch (amo_op) {
            case NVSHMEMI_AMO_SIGNAL:
            case NVSHMEMI_AMO_SIGNAL_SET:
            case NVSHMEMI_AMO_SWAP:
            case NVSHMEMI_AMO_SET:
            case NVSHMEMI_AMO_FETCH_AND:
            case NVSHMEMI_AMO_AND:
            case NVSHMEMI_AMO_FETCH_OR:
            case NVSHMEMI_AMO_OR:
                return 2;
        }
    }
    return 1;
}

template <bool support_half_av_seg>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_write_atomic_wqe(
    nvshmemi_ibgda_device_qp_t *qp, const void *val_1,
    const void *val_2, uint64_t laddr, __be32 lkey, uint64_t raddr, __be32 rkey, uint32_t bytes,
    uint64_t wqe_slot_idx, nvshmemi_amo_t amo_op, uint8_t fm_ce_se) {
#ifdef IBGDA_USE_AMO_OPS
    // TODO: Depending on the specific operation, different RDMA ops might be used
    switch (amo_op) {
        case NVSHMEMI_AMO_FETCH_INC:
        case NVSHMEMI_AMO_INC:
        case NVSHMEMI_AMO_SIGNAL:
        case NVSHMEMI_AMO_SIGNAL_SET:
        case NVSHMEMI_AMO_SWAP:
        case NVSHMEMI_AMO_SET:
        case NVSHMEMI_AMO_SIGNAL_ADD:
        case NVSHMEMI_AMO_ADD:
        case NVSHMEMI_AMO_FETCH_AND:
        case NVSHMEMI_AMO_AND:
        case NVSHMEMI_AMO_FETCH_OR:
        case NVSHMEMI_AMO_OR:
        case NVSHMEMI_AMO_FETCH_XOR:
        case NVSHMEMI_AMO_XOR:
        case NVSHMEMI_AMO_FETCH:
        case NVSHMEMI_AMO_FETCH_ADD:
        case NVSHMEMI_AMO_COMPARE_SWAP:
        default:
            assert(0);
            break;
    }
#else
    assert(0);
#endif
}

// A NOP WQE essentially is a WQE which does nothing operational
// The goal of issuing this WQE is to get a completion
// TODO:  Use a local-invalidate WQE with a MW which is already freed
// The HW will treat it as a NOP and will get completed if signaled
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_write_nop_wqe(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t wqe_slot_idx) {
    // A local-invalidate with a freed MW WQE allows the NIC to perform a NOP
    // and at the same time generate fence/completion per request
    assert(0);
}

// A DUMP WQE essentially is a WQE which performs nothing operational
// except that it does touch the local address provided and generate a completion
// TODO:  Use a zero-length RDMA write WQE instead
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_write_dump_wqe(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t laddr, __be32 lkey, uint32_t bytes,
    uint64_t wqe_slot_idx, uint8_t fm_ce_se) {
    // A zero-len write WQE allows the NIC to read laddr, which is always on GPU memory.
    //ibgda_write_rdma_write_inl_wqe<false>(qp, 0, laddr, lkey, 0, wqe_idx,
    //                                      fm_ce_se, out_wqes);
    assert(0);
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void bnxt_re_init_db_hdr(
    struct bnxt_re_db_hdr *hdr, uint32_t indx, uint32_t epoch,
    uint32_t qid, uint32_t typ) {
   uint64_t key_lo, key_hi;

   key_lo = HTOLE32((indx & BNXT_RE_DB_INDX_MASK) |
                    (epoch << BNXT_RE_DB_EPOCH_SHIFT));
   key_hi = HTOLE32((qid & BNXT_RE_DB_QID_MASK) |
            ((typ & BNXT_RE_DB_TYP_MASK) << BNXT_RE_DB_TYP_SHIFT) |
            (0x1UL << BNXT_RE_DB_VALID_SHIFT));
   hdr->typ_qid_indx = HTOLE32((key_lo | (key_hi << 32)));
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_ring_db(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t slot_idx) {
    uint32_t prod_slot_idx = (uint32_t)slot_idx;
    uint32_t epoch = (uint32_t)ibgda_atomic_read(&qp->mvars.tx_wq.epoch);
    uint64_t *bf_ptr = (uint64_t *)qp->tx_wq.bf;
    struct bnxt_re_db_hdr hdr;

    bnxt_re_init_db_hdr(&hdr, prod_slot_idx, epoch, qp->qpn, BNXT_RE_QUE_TYPE_SQ);

#ifdef NVSHMEM_IBGDA_DEBUG
    printf("From %s %d qpn 0x%x prod_slot_idx %#x at 0x%lx nwqes 0x%x cq_handle 0x%lx\n",
                    __func__, __LINE__, qp->qpn, prod_slot_idx,
                    (unsigned long)bf_ptr, qp->tx_wq.nwqes, (unsigned long)qp->tx_wq.cq);
#endif
    // Write to the actual DB
    ibgda_store_release(bf_ptr, *((uint64_t *)&hdr));
}

template <bool need_strong_flush>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_post_send(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t new_prod_idx, uint64_t new_slot_idx) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t old_prod_idx;

    // Update prod_idx before ringing the db so that we know which index is needed in quiet/fence.
    global_lock_acquire<NVSHMEMI_THREADGROUP_THREAD>(&mvars->post_send_lock);

    if (need_strong_flush)
        old_prod_idx = atomicMax((unsigned long long int *)&mvars->tx_wq.prod_idx,
                                 (unsigned long long int)new_prod_idx);
    else
        old_prod_idx = atomicMax_block((unsigned long long int *)&mvars->tx_wq.prod_idx,
                                       (unsigned long long int)new_prod_idx);

    if (likely(new_prod_idx > old_prod_idx)) {
        if (new_slot_idx < atomicAdd((unsigned long long int *)&mvars->tx_wq.posted_prod_slot_idx, 0ULL))
            atomicXor((unsigned long long int *)&mvars->tx_wq.epoch, 0x1ULL);

        if (need_strong_flush)
            atomicExch((unsigned long long int *)&mvars->tx_wq.posted_prod_slot_idx,
                       (unsigned long long int)new_slot_idx);
        else
            atomicExch_block((unsigned long long int *)&mvars->tx_wq.posted_prod_slot_idx,
                             (unsigned long long int)new_slot_idx);
        IBGDA_MEMBAR();
        ibgda_ring_db(qp, new_slot_idx);
    }

    global_lock_release<NVSHMEMI_THREADGROUP_THREAD>(&mvars->post_send_lock);
}

// If `qp` is shared among CTAs, need_strong_flush must be set to true because
// we must push prior writes from this CTA to L2 before coalescing DB.
template <bool need_strong_flush>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_submit_requests(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t base_wqe_idx, uint16_t num_wqes,
    uint64_t base_slot_idx, uint64_t num_slots) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t mask = ~((uint64_t)(state->num_requests_in_batch - 1));

    uint64_t new_wqe_idx = base_wqe_idx + num_wqes;
    uint64_t new_slot_idx = (base_slot_idx + num_slots) % qp->tx_wq.sq_depth;

    unsigned long long int *ready_idx =
        (unsigned long long int *)(state->use_async_postsend ? qp->tx_wq.prod_idx
                                                             : &mvars->tx_wq.ready_head);
    // TBD: no async postsend support yet
    unsigned long long int *ready_slot =
        (unsigned long long int *)&mvars->tx_wq.ready_prod_slot_idx;

    // WQE writes must be finished first.
    if (need_strong_flush)
        // membar from a different CTA does not push prior writes of this CTA.
        // We must push them out first because a different CTA might post-send for us.
        IBGDA_MEMBAR_NO_OPTIMIZATION();
    else
        // It is ok for those wqes to not be visible to the GPU scope yet.
        // ibgda_post_send will take care of that (if we choose to call it).
        IBGDA_MFENCE();

    // Wait for prior WQE slots to be filled first.
    // They might not be post-sent yet.
    if (need_strong_flush) {
        while (atomicCAS(ready_idx, (unsigned long long int)base_wqe_idx,
                         (unsigned long long int)new_wqe_idx) != base_wqe_idx)
            ;  // wait here
        while (atomicCAS(ready_slot, (unsigned long long int)base_slot_idx,
                         (unsigned long long int)new_slot_idx) != base_slot_idx)
            ;  // wait here
    } else {
        while (atomicCAS_block(ready_idx, (unsigned long long int)base_wqe_idx,
                               (unsigned long long int)new_wqe_idx) != base_wqe_idx)
            ;  // wait here
        while (atomicCAS_block(ready_slot, (unsigned long long int)base_slot_idx,
                               (unsigned long long int)new_slot_idx) != base_slot_idx)
            ;  // wait here
    }

    IBGDA_MFENCE();

    if (!state->use_async_postsend) {
        bool do_post_send =
            (new_wqe_idx ==
             ibgda_atomic_read(&mvars->tx_wq.resv_head))  // No concurrent submissions
            || ((base_wqe_idx & mask) !=
                (new_wqe_idx & mask))  // Num of not-yet-posted wqes is beyond the threshold.
            || (num_wqes >= state->num_requests_in_batch);  // The number of wqes in this submission
                                                            // reaches the threshold.

        if (do_post_send) ibgda_post_send<need_strong_flush>(qp, new_wqe_idx, new_slot_idx);
    }
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t
ibgda_quiet(nvshmemi_ibgda_device_qp_t *qp) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    uint64_t prod_idx = state->use_async_postsend ? ibgda_atomic_read(qp->tx_wq.prod_idx)
                                                  : ibgda_atomic_read(&qp->mvars.tx_wq.ready_head);
    nvshmemi_ibgda_device_cq_t cq = *qp->tx_wq.cq;

    int err = 0;
    int status = ibgda_poll_cq(&cq, prod_idx, &err);
    // TODO: Integrate the error handler with the core NVSHMEM
#ifdef NVSHMEM_IBGDA_DEBUG
    if (status) {
        printf("ibgda_poll_cq failed with error=%d. (%d 0x%lx 0x%lx 0x%lx)\n", err,
                    state->use_async_postsend, prod_idx,
                ibgda_atomic_read(qp->tx_wq.prod_idx),
                    ibgda_atomic_read(&qp->mvars.tx_wq.ready_head));
    }
#endif
    assert(likely(status == 0));
    return prod_idx;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_wait_for_slot_availability(
    nvshmemi_ibgda_device_qp_t *qp, uint64_t wqe_idx) {
    int status = 0;
    int err = 0;
    uint16_t nwqes = qp->tx_wq.nwqes;

    // We don't want wqe_idx - nwqes to wraparound.
    if (likely(wqe_idx >= nwqes)) {
        nvshmemi_ibgda_device_cq_t cq = *qp->tx_wq.cq;
        status = ibgda_poll_cq(&cq, wqe_idx - nwqes, &err);
        // TODO: Integrate the error handler with the core NVSHMEM
        if (status) {
            printf("ibgda_poll_cq failed with error=%d.\n", err);
        }
        assert(likely(status == 0));
    }
    IBGDA_MFENCE();
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE int ibgda_get_proxy_pe(int pe) {
    if (nvshmemi_device_state_d.enable_rail_opt == 1)
        return (pe / nvshmemi_device_state_d.node_npes) * nvshmemi_device_state_d.node_npes +
               nvshmemi_device_state_d.node_mype;
    return pe;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE nvshmemi_ibgda_device_qp_t *ibgda_get_rc(
    int pe, bool *out_shared_among_ctas) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    uint32_t id;
    uint32_t idx;
    uint32_t dev_offset;
    uint32_t warpid = nvshmemi_thread_id_in_block() / nvshmemi_warp_size();

#ifdef NVSHMEM_IBGDA_USE_RC_LOOPBACK
    // Loopback RC QP should have been created
#else
    assert(pe != nvshmemi_device_state_d.mype);
#endif
    switch (state->rc_map_type) {
        case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_CTA:
            id = ibgda_get_ctaid();
            break;
        case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_SM:
            id = ibgda_get_smid();
            break;
        case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_WARP:
            id = ibgda_get_ctaid() * nvshmemi_block_size() / nvshmemi_warp_size() + warpid;
            break;
        default:
            assert(0);
            break;
    }

    dev_offset = ++state->globalmem.qp_group_switches[id % state->num_qp_groups];

    /* round down */
    id = id / state->num_devices_initialized;
    id = (id * state->num_devices_initialized) + (dev_offset % state->num_devices_initialized);

    idx = (pe * state->num_rc_per_pe * state->num_devices_initialized) +
          (id % (state->num_rc_per_pe * state->num_devices_initialized));

    *out_shared_among_ctas = true;
    return &state->globalmem.rcs[idx];
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE nvshmemi_ibgda_device_qp_t *ibgda_get_qp(
    int pe, bool *out_shared_among_ctas) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    // Allow self PE QP to be retrieved
    return ibgda_get_rc(pe, out_shared_among_ctas);
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_get_lkey(
    uint64_t addr, __be32 *lkey, size_t *chunk_size, bool *is_sysmem_scope, uint32_t dev_idx) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    uint64_t heap_start = (uint64_t)nvshmemi_device_state_d.heap_base;
    uint64_t heap_end = heap_start + nvshmemi_device_state_d.heap_size - 1;
    size_t max_len = 1ULL << 30;
    if (heap_start <= addr && addr <= heap_end) {
        // addr in the symmetric heap
        uint64_t idx = ((addr - heap_start) >> state->log2_cumem_granularity) *
                           state->num_devices_initialized +
                       dev_idx;
        nvshmemi_ibgda_device_key_t device_key;

        if (idx < NVSHMEMI_IBGDA_MAX_CONST_LKEYS)
            device_key = state->constmem.lkeys[idx];
        else
            device_key = state->globalmem.lkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_LKEYS];

        assert(addr < device_key.next_addr);

        *lkey = device_key.key;
        *chunk_size = device_key.next_addr - addr;
        *chunk_size = *chunk_size < max_len ? *chunk_size : max_len;
        *is_sysmem_scope = (nvshmemi_device_state_d.symmetric_heap_kind == 1);
        return;
    } else {
        // local-only addr
        nvshmemi_ibgda_device_local_only_mhandle_t *mhandle =
            state->globalmem.local_only_mhandle_head;

        while (mhandle) {
            if (mhandle->start <= addr && addr <= mhandle->end) {
                *lkey = mhandle->lkeys[dev_idx];
                *chunk_size = mhandle->end - addr + 1;
                *chunk_size = *chunk_size < max_len ? *chunk_size : max_len;
                *is_sysmem_scope = mhandle->is_sysmem_scope;
                return;
            }
            mhandle = mhandle->next;
        }
    }

    // lkey is not found.
    assert(0);
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_get_raddr_rkey(
    uint64_t addr, int dst_pe, int proxy_pe, uint64_t *out_raddr, __be32 *out_rkey,
    size_t *out_chunk_size, uint32_t dev_idx) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    uint64_t heap_start = (uint64_t)nvshmemi_device_state_d.heap_base;
    uint64_t roffset = addr - heap_start;
    int npes;
    // nvcc from CUDA12.0 - 12.2 seems to have a bug. It causes
    // nvshmemi_device_state_d.npes to become 0 in this function.
    // WAR: Force reload of nvshmemi_device_state_d.npes. We may reload from L1
    // most of the time, so the performance hit is minimal.
    asm volatile("ld.b32 %0, [%1];" : "=r"(npes) : "l"(&nvshmemi_device_state_d.npes));

    uint64_t idx =
        ((roffset >> state->log2_cumem_granularity) * npes * state->num_devices_initialized) +
        (proxy_pe * state->num_devices_initialized) + dev_idx;
    nvshmemi_ibgda_device_key_t device_key;
    uint64_t raddr;

    if (idx < NVSHMEMI_IBGDA_MAX_CONST_RKEYS)
        device_key = state->constmem.rkeys[idx];
    else
        device_key = state->globalmem.rkeys[idx - NVSHMEMI_IBGDA_MAX_CONST_RKEYS];

    assert(roffset < device_key.next_addr);

    raddr = (uint64_t)nvshmemi_device_state_d.peer_heap_base_remote[proxy_pe] + roffset;
    if (dst_pe != proxy_pe)
        raddr += (dst_pe % nvshmemi_device_state_d.node_npes - nvshmemi_device_state_d.node_mype) *
                 nvshmemi_device_state_d.heap_size;

    *out_raddr = raddr;
    *out_rkey = device_key.key;
    *out_chunk_size = device_key.next_addr - roffset;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t ibgda_reserve_wqe_slots(
    nvshmemi_ibgda_device_qp_t *qp, unsigned long long int num_wqes, bool is_qp_shared_among_ctas,
    int wqe_size, int num_msn, int num_pkts, uint64_t *slot_idx, uint64_t *msn, uint64_t *psn) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint64_t wqe_idx;

    global_lock_acquire<NVSHMEMI_THREADGROUP_THREAD>(&qp->mvars.resv_lock);
    // 1. Reserve wqe_idx
// OK to keep this conditional since we only support one build per major verion.
#if CUDART_VERSION >= 12000
    if (is_qp_shared_among_ctas)
        wqe_idx = atomicAdd((unsigned long long int *)&mvars->tx_wq.resv_head, num_wqes);
    else
        wqe_idx = atomicAdd_block((unsigned long long int *)&mvars->tx_wq.resv_head, num_wqes);
#else
    // WAR NVBUG 3749055. The fix is in nvcc of CUDA 12.0 and later.
    if (is_qp_shared_among_ctas)
        asm volatile("atom.relaxed.gpu.global.add.u64 %0, [%1], %2;"
                     : "=l"(wqe_idx)
                     : "l"(&mvars->tx_wq.resv_head), "l"(num_wqes));
    else
        asm volatile("atom.relaxed.cta.global.add.u64 %0, [%1], %2;"
                     : "=l"(wqe_idx)
                     : "l"(&mvars->tx_wq.resv_head), "l"(num_wqes));
#endif

    // 2. Reserve slot_idx
    *slot_idx = atomicAdd((unsigned long long int *)&mvars->tx_wq.resv_prod_slot_idx, 0);
    atomicExch((unsigned long long int *)&mvars->tx_wq.resv_prod_slot_idx,
               (*slot_idx + num_wqes * wqe_size) % qp->tx_wq.sq_depth);

    // 3. Reserve msn and psn idx
    *msn = atomicAdd((unsigned long long int *)&mvars->tx_wq.msn_idx, 0);
    atomicExch((unsigned long long int *)&mvars->tx_wq.msn_idx,
               (*msn + num_msn) % qp->msn_tbl_sz);
    *psn = atomicAdd((unsigned long long int *)&mvars->tx_wq.psn, num_pkts);

    global_lock_release<NVSHMEMI_THREADGROUP_THREAD>(&qp->mvars.resv_lock);

    // If last slot is available, all prior slots are also available.
    ibgda_wait_for_slot_availability(qp, wqe_idx + num_wqes);
    return wqe_idx;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t
ibgda_reserve_ibuf_slots(nvshmemi_ibgda_device_qp_t *qp, unsigned long long int num_slots) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    uint32_t nslots = qp->ibuf.nslots;
    uint64_t base_idx = atomicAdd((unsigned long long int *)&mvars->ibuf.head, num_slots);
    uint64_t idx = base_idx + num_slots;

    // Wait until the slots become available.
    while (idx - ibgda_atomic_read(&mvars->ibuf.tail) > nslots)
        ;

    // Prevent the reordering of the above wait loop.
    IBGDA_MFENCE();

    return base_idx;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_release_ibuf(
    nvshmemi_ibgda_device_qp_t *qp, unsigned long long int base_idx,
    unsigned long long int num_slots) {
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;
    unsigned long long int new_idx = base_idx + num_slots;
    IBGDA_MFENCE();
    // Wait here.
    while (atomicCAS((unsigned long long int *)&mvars->ibuf.tail, (unsigned long long int)base_idx,
                     new_idx) != base_idx)
        ;
    IBGDA_MFENCE();
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t
ibgda_get_ibuf_addr(nvshmemi_ibgda_device_qp_t *qp, uint64_t idx) {
    idx = idx & (qp->ibuf.nslots - 1);

    // buf[0] is reserved for non-fetch operations
    return (uint64_t)qp->ibuf.buf + NVSHMEMI_IBGDA_IBUF_SLOT_SIZE * (idx + 1);
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE bool ibgda_can_coalesce_warp(
    unsigned int amask, nvshmemi_ibgda_device_qp_t *qp) {
    int pred_same_qp;

    if (amask != IBGDA_FULL_WARP) return false;

    __match_all_sync(amask, qp->qpn, &pred_same_qp);
    if (!pred_same_qp) return false;

    return true;
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE bool ibgda_can_coalesce_warp_pe(
    unsigned int amask, int pe) {
    int pred_same_pe;

    if (amask != IBGDA_FULL_WARP) return false;

    __match_all_sync(amask, pe, &pred_same_pe);
    if (!pred_same_pe) return false;

    return true;
}

// CST is used for GPUs which do not natively support ordering of remote writes
// This means it'll need additional sync ops
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t
ibgda_cst(nvshmemi_ibgda_device_qp_t *qp, bool is_qp_shared_among_ctas) {
    assert(likely(qp->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC));

    uint64_t laddr = (uint64_t)qp->ibuf.buf;
    __be32 lkey = qp->ibuf.lkey;
    const int num_wqes = 1;
    const int num_slots_per_wqe = 3;

    uint64_t base_wqe_idx, base_slot_idx, base_msn_idx, base_psn;
    base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas,
                                           num_slots_per_wqe, 1, 1,
                                           &base_slot_idx, &base_msn_idx, &base_psn);

    // DUMP OP causes the NIC to read laddr, which is always on GPU memory.
    // For CST, it is cheaper than RDMA READ.
    ibgda_write_dump_wqe(qp, laddr, lkey, 0, base_slot_idx,
                         BNXT_RE_WR_FLAGS_RD_FENCE);

    // Don't update get_head here because this is internal cst
    if (is_qp_shared_among_ctas)
        ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);
    else
        ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);

    return ibgda_quiet(qp);
}

__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE uint64_t
ibgda_quiet_with_cst(nvshmemi_ibgda_device_qp_t *qp, bool is_qp_shared_among_ctas) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    nvshmemi_ibgda_device_qp_management_t *mvars = &qp->mvars;

    uint64_t get_head;
    uint64_t ticket;
    uint64_t get_tail;

    if (state->may_skip_cst) {
        ticket = ibgda_quiet(qp);
    } else {
        // We want to read get_head before calling ibgda_quiet. Thus, ticket =
        // ibgda_quiet(qp) cannot be combined.
        get_head = ibgda_atomic_read(&mvars->tx_wq.get_head);
        ticket = ibgda_quiet(qp);
        get_tail = ibgda_atomic_read(&mvars->tx_wq.get_tail);

        // TODO: Change to WAIT + DUMP
        // In that case, we don't have to do quiet first
        if (get_tail < get_head) {
            bool is_qp_shared_among_ctas;
            nvshmemi_ibgda_device_qp_t *myqp =
                ibgda_get_qp(nvshmemi_device_state_d.mype, &is_qp_shared_among_ctas);
            uint64_t cst_ticket = ibgda_cst(myqp, is_qp_shared_among_ctas);
            ibgda_update_get_tail(myqp, cst_ticket);
            ibgda_update_get_tail(qp, ticket);
        }
    }

    return ticket;
}

template <nvshmemi_op_t channel_op, bool nbi, bool support_half_av_seg>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_rma_thread(
    uint64_t rptr, uint64_t lptr, size_t remaining_size, int dst_pe, int proxy_pe) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    unsigned int amask = __activemask();
    bool can_coalesce_warp = ibgda_can_coalesce_warp_pe(amask, proxy_pe);
    int my_tid;
    int tg_size;

    const bool need_cst = (channel_op == NVSHMEMI_OP_GET) && !state->may_skip_cst;
    const bool need_immediate_cst = !nbi && need_cst;

    int is_qp_shared_among_ctas;
    nvshmemi_ibgda_device_qp_t *qp;

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = ibgda_get_qp(proxy_pe, (bool *)&is_qp_shared_among_ctas);
        }
        qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
        is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);
    } else {
        qp = ibgda_get_qp(proxy_pe, (bool *)&is_qp_shared_among_ctas);
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
    }

    const bool need_additional_wqe = need_immediate_cst;
    int num_wqes_per_cmd = 1;
    int num_slots_per_wqe = 3;

    bool did_quiet = false;

    if (unlikely(remaining_size == 0)) return;

    while (remaining_size > 0) {
        amask = __activemask();

        bool is_data_buf_in_sysmem;

        __be32 lkey;
        size_t lchunk_size;
        ibgda_get_lkey(lptr, &lkey, &lchunk_size, &is_data_buf_in_sysmem, qp->dev_idx);

        __be32 rkey;
        uint64_t raddr;
        size_t rchunk_size;
        ibgda_get_raddr_rkey(rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);

        size_t transfer_size = ibgda_cal_transfer_size(remaining_size, lchunk_size, rchunk_size);

        can_coalesce_warp = ibgda_can_coalesce_warp(amask, qp);
        if (can_coalesce_warp) {
            my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
            tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        } else {
            my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
            tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        }

        int num_wqes = num_wqes_per_cmd * tg_size + (need_additional_wqe ? 1 : 0);
        int total_msn = num_wqes_per_cmd * tg_size;
        int ppw = bnxt_re_get_pkts_per_wqe(qp, transfer_size);
        int total_pkts = total_msn * ppw;

        uint64_t base_wqe_idx, base_slot_idx, base_msn_idx, base_psn;

        if (my_tid == 0) {
            base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas,
                                            num_slots_per_wqe, total_msn, total_pkts,
                                            &base_slot_idx, &base_msn_idx, &base_psn);
        }
        if (can_coalesce_warp) {
            base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
            base_slot_idx = __shfl_sync(amask, base_slot_idx, 0);
            base_msn_idx = __shfl_sync(amask, base_msn_idx, 0);
            base_psn = __shfl_sync(amask, base_psn, 0);
        }

        uint64_t my_slot_idx = base_slot_idx + (my_tid * num_wqes_per_cmd * num_slots_per_wqe);
        uint64_t my_msn_idx = base_msn_idx + (my_tid * num_wqes_per_cmd);
        uint64_t my_psn = base_psn + (my_tid * num_wqes_per_cmd * ppw);

        // Generate CQE only if we create the last WQE in the group.
        // TBD - Check this
        uint8_t fm_ce_se = 0;

        switch (channel_op) {
            case NVSHMEMI_OP_PUT:
                ibgda_write_rdma_write_wqe<support_half_av_seg>(qp, lptr, lkey, raddr, rkey,
                                                    transfer_size, my_slot_idx, my_msn_idx,
                                                    my_psn, ppw, fm_ce_se);
                break;
            case NVSHMEMI_OP_GET:
                ibgda_write_rdma_read_wqe<support_half_av_seg>(qp, lptr, lkey, raddr, rkey,
                                                               transfer_size, my_slot_idx, fm_ce_se);
                break;
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unsupported channel_op.\n");
#endif
                assert(0);
        }

        if (can_coalesce_warp) {
            nvshmemi_warp_sync();
        }

        if (my_tid == tg_size - 1) {
            if (need_immediate_cst) {
                // Enqueue CST op in the QP.  This command has NIC Fence, which
                // waits for all prior READ/ATOMIC to finish before issuing this
                // DUMP.
                my_slot_idx += num_wqes_per_cmd * num_slots_per_wqe;
                ibgda_write_dump_wqe(qp, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char),
                                     my_slot_idx, 2 << 5);
            } else {
                if (need_additional_wqe) {
                    my_slot_idx += num_wqes_per_cmd * num_slots_per_wqe;
                    ibgda_write_nop_wqe(qp, my_slot_idx);
                }

                if (need_cst) {
                    // For nbi, we will do CST in QUIET.
                    // GET index must be visible before the new cons index.
                    ibgda_update_get_head(qp, base_wqe_idx + num_wqes);
                }
            }

            // Require membar.sys to push data buffer to the point of consistency.
            if (channel_op == NVSHMEMI_OP_PUT && is_data_buf_in_sysmem) __threadfence_system();

            if (is_qp_shared_among_ctas)
                ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);
            else
                ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);
        }

        remaining_size -= transfer_size;

        rptr += transfer_size;
        lptr += transfer_size;

        if (can_coalesce_warp) {
            if (!nbi) {
                bool do_coalesce_quiet = __all_sync(amask, remaining_size == 0);
                if (do_coalesce_quiet && my_tid == tg_size - 1) {
                    // CST, if required, has already been enqueued. We simply need to
                    // do ibgda_quiet here.
                    ibgda_quiet(qp);
                }
                did_quiet |= do_coalesce_quiet;
            }
            nvshmemi_warp_sync();
        }
    }

    if (!nbi && !did_quiet) {
        // CST, if required, has already been enqueued. We simply need to
        // do ibgda_quiet here.
        ibgda_quiet(qp);
    }
}

#if __cplusplus >= 201103L
static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64,
              "static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64) failed");
#endif
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op, bool nbi, bool support_half_av_seg>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void ibgda_rma(uint64_t req_rptr,
                                                                        uint64_t req_lptr,
                                                                        size_t bytes, int dst_pe,
                                                                        int proxy_pe) {
    assert(SCOPE == NVSHMEMI_THREADGROUP_WARP || SCOPE == NVSHMEMI_THREADGROUP_BLOCK);

    // Use only warp 0
    int my_tid = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    const bool need_cst = (channel_op == NVSHMEMI_OP_GET) && !state->may_skip_cst;
    const bool need_immediate_cst = !nbi && need_cst;
    bool need_additional_wqe;

    int is_qp_shared_among_ctas = 0;
    nvshmemi_ibgda_device_qp_t *qp;

    int num_slots_per_wqe = 0;
    int num_wqes;
    int num_wqes_per_cmd;
    int ppw, total_msn, total_pkts;

    uint64_t base_wqe_idx, base_slot_idx, base_msn_idx, base_psn;
    uint64_t my_slot_idx, my_msn_idx, my_psn;

    size_t remaining_size = bytes;

    size_t transfer_size;
    size_t my_transfer_size = 0;

    uint64_t rptr = req_rptr;
    uint64_t lptr = req_lptr;

    __be32 lkey;
    __be32 my_lkey = 0;
    uint64_t my_laddr;
    size_t lchunk_size;

    __be32 rkey;
    __be32 my_rkey = 0;
    uint64_t raddr;
    uint64_t my_raddr;
    size_t rchunk_size;

    int chunk_idx = 0;

    bool is_data_buf_in_sysmem;

    uint8_t fm_ce_se;

    if (unlikely(remaining_size == 0)) goto out;

    // Not warp 0, wait at the exit.
    if (my_tid >= tg_size) {
        goto out;
    }
    my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();

    if (my_tid == 0) {
        qp = ibgda_get_qp(proxy_pe, (bool *)&is_qp_shared_among_ctas);
    }
    qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
    is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);

    need_additional_wqe = need_immediate_cst;

    num_wqes_per_cmd = 1;
    num_slots_per_wqe = 3;

    // Calculate how many chunks we need to send.
    while (remaining_size > 0) {
        ibgda_get_lkey(lptr, &lkey, &lchunk_size, &is_data_buf_in_sysmem, qp->dev_idx);
        ibgda_get_raddr_rkey(rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);
        transfer_size = ibgda_cal_transfer_size(remaining_size, lchunk_size, rchunk_size);
        if (my_tid == chunk_idx) {
            my_lkey = lkey;
            my_laddr = lptr;
            my_rkey = rkey;
            my_raddr = raddr;
            my_transfer_size = transfer_size;
        }

        remaining_size -= transfer_size;
        rptr += transfer_size;
        lptr += transfer_size;

        ++chunk_idx;
    }

    // Too many chunks. Use ibgda_rma_thread to handle it instead.
    if (unlikely(chunk_idx > tg_size)) {
        if (my_tid == 0) {
            ibgda_rma_thread<channel_op, nbi, support_half_av_seg>(req_rptr, req_lptr, bytes,
                                                                   dst_pe, proxy_pe);
        }
        goto out;
    }

    num_wqes = num_wqes_per_cmd * chunk_idx + (need_additional_wqe ? 1 : 0);
    ppw = bnxt_re_get_pkts_per_wqe(qp, transfer_size);
    total_msn = num_wqes_per_cmd * chunk_idx;
    total_pkts = total_msn * ppw;

    if (my_tid == 0) {
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas,
                                            num_slots_per_wqe, total_msn, total_pkts,
                                            &base_slot_idx, &base_msn_idx, &base_psn);
    }

    base_wqe_idx = __shfl_sync(IBGDA_FULL_WARP, base_wqe_idx, 0);
    base_slot_idx = __shfl_sync(IBGDA_FULL_WARP, base_slot_idx, 0);
    base_msn_idx = __shfl_sync(IBGDA_FULL_WARP, base_msn_idx, 0);
    base_psn = __shfl_sync(IBGDA_FULL_WARP, base_psn, 0);
    my_slot_idx = base_slot_idx + (my_tid * num_wqes_per_cmd * num_slots_per_wqe);
    my_msn_idx = base_msn_idx + (my_tid * num_wqes_per_cmd);
    my_psn = base_psn + (my_tid * num_wqes_per_cmd * ppw);

    // Generate CQE only if we create the last WQE in the group.
    fm_ce_se = 0;

    if (my_tid < chunk_idx) {
        switch (channel_op) {
            case NVSHMEMI_OP_PUT:
                ibgda_write_rdma_write_wqe<support_half_av_seg>(qp, my_laddr, my_lkey,
                                                  my_raddr, my_rkey, my_transfer_size,
                                                  my_slot_idx, my_msn_idx, my_psn,
                                                  ppw, fm_ce_se);
                break;
            case NVSHMEMI_OP_GET:
                ibgda_write_rdma_read_wqe<support_half_av_seg>(qp, my_laddr, my_lkey, my_raddr,
                                                               my_rkey, my_transfer_size,
                                                               my_slot_idx, fm_ce_se);
                break;
            default:
#ifdef NVSHMEM_IBGDA_DEBUG
                printf("Unsupported channel_op.\n");
#endif
                assert(0);
        }
    }

    nvshmemi_warp_sync();

    if (my_tid == chunk_idx - 1) {
        if (need_immediate_cst) {
            my_slot_idx += num_wqes_per_cmd * num_slots_per_wqe;
            // Enqueue CST op in the QP.  This command has NIC Fence, which
            // waits for all prior READ/ATOMIC to finish before issuing this
            // DUMP.
            ibgda_write_dump_wqe(qp, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char),
                                 my_slot_idx, BNXT_RE_WR_FLAGS_RD_FENCE);
        } else {
            if (need_additional_wqe) {
                my_slot_idx += num_wqes_per_cmd * num_slots_per_wqe;
                ibgda_write_nop_wqe(qp, my_slot_idx);
            }

            if (need_cst) {
                // For nbi, we will do CST in QUIET.
                // GET index must be visible before the new cons index.
                // ibgda_submit_requests has fence, which guarantees the ordering.
                ibgda_update_get_head(qp, base_wqe_idx + num_wqes);
            }
        }

        // Require membar.sys to push data buffer to the point of consistency.
        if (channel_op == NVSHMEMI_OP_PUT && is_data_buf_in_sysmem) __threadfence_system();

        if (is_qp_shared_among_ctas)
            ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);
        else
            ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);

        if (!nbi) {
            // CST, if required, has already been enqueued. We simply need to
            // do ibgda_quiet here.
            ibgda_quiet(qp);
        }
    }

out:
    nvshmemi_threadgroup_sync<SCOPE>();
}

/**
 * RMA P base
 */
#if __cplusplus >= 201103L
static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64,
              "static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64) failed");
#endif
template <typename T, bool is_full_warp, bool can_combine_data, bool support_half_av_seg>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_ibgda_rma_p_impl(
    void *rptr, const T value, int dst_pe) {
    static_assert((can_combine_data && is_full_warp) || (!can_combine_data),
                  "can_combine_data check 1 failed.\n");
    static_assert((can_combine_data && support_half_av_seg) || (!can_combine_data),
                  "can_combine_data check 2 failed.\n");

    int my_tid;
    int tg_size;
    int proxy_pe = ibgda_get_proxy_pe(dst_pe);
    int is_qp_shared_among_ctas;
    nvshmemi_ibgda_device_qp_t *qp;

    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    if (is_full_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = ibgda_get_qp(proxy_pe, (bool *)&is_qp_shared_among_ctas);
        }
        qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
        is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);
    } else {
        qp = ibgda_get_qp(proxy_pe, (bool *)&is_qp_shared_among_ctas);
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
    }

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;
    ibgda_get_raddr_rkey((uint64_t)rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size,
                         qp->dev_idx);

    // With proper alignment (requirement of NVSHMEM), one element cannot span multiple chunks.
    assert(rchunk_size >= sizeof(T));

    int num_slots_per_wqe = 3;
    int num_wqes_per_cmd;
    int num_wqes;
    int total_msn, total_pkts;

    bool need_additional_wqe = false;

    if (can_combine_data) {
        num_wqes_per_cmd =
            ibgda_get_num_wqes_in_inl_combine_warp<T, NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC>();
        num_wqes = num_wqes_per_cmd;
    } else {
        num_wqes_per_cmd = 1;
        num_wqes = num_wqes_per_cmd * tg_size;
    }
    total_msn = num_wqes;
    total_pkts = total_msn * num_wqes_per_cmd; // 1 pkt per inline wqe

    // additional nop wqe doesn't need MSN entry
    if (!can_combine_data && num_wqes_per_cmd > 1) {
        ++num_wqes;
        need_additional_wqe = true;
    }

    uint64_t base_wqe_idx = 0, base_slot_idx = 0, base_msn_idx = 0, base_psn = 0;

    if (my_tid == 0) {
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas,
                                               num_slots_per_wqe, total_msn, total_pkts,
                                               &base_slot_idx, &base_msn_idx, &base_psn);
    }

    if (is_full_warp) {
        // Sync the following base variables for all threads within a WARP
        base_wqe_idx = __shfl_sync(IBGDA_FULL_WARP, base_wqe_idx, 0);
        base_slot_idx = __shfl_sync(IBGDA_FULL_WARP, base_slot_idx, 0);
        base_msn_idx = __shfl_sync(IBGDA_FULL_WARP, base_msn_idx, 0);
        base_psn = __shfl_sync(IBGDA_FULL_WARP, base_psn, 0);
    }

    // Generate CQE only if we create the last WQE in the group.
    uint8_t fm_ce_se = 0;

    uint64_t my_slot_idx = base_slot_idx + (my_tid * num_wqes_per_cmd * num_slots_per_wqe);
    uint64_t my_msn_idx = base_msn_idx + (my_tid * num_wqes_per_cmd);
    uint64_t my_psn = base_psn + (my_tid * num_wqes_per_cmd * 1);   // 1 pkt per wqe

    if (can_combine_data)
        ibgda_write_rdma_write_inl_wqe_combine_warp<T>(qp, value, raddr, rkey, my_slot_idx,
                                                       my_tid);
    else
        ibgda_write_rdma_write_inl_wqe<support_half_av_seg>(qp, &value, raddr, rkey, sizeof(T),
                                                    my_slot_idx, my_msn_idx, my_psn, fm_ce_se);

    if (is_full_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) {
        if (need_additional_wqe) {
            my_slot_idx += num_wqes_per_cmd * num_slots_per_wqe;
            ibgda_write_nop_wqe(qp, my_slot_idx);
        }

        if (is_qp_shared_among_ctas)
            ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);
        else
            ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);
    }

    // TBD - Review and conclude later.
    ibgda_quiet(qp);

    if (is_full_warp) nvshmemi_warp_sync();
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_ibgda_rma_p(void *rptr, const T value,
                                                                   int dst_pe) {
    unsigned int amask = __activemask();
    bool can_combine_data = false;
    int pred_pe = 0;
    int pred_contiguous = 0;
    int pred_rkey = 0;
    int my_tid;

    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    if (amask == IBGDA_FULL_WARP) {
        /* TODO: Adding multi-dev support could have caused a regression with coalescing. */
        nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
        __be32 rkey;
        uint64_t raddr;
        size_t rchunk_size;
        int proxy_pe = ibgda_get_proxy_pe(dst_pe);
        ibgda_get_raddr_rkey((uint64_t)rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size, 0);
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        __match_all_sync(IBGDA_FULL_WARP, dst_pe, &pred_pe);
        __match_all_sync(IBGDA_FULL_WARP, (uintptr_t)(rptr) - (my_tid * sizeof(T)),
                         &pred_contiguous);
        __match_all_sync(IBGDA_FULL_WARP, rkey, &pred_rkey);
        can_combine_data = (pred_pe && pred_contiguous && pred_rkey && state->support_half_av_seg);

        if (can_combine_data)
            nvshmemi_ibgda_rma_p_impl<T, true, true, true>(rptr, value, dst_pe);
        else if (state->support_half_av_seg)
            nvshmemi_ibgda_rma_p_impl<T, true, false, true>(rptr, value, dst_pe);
        else
            nvshmemi_ibgda_rma_p_impl<T, true, false, false>(rptr, value, dst_pe);
    } else if (state->support_half_av_seg)
        nvshmemi_ibgda_rma_p_impl<T, false, false, true>(rptr, value, dst_pe);
    else
        nvshmemi_ibgda_rma_p_impl<T, false, false, false>(rptr, value, dst_pe);
}

/**
 * RMA G base
 */
template <typename T, bool support_half_av_seg>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T nvshmemi_ibgda_rma_g_impl(void *rptr, int dst_pe,
                                                                     int proxy_pe) {
    assert(0);
    return false;
#if 0
    unsigned int amask = __activemask();
    int my_tid;
    int tg_size;

    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    const bool need_cst = !state->may_skip_cst;

    uint64_t base_wqe_idx;
    uint64_t base_ibuf_idx;

    T ret;

    int is_qp_shared_among_ctas;
    nvshmemi_ibgda_device_qp_t *qp;

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;

    bool can_coalesce_warp = ibgda_can_coalesce_warp_pe(amask, proxy_pe);
    bool can_combine_data = false;
    int pred_contiguous = 0;
    int pred_rkey = 0;

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = ibgda_get_qp(proxy_pe, (bool *)&is_qp_shared_among_ctas);
        }
        qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
        is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);
        ibgda_get_raddr_rkey((uint64_t)rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size,
                             qp->dev_idx);

        __match_all_sync(IBGDA_FULL_WARP, (uintptr_t)(rptr) - (my_tid * sizeof(T)),
                         &pred_contiguous);
        __match_all_sync(IBGDA_FULL_WARP, rkey, &pred_rkey);
        can_combine_data = (pred_contiguous && pred_rkey);
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        qp = ibgda_get_qp(proxy_pe, (bool *)&is_qp_shared_among_ctas);
        ibgda_get_raddr_rkey((uint64_t)rptr, dst_pe, proxy_pe, &raddr, &rkey, &rchunk_size,
                             qp->dev_idx);
    }

    const bool need_additional_wqe = need_cst;

    int num_wqes_per_cmd = 1;

    int num_wqes = (can_combine_data ? num_wqes_per_cmd : num_wqes_per_cmd * tg_size) +
                   (need_additional_wqe ? 1 : 0);

    int num_ibuf_slots = can_coalesce_warp ? 1 : tg_size;

    if (my_tid == 0) {
        base_ibuf_idx = ibgda_reserve_ibuf_slots(qp, num_ibuf_slots);
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas);
    }

    if (can_coalesce_warp) {
        base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
        base_ibuf_idx = __shfl_sync(amask, base_ibuf_idx, 0);
    }

    uint64_t my_wqe_idx =
        can_combine_data ? base_wqe_idx : base_wqe_idx + (my_tid * num_wqes_per_cmd);
    uint64_t my_ibuf_idx = can_coalesce_warp ? base_ibuf_idx : base_ibuf_idx + my_tid;

    void *wqe_ptrs[2];
    wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
    wqe_ptrs[1] = ibgda_get_wqe_ptr(qp, my_wqe_idx + 1);

    uint64_t laddr =
        ibgda_get_ibuf_addr(qp, my_ibuf_idx) + (can_coalesce_warp ? my_tid * sizeof(T) : 0);
    __be32 lkey = qp->ibuf.lkey;

    // Generate CQE only if we create the last WQE in the group.
    uint8_t fm_ce_se = (!need_additional_wqe && ((can_combine_data && (my_tid == 0)) ||
                                                 (!can_combine_data && (my_tid == tg_size - 1))))
                           ? BNXT_RE_WR_FLAGS_SIGNALED
                           : 0;

    if (!can_combine_data) {
        ibgda_write_rdma_read_wqe<support_half_av_seg>(qp, laddr, lkey, raddr, rkey, sizeof(T),
                                                       my_wqe_idx, fm_ce_se, wqe_ptrs);

    } else if (my_tid == 0) {
        ibgda_write_rdma_read_wqe<support_half_av_seg>(
            qp, laddr, lkey, raddr, rkey, sizeof(T) * tg_size, my_wqe_idx, fm_ce_se, wqe_ptrs);
    }

    if (can_coalesce_warp) nvshmemi_warp_sync();

    if (need_additional_wqe && (my_tid == (tg_size - 1))) {
        my_wqe_idx += num_wqes_per_cmd;
        wqe_ptrs[0] = ibgda_get_wqe_ptr(qp, my_wqe_idx);
        fm_ce_se = BNXT_RE_WR_FLAGS_SIGNALED;

        if (need_cst)
            // Enqueue CST op in the QP.  This command has NIC Fence, which
            // waits for all prior READ/ATOMIC to finish before issuing this
            // DUMP.
            ibgda_write_dump_wqe(qp, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char),
                                 my_wqe_idx, BNXT_RE_WR_FLAGS_RD_FENCE, wqe_ptrs);
        else
            ibgda_write_nop_wqe(qp, my_wqe_idx, wqe_ptrs);
    }
    if (fm_ce_se > 0) {
        if (is_qp_shared_among_ctas)
            ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes);
        else
            ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes);

        ibgda_quiet(qp);
    }

    if (can_coalesce_warp) nvshmemi_warp_sync();

    ret = READ_ONCE(*(T *)laddr);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) ibgda_release_ibuf(qp, base_ibuf_idx, num_ibuf_slots);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    return ret;
#endif
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T nvshmemi_ibgda_rma_g(void *rptr, int dst_pe) {
    T ret;
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    int proxy_pe = ibgda_get_proxy_pe(dst_pe);

    if (state->support_half_av_seg)
        ret = nvshmemi_ibgda_rma_g_impl<T, true>(rptr, dst_pe, proxy_pe);
    else
        ret = nvshmemi_ibgda_rma_g_impl<T, false>(rptr, dst_pe, proxy_pe);
    return ret;
}

/**
 * RMA NBI base
 */
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_ibgda_rma_nbi(void *rptr, void *lptr,
                                                                     size_t bytes, int dst_pe) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    int proxy_pe = ibgda_get_proxy_pe(dst_pe);
    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        if (state->support_half_av_seg) {
            ibgda_rma_thread<channel_op, true, true>((uint64_t)rptr, (uint64_t)lptr, bytes, dst_pe,
                                                     proxy_pe);
        } else {
            ibgda_rma_thread<channel_op, true, false>((uint64_t)rptr, (uint64_t)lptr, bytes, dst_pe,
                                                      proxy_pe);
        }
    } else {
        if (state->support_half_av_seg) {
            ibgda_rma<SCOPE, channel_op, true, true>((uint64_t)rptr, (uint64_t)lptr, bytes, dst_pe,
                                                     proxy_pe);
        } else {
            ibgda_rma<SCOPE, channel_op, true, false>((uint64_t)rptr, (uint64_t)lptr, bytes, dst_pe,
                                                      proxy_pe);
        }
    }
}

/**
 * RMA (blocking) base
 */
template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_ibgda_rma(void *rptr, void *lptr,
                                                                 size_t bytes, int dst_pe) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    int proxy_pe = ibgda_get_proxy_pe(dst_pe);
    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        if (state->support_half_av_seg) {
            ibgda_rma_thread<channel_op, false, true>((uint64_t)rptr, (uint64_t)lptr, bytes, dst_pe,
                                                      proxy_pe);
        } else {
            ibgda_rma_thread<channel_op, false, false>((uint64_t)rptr, (uint64_t)lptr, bytes,
                                                       dst_pe, proxy_pe);
        }
    } else {
        if (state->support_half_av_seg) {
            ibgda_rma<SCOPE, channel_op, false, true>((uint64_t)rptr, (uint64_t)lptr, bytes, dst_pe,
                                                      proxy_pe);
        } else {
            ibgda_rma<SCOPE, channel_op, false, false>((uint64_t)rptr, (uint64_t)lptr, bytes,
                                                       dst_pe, proxy_pe);
        }
    }
}

/**
 * AMO non-fetch base
 */
template <typename T, bool support_half_av_seg>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_ibgda_amo_nonfetch_impl(void *rptr,
                                                                               const T value,
                                                                               int pe,
                                                                               nvshmemi_amo_t op) {
    unsigned int amask = __activemask();
    int my_tid;
    int tg_size;

    int is_qp_shared_among_ctas;
    nvshmemi_ibgda_device_qp_t *qp;

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;

    bool can_coalesce_warp = ibgda_can_coalesce_warp_pe(amask, pe);

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = ibgda_get_qp(pe, (bool *)&is_qp_shared_among_ctas);
        }
        qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
        is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        qp = ibgda_get_qp(pe, (bool *)&is_qp_shared_among_ctas);
    }
    ibgda_get_raddr_rkey((uint64_t)rptr, pe, pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);

    int num_wqes_per_cmd = ibgda_get_num_wqes_in_atomic<T>(op, qp->qp_type);

    const bool need_additional_wqe = (num_wqes_per_cmd > 1);

    int num_wqes = num_wqes_per_cmd * tg_size + (need_additional_wqe ? 1 : 0);
    int num_slots_per_wqe = 3;

    uint64_t base_wqe_idx, base_slot_idx, base_msn_idx, base_psn;

    if (my_tid == 0) {
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas,
                                               num_slots_per_wqe, 1, &base_slot_idx,
                                               &base_msn_idx, &base_psn);
    }

    if (can_coalesce_warp) {
        base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
        base_slot_idx = __shfl_sync(amask, base_slot_idx, 0);
        base_msn_idx = __shfl_sync(amask, base_msn_idx, 0);
        base_psn = __shfl_sync(amask, base_psn, 0);
    }

    uint64_t my_slot_idx = base_slot_idx + (my_tid * num_wqes_per_cmd * num_slots_per_wqe);
    uint64_t my_msn_idx = base_msn_idx + (my_tid * num_wqes_per_cmd);
    uint64_t my_psn = base_slot_idx + (my_tid * num_wqes_per_cmd * 1);  // 1 pkt per wqe
    uint8_t fm_ce_se = 0;

    ibgda_write_atomic_wqe<support_half_av_seg>(qp, &value, NULL, (uint64_t)qp->ibuf.buf,
                                                qp->ibuf.lkey, raddr, rkey, sizeof(T), my_slot_idx,
                                                op, fm_ce_se);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) {
        if (need_additional_wqe) {
            my_slot_idx += num_wqes_per_cmd * num_slots_per_wqe;
            ibgda_write_nop_wqe(qp, my_slot_idx);
        }

        if (is_qp_shared_among_ctas)
            ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);
        else
            ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);
    }

    if (can_coalesce_warp) nvshmemi_warp_sync();
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_ibgda_amo_nonfetch(void *rptr, const T value,
                                                                          int pe,
                                                                          nvshmemi_amo_t op) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    if (state->support_half_av_seg)
        nvshmemi_ibgda_amo_nonfetch_impl<T, true>(rptr, value, pe, op);
    else
        nvshmemi_ibgda_amo_nonfetch_impl<T, false>(rptr, value, pe, op);
}

/**
 * AMO fetch base
 */
template <typename T, bool support_half_av_seg>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T nvshmemi_ibgda_amo_fetch_impl(void *rptr, const T value,
                                                                         const T compare, int pe,
                                                                         nvshmemi_amo_t op) {
    unsigned int amask = __activemask();
    int my_tid;
    int tg_size;

    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    const bool need_cst = !state->may_skip_cst;

    T ret;

    int is_qp_shared_among_ctas;
    nvshmemi_ibgda_device_qp_t *qp;

    __be32 rkey;
    uint64_t raddr;
    size_t rchunk_size;

    bool can_coalesce_warp = ibgda_can_coalesce_warp_pe(amask, pe);

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = ibgda_get_qp(pe, (bool *)&is_qp_shared_among_ctas);
        }
        qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
        is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        qp = ibgda_get_qp(pe, (bool *)&is_qp_shared_among_ctas);
    }
    ibgda_get_raddr_rkey((uint64_t)rptr, pe, pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);

    int num_wqes_per_cmd = ibgda_get_num_wqes_in_atomic<T>(op, qp->qp_type);

    const bool need_additional_wqe = (num_wqes_per_cmd > 1) || need_cst;

    int num_wqes = num_wqes_per_cmd * tg_size + (need_additional_wqe ? 1 : 0);
    int total_msn = num_wqes_per_cmd * tg_size;
    int ppw = 1;    // 1 pkt per atomic wqe
    int total_pkts = total_msn * ppw;
    int num_slots_per_wqe = 3;

    uint64_t base_ibuf_idx;
    uint64_t base_wqe_idx, base_slot_idx, base_msn_idx, base_psn;

    if (my_tid == 0) {
        base_ibuf_idx = ibgda_reserve_ibuf_slots(qp, tg_size);
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas,
                                            num_slots_per_wqe, total_msn, total_pkts,
                                            &base_slot_idx, &base_msn_idx, &base_psn);
    }

    if (can_coalesce_warp) {
        base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
        base_ibuf_idx = __shfl_sync(amask, base_ibuf_idx, 0);
        base_slot_idx = __shfl_sync(amask, base_slot_idx, 0);
        base_msn_idx = __shfl_sync(amask, base_msn_idx, 0);
        base_psn = __shfl_sync(amask, base_psn, 0);
    }

    uint64_t my_ibuf_idx = base_ibuf_idx + my_tid;
    uint64_t my_slot_idx = base_slot_idx + (my_tid * num_wqes_per_cmd * num_slots_per_wqe);
    uint64_t laddr = ibgda_get_ibuf_addr(qp, my_ibuf_idx);
    __be32 lkey = qp->ibuf.lkey;
    uint8_t fm_ce_se = 0;

    ibgda_write_atomic_wqe<support_half_av_seg>(qp, &value, &compare, laddr, lkey, raddr, rkey,
                                                sizeof(T), my_slot_idx, op, fm_ce_se);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) {
        if (need_additional_wqe) {
            my_slot_idx += num_wqes_per_cmd * num_slots_per_wqe;
            if (need_cst)
                // Enqueue CST op in the QP.  This command has NIC Fence, which
                // waits for all prior READ/ATOMIC to finish before issuing this
                // DUMP.
                ibgda_write_dump_wqe(qp, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sizeof(char),
                                     my_slot_idx, 2 << 5);
            else
                ibgda_write_nop_wqe(qp, my_slot_idx);
        }

        if (is_qp_shared_among_ctas)
            ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);
        else
            ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);

        ibgda_quiet(qp);
    }

    if (can_coalesce_warp) nvshmemi_warp_sync();

    ret = READ_ONCE(*(T *)laddr);
    if (sizeof(T) == 4) ret = BSWAP32((uint32_t)ret);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    if (my_tid == tg_size - 1) ibgda_release_ibuf(qp, base_ibuf_idx, tg_size);

    if (can_coalesce_warp) nvshmemi_warp_sync();

    return ret;
}

template <typename T>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE T nvshmemi_ibgda_amo_fetch(void *rptr, const T value,
                                                                    const T compare, int pe,
                                                                    nvshmemi_amo_t op) {
    T ret;
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    if (state->support_half_av_seg)
        ret = nvshmemi_ibgda_amo_fetch_impl<T, true>(rptr, value, compare, pe, op);
    else
        ret = nvshmemi_ibgda_amo_fetch_impl<T, false>(rptr, value, compare, pe, op);
    return ret;
}

#if __cplusplus >= 201103L
static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 128,
              "static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 128) failed");
#endif
template <bool is_nbi, bool support_half_av_seg>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_ibgda_put_signal_thread_impl(
    void *rptr, void *lptr, size_t bytes, void *sig_rptr, uint64_t signal, nvshmemi_amo_t sig_op,
    int pe) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    nvshmemi_ibgda_device_qp_t *qp;
    size_t lchunk_size;
    size_t rchunk_size;
    size_t sig_rchunk_size;
    uint64_t sig_raddr;
    uint64_t raddr;

    unsigned int amask = __activemask();
    int my_tid;
    int tg_size;
    __be32 lkey;
    __be32 rkey;
    __be32 sig_rkey;

    bool can_coalesce_warp = ibgda_can_coalesce_warp_pe(amask, pe);
    int is_qp_shared_among_ctas;
    bool is_data_buf_in_sysmem;

    if (can_coalesce_warp) {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        if (my_tid == 0) {
            qp = ibgda_get_qp(pe, (bool *)&is_qp_shared_among_ctas);
        }
        qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
        is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);
    } else {
        my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
        tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        qp = ibgda_get_qp(pe, (bool *)&is_qp_shared_among_ctas);
    }
    ibgda_get_lkey((uint64_t)lptr, &lkey, &lchunk_size, &is_data_buf_in_sysmem, qp->dev_idx);
    ibgda_get_raddr_rkey((uint64_t)rptr, pe, pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);
    ibgda_get_raddr_rkey((uint64_t)sig_rptr, pe, pe, &sig_raddr, &sig_rkey, &sig_rchunk_size,
                         qp->dev_idx);

    const int num_atomic_wqes_per_cmd = ibgda_get_num_wqes_in_atomic<uint64_t>(sig_op, qp->qp_type);
    const bool need_additional_wqe = (num_atomic_wqes_per_cmd > 1);
    int num_wqes;
    int num_slots_per_wqe = 3;
    uint8_t fm_ce_se;

    size_t transfer_size = ibgda_cal_transfer_size(bytes, lchunk_size, rchunk_size);
    uint64_t base_wqe_idx, base_slot_idx, base_msn_idx, base_psn;
    uint64_t my_slot_idx, my_msn_idx, my_psn;
    int ppw, total_msn, total_pkts;

    if (transfer_size == bytes) {
        amask = __activemask();
        can_coalesce_warp = ibgda_can_coalesce_warp(amask, qp);
        if (can_coalesce_warp) {
            my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();
            tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
        } else {
            my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_THREAD>();
            tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_THREAD>();
        }

        int num_rdma_write_wqes_per_cmd = 1;

        int num_wqes_per_cmd = num_rdma_write_wqes_per_cmd + num_atomic_wqes_per_cmd;
        num_wqes = num_wqes_per_cmd * tg_size + (need_additional_wqe ? 1 : 0);
        ppw = bnxt_re_get_pkts_per_wqe(qp, transfer_size);
        total_msn = num_wqes_per_cmd * tg_size;
        total_pkts = total_msn * ppw;

        if (my_tid == 0) {
            base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas,
                                                   num_slots_per_wqe, total_msn, total_pkts,
                                                   &base_slot_idx, &base_msn_idx, &base_psn);
        }

        if (can_coalesce_warp) {
            base_wqe_idx = __shfl_sync(amask, base_wqe_idx, 0);
            base_slot_idx = __shfl_sync(amask, base_slot_idx, 0);
            base_msn_idx = __shfl_sync(amask, base_msn_idx, 0);
            base_psn = __shfl_sync(amask, base_psn, 0);
        }

        my_slot_idx = base_slot_idx + (my_tid * num_wqes_per_cmd * num_slots_per_wqe);
        my_msn_idx = base_msn_idx + (my_tid * num_wqes_per_cmd);
        my_psn = base_psn + (my_tid * num_wqes_per_cmd * ppw);

        ibgda_write_rdma_write_wqe<support_half_av_seg>(qp, (uint64_t)lptr, lkey, raddr, rkey,
                                                        bytes, my_slot_idx, my_msn_idx, my_psn,
                                                        ppw, 0);

        fm_ce_se = 0;

        //TBD: atomic wqe posted without reserve?
        ibgda_write_atomic_wqe<support_half_av_seg>(
            qp, &signal, NULL, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sig_raddr, sig_rkey,
            sizeof(signal), my_slot_idx + num_rdma_write_wqes_per_cmd * num_slots_per_wqe,
            sig_op, fm_ce_se);
        if (can_coalesce_warp) {
            nvshmemi_warp_sync();
        }

        if (my_tid == tg_size - 1) {
            if (need_additional_wqe) {
                my_slot_idx += num_wqes_per_cmd * num_slots_per_wqe;
                ibgda_write_nop_wqe(qp, my_slot_idx);
            }

            // Require membar.sys to push data buffer to the point of consistency.
            if (is_data_buf_in_sysmem) __threadfence_system();
            if (is_qp_shared_among_ctas)
                ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);
            else
                ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);

            if (!is_nbi) {
                ibgda_quiet(qp);
            }
        }

        if (can_coalesce_warp) {
            nvshmemi_warp_sync();
        }
    } else {
        ibgda_rma_thread<NVSHMEMI_OP_PUT, true, support_half_av_seg>(
            (uintptr_t)rptr, (uintptr_t)lptr, bytes, pe, pe);

        num_wqes = num_atomic_wqes_per_cmd + (need_additional_wqe ? 1 : 0);
        ppw = 1;
        total_msn = num_atomic_wqes_per_cmd;
        total_pkts = total_msn * ppw;

        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas,
                                               num_slots_per_wqe, total_msn, total_pkts,
                                               &base_slot_idx, &base_msn_idx, &base_psn);
        my_slot_idx = base_slot_idx;
        my_msn_idx = base_msn_idx;
        my_psn = base_psn;

        fm_ce_se = 0;

        ibgda_write_atomic_wqe<support_half_av_seg>(
            qp, &signal, NULL, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sig_raddr, sig_rkey,
            sizeof(signal), my_slot_idx, sig_op, fm_ce_se);

        if (need_additional_wqe) {
            my_slot_idx += num_atomic_wqes_per_cmd * num_slots_per_wqe;
            ibgda_write_nop_wqe(qp, my_slot_idx);
        }
        if (is_qp_shared_among_ctas)
            ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);
        else
            ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);

        if (!is_nbi) {
            ibgda_quiet(qp);
        }
    }
}

/**
 * PUT SIGNAL base
 */
#if __cplusplus >= 201103L
static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64,
              "static_assert(NVSHMEMI_IBGDA_MIN_QP_DEPTH >= 64) failed");
#endif
template <threadgroup_t SCOPE, bool is_nbi, bool support_half_av_seg>
__device__ NVSHMEMI_STATIC NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_ibgda_put_signal_impl(
    void *req_rptr, void *req_lptr, size_t bytes, void *sig_rptr, uint64_t signal,
    nvshmemi_amo_t sig_op, int pe) {
    assert(SCOPE == NVSHMEMI_THREADGROUP_WARP || SCOPE == NVSHMEMI_THREADGROUP_BLOCK);

    // Use only wrap 0
    int my_tid = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    int tg_size = nvshmemi_threadgroup_size<NVSHMEMI_THREADGROUP_WARP>();
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

    int is_qp_shared_among_ctas;
    nvshmemi_ibgda_device_qp_t *qp;

    int num_rdma_write_wqes_per_cmd;
    int num_atomic_wqes_per_cmd;
    bool need_additional_wqe;

    int num_wqes;
    int num_slots_per_wqe = 3;
    int ppw, total_msn, total_pkts;

    uint64_t base_wqe_idx, base_slot_idx, base_msn_idx, base_psn;
    uint64_t my_slot_idx, my_msn_idx, my_psn;

    size_t remaining_size = bytes;

    size_t transfer_size;
    size_t my_transfer_size = 0;

    uint64_t rptr = (uint64_t)req_rptr;
    uint64_t lptr = (uint64_t)req_lptr;

    __be32 lkey;
    __be32 my_lkey = 0;
    uint64_t my_laddr;
    size_t lchunk_size;

    __be32 rkey;
    __be32 my_rkey = 0;
    uint64_t raddr;
    uint64_t my_raddr;
    size_t rchunk_size;

    int chunk_idx = 0;

    bool is_data_buf_in_sysmem;

    // Not warp 0, wait at the exit.
    if (my_tid >= tg_size) {
        goto out;
    }

    my_tid = nvshmemi_thread_id_in_threadgroup<NVSHMEMI_THREADGROUP_WARP>();

    if (my_tid == 0) {
        qp = ibgda_get_qp(pe, (bool *)&is_qp_shared_among_ctas);
    }
    qp = (nvshmemi_ibgda_device_qp_t *)__shfl_sync(IBGDA_FULL_WARP, (uintptr_t)qp, 0);
    is_qp_shared_among_ctas = __shfl_sync(IBGDA_FULL_WARP, is_qp_shared_among_ctas, 0);

    num_rdma_write_wqes_per_cmd = 1;

    num_atomic_wqes_per_cmd = ibgda_get_num_wqes_in_atomic<uint64_t>(sig_op, qp->qp_type);
    need_additional_wqe = (num_atomic_wqes_per_cmd > 1);

    // Calculate how many chunks we need to send.
    while (remaining_size > 0) {
        ibgda_get_lkey(lptr, &lkey, &lchunk_size, &is_data_buf_in_sysmem, qp->dev_idx);
        ibgda_get_raddr_rkey(rptr, pe, pe, &raddr, &rkey, &rchunk_size, qp->dev_idx);
        transfer_size = ibgda_cal_transfer_size(remaining_size, lchunk_size, rchunk_size);
        if (my_tid == chunk_idx) {
            my_lkey = lkey;
            my_laddr = lptr;
            my_rkey = rkey;
            my_raddr = raddr;
            my_transfer_size = transfer_size;
        }

        remaining_size -= transfer_size;
        rptr += transfer_size;
        lptr += transfer_size;

        ++chunk_idx;
    }

    // Too many chunks. Use nvshmemi_ibgda_put_signal_thread_impl to handle it instead.
    // Note that we need one thread to handle amo.
    if (unlikely(chunk_idx > tg_size - 1)) {
        if (my_tid == 0) {
            nvshmemi_ibgda_put_signal_thread_impl<is_nbi, support_half_av_seg>(
                req_rptr, req_lptr, bytes, sig_rptr, signal, sig_op, pe);
        }
        goto out;
    }

    ppw = bnxt_re_get_pkts_per_wqe(qp, transfer_size);
    num_wqes = num_rdma_write_wqes_per_cmd * chunk_idx;
    total_msn = num_wqes;
    total_pkts = total_msn * ppw;
    num_wqes += num_atomic_wqes_per_cmd + (need_additional_wqe ? 1 : 0);
    total_msn += num_atomic_wqes_per_cmd;
    total_pkts += num_atomic_wqes_per_cmd * 1; // 1 pkt per atomic wqe
    if (my_tid == 0) {
        base_wqe_idx = ibgda_reserve_wqe_slots(qp, num_wqes, is_qp_shared_among_ctas,
                                               num_slots_per_wqe, total_msn, total_pkts,
                                               &base_slot_idx, &base_msn_idx, &base_psn);
    }

    base_wqe_idx = __shfl_sync(IBGDA_FULL_WARP, base_wqe_idx, 0);
    base_slot_idx = __shfl_sync(IBGDA_FULL_WARP, base_slot_idx, 0);
    base_msn_idx = __shfl_sync(IBGDA_FULL_WARP, base_msn_idx, 0);
    base_psn = __shfl_sync(IBGDA_FULL_WARP, base_psn, 0);
    my_slot_idx = base_slot_idx + (my_tid * num_rdma_write_wqes_per_cmd * num_slots_per_wqe);
    my_msn_idx = base_msn_idx + (my_tid * num_rdma_write_wqes_per_cmd);
    my_psn = base_psn + (my_tid * num_rdma_write_wqes_per_cmd * ppw);

    if (my_tid < chunk_idx) {
        ibgda_write_rdma_write_wqe<support_half_av_seg>(qp, my_laddr, my_lkey, my_raddr,
                                                    my_rkey, my_transfer_size, my_slot_idx,
                                                    my_msn_idx, my_psn, ppw, 0);
    } else if (my_tid == chunk_idx) {
        __be32 sig_rkey;
        uint64_t sig_raddr;
        size_t sig_rchunk_size;
        ibgda_get_raddr_rkey((uint64_t)sig_rptr, pe, pe, &sig_raddr, &sig_rkey, &sig_rchunk_size,
                             qp->dev_idx);

        uint8_t fm_ce_se = 0;

        ibgda_write_atomic_wqe<support_half_av_seg>(
            qp, &signal, NULL, (uint64_t)qp->ibuf.buf, qp->ibuf.lkey, sig_raddr, sig_rkey,
            sizeof(signal), my_slot_idx, sig_op, fm_ce_se);

        if (need_additional_wqe) {
            my_slot_idx += num_atomic_wqes_per_cmd * num_slots_per_wqe;
            ibgda_write_nop_wqe(qp, my_slot_idx);
        }
    }

    nvshmemi_warp_sync();

    if (my_tid == chunk_idx) {
        // Require membar.sys to push data buffer to the point of consistency.
        if (is_data_buf_in_sysmem) __threadfence_system();

        if (is_qp_shared_among_ctas)
            ibgda_submit_requests<true>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);
        else
            ibgda_submit_requests<false>(qp, base_wqe_idx, num_wqes, base_slot_idx, num_wqes * num_slots_per_wqe);

        if (!is_nbi) {
            ibgda_quiet(qp);
        }
    }

out:
    nvshmemi_threadgroup_sync<SCOPE>();
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_ibgda_put_signal(
    void *rptr, void *lptr, size_t bytes, void *sig_rptr, uint64_t signal, nvshmemi_amo_t sig_op,
    int pe, bool is_nbi) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    if (SCOPE == NVSHMEMI_THREADGROUP_THREAD) {
        if (is_nbi && state->support_half_av_seg)
            nvshmemi_ibgda_put_signal_thread_impl<true, true>(rptr, lptr, bytes, sig_rptr, signal,
                                                              sig_op, pe);
        else if (is_nbi && !state->support_half_av_seg)
            nvshmemi_ibgda_put_signal_thread_impl<true, false>(rptr, lptr, bytes, sig_rptr, signal,
                                                               sig_op, pe);
        else if (!is_nbi && state->support_half_av_seg)
            nvshmemi_ibgda_put_signal_thread_impl<false, true>(rptr, lptr, bytes, sig_rptr, signal,
                                                               sig_op, pe);
        else
            nvshmemi_ibgda_put_signal_thread_impl<false, false>(rptr, lptr, bytes, sig_rptr, signal,
                                                                sig_op, pe);
    } else {
        if (is_nbi && state->support_half_av_seg)
            nvshmemi_ibgda_put_signal_impl<SCOPE, true, true>(rptr, lptr, bytes, sig_rptr, signal,
                                                              sig_op, pe);
        else if (is_nbi && !state->support_half_av_seg)
            nvshmemi_ibgda_put_signal_impl<SCOPE, true, false>(rptr, lptr, bytes, sig_rptr, signal,
                                                               sig_op, pe);
        else if (!is_nbi && state->support_half_av_seg)
            nvshmemi_ibgda_put_signal_impl<SCOPE, false, true>(rptr, lptr, bytes, sig_rptr, signal,
                                                               sig_op, pe);
        else
            nvshmemi_ibgda_put_signal_impl<SCOPE, false, false>(rptr, lptr, bytes, sig_rptr, signal,
                                                                sig_op, pe);
    }
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_ibgda_quiet() {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    nvshmemi_ibgda_device_qp_t *qp;
    uint32_t nrcs =
        state->num_rc_per_pe * nvshmemi_device_state_d.npes * state->num_devices_initialized;
    uint32_t index_in_scope = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    uint32_t scope_size = nvshmemi_threadgroup_size<SCOPE>();

    scope_size =
        scope_size > IBGDA_MAX_THREADS_PER_QUIET ? IBGDA_MAX_THREADS_PER_QUIET : scope_size;

    if (index_in_scope < scope_size) {
#ifdef NVSHMEM_IBGDA_USE_RC_LOOPBACK
        uint32_t rc_lb = 0;
#endif
        for (uint32_t i = index_in_scope; i < nrcs; i += scope_size) {
            if (i / (state->num_rc_per_pe * state->num_devices_initialized) ==
                nvshmemi_device_state_d.mype) {
#ifdef NVSHMEM_IBGDA_USE_RC_LOOPBACK
                // Allow loopback transactions
                if (rc_lb++)
#else
                // No loopback to oneself
#endif
                    continue;
            }
            qp = &state->globalmem.rcs[i];
            ibgda_quiet_with_cst(qp, true);
        }
    }
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_ibgda_fence() {
    // Multiple QPs may target the same PE before fence.
    // We need to quiet those QPs.
    // TODO: Make it more efficient.
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();
    uint32_t index_in_scope = nvshmemi_thread_id_in_threadgroup<SCOPE>();
    uint32_t scope_size = nvshmemi_threadgroup_size<SCOPE>();
    uint32_t nrcs = state->num_rc_per_pe * nvshmemi_device_state_d.npes;
    nvshmemi_ibgda_device_qp_t *qp;

    // As all WQEs always go to the same QP, FENCE is naturally guaranteed.
    if (unlikely(nrcs <= 1)) return;

    scope_size =
        scope_size > IBGDA_MAX_THREADS_PER_QUIET ? IBGDA_MAX_THREADS_PER_QUIET : scope_size;

    // Fence does not guarantee the completion of prior operations.
    // It is ok for GET to finish without data arrival.
    // Use ibgda_quiet here instead of ibgda_quiet_with_cst since it is cheaper.
    if (index_in_scope < scope_size) {
        for (uint32_t i = index_in_scope; i < nrcs; i += scope_size) {
            if (i / state->num_rc_per_pe == nvshmemi_device_state_d.mype) continue;
            qp = &state->globalmem.rcs[i];
            ibgda_quiet(qp);
        }
    }

    nvshmemi_threadgroup_sync<SCOPE>();
}

__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void nvshmemi_ibgda_enforce_consistency_at_target(
    bool use_membar) {
    nvshmemi_ibgda_device_state_t *state = ibgda_get_state();

#ifdef NVSHMEM_IBGDA_USE_RC_LOOPBACK
    if (!state->may_skip_cst) {
        bool is_qp_shared_among_ctas;
        // Use the RC loopback QP for CST
        nvshmemi_ibgda_device_qp_t *qp;

        /* We must run the cst op on all devices */
        for (int i = 0; i < state->num_devices_initialized; i++) {
            qp = ibgda_get_qp(nvshmemi_device_state_d.mype, &is_qp_shared_among_ctas);
            ibgda_cst(qp, is_qp_shared_among_ctas);
        }
    }
#endif
    // TODO: This fence is from the design of Proxy.
    // Review if we still need it when we fully move to IBGDA -- especially for on-stream API.
    if (use_membar) {
        __threadfence_system();  // XXX: prevents store to issue_d reordered to before load from
                                 // cst_ack_d (breaks cst -> rma)
    }
}


#endif /* __CUDA_ARCH__ */

#endif /* _NVSHMEMI_IBGDA_DEVICE_H_ */
