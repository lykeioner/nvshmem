/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <assert.h>                                      // for assert
#include <cuda.h>                                        // for CUDA_SUCCESS, CUdevice, CUd...
#include <cuda_runtime.h>                                // for cudaFree, cudaMalloc, cudaM...
#include <driver_types.h>                                // for cudaSuccess, cudaMemcpyHost...
#include <endian.h>                                      // for htobe32, htobe64
#include <errno.h>                                       // for ENOMEM
#include <linux/types.h>                                 // for __be32
#include <math.h>                                        // for ceil, log2
#include <stddef.h>                                      // for NULL, size_t, offsetof
#include <stdint.h>                                      // for uint8_t, uint64_t, uint32_t
#include <stdio.h>                                       // for fprintf, stderr, printf
#include <stdlib.h>                                      // for free, calloc, malloc, posix...
#include <unistd.h>                                      // for _SC_PAGESIZE
#include <string.h>                                      // for memset, memcpy, strcmp, strstr
#include <sys/types.h>                                   // for off_t
#include <algorithm>                                     // for for_each, remove_if, max
#include <cctype>                                        // for tolower, isspace
#include <string>                                        // for basic_string, string, opera...
#include <vector>                                        // for vector
#include "device_host_transport/nvshmem_common_ibgda.h"  // for nvshmemi_ibgda_device_state_t
#include "internal/host_transport/cudawrap.h"            // for CUPFN, nvshmemi_cuda_fn_table
#include "bootstrap_host_transport/env_defs_internal.h"  // for nvshmemi_options_s, nvshmem...
#include "non_abi/nvshmemx_error.h"                      // for NVSHMEMX_ERROR_INTERNAL
#include "non_abi/nvshmem_build_options.h"               // IWYU pragma: keep
#include "non_abi/nvshmem_version.h"
#include "infiniband/verbs.h"   // for ibv_ah_attr, ibv_port_attr
#include "internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h"  // for bootstrap_handle_t
#include "internal/host_transport/nvshmemi_transport_defines.h"  // for nvshmem_mem_handle_t, NVSHM...
#include "internal/host_transport/transport.h"  // for nvshmem_transport, nvshmem_...
#include "transport_common.h"                   // for nvshmemt_ibv_function_table
#include "transport_ib_common.h"                // for nvshmemt_ib_common_mem_handle
#include "infiniband/bnxt_re_dv.h"
#ifdef NVSHMEM_USE_GDRCOPY
#include "transport_gdr_common.h"
#endif

#define CUDA_RUNTIME_ERROR_STRING(result)                                         \
    do {                                                                          \
        if (unlikely(cudaSuccess != result)) {                                    \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
        }                                                                         \
    } while (0)

// TBD - Duplicate from bnxt_re_fp_defs.h. Fix it later.
#define BNXT_RE_SLOT_SIZE_BB            16
#define BNXT_RE_STATIC_WQE_SIZE_SLOTS   4
#define BNXT_RE_STATIC_WQE_BB           (BNXT_RE_STATIC_WQE_SIZE_SLOTS * BNXT_RE_SLOT_SIZE_BB)
#define BNXT_RE_STATIC_WQE_SHIFT        6

#define BNXT_RE_STATIC_RQE_SIZE_SLOTS   4
#define BNXT_RE_STATIC_RQE_BB           (BNXT_RE_STATIC_RQE_SIZE_SLOTS * BNXT_RE_SLOT_SIZE_BB)
#define BNXT_RE_STATIC_RQE_SHIFT        6

#define BNXT_RE_STATIC_CQE_SIZE_SLOTS   4
#define BNXT_RE_STATIC_CQE_BB           (BNXT_RE_STATIC_CQE_SIZE_SLOTS * BNXT_RE_SLOT_SIZE_BB)
#define BNXT_RE_STATIC_CQE_SHIFT        6
#define BNXT_RE_QUEUE_START_PHASE       0x01

#define NVSHMEMI_IBGDA_CQE_SIZE 64
/* TBD - Hardcoding based on max sges are 13 */
#define NVSHMEMI_IBGDA_BNXT_MAX_INLINE_SIZE (16 * 13)
#define NVSHMEMI_IBGDA_BNXT_SEND_SGE 2
#define NVSHMEMI_IBGDA_BNXT_RECV_SGE 2

#define MAX_NUM_HCAS 16
#define MAX_NUM_PORTS 4
#define MAX_NUM_PES_PER_NODE 32

#define IBGDA_DC_ACCESS_KEY 0x5623CEAF

#define IBGDA_DBRSIZE 8
#define IBGDA_SRQ_TYPE_VALUE 0x1

#define IBGDA_LOG_MAX_MSG_SIZE 30  // 30 is max allowed on IB QPs
#define IBGDA_MIN_RNR_NAK 12

#define IBGDA_GRH_HOP_LIMIT 255

#define IBGDA_ROCE_V1_UDP_SPORT_BASE 0x0000
#define IBGDA_ROCE_V2_UDP_SPORT_BASE 0xC000

// First slot is reserved for non-fetch operations.
#define IBGDA_IBUF_RESERVED_SLOTS 1

#define IBGDA_GPAGE_BITS 16
#define IBGDA_GPAGE_SIZE (1ULL << IBGDA_GPAGE_BITS)
#define IBGDA_GPAGE_OFF (IBGDA_GPAGE_SIZE - 1)
#define IBGDA_GPAGE_MASK (~(IBGDA_GPAGE_OFF))

#define IBGDA_ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#define IBGDA_READ_ONCE(x) IBGDA_ACCESS_ONCE(x)
#define IBGDA_WRITE_ONCE(x, v) (IBGDA_ACCESS_ONCE(x) = (v))

#define IBGDA_MIN(x, y) ((x) < (y) ? (x) : (y))
#define IBGDA_MAX(x, y) ((x) > (y) ? (x) : (y))

#define IBGDA_ROUND_UP(V, SIZE) (((V) + (SIZE)-1) / (SIZE) * (SIZE))

#define IBGDA_ROUND_UP_POW2(_n)                 \
    ({                                          \
        typeof(_n) pow2 = 0;                    \
        assert((_n) >= 1);                      \
        for (pow2 = 1; pow2 < (_n); pow2 <<= 1) \
            ;                                   \
        pow2;                                   \
    })

#define IBGDA_ROUND_UP_POW2_OR_0(_n) (((_n) == 0) ? 0 : IBGDA_ROUND_UP_POW2(_n))

#define IBGDA_ROUND_DOWN_POW2_OR_0(_n)                  \
    ({                                                  \
        typeof(_n) pow2 = IBGDA_ROUND_UP_POW2_OR_0(_n); \
        (((_n) < pow2) ? pow2 / 2 : pow2);              \
    })

template <typename T>
inline T IBGDA_ILOG2(T _n) {
    return (T)ceil(log2((double)_n));
}

#define IBGDA_ILOG2_OR0(_n) (((_n) == 0) ? 0 : IBGDA_ILOG2(_n))

/* TBD - Fix below */
enum {
        BNXT_SEND_WQE_BB        = 64,
        BNXT_SEND_WQE_SHIFT     = 6,
};

/* TBD - Fix below */
enum { IBGDA_BNXT_NC_UAR_SIZE = 12 };

typedef enum {
    IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_AUTO = 0,
    IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM,
    IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_HOSTMEM,
} ibgda_nic_mapping_memtype_reqeust_t;

typedef enum {
    IBGDA_MEM_TYPE_HOST = 0,
    IBGDA_MEM_TYPE_GPU = 1,
    IBGDA_MEM_TYPE_NIC = 2,
} ibgda_mem_type_t;

typedef enum {
    IBGDA_NIC_HANDLER_AUTO = 0,
    IBGDA_NIC_HANDLER_GPU,
    IBGDA_NIC_HANDLER_CPU,
} ibgda_nic_handler_t;

struct ibgda_mem_object {
    ibgda_mem_type_t mem_type;
    struct {
        void *cpu_ptr;
        void *gpu_ptr;
        size_t size;
    } base;
    struct {
        void *cpu_ptr;
        void *gpu_ptr;
        size_t size;
    } aligned;
    union {
        struct bnxt_re_dv_umem *umem;
        struct bnxt_re_dv_devx_uar *uar;
    };
    bool has_cpu_mapping : 1;
    bool has_gpu_mapping : 1;
    bool has_nic_mapping : 1;
#ifdef NVSHMEM_USE_GDRCOPY
    gdr_mh_t mh;
#endif
    struct bnxt_re_dv_qp_mem_info qp_mem;
};

/* TBD - Move this to DV API */
struct bnxt_re_dv_devx_uar {
    struct ibv_context *context;
    void *reg_addr;
    uint32_t dpi_idx;
    // TBD. Remove below
    void *base_addr;
    off_t mmap_off;
    uint64_t comp_mask;
};

struct ibgda_cq {
    struct ibv_cq *devx_cq;
    uint32_t cqn;
    uint32_t num_cqe;
    struct ibgda_mem_object *cq_mobject;
    struct ibgda_mem_object *dbr_mobject;
    off_t cq_offset;
    off_t dbr_offset;
};

struct ibgda_ep {
    nvshmemi_ibgda_device_qp_type_t qp_type;

    union {
        /* TBD
         * Ideally we do not require ibv_qp in DV path.
         */
        struct ibv_qp *devx_qp;
        struct ibv_qp *ib_qp;
    };
    uint32_t qpn;
    int portid;

    size_t sq_cnt;
    size_t rq_cnt;

    struct ibgda_mem_object *wq_mobject;
    struct ibgda_mem_object *rq_mobject;
    struct ibgda_mem_object *dbr_mobject;
    struct ibgda_mem_object *uar_mobject;

    off_t wq_offset;
    off_t rq_offset;
    off_t dbr_offset;

    struct ibgda_cq *send_cq;
    struct ibgda_cq *recv_cq;
    struct ibv_ah *ah;

    uint32_t user_index;

    // MSN table related parameters
    uint16_t mtu;
    uint32_t sq_psn;
    uint32_t msn;
    uint32_t msn_tbl_sz;
    void *pad;
};

struct ibgda_mem_handle {
    struct nvshmemt_ib_common_mem_handle dev_mem_handles[NVSHMEMI_IBGDA_MAX_DEVICES_PER_PE];
    int num_devs;
};

struct ibgda_rc_handle {
    uint32_t qpn;
    uint16_t lid;
    // RoCE
    uint64_t spn;
    uint64_t iid;
};

struct ibgda_internal_buffer {
    struct ibgda_mem_object *mem_object;
    struct nvshmemt_ib_common_mem_handle *mem_handle;
};

struct ibgda_device {
    struct ibv_device *dev;
    struct ibv_pd *pd; /* protection domain */
    struct ibv_context *context;
    struct ibv_device_attr device_attr;
    struct ibv_port_attr port_attr[MAX_NUM_PORTS];
    struct nvshmemt_ib_gid_info gid_info[MAX_NUM_PORTS];
    struct {
        struct ibv_srq *srq;
        struct ibv_cq *recv_cq;
        struct ibgda_mem_object *wq_mobject;
        size_t wq_buf_size_per_qp;
        off_t cur_wq_off;
        struct ibgda_mem_object *rq_mobject;
        size_t rq_buf_size_per_qp;
        off_t cur_rq_off;
        struct ibgda_mem_object *dbr_mobject;
        struct ibgda_internal_buffer internal_buf;
        off_t cur_dbr_off;
        int pdn;
        int srqn;
        int rcqn;
        struct ibgda_mem_object *prod_idx_mobject;
        uint64_t *prod_idx_cache;
        uint64_t *prod_idx_snapshot;
    } qp_shared_object;  // For RC
    struct {
        size_t cq_buf_size_per_cq;
        struct ibgda_mem_object *cq_mobject;
        struct ibgda_mem_object *dbr_mobject;
        off_t cur_cq_off;
        off_t cur_dbr_off;
    } cq_shared_object;
    struct {
        struct ibgda_ep **eps;
        struct ibgda_rc_handle *peer_ep_handles;
        int num_eps_per_pe;
        nvshmemi_ibgda_device_qp_map_type_t map_by;
    } rc;
    bool support_nic_buf_on_gpumem;
    bool support_nic_buf_on_hostmem;
    bool support_half_av_seg;
    bool may_skip_cst;
    ibgda_nic_handler_t nic_handler;
};

typedef struct {
    struct nvshmemi_options_s *options;
    void *devices;
    int *dev_ids;
    int *port_ids;
    int *selected_dev_ids;
    int n_dev_ids;
    int n_devs_selected;
    int log_level;
    bool cuda_support_dmabuf;
    bool dmabuf_support_for_data_buffers;
    bool dmabuf_support_for_control_buffers;
    cudaStream_t my_stream;
} nvshmemt_ibgda_state_t;

struct ibgda_device_local_only_mhandle_cache {
    nvshmemi_ibgda_device_local_only_mhandle_t mhandle;
    void *
        dev_ptr;  // Ptr to GPU buffer that contains a copy of this mhandle. CPU cannot dereference.
};

// CPU cannot dereference next
static std::vector<struct ibgda_device_local_only_mhandle_cache> ibgda_device_local_only_mhandles;

static std::vector<nvshmemi_ibgda_device_key_t> ibgda_device_lkeys;
static std::vector<nvshmemi_ibgda_device_key_t> ibgda_device_rkeys;

// Ptr to GPU buffer. CPU cannot dereference.
static void *ibgda_device_lkeys_d = 0;
static void *ibgda_device_rkeys_d = 0;

/* transport constants */
static int ibgda_qp_depth = 0;
static int ibgda_srq_depth;
static int ibgda_num_requests_in_batch;
static int ibgda_num_fetch_slots_per_rc;

/* ibv state */
static struct nvshmemt_ibv_function_table ftable;
static void *ibv_handle;

/* CUDA function table */
static struct nvshmemi_cuda_fn_table *ibgda_cuda_syms;

#ifdef NVSHMEM_USE_GDRCOPY
static gdr_t gdr_desc;
static struct gdrcopy_function_table gdrcopy_ftable;
static void *gdrcopy_handle = NULL;
#endif
static bool use_gdrcopy = 0;

static ibgda_mem_type_t ibgda_nic_buf_location;
static ibgda_nic_handler_t ibgda_nic_handler;

static int ibgda_parse_qp_map_by(nvshmemi_ibgda_device_qp_map_type_t *out_map_by, const char *str) {
    int status = 0;
    nvshmemi_ibgda_device_qp_map_type_t map_by;
    std::string req = str;

    // Trim whitespace
    req.erase(std::remove_if(req.begin(), req.end(), ::isspace), req.end());

    // To lower case
    std::for_each(req.begin(), req.end(), [](decltype(*req.begin()) &c) { c = ::tolower(c); });

    if (req == "cta") {
        map_by = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_CTA;
    } else if (req == "sm") {
        map_by = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_SM;
    } else if (req == "warp") {
        map_by = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_WARP;
    } else {
        status = NVSHMEMX_ERROR_INVALID_VALUE;
    }

    if (status == 0) {
        *out_map_by = map_by;
    }

    return status;
}

static int ibgda_parse_nic_handler_request(ibgda_nic_handler_t *out_loc, const char *str) {
    int status = 0;
    ibgda_nic_handler_t loc;
    std::string req = str;

    // Trim whitespace
    req.erase(std::remove_if(req.begin(), req.end(), ::isspace), req.end());

    // To lower case
    std::for_each(req.begin(), req.end(), [](decltype(*req.begin()) &c) { c = ::tolower(c); });

    if (req == "auto") {
        loc = IBGDA_NIC_HANDLER_AUTO;
    } else if (req == "gpu") {
        loc = IBGDA_NIC_HANDLER_GPU;
    } else if (req == "cpu") {
        loc = IBGDA_NIC_HANDLER_CPU;
    } else {
        status = NVSHMEMX_ERROR_INVALID_VALUE;
    }

    if (status == 0) {
        *out_loc = loc;
    }

    return status;
}

static size_t ibgda_get_host_page_size() {
    static size_t host_page_size = 0;
    if (!host_page_size) host_page_size = sysconf(_SC_PAGESIZE);
    return host_page_size;
}

int nvshmemt_ibgda_progress(nvshmem_transport_t t) {
    NVSHMEMI_ERROR_PRINT("ibgda progress not implemented");
    return NVSHMEMX_ERROR_NOT_SUPPORTED;
}

int nvshmemt_ibgda_show_info(struct nvshmem_transport *transport, int style) {
    NVSHMEMI_ERROR_PRINT("ibgda show info not implemented");
    return 0;
}

static int get_pci_path(int dev, char **pci_path, nvshmem_transport_t t) {
    int status = NVSHMEMX_SUCCESS;

    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)transport->state;
    int dev_id = ibgda_state->dev_ids[dev];
    const char *ib_name =
        (const char *)((struct ibgda_device *)ibgda_state->devices)[dev_id].dev->name;
    /* TBD - Do we need below ? */
#if 0
    status = nvshmemt_ib_iface_get_bnxt_path(ib_name, pci_path);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "nvshmemt_ib_iface_get_bnxt_path failed \n");
#endif
out:
    return status;
}

int nvshmemt_ibgda_can_reach_peer(int *access, struct nvshmem_transport_pe_info *peer_info,
                                  nvshmem_transport_t t) {
    int status = 0;

    *access = NVSHMEM_TRANSPORT_CAP_GPU_WRITE | NVSHMEM_TRANSPORT_CAP_GPU_READ |
              NVSHMEM_TRANSPORT_CAP_GPU_ATOMICS;

    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: from %s %d \n", __func__, __LINE__);
    return status;
}

int nvshmemt_ibgda_get_mem_handle(nvshmem_mem_handle_t *mem_handle,
                                  nvshmem_mem_handle_t *mem_handle_in, void *buf, size_t length,
                                  nvshmem_transport_t t, bool local_only) {
    int status = 0;
    struct nvshmem_transport *transport = (struct nvshmem_transport *)t;
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)transport->state;
    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: from %s %d \n", __func__, __LINE__);

    __be32 device_lkey;
    struct ibgda_mem_handle *handle;

    nvshmemi_ibgda_device_local_only_mhandle_t *device_mhandle_d = NULL;
    bool did_emplace = false;

    nvshmemi_ibgda_device_state_t *ibgda_device_state;
    ibgda_device_state = (nvshmemi_ibgda_device_state_t *)transport->type_specific_shared_state;
    assert(ibgda_device_state != NULL);
    int n_devs_selected = ibgda_state->n_devs_selected;

    memset((void *)mem_handle, 0, sizeof(*mem_handle));
    handle = (struct ibgda_mem_handle *)mem_handle;
    handle->num_devs = n_devs_selected;

    for (int i = 0; i < n_devs_selected; ++i) {
        struct ibgda_device *device =
            ((struct ibgda_device *)ibgda_state->devices + ibgda_state->selected_dev_ids[i]);
        nvshmem_mem_handle_t *dev_handle = (nvshmem_mem_handle_t *)&handle->dev_mem_handles[i];

        status = nvshmemt_ib_common_reg_mem_handle(
            &ftable, device->pd, dev_handle, buf, length, local_only,
            ibgda_state->dmabuf_support_for_data_buffers, ibgda_cuda_syms, ibgda_state->log_level,
            ibgda_state->options->IB_ENABLE_RELAXED_ORDERING);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Unable to register memory handle.\n");
    }

    if (local_only) {
        struct ibgda_device_local_only_mhandle_cache device_mhandle_cache;
        nvshmemi_ibgda_device_local_only_mhandle_t *device_mhandle_h =
            &device_mhandle_cache.mhandle;
        nvshmemi_init_ibgda_device_local_only_memhandle((*device_mhandle_h));

        void *mhandle_gpu_ptr;

        cudaPointerAttributes buf_attributes;

        status = cudaPointerGetAttributes(&buf_attributes, buf);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaPointerGetAttributes failed.\n");

        status = cudaMalloc((void **)&device_mhandle_d, sizeof(*device_mhandle_d));
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                              "cudaMalloc failed.\n");

        device_mhandle_h->start = (uint64_t)buf;
        device_mhandle_h->end = (uint64_t)buf + length - 1;
        device_mhandle_h->is_sysmem_scope = (buf_attributes.type != cudaMemoryTypeDevice);
        device_mhandle_h->next = NULL;
        for (int i = 0; i < n_devs_selected; ++i) {
            device_lkey = htobe32(handle->dev_mem_handles[i].lkey);
            device_mhandle_h->lkeys[i] = device_lkey;
        }

        status = cudaMemcpyAsync((void *)device_mhandle_d, (const void *)device_mhandle_h,
                                 sizeof(*device_mhandle_d), cudaMemcpyHostToDevice,
                                 ibgda_state->my_stream);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "Copying device_mhandle to GPU memory failed.\n");

        device_mhandle_cache.dev_ptr = device_mhandle_d;

        if (ibgda_device_local_only_mhandles.empty()) {
            ibgda_device_state->globalmem.local_only_mhandle_head = device_mhandle_d;
        } else {
            struct ibgda_device_local_only_mhandle_cache *last_mhandle_cache =
                &ibgda_device_local_only_mhandles.back();
            mhandle_gpu_ptr = (void *)((uintptr_t)last_mhandle_cache->dev_ptr +
                                       offsetof(nvshmemi_ibgda_device_local_only_mhandle_t, next));
            last_mhandle_cache->mhandle.next = device_mhandle_d;
            status = cudaMemcpyAsync(mhandle_gpu_ptr, (const void *)&device_mhandle_d,
                                     sizeof(device_mhandle_d), cudaMemcpyHostToDevice,
                                     ibgda_state->my_stream);
            NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Setting local_only_mhandle in GPU memory failed.\n");
        }

        ibgda_device_local_only_mhandles.emplace_back(device_mhandle_cache);
        did_emplace = true;
    } else {
        size_t num_lkeys;
        size_t num_elements;

        // length must be divisible by cumem_granularity, which is a power of 2.
        assert((length & ((1ULL << transport->log2_cumem_granularity) - 1)) == 0);

        num_elements = length >> transport->log2_cumem_granularity;
        while (num_elements > 0) {
            for (int i = 0; i < n_devs_selected; i++) {
                device_lkey = htobe32(handle->dev_mem_handles[i].lkey);
                nvshmemi_ibgda_device_key_t dev_key;
                dev_key.key = device_lkey;
                dev_key.next_addr = (uint64_t)buf + length;
                ibgda_device_lkeys.emplace_back(dev_key);
            }
            --num_elements;
        }

        did_emplace = true;

        if (ibgda_device_lkeys_d) {
            status = cudaFree(ibgda_device_lkeys_d);
            NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                                  "cudaFree failed.\n");
            ibgda_device_lkeys_d = 0;
        }

        num_lkeys = ibgda_device_lkeys.size();

        // Put lkeys in constant memory first for cache optimization
        memcpy(ibgda_device_state->constmem.lkeys, ibgda_device_lkeys.data(),
               IBGDA_MIN(num_lkeys, NVSHMEMI_IBGDA_MAX_CONST_LKEYS) *
                   sizeof(nvshmemi_ibgda_device_key_t));

        // If we have overflow, put the rest in global memory
        if (num_lkeys > NVSHMEMI_IBGDA_MAX_CONST_LKEYS) {
            size_t lkeys_array_size =
                sizeof(nvshmemi_ibgda_device_key_t) * (num_lkeys - NVSHMEMI_IBGDA_MAX_CONST_LKEYS);

            nvshmemi_ibgda_device_key_t *data_ptr =
                &ibgda_device_lkeys.data()[NVSHMEMI_IBGDA_MAX_CONST_LKEYS];

            status = cudaMalloc(&ibgda_device_lkeys_d, lkeys_array_size);
            NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                  "cudaMalloc failed.\n");

            status = cudaMemcpyAsync(ibgda_device_lkeys_d, (const void *)data_ptr, lkeys_array_size,
                                     cudaMemcpyHostToDevice, ibgda_state->my_stream);
            NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Copying lkeys to GPU memory failed.\n");
        }
        ibgda_device_state->globalmem.lkeys = (nvshmemi_ibgda_device_key_t *)ibgda_device_lkeys_d;
    }

    status = cudaStreamSynchronize(ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "stream synchronize failed.\n");

out:
    if (status) {
        if (device_mhandle_d) cudaFree(device_mhandle_d);
        if (did_emplace) {
            if (local_only) {
                // Recoverable
                ibgda_device_local_only_mhandles.pop_back();
            } else {
                // Unrecoverable
                ibgda_device_lkeys.clear();
            }
        }

        for (int i = 0; i < n_devs_selected; ++i) {
            nvshmemt_ib_common_release_mem_handle(
                &ftable, (nvshmem_mem_handle_t *)&handle->dev_mem_handles[i],
                ibgda_state->log_level);
        }
    }
    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: from %s %d \n", __func__, __LINE__);
    return status;
}

static int ibgda_mobject_nic_map(struct ibgda_mem_object *mobject, struct ibv_context *context,
                                 uint32_t access, bool use_dmabuf = false) {
    int status = 0;
    void *addr;
    struct bnxt_re_dv_umem_reg_attr attr = {
            0,
    };
    struct bnxt_re_dv_umem *umem = NULL;

    assert(mobject);
    assert(!mobject->has_nic_mapping);
    assert(context);

    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: use_dmabuf %d from %s %d object size %d\n",
                         use_dmabuf, __func__, __LINE__, mobject->aligned.size);

    if (mobject->mem_type == IBGDA_MEM_TYPE_GPU) {
        addr = (void *)mobject->aligned.gpu_ptr;
    } else if (mobject->mem_type == IBGDA_MEM_TYPE_HOST) {
        addr = mobject->aligned.cpu_ptr;
    } else {
        status = NVSHMEMX_ERROR_INTERNAL;
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "invalid mem_type specified.\n");
        assert(0);
    }

    if (use_dmabuf && mobject->mem_type == IBGDA_MEM_TYPE_GPU) {
/* TBD - use_dmabuf is not handled */
#if 0
        int fd;
        struct mlx5dv_devx_umem_in umem_in = {
            0,
        };
        const size_t host_page_size = ibgda_get_host_page_size();
        size_t dmabuf_size = IBGDA_ROUND_UP(mobject->aligned.size, host_page_size);
        CUCHECKGOTO(ibgda_cuda_syms,
                    cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)addr, dmabuf_size,
                                                  CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
                    status, out);
        umem_in.addr = 0;
        umem_in.size = mobject->aligned.size;
        umem_in.access = access;
        umem_in.pgsz_bitmap = UINT64_MAX & ~(host_page_size - 1);
        umem_in.comp_mask = MLX5DV_UMEM_MASK_DMABUF;
        umem_in.dmabuf_fd = fd;
        umem = mlx5dv_devx_umem_reg_ex(context, &umem_in);
        close(fd);
#else
        status = NVSHMEMX_ERROR_NOT_SUPPORTED;
        goto out;
#endif
    } else {
        attr.addr = addr;
        attr.size = mobject->aligned.size;
        attr.access_flags = access;
        attr.comp_mask = BNXT_RE_DV_UMEM_FLAGS_DMABUF;

        umem = (struct bnxt_re_dv_umem*) bnxt_re_dv_umem_reg(context, &attr);
    }

    if (!umem) {
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    mobject->umem = umem;
    mobject->has_nic_mapping = true;

out:
    return status;
}

static void ibgda_mobject_nic_unmap(struct ibgda_mem_object *mobject) {
    int status = 0;

    assert(mobject);
    assert(mobject->has_nic_mapping);
    assert(mobject->mem_type != IBGDA_MEM_TYPE_NIC);
    assert(mobject->umem);

    status = bnxt_re_dv_umem_dereg(mobject->umem);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bnxt_re_dv_`umem_dereg failed.\n");

    mobject->has_nic_mapping = false;
    mobject->umem = NULL;

out:
    return;
}

static int ibgda_gpu_mem_alloc(struct ibgda_mem_object **pmobject, size_t size, size_t alignment,
                               bool host_mapping) {
    int status = 0;

    int attr_val;

    void *ptr = 0;
    void *aligned_ptr;
    size_t bufsize = size;

    void *cpu_ptr_base = NULL;
    void *cpu_ptr = NULL;

    struct ibgda_mem_object *mobject =
        (struct ibgda_mem_object *)calloc(1, sizeof(struct ibgda_mem_object));
    NVSHMEMI_NULL_ERROR_JMP(mobject, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate a new mobject.\n");

    if (alignment > 0) bufsize = size + alignment - 1;

    status = cudaMalloc(&ptr, bufsize);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaMalloc failed.\n");

    attr_val = 1;
    status =
        CUPFN(ibgda_cuda_syms,
              cuPointerSetAttribute(&attr_val, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)ptr));
    NVSHMEMI_NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                          "cuPointerSetAttribute failed.\n");

    status = cudaMemset(ptr, 0, bufsize);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaMemset failed.\n");

    if (alignment > 0) {
        aligned_ptr = (void *)((size_t)((char *)ptr + alignment - 1) & (~(alignment - 1)));
    } else {
        aligned_ptr = ptr;
    }

    if (host_mapping) {
#ifdef NVSHMEM_USE_GDRCOPY
        if (use_gdrcopy) {
            status = gdrcopy_ftable.pin_buffer(gdr_desc, (unsigned long)aligned_ptr,
                                               IBGDA_ROUND_UP(size, IBGDA_GPAGE_SIZE), 0, 0,
                                               &mobject->mh);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "gdrcopy pin_buffer failed \n");

            status = gdrcopy_ftable.map(gdr_desc, mobject->mh, &cpu_ptr_base, size);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "gdrcopy map failed \n");

            gdr_info_t info;
            status = gdrcopy_ftable.get_info(gdr_desc, mobject->mh, &info);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "gdrcopy get_info failed \n");

            // remember that mappings start on a 64KB boundary, so let's
            // calculate the offset from the head of the mapping to the
            // beginning of the buffer
            uintptr_t off;
            off = (uintptr_t)aligned_ptr - info.va;
            cpu_ptr = (void *)((uintptr_t)cpu_ptr_base + off);

            mobject->base.cpu_ptr = cpu_ptr_base;
            mobject->aligned.cpu_ptr = cpu_ptr;
        } else
#endif
        {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_NOT_SUPPORTED, out,
                               "host_mapping is not supported as GDRCopy is disable \n");
        }
    }

    mobject->mem_type = IBGDA_MEM_TYPE_GPU;

    mobject->base.gpu_ptr = ptr;
    mobject->base.size = bufsize;

    mobject->aligned.gpu_ptr = aligned_ptr;
    mobject->aligned.size = size;

    mobject->has_cpu_mapping = host_mapping;
    mobject->has_gpu_mapping = true;
    mobject->has_nic_mapping = false;

    *pmobject = mobject;

out:
    if (status) {
        if (ptr) {
            cudaError_t _status = cudaFree(ptr);
            CUDA_RUNTIME_ERROR_STRING(_status);
        }

        if (mobject) free(mobject);
    }
    return status;
}

static void ibgda_gpu_mem_free(struct ibgda_mem_object *mobject) {
    int status = 0;

    if (!mobject) return;

    assert(mobject->mem_type == IBGDA_MEM_TYPE_GPU);

#ifdef NVSHMEM_USE_GDRCOPY
    if (mobject->has_cpu_mapping) {
        assert(use_gdrcopy);

        status = gdrcopy_ftable.unmap(gdr_desc, mobject->mh, mobject->base.cpu_ptr,
                                      mobject->aligned.size);
        if (status) {
            NVSHMEMI_WARN_PRINT("gdr_unmap failed ... Continue\n");
        }

        status = gdrcopy_ftable.unpin_buffer(gdr_desc, mobject->mh);
        if (status) {
            NVSHMEMI_WARN_PRINT("gdr_unpin failed ... Continue\n");
        }
    }
#endif

    status = cudaFree(mobject->base.gpu_ptr);
    CUDA_RUNTIME_ERROR_STRING((cudaError_t)status);

    free(mobject);
}

static int ibgda_host_mem_alloc(struct ibgda_mem_object **pmobject, size_t size, size_t alignment,
                                bool gpu_mapping) {
    int status;

    void *ptr = NULL;

    bool did_host_reg = false;
    void *gpu_ptr;

    struct ibgda_mem_object *mobject =
        (struct ibgda_mem_object *)calloc(1, sizeof(struct ibgda_mem_object));
    NVSHMEMI_NULL_ERROR_JMP(mobject, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate a new mobject.\n");

    status = posix_memalign(&ptr, alignment, size);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "posix_memalign failed.\n");

    memset(ptr, 0, size);

    if (gpu_mapping) {
        status = cudaHostRegister(ptr, size, cudaHostRegisterPortable | cudaHostRegisterMapped);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaHostRegister failed.\n");
        did_host_reg = true;

        status = cudaHostGetDevicePointer(&gpu_ptr, ptr, 0);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaHostGetDevicePointer failed.\n");

        mobject->base.gpu_ptr = gpu_ptr;
        mobject->aligned.gpu_ptr = gpu_ptr;
        mobject->has_gpu_mapping = true;
    }

    mobject->base.cpu_ptr = ptr;
    mobject->base.size = size;

    mobject->aligned.cpu_ptr = ptr;
    mobject->aligned.size = size;

    mobject->has_cpu_mapping = true;

    *pmobject = mobject;

out:
    if (status) {
        if (did_host_reg) {
            cudaError_t _status = cudaHostUnregister(ptr);
            CUDA_RUNTIME_ERROR_STRING(_status);
        }
        if (ptr) free(ptr);
        if (mobject) free(mobject);
    }
    return status;
}

static void ibgda_host_mem_free(struct ibgda_mem_object *mobject) {
    cudaError_t status;

    if (!mobject) return;

    assert(mobject->mem_type == IBGDA_MEM_TYPE_HOST);

    if (mobject->has_gpu_mapping) {
        status = cudaHostUnregister(mobject->base.cpu_ptr);
        CUDA_RUNTIME_ERROR_STRING(status);
    }

    free(mobject->base.cpu_ptr);

    free(mobject);
}

static int ibgda_nic_mem_gpu_map(struct ibgda_mem_object **pmobject, struct bnxt_re_dv_devx_uar *uar,
                                 size_t size) {
    int status = 0;
    bool did_host_reg = false;

    void *ptr = 0;

    struct ibgda_mem_object *mobject =
        (struct ibgda_mem_object *)calloc(1, sizeof(struct ibgda_mem_object));
    NVSHMEMI_NULL_ERROR_JMP(mobject, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate a new mobject.\n");

    status = cudaHostRegister(
        uar->reg_addr, size,
        cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterIoMemory);

    /* TBD Multiple DPI mapping needs to be avoided.
     * Use alloc DPI instead of default DPI.
     * */
    if (status == 712)
        goto skip;

    if (status != cudaSuccess) {
        NVSHMEMI_WARN_PRINT(
            "cudaHostRegister with IoMemory failed with error=%d. We may need to use a fallback "
            "path.\n",
            status);
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

skip:
    did_host_reg = true;

    status = cudaHostGetDevicePointer(&ptr, uar->reg_addr, 0);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaHostGetDevicePointer failed.\n");

    mobject->mem_type = IBGDA_MEM_TYPE_NIC;

    mobject->base.cpu_ptr = uar->reg_addr;
    mobject->base.gpu_ptr = ptr;
    mobject->base.size = size;

    mobject->aligned.cpu_ptr = uar->reg_addr;
    mobject->aligned.gpu_ptr = ptr;
    mobject->aligned.size = size;

    mobject->uar = uar;

    mobject->has_cpu_mapping = true;
    mobject->has_gpu_mapping = true;
    mobject->has_nic_mapping = true;

    *pmobject = mobject;

out:
    if (status) {
        if (did_host_reg) {
            cudaError_t _status = cudaHostUnregister(uar->reg_addr);
            CUDA_RUNTIME_ERROR_STRING(_status);
        }
        if (mobject) free(mobject);
    }
    return status;
}

static void ibgda_nic_mem_gpu_unmap(struct ibgda_mem_object *mobject) {
    cudaError_t status;

    if (!mobject) return;

    assert(mobject->mem_type == IBGDA_MEM_TYPE_NIC);

    status = cudaHostUnregister(mobject->uar->reg_addr);
    CUDA_RUNTIME_ERROR_STRING(status);

    free(mobject);
}

static int ibgda_nic_mem_cpu_map(struct ibgda_mem_object **pmobject, struct bnxt_re_dv_devx_uar *uar,
                                 size_t size) {
    int status = 0;

    struct ibgda_mem_object *mobject =
        (struct ibgda_mem_object *)calloc(1, sizeof(struct ibgda_mem_object));
    NVSHMEMI_NULL_ERROR_JMP(mobject, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate a new mobject.\n");

    mobject->mem_type = IBGDA_MEM_TYPE_NIC;

    mobject->base.cpu_ptr = uar->reg_addr;
    mobject->base.size = size;

    mobject->aligned.cpu_ptr = uar->reg_addr;
    mobject->aligned.size = size;

    mobject->uar = uar;

    mobject->has_cpu_mapping = true;
    mobject->has_nic_mapping = true;

    *pmobject = mobject;

out:
    return status;
}

static void ibgda_nic_mem_cpu_unmap(struct ibgda_mem_object *mobject) {
    assert(mobject->mem_type == IBGDA_MEM_TYPE_NIC);

    free(mobject);
}

static inline int ibgda_nic_control_alloc(struct ibgda_mem_object **pmobject, size_t size,
                                          size_t alignment) {
    assert(ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU ||
           ibgda_nic_buf_location == IBGDA_MEM_TYPE_HOST);
    if (ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU)
        return ibgda_gpu_mem_alloc(pmobject, size, alignment, false);
    else
        return ibgda_host_mem_alloc(pmobject, size, alignment, true);
}

static inline void ibgda_nic_control_free(struct ibgda_mem_object *mobject) {
    assert(ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU ||
           ibgda_nic_buf_location == IBGDA_MEM_TYPE_HOST);
    if (ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU)
        ibgda_gpu_mem_free(mobject);
    else
        ibgda_host_mem_free(mobject);
}

static int ibgda_create_cq(struct ibgda_cq **pgcq, struct ibgda_device *device) {
    int status = 0;

    struct bnxt_re_dv_cq_init_attr cq_dv_attr = {};
    struct ibgda_cq *gcq = NULL;

    struct ibv_pd *pd = device->pd;
    struct ibv_context *context = pd->context;
    bnxt_re_dv_obj dv_obj;
    struct bnxt_re_dv_cq dvscq;

    void *cq_context;

    size_t num_cqe = IBGDA_ROUND_UP_POW2_OR_0(ibgda_qp_depth);

    struct bnxt_re_dv_umem *cq_umem = device->cq_shared_object.cq_mobject->umem;
    off_t cq_offset = device->cq_shared_object.cur_cq_off;

    struct bnxt_re_dv_umem *dbr_umem = device->cq_shared_object.dbr_mobject->umem;
    off_t dbr_offset = device->cq_shared_object.cur_dbr_off;

    gcq = (struct ibgda_cq *)calloc(1, sizeof(struct ibgda_cq));
    NVSHMEMI_NULL_ERROR_JMP(gcq, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate mem for cq.\n");

    cq_dv_attr.umem_handle = cq_umem;
    cq_dv_attr.cq_umem_offset = cq_offset;
    cq_dv_attr.ncqe = num_cqe;

    gcq->devx_cq = bnxt_re_dv_create_cq(context, &cq_dv_attr);
    NVSHMEMI_NULL_ERROR_JMP(gcq->devx_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to create CQ.\n");
    memset(&dv_obj, 0, sizeof(dv_obj));
    dv_obj.cq.in = gcq->devx_cq;
    dv_obj.cq.out = &dvscq;

    status = bnxt_re_dv_init_obj(&dv_obj, BNXT_RE_DV_OBJ_CQ);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "DV init object CQ failed.\n");

    gcq->cqn = dvscq.cqn;
    gcq->num_cqe = num_cqe;
    gcq->cq_mobject = device->cq_shared_object.cq_mobject;
    gcq->cq_offset = cq_offset;
    gcq->dbr_mobject = device->cq_shared_object.dbr_mobject;
    gcq->dbr_offset = dbr_offset;

    device->cq_shared_object.cur_cq_off += device->cq_shared_object.cq_buf_size_per_cq;
    device->cq_shared_object.cur_dbr_off += IBGDA_DBRSIZE;

    *pgcq = gcq;

out:
    if (status) {
        if (gcq) free(gcq);
    }
    return status;
}

static void ibgda_destroy_cq(struct ibgda_cq *gcq) {
    if (!gcq) return;

    if (gcq->devx_cq) {
        bnxt_re_dv_destroy_cq(gcq->devx_cq);
    }

    free(gcq);
}

static void ibgda_get_device_cq(nvshmemi_ibgda_device_cq_t *dev_cq, const struct ibgda_cq *cq) {
    dev_cq->cqn = cq->cqn;
    dev_cq->ncqes = cq->num_cqe;

    assert(cq->cq_mobject->has_gpu_mapping);
    dev_cq->cqe = (void *)((uintptr_t)cq->cq_mobject->aligned.gpu_ptr + cq->cq_offset);

    assert(cq->dbr_mobject->has_gpu_mapping);
    dev_cq->dbrec = (__be32 *)((uintptr_t)cq->dbr_mobject->aligned.gpu_ptr + cq->dbr_offset);
}

static int ibgda_qp_rst2init(struct ibgda_ep *ep, const struct ibgda_device *device, int portid) {
    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);
    struct ibv_qp* ib_qp = ep->devx_qp;
    struct ibv_qp_attr ib_qp_attr;
    struct ib_uverbs_qp_attr ib_qp_uattr;
    int status = 0, flags;

    // RST2INIT
    memset(&ib_qp_attr, 0, sizeof(ib_qp_attr));
    ib_qp_attr.qp_state = IBV_QPS_INIT;
    ib_qp_attr.pkey_index = 0;
    ib_qp_attr.port_num = portid;
    ib_qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
    flags = (IBV_QP_STATE | IBV_QP_PKEY_INDEX |
             IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);

    status = bnxt_re_dv_modify_qp(ib_qp, &ib_qp_attr, flags, 0, 0);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                         "bnxt_re_dv_modify_qp rst2init for RC failed.\n");
    ep->portid = portid;

out:
    return status;
}

static int ibgda_rc_init2rtr(nvshmemt_ibgda_state_t *ibgda_state, struct ibgda_ep *ep,
                             const struct ibgda_device *device, int portid,
                             struct ibgda_rc_handle *peer_ep_handle) {
    struct ibv_qp_attr ib_qp_attr = {};
    int status = 0, flags;

    const struct ibv_port_attr *port_attr = device->port_attr + (portid - 1);

    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    if (port_attr->link_layer == IBV_LINK_LAYER_ETHERNET) {

        const char *nic_device_name = ftable.get_device_name(device->context->device);

        ib_get_gid_index(&ftable, device->context, portid, port_attr->gid_tbl_len,
                         (int *)&device->gid_info[portid - 1].local_gid_index,
                         ibgda_state->log_level, ibgda_state->options);
        ftable.query_gid(device->context, portid, device->gid_info[portid - 1].local_gid_index,
                         (ibv_gid *)&device->gid_info[portid - 1].local_gid);

        ib_qp_attr.alt_ah_attr.is_global = 0;

        ib_qp_attr.qp_state = IBV_QPS_RTR;
        ib_qp_attr.path_mtu = port_attr->active_mtu;
        ib_qp_attr.min_rnr_timer = 12;
        ib_qp_attr.dest_qp_num = peer_ep_handle->qpn;
        ib_qp_attr.rq_psn = 0;
        //ib_qp_uattr.max_dest_rd_atomic = NVSHMEMT_IBRC_MAX_RD_ATOMIC;
        ib_qp_attr.max_dest_rd_atomic = 126;
        ib_qp_attr.ah_attr.sl = ibgda_state->options->IB_SL;
        ib_qp_attr.ah_attr.src_path_bits = 0;
        ib_qp_attr.ah_attr.port_num = portid;
        /* TBD - ROCEV2 only */
        ib_qp_attr.ah_attr.dlid = port_attr->lid | (IBGDA_ROCE_V2_UDP_SPORT_BASE);
        ib_qp_attr.ah_attr.is_global = 1;
        ib_qp_attr.ah_attr.grh.dgid.global.subnet_prefix = peer_ep_handle->spn;
        ib_qp_attr.ah_attr.grh.dgid.global.interface_id = peer_ep_handle->iid;
        ib_qp_attr.ah_attr.grh.sgid_index = device->gid_info[portid - 1].local_gid_index;
        flags = (IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                 IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MIN_RNR_TIMER |
                 IBV_QP_MAX_DEST_RD_ATOMIC);
    }

    /* MSN Table */
    ep->mtu = 0x80 << port_attr->active_mtu;
    status = bnxt_re_dv_modify_qp(ep->devx_qp, &ib_qp_attr, flags, 0, 0);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Error in bnxt_re_dv_modify_qp for INIT2RTR_QP peer qpn %d \n", ep->qpn);

out:
    return status;
}

static int ibgda_qp_rtr2rts(struct ibgda_ep *ep, const struct ibgda_device *device, int portid) {
    int status = 0, flags;
    struct ibv_qp_attr ib_qp_attr = {};

    memset(&ib_qp_attr, 0, sizeof(struct ibv_qp_attr));
    ib_qp_attr.qp_state = IBV_QPS_RTS;
    ib_qp_attr.sq_psn = 0;
    ib_qp_attr.timeout = 20;
    ib_qp_attr.retry_cnt = 7;
    ib_qp_attr.rnr_retry = 7;
    ib_qp_attr.max_rd_atomic = 7;

    flags = (IBV_QP_STATE | IBV_QP_SQ_PSN |
                                IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);

    status = bnxt_re_dv_modify_qp(ep->devx_qp, &ib_qp_attr, flags, 0, 0);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Error in bnxt_re_dv_modify_qp for RTR2RTS_QP peer qpn %d \n", ep->qpn);

    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: from %s %d \n", __func__, __LINE__);
out:
    return status;
}

static int ibgda_destroy_internal_buffer(nvshmemt_ibgda_state_t *ibgda_state,
                                         struct ibgda_device *device) {
    int status = 0;

    struct ibgda_mem_object *internal_buf_mobject = NULL;
    struct nvshmemt_ib_common_mem_handle *internal_buf_mhandle = NULL;

    internal_buf_mobject = device->qp_shared_object.internal_buf.mem_object;
    internal_buf_mhandle = device->qp_shared_object.internal_buf.mem_handle;

    if (internal_buf_mhandle) {
        nvshmemt_ib_common_release_mem_handle(&ftable, (nvshmem_mem_handle_t *)internal_buf_mhandle,
                                              ibgda_state->log_level);
        free(internal_buf_mhandle);
    }

    if (internal_buf_mobject) {
        ibgda_gpu_mem_free(internal_buf_mobject);
    }

    return status;
}

static int ibgda_create_internal_buffer(struct ibgda_internal_buffer *internal_buf,
                                        nvshmemt_ibgda_state_t *ibgda_state,
                                        struct ibgda_device *device, int n_pes) {
    int status = 0;

    struct ibgda_mem_object *internal_buf_mobject = NULL;
    struct nvshmemt_ib_common_mem_handle *internal_buf_mhandle = NULL;

    size_t size_per_rc =
        NVSHMEMI_IBGDA_IBUF_SLOT_SIZE * (ibgda_num_fetch_slots_per_rc + IBGDA_IBUF_RESERVED_SLOTS);
    size_t buf_size =
        size_per_rc * device->rc.num_eps_per_pe * n_pes;

    status = ibgda_gpu_mem_alloc(&internal_buf_mobject, buf_size, IBGDA_GPAGE_SIZE, false);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "cannot allocate internal buffer.\n");

    internal_buf_mhandle =
        (struct nvshmemt_ib_common_mem_handle *)calloc(1, sizeof(*internal_buf_mhandle));
    NVSHMEMI_NULL_ERROR_JMP(internal_buf_mhandle, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate internal_buf_mhandle.\n");

    status = nvshmemt_ib_common_reg_mem_handle(
        &ftable, device->pd, (nvshmem_mem_handle_t *)internal_buf_mhandle,
        (void *)internal_buf_mobject->aligned.gpu_ptr, internal_buf_mobject->aligned.size, false,
        ibgda_state->dmabuf_support_for_data_buffers, ibgda_cuda_syms, ibgda_state->log_level,
        ibgda_state->options->IB_ENABLE_RELAXED_ORDERING);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to register memory for IBGDA transport.\n");

    internal_buf->mem_object = internal_buf_mobject;
    internal_buf->mem_handle = internal_buf_mhandle;

out:
    if (status) {
        if (internal_buf_mhandle) {
            nvshmemt_ib_common_release_mem_handle(
                &ftable, (nvshmem_mem_handle_t *)internal_buf_mhandle, ibgda_state->log_level);
            free(internal_buf_mhandle);
        }
        if (internal_buf_mobject) ibgda_gpu_mem_free(internal_buf_mobject);
    }
    return status;
}

static void ibgda_destroy_cq_shared_objects(nvshmemt_ibgda_state_t *ibgda_state,
                                            struct ibgda_device *device) {
    if (device->cq_shared_object.dbr_mobject) {
        if (device->cq_shared_object.dbr_mobject->has_nic_mapping)
            ibgda_mobject_nic_unmap(device->cq_shared_object.dbr_mobject);
        ibgda_nic_control_free(device->cq_shared_object.dbr_mobject);
    }

    if (device->cq_shared_object.cq_mobject) {
        if (device->cq_shared_object.cq_mobject->has_nic_mapping)
            ibgda_mobject_nic_unmap(device->cq_shared_object.cq_mobject);
        ibgda_nic_control_free(device->cq_shared_object.cq_mobject);
    }
}

static int ibgda_destroy_qp_shared_objects(nvshmemt_ibgda_state_t *ibgda_state,
                                           struct ibgda_device *device) {
    int status = 0;

    status = ibgda_destroy_internal_buffer(ibgda_state, device);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibgda_destroy_internal_buffer failed.\n");

    if (device->qp_shared_object.recv_cq) {
        status = ftable.destroy_cq(device->qp_shared_object.recv_cq);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_destroy_cq failed for recv_cq.\n");
    }

    if (device->qp_shared_object.srq) {
        status = ftable.destroy_srq(device->qp_shared_object.srq);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "destroy_srq failed.\n");
    }

    if (device->qp_shared_object.prod_idx_mobject)
        ibgda_gpu_mem_free(device->qp_shared_object.prod_idx_mobject);

    if (device->qp_shared_object.prod_idx_cache) free(device->qp_shared_object.prod_idx_cache);

    if (device->qp_shared_object.prod_idx_snapshot)
        free(device->qp_shared_object.prod_idx_snapshot);

    if (device->qp_shared_object.dbr_mobject) {
        if (device->qp_shared_object.dbr_mobject->has_nic_mapping)
            ibgda_mobject_nic_unmap(device->qp_shared_object.dbr_mobject);
        if (ibgda_nic_handler == IBGDA_NIC_HANDLER_GPU)
            ibgda_nic_control_free(device->qp_shared_object.dbr_mobject);
        else
            ibgda_host_mem_free(device->qp_shared_object.dbr_mobject);
    }

    if (device->qp_shared_object.wq_mobject) {
        if (device->qp_shared_object.wq_mobject->has_nic_mapping)
            ibgda_mobject_nic_unmap(device->qp_shared_object.wq_mobject);
        ibgda_nic_control_free(device->qp_shared_object.wq_mobject);
    }

    if (device->qp_shared_object.rq_mobject) {
        if (device->qp_shared_object.rq_mobject->has_nic_mapping)
            ibgda_mobject_nic_unmap(device->qp_shared_object.rq_mobject);
        ibgda_nic_control_free(device->qp_shared_object.rq_mobject);
    }

out:
    return status;
}

static int ibgda_create_cq_shared_objects(nvshmemt_ibgda_state_t *ibgda_state,
                                          struct ibgda_device *device, int n_pes) {
    int status = 0;

    struct ibv_context *context = device->context;

    // Each RC qp has one send CQ and one recv CQ.
    unsigned int num_cqs = device->rc.num_eps_per_pe * n_pes * 2;

    assert(ibgda_qp_depth > 0);
    /* BNXT specific check */
    assert(num_cqs > 0);
    size_t num_cqe = IBGDA_ROUND_UP_POW2_OR_0(ibgda_qp_depth);
    size_t cq_buf_size_per_cq = num_cqe * NVSHMEMI_IBGDA_CQE_SIZE;
    size_t cq_buf_size = num_cqs * cq_buf_size_per_cq;

    size_t dbr_buf_size = IBGDA_DBRSIZE * num_cqs;

    struct ibgda_mem_object *cq_mobject = NULL;
    struct ibgda_mem_object *dbr_mobject = NULL;

    // Allocate and map CQ buffer for all CQs.
    status = ibgda_nic_control_alloc(&cq_mobject, cq_buf_size, IBGDA_GPAGE_SIZE);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot allocate cq buf.\n");

    status = cudaMemset(cq_mobject->base.gpu_ptr, 0x00, cq_mobject->base.size);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "cudaMemset failed.\n");

    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: cq_mobject from %s %d \n",
        __func__, __LINE__);
    status = ibgda_mobject_nic_map(cq_mobject, context, IBV_ACCESS_LOCAL_WRITE,
                                   ibgda_state->dmabuf_support_for_control_buffers);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot register cq buf.\n");

    // Allocate and map Doorbell Record buffer for all CQs.
    status = ibgda_nic_control_alloc(&dbr_mobject, dbr_buf_size, IBGDA_GPAGE_SIZE);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot allocate dbr buf.\n");

    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: CQ dbr_mobject from %s %d \n",
        __func__, __LINE__);
    status = ibgda_mobject_nic_map(dbr_mobject, context, IBV_ACCESS_LOCAL_WRITE,
                                   ibgda_state->dmabuf_support_for_control_buffers);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot register dbr buf.\n");

    // Output
    device->cq_shared_object.cq_buf_size_per_cq = cq_buf_size_per_cq;
    device->cq_shared_object.cq_mobject = cq_mobject;
    device->cq_shared_object.dbr_mobject = dbr_mobject;

out:
    if (status) {
        if (dbr_mobject) {
            if (dbr_mobject->has_nic_mapping) ibgda_mobject_nic_unmap(dbr_mobject);
            ibgda_nic_control_free(dbr_mobject);
        }
        if (cq_mobject) {
            if (cq_mobject->has_nic_mapping) ibgda_mobject_nic_unmap(cq_mobject);
            ibgda_nic_control_free(cq_mobject);
        }
    }
    return status;
}

static int ibgda_create_qp_shared_objects(nvshmemt_ibgda_state_t *ibgda_state,
                                          struct ibgda_device *device, int n_pes) {
    int status = 0;

    struct ibv_context *context = device->context;
    struct ibv_pd *pd = device->pd;

    struct ibv_srq *srq = NULL;
    struct ibv_srq_init_attr srq_init_attr;

    struct ibv_cq *recv_cq = NULL;

    struct ibgda_mem_object *prod_idx_mobject = NULL;
    uint64_t *prod_idx_cache = NULL;
    uint64_t *prod_idx_snapshot = NULL;
    unsigned int num_eps = device->rc.num_eps_per_pe * n_pes;

    bnxt_re_dv_obj dv_obj;
    struct bnxt_re_dv_pd dvpd;
    struct bnxt_re_dv_cq dvscq;
    struct bnxt_re_dv_cq dvrcq;
    struct bnxt_re_dv_srq dvsrq;

    struct bnxt_re_dv_qp_init_attr dv_qp_attr = {};
    struct bnxt_re_dv_qp_mem_info qp_mem = {};
    struct ibv_qp_init_attr attr = {};

    int pdn = 0;
    int srqn = 0;
    int rcqn = 0;
    int nslots;
    int psn_nslots;

    assert(ibgda_qp_depth > 0);
    size_t num_wqebb = IBGDA_ROUND_UP_POW2_OR_0(ibgda_qp_depth);

    size_t wq_buf_size_per_qp;
    size_t wq_buf_size;
    size_t rq_buf_size_per_qp;
    size_t rq_buf_size;
    struct ibgda_mem_object *wq_mobject = NULL;
    struct ibgda_mem_object *rq_mobject = NULL;

    size_t dbr_buf_size;
    struct ibgda_mem_object *dbr_mobject = NULL;

    // Initialization
    memset(&srq_init_attr, 0, sizeof(srq_init_attr));
    memset(&dvpd, 0, sizeof(dvpd));
    memset(&dvscq, 0, sizeof(dvscq));
    memset(&dvrcq, 0, sizeof(dvrcq));
    memset(&dvsrq, 0, sizeof(dvsrq));

    // Query pdn
    memset(&dv_obj, 0, sizeof(dv_obj));
    dv_obj.pd.in = pd;
    dv_obj.pd.out = &dvpd;

    status = bnxt_re_dv_init_obj(&dv_obj, BNXT_RE_DV_OBJ_PD);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "bnxt_re_dv PD initialization failed.\n");

    pdn = dvpd.pdn;

    // Create srq on host memory.
    srq_init_attr.attr.max_wr = ibgda_srq_depth;
    srq_init_attr.attr.max_sge = 1;

    srq = ftable.create_srq(pd, &srq_init_attr);
    NVSHMEMI_NULL_ERROR_JMP(srq, status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_create_srq failed.\n");

    memset(&dv_obj, 0, sizeof(dv_obj));
    //dvsrq.comp_mask = BNXT_RE_DV_SRQ_MASK_SRQN;
    dv_obj.srq.in = srq;
    dv_obj.srq.out = &dvsrq;

    status = bnxt_re_dv_init_obj(&dv_obj, BNXT_RE_DV_OBJ_SRQ);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "bnxt_re_dv SRQ initialization failed.\n");

    srqn = dvsrq.srqn;
    NVSHMEMI_EQ_ERROR_JMP(srqn, 0, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to allocate SRQ for your device. "
                          "This may occur if your ofed is older than version 5.0.\n");

    // Create recv_cq on host memory.
    recv_cq = ftable.create_cq(context, ibgda_srq_depth, NULL, NULL, 0);
    NVSHMEMI_NULL_ERROR_JMP(recv_cq, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "ibv_create_cq for recv_cq failed.\n");

    memset(&dv_obj, 0, sizeof(dv_obj));
    dv_obj.cq.in = recv_cq;
    dv_obj.cq.out = &dvrcq;

    status = bnxt_re_dv_init_obj(&dv_obj, BNXT_RE_DV_OBJ_CQ);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "bnxt_re_dv RCQ initialization failed.\n");

    rcqn = dvrcq.cqn;

    status = ibgda_create_internal_buffer(&device->qp_shared_object.internal_buf, ibgda_state,
                                          device, n_pes);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibgda_create_internal_buffer failed.\n");

    if (ibgda_nic_handler == IBGDA_NIC_HANDLER_CPU) {
        status = ibgda_gpu_mem_alloc(&prod_idx_mobject, sizeof(uint64_t) * num_eps,
                                     IBGDA_GPAGE_SIZE, true);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "cannot allocate prod_idx_mobject.\n");

        prod_idx_cache = (uint64_t *)calloc(num_eps, sizeof(uint64_t));
        NVSHMEMI_NULL_ERROR_JMP(prod_idx_cache, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "Unable to allocate mem for prod_idx_cache.\n");

        prod_idx_snapshot = (uint64_t *)calloc(num_eps, sizeof(uint64_t));
        NVSHMEMI_NULL_ERROR_JMP(prod_idx_snapshot, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "Unable to allocate mem for prod_idx_snapshot.\n");
    }

    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: pdn %d srqn %d rcqn %d from %s %d \n",
                        pdn, srqn, rcqn, __func__, __LINE__);

    // Allocate and map WQ buffer for all QPs.
    dv_qp_attr.qp_type = IBV_QPT_RC;
    dv_qp_attr.max_send_wr = num_wqebb;  // num_wqebb is always a power of 2
    dv_qp_attr.max_recv_wr = 1;
    dv_qp_attr.max_send_sge = NVSHMEMI_IBGDA_BNXT_SEND_SGE;
    dv_qp_attr.max_recv_sge = NVSHMEMI_IBGDA_BNXT_RECV_SGE;
    dv_qp_attr.max_inline_data = NVSHMEMI_IBGDA_BNXT_MAX_INLINE_SIZE;

    /* Try max case scenario for now.
     * IBGDA_GPAGE_SIZE is 64K so need to handle all the cases.
     * For each wqe max possible slots are 15.
     * Current implemenation restrict max slots to 4 for IBGDA.
     *
     * */
    nslots = num_wqebb * BNXT_RE_STATIC_WQE_SIZE_SLOTS;
    psn_nslots = IBGDA_ROUND_UP_POW2_OR_0(nslots);

    /* slots size is 16 bytes. One PSN entry size is 8 bytes */
    wq_buf_size_per_qp = (nslots * 16) + (psn_nslots * 8);

    /* Alignment */
    wq_buf_size_per_qp = (wq_buf_size_per_qp + IBGDA_GPAGE_SIZE - 1)  & ~(IBGDA_GPAGE_SIZE - 1);
    wq_buf_size = wq_buf_size_per_qp * num_eps;

    status = ibgda_nic_control_alloc(&wq_mobject, wq_buf_size, IBGDA_GPAGE_SIZE);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot allocate wq buf.\n");

    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: QP wq_mobject from %s %d \n",
                        __func__, __LINE__);
    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: wq_buf_size (%d * %d * %d) = %d -> (0x%x * %d) = %d"
                        " slots 0x%x psn_slots 0x%x\n",
                        num_wqebb, BNXT_SEND_WQE_BB, num_eps,
                        (num_wqebb * BNXT_SEND_WQE_BB * num_eps),
                        wq_buf_size_per_qp, num_eps,
                        wq_buf_size, nslots, psn_nslots);

    wq_mobject->qp_mem.qp_handle = 0;
    wq_mobject->qp_mem.sq_len = wq_buf_size_per_qp;
    wq_mobject->qp_mem.sq_slots = nslots;
    wq_mobject->qp_mem.sq_wqe_sz = 0x40; /* Max sq wqe size based on 4 sges */
    wq_mobject->qp_mem.sq_psn_sz = 0x8;
    wq_mobject->qp_mem.sq_npsn = psn_nslots;

    status = ibgda_mobject_nic_map(wq_mobject, context, IBV_ACCESS_LOCAL_WRITE,
                                   ibgda_state->dmabuf_support_for_control_buffers);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot register wq buf.\n");

    // Allocate and map RQ buffer for all QPs.
    rq_buf_size_per_qp = num_wqebb * BNXT_SEND_WQE_BB;  // num_wqebb is always a power of 2
    rq_buf_size = rq_buf_size_per_qp * num_eps;

    /* TBD.
     * 2 slots for header + 2 slots per sge. */
    nslots = (2 + NVSHMEMI_IBGDA_BNXT_RECV_SGE) * num_wqebb;
    /* slots size is 16 bytes */
    rq_buf_size_per_qp = (nslots * 16);
    /* Alignment */
    rq_buf_size_per_qp = (rq_buf_size_per_qp + IBGDA_GPAGE_SIZE - 1) & ~(IBGDA_GPAGE_SIZE - 1);
    rq_buf_size = rq_buf_size_per_qp * num_eps;

    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: rq_buf_size 0x%x nslots 0x%x \n",
                        rq_buf_size_per_qp, nslots);

    status = ibgda_nic_control_alloc(&rq_mobject, rq_buf_size, IBGDA_GPAGE_SIZE);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot allocate wq buf.\n");

    rq_mobject->qp_mem.rq_len = rq_buf_size_per_qp;
    rq_mobject->qp_mem.rq_slots = nslots;
    rq_mobject->qp_mem.rq_wqe_sz = 0x40; /* size based on 2 sge */

    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: QP rq_mobject from %s %d \n",
                        __func__, __LINE__);
    status = ibgda_mobject_nic_map(rq_mobject, context, IBV_ACCESS_LOCAL_WRITE,
                                   ibgda_state->dmabuf_support_for_control_buffers);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot register wq buf.\n");

    // Allocate and map Doorbell Record buffer for all QPs.
    dbr_buf_size = IBGDA_DBRSIZE * num_eps;
    if (ibgda_nic_handler == IBGDA_NIC_HANDLER_GPU)
        status = ibgda_nic_control_alloc(&dbr_mobject, dbr_buf_size, IBGDA_GPAGE_SIZE);
    else
        status = ibgda_host_mem_alloc(&dbr_mobject, dbr_buf_size, IBGDA_GPAGE_SIZE, true);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot allocate dbr buf.\n");

    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: QP dbr_mobject from %s %d \n",
                        __func__, __LINE__);
    status = ibgda_mobject_nic_map(dbr_mobject, context, IBV_ACCESS_LOCAL_WRITE,
                                   ibgda_state->dmabuf_support_for_control_buffers);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cannot register dbr buf.\n");

    // Output
    device->qp_shared_object.srq = srq;
    device->qp_shared_object.recv_cq = recv_cq;
    device->qp_shared_object.pdn = pdn;
    device->qp_shared_object.srqn = srqn;
    device->qp_shared_object.rcqn = rcqn;
    device->qp_shared_object.prod_idx_mobject = prod_idx_mobject;
    device->qp_shared_object.prod_idx_cache = prod_idx_cache;
    device->qp_shared_object.prod_idx_snapshot = prod_idx_snapshot;
    device->qp_shared_object.wq_buf_size_per_qp = wq_buf_size_per_qp;
    device->qp_shared_object.wq_mobject = wq_mobject;
    device->qp_shared_object.rq_buf_size_per_qp = rq_buf_size_per_qp;
    device->qp_shared_object.rq_mobject = rq_mobject;
    device->qp_shared_object.dbr_mobject = dbr_mobject;

out:
    if (status) {
        if (dbr_mobject) {
            if (dbr_mobject->has_nic_mapping) ibgda_mobject_nic_unmap(dbr_mobject);
            ibgda_nic_control_free(dbr_mobject);
        }
        if (wq_mobject) {
            if (wq_mobject->has_nic_mapping) ibgda_mobject_nic_unmap(wq_mobject);
            ibgda_nic_control_free(wq_mobject);
        }
        if (rq_mobject) {
            if (rq_mobject->has_nic_mapping) ibgda_mobject_nic_unmap(rq_mobject);
            ibgda_nic_control_free(rq_mobject);
        }
        if (recv_cq) ftable.destroy_cq(recv_cq);
        if (srq) ftable.destroy_srq(srq);
        if (prod_idx_mobject) ibgda_gpu_mem_free(prod_idx_mobject);
        if (prod_idx_cache) free(prod_idx_cache);
        if (prod_idx_snapshot) free(prod_idx_snapshot);
    }
    return status;
}

static int ibgda_alloc_and_map_qp_uar(struct ibv_context *context, ibgda_nic_handler_t handler,
                                      struct ibgda_mem_object **out_mobject) {
    int status = 0;

    struct bnxt_re_dv_devx_uar *uar = NULL;
    struct ibgda_mem_object *uar_mobject = NULL;
    size_t uar_reg_size = 0;
    uint8_t log_bf_reg_size = 0;

    struct bnxt_re_dv_db_region_attr attr = {};
    memset(&attr, 0, sizeof(struct bnxt_re_dv_db_region_attr));

    /* allocate host memory for rc, cq start */
    uar = (struct bnxt_re_dv_devx_uar *)malloc(sizeof(struct bnxt_re_dv_devx_uar));
    NVSHMEMI_NULL_ERROR_JMP(uar, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "bnxt_re uar err.");

    if (bnxt_re_dv_get_default_db_region(context, &attr)) {
        NVSHMEMI_WARN_PRINT(
            "bnxt_re_dv_get_default_dbr failed.\n");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }
    uar->reg_addr = (void*)attr.dbr;
    uar->dpi_idx = attr.dpi;
    uar->context  = context;

    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: dpi 0x%lx dpi_id 0x%x from %s %d \n",
                        uar->reg_addr, uar->dpi_idx, __func__, __LINE__);
    // TBD - Hardcoding
    log_bf_reg_size = 16;
    uar_reg_size = 1LLU << log_bf_reg_size;
    uar_reg_size = 1LLU << IBGDA_BNXT_NC_UAR_SIZE;

    // Map the UAR to GPU
    if (handler == IBGDA_NIC_HANDLER_GPU) {
        status = ibgda_nic_mem_gpu_map(&uar_mobject, uar, uar_reg_size);
        if (status) {
            NVSHMEMI_WARN_PRINT(
                "ibgda_nic_mem_gpu_map failed. We may need to use the CPU fallback path.\n");
            status = NVSHMEMX_ERROR_INTERNAL;
            goto out;
        }
    } else {
        status = ibgda_nic_mem_cpu_map(&uar_mobject, uar, uar_reg_size);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_nic_mem_cpu_map failed.\n");
    }

    *out_mobject = uar_mobject;

out:
    if (status) {
        if (uar_mobject) {
            if (handler == IBGDA_NIC_HANDLER_GPU)
                ibgda_nic_mem_gpu_unmap(uar_mobject);
            else
                ibgda_nic_mem_cpu_unmap(uar_mobject);
        }
        if (uar) {
            NVSHMEMI_WARN_PRINT("IBGDA_BNXT: dpi 0x%lx dpi_id 0x%x from %s %d \n",
                uar->reg_addr, uar->dpi_idx, __func__, __LINE__);
            free(uar);
        }
    }
    return status;
}

static void ibgda_unmap_and_free_qp_uar(struct ibgda_mem_object *mobject) {
    struct bnxt_re_dv_devx_uar *uar = NULL;

    if (!mobject) return;

    uar = mobject->uar;

    if (mobject->has_gpu_mapping)
        ibgda_nic_mem_gpu_unmap(mobject);
    else
        ibgda_nic_mem_cpu_unmap(mobject);

    if (uar) {
        NVSHMEMI_WARN_PRINT("IBGDA_BNXT: dpi 0x%lx dpi_id 0x%x from %s %d \n",
            uar->reg_addr, uar->dpi_idx, __func__, __LINE__);
        free(uar);
    }
}

/**
 * Create a RC QP
 */
static int ibgda_create_qp(struct ibgda_ep **ep_ptr, struct ibgda_device *device, int portid,
                           uint32_t qp_idx, nvshmemi_ibgda_device_qp_type_t qp_type) {
    struct ibv_pd *pd = device->pd;
    struct ibv_context *context = pd->context;
    struct ibgda_ep *ep = NULL;
    struct bnxt_re_dv_qp_init_attr dv_qp_attr = {};
    struct bnxt_re_dv_qp_mem_info qp_mem = {};
    struct ibv_qp_init_attr attr = {};

    void *qp_context;

    struct ibgda_mem_object *uar_mobject = NULL;

    struct bnxt_re_dv_umem *wq_umem = NULL;
    off_t wq_offset = 0;

    struct bnxt_re_dv_umem *rq_umem = NULL;
    off_t rq_offset = 0;

    struct bnxt_re_dv_umem *dbr_umem = NULL;
    off_t dbr_offset = 0;

    int cqe_version = 0;

    struct ibgda_cq *send_cq = NULL;
    struct ibgda_cq *recv_cq = NULL;

    size_t num_wqebb = IBGDA_ROUND_UP_POW2_OR_0(ibgda_qp_depth);
    size_t num_recv_wqe = ibgda_qp_depth;
    size_t recv_wqe_size = 16;

    int status = 0;

    assert(qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    INFO(5, "IBGDA_BNXT: from %s %d", __func__, __LINE__);
    // Create send_cq on GPU memory.
    status = ibgda_create_cq(&send_cq, device);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibgda_create_cq failed.\n");

    if (qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC) {
        status = ibgda_create_cq(&recv_cq, device);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibgda_create_cq failed.\n");
    }

    ep = (struct ibgda_ep *)calloc(1, sizeof(struct ibgda_ep));
    NVSHMEMI_NULL_ERROR_JMP(ep, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate mem for ep.\n");

    // Allocate and map UAR. This will be used as a DB/BF register.
    status = ibgda_alloc_and_map_qp_uar(context, ibgda_nic_handler, &uar_mobject);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "ibgda_alloc_and_map_qp_uar failed\n");

    wq_umem = device->qp_shared_object.wq_mobject->umem;
    wq_offset = device->qp_shared_object.cur_wq_off;

    rq_umem = device->qp_shared_object.rq_mobject->umem;
    // TBD - We may need to adjust rq_offset within wq_mobject.
    rq_offset = device->qp_shared_object.cur_rq_off;

    dbr_umem = device->qp_shared_object.dbr_mobject->umem;
    dbr_offset = device->qp_shared_object.cur_dbr_off;

    dv_qp_attr.qp_type = IBV_QPT_RC;
    dv_qp_attr.max_send_wr = num_wqebb;
    // TBD - Hardcoding. DeepEP required higher recv queue size
    dv_qp_attr.max_recv_wr = num_wqebb;
    dv_qp_attr.max_send_sge = NVSHMEMI_IBGDA_BNXT_SEND_SGE;
    dv_qp_attr.max_recv_sge = NVSHMEMI_IBGDA_BNXT_RECV_SGE;
    // TBD - Hardcoding
    dv_qp_attr.max_inline_data = NVSHMEMI_IBGDA_BNXT_MAX_INLINE_SIZE;
    dv_qp_attr.sq_umem_handle = wq_umem;
    dv_qp_attr.sq_umem_offset = wq_offset;
    dv_qp_attr.rq_umem_handle = rq_umem;
    dv_qp_attr.rq_umem_offset = rq_offset;
    dv_qp_attr.send_cq = send_cq->devx_cq;

    /* TBD - recv_cq is required for current DV implementaion.
     * Improve once DV changes are available.
     * In case of DeepEP use recv_qp from DV method else
     * use from shared object.
     */
    dv_qp_attr.recv_cq = device->qp_shared_object.recv_cq;
    if (qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC)
            dv_qp_attr.recv_cq = recv_cq->devx_cq;
    else
        dv_qp_attr.srq = device->qp_shared_object.srq;

    dv_qp_attr.qp_handle = device->qp_shared_object.wq_mobject->qp_mem.qp_handle;
    dv_qp_attr.sq_len = device->qp_shared_object.wq_mobject->qp_mem.sq_len;
    dv_qp_attr.sq_slots = device->qp_shared_object.wq_mobject->qp_mem.sq_slots;
    dv_qp_attr.sq_wqe_sz = device->qp_shared_object.wq_mobject->qp_mem.sq_wqe_sz;
    dv_qp_attr.sq_psn_sz = device->qp_shared_object.wq_mobject->qp_mem.sq_psn_sz;
    dv_qp_attr.sq_npsn = device->qp_shared_object.wq_mobject->qp_mem.sq_npsn;
    dv_qp_attr.rq_len = device->qp_shared_object.rq_mobject->qp_mem.rq_len;
    dv_qp_attr.rq_slots = device->qp_shared_object.rq_mobject->qp_mem.rq_slots;
    dv_qp_attr.rq_wqe_sz = device->qp_shared_object.rq_mobject->qp_mem.rq_wqe_sz;

    INFO(5, "IBGDA_BNXT: from %s %d 0x%lx sq_len 0x%x sq_slots 0x%x sq_wqe_sz 0x%x"
                   "sq_psn_sz 0x%x sq_npsn 0x%x rq_len 0x%x rq_slots 0x%x rq_wqe_sz 0x%x\n",
                    __func__, __LINE__,
                    dv_qp_attr.qp_handle, dv_qp_attr.sq_len,
                    dv_qp_attr.sq_slots, dv_qp_attr.sq_wqe_sz,
                    dv_qp_attr.sq_psn_sz, dv_qp_attr.sq_npsn,
                    dv_qp_attr.rq_len, dv_qp_attr.rq_slots,
                    dv_qp_attr.rq_wqe_sz);

    ep->devx_qp = bnxt_re_dv_create_qp(pd, &dv_qp_attr);

    NVSHMEMI_NULL_ERROR_JMP(ep->devx_qp, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to create QP for EP.\n");
    ep->portid = portid;

    ep->sq_cnt = num_wqebb;
    ep->rq_cnt = num_recv_wqe;

    ep->wq_mobject = device->qp_shared_object.wq_mobject;
    ep->wq_offset = wq_offset;
    device->qp_shared_object.cur_wq_off += device->qp_shared_object.wq_buf_size_per_qp;

    ep->rq_mobject = device->qp_shared_object.rq_mobject;
    ep->rq_offset = rq_offset;
    device->qp_shared_object.cur_rq_off += device->qp_shared_object.rq_buf_size_per_qp;

    ep->dbr_mobject = device->qp_shared_object.dbr_mobject;
    ep->dbr_offset = dbr_offset;
    device->qp_shared_object.cur_dbr_off += IBGDA_DBRSIZE;

    ep->uar_mobject = uar_mobject;

    ep->send_cq = send_cq;
    ep->recv_cq = recv_cq;

    ep->qp_type = qp_type;

    ep->user_index = qp_idx;
    ep->qpn = ep->devx_qp->qp_num;

    // Initialize the QP specific parameters for MSN tbl
    ep->sq_psn = 0;     // Start PSN of the very first WQE
    ep->msn = 0;        // Start MSN idx of the very first WQE
    ep->msn_tbl_sz = ep->sq_cnt * BNXT_RE_STATIC_WQE_SIZE_SLOTS / 2;
    ep->pad = NULL;     // Address of the first location of the MSN table

    *ep_ptr = ep;

out:
    if (status) {
        if (uar_mobject) ibgda_unmap_and_free_qp_uar(uar_mobject);
        if (send_cq) ibgda_destroy_cq(send_cq);
        if (recv_cq) ibgda_destroy_cq(recv_cq);
        if (ep) free(ep);
    }

    return status;
}

static int ibgda_get_rc_handle(struct ibgda_rc_handle *rc_handle, const struct ibgda_ep *ep,
                               const struct ibgda_device *device) {
    const struct ibv_port_attr *port_attr = &device->port_attr[ep->portid - 1];
    const union ibv_gid *gid = &device->gid_info[ep->portid - 1].local_gid;

    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    rc_handle->qpn = ep->qpn;
    rc_handle->lid = port_attr->lid;
    if (rc_handle->lid == 0) {
        rc_handle->spn = gid->global.subnet_prefix;
        rc_handle->iid = gid->global.interface_id;

        NVSHMEMI_WARN_PRINT("IBGDA_BNXT: spn %lx iid %lx from %s %d \n",
                rc_handle->spn, rc_handle->iid, __func__, __LINE__);
    }

    return 0;
}

static int ibgda_destroy_ep(struct ibgda_ep *ep) {
    int status = 0;

    if (!ep) return status;
    if (ep->devx_qp) {
        bnxt_re_dv_destroy_qp(ep->devx_qp);
    }

    if (ep->send_cq) {
        ibgda_destroy_cq(ep->send_cq);
    }

    if (ep->recv_cq) {
        ibgda_destroy_cq(ep->recv_cq);
    }

    if (ep->ah) {
        ftable.destroy_ah(ep->ah);
    }

    free(ep);

    return status;
}

static void ibgda_get_device_qp_mvars(nvshmemi_ibgda_device_qp_management_t *dev_mvars,
                                      struct ibgda_device *device, const struct ibgda_ep *ep) {
    memset(dev_mvars, 0, sizeof(*dev_mvars));
}

static void ibgda_get_device_qp(nvshmemi_ibgda_device_qp_t *dev_qp, struct ibgda_device *device,
                                const struct ibgda_ep *ep, int selected_dev_idx) {
    uintptr_t ibuf_rc_start;
    void *ibuf_ptr = NULL;
    uint64_t key_lo, key_hi, typ_qid_indx;
    uint64_t *dpi_cpu;

    size_t size_per_rc =
        NVSHMEMI_IBGDA_IBUF_SLOT_SIZE * (ibgda_num_fetch_slots_per_rc + IBGDA_IBUF_RESERVED_SLOTS);

    assert(ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);

    dev_qp->qpn = ep->qpn;

    assert(ep->wq_mobject->has_gpu_mapping);
    // For Deep ep.
    //assert(ep->rq_mobject->has_gpu_mapping);
    dev_qp->tx_wq.wqe = (void *)((uintptr_t)ep->wq_mobject->aligned.gpu_ptr + ep->wq_offset);
    if (ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC) {
        dev_qp->rx_wq.nwqes = ep->rq_cnt;
        dev_qp->rx_wq.wqe = (void *)((uintptr_t)ep->rq_mobject->aligned.gpu_ptr + ep->rq_offset);
        dev_qp->rx_wq.bf = (void *)ep->uar_mobject->aligned.gpu_ptr;

        dpi_cpu = (uint64_t*)ep->uar_mobject->aligned.cpu_ptr;
        // TBD - When used gpu_ptr in cuda fast path, rq doorbell ring
        // is not working as expected.
        // Current code is workaround until we figure out final solution.
        // Below code will make sure rq_prod_indx is set correctly to the
        // max QD.
        key_lo = ibgda_qp_depth - 1;
        key_hi = ((ep->qpn & 0xFFFFFUL) | (0x14000000));
        typ_qid_indx = (key_lo | (key_hi << 32));
        IBGDA_WRITE_ONCE(*dpi_cpu, typ_qid_indx);
    }

    if (ibgda_nic_handler == IBGDA_NIC_HANDLER_GPU) {
        assert(ep->dbr_mobject->has_gpu_mapping);
        dev_qp->tx_wq.dbrec = (__be32 *)((uintptr_t)ep->dbr_mobject->aligned.gpu_ptr +
                                         ep->dbr_offset + sizeof(__be32));

        assert(ep->uar_mobject->has_gpu_mapping);
        dev_qp->tx_wq.bf = (void *)ep->uar_mobject->aligned.gpu_ptr;
    }

    dev_qp->tx_wq.nwqes = ep->sq_cnt;
    dev_qp->tx_wq.sq_depth = dev_qp->tx_wq.nwqes * BNXT_RE_STATIC_WQE_SIZE_SLOTS;

    ibuf_rc_start = (uintptr_t)device->qp_shared_object.internal_buf.mem_object->aligned.gpu_ptr;

    if (ep->qp_type == NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC) {
        ibuf_ptr = (void *)(ibuf_rc_start + (size_per_rc * ep->user_index));
        dev_qp->ibuf.nslots = ibgda_num_fetch_slots_per_rc;
    }

    dev_qp->ibuf.lkey = htobe32(device->qp_shared_object.internal_buf.mem_handle->lkey);
    dev_qp->ibuf.rkey = htobe32(device->qp_shared_object.internal_buf.mem_handle->rkey);
    dev_qp->ibuf.buf = ibuf_ptr;

    dev_qp->qp_type = ep->qp_type;
    dev_qp->dev_idx = selected_dev_idx;

    // Initialize the QP specific parameters for MSN tbl
    dev_qp->mtu = ep->mtu;
    dev_qp->msn_tbl_sz = ep->msn_tbl_sz;
    // First MSN tbl entry GPU VA
    dev_qp->pad = (void *)((uintptr_t)dev_qp->tx_wq.wqe +
                            (dev_qp->tx_wq.nwqes * BNXT_RE_STATIC_WQE_SIZE_SLOTS *
                            BNXT_RE_SLOT_SIZE_BB));

    ibgda_get_device_qp_mvars(&dev_qp->mvars, device, ep);
}

static int ibgda_setup_gpu_state(nvshmem_transport_t t) {
    nvshmemt_ibgda_state_t *ibgda_state;
    ibgda_state = (nvshmemt_ibgda_state_t *)t->state;

    nvshmemi_ibgda_device_state_t *ibgda_device_state_h;
    ibgda_device_state_h = (nvshmemi_ibgda_device_state_t *)t->type_specific_shared_state;

    nvshmemi_ibgda_device_qp_t *rc_d = NULL;
    nvshmemi_ibgda_device_qp_t *rc_h = NULL;

    nvshmemi_ibgda_device_cq_t *cq_d = NULL;
    nvshmemi_ibgda_device_cq_t *cq_h = NULL;

    nvshmemi_ibgda_device_cq_t *recv_cq_d = NULL;
    nvshmemi_ibgda_device_cq_t *recv_cq_h = NULL;

    uint8_t *qp_group_switches_d = NULL;

    const size_t mvars_offset = offsetof(nvshmemi_ibgda_device_qp_t, mvars);
    const size_t cq_lock_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, poll_cq_lock);
    const size_t prod_idx_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.prod_idx);
    const size_t cons_t_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.cons_idx);
    const size_t cons_t_done_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.sq_cons_idx);
    const size_t wqe_h_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.resv_head);
    const size_t wqe_t_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.ready_head);
    const size_t cq_t_ph_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.cq_phase);
    const size_t cqe_idx_t_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, tx_wq.cqe_idx);
    const size_t rx_resv_head_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, rx_wq.resv_head);
    const size_t rx_cons_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, rx_wq.cons_idx);
    const size_t rx_cq_phase_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, rx_wq.cq_phase);
    const size_t rx_cqe_idx_t_offset = offsetof(nvshmemi_ibgda_device_qp_management_t, rx_wq.cqe_idx);
    nvshmemi_ibgda_device_qp_map_type_t rc_map_type = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_INVALID;

    int n_pes = t->n_pes;
    int mype = t->my_pe;
    int n_devs_selected = ibgda_state->n_devs_selected;
    int num_qp_groups = 0;
    int num_rc_handles = 0;
    int num_cq_handles = 0;
    int status = 0;
    int cq_idx = 0;
    bool skip_cst = true;
    bool support_half_av_seg = false;

    assert(ibgda_device_state_h != 0);
    memset(ibgda_device_state_h, 0, sizeof(*ibgda_device_state_h));

    /* calculate buffer sizes and constants start */
    for (int j = 0; j < n_devs_selected; j++) {
        struct ibgda_device *device;
        int dev_idx;
        dev_idx = ibgda_state->selected_dev_ids[j];
        device = (struct ibgda_device *)ibgda_state->devices + dev_idx;
        rc_map_type = device->rc.map_by;
        skip_cst &= device->may_skip_cst;
        support_half_av_seg &= device->support_half_av_seg;
        num_rc_handles += device->rc.num_eps_per_pe * n_pes;
        num_cq_handles += (device->rc.num_eps_per_pe * (n_pes - 1) * 2);
    }
    assert(num_rc_handles >= 0);

    num_qp_groups = num_rc_handles / n_devs_selected / n_pes;
    /* calculate buffer sizes and constants end */

    recv_cq_h = (nvshmemi_ibgda_device_cq_t *)calloc(1, sizeof(*recv_cq_h));
    NVSHMEMI_NULL_ERROR_JMP(recv_cq_h, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "recv_cq calloc err.");
    nvshmemi_init_ibgda_device_cq(recv_cq_h[0]);

    /* allocate host memory for rc, cq start */
    if (num_rc_handles > 0) {
        rc_h = (nvshmemi_ibgda_device_qp_t *)calloc(num_rc_handles, sizeof(*rc_h));
        NVSHMEMI_NULL_ERROR_JMP(rc_h, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "rc calloc err.");
        /* TBD - Looks like original code used incorrect array.
         * Review and fix it later.
         */
            for (int i = 0; i < num_rc_handles; i++) {
                nvshmemi_init_ibgda_device_qp(rc_h[i]);
            }
    }

    cq_h = (nvshmemi_ibgda_device_cq_t *)calloc(num_cq_handles, sizeof(*cq_h));
    NVSHMEMI_NULL_ERROR_JMP(cq_h, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "cq calloc err.");
    for (int i = 0; i < num_cq_handles; i++) {
        nvshmemi_init_ibgda_device_cq(cq_h[i]);
    }

    if (num_rc_handles > 0) {
        status = cudaMalloc(&rc_d, num_rc_handles * sizeof(*rc_d));
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                              "rc_d cudaM err.\n");
    }

    status = cudaMalloc(&cq_d, num_cq_handles * sizeof(*cq_d));
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out, "cq cudaM err.");

    status = cudaMalloc(&qp_group_switches_d, num_qp_groups * sizeof(*qp_group_switches_d));
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
        "qp_group_switches_d cudaM err.");
    /* allocate device memory for rc, cq end */

    if (num_rc_handles > 0) {
        for (int i = 0, rc_lb = 0; i < num_rc_handles / n_devs_selected; ++i) {
            int arr_offset = i * n_devs_selected;
            /* No RC QP to self */
            if ((i / (num_rc_handles / n_devs_selected / n_pes)) == mype) {
#ifdef NVSHMEM_IBGDA_USE_RC_LOOPBACK
                // Allow loopback to itelf
                if (rc_lb++)
#else
                // Do not create loopback to self
#endif
                    continue;
            }
            for (int j = 0; j < n_devs_selected; j++) {
                int arr_idx = arr_offset + j;
                int dev_idx = ibgda_state->selected_dev_ids[j];
                struct ibgda_device *device = (struct ibgda_device *)ibgda_state->devices + dev_idx;
                uintptr_t base_mvars_d_addr = (uintptr_t)(&rc_d[arr_idx]) + mvars_offset;

                ibgda_get_device_qp(&rc_h[arr_idx], device, device->rc.eps[i], j);
                // Anything non-zero in management variables set here.
                rc_h[arr_idx].mvars.tx_wq.cq_phase = 1;
                rc_h[arr_idx].mvars.rx_wq.cq_phase = 1;

                rc_h[arr_idx].tx_wq.cq = &cq_d[cq_idx];

                ibgda_get_device_cq(&cq_h[cq_idx], device->rc.eps[i]->send_cq);
                cq_h[cq_idx].poll_cq_lock = (int *)(base_mvars_d_addr + cq_lock_offset);
                cq_h[cq_idx].cons_idx = (uint64_t *)(base_mvars_d_addr + cons_t_offset);
                cq_h[cq_idx].sq_cons_idx = (uint64_t *)(base_mvars_d_addr + cons_t_done_offset);
                cq_h[cq_idx].resv_head = (uint64_t *)(base_mvars_d_addr + wqe_h_offset);
                cq_h[cq_idx].ready_head = (uint64_t *)(base_mvars_d_addr + wqe_t_offset);
                cq_h[cq_idx].cq_phase = (uint64_t *)(base_mvars_d_addr + cq_t_ph_offset);
                cq_h[cq_idx].cqe_idx = (uint64_t *)(base_mvars_d_addr + cqe_idx_t_offset);
                cq_h[cq_idx].qpn = rc_h[arr_idx].qpn;
                cq_h[cq_idx].qp_type = rc_h[arr_idx].qp_type;
                cq_h[cq_idx].sq_size = rc_h[arr_idx].tx_wq.sq_depth;

                if (ibgda_nic_handler == IBGDA_NIC_HANDLER_GPU) {
                    rc_h[arr_idx].tx_wq.prod_idx =
                        (uint64_t *)(base_mvars_d_addr + prod_idx_offset);
                    cq_h[cq_idx].prod_idx = (uint64_t *)(base_mvars_d_addr + prod_idx_offset);
                } else {
                    rc_h[arr_idx].tx_wq.prod_idx =
                        &((uint64_t *)device->qp_shared_object.prod_idx_mobject->aligned
                              .gpu_ptr)[i];
                    cq_h[cq_idx].prod_idx = rc_h[arr_idx].tx_wq.prod_idx;
                }

                ++cq_idx;

                rc_h[arr_idx].rx_wq.cq = &cq_d[cq_idx];

                ibgda_get_device_cq(&cq_h[cq_idx], device->rc.eps[i]->recv_cq);
                cq_h[cq_idx].resv_head = (uint64_t *)(base_mvars_d_addr + rx_resv_head_offset);
                cq_h[cq_idx].cons_idx = (uint64_t *)(base_mvars_d_addr + rx_cons_offset);
                cq_h[cq_idx].cqe_idx = (uint64_t *)(base_mvars_d_addr + rx_cqe_idx_t_offset);
                cq_h[cq_idx].cq_phase = (uint64_t *)(base_mvars_d_addr + rx_cq_phase_offset);
                cq_h[cq_idx].qpn = rc_h[arr_idx].qpn;
                cq_h[cq_idx].qp_type = rc_h[arr_idx].qp_type;
                ++cq_idx;

            }
        }
    }
    cudaMemsetAsync(qp_group_switches_d, 0, num_qp_groups * sizeof(*qp_group_switches_d),
                    ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "qp_group_switches_d set err.");
    /* Get and store information for rc, cq end */

    if (num_rc_handles > 0) {
        status = cudaMemcpyAsync(rc_d, (const void *)rc_h, sizeof(*rc_h) * num_rc_handles,
                                 cudaMemcpyHostToDevice, ibgda_state->my_stream);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out, "rc copy err.");
    }

    status = cudaMemcpyAsync(cq_d, (const void *)cq_h, sizeof(*cq_h) * num_cq_handles,
                             cudaMemcpyHostToDevice, ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out, "cq copy err.");
    /* Copy host side structs to device side structs end */

    /* Post the device state start */
    ibgda_device_state_h->globalmem.qp_group_switches = qp_group_switches_d;
    ibgda_device_state_h->globalmem.rcs = rc_d;
    ibgda_device_state_h->globalmem.cqs = cq_d;

    ibgda_device_state_h->num_qp_groups = num_qp_groups;
    ibgda_device_state_h->log2_cumem_granularity = t->log2_cumem_granularity;
    ibgda_device_state_h->num_rc_per_pe = num_rc_handles / n_devs_selected / n_pes;
    ibgda_device_state_h->rc_map_type = rc_map_type;
    ibgda_device_state_h->num_requests_in_batch = ibgda_num_requests_in_batch;
    ibgda_device_state_h->support_half_av_seg = support_half_av_seg;
    ibgda_device_state_h->may_skip_cst = skip_cst;
    ibgda_device_state_h->use_async_postsend = (ibgda_nic_handler == IBGDA_NIC_HANDLER_CPU);
    ibgda_device_state_h->num_devices_initialized = n_devs_selected;
    assert(ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU ||
           ibgda_nic_buf_location == IBGDA_MEM_TYPE_HOST);
    ibgda_device_state_h->nic_buf_on_gpumem = (ibgda_nic_buf_location == IBGDA_MEM_TYPE_GPU);
    status = cudaStreamSynchronize(ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out, "stream sync err.");
    /* Post the device state start */

out:
    if (status) {
        if (cq_d) cudaFree(cq_d);
        if (rc_d) cudaFree(rc_d);
        if (qp_group_switches_d) cudaFree(qp_group_switches_d);
    }
    if (cq_h) free(cq_h);
    if (rc_h) free(rc_h);
    return status;
}

// Platform native ordering for GPUDirect RDMA writes
//  CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE = 0
//      The device does not natively support ordering of remote writes.  This would require CST
//  CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER = 100
//      Natively, the device can consistently consume remote writes, although other CUDA devices may not.
//  CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES = 200
//      Any CUDA device in the system can consistently consume remote writes to this device

static bool ibgda_cst_is_required(struct ibgda_device *device, CUdevice dev_id) {
    bool rval = true;

    int order = 0;
    if (CUPFN(ibgda_cuda_syms,
              cuDeviceGetAttribute(
                  &order, (CUdevice_attribute)CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
                  dev_id))) {
        NVSHMEMI_WARN_PRINT("Cannot query dev attr. Assuming no GDR write ordering\n");
    } else {
        // GPU guarantees incoming PCIe write ordering. No need to do CST.
        if (order >= CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER) rval = false;
    }

    return rval;
}

int nvshmemt_ibgda_connect_endpoints(nvshmem_transport_t t, int *selected_dev_ids,
                                     int num_selected_devs) {
    /* global state start */
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)t->state;
    struct nvshmemi_options_s *options = ibgda_state->options;
    int status = 0;
    /* global state end */

    if (!options->IBGDA_ENABLE_MULTI_PORT && num_selected_devs > 1) {
        INFO(ibgda_state->log_level,
             "Multi-port for IBGDA is disabled by the env. Using 1 device instead "
             "of %d.",
             num_selected_devs);
        num_selected_devs = 1;
    }

    if (num_selected_devs > NVSHMEMI_IBGDA_MAX_DEVICES_PER_PE) {
        NVSHMEMI_WARN_PRINT("IBGDA only supports %d devices, but the lib has requested %d.\n",
                            NVSHMEMI_IBGDA_MAX_DEVICES_PER_PE, num_selected_devs);
        num_selected_devs = NVSHMEMI_IBGDA_MAX_DEVICES_PER_PE;
        NVSHMEMI_WARN_PRINT("Using %d devices.\n", num_selected_devs);
    }
    /* Constants for resource creation start */
    int mype = t->my_pe;
    int n_pes = t->n_pes;
    int num_rc_eps_per_pe = options->IBGDA_NUM_RC_PER_PE;
    int num_rc_eps = num_rc_eps_per_pe * n_pes;
    /* constants for resource creation end */

    /* loop variables start */
    struct ibgda_rc_handle *local_rc_handles = NULL;
    struct ibgda_device *device = NULL;
    int curr_dev_id = 0;
    int init_dev_cnt = 0;
    int portid = 0;
    /* loop variables end */

    /* cuda info start */
    CUdevice gpu_device_id;
    int mtpb;
    int mpc;
    int warp_size;
    /* cuda info end */

    /* shared dev info start */
    nvshmemi_ibgda_device_qp_map_type_t rc_map_type = NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_INVALID;
    bool support_half_av_seg = false;
    bool skip_cst = true;
    /* shared dev info end */

    if (ibgda_state->selected_dev_ids) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out_already_connected,
                           "Device already selected. IBGDA only supports"
                           " one initialization per PE.\n");
    }

    /* Get CUDA information start */
    if (CUPFN(ibgda_cuda_syms, cuCtxGetDevice(&gpu_device_id))) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "cuCtxGetDevice failed.\n");
    }

    if (cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, gpu_device_id)) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "querying warp size failed.");
    }

    if (cudaDeviceGetAttribute(&mtpb, cudaDevAttrMaxThreadsPerBlock, gpu_device_id)) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "query max threads per block fail.");
    }

    if (cudaDeviceGetAttribute(&mpc, cudaDevAttrMultiProcessorCount, gpu_device_id)) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "query mpc count fail.");
    }
    /* Get CUDA information end */

    /* Get shared dev info start */
    status = ibgda_parse_qp_map_by(&rc_map_type, options->IBGDA_RC_MAP_BY);
    NVSHMEMI_NZ_ERROR_JMP(status, status, out, "IBGDA_RC_MAP_BY is not valid.");
    INFO(ibgda_state->log_level, "IBGDA_RC_MAP_BY is set to %s.", options->IBGDA_RC_MAP_BY);
    /* Get shared dev info end */

    /* Allocate global structs start */
    ibgda_state->selected_dev_ids = (int *)calloc(num_selected_devs, sizeof(*selected_dev_ids));
    NVSHMEMI_NULL_ERROR_JMP(ibgda_state->selected_dev_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY,
                            out, "allocation of selected_device_ids failed.\n");

    local_rc_handles = (struct ibgda_rc_handle *)calloc(num_rc_eps, sizeof(*local_rc_handles));
    NVSHMEMI_NULL_ERROR_JMP(local_rc_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "allocation of local_rc_handles failed.\n");
    /* Allocate global structs end */

    if (num_rc_eps_per_pe < 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_IBGDA_NUM_RC_PER_PE must be positive or zero.\n");
    } else if (num_rc_eps_per_pe > 0) {
        if (ibgda_num_fetch_slots_per_rc < warp_size) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                               "NVSHMEM_IBGDA_NUM_FETCH_SLOTS_PER_RC must be at least %d.\n",
                               warp_size);
        }

        switch (rc_map_type) {
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_CTA:
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_SM:
            case NVSHMEMI_IBGDA_DEVICE_QP_MAP_TYPE_WARP:
                break;
            default:
                NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                                   "NVSHMEM_IBGDA_RC_MAP_BY=%s is not supported.\n",
                                   ibgda_state->options->IBGDA_RC_MAP_BY);
                break;
        }
    }
    /* recalculate mappings for QP types end */

    /* check configured args stop */

    for (int i = 0; i < num_selected_devs; i++) {
        if (selected_dev_ids[i] < 0 || selected_dev_ids[i] >= ibgda_state->n_dev_ids) {
            NVSHMEMI_ERROR_PRINT("Invalid device ID %d.\n", selected_dev_ids[i]);
            if (i > 0) {
                goto out_already_connected;
            } else {
                goto out;
            }
        }
        curr_dev_id = ibgda_state->dev_ids[selected_dev_ids[i]];

        /* set device info start */
        device = ((struct ibgda_device *)ibgda_state->devices + curr_dev_id);
        portid = ibgda_state->port_ids[selected_dev_ids[i]];
        skip_cst &= (!ibgda_cst_is_required(device, gpu_device_id));
        device->rc.map_by = rc_map_type;
        device->rc.num_eps_per_pe = num_rc_eps_per_pe;
        /* set device info end */

        /* allocate device structs start */

        device->rc.peer_ep_handles =
            (struct ibgda_rc_handle *)calloc(num_rc_eps, sizeof(*device->rc.peer_ep_handles));
        NVSHMEMI_NULL_ERROR_JMP(device->rc.peer_ep_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY,
                                out, "allocation of rc.peer_ep_handles failed.");

        device->rc.eps = (struct ibgda_ep **)calloc(num_rc_eps, sizeof(*device->rc.eps));
        NVSHMEMI_NULL_ERROR_JMP(device->rc.eps, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                                "allocation of rc.eps failed.");
        /* allocate device structs end */

        /* create shared device objects start */
        status = ibgda_create_cq_shared_objects(ibgda_state, device, n_pes);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_create_cq_shared_objects failed.\n");

        status = ibgda_create_qp_shared_objects(ibgda_state, device, n_pes);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_create_qp_shared_objects failed.");

        /* create shared device objects end */

        /* create and assign RCs start */
        INFO(ibgda_state->log_level, "Creating %d RC QPs", device->rc.num_eps_per_pe);
        for (int i = 0, rc_lb = 0; i < num_rc_eps; ++i) {
            int dst_pe = (i + 1 + mype) % n_pes;
            int offset = i / n_pes;
            int mapped_i = dst_pe * device->rc.num_eps_per_pe + offset;
            if (dst_pe == mype) {
#ifdef NVSHMEM_IBGDA_USE_RC_LOOPBACK
                // Allow loopback to itelf
                if (rc_lb++)
#else
                // Do not create loopback to self
#endif
                    continue;
            }
            INFO(ibgda_state->log_level, "ibgda_create_qp loop i|rc_lb|mype|dst_pe|offset|mapped_i %d %d %d %d %d %d",
                 i, rc_lb, mype, dst_pe, offset, mapped_i);
            status = ibgda_create_qp(&device->rc.eps[mapped_i], device, portid, mapped_i,
                                     NVSHMEMI_IBGDA_DEVICE_QP_TYPE_RC);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_create_qp failed on RC #%d.", mapped_i);

            status = ibgda_get_rc_handle(&local_rc_handles[mapped_i], device->rc.eps[mapped_i], device);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_get_rc_handle failed on RC #%d.", mapped_i);
        }

        if (num_rc_eps) {
            status = t->boot_handle->alltoall(
                (void *)local_rc_handles, (void *)device->rc.peer_ep_handles,
                sizeof(*local_rc_handles) * device->rc.num_eps_per_pe, t->boot_handle);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "alltoall of rc failed.");
        }
        INFO(0, "IBGDA_BNXT from %s %d num_rc_eps %d", __func__, __LINE__, num_rc_eps);

        for (int i = 0, rc_lb = 0; i < num_rc_eps; ++i) {
            if (i / device->rc.num_eps_per_pe == mype) {
#ifdef NVSHMEM_IBGDA_USE_RC_LOOPBACK
                // Allow loopback to itself
                if (rc_lb++)
#else
                // No loopback to self
#endif
                    continue;
            }
            status = ibgda_qp_rst2init(device->rc.eps[i], device, portid);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_qp_rst2init failed on RC #%d.", i);

            status = ibgda_rc_init2rtr(ibgda_state, device->rc.eps[i], device, portid,
                                       &device->rc.peer_ep_handles[i]);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_rc_init2rtr failed on RC #%d.", i);

            status = ibgda_qp_rtr2rts(device->rc.eps[i], device, portid);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibgda_qp_rtr2rts failed on RC #%d.", i);
        }
        /* create and assign RCs end */

        ibgda_state->selected_dev_ids[init_dev_cnt] = curr_dev_id;
        ++init_dev_cnt;
    }

    /* Multiple devices break our CST optimizations. */
    if (init_dev_cnt > 1) {
        skip_cst = false;
    }

    /* set all device to not support_half_av_seg and need_cst start */
    for (int i = 0; i < init_dev_cnt; i++) {
        curr_dev_id = ibgda_state->selected_dev_ids[i];
        device = ((struct ibgda_device *)ibgda_state->devices + curr_dev_id);
        device->support_half_av_seg = support_half_av_seg;
        device->may_skip_cst = skip_cst;
    }
    INFO(5, "IBGDA_BNXT from %s %d num_rc_eps %d", __func__, __LINE__, num_rc_eps);
    /* set all device to not support_half_av_seg and need_cst end */

out:
    if (status) {
        if (ibgda_state->selected_dev_ids && init_dev_cnt == 0) {
            free(ibgda_state->selected_dev_ids);
            ibgda_state->selected_dev_ids = NULL;
        } else {
            status = 0;
        }
    }

    if (init_dev_cnt) {
        ibgda_state->n_devs_selected = init_dev_cnt;
        // Setup QPs / CQs on GPU.
        status = ibgda_setup_gpu_state(t);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_setup_gpu_state failed.");
    }

    if (init_dev_cnt < num_selected_devs) {
        NVSHMEMI_WARN_PRINT("Failed to initialize all selected devices. Perf may be limited.");
    }

out_already_connected:
    if (local_rc_handles) {
        free(local_rc_handles);
    }

    return status;
}

int nvshmemt_ibgda_release_mem_handle(nvshmem_mem_handle_t *mem_handle, nvshmem_transport_t t) {
    int status = 0;
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)t->state;
    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: from %s %d \n", __func__, __LINE__);

    nvshmemi_ibgda_device_state_t *ibgda_device_state =
        (nvshmemi_ibgda_device_state_t *)t->type_specific_shared_state;
    assert(ibgda_device_state != NULL);

    struct ibgda_mem_handle *ibgda_mem_handle = (struct ibgda_mem_handle *)mem_handle;
    struct nvshmemt_ib_common_mem_handle *handle =
        (struct nvshmemt_ib_common_mem_handle *)&ibgda_mem_handle->dev_mem_handles[0];
    if (handle->local_only) {
        uint32_t position = 0;
        struct ibgda_device_local_only_mhandle_cache *prev_mhandle_cache = NULL;
        struct ibgda_device_local_only_mhandle_cache *next_mhandle_cache = NULL;
        struct ibgda_device_local_only_mhandle_cache *curr_mhandle_cache = NULL;
        void *mhandle_gpu_ptr;

        // Find the position in the host-side cache.
        for (auto it = ibgda_device_local_only_mhandles.begin();
             it != ibgda_device_local_only_mhandles.end(); ++it) {
            if (it->mhandle.start == (uint64_t)handle->buf) {
                curr_mhandle_cache = &ibgda_device_local_only_mhandles.data()[position];
                if (position > 0)
                    prev_mhandle_cache = &ibgda_device_local_only_mhandles.data()[position - 1];
                if (position < ibgda_device_local_only_mhandles.size() - 1)
                    next_mhandle_cache = &ibgda_device_local_only_mhandles.data()[position + 1];
                break;
            }
            ++position;
        }
        NVSHMEMI_NULL_ERROR_JMP(curr_mhandle_cache, status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                                "mem_handle is not registered.\n");

        // Remove this element from the linked list on both host and GPU.
        if (prev_mhandle_cache) {
            if (next_mhandle_cache)
                prev_mhandle_cache->mhandle.next =
                    (nvshmemi_ibgda_device_local_only_mhandle_t *)next_mhandle_cache->dev_ptr;
            else
                prev_mhandle_cache->mhandle.next = NULL;
            mhandle_gpu_ptr = (void *)((uintptr_t)prev_mhandle_cache->dev_ptr +
                                       offsetof(nvshmemi_ibgda_device_local_only_mhandle_t, next));
            status =
                cudaMemcpyAsync(mhandle_gpu_ptr, (const void *)&prev_mhandle_cache->mhandle.next,
                                sizeof(prev_mhandle_cache->mhandle.next), cudaMemcpyHostToDevice,
                                ibgda_state->my_stream);
            NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                                  "Setting local_only_mhandle in GPU memory failed.\n");
        } else {
            // The caller will trigger device state update.
            if (next_mhandle_cache)
                ibgda_device_state->globalmem.local_only_mhandle_head =
                    (nvshmemi_ibgda_device_local_only_mhandle_t *)next_mhandle_cache->dev_ptr;
            else
                ibgda_device_state->globalmem.local_only_mhandle_head = NULL;
        }

        // Free the copy of this element on GPU.
        status = cudaFree(curr_mhandle_cache->dev_ptr);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaFree failed.\n");

        ibgda_device_local_only_mhandles.erase(ibgda_device_local_only_mhandles.begin() + position);
    }

    // TODO: Clean up non-local-only mem_handle

    for (int i = 0; i < ibgda_state->n_devs_selected; i++) {
        handle = (struct nvshmemt_ib_common_mem_handle *)&ibgda_mem_handle->dev_mem_handles[i];
        status = nvshmemt_ib_common_release_mem_handle(&ftable, (nvshmem_mem_handle_t *)handle,
                                                       ibgda_state->log_level);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmemt_ib_common_release_mem_handle failed.\n");
    }

    status = cudaStreamSynchronize(ibgda_state->my_stream);
    NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                          "stream synchronize failed.\n");

out:
    return status;
}

int nvshmemt_ibgda_finalize(nvshmem_transport_t transport) {
    struct ibgda_device *device = NULL;
    assert(transport != NULL);
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)transport->state;
    nvshmemi_ibgda_device_state_t *ibgda_device_state_h;

    int status = 0, dev_id;
    int n_pes = transport->n_pes;
    int mype = transport->my_pe;
    int num_rc_eps;

    if (!ibgda_state) {
        goto out;
    }

    ibgda_device_lkeys.clear();
    ibgda_device_rkeys.clear();

    if (ibgda_device_lkeys_d) {
        cudaFree(ibgda_device_lkeys_d);
        ibgda_device_lkeys_d = 0;
    }
    if (ibgda_device_rkeys_d) {
        cudaFree(ibgda_device_rkeys_d);
        ibgda_device_rkeys_d = 0;
    }

    ibgda_device_state_h = (nvshmemi_ibgda_device_state_t *)transport->type_specific_shared_state;
    if (ibgda_device_state_h) {
        if (ibgda_device_state_h->globalmem.cqs) cudaFree(ibgda_device_state_h->globalmem.cqs);
        if (ibgda_device_state_h->globalmem.rcs) cudaFree(ibgda_device_state_h->globalmem.rcs);
        if (ibgda_device_state_h->globalmem.qp_group_switches)
            cudaFree(ibgda_device_state_h->globalmem.qp_group_switches);
    }

    for (int i = 0, rc_lb = 0; i < ibgda_state->n_devs_selected; i++) {
        dev_id = ibgda_state->selected_dev_ids[i];
        device = ((struct ibgda_device *)ibgda_state->devices + dev_id);

        num_rc_eps = device->rc.num_eps_per_pe * n_pes;
        for (int j = 0, rc_lb = 0; j < num_rc_eps; ++j) {
            if (j / device->rc.num_eps_per_pe == mype) {
#ifdef NVSHMEM_IBGDA_USE_RC_LOOPBACK
                // Allow loopback to itself
                if (rc_lb++)
#else
                // No loopback to self
#endif
                    continue;
            }
            status = ibgda_destroy_ep(device->rc.eps[j]);
        }

        status = ibgda_destroy_qp_shared_objects(ibgda_state, device);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "ibgda_destroy_qp_shared_objects failed.\n");

        ibgda_destroy_cq_shared_objects(ibgda_state, device);
    }

    /* Free all devices, not just ones we used. */
    for (int i = 0; i < ibgda_state->n_dev_ids; i++) {
        device = (struct ibgda_device *)ibgda_state->devices + ibgda_state->dev_ids[i];
        if (device->pd) {
            status = ftable.dealloc_pd(device->pd);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_dealloc_pd failed \n");
        }

        if (device->context) {
            status = ftable.close_device(device->context);
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibv_close_device failed \n");
        }
    }

#ifdef NVSHMEM_USE_GDRCOPY
    if (use_gdrcopy) {
        nvshmemt_gdrcopy_ftable_fini(&gdrcopy_ftable, &gdr_desc, &gdrcopy_handle);
    }
#endif

    nvshmemt_ibv_ftable_fini(&ibv_handle);

    if (transport->state) {
        if (ibgda_state->selected_dev_ids) {
            free(ibgda_state->selected_dev_ids);
            ibgda_state->selected_dev_ids = NULL;
        }
        free(transport->state);
    }

    if (transport->device_pci_paths) {
        for (int i = 0; i < transport->n_devices; i++) {
            free(transport->device_pci_paths[i]);
        }
        free(transport->device_pci_paths);
    }

out:
    free(transport);
    return status;
}

int nvshmemt_ibgda_add_device_remote_mem_handles(nvshmem_transport_t t, int transport_stride,
                                                 nvshmem_mem_handle_t *mem_handles,
                                                 uint64_t heap_offset, size_t size) {
    nvshmemt_ibgda_state_t *ibgda_state = (nvshmemt_ibgda_state_t *)t->state;
    int status = 0;
    int n_pes = t->n_pes;
    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: from %s %d \n", __func__, __LINE__);

    size_t num_rkeys;

    nvshmemi_ibgda_device_state_t *ibgda_device_state;

    ibgda_device_state = (nvshmemi_ibgda_device_state_t *)t->type_specific_shared_state;
    assert(ibgda_device_state != NULL);

    static_assert(sizeof(struct nvshmemt_ib_common_mem_handle) <= NVSHMEM_MEM_HANDLE_SIZE,
                  "static_assert(sizeof(T) <= NVSHMEM_MEM_HANDLE_SIZE) failed");

    size_t num_elements;
    // size must be divisible by cumem_granularity, which is a power of 2.
    assert((size & ((1ULL << t->log2_cumem_granularity) - 1)) == 0);

    num_elements = size >> t->log2_cumem_granularity;
    while (num_elements > 0) {
        for (int i = 0; i < n_pes; ++i) {
            // sizeof(struct ibgda_mem_handle) <= sizeof(nvshmem_mem_handle_t)
            // So, we calculate the pointer with nvshmem_mem_handle_t and convert to
            // ibgda_mem_handle later.
            struct ibgda_mem_handle *gmhandle =
                (struct ibgda_mem_handle *)&mem_handles[i * transport_stride + t->index];
            for (int j = 0; j < gmhandle->num_devs; j++) {
                struct nvshmemt_ib_common_mem_handle *handle =
                    (struct nvshmemt_ib_common_mem_handle *)&gmhandle->dev_mem_handles[j];
                nvshmemi_ibgda_device_key_t device_key;
                device_key.key = htobe32(handle->rkey);
                device_key.next_addr = heap_offset + size;

                ibgda_device_rkeys.emplace_back(device_key);
            }
        }
        --num_elements;
    }

    if (ibgda_device_rkeys_d) {
        status = cudaFree(ibgda_device_rkeys_d);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "cudaFree failed.\n");
        ibgda_device_rkeys_d = 0;
    }

    num_rkeys = ibgda_device_rkeys.size();

    // For cache optimization, put rkeys in constant memory first.
    memcpy(
        ibgda_device_state->constmem.rkeys, ibgda_device_rkeys.data(),
        IBGDA_MIN(num_rkeys, NVSHMEMI_IBGDA_MAX_CONST_RKEYS) * sizeof(nvshmemi_ibgda_device_key_t));

    // Put the rest that don't fit in constant memory in global memory
    if (num_rkeys > NVSHMEMI_IBGDA_MAX_CONST_RKEYS) {
        size_t rkeys_array_size =
            sizeof(nvshmemi_ibgda_device_key_t) * (num_rkeys - NVSHMEMI_IBGDA_MAX_CONST_RKEYS);

        nvshmemi_ibgda_device_key_t *data_ptr =
            &ibgda_device_rkeys.data()[NVSHMEMI_IBGDA_MAX_CONST_RKEYS];

        status = cudaMalloc(&ibgda_device_rkeys_d, rkeys_array_size);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                              "cudaMalloc failed.\n");

        status = cudaMemcpyAsync(ibgda_device_rkeys_d, (const void *)data_ptr, rkeys_array_size,
                                 cudaMemcpyHostToDevice, ibgda_state->my_stream);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "Copying rkeys to GPU memory failed.\n");

        status = cudaStreamSynchronize(ibgda_state->my_stream);
        NVSHMEMI_NE_ERROR_JMP(status, cudaSuccess, NVSHMEMX_ERROR_INTERNAL, out,
                              "stream synchronize failed.\n");
    }

    ibgda_device_state->globalmem.rkeys = (nvshmemi_ibgda_device_key_t *)ibgda_device_rkeys_d;
out:
    if (status) {
        // Unrecoverable error
        if (ibgda_device_rkeys_d) cudaFree(ibgda_device_rkeys_d);
        ibgda_device_rkeys.clear();
    }
    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: from %s %d \n", __func__, __LINE__);
    return status;
}

static ibgda_nic_mapping_memtype_reqeust_t ibgda_parse_nic_mapping_memtype_request(
    const char *str) {
    std::string req = str;

    // Trim whitespace
    req.erase(std::remove_if(req.begin(), req.end(), ::isspace), req.end());

    // To lower case
    std::for_each(req.begin(), req.end(), [](decltype(*req.begin()) &c) { c = ::tolower(c); });

    if (req == "gpumem")
        return IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM;
    else if (req == "hostmem")
        return IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_HOSTMEM;
    else
        return IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_AUTO;
}

static int ibgda_check_nic_mapping_memtypes(nvshmemt_ibgda_state_t *ibgda_state,
                                            struct ibgda_device *device,
                                            ibgda_nic_mapping_memtype_reqeust_t request_memtype) {
    int status = 0;

    bool try_gpumem = ((request_memtype == IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_AUTO) ||
                       (request_memtype == IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM));
    bool try_hostmem = ((request_memtype == IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_AUTO) ||
                        (request_memtype == IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_HOSTMEM));

    bool can_use_gpumem = false;
    bool can_use_hostmem = false;

    struct ibgda_mem_object *mobject = NULL;

    if (try_gpumem) {
        status = ibgda_gpu_mem_alloc(&mobject, IBGDA_DBRSIZE, IBGDA_GPAGE_SIZE, false);
        if (status) goto out_try_gpumem;

        if (!ibgda_state->options->IB_DISABLE_DMABUF && ibgda_state->cuda_support_dmabuf) {
            status = ibgda_mobject_nic_map(mobject, device->context, IBV_ACCESS_LOCAL_WRITE, true);
            ibgda_state->dmabuf_support_for_control_buffers = (status == 0);
        }

        if (!ibgda_state->dmabuf_support_for_control_buffers) {
            status = ibgda_mobject_nic_map(mobject, device->context, IBV_ACCESS_LOCAL_WRITE, false);
            if (status) goto out_try_gpumem;
        }

        can_use_gpumem = true;

    out_try_gpumem:
        if (mobject) {
            if (mobject->has_nic_mapping) ibgda_mobject_nic_unmap(mobject);
            ibgda_gpu_mem_free(mobject);
        }
        mobject = NULL;
        status = 0;
    }

    if (try_hostmem) {
        status = ibgda_host_mem_alloc(&mobject, IBGDA_DBRSIZE, IBGDA_GPAGE_SIZE, true);
        if (status) goto out_try_hostmem;

        status = ibgda_mobject_nic_map(mobject, device->context, IBV_ACCESS_LOCAL_WRITE);
        if (status) goto out_try_hostmem;

        can_use_hostmem = true;

    out_try_hostmem:
        if (mobject) {
            if (mobject->has_nic_mapping) ibgda_mobject_nic_unmap(mobject);
            ibgda_host_mem_free(mobject);
        }
        mobject = NULL;
        status = 0;
    }

    device->support_nic_buf_on_gpumem = can_use_gpumem;
    device->support_nic_buf_on_hostmem = can_use_hostmem;
    NVSHMEMI_WARN_PRINT("IBGDA_BNXT: Testing umem mobject from %s %d "
                        "can_use_gpumem %d can_use_hostmem %d\n",
                        __func__, __LINE__, can_use_gpumem, can_use_hostmem);

    if (!can_use_gpumem && !can_use_hostmem) return NVSHMEMX_ERROR_NOT_SUPPORTED;

    return 0;
}

static int ibgda_check_gpu_mapping_nic_uar(struct ibgda_device *device) {
    int status = 0;

    struct ibgda_mem_object *mobject = NULL;

    status = ibgda_alloc_and_map_qp_uar(device->context, IBGDA_NIC_HANDLER_GPU, &mobject);
    if (status) {
        NVSHMEMI_WARN_PRINT(
            "ibgda_alloc_and_map_qp_uar with GPU as handler failed. We may need to enter the CPU "
            "fallback path.\n");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

out:
    if (mobject) ibgda_unmap_and_free_qp_uar(mobject);
    return status;
}

int nvshmemt_init(nvshmem_transport_t *t, struct nvshmemi_cuda_fn_table *table, int api_version) {
    struct nvshmemt_hca_info hca_list[MAX_NUM_HCAS];
    struct nvshmemt_hca_info pe_hca_mapping[MAX_NUM_PES_PER_NODE];
    struct nvshmemi_options_s *options = NULL;

    int status = 0;
    int exclude_list = 0;
    int hca_list_count = 0;
    int pe_hca_map_count = 0;
    int user_selection = 0;
    int offset = 0;
    int num_devices = 0;
    int lowest_stream_priority;
    int highest_stream_priority;
    int flag;
    uint32_t atomic_host_endian_size = 0;
    CUdevice gpu_device_id;

    struct nvshmem_transport *transport = NULL;
    nvshmemt_ibgda_state_t *ibgda_state;
    struct ibgda_device *device;
    struct ibv_device **dev_list = NULL;

    bool nic_buf_on_gpumem = true;
    bool nic_buf_on_hostmem = true;

    ibgda_nic_mapping_memtype_reqeust_t nic_mapping_memtype_request;

    ibgda_nic_handler_t nic_handler_request;
    ibgda_nic_handler_t nic_handler = IBGDA_NIC_HANDLER_GPU;

    if (NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version) != NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION) {
        NVSHMEMI_ERROR_PRINT(
            "NVSHMEM provided an incompatible version of the transport interface. "
            "This transport supports transport API major version %d. Host has %d",
            NVSHMEM_TRANSPORT_PLUGIN_MAJOR_VERSION, NVSHMEM_TRANSPORT_MAJOR_VERSION(api_version));
        return NVSHMEMX_ERROR_INVALID_VALUE;
    }

    ibgda_cuda_syms = table;

    options = (struct nvshmemi_options_s *)calloc(1, sizeof(struct nvshmemi_options_s));
    NVSHMEMI_NULL_ERROR_JMP(options, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate options stuct for ibgda transport.\n");

    status = nvshmemi_env_options_init(options);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "Unable to initialize NVSHMEM options.\n");

    transport = (struct nvshmem_transport *)malloc(sizeof(struct nvshmem_transport));
    NVSHMEMI_NULL_ERROR_JMP(transport, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "Unable to allocate transport stuct for ibgda transport.\n");
    memset(transport, 0, sizeof(struct nvshmem_transport));

    ibgda_srq_depth = options->SRQ_DEPTH;
    if (ibgda_srq_depth <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_SRQ_DEPTH must be a positive number.\n");
    }

    // TBD - Fix lower QD later.
    //ibgda_qp_depth = options->QP_DEPTH;
    ibgda_qp_depth = 8192;
    if (ibgda_qp_depth > 0) {
        ibgda_qp_depth = IBGDA_ROUND_UP_POW2_OR_0(ibgda_qp_depth);
    }
    if (ibgda_qp_depth <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_QP_DEPTH must be a positive number.\n");
    } else if (ibgda_qp_depth < NVSHMEMI_IBGDA_MIN_QP_DEPTH) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_QP_DEPTH must be at least %d.\n", NVSHMEMI_IBGDA_MIN_QP_DEPTH);
    } else if (ibgda_qp_depth > NVSHMEMI_IBGDA_MAX_QP_DEPTH) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_QP_DEPTH can be at most %d.\n", NVSHMEMI_IBGDA_MAX_QP_DEPTH);
    }

    ibgda_num_requests_in_batch = options->IBGDA_NUM_REQUESTS_IN_BATCH;
    if (ibgda_num_requests_in_batch > 0) {
        ibgda_num_requests_in_batch = IBGDA_ROUND_UP_POW2_OR_0(ibgda_num_requests_in_batch);
    }
    if (ibgda_num_requests_in_batch <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH must be a positive number.\n");
    } else if (ibgda_num_requests_in_batch > ibgda_qp_depth) {
        NVSHMEMI_ERROR_JMP(
            status, NVSHMEMX_ERROR_INVALID_VALUE, out,
            "NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH must not be larger than QP depth.\n");
    }
    ibgda_num_fetch_slots_per_rc = options->IBGDA_NUM_FETCH_SLOTS_PER_RC;
    if (ibgda_num_fetch_slots_per_rc > 0) {
        ibgda_num_fetch_slots_per_rc = IBGDA_ROUND_UP_POW2_OR_0(ibgda_num_fetch_slots_per_rc);
    }
    if (ibgda_num_fetch_slots_per_rc <= 0) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INVALID_VALUE, out,
                           "NVSHMEM_IBGDA_NUM_FETCH_SLOTS_PER_RC must be a positive number.\n");
    }

    ibgda_state = (nvshmemt_ibgda_state_t *)calloc(1, sizeof(nvshmemt_ibgda_state_t));
    NVSHMEMI_NULL_ERROR_JMP(ibgda_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "p2p state allocation failed \n");
    transport->state = (void *)ibgda_state;

    ibgda_state->log_level = nvshmemt_common_get_log_level(options);
    ibgda_state->options = options;

    if (nvshmemt_ibv_ftable_init(&ibv_handle, &ftable, ibgda_state->log_level)) {
        NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                           "Unable to dlopen libibverbs. Skipping IBGDA transport.\n");
    }

#ifdef NVSHMEM_USE_GDRCOPY
    if (options->DISABLE_GDRCOPY) {
        use_gdrcopy = false;
    } else {
        use_gdrcopy = nvshmemt_gdrcopy_ftable_init(&gdrcopy_ftable, &gdr_desc, &gdrcopy_handle,
                                                   ibgda_state->log_level);
    }
#endif

    status = CUPFN(ibgda_cuda_syms, cuCtxGetDevice(&gpu_device_id));
    if (status != CUDA_SUCCESS) {
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }
    status =
        CUPFN(ibgda_cuda_syms,
              cuDeviceGetAttribute(&flag, (CUdevice_attribute)CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
                                   gpu_device_id));
    if (status != CUDA_SUCCESS) {
        status = 0;
        cudaGetLastError();
        ibgda_state->cuda_support_dmabuf = false;
    } else {
        ibgda_state->cuda_support_dmabuf = (flag == 1);
    }

    ibgda_state->dmabuf_support_for_data_buffers = ibgda_state->cuda_support_dmabuf;
    if (options->IB_DISABLE_DMABUF) {
        ibgda_state->dmabuf_support_for_data_buffers = false;
    }

    if (ibgda_state->dmabuf_support_for_data_buffers == false) {
        if (nvshmemt_ib_common_nv_peer_mem_available() != NVSHMEMX_SUCCESS) {
            NVSHMEMI_ERROR_PRINT(
                "neither nv_peer_mem, or nvidia_peermem detected. Skipping transport.\n");
            status = NVSHMEMX_ERROR_INTERNAL;
            goto out;
        }
    }

    dev_list = ftable.get_device_list(&num_devices);
    NVSHMEMI_NULL_ERROR_JMP(dev_list, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "get_device_list failed \n");

    ibgda_state->devices = calloc(MAX_NUM_HCAS, sizeof(struct ibgda_device));
    NVSHMEMI_NULL_ERROR_JMP(ibgda_state->devices, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "get_device_list failed \n");

    ibgda_state->dev_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(ibgda_state->dev_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "malloc failed \n");

    ibgda_state->port_ids = (int *)malloc(MAX_NUM_PES_PER_NODE * sizeof(int));
    NVSHMEMI_NULL_ERROR_JMP(ibgda_state->port_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                            "malloc failed \n");
    if (options->HCA_LIST_provided) {
        user_selection = 1;
        exclude_list = (options->HCA_LIST[0] == '^');
        hca_list_count = nvshmemt_parse_hca_list(options->HCA_LIST, hca_list, MAX_NUM_HCAS,
                                                 ibgda_state->log_level);
    }

    if (options->HCA_PE_MAPPING_provided) {
        if (hca_list_count) {
            NVSHMEMI_WARN_PRINT(
                "Found conflicting parameters NVSHMEM_HCA_LIST and NVSHMEM_HCA_PE_MAPPING, "
                "ignoring "
                "NVSHMEM_HCA_PE_MAPPING \n");
        } else {
            user_selection = 1;
            pe_hca_map_count =
                nvshmemt_parse_hca_list(options->HCA_PE_MAPPING, pe_hca_mapping,
                                        MAX_NUM_PES_PER_NODE, ibgda_state->log_level);
        }
    }

    nic_mapping_memtype_request =
        ibgda_parse_nic_mapping_memtype_request(options->IBGDA_FORCE_NIC_BUF_MEMTYPE);
#ifdef NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY
    if (nic_mapping_memtype_request == IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_AUTO) {
        nic_mapping_memtype_request = IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM;
    }
    if (nic_mapping_memtype_request != IBGDA_NIC_MAPPING_MEMTYPE_REQUEST_GPUMEM) {
        NVSHMEMI_ERROR_JMP(
            status, NVSHMEMX_ERROR_NOT_SUPPORTED, out,
            "GPU-initiated communication is compiled with GPU memory support only.\n");
    }
#endif

    status = ibgda_parse_nic_handler_request(&nic_handler_request, options->IBGDA_NIC_HANDLER);
    NVSHMEMI_NZ_ERROR_JMP(status, status, out, "NVSHMEM_IBGDA_NIC_HANDLER is not valid.");

    if (!use_gdrcopy) {
        if (nic_handler_request == IBGDA_NIC_HANDLER_AUTO) {
            nic_handler_request = IBGDA_NIC_HANDLER_GPU;
        } else if (nic_handler_request == IBGDA_NIC_HANDLER_CPU) {
            NVSHMEMI_ERROR_JMP(status, NVSHMEMX_ERROR_NOT_SUPPORTED, out,
                               "NVSHMEM_IBGDA_NIC_HANDLER=cpu requires GDRCopy.\n");
        }
    }

    INFO(ibgda_state->log_level,
         "Begin - Enumerating IB devices in the system ([<dev_id, device_name, num_ports>]) - \n");
    for (int i = 0; i < num_devices; i++) {
        device = (struct ibgda_device *)ibgda_state->devices + i;
        device->dev = dev_list[i];

        device->context = ftable.open_device(device->dev);
        if (!device->context) {
            INFO(ibgda_state->log_level, "open_device failed for IB device at index %d\n", i);
            continue;
        }

        const char *name = ftable.get_device_name(device->dev);
        NVSHMEMI_NULL_ERROR_JMP(name, status, NVSHMEMX_ERROR_INTERNAL, out,
                                "ibv_get_device_name failed \n");
        if (!strstr(name, "bnxt_re")) {
            ftable.close_device(device->context);
            device->context = NULL;
            NVSHMEMI_WARN_PRINT("device %s log_level %d is not enumerated as an bnxt_re device. Skipping...\n",
                                name, ibgda_state->log_level);
            continue;
        }

        status = ftable.query_device(device->context, &device->device_attr);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "ibv_query_device failed \n");

        //if (!nvshmemt_ib_common_query_bnxt_re_caps(device->context)) {
        if (0) {
            ftable.close_device(device->context);
            device->context = NULL;
            NVSHMEMI_WARN_PRINT("device %s is not enumerated as an bnxt_re device. Skipping...\n",
                                name);
            continue;
        }

        if (nic_handler_request == IBGDA_NIC_HANDLER_CPU) {
            device->nic_handler = IBGDA_NIC_HANDLER_CPU;
        } else {
            status = ibgda_check_gpu_mapping_nic_uar(device);
            device->nic_handler = status ? IBGDA_NIC_HANDLER_CPU : IBGDA_NIC_HANDLER_GPU;
            if (status && nic_handler_request == IBGDA_NIC_HANDLER_GPU) {
                ftable.close_device(device->context);
                device->context = NULL;
                NVSHMEMI_WARN_PRINT("GPU cannot map UAR of device %s. Skipping...\n", name);
                continue;
            }
        }

        status = ibgda_check_nic_mapping_memtypes(ibgda_state, device, nic_mapping_memtype_request);
        if (status) {
            ftable.close_device(device->context);
            device->context = NULL;
            NVSHMEMI_WARN_PRINT(
                "device %s cannot allocate buffer on the specified memory type. Skipping...\n",
                name);
            continue;
        }

        if (device->support_nic_buf_on_gpumem && !ibgda_state->options->IB_DISABLE_DMABUF &&
            !ibgda_state->dmabuf_support_for_control_buffers) {
            INFO(ibgda_state->log_level,
                 "The system does not support registering the NIC control buffers with DMABUF. "
                 "Fallback to use either nv_peer_mem or nvidia_peermem.\n");
        }

        //status = nvshmemt_ib_common_check_nic_ext_atomic_support(device->context);
        //if (status) {
        if (0) {
            ftable.close_device(device->context);
            device->context = NULL;
            NVSHMEMI_WARN_PRINT(
                "device %s does not support all necessary atomic operations. You may want to check "
                "the PCI_ATOMIC_MODE value in the NIC firmware. Skipping...\n",
                name);
            continue;
        }

        INFO(ibgda_state->log_level,
             "Enumerated IB devices in the system - device id=%d (of %d), name=%s, num_ports=%d\n",
             i, num_devices, name, device->device_attr.phys_port_cnt);
        int device_used = 0;
        for (int p = 1; p <= device->device_attr.phys_port_cnt; p++) {
            int allowed_device = 1;
            int replicate_count = 1;
            if (hca_list_count) {
                // filter out based on user hca list
                allowed_device = exclude_list;
                for (int j = 0; j < hca_list_count; j++) {
                    if (!strcmp(hca_list[j].name, name)) {
                        if (hca_list[j].port == -1 || hca_list[j].port == p) {
                            hca_list[j].found = 1;
                            allowed_device = !exclude_list;
                        }
                    }
                }
            } else if (pe_hca_map_count) {
                // filter devices based on user hca-pe mapping
                allowed_device = 0;
                for (int j = 0; j < pe_hca_map_count; j++) {
                    if (!strcmp(pe_hca_mapping[j].name, name)) {
                        if (pe_hca_mapping[j].port == -1 || pe_hca_mapping[j].port == p) {
                            allowed_device = 1;
                            pe_hca_mapping[j].found = 1;
                            replicate_count = pe_hca_mapping[j].count;
                        }
                    }
                }
            }

            if (!allowed_device) {
                continue;
            } else {
                status = ftable.query_port(device->context, p, &device->port_attr[p - 1]);
                NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                      "ibv_port_query failed \n");

                // IBGDA supports IB and RoCE.
                if ((device->port_attr[p - 1].state != IBV_PORT_ACTIVE) ||
                    ((device->port_attr[p - 1].link_layer != IBV_LINK_LAYER_INFINIBAND) &&
                     (device->port_attr[p - 1].link_layer != IBV_LINK_LAYER_ETHERNET))) {
                    if (user_selection) {
                        NVSHMEMI_WARN_PRINT(
                            "found inactive port or port with non IB/RoCE link layer protocol, "
                            "skipping...\n");
                    }
                    continue;
                }

                ib_get_gid_index(&ftable, device->context, p, device->port_attr[p - 1].gid_tbl_len,
                                 &device->gid_info[p - 1].local_gid_index, ibgda_state->log_level,
                                 options);
                status =
                    ftable.query_gid(device->context, p, device->gid_info[p - 1].local_gid_index,
                                     &device->gid_info[p - 1].local_gid);
                NVSHMEMI_NULL_ERROR_JMP(dev_list, status, NVSHMEMX_ERROR_INTERNAL, out,
                                        "query_gid failed \n");

                if (!device->pd) {
                    device->pd = ftable.alloc_pd(device->context);
                    NVSHMEMI_NULL_ERROR_JMP(device->pd, status, NVSHMEMX_ERROR_INTERNAL, out,
                                            "ibv_alloc_pd failed \n");
                }

                for (int k = 0; k < replicate_count; k++) {
                    ibgda_state->dev_ids[offset] = i;
                    ibgda_state->port_ids[offset] = p;
                    offset++;
                }

                device_used = 1;
            }
        }

        if (!device_used) {
            status = ftable.close_device(device->context);
            if (device->pd) {
                status = ftable.dealloc_pd(device->pd);
            }

            device->context = NULL;
            device->pd = NULL;
            NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                  "ibv_close_device or ibv_dealloc_pd failed \n");
            continue;
        }
#if 0
        /* Report whether we need to do atomic endianness conversions on 8 byte operands. */
        status = nvshmemt_ib_common_query_endianness_conversion_size(&atomic_host_endian_size,
                                                                     device->context);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "nvshmemt_ib_common_query_endianness_conversion_size failed.\n");
#endif
    }
    INFO(ibgda_state->log_level, "End - Enumerating IB devices in the system\n");

    ibgda_state->n_dev_ids = offset;
    INFO(ibgda_state->log_level,
         "Begin - Ordered list of devices for assignment (after processing user provdied env vars "
         "(if any))  - \n");
    for (int i = 0; i < ibgda_state->n_dev_ids; i++) {
        INFO(ibgda_state->log_level,
             "Ordered list of devices for assignment - idx=%d (of %d), device id=%d, port_num=%d\n",
             i, ibgda_state->n_dev_ids, ibgda_state->dev_ids[i], ibgda_state->port_ids[i]);

        device = (struct ibgda_device *)ibgda_state->devices + ibgda_state->dev_ids[i];
        nic_buf_on_gpumem &= device->support_nic_buf_on_gpumem;
        nic_buf_on_hostmem &= device->support_nic_buf_on_hostmem;
        if (device->nic_handler == IBGDA_NIC_HANDLER_CPU) nic_handler = IBGDA_NIC_HANDLER_CPU;
    }
    INFO(ibgda_state->log_level,
         "End - Ordered list of devices for assignment (after processing user provdied env vars "
         "(if any))\n");

    if (!ibgda_state->n_dev_ids) {
        INFO(
            ibgda_state->log_level,
            "no active IB device that supports GPU-initiated communication is found, exiting...\n");
        status = NVSHMEMX_ERROR_INTERNAL;
        goto out;
    }

    transport->n_devices = ibgda_state->n_dev_ids;
    transport->device_pci_paths = (char **)calloc(transport->n_devices, sizeof(char *));
    NVSHMEMI_NULL_ERROR_JMP(transport->device_pci_paths, status, NVSHMEMX_ERROR_INTERNAL, out,
                            "Unable to allocate paths for IB transport.\n");
    for (int i = 0; i < transport->n_devices; i++) {
        status = get_pci_path(i, &transport->device_pci_paths[i], transport);
        NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                              "Failed to get paths for PCI devices.\n");
    }

    assert(nic_buf_on_gpumem || nic_buf_on_hostmem);
    if (nic_buf_on_gpumem) {
        ibgda_nic_buf_location = IBGDA_MEM_TYPE_GPU;
        INFO(ibgda_state->log_level, "NIC buffer will be on GPU memory.\n");
    } else {
        ibgda_nic_buf_location = IBGDA_MEM_TYPE_HOST;
        INFO(ibgda_state->log_level, "NIC buffer will be on host memory.\n");
    }

    assert(nic_handler == IBGDA_NIC_HANDLER_GPU || nic_handler == IBGDA_NIC_HANDLER_CPU);
    if (nic_handler == IBGDA_NIC_HANDLER_CPU) {
        assert(use_gdrcopy);
        INFO(ibgda_state->log_level, "NIC handler will be CPU.\n");
    } else {
        INFO(ibgda_state->log_level, "NIC handler will be GPU.\n");
    }
    ibgda_nic_handler = nic_handler;

    // print devices that were not found
    if (hca_list_count) {
        for (int j = 0; j < hca_list_count; j++) {
            if (hca_list[j].found != 1) {
                NVSHMEMI_WARN_PRINT(
                    "cound not find user specified HCA name: %s port: %d, skipping\n",
                    hca_list[j].name, hca_list[j].port);
            }
        }
    } else if (pe_hca_map_count) {
        // filter devices based on user hca-pe mapping
        for (int j = 0; j < pe_hca_map_count; j++) {
            if (pe_hca_mapping[j].found != 1) {
                NVSHMEMI_WARN_PRINT(
                    "cound not find user specified HCA name: %s port: %d, skipping\n",
                    pe_hca_mapping[j].name, pe_hca_mapping[j].port);
            }
        }
    }

    status = cudaDeviceGetStreamPriorityRange(&lowest_stream_priority, &highest_stream_priority);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "stream priority query failed. \n");
    status = cudaStreamCreateWithPriority(&ibgda_state->my_stream, cudaStreamNonBlocking,
                                          highest_stream_priority);
    NVSHMEMI_NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "internal stream creation failed. \n");

    transport->host_ops.can_reach_peer = nvshmemt_ibgda_can_reach_peer;
    transport->host_ops.connect_endpoints = nvshmemt_ibgda_connect_endpoints;
    transport->host_ops.get_mem_handle = nvshmemt_ibgda_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_ibgda_release_mem_handle;
    transport->host_ops.show_info = nvshmemt_ibgda_show_info;
    transport->host_ops.progress =
        ((nic_handler == IBGDA_NIC_HANDLER_GPU) ? NULL : nvshmemt_ibgda_progress);
    transport->host_ops.finalize = nvshmemt_ibgda_finalize;
    transport->host_ops.rma = NULL;
    transport->host_ops.amo = NULL;
    transport->host_ops.fence = NULL;
    transport->host_ops.quiet = NULL;
    transport->host_ops.enforce_cst = NULL;
    transport->host_ops.add_device_remote_mem_handles =
        nvshmemt_ibgda_add_device_remote_mem_handles;

    transport->attr = NVSHMEM_TRANSPORT_ATTR_CONNECTED;
    transport->is_successfully_initialized = true;
    transport->max_op_len = 1ULL << 30;
    transport->atomic_host_endian_min_size = atomic_host_endian_size;
    transport->no_proxy = (nic_handler == IBGDA_NIC_HANDLER_GPU);
    transport->type = NVSHMEM_TRANSPORT_LIB_CODE_IBGDA;
    transport->api_version = api_version < NVSHMEM_TRANSPORT_INTERFACE_VERSION
                                 ? api_version
                                 : NVSHMEM_TRANSPORT_INTERFACE_VERSION;

    *t = transport;

out:
    if (status) {
        if (options) {
            free(options);
        }
        if (transport) {
            if (transport->device_pci_paths) {
                free(transport->device_pci_paths);
            }
            free(transport);
        }
    }
    // TODO: Implement cleanup
    return status;
}
