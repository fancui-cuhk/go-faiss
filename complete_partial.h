#ifndef GOFAISS_COMPLETE_PARTIAL_H
#define GOFAISS_COMPLETE_PARTIAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* gofaiss_complete_last_error(void);

/* Deserialize an index from a memory buffer (used for O_DIRECT page loads). */
int gofaiss_read_index_bytes(const void* data, size_t n, int io_flags, void** p_out);

/* IVF header + empty ArrayInvertedLists. Does not heap-load all invlists. */
int gofaiss_read_index_header_ram(const char* fname, void** p_out);

/* Copy selected lists from `{complete}` / `{complete}.ivfdata` into dest RAM.
 * n_lists==0 copies every still-empty list. Already-loaded lists are skipped.
 * Return 2 when a full copy (n_lists==0) stops early because a query asked it
 * to yield. Lists copied before the yield stay loaded.
 */
int gofaiss_absorb_lists_from_complete(
		void* dest,
		const int64_t* list_ids,
		size_t n_lists,
		const char* complete_path);

/* Nonzero: full-index copies should stop so a query absorb can use the disk. */
void gofaiss_set_absorb_yield(int yield_to_query);

/* Codes+ids bytes in RAM inverted lists. Uses invlists->nlist, not ivf->nlist. */
int64_t gofaiss_ivf_resident_list_bytes(void* index);

/* One list inside a caller-mmap'd block. Offsets are bytes from block.
 * id_off must be 8-byte aligned.
 * Return 0: block is owned by the index (or already munmap'd if nothing was
 * adopted). Return -1: caller still owns the block. Return -2: this function
 * owns or already munmap'd the block; the caller must not munmap.
 */
typedef struct GofaissInvlistSpan {
	int64_t list_id;
	size_t nvec;
	size_t code_off;
	size_t id_off;
} GofaissInvlistSpan;

int gofaiss_adopt_invlist_block(
		void* index,
		void* block,
		size_t block_bytes,
		const GofaissInvlistSpan* spans,
		size_t n_spans);

#ifdef __cplusplus
}
#endif

#endif
