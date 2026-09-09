#ifndef GOFAISS_COMPLETE_PARTIAL_H
#define GOFAISS_COMPLETE_PARTIAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* gofaiss_complete_last_error(void);

/* IVF header + empty ArrayInvertedLists. Does not heap-load all invlists. */
int gofaiss_read_index_header_ram(const char* fname, void** p_out);

/* Copy selected lists from `{complete}` / `{complete}.ivfdata` into dest RAM.
 * n_lists==0 copies every still-empty list. Already-loaded lists are skipped.
 */
int gofaiss_absorb_lists_from_complete(
		void* dest,
		const int64_t* list_ids,
		size_t n_lists,
		const char* complete_path);

/* Codes+ids bytes in RAM inverted lists. Uses invlists->nlist, not ivf->nlist. */
int64_t gofaiss_ivf_resident_list_bytes(void* index);

#ifdef __cplusplus
}
#endif

#endif
