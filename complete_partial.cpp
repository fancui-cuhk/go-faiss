#include "complete_partial.h"

#include <faiss/IndexIVF.h>
#include <faiss/index_io.h>
#include <faiss/impl/index_read_utils.h>
#include <faiss/invlists/InvertedLists.h>

#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

thread_local std::string g_err;

int complete_read_flags(const char* path) {
	int flags = faiss::IO_FLAG_ONDISK_SAME_DIR | faiss::IO_FLAG_READ_ONLY;
	std::string sidecar = std::string(path) + ".ivfdata";
	struct stat st;
	if (stat(sidecar.c_str(), &st) != 0) {
		// Single-file ArrayInvertedLists: mmap via IO_FLAG_MMAP, no heap copy.
		flags |= faiss::IO_FLAG_MMAP;
	}
	return flags;
}

} // namespace

extern "C" const char* gofaiss_complete_last_error(void) {
	return g_err.c_str();
}

extern "C" int gofaiss_read_index_header_ram(const char* fname, void** p_out) {
	g_err.clear();
	if (fname == nullptr || p_out == nullptr) {
		g_err = "read_index_header_ram: nil argument";
		return -1;
	}
	try {
		std::unique_ptr<faiss::Index> idx(
				faiss::read_index(fname, complete_read_flags(fname)));
		auto* ivf = dynamic_cast<faiss::IndexIVF*>(idx.get());
		if (ivf == nullptr) {
			g_err = "read_index_header_ram: not IVF";
			return -1;
		}
		faiss::init_ram_invlists(ivf);
		if (ivf->list_to_file.size() != ivf->nlist) {
			ivf->list_to_file.assign(ivf->nlist, 0);
		}
		*p_out = idx.release();
		return 0;
	} catch (const std::exception& e) {
		g_err = e.what();
		return -1;
	}
}

extern "C" int gofaiss_absorb_lists_from_complete(
		void* dest,
		const int64_t* list_ids,
		size_t n_lists,
		const char* complete_path) {
	g_err.clear();
	if (dest == nullptr || complete_path == nullptr) {
		g_err = "absorb_lists_from_complete: nil argument";
		return -1;
	}
	try {
		auto* ivf = dynamic_cast<faiss::IndexIVF*>(
				reinterpret_cast<faiss::Index*>(dest));
		if (ivf == nullptr) {
			g_err = "absorb_lists_from_complete: dest is not IVF";
			return -1;
		}
		auto* dst = dynamic_cast<faiss::ArrayInvertedLists*>(ivf->invlists);
		if (dst == nullptr) {
			faiss::init_ram_invlists(ivf);
			dst = dynamic_cast<faiss::ArrayInvertedLists*>(ivf->invlists);
		}
		if (dst == nullptr) {
			g_err = "absorb_lists_from_complete: no ArrayInvertedLists";
			return -1;
		}

		std::unique_ptr<faiss::Index> src_idx(
				faiss::read_index(complete_path, complete_read_flags(complete_path)));
		auto* src = dynamic_cast<faiss::IndexIVF*>(src_idx.get());
		if (src == nullptr || src->invlists == nullptr) {
			g_err = "absorb_lists_from_complete: source is not IVF";
			return -1;
		}
		faiss::InvertedLists* sil = src->invlists;

		std::vector<int64_t> all;
		if (n_lists == 0) {
			all.resize(ivf->nlist);
			for (size_t i = 0; i < ivf->nlist; i++) {
				all[i] = static_cast<int64_t>(i);
			}
			list_ids = all.data();
			n_lists = all.size();
		}
		if (n_lists > 0 && list_ids == nullptr) {
			g_err = "absorb_lists_from_complete: nil list_ids";
			return -1;
		}

		for (size_t i = 0; i < n_lists; i++) {
			auto lid = static_cast<size_t>(list_ids[i]);
			if (list_ids[i] < 0 || lid >= dst->nlist || lid >= sil->nlist) {
				continue;
			}
			if (dst->list_size(lid) > 0) {
				continue;
			}
			size_t ls = sil->list_size(lid);
			if (ls == 0) {
				continue;
			}
			const uint8_t* codes = sil->get_codes(lid);
			const faiss::idx_t* ids = sil->get_ids(lid);
			dst->add_entries(lid, ls, ids, codes);
			sil->release_codes(lid, codes);
			sil->release_ids(lid, ids);
		}
		size_t tot = 0;
		for (size_t i = 0; i < dst->nlist; i++) {
			tot += dst->list_size(i);
		}
		ivf->ntotal = static_cast<faiss::idx_t>(tot);
		return 0;
	} catch (const std::exception& e) {
		g_err = e.what();
		return -1;
	}
}

extern "C" int64_t gofaiss_ivf_resident_list_bytes(void* index) {
	g_err.clear();
	if (index == nullptr) {
		g_err = "ivf_resident_list_bytes: nil index";
		return -1;
	}
	try {
		auto* ivf = dynamic_cast<faiss::IndexIVF*>(
				reinterpret_cast<faiss::Index*>(index));
		if (ivf == nullptr || ivf->invlists == nullptr) {
			return 0;
		}
		faiss::InvertedLists* il = ivf->invlists;
		size_t n = il->nlist;
		if (n > ivf->nlist) {
			n = ivf->nlist;
		}
		const int64_t vec_bytes = static_cast<int64_t>(ivf->d) * 4 + 8;
		int64_t bytes = 0;
		for (size_t i = 0; i < n; i++) {
			bytes += static_cast<int64_t>(il->list_size(i)) * vec_bytes;
		}
		return bytes;
	} catch (const std::exception& e) {
		g_err = e.what();
		return -1;
	}
}
