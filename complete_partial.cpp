#include "complete_partial.h"

#include <faiss/IndexIVF.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <faiss/impl/index_read_utils.h>
#include <faiss/invlists/InvertedLists.h>

#include <faiss/impl/maybe_owned_vector.h>

#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <vector>

namespace {

thread_local std::string g_err;

struct MmapBlock final : faiss::MaybeOwnedVectorOwner {
	void* p = nullptr;
	size_t n = 0;
	MmapBlock(void* ptr, size_t nbytes) : p(ptr), n(nbytes) {}
	~MmapBlock() override {
		if (p != nullptr && p != MAP_FAILED && n > 0) {
			munmap(p, n);
			p = nullptr;
		}
	}
};

void refresh_ntotal(faiss::IndexIVF* ivf, faiss::ArrayInvertedLists* ils) {
	size_t tot = 0;
	for (size_t i = 0; i < ils->nlist; i++) {
		tot += ils->list_size(i);
	}
	ivf->ntotal = static_cast<faiss::idx_t>(tot);
}

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

extern "C" int gofaiss_read_index_bytes(const void* data, size_t n, int io_flags, void** p_out) {
	g_err.clear();
	if (data == nullptr || p_out == nullptr || n == 0) {
		g_err = "read_index_bytes: nil argument";
		return -1;
	}
	try {
		faiss::VectorIOReader reader;
		reader.name = "odirect";
		const auto* bytes = static_cast<const uint8_t*>(data);
		reader.data.assign(bytes, bytes + n);
		faiss::Index* idx = faiss::read_index(&reader, io_flags);
		*p_out = idx;
		return 0;
	} catch (const std::exception& e) {
		g_err = e.what();
		return -1;
	}
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

extern "C" int gofaiss_adopt_invlist_block(
		void* index,
		void* block,
		size_t block_bytes,
		const GofaissInvlistSpan* spans,
		size_t n_spans) {
	g_err.clear();
	if (index == nullptr || block == nullptr || (n_spans > 0 && spans == nullptr)) {
		g_err = "adopt_invlist_block: nil argument";
		return -1;
	}
	bool taken = false;
	std::shared_ptr<MmapBlock> owner;
	try {
		auto* ivf = dynamic_cast<faiss::IndexIVF*>(reinterpret_cast<faiss::Index*>(index));
		if (ivf == nullptr) {
			g_err = "adopt_invlist_block: not IVF";
			return -1;
		}
		auto* ils = dynamic_cast<faiss::ArrayInvertedLists*>(ivf->invlists);
		if (ils == nullptr) {
			faiss::init_ram_invlists(ivf);
			ils = dynamic_cast<faiss::ArrayInvertedLists*>(ivf->invlists);
		}
		if (ils == nullptr) {
			g_err = "adopt_invlist_block: no ArrayInvertedLists";
			return -1;
		}
		if (ils->code_size == faiss::InvertedLists::INVALID_CODE_SIZE) {
			ils->code_size = ivf->code_size;
		}
		if (ils->code_size != ivf->code_size) {
			g_err = "adopt_invlist_block: code_size mismatch";
			return -1;
		}
		owner = std::make_shared<MmapBlock>(block, block_bytes);
		taken = true;
		if (n_spans == 0) {
			owner.reset();
			refresh_ntotal(ivf, ils);
			return 0;
		}
		auto* base = static_cast<uint8_t*>(block);
		for (size_t i = 0; i < n_spans; i++) {
			const GofaissInvlistSpan& sp = spans[i];
			if (sp.list_id < 0 || static_cast<size_t>(sp.list_id) >= ils->nlist) {
				g_err = "adopt_invlist_block: list id out of range";
				owner.reset();
				return -2;
			}
			if (sp.nvec == 0 || ils->list_size(static_cast<size_t>(sp.list_id)) > 0) {
				continue;
			}
			size_t code_bytes = sp.nvec * ils->code_size;
			size_t id_bytes = sp.nvec * sizeof(faiss::idx_t);
			if (sp.code_off > block_bytes || code_bytes > block_bytes - sp.code_off ||
					sp.id_off > block_bytes || id_bytes > block_bytes - sp.id_off ||
					(sp.id_off % sizeof(faiss::idx_t)) != 0) {
				g_err = "adopt_invlist_block: span out of block";
				owner.reset();
				return -2;
			}
		}
		size_t adopted = 0;
		for (size_t i = 0; i < n_spans; i++) {
			const GofaissInvlistSpan& sp = spans[i];
			size_t list_no = static_cast<size_t>(sp.list_id);
			if (sp.nvec == 0 || ils->list_size(list_no) > 0) {
				continue;
			}
			size_t code_bytes = sp.nvec * ils->code_size;
			ils->codes[list_no] = faiss::MaybeOwnedVector<uint8_t>::create_view(
					base + sp.code_off, code_bytes, owner);
			ils->ids[list_no] = faiss::MaybeOwnedVector<faiss::idx_t>::create_view(
					base + sp.id_off, sp.nvec, owner);
			adopted++;
		}
		if (adopted == 0) {
			owner.reset();
		}
		refresh_ntotal(ivf, ils);
		return 0;
	} catch (const std::exception& e) {
		g_err = e.what();
		if (!taken) {
			return -1;
		}
		return -2;
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
