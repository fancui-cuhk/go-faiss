#include "ivf_tune.h"

#include <faiss/IndexIVF.h>
#include <faiss/MetaIndexes.h>
#include <omp.h>

extern "C" int gofaiss_set_ivf_search_tune(void* index, int nprobe, int parallel_mode) {
	auto* idx = reinterpret_cast<faiss::Index*>(index);
	if (idx == nullptr) {
		return -1;
	}
	auto* ivf = dynamic_cast<faiss::IndexIVF*>(idx);
	if (ivf == nullptr) {
		if (auto* idmap = dynamic_cast<faiss::IndexIDMap*>(idx)) {
			ivf = dynamic_cast<faiss::IndexIVF*>(idmap->index);
		}
	}
	if (ivf == nullptr) {
		return -1;
	}
	if (nprobe > 0) {
		ivf->nprobe = static_cast<size_t>(nprobe);
	}
	ivf->parallel_mode = parallel_mode;
	return 0;
}

extern "C" void gofaiss_set_omp_threads(int n) {
	if (n < 1) {
		n = 1;
	}
	omp_set_dynamic(0);
	omp_set_nested(0);
	omp_set_num_threads(n);
}

extern "C" int gofaiss_omp_max_threads(void) {
	return omp_get_max_threads();
}
