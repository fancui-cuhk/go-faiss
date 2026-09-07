#ifndef GOFAISS_IVF_TUNE_H
#define GOFAISS_IVF_TUNE_H

#ifdef __cplusplus
extern "C" {
#endif

int gofaiss_set_ivf_search_tune(void* index, int nprobe, int parallel_mode);
void gofaiss_set_omp_threads(int n);
int gofaiss_omp_max_threads(void);

#ifdef __cplusplus
}
#endif

#endif
