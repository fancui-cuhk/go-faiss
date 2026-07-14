//go:build faiss_ext

package faiss

/*
#include <stdlib.h>
#include <faiss/c_api/index_io_c.h>
*/
import "C"
import (
	"unsafe"
)

func WriteIndexDistGrouped(idx Index, mainFilename string, groups [][]int) error {
	cfname := C.CString(mainFilename)
	defer C.free(unsafe.Pointer(cfname))

	nGroups := len(groups)
	groupSizes := make([]C.size_t, nGroups)
	flatIDs := make([]C.size_t, 0)
	for i, g := range groups {
		groupSizes[i] = C.size_t(len(g))
		for _, cid := range g {
			flatIDs = append(flatIDs, C.size_t(cid))
		}
	}
	var cFlat *C.size_t
	if len(flatIDs) > 0 {
		cFlat = &flatIDs[0]
	}
	var cSizes *C.size_t
	if nGroups > 0 {
		cSizes = &groupSizes[0]
	}
	if c := C.faiss_write_index_fname_dist_grouped(
		idx.cPtr(), cfname, C.size_t(nGroups), cSizes, cFlat,
	); c != 0 {
		return getLastError()
	}
	return nil
}

func IVFCentroids(idx Index) (centroids []float32, nlist, d int, err error) {
	var cCentroids *C.float
	var cNlist, cD C.size_t
	if c := C.faiss_get_ivf_centroids(idx.cPtr(), &cCentroids, &cNlist, &cD); c != 0 {
		return nil, 0, 0, getLastError()
	}
	nlist = int(cNlist)
	d = int(cD)
	centroids = make([]float32, nlist*d)
	if nlist*d > 0 {
		hdr := (*[1 << 28]C.float)(unsafe.Pointer(cCentroids))[: nlist*d : nlist*d]
		for i := range centroids {
			centroids[i] = float32(hdr[i])
		}
	}
	return centroids, nlist, d, nil
}

func IVFClusterSizes(idx Index) (sizes []int64, codeSize int, err error) {
	var cSizes *C.size_t
	var cNlist, cCodeSize C.size_t
	if c := C.faiss_get_ivf_cluster_sizes(idx.cPtr(), &cSizes, &cNlist, &cCodeSize); c != 0 {
		return nil, 0, getLastError()
	}
	n := int(cNlist)
	sizes = make([]int64, n)
	if n > 0 {
		src := unsafe.Slice(cSizes, n)
		for i := 0; i < n; i++ {
			sizes[i] = int64(src[i])
		}
	}
	return sizes, int(cCodeSize), nil
}
