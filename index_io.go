package faiss

/*
#include <stdlib.h>
#include <faiss/c_api/index_io_c.h>
*/
import "C"
import (
	"unsafe"
)

// IO flags
const (
	IOFlagMmap     = C.FAISS_IO_FLAG_MMAP
	IOFlagReadOnly = C.FAISS_IO_FLAG_READ_ONLY
)

// WriteIndex writes an index to a file.
func WriteIndex(idx Index, filename string) error {
	cfname := C.CString(filename)
	defer C.free(unsafe.Pointer(cfname))
	if c := C.faiss_write_index_fname(idx.cPtr(), cfname); c != 0 {
		return getLastError()
	}
	return nil
}

// ReadIndex reads an index from a file.
func ReadIndex(filename string, ioflags int) (*IndexImpl, error) {
	cfname := C.CString(filename)
	defer C.free(unsafe.Pointer(cfname))
	var idx faissIndex
	if c := C.faiss_read_index_fname(cfname, C.int(ioflags), &idx.idx); c != 0 {
		return nil, getLastError()
	}
	return &IndexImpl{&idx}, nil
}

// [DIST] WriteIndexDist writes an index to files.
func WriteIndexDist(idx Index, main_filename string) error {
	cfname := C.CString(main_filename)
	defer C.free(unsafe.Pointer(cfname))
	if c := C.faiss_write_index_fname_dist(idx.cPtr(), cfname); c != 0 {
		return getLastError()
	}
	return nil
}

// [DIST] ReadIndexDist reads an index from file.
func ReadIndexDist(main_filename string, ioflags int) (*IndexImpl, error) {
	cfname := C.CString(main_filename)
	defer C.free(unsafe.Pointer(cfname))
	var idx faissIndex
	if c := C.faiss_read_index_fname_dist(cfname, C.int(ioflags), &idx.idx); c != 0 {
		return nil, getLastError()
	}
	return &IndexImpl{&idx}, nil
}

// [DIST] GetListToFileMapping retrieves the cluster-to-file mapping from IndexIVF.
// Returns a slice where index is cluster_id and value is file_id.
func GetListToFileMapping(idx Index) ([]int64, error) {
	var listToFile *C.size_t
	var nlist C.size_t
	
	if c := C.faiss_get_list_to_file_mapping(idx.cPtr(), &listToFile, &nlist); c != 0 {
		return nil, getLastError()
	}
	
	// Convert C array to Go slice
	goSlice := make([]int64, int(nlist))
	cSlice := unsafe.Slice(listToFile, int(nlist))
	for i := 0; i < int(nlist); i++ {
		goSlice[i] = int64(cSlice[i])
	}
	
	return goSlice, nil
}

// WriteIndexDistGrouped writes a distributed IVF index with explicit cluster file groups.
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

// IVFCentroids returns coarse quantizer centroids (nlist * d floats).
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
		src := unsafe.Slice(cCentroids, nlist*d)
		copy(centroids, src)
	}
	return centroids, nlist, d, nil
}

// IVFClusterSizes returns per-list vector counts and code size.
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
