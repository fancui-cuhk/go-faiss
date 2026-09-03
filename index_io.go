package faiss

/*
#include <stdlib.h>
#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/index_io_c.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// IO flags
const (
	IOFlagMmap     = C.FAISS_IO_FLAG_MMAP
	IOFlagReadOnly = C.FAISS_IO_FLAG_READ_ONLY
	// IOFlagOndiskSameDir resolves the OnDiskInvertedLists data-file reference
	// relative to the directory of the index file being read. Required for
	// indexes built by MergeIVFOnDisk once they are copied elsewhere.
	IOFlagOndiskSameDir = C.FAISS_IO_FLAG_ONDISK_SAME_DIR
)

// MergeIVFOnDisk merges per-slice IVF block index files into one IVF index
// whose inverted lists are stored in ivfdataPath (OnDiskInvertedLists). The
// trainedPath file must be an empty, trained IVF index providing the
// quantizer; blocks must share that quantizer and hold globally assigned ids.
// Reading the resulting outPath requires IOFlagOndiskSameDir when the file was
// moved from where it was written.
func MergeIVFOnDisk(trainedPath string, blockPaths []string, ivfdataPath, outPath string) error {
	if len(blockPaths) == 0 {
		return fmt.Errorf("MergeIVFOnDisk: no block indexes")
	}
	cTrained := C.CString(trainedPath)
	defer C.free(unsafe.Pointer(cTrained))
	cIvfdata := C.CString(ivfdataPath)
	defer C.free(unsafe.Pointer(cIvfdata))
	cOut := C.CString(outPath)
	defer C.free(unsafe.Pointer(cOut))

	cBlocks := make([]*C.char, len(blockPaths))
	for i, p := range blockPaths {
		cBlocks[i] = C.CString(p)
		defer C.free(unsafe.Pointer(cBlocks[i]))
	}
	if c := C.faiss_merge_ivf_ondisk(
		cTrained,
		(**C.char)(unsafe.Pointer(&cBlocks[0])),
		C.size_t(len(cBlocks)),
		cIvfdata,
		cOut); c != 0 {
		return getLastError()
	}
	return nil
}

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

// ReadIndexDist reads a distributed IVF header.
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

// IVFCopyList copies one inverted list's ids and raw codes (IVFFlat codes are
// float32 vectors). Works for OnDiskInvertedLists.
func IVFCopyList(idx Index, listNo int) (ids []int64, codes []byte, codeSize int, err error) {
	var ls, cs C.size_t
	if c := C.faiss_ivf_copy_list(idx.cPtr(), C.size_t(listNo), nil, nil, &ls, &cs); c != 0 {
		return nil, nil, 0, getLastError()
	}
	codeSize = int(cs)
	if ls == 0 {
		return nil, nil, codeSize, nil
	}
	ids = make([]int64, ls)
	codes = make([]byte, int(ls)*codeSize)
	if c := C.faiss_ivf_copy_list(
		idx.cPtr(),
		C.size_t(listNo),
		(*C.idx_t)(unsafe.Pointer(&ids[0])),
		(*C.uint8_t)(unsafe.Pointer(&codes[0])),
		&ls,
		&cs,
	); c != 0 {
		return nil, nil, 0, getLastError()
	}
	return ids, codes, int(cs), nil
}

// HNSWLevel0Neighbors returns the level-0 neighbor vertex ids of i.
func HNSWLevel0Neighbors(idx Index, i int64) ([]int64, error) {
	buf := make([]int64, 64)
	n, err := HNSWFillLevel0Neighbors(idx, i, buf)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, nil
	}
	return append([]int64(nil), buf[:n]...), nil
}

// HNSWFillLevel0Neighbors copies level-0 neighbor ids of i into buf and
// returns how many were written. buf must be large enough (Faiss L0 is 2M).
func HNSWFillLevel0Neighbors(idx Index, i int64, buf []int64) (int, error) {
	if len(buf) == 0 {
		return 0, fmt.Errorf("HNSWFillLevel0Neighbors: empty buffer")
	}
	var n C.size_t
	if c := C.faiss_hnsw_copy_level0_neighbors(
		idx.cPtr(),
		C.idx_t(i),
		(*C.idx_t)(unsafe.Pointer(&buf[0])),
		C.size_t(len(buf)),
		&n,
	); c != 0 {
		return 0, getLastError()
	}
	return int(n), nil
}

// Reconstruct returns the (possibly approximate) vector stored at key.
func Reconstruct(idx Index, key int64) ([]float32, error) {
	d := idx.D()
	out := make([]float32, d)
	if c := C.faiss_Index_reconstruct(idx.cPtr(), C.idx_t(key), (*C.float)(&out[0])); c != 0 {
		return nil, getLastError()
	}
	return out, nil
}
