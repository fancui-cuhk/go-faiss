package faiss

/*
#include <stdlib.h>
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
