package faiss

/*
#include <stdlib.h>
#include <faiss/c_api/index_io_c.h>
*/
import "C"
import "unsafe"

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
