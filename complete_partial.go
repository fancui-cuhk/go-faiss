package faiss

/*
#cgo CXXFLAGS: -std=c++17
#include <stdlib.h>
#include <faiss/c_api/Index_c.h>
#include "complete_partial.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func completeLastError() error {
	return fmt.Errorf("%s", C.GoString(C.gofaiss_complete_last_error()))
}

// ReadIndexHeaderRam loads an IVF quantizer/header and empty ArrayInvertedLists.
// It does not heap-copy every inverted list from `{complete}` / `{complete}.ivfdata`.
func ReadIndexHeaderRam(filename string) (*IndexImpl, error) {
	cfname := C.CString(filename)
	defer C.free(unsafe.Pointer(cfname))
	var raw unsafe.Pointer
	if C.gofaiss_read_index_header_ram(cfname, &raw) != 0 {
		return nil, completeLastError()
	}
	idx := faissIndex{idx: (*C.FaissIndex)(raw)}
	return &IndexImpl{&idx}, nil
}

// AbsorbListsFromComplete copies selected inverted lists from a complete IVF
// file into dest (header-only RAM index). Empty listIDs copies every still-empty
// list. Lists that already have entries are skipped.
func AbsorbListsFromComplete(dest Index, listIDs []int64, completePath string) error {
	if dest == nil {
		return fmt.Errorf("AbsorbListsFromComplete: nil dest")
	}
	cpath := C.CString(completePath)
	defer C.free(unsafe.Pointer(cpath))
	var cLists *C.int64_t
	n := len(listIDs)
	if n > 0 {
		cLists = (*C.int64_t)(unsafe.Pointer(&listIDs[0]))
	}
	if C.gofaiss_absorb_lists_from_complete(
		unsafe.Pointer(dest.cPtr()),
		cLists,
		C.size_t(n),
		cpath,
	) != 0 {
		return completeLastError()
	}
	return nil
}

// IVFResidentListBytes is codes+ids currently in RAM inverted lists.
// Walks invlists->nlist in one C call so nlist cannot run past the array.
func IVFResidentListBytes(idx Index) (int64, error) {
	if idx == nil {
		return 0, fmt.Errorf("IVFResidentListBytes: nil index")
	}
	n := int64(C.gofaiss_ivf_resident_list_bytes(unsafe.Pointer(idx.cPtr())))
	if n < 0 {
		return 0, completeLastError()
	}
	return n, nil
}
