package faiss

/*
#cgo CXXFLAGS: -std=c++17
#include <stdlib.h>
#include <faiss/c_api/Index_c.h>
#include "complete_partial.h"
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// ErrAbsorbYielded means a full-index copy stopped so a query absorb can run.
// Lists copied before the stop stay loaded; the caller should retry.
var ErrAbsorbYielded = errors.New("absorb yielded to query")

func completeLastError() error {
	return fmt.Errorf("%s", C.GoString(C.gofaiss_complete_last_error()))
}

// ReadIndexBytes deserializes an index from an in-memory Faiss file image.
func ReadIndexBytes(data []byte, ioflags int) (*IndexImpl, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("ReadIndexBytes: empty")
	}
	var raw unsafe.Pointer
	if C.gofaiss_read_index_bytes(unsafe.Pointer(&data[0]), C.size_t(len(data)), C.int(ioflags), &raw) != 0 {
		return nil, completeLastError()
	}
	idx := faissIndex{idx: (*C.FaissIndex)(raw)}
	return &IndexImpl{&idx}, nil
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

// SetAbsorbYield asks in-flight full-index copies to stop. Query absorbs
// (n_lists > 0) are not interrupted. Pass false when no query absorb is running.
func SetAbsorbYield(yieldToQuery bool) {
	v := C.int(0)
	if yieldToQuery {
		v = 1
	}
	C.gofaiss_set_absorb_yield(v)
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
	rc := C.gofaiss_absorb_lists_from_complete(
		unsafe.Pointer(dest.cPtr()),
		cLists,
		C.size_t(n),
		cpath,
	)
	if rc == 2 {
		return ErrAbsorbYielded
	}
	if rc != 0 {
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

// adoptInvlistBlock hands one mmap block to the IVF index.
// A nil error means C++ owns the block (or already unmapped it).
// On error, the block is unmapped here when C++ did not take it.
func adoptInvlistBlock(idx Index, block []byte, spans []C.GofaissInvlistSpan) error {
	if idx == nil {
		mapFree(block)
		return fmt.Errorf("adopt_invlist_block: nil index")
	}
	if len(block) == 0 {
		return fmt.Errorf("adopt_invlist_block: empty block")
	}
	var sp *C.GofaissInvlistSpan
	if len(spans) > 0 {
		sp = &spans[0]
	}
	rc := C.gofaiss_adopt_invlist_block(
		unsafe.Pointer(idx.cPtr()),
		unsafe.Pointer(&block[0]),
		C.size_t(len(block)),
		sp,
		C.size_t(len(spans)),
	)
	runtime.KeepAlive(block)
	runtime.KeepAlive(spans)
	if rc == 0 {
		return nil
	}
	if rc == -1 {
		mapFree(block)
	}
	return completeLastError()
}
