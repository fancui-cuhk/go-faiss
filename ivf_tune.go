package faiss

/*
#cgo CXXFLAGS: -std=c++17
#cgo LDFLAGS: -lgomp
#include "ivf_tune.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// IVFParallelLists makes one Search() split nprobe lists across OpenMP threads.
const IVFParallelLists = 1

// SetOMPThreads is how many cores one Faiss Search may use.
func SetOMPThreads(n int) {
	if n < 1 {
		n = runtime.NumCPU()
	}
	C.gofaiss_set_omp_threads(C.int(n))
}

// OMPMaxThreads is the OpenMP team size Faiss will use on the next Search.
func OMPMaxThreads() int {
	return int(C.gofaiss_omp_max_threads())
}

// TuneIVFSearch sets nprobe and OpenMP-over-lists on an IVF (or IDMap-wrapped IVF).
func TuneIVFSearch(idx Index, nprobe, parallelMode int) error {
	if idx == nil {
		return fmt.Errorf("nil index")
	}
	rc := C.gofaiss_set_ivf_search_tune(unsafe.Pointer(idx.cPtr()), C.int(nprobe), C.int(parallelMode))
	if rc != 0 {
		return fmt.Errorf("index is not IVF; cannot set nprobe/parallel_mode")
	}
	return nil
}
