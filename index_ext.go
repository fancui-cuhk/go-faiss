package faiss

/*
#include <stdlib.h>
#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/IndexIVF_c.h>
#include <faiss/c_api/IndexIVFFlat_c.h>
#include <faiss/c_api/MetaIndexes_c.h>
#include <faiss/c_api/impl/AuxIndexStructures_c.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// DefaultSeekGapBytes merges invlist payload holes cheaper than a typical HDD seek.
const DefaultSeekGapBytes uint64 = 2 << 20

// InvertedListsIOStats is byte accounting from a skippable probe_clusters load.
type InvertedListsIOStats struct {
	TableBytes   uint64
	PayloadBytes uint64
	SkipBytes    uint64
	ReadOps      uint64
	MergedRanges uint64
	IoMs         float64
	ComputeMs    float64
}

// Bytes is the number of bytes actually transferred (table + payload + merged holes).
func (s InvertedListsIOStats) Bytes() uint64 {
	return s.TableBytes + s.PayloadBytes + s.SkipBytes
}

func ProbeClustersWithIO(
	idx Index,
	x []float32,
	k int64,
	nclusters int64,
	clusterIDs []int64,
	fileIDs []int64,
	centroidDis []float32,
	invlistBasePath string,
	seekGapBytes uint64,
	stats *InvertedListsIOStats,
) (distances []float32, labels []int64, err error) {
	n := len(x) / idx.D()
	distances = make([]float32, int64(n)*k)
	labels = make([]int64, int64(n)*k)

	var cInvPath *C.char
	if invlistBasePath != "" {
		cInvPath = C.CString(invlistBasePath)
		defer C.free(unsafe.Pointer(cInvPath))
	}

	var cStats *C.FaissInvertedListsIOStats
	var st C.FaissInvertedListsIOStats
	if stats != nil {
		cStats = &st
	}

	if c := C.faiss_probe_clusters(
		idx.cPtr(),
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		C.size_t(nclusters),
		(*C.idx_t)(&clusterIDs[0]),
		(*C.idx_t)(&fileIDs[0]),
		(*C.float)(&centroidDis[0]),
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
		cInvPath,
		C.size_t(seekGapBytes),
		cStats,
	); c != 0 {
		return distances, labels, getLastError()
	}
	if stats != nil {
		stats.TableBytes = uint64(st.table_bytes)
		stats.PayloadBytes = uint64(st.payload_bytes)
		stats.SkipBytes = uint64(st.skip_bytes)
		stats.ReadOps = uint64(st.read_ops)
		stats.MergedRanges = uint64(st.merged_ranges)
		// io_ms / compute_ms live in source Faiss headers; the installed
		// libfaiss C struct does not have them yet, so leave these at 0.
	}
	return distances, labels, nil
}

// SetIsTrained marks an index trained without calling Train. Needed when
// filling IVF lists with add_core instead of Add.
func SetIsTrained(idx Index, trained bool) error {
	v := C.int(0)
	if trained {
		v = 1
	}
	if c := C.faiss_Index_set_is_trained(idx.cPtr(), v); c != 0 {
		return getLastError()
	}
	return nil
}

// NewIndexIVFFlat builds an IVF-Flat index around a quantizer. The returned
// index owns the quantizer; do not Delete quantizer separately.
func NewIndexIVFFlat(quantizer Index, d, nlist int) (*IndexImpl, error) {
	var p *C.FaissIndexIVFFlat
	if c := C.faiss_IndexIVFFlat_new_with(&p, quantizer.cPtr(), C.size_t(d), C.size_t(nlist)); c != 0 {
		return nil, getLastError()
	}
	C.faiss_IndexIVFFlat_set_own_fields(p, 1)
	return &IndexImpl{&faissIndex{(*C.FaissIndex)(unsafe.Pointer(p))}}, nil
}

// IVFAddCore appends vectors to given inverted lists (precomputed list ids).
// The same vector id may be added to more than one list.
func IVFAddCore(idx Index, x []float32, ids, listNos []int64) error {
	if len(ids) != len(listNos) {
		return fmt.Errorf("IVFAddCore: ids (%d) and list ids (%d) differ", len(ids), len(listNos))
	}
	d := idx.D()
	if d <= 0 {
		return fmt.Errorf("IVFAddCore: invalid dimension")
	}
	n := len(ids)
	if n == 0 {
		return nil
	}
	if len(x) != n*d {
		return fmt.Errorf("IVFAddCore: vectors length %d want %d", len(x), n*d)
	}
	ivf := C.faiss_IndexIVFFlat_cast(idx.cPtr())
	if ivf == nil {
		return fmt.Errorf("IVFAddCore: index is not IndexIVFFlat")
	}
	if c := C.faiss_IndexIVFFlat_add_core(
		ivf,
		C.idx_t(n),
		(*C.float)(&x[0]),
		(*C.idx_t)(&ids[0]),
		(*C.int64_t)(&listNos[0]),
	); c != 0 {
		return getLastError()
	}
	return nil
}

// NewIndexIDMap2 wraps a sub-index so Search returns the provided IDs.
// The returned index owns sub; do not Delete sub separately.
func NewIndexIDMap2(sub Index) (*IndexImpl, error) {
	var wrapped *C.FaissIndexIDMap2
	if c := C.faiss_IndexIDMap2_new(&wrapped, sub.cPtr()); c != 0 {
		return nil, getLastError()
	}
	C.faiss_IndexIDMap2_set_own_fields(wrapped, 1)
	return &IndexImpl{&faissIndex{(*C.FaissIndex)(unsafe.Pointer(wrapped))}}, nil
}

// SearchWithSelector runs Search, optionally restricted to the given IDs.
// An empty filter searches every stored vector. k<=0 searches ntotal.
func SearchWithSelector(idx Index, x []float32, k int64, filter []int64) (distances []float32, labels []int64, err error) {
	if k <= 0 {
		k = idx.Ntotal()
		if k <= 0 {
			k = 1
		}
	}
	n := len(x) / idx.D()
	distances = make([]float32, int64(n)*k)
	labels = make([]int64, int64(n)*k)
	if len(filter) == 0 {
		if c := C.faiss_Index_search(
			idx.cPtr(),
			C.idx_t(n),
			(*C.float)(&x[0]),
			C.idx_t(k),
			(*C.float)(&distances[0]),
			(*C.idx_t)(&labels[0]),
		); c != 0 {
			return nil, nil, getLastError()
		}
		return distances, labels, nil
	}
	sel, err := NewIDSelectorBatch(filter)
	if err != nil {
		return nil, nil, err
	}
	defer sel.Delete()
	var params *C.FaissSearchParameters
	if c := C.faiss_SearchParameters_new(&params, sel.sel); c != 0 {
		return nil, nil, getLastError()
	}
	defer C.faiss_SearchParameters_free(params)
	if c := C.faiss_Index_search_with_params(
		idx.cPtr(),
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		params,
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		return nil, nil, getLastError()
	}
	return distances, labels, nil
}

// InitRAMInvlists replaces inverted lists with empty ArrayInvertedLists and
// sets ntotal=0. The quantizer, list_to_file mapping, and fname are kept.
func InitRAMInvlists(idx Index) error {
	if idx == nil {
		return fmt.Errorf("InitRAMInvlists: nil index")
	}
	if c := C.faiss_ivf_init_ram_invlists(idx.cPtr()); c != 0 {
		return getLastError()
	}
	return nil
}

// IVFNlist is IndexIVF.nlist.
func IVFNlist(idx Index) (int, error) {
	if idx == nil {
		return 0, fmt.Errorf("IVFNlist: nil index")
	}
	ivf := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivf == nil {
		return 0, fmt.Errorf("IVFNlist: not IVF")
	}
	return int(C.faiss_IndexIVF_nlist(ivf)), nil
}

// IVFListSize is the number of vectors currently stored in one RAM inverted list.
func IVFListSize(idx Index, listNo int64) (int64, error) {
	if idx == nil {
		return 0, fmt.Errorf("IVFListSize: nil index")
	}
	ivf := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivf == nil {
		return 0, fmt.Errorf("IVFListSize: index is not IVF")
	}
	return int64(C.faiss_IndexIVF_get_list_size(ivf, C.size_t(listNo))), nil
}

// SearchPreassigned runs Search over already-assigned IVF lists in RAM.
// Unlike ProbeClusters, this does not reload or replace inverted lists.
func SearchPreassigned(idx Index, x []float32, k int64, clusterIDs []int64, centroidDis []float32) (distances []float32, labels []int64, err error) {
	if idx == nil {
		return nil, nil, fmt.Errorf("SearchPreassigned: nil index")
	}
	if k <= 0 {
		return nil, nil, fmt.Errorf("SearchPreassigned: k must be positive")
	}
	if len(clusterIDs) != len(centroidDis) {
		return nil, nil, fmt.Errorf("SearchPreassigned: cluster_ids (%d) and centroid_distances (%d) differ", len(clusterIDs), len(centroidDis))
	}
	ivf := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivf == nil {
		return nil, nil, fmt.Errorf("SearchPreassigned: index is not IVF")
	}
	n := len(x) / idx.D()
	distances = make([]float32, int64(n)*k)
	labels = make([]int64, int64(n)*k)
	if n == 0 || len(clusterIDs) == 0 {
		return distances, labels, nil
	}
	if c := C.faiss_IndexIVF_search_preassigned(
		ivf,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		(*C.idx_t)(&clusterIDs[0]),
		(*C.float)(&centroidDis[0]),
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
		0,
	); c != 0 {
		return distances, labels, getLastError()
	}
	return distances, labels, nil
}

// AbsorbInvlists merges selected lists from `{base}_invlists_{fid}` into the
// resident ArrayInvertedLists. Lists that already have entries are skipped.
func AbsorbInvlists(idx Index, listIDs, fileIDs []int64, invlistBasePath string) error {
	if idx == nil {
		return fmt.Errorf("AbsorbInvlists: nil index")
	}
	if len(listIDs) != len(fileIDs) {
		return fmt.Errorf("AbsorbInvlists: list_ids (%d) and file_ids (%d) differ", len(listIDs), len(fileIDs))
	}
	var cPath *C.char
	if invlistBasePath != "" {
		cPath = C.CString(invlistBasePath)
		defer C.free(unsafe.Pointer(cPath))
	}
	var cLists, cFiles *C.idx_t
	n := len(listIDs)
	if n > 0 {
		cLists = (*C.idx_t)(unsafe.Pointer(&listIDs[0]))
		cFiles = (*C.idx_t)(unsafe.Pointer(&fileIDs[0]))
	}
	if c := C.faiss_ivf_absorb_invlists_from_files(
		idx.cPtr(),
		cLists,
		C.size_t(n),
		cFiles,
		cPath,
	); c != 0 {
		return getLastError()
	}
	return nil
}
