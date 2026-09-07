//go:build faiss_ext

package faiss_test

import (
	"os"
	"path/filepath"
	"testing"

	faiss "github.com/fancui-cuhk/go-faiss"
)

func TestProbeClustersSkipsUnreadLists(t *testing.T) {
	dir := t.TempDir()
	prefix := filepath.Join(dir, "db")
	const d = 4
	quant, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatal(err)
	}
	cents := []float32{
		0, 0, 0, 0,
		10, 0, 0, 0,
		0, 10, 0, 0,
	}
	if err := quant.Add(cents); err != nil {
		t.Fatal(err)
	}
	idx, err := faiss.NewIndexIVFFlat(quant, d, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Delete()
	if err := faiss.SetIsTrained(idx, true); err != nil {
		t.Fatal(err)
	}

	if err := faiss.IVFAddCore(idx, []float32{0, 0, 0, 0}, []int64{7}, []int64{0}); err != nil {
		t.Fatal(err)
	}
	big := make([]float32, d*4000)
	ids := make([]int64, 4000)
	lists := make([]int64, 4000)
	for i := 0; i < 4000; i++ {
		big[i*d] = 10
		ids[i] = 1000 + int64(i)
		lists[i] = 1
	}
	if err := faiss.IVFAddCore(idx, big, ids, lists); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4000; i++ {
		big[i*d] = 0
		big[i*d+1] = 10
		ids[i] = 9000 + int64(i)
		lists[i] = 2
	}
	if err := faiss.IVFAddCore(idx, big, ids, lists); err != nil {
		t.Fatal(err)
	}

	if err := faiss.WriteIndexDistGrouped(idx, prefix, [][]int{{0, 1, 2}}); err != nil {
		t.Fatal(err)
	}
	fi, err := os.Stat(prefix + "_invlists_0")
	if err != nil {
		t.Fatal(err)
	}

	header, err := faiss.ReadIndexDist(prefix, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer header.Delete()

	var st faiss.InvertedListsIOStats
	dists, labels, err := faiss.ProbeClustersWithIO(
		header, []float32{0, 0, 0, 0}, 1, 1,
		[]int64{0}, []int64{0}, []float32{0},
		"", 0, &st,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(labels) == 0 || labels[0] != 7 || dists[0] > 1e-5 {
		t.Fatalf("want id 7 dist~0, got labels=%v dists=%v", labels, dists)
	}
	if st.Bytes() >= uint64(fi.Size())/2 {
		t.Fatalf("skip reader transferred %d of %d-byte shard", st.Bytes(), fi.Size())
	}
	if st.SkipBytes != 0 {
		t.Fatalf("gap=0 should not read holes, skip_bytes=%d", st.SkipBytes)
	}
	if st.MergedRanges != 1 {
		t.Fatalf("merged_ranges=%d want 1", st.MergedRanges)
	}
	if st.TableBytes == 0 {
		t.Fatal("first probe should read the offset table")
	}

	var stMerge faiss.InvertedListsIOStats
	if _, _, err := faiss.ProbeClustersWithIO(
		header, []float32{0, 0, 0, 0}, 2, 2,
		[]int64{0, 2}, []int64{0, 0}, []float32{0, 0},
		"", 1<<30, &stMerge,
	); err != nil {
		t.Fatal(err)
	}
	if stMerge.MergedRanges != 1 {
		t.Fatalf("huge gap should merge lists 0 and 2 across the unread hole, got %d", stMerge.MergedRanges)
	}
	if stMerge.SkipBytes == 0 {
		t.Fatal("merged hole should show up as skip_bytes")
	}
	if stMerge.TableBytes != 0 {
		t.Fatalf("second probe should keep the table in RAM, table_bytes=%d", stMerge.TableBytes)
	}
}

func writeThreeListIVF(t *testing.T, prefix string, groups [][]int) {
	t.Helper()
	const d = 4
	quant, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatal(err)
	}
	if err := quant.Add([]float32{
		0, 0, 0, 0,
		10, 0, 0, 0,
		0, 10, 0, 0,
	}); err != nil {
		t.Fatal(err)
	}
	idx, err := faiss.NewIndexIVFFlat(quant, d, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Delete()
	if err := faiss.SetIsTrained(idx, true); err != nil {
		t.Fatal(err)
	}
	if err := faiss.IVFAddCore(idx, []float32{0, 0, 0, 0}, []int64{7}, []int64{0}); err != nil {
		t.Fatal(err)
	}
	big := make([]float32, d*4000)
	ids := make([]int64, 4000)
	lists := make([]int64, 4000)
	for i := 0; i < 4000; i++ {
		big[i*d] = 10
		ids[i] = 1000 + int64(i)
		lists[i] = 1
	}
	if err := faiss.IVFAddCore(idx, big, ids, lists); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4000; i++ {
		big[i*d] = 0
		big[i*d+1] = 10
		ids[i] = 9000 + int64(i)
		lists[i] = 2
	}
	if err := faiss.IVFAddCore(idx, big, ids, lists); err != nil {
		t.Fatal(err)
	}
	if err := faiss.WriteIndexDistGrouped(idx, prefix, groups); err != nil {
		t.Fatal(err)
	}
}

func TestInvlistDirCache(t *testing.T) {
	q := []float32{0, 0, 0, 0}

	t.Run("sameQueryMatchesAndDropsTableIO", func(t *testing.T) {
		prefix := filepath.Join(t.TempDir(), "db")
		writeThreeListIVF(t, prefix, [][]int{{0, 1, 2}})
		fi, err := os.Stat(prefix + "_invlists_0")
		if err != nil {
			t.Fatal(err)
		}
		header, err := faiss.ReadIndexDist(prefix, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer header.Delete()

		var first, second faiss.InvertedListsIOStats
		d1, l1, err := faiss.ProbeClustersWithIO(header, q, 1, 1, []int64{0}, []int64{0}, []float32{0}, "", 0, &first)
		if err != nil {
			t.Fatal(err)
		}
		d2, l2, err := faiss.ProbeClustersWithIO(header, q, 1, 1, []int64{0}, []int64{0}, []float32{0}, "", 0, &second)
		if err != nil {
			t.Fatal(err)
		}
		if l1[0] != 7 || l2[0] != 7 || d1[0] > 1e-5 || d2[0] > 1e-5 {
			t.Fatalf("hits drifted: first=%v %v second=%v %v", l1, d1, l2, d2)
		}
		if first.TableBytes == 0 {
			t.Fatal("cold probe must read idsizes")
		}
		if second.TableBytes != 0 {
			t.Fatalf("warm probe still read the table: %d", second.TableBytes)
		}
		if second.PayloadBytes != first.PayloadBytes {
			t.Fatalf("payload bytes changed after cache: cold=%d warm=%d", first.PayloadBytes, second.PayloadBytes)
		}
		if second.ReadOps >= first.ReadOps {
			t.Fatalf("warm probe should do fewer reads: cold=%d warm=%d", first.ReadOps, second.ReadOps)
		}
		if first.Bytes() >= uint64(fi.Size())/2 || second.Bytes() >= uint64(fi.Size())/2 {
			t.Fatalf("cached skip leaked unread lists: file=%d cold=%d warm=%d", fi.Size(), first.Bytes(), second.Bytes())
		}
	})

	t.Run("secondFileHasItsOwnColdTable", func(t *testing.T) {
		prefix := filepath.Join(t.TempDir(), "db")
		writeThreeListIVF(t, prefix, [][]int{{0}, {1, 2}})
		header, err := faiss.ReadIndexDist(prefix, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer header.Delete()

		var a1, a2, b1 faiss.InvertedListsIOStats
		if _, _, err := faiss.ProbeClustersWithIO(header, q, 1, 1, []int64{0}, []int64{0}, []float32{0}, "", 0, &a1); err != nil {
			t.Fatal(err)
		}
		if _, _, err := faiss.ProbeClustersWithIO(header, q, 1, 1, []int64{0}, []int64{0}, []float32{0}, "", 0, &a2); err != nil {
			t.Fatal(err)
		}
		if a2.TableBytes != 0 {
			t.Fatalf("file 0 warm table_bytes=%d", a2.TableBytes)
		}
		if _, labels, err := faiss.ProbeClustersWithIO(header, []float32{10, 0, 0, 0}, 1, 1, []int64{1}, []int64{1}, []float32{0}, "", 0, &b1); err != nil {
			t.Fatal(err)
		} else if labels[0] < 1000 || labels[0] >= 5000 {
			t.Fatalf("file 1 should hit list 1 ids, got %v", labels)
		}
		if b1.TableBytes == 0 {
			t.Fatal("first probe of a second invlist file must read its own table")
		}
	})

	t.Run("resetDropsCache", func(t *testing.T) {
		prefix := filepath.Join(t.TempDir(), "db")
		writeThreeListIVF(t, prefix, [][]int{{0, 1, 2}})
		header, err := faiss.ReadIndexDist(prefix, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer header.Delete()
		var cold, warm, after faiss.InvertedListsIOStats
		if _, _, err := faiss.ProbeClustersWithIO(header, q, 1, 1, []int64{0}, []int64{0}, []float32{0}, "", 0, &cold); err != nil {
			t.Fatal(err)
		}
		if _, _, err := faiss.ProbeClustersWithIO(header, q, 1, 1, []int64{0}, []int64{0}, []float32{0}, "", 0, &warm); err != nil {
			t.Fatal(err)
		}
		if warm.TableBytes != 0 {
			t.Fatal("expected warm cache before reset")
		}
		if err := header.Reset(); err != nil {
			t.Fatal(err)
		}
		if _, labels, err := faiss.ProbeClustersWithIO(header, q, 1, 1, []int64{0}, []int64{0}, []float32{0}, "", 0, &after); err != nil {
			t.Fatal(err)
		} else if labels[0] != 7 {
			t.Fatalf("after reset lost id 7: %v", labels)
		}
		if after.TableBytes == 0 {
			t.Fatal("reset must drop the cached idsizes table")
		}
	})

	t.Run("gapMergeAfterCache", func(t *testing.T) {
		prefix := filepath.Join(t.TempDir(), "db")
		writeThreeListIVF(t, prefix, [][]int{{0, 1, 2}})
		header, err := faiss.ReadIndexDist(prefix, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer header.Delete()
		if _, _, err := faiss.ProbeClustersWithIO(header, q, 1, 1, []int64{0}, []int64{0}, []float32{0}, "", 0, nil); err != nil {
			t.Fatal(err)
		}
		var merged, split faiss.InvertedListsIOStats
		if _, _, err := faiss.ProbeClustersWithIO(header, q, 2, 2, []int64{0, 2}, []int64{0, 0}, []float32{0, 0}, "", 1<<30, &merged); err != nil {
			t.Fatal(err)
		}
		if merged.TableBytes != 0 || merged.MergedRanges != 1 || merged.SkipBytes == 0 {
			t.Fatalf("cached merge of lists 0 and 2: table=%d ranges=%d skip=%d", merged.TableBytes, merged.MergedRanges, merged.SkipBytes)
		}
		if _, _, err := faiss.ProbeClustersWithIO(header, q, 2, 2, []int64{0, 2}, []int64{0, 0}, []float32{0, 0}, "", 0, &split); err != nil {
			t.Fatal(err)
		}
		if split.TableBytes != 0 || split.MergedRanges < 2 {
			t.Fatalf("cached gap=0 across unread list 1: table=%d ranges=%d", split.TableBytes, split.MergedRanges)
		}
		if split.SkipBytes != 0 {
			t.Fatalf("gap=0 should not read the hole, skip_bytes=%d", split.SkipBytes)
		}
	})
}

func TestIVFAddCoreAllowsOverlap(t *testing.T) {
	quant, err := faiss.NewIndexFlatL2(2)
	if err != nil {
		t.Fatal(err)
	}
	if err := quant.Add([]float32{0, 0, 1, 0}); err != nil {
		t.Fatal(err)
	}
	idx, err := faiss.NewIndexIVFFlat(quant, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Delete()
	if err := faiss.SetIsTrained(idx, true); err != nil {
		t.Fatal(err)
	}
	vec := []float32{0, 0}
	if err := faiss.IVFAddCore(idx, vec, []int64{42}, []int64{0}); err != nil {
		t.Fatal(err)
	}
	if err := faiss.IVFAddCore(idx, vec, []int64{42}, []int64{1}); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	prefix := filepath.Join(dir, "db")
	if err := faiss.WriteIndexDistGrouped(idx, prefix, [][]int{{0, 1}}); err != nil {
		t.Fatal(err)
	}
	header, err := faiss.ReadIndexDist(prefix, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer header.Delete()
	_, labels, err := faiss.ProbeClustersWithIO(
		header, vec, 2, 2,
		[]int64{0, 1}, []int64{0, 0}, []float32{0, 0},
		"", 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	found := 0
	for _, id := range labels {
		if id == 42 {
			found++
		}
	}
	if found == 0 {
		t.Fatalf("overlapping id 42 missing from probe: %v", labels)
	}
}

func TestProbeClustersOverlapNeedsExtraK(t *testing.T) {
	quant, err := faiss.NewIndexFlatL2(2)
	if err != nil {
		t.Fatal(err)
	}
	if err := quant.Add([]float32{0, 0, 1, 0}); err != nil {
		t.Fatal(err)
	}
	idx, err := faiss.NewIndexIVFFlat(quant, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Delete()
	if err := faiss.SetIsTrained(idx, true); err != nil {
		t.Fatal(err)
	}
	near := []float32{0, 0}
	far := []float32{0.5, 0}
	if err := faiss.IVFAddCore(idx, near, []int64{1}, []int64{0}); err != nil {
		t.Fatal(err)
	}
	if err := faiss.IVFAddCore(idx, near, []int64{1}, []int64{1}); err != nil {
		t.Fatal(err)
	}
	if err := faiss.IVFAddCore(idx, far, []int64{2}, []int64{0}); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	prefix := filepath.Join(dir, "db")
	if err := faiss.WriteIndexDistGrouped(idx, prefix, [][]int{{0, 1}}); err != nil {
		t.Fatal(err)
	}
	header, err := faiss.ReadIndexDist(prefix, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer header.Delete()

	_, tight, err := faiss.ProbeClustersWithIO(
		header, near, 2, 2,
		[]int64{0, 1}, []int64{0, 0}, []float32{0, 0},
		"", 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	uniqueTight := map[int64]struct{}{}
	for _, id := range tight {
		if id >= 0 {
			uniqueTight[id] = struct{}{}
		}
	}
	if _, ok := uniqueTight[2]; ok {
		t.Skip("this Faiss heap already unique-by-id; over-fetch still safe")
	}
	if _, ok := uniqueTight[1]; !ok {
		t.Fatalf("expected duplicate id 1 to fill k=2, got %v", tight)
	}

	_, wide, err := faiss.ProbeClustersWithIO(
		header, near, 8, 2,
		[]int64{0, 1}, []int64{0, 0}, []float32{0, 0},
		"", 0, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	found2 := false
	for _, id := range wide {
		if id == 2 {
			found2 = true
		}
	}
	if !found2 {
		t.Fatalf("over-fetch k=8 should surface id 2, got %v", wide)
	}
}

func TestIndexIDMapSearchAndFilter(t *testing.T) {
	flat, err := faiss.NewIndexFlatL2(2)
	if err != nil {
		t.Fatal(err)
	}
	idx, err := faiss.NewIndexIDMap2(flat)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Delete()
	if err := idx.AddWithIDs([]float32{0, 0, 1, 0, 0, 1}, []int64{10, 11, 12}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "page")
	if err := faiss.WriteIndex(idx, path); err != nil {
		t.Fatal(err)
	}
	loaded, err := faiss.ReadIndex(path, faiss.IOFlagReadOnly)
	if err != nil {
		t.Fatal(err)
	}
	defer loaded.Delete()
	_, labels, err := faiss.SearchWithSelector(loaded, []float32{0, 0}, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if labels[0] != 10 {
		t.Fatalf("unfiltered: %v", labels)
	}
	_, filtered, err := faiss.SearchWithSelector(loaded, []float32{0, 0}, 2, []int64{11, 12})
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range filtered {
		if id == 10 {
			t.Fatalf("filter leaked id 10: %v", filtered)
		}
	}
}

func TestAbsorbInvlistsMergesDisjointGroups(t *testing.T) {
	dir := t.TempDir()
	prefix := filepath.Join(dir, "db")
	const d = 4
	quant, err := faiss.NewIndexFlatL2(d)
	if err != nil {
		t.Fatal(err)
	}
	cents := []float32{
		0, 0, 0, 0,
		10, 0, 0, 0,
		0, 10, 0, 0,
	}
	if err := quant.Add(cents); err != nil {
		t.Fatal(err)
	}
	idx, err := faiss.NewIndexIVFFlat(quant, d, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Delete()
	if err := faiss.SetIsTrained(idx, true); err != nil {
		t.Fatal(err)
	}
	if err := faiss.IVFAddCore(idx, []float32{0, 0, 0, 0}, []int64{7}, []int64{0}); err != nil {
		t.Fatal(err)
	}
	if err := faiss.IVFAddCore(idx, []float32{10, 0, 0, 0}, []int64{8}, []int64{1}); err != nil {
		t.Fatal(err)
	}
	if err := faiss.IVFAddCore(idx, []float32{0, 10, 0, 0}, []int64{9}, []int64{2}); err != nil {
		t.Fatal(err)
	}
	if err := faiss.WriteIndexDistGrouped(idx, prefix, [][]int{{0}, {1, 2}}); err != nil {
		t.Fatal(err)
	}

	header, err := faiss.ReadIndexDist(prefix, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer header.Delete()
	if err := faiss.InitRAMInvlists(header); err != nil {
		t.Fatal(err)
	}
	if header.Ntotal() != 0 {
		t.Fatalf("InitRAMInvlists ntotal=%d want 0", header.Ntotal())
	}

	mapping, err := faiss.GetListToFileMapping(header)
	if err != nil {
		t.Fatal(err)
	}
	if err := faiss.AbsorbInvlists(header, []int64{0}, []int64{mapping[0]}, prefix); err != nil {
		t.Fatal(err)
	}
	sz0, err := faiss.IVFListSize(header, 0)
	if err != nil {
		t.Fatal(err)
	}
	if sz0 != 1 || header.Ntotal() != 1 {
		t.Fatalf("after first absorb list0=%d ntotal=%d", sz0, header.Ntotal())
	}
	sz1, _ := faiss.IVFListSize(header, 1)
	if sz1 != 0 {
		t.Fatalf("list 1 should still be empty, got %d", sz1)
	}

	if err := faiss.AbsorbInvlists(header, []int64{1, 2}, []int64{mapping[1], mapping[2]}, prefix); err != nil {
		t.Fatal(err)
	}
	if header.Ntotal() != 3 {
		t.Fatalf("after second absorb ntotal=%d want 3", header.Ntotal())
	}
	before := header.Ntotal()
	if err := faiss.AbsorbInvlists(header, []int64{0, 1, 2}, []int64{mapping[0], mapping[1], mapping[2]}, prefix); err != nil {
		t.Fatal(err)
	}
	if header.Ntotal() != before {
		t.Fatalf("re-absorb should be a no-op, ntotal %d -> %d", before, header.Ntotal())
	}

	if err := faiss.TuneIVFSearch(header, 3, faiss.IVFParallelLists); err != nil {
		t.Fatal(err)
	}
	_, labels, err := header.Search([]float32{0, 0, 0, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if labels[0] != 7 {
		t.Fatalf("search recall: got id %d want 7", labels[0])
	}
}
