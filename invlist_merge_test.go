package faiss

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func writeTinyInvlistFile(t *testing.T, path string, listIDs []int64) {
	t.Helper()
	const codeSize = 4
	n := uint64(len(listIDs))
	var raw []byte
	raw = binary.LittleEndian.AppendUint64(raw, n)
	raw = binary.LittleEndian.AppendUint64(raw, codeSize)
	raw = binary.LittleEndian.AppendUint64(raw, n*2)
	for _, id := range listIDs {
		raw = binary.LittleEndian.AppendUint64(raw, uint64(id))
		raw = binary.LittleEndian.AppendUint64(raw, 1)
	}
	for i, id := range listIDs {
		raw = append(raw, byte(i), 0, 0, 0)
		raw = binary.LittleEndian.AppendUint64(raw, uint64(id))
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0644); err != nil {
		t.Fatal(err)
	}
}

func TestMergeInvlistFiles_combinesListIDs(t *testing.T) {
	dir := t.TempDir()
	a := filepath.Join(dir, "a")
	b := filepath.Join(dir, "b")
	writeTinyInvlistFile(t, a, []int64{1, 3})
	writeTinyInvlistFile(t, b, []int64{2})
	dst := filepath.Join(dir, "merged")
	if err := MergeInvlistFiles(dst, []string{a, b}); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(dst)
	if err != nil {
		t.Fatal(err)
	}
	locs, codeSize, _, err := ParseInvlistTableBytes(raw)
	if err != nil {
		t.Fatal(err)
	}
	if codeSize != 4 || len(locs) != 3 {
		t.Fatalf("codeSize=%d locs=%d", codeSize, len(locs))
	}
	got := make([]int64, len(locs))
	for i, loc := range locs {
		got[i] = loc.ListID
	}
	if got[0] != 1 || got[1] != 2 || got[2] != 3 {
		t.Fatalf("list ids %v want 1,2,3", got)
	}
}

func TestParseInvlistTableBytes_prefixTooShort(t *testing.T) {
	if _, _, _, err := ParseInvlistTableBytes([]byte{1, 2, 3}); err == nil {
		t.Fatal("short header must fail")
	}
}
