package faiss

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sort"
)

// MergeInvlistFiles writes one OnDiskInvertedLists file that contains every
// list from srcs. Lists are written in list-id order. Callers must not pass
// overlapping list IDs.
func MergeInvlistFiles(dst string, srcs []string) error {
	if dst == "" {
		return fmt.Errorf("MergeInvlistFiles: empty dest")
	}
	if len(srcs) == 0 {
		return fmt.Errorf("MergeInvlistFiles: no sources")
	}
	type srcList struct {
		src    string
		loc    invlistLoc
		codeSz uint64
	}
	var all []srcList
	var codeSize uint64
	seen := make(map[int64]string, 64)
	for _, src := range srcs {
		f, oDirect, err := openInvlist(src)
		if err != nil {
			return fmt.Errorf("MergeInvlistFiles: open %s: %w", src, err)
		}
		_, cs, _, idsizes, tableEnd, err := readInvlistTable(f, oDirect)
		if err != nil {
			f.Close()
			return fmt.Errorf("MergeInvlistFiles: %s %w", src, err)
		}
		if codeSize == 0 {
			codeSize = cs
		} else if cs != codeSize {
			f.Close()
			return fmt.Errorf("MergeInvlistFiles: %s codeSize %d want %d", src, cs, codeSize)
		}
		off := uint64(tableEnd)
		for i := 0; i+1 < len(idsizes); i += 2 {
			listID := int64(idsizes[i])
			nvec := idsizes[i+1]
			nbytes := nvec*codeSize + nvec*8
			if prev, ok := seen[listID]; ok {
				f.Close()
				return fmt.Errorf("MergeInvlistFiles: list %d in both %s and %s", listID, prev, src)
			}
			seen[listID] = src
			all = append(all, srcList{
				src:    src,
				loc:    invlistLoc{listID: listID, nvec: nvec, offset: int64(off), bytes: nbytes},
				codeSz: codeSize,
			})
			off += nbytes
		}
		if err := f.Close(); err != nil {
			return err
		}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].loc.listID < all[j].loc.listID })

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	keep := false
	defer func() {
		_ = out.Close()
		if !keep {
			_ = os.Remove(dst)
		}
	}()

	numList := uint64(len(all))
	idsizesLen := numList * 2
	if err := binary.Write(out, binary.LittleEndian, numList); err != nil {
		return err
	}
	if err := binary.Write(out, binary.LittleEndian, codeSize); err != nil {
		return err
	}
	if err := binary.Write(out, binary.LittleEndian, idsizesLen); err != nil {
		return err
	}
	for _, item := range all {
		if err := binary.Write(out, binary.LittleEndian, uint64(item.loc.listID)); err != nil {
			return err
		}
		if err := binary.Write(out, binary.LittleEndian, item.loc.nvec); err != nil {
			return err
		}
	}

	open := make(map[string]*os.File)
	defer func() {
		for _, f := range open {
			_ = f.Close()
		}
	}()
	srcFile := func(path string) (*os.File, error) {
		if f, ok := open[path]; ok {
			return f, nil
		}
		f, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		open[path] = f
		return f, nil
	}
	for _, item := range all {
		if item.loc.bytes == 0 {
			continue
		}
		f, err := srcFile(item.src)
		if err != nil {
			return err
		}
		if _, err := f.Seek(item.loc.offset, io.SeekStart); err != nil {
			return fmt.Errorf("MergeInvlistFiles: seek %s list %d: %w", item.src, item.loc.listID, err)
		}
		if _, err := io.CopyN(out, f, int64(item.loc.bytes)); err != nil {
			return fmt.Errorf("MergeInvlistFiles: copy %s list %d: %w", item.src, item.loc.listID, err)
		}
	}
	if err := out.Sync(); err != nil {
		return err
	}
	keep = true
	return nil
}
