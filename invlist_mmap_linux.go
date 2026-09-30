//go:build linux

package faiss

import (
	"fmt"
	"io"
	"os"
	"syscall"
)

func mapAlloc(n int) ([]byte, error) {
	if n <= 0 {
		return nil, fmt.Errorf("mapAlloc: empty")
	}
	page := syscall.Getpagesize()
	if page <= 0 {
		return nil, fmt.Errorf("mapAlloc: bad page size")
	}
	if n > int(^uint(0)>>1)-page {
		return nil, fmt.Errorf("mapAlloc: too large")
	}
	n = (n + page - 1) / page * page
	b, err := syscall.Mmap(-1, 0, n, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANONYMOUS)
	if err != nil {
		return nil, fmt.Errorf("mmap %d: %w", n, err)
	}
	return b, nil
}

func mapFree(b []byte) {
	if len(b) == 0 {
		return
	}
	_ = syscall.Munmap(b)
}

// readPayloadBlock reads every merged range into one anonymous mapping.
// windows[i] is the logical file bytes of ranges[i] (O_DIRECT alignment
// prefix is not part of the window). release munmaps the original mapping.
func readPayloadBlock(f *os.File, ranges []mergedRange, oDirect bool) (block []byte, windows [][]byte, release func(), err error) {
	if len(ranges) == 0 {
		return nil, nil, func() {}, nil
	}
	page := syscall.Getpagesize()
	type slot struct {
		dest    int
		alen    int
		extra   int
		n       int
		fileOff int64
	}
	slots := make([]slot, len(ranges))
	cursor := 0
	for i, rg := range ranges {
		if rg.length > uint64(int(^uint(0)>>1)) {
			return nil, nil, nil, fmt.Errorf("invlist range too large")
		}
		n := int(rg.length)
		extra := 0
		alen := n
		fileOff := rg.start
		if oDirect {
			page64 := int64(page)
			if page64 <= 0 || rg.start < 0 {
				return nil, nil, nil, fmt.Errorf("invlist O_DIRECT bad offset %d", rg.start)
			}
			aoff := rg.start - rg.start%page64
			extra = int(rg.start - aoff)
			alen = extra + n
			alen = (alen + page - 1) / page * page
			fileOff = aoff
		}
		if rem := cursor % page; rem != 0 {
			cursor += page - rem
		}
		if cursor > int(^uint(0)>>1)-alen {
			return nil, nil, nil, fmt.Errorf("invlist payload too large")
		}
		slots[i] = slot{dest: cursor, alen: alen, extra: extra, n: n, fileOff: fileOff}
		cursor += alen
	}
	block, err = mapAlloc(cursor)
	if err != nil {
		return nil, nil, nil, err
	}
	release = func() { mapFree(block) }
	windows = make([][]byte, len(ranges))
	for i, sl := range slots {
		need := sl.extra + sl.n
		dst := block[sl.dest : sl.dest+sl.alen]
		var got int
		var rerr error
		if oDirect {
			got, rerr = f.ReadAt(dst, sl.fileOff)
		} else {
			got, rerr = f.ReadAt(dst[:sl.n], sl.fileOff)
			need = sl.n
		}
		if got < need {
			release()
			if rerr == nil {
				rerr = io.ErrUnexpectedEOF
			}
			return nil, nil, nil, rerr
		}
		windows[i] = block[sl.dest+sl.extra : sl.dest+sl.extra+sl.n]
	}
	return block, windows, release, nil
}
