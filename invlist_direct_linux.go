//go:build linux

package faiss

import (
	"fmt"
	"io"
	"os"
	"sync/atomic"
	"syscall"
	"unsafe"
)

var oDirectLogged atomic.Bool

const ioAlign = 4096

func init() {
	openInvlist = openInvlistDirect
}

func openInvlistDirect(fname string) (*os.File, bool, error) {
	if os.Getenv("EMBER_INVLIST_ODIRECT") != "1" {
		return openInvlistBuffered(fname)
	}
	f, err := os.OpenFile(fname, os.O_RDONLY|syscall.O_DIRECT, 0)
	if err == nil && oDirectLogged.CompareAndSwap(false, true) {
		fmt.Fprintln(os.Stderr, "ReadInvlists: O_DIRECT enabled")
	}
	return f, true, err
}

func alignedSlice(n int) []byte {
	raw := make([]byte, n+ioAlign*2)
	p := uintptr(unsafe.Pointer(&raw[0]))
	pad := int((ioAlign - p%ioAlign) % ioAlign)
	return raw[pad : pad+n]
}

// ReadIndexDirect reads path with O_DIRECT and deserializes the bytes.
// It does not fall back to a buffered read.
func ReadIndexDirect(path string, ioflags int) (*IndexImpl, error) {
	f, err := os.OpenFile(path, os.O_RDONLY|syscall.O_DIRECT, 0)
	if err != nil {
		return nil, fmt.Errorf("ReadIndexDirect: open %s: %w", path, err)
	}
	defer f.Close()
	if oDirectLogged.CompareAndSwap(false, true) {
		fmt.Fprintln(os.Stderr, "ReadIndex: O_DIRECT enabled")
	}
	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("ReadIndexDirect: stat %s: %w", path, err)
	}
	if st.Size() > int64(^uint(0)>>1) {
		return nil, fmt.Errorf("ReadIndexDirect: %s too large", path)
	}
	buf, err := readAtDirect(f, 0, int(st.Size()))
	if err != nil {
		return nil, fmt.Errorf("ReadIndexDirect: %s: %w", path, err)
	}
	return ReadIndexBytes(buf, ioflags)
}

func readAtDirect(f *os.File, off int64, n int) ([]byte, error) {
	aoff := off - off%ioAlign
	extra := int(off - aoff)
	alen := extra + n
	alen = (alen + ioAlign - 1) / ioAlign * ioAlign
	buf := alignedSlice(alen)
	got, err := f.ReadAt(buf, aoff)
	need := extra + n
	if got < need {
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		return nil, fmt.Errorf("o_direct short read off=%d n=%d got=%d: %w", off, n, got, err)
	}
	out := make([]byte, n)
	copy(out, buf[extra:need])
	return out, nil
}
