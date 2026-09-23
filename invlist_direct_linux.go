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
