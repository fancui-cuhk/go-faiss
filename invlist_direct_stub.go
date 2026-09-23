//go:build !linux

package faiss

import (
	"os"
)

func readAtDirect(f *os.File, off int64, n int) ([]byte, error) {
	buf := make([]byte, n)
	_, err := f.ReadAt(buf, off)
	return buf, err
}
