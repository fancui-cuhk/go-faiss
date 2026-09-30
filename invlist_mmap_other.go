//go:build !linux

package faiss

import (
	"fmt"
	"os"
)

func mapAlloc(n int) ([]byte, error) {
	return nil, fmt.Errorf("mapAlloc: anonymous mmap is linux-only (%d bytes)", n)
}

func mapFree(b []byte) {}

func readPayloadBlock(f *os.File, ranges []mergedRange, oDirect bool) ([]byte, [][]byte, func(), error) {
	return nil, nil, nil, fmt.Errorf("invlist payload mmap is linux-only")
}
