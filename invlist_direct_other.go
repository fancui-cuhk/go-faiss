//go:build !linux

package faiss

import "fmt"

// ReadIndexDirect is the O_DIRECT page load. This platform has no O_DIRECT.
func ReadIndexDirect(path string, ioflags int) (*IndexImpl, error) {
	_ = ioflags
	return nil, fmt.Errorf("ReadIndexDirect: O_DIRECT is not implemented on this platform (%s)", path)
}
