//go:build !faiss_ext

package faiss

import "fmt"

func WriteIndexDistGrouped(idx Index, mainFilename string, groups [][]int) error {
	return fmt.Errorf("WriteIndexDistGrouped: rebuild Faiss from submodule and build with -tags faiss_ext")
}

func IVFCentroids(idx Index) (centroids []float32, nlist, d int, err error) {
	return nil, 0, 0, fmt.Errorf("IVFCentroids: build with -tags faiss_ext or use ExtractIVFStats fallback")
}

func IVFClusterSizes(idx Index) (sizes []int64, codeSize int, err error) {
	return nil, 0, fmt.Errorf("IVFClusterSizes: build with -tags faiss_ext or use ExtractIVFStats fallback")
}
