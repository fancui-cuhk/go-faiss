package faiss

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

type invlistLoc struct {
	listID int64
	nvec   uint64
	offset int64
	bytes  uint64
}

func readInvlistFile(fname string, wanted []int64) ([]InvlistPayload, error) {
	need := make(map[int64]struct{}, len(wanted))
	for _, id := range wanted {
		need[id] = struct{}{}
	}
	f, err := os.Open(fname)
	if err != nil {
		return nil, fmt.Errorf("ReadInvlists: open %s: %w", fname, err)
	}
	defer f.Close()

	numList, err := readU64(f)
	if err != nil {
		return nil, fmt.Errorf("ReadInvlists: %s num_list: %w", fname, err)
	}
	codeSize, err := readU64(f)
	if err != nil {
		return nil, fmt.Errorf("ReadInvlists: %s code_size: %w", fname, err)
	}
	idsizesLen, err := readU64(f)
	if err != nil {
		return nil, fmt.Errorf("ReadInvlists: %s idsizes len: %w", fname, err)
	}
	if idsizesLen != numList*2 {
		return nil, fmt.Errorf("ReadInvlists: %s idsizes len %d want %d", fname, idsizesLen, numList*2)
	}
	idsizes := make([]uint64, idsizesLen)
	if err := binary.Read(f, binary.LittleEndian, idsizes); err != nil {
		return nil, fmt.Errorf("ReadInvlists: %s idsizes: %w", fname, err)
	}
	tableEnd, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	locs := make([]invlistLoc, 0, numList)
	present := make(map[int64]struct{}, numList)
	off := uint64(tableEnd)
	for i := 0; i+1 < len(idsizes); i += 2 {
		listID := int64(idsizes[i])
		nvec := idsizes[i+1]
		nbytes := nvec*codeSize + nvec*8
		locs = append(locs, invlistLoc{listID: listID, nvec: nvec, offset: int64(off), bytes: nbytes})
		present[listID] = struct{}{}
		off += nbytes
	}
	for id := range need {
		if _, ok := present[id]; !ok {
			return nil, fmt.Errorf("ReadInvlists: list %d is not in %s", id, fname)
		}
	}

	out := make([]InvlistPayload, 0, len(need))
	for _, loc := range locs {
		if _, ok := need[loc.listID]; !ok {
			continue
		}
		if loc.nvec == 0 {
			out = append(out, InvlistPayload{ListID: loc.listID})
			continue
		}
		buf := make([]byte, loc.bytes)
		if _, err := f.ReadAt(buf, loc.offset); err != nil {
			return nil, fmt.Errorf("ReadInvlists: %s list %d: %w", fname, loc.listID, err)
		}
		codeBytes := int(loc.nvec * codeSize)
		ids := make([]int64, loc.nvec)
		if err := binary.Read(bytes.NewReader(buf[codeBytes:]), binary.LittleEndian, ids); err != nil {
			return nil, fmt.Errorf("ReadInvlists: %s list %d ids: %w", fname, loc.listID, err)
		}
		out = append(out, InvlistPayload{
			ListID: loc.listID,
			Codes:  buf[:codeBytes],
			IDs:    ids,
		})
	}
	return out, nil
}

func readU64(r io.Reader) (uint64, error) {
	var v uint64
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		return 0, err
	}
	return v, nil
}
