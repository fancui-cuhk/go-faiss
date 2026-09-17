package faiss

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sort"
)

type invlistLoc struct {
	listID int64
	nvec   uint64
	offset int64
	bytes  uint64
}

type mergedRange struct {
	start  int64
	length uint64
	lists  []invlistLoc
}

// mergeInvlistRanges matches faiss merge_invlist_ranges: sort by offset,
// join if the hole is <= seekGapBytes (0 still joins touching clusters).
func mergeInvlistRanges(need []invlistLoc, seekGapBytes uint64) []mergedRange {
	sort.Slice(need, func(i, j int) bool {
		if need[i].offset != need[j].offset {
			return need[i].offset < need[j].offset
		}
		return need[i].listID < need[j].listID
	})
	var out []mergedRange
	for _, loc := range need {
		if loc.bytes == 0 {
			continue
		}
		if len(out) == 0 {
			out = append(out, mergedRange{start: loc.offset, length: loc.bytes, lists: []invlistLoc{loc}})
			continue
		}
		last := &out[len(out)-1]
		end := uint64(last.start) + last.length
		var gap uint64
		if uint64(loc.offset) > end {
			gap = uint64(loc.offset) - end
		}
		if gap <= seekGapBytes {
			newEnd := uint64(loc.offset) + loc.bytes
			if newEnd > end {
				last.length = newEnd - uint64(last.start)
			}
			last.lists = append(last.lists, loc)
			continue
		}
		out = append(out, mergedRange{start: loc.offset, length: loc.bytes, lists: []invlistLoc{loc}})
	}
	return out
}

// InvlistReadStats is one Go ReadInvlistsGap pass (merged-range ReadAts only).
type InvlistReadStats struct {
	ReadAts    int64
	RangeBytes int64
}

func (s *InvlistReadStats) add(o InvlistReadStats) {
	s.ReadAts += o.ReadAts
	s.RangeBytes += o.RangeBytes
}

func readInvlistFile(fname string, wanted []int64, seekGapBytes uint64) ([]InvlistPayload, InvlistReadStats, error) {
	need := make(map[int64]struct{}, len(wanted))
	for _, id := range wanted {
		need[id] = struct{}{}
	}
	f, err := os.Open(fname)
	if err != nil {
		return nil, InvlistReadStats{}, fmt.Errorf("ReadInvlists: open %s: %w", fname, err)
	}
	defer f.Close()

	numList, err := readU64(f)
	if err != nil {
		return nil, InvlistReadStats{}, fmt.Errorf("ReadInvlists: %s num_list: %w", fname, err)
	}
	codeSize, err := readU64(f)
	if err != nil {
		return nil, InvlistReadStats{}, fmt.Errorf("ReadInvlists: %s code_size: %w", fname, err)
	}
	idsizesLen, err := readU64(f)
	if err != nil {
		return nil, InvlistReadStats{}, fmt.Errorf("ReadInvlists: %s idsizes len: %w", fname, err)
	}
	if idsizesLen != numList*2 {
		return nil, InvlistReadStats{}, fmt.Errorf("ReadInvlists: %s idsizes len %d want %d", fname, idsizesLen, numList*2)
	}
	idsizes := make([]uint64, idsizesLen)
	if err := binary.Read(f, binary.LittleEndian, idsizes); err != nil {
		return nil, InvlistReadStats{}, fmt.Errorf("ReadInvlists: %s idsizes: %w", fname, err)
	}
	tableEnd, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, InvlistReadStats{}, err
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
			return nil, InvlistReadStats{}, fmt.Errorf("ReadInvlists: list %d is not in %s", id, fname)
		}
	}

	out := make([]InvlistPayload, 0, len(need))
	needLocs := make([]invlistLoc, 0, len(need))
	for _, loc := range locs {
		if _, ok := need[loc.listID]; !ok {
			continue
		}
		if loc.nvec == 0 {
			out = append(out, InvlistPayload{ListID: loc.listID})
			continue
		}
		needLocs = append(needLocs, loc)
	}

	var st InvlistReadStats
	for _, rg := range mergeInvlistRanges(needLocs, seekGapBytes) {
		buf := make([]byte, rg.length)
		if _, err := f.ReadAt(buf, rg.start); err != nil {
			return nil, st, fmt.Errorf("ReadInvlists: %s range off=%d len=%d: %w", fname, rg.start, rg.length, err)
		}
		st.ReadAts++
		st.RangeBytes += int64(rg.length)
		for _, loc := range rg.lists {
			rel := loc.offset - rg.start
			piece := buf[rel : rel+int64(loc.bytes)]
			p, err := decodeInvlistPayload(loc, piece, codeSize)
			if err != nil {
				return nil, st, fmt.Errorf("ReadInvlists: %s list %d: %w", fname, loc.listID, err)
			}
			out = append(out, p)
		}
	}
	return out, st, nil
}

func decodeInvlistPayload(loc invlistLoc, buf []byte, codeSize uint64) (InvlistPayload, error) {
	codeBytes := int(loc.nvec * codeSize)
	ids := make([]int64, loc.nvec)
	if err := binary.Read(bytes.NewReader(buf[codeBytes:]), binary.LittleEndian, ids); err != nil {
		return InvlistPayload{}, err
	}
	return InvlistPayload{
		ListID: loc.listID,
		Codes:  buf[:codeBytes],
		IDs:    ids,
	}, nil
}

func readU64(r io.Reader) (uint64, error) {
	var v uint64
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		return 0, err
	}
	return v, nil
}
