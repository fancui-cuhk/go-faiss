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
	f, oDirect, err := openInvlist(fname)
	if err != nil {
		return nil, InvlistReadStats{}, fmt.Errorf("ReadInvlists: open %s: %w", fname, err)
	}
	defer f.Close()

	numList, codeSize, _, idsizes, tableEnd, err := readInvlistTable(f, oDirect)
	if err != nil {
		return nil, InvlistReadStats{}, fmt.Errorf("ReadInvlists: %s %w", fname, err)
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

	ranges := mergeInvlistRanges(needLocs, seekGapBytes)
	if len(ranges) == 0 {
		return out, InvlistReadStats{}, nil
	}
	_, windows, release, err := readPayloadBlock(f, ranges, oDirect)
	if err != nil {
		return nil, InvlistReadStats{}, fmt.Errorf("ReadInvlists: %s: %w", fname, err)
	}
	pr := &payloadRelease{fn: release}
	var st InvlistReadStats
	for i, rg := range ranges {
		st.ReadAts++
		st.RangeBytes += int64(rg.length)
		buf := windows[i]
		for _, loc := range rg.lists {
			rel := loc.offset - rg.start
			if rel < 0 || int(rel)+int(loc.bytes) > len(buf) {
				pr.call()
				return nil, st, fmt.Errorf("ReadInvlists: %s list %d outside range", fname, loc.listID)
			}
			piece := buf[rel : rel+int64(loc.bytes)]
			p, err := decodeInvlistPayload(loc, piece, codeSize)
			if err != nil {
				pr.call()
				return nil, st, fmt.Errorf("ReadInvlists: %s list %d: %w", fname, loc.listID, err)
			}
			if p.release == nil {
				p.release = pr
			}
			out = append(out, p)
		}
	}
	return out, st, nil
}

// InvlistLoc is one list's byte span inside a single OnDiskInvertedLists file.
type InvlistLoc struct {
	ListID int64
	Nvec   uint64
	Offset int64
	Bytes  uint64
}

// MergedRange is a contiguous ReadAt/Range after mergeInvlistRanges.
type MergedRange struct {
	Start  int64
	Length uint64
	Lists  []InvlistLoc
}

// ParseInvlistTableBytes parses a buffer that starts at file offset 0 and is
// at least tableEnd bytes long (header + idsizes).
func ParseInvlistTableBytes(raw []byte) (locs []InvlistLoc, codeSize uint64, tableEnd int64, err error) {
	if len(raw) < 24 {
		return nil, 0, 0, fmt.Errorf("invlist table: short header (%d bytes)", len(raw))
	}
	numList := binary.LittleEndian.Uint64(raw[0:8])
	codeSize = binary.LittleEndian.Uint64(raw[8:16])
	idsizesLen := binary.LittleEndian.Uint64(raw[16:24])
	if idsizesLen != numList*2 {
		return nil, 0, 0, fmt.Errorf("invlist table: idsizes len %d want %d", idsizesLen, numList*2)
	}
	tableEnd = 24 + int64(idsizesLen)*8
	if int64(len(raw)) < tableEnd {
		return nil, 0, 0, fmt.Errorf("invlist table: short idsizes have %d want %d", len(raw), tableEnd)
	}
	idsizes := make([]uint64, idsizesLen)
	if err := binary.Read(bytes.NewReader(raw[24:tableEnd]), binary.LittleEndian, idsizes); err != nil {
		return nil, 0, 0, err
	}
	locs = make([]InvlistLoc, 0, numList)
	off := uint64(tableEnd)
	for i := 0; i+1 < len(idsizes); i += 2 {
		nvec := idsizes[i+1]
		nbytes := nvec*codeSize + nvec*8
		locs = append(locs, InvlistLoc{
			ListID: int64(idsizes[i]),
			Nvec:   nvec,
			Offset: int64(off),
			Bytes:  nbytes,
		})
		off += nbytes
	}
	return locs, codeSize, tableEnd, nil
}

// InvlistTablePrefixLen is the first Range needed to learn tableEnd.
const InvlistTablePrefixLen = 24

// MergeInvlistRanges joins needed list spans when the hole is <= seekGapBytes.
func MergeInvlistRanges(need []InvlistLoc, seekGapBytes uint64) []MergedRange {
	internal := make([]invlistLoc, len(need))
	for i, loc := range need {
		internal[i] = invlistLoc{listID: loc.ListID, nvec: loc.Nvec, offset: loc.Offset, bytes: loc.Bytes}
	}
	merged := mergeInvlistRanges(internal, seekGapBytes)
	out := make([]MergedRange, len(merged))
	for i, rg := range merged {
		lists := make([]InvlistLoc, len(rg.lists))
		for j, loc := range rg.lists {
			lists[j] = InvlistLoc{ListID: loc.listID, Nvec: loc.nvec, Offset: loc.offset, Bytes: loc.bytes}
		}
		out[i] = MergedRange{Start: rg.start, Length: rg.length, Lists: lists}
	}
	return out
}

// DecodeInvlistBytes turns one list's raw codes+ids into an install payload.
func DecodeInvlistBytes(loc InvlistLoc, buf []byte, codeSize uint64) (InvlistPayload, error) {
	return decodeInvlistPayload(invlistLoc{
		listID: loc.ListID,
		nvec:   loc.Nvec,
		offset: loc.Offset,
		bytes:  loc.Bytes,
	}, buf, codeSize)
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

// openInvlist is O_DIRECT on Linux when EMBER_INVLIST_ODIRECT=1.
var openInvlist = openInvlistBuffered

func openInvlistBuffered(fname string) (*os.File, bool, error) {
	f, err := os.Open(fname)
	return f, false, err
}

func readInvlistTable(f *os.File, oDirect bool) (numList, codeSize, idsizesLen uint64, idsizes []uint64, tableEnd int64, err error) {
	head, err := readInvlistRange(f, 0, 24, oDirect)
	if err != nil {
		return 0, 0, 0, nil, 0, fmt.Errorf("header: %w", err)
	}
	numList = binary.LittleEndian.Uint64(head[0:8])
	codeSize = binary.LittleEndian.Uint64(head[8:16])
	idsizesLen = binary.LittleEndian.Uint64(head[16:24])
	if idsizesLen != numList*2 {
		return 0, 0, 0, nil, 0, fmt.Errorf("idsizes len %d want %d", idsizesLen, numList*2)
	}
	tableEnd = 24 + int64(idsizesLen)*8
	raw, err := readInvlistRange(f, 0, int(tableEnd), oDirect)
	if err != nil {
		return 0, 0, 0, nil, 0, fmt.Errorf("idsizes: %w", err)
	}
	idsizes = make([]uint64, idsizesLen)
	if err := binary.Read(bytes.NewReader(raw[24:tableEnd]), binary.LittleEndian, idsizes); err != nil {
		return 0, 0, 0, nil, 0, err
	}
	return numList, codeSize, idsizesLen, idsizes, tableEnd, nil
}

func readInvlistRange(f *os.File, off int64, n int, oDirect bool) ([]byte, error) {
	if n <= 0 {
		return nil, nil
	}
	if !oDirect {
		buf := make([]byte, n)
		got, err := f.ReadAt(buf, off)
		if got == n {
			return buf, nil
		}
		if err == nil {
			err = io.ErrUnexpectedEOF
		}
		return nil, err
	}
	return readAtDirect(f, off, n)
}
