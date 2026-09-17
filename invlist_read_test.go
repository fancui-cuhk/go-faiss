package faiss

import "testing"

func TestMergeInvlistRangesTouchesJoinEvenIfGapZero(t *testing.T) {
	need := []invlistLoc{
		{listID: 1, offset: 150, bytes: 50},
		{listID: 0, offset: 100, bytes: 50},
	}
	got := mergeInvlistRanges(need, 0)
	if len(got) != 1 {
		t.Fatalf("touching clusters: %d ranges, want 1", len(got))
	}
	if got[0].start != 100 || got[0].length != 100 || len(got[0].lists) != 2 {
		t.Fatalf("range %+v", got[0])
	}
}

func TestMergeInvlistRangesHoleNeedsBudget(t *testing.T) {
	const hole = 100
	need := []invlistLoc{
		{listID: 0, offset: 0, bytes: 50},
		{listID: 2, offset: 50 + hole, bytes: 50},
	}
	split := mergeInvlistRanges(need, hole-1)
	if len(split) != 2 {
		t.Fatalf("hole %d with budget %d: %d ranges, want 2", hole, hole-1, len(split))
	}
	joined := mergeInvlistRanges(need, hole)
	if len(joined) != 1 {
		t.Fatalf("hole %d with budget %d: %d ranges, want 1", hole, hole, len(joined))
	}
	if joined[0].length != 50+hole+50 {
		t.Fatalf("joined length %d want %d", joined[0].length, 50+hole+50)
	}
}

func TestMergeInvlistRangesSkipsEmpty(t *testing.T) {
	need := []invlistLoc{
		{listID: 0, offset: 0, bytes: 0},
		{listID: 1, offset: 0, bytes: 40},
	}
	got := mergeInvlistRanges(need, 1<<20)
	if len(got) != 1 || got[0].lists[0].listID != 1 {
		t.Fatalf("%+v", got)
	}
}

func TestMergeInvlistRangesCountIsReadAtCount(t *testing.T) {
	need := []invlistLoc{
		{listID: 0, offset: 0, bytes: 50},
		{listID: 1, offset: 50, bytes: 50},
		{listID: 2, offset: 50 + 50 + 3<<20, bytes: 50},
	}
	got := mergeInvlistRanges(need, 2<<20)
	if len(got) != 2 {
		t.Fatalf("want 2 ReadAts after merge, got %d", len(got))
	}
}
