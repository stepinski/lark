package nest_test

import (
	"context"
	"testing"
	"time"

	"github.com/stepinski/lark/internal/nest"
)

func TestNest_Push_NonBlocking(t *testing.T) {
	// 1. Create a tiny nest (capacity 1)
	n := nest.New[string](1)

	// 2. First push should succeed
	if err := n.Push("egg-1"); err != nil {
		t.Fatalf("First push failed: %v", err)
	}

	// 3. Second push should FAIL immediately (Nest is full)
	if err := n.Push("egg-2"); err != nest.ErrNestFull {
		t.Fatalf("Expected ErrNestFull, got %v", err)
	}
}

func TestNest_Pop_ContextCancel(t *testing.T) {
	n := nest.New[int](5)
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// 4. Try to Pop from an empty nest
	// This should block until timeout, then return error
	_, err := n.Pop(ctx)

	if err != context.DeadlineExceeded {
		t.Errorf("Expected DeadlineExceeded, got %v", err)
	}
}
