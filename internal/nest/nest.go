package nest

import (
	"context"
	"errors"
)

var (
	ErrNestFull  = errors.New("nest is full - come back later")
	ErrNestEmpty = errors.New("nest is empty")
)

type Nest[T any] struct {
	ch chan T
}

func New[T any](capacity int) *Nest[T] {
	return &Nest[T]{
		// Create a buffered channel with fixed size
		ch: make(chan T, capacity),
	}
}

// Push tries to add an item.
// If full, it returns ErrNestFull IMMEDIATELY (The "Bounce").
func (n *Nest[T]) Push(item T) error {
	select {
	case n.ch <- item:
		// Success! The item is in the buffer.
		return nil
	default:
		// The channel is full.
		// Instead of blocking (waiting), we bounce the request.
		return ErrNestFull
	}
}

// Pop waits for an item.
// If context is cancelled (timeout/disconnect), returns error.
func (n *Nest[T]) Pop(ctx context.Context) (T, error) {
	var zero T // Needed to return a "nil" value for generic T

	select {
	case item := <-n.ch:
		// We got an egg!
		return item, nil
	case <-ctx.Done():
		// The caller gave up (timeout) before we found an egg.
		return zero, ctx.Err()
	}
}
