/*
Tests for SGD

http://golang.org/doc/code.html#Testing
http://www.golang-book.com/12

Example:

package newmath

import "testing"

func TestSqrt(t *testing.T) {
    const in, out = 4, 2
    if x := Sqrt(in); x != out {
        t.Errorf("Sqrt(%v) = %v, want %v", in, x, out)
    }
}

*/

package sgd

import "testing"

func Test(t *testing.T) {
    const in, out = 4, 2
    if x := Sqrt(in); x != out {
        t.Errorf("Sqrt(%v) = %v, want %v", in, x, out)
    }
}
