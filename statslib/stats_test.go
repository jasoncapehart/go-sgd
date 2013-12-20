/*
Tests for stats

Ref:
http://golang.org/doc/code.html#Testing
http://www.golang-book.com/12

*/

package stats

import (
    "math"
    "testing"
)

func TestIdentityLink(t *testing.T) {
    x := 2.0
    output := link_map["identity"](x)

    if output != 2.0 {
        t.Errorf("Identity link function broken")
    }
}

func TestLogitLink(t *testing.T) {
    x := 2.0
    y := 1 / (1.0 + math.Exp(-x))
    output := link_map["logit"](x)

    if output != y {
        t.Errorf("Logit link function broken")
    }
}
