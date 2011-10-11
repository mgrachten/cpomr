#!/bin/bash

s="barpart2-top-middle.png"

t="barpart2-bottom-middle.png"
gm convert $s -flip $t

s="barpart2-top-begin.png"

t="barpart2-top-end.png"
gm convert $s -flop $t

t="barpart2-bottom-begin.png"
gm convert $s -flip $t

t="barpart2-bottom-end.png"
gm convert $s -flip -flop $t

exit 0
#source
s="barpart-top-begin.png"
t1="barpart-bottom-begin.png"
t2="barpart-top-end.png"
t3="barpart-bottom-end.png"
# vertical
gm convert $s -flip $t1
gm convert $s -flop $t2
gm convert $s -flip -flop $t3