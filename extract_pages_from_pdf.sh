#!/bin/bash

for pdf in *.pdf;
do
    dir="${pdf/.pdf/}"
    mkdir -p $dir
    pdfimages $pdf "$dir/$dir-page"
    pushd $dir
    ls *.ppm | parallel gm convert {} -depth 8 -resize 2048x2048 {.}.png
    rm *.ppm
    ls *.pbm | parallel gm convert {} -operator All Negate 0 -depth 8 -resize 3500x3500 {.}.png
    rm *.pbm
    popd
done
