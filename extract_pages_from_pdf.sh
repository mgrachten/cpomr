#!/bin/bash

for pdf in *.pdf;
do
    dir="${pdf/.pdf/}"
    mkdir -p $dir
    pdfimages $pdf "$dir/$dir-page"
    pushd $dir
    ls *.ppm | parallel convert {} -depth 8 -resize 4096x4096 {.}.png
    rm *.ppm
    popd
done
