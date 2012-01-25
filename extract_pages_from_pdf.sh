#!/bin/bash

maxDim=1500
for pdf in *.pdf;
do
    dir="${pdf/.pdf/}"
    mkdir -p $dir
    pdfimages $pdf "$dir/$dir-page"
    pushd $dir
    ls *.ppm | parallel gm convert {} -depth 8 -resize ${maxDim}x${maxDim} {.}.png
    rm *.ppm
    ls *.pbm | parallel gm convert {} -flip -operator All Negate 0 -depth 8 -resize ${maxDim}x${maxDim} {.}.png
    #ls *.pbm | parallel gm convert {} -depth 8 -resize ${maxDim}x${maxDim} {.}.png
    rm *.pbm
    popd
done
