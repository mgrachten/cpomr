#!/bin/bash

IFS='
'

which parallel > /dev/null
haveparallel=$?

maxDim=2500
for pdf in $@
do
    bname=$(basename $pdf)
    dir="${bname/.pdf/}"
    mkdir -p $dir
    pdfimages $pdf "$dir/$dir-page"
    pushd $dir
    if [ $haveparallel -eq 0 ]
    then
	ls *.p[pb]m | parallel gm convert {} -depth 8 -resize ${maxDim}x${maxDim} {.}.png
        # ls *.pbm | parallel gm convert {} -flip -operator All Negate 0 -depth 8 -resize ${maxDim}x${maxDim} {.}.png
	#ls *.pbm | parallel gm convert {} -depth 8 -resize ${maxDim}x${maxDim} {.}.png
    else
	for i in $(/bin/ls *.p[pb]m)
	do
	    outname=$(echo $i|perl -pe 's/(.*)\.[^.]*$/\1.png/')
	    echo $i $outname
	    gm convert $i -depth 8 -resize ${maxDim}x${maxDim} ${outname}
	done
    fi   
    rm *.ppm 2&> /dev/null
    rm *.pbm 2&> /dev/null
    popd
done
