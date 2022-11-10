#!/bin/bash

for i in {01..16..1}; do
  echo JPEG2000 Encoding test/images/kodim$i.png
  mkdir -p test/jpeg2000/$i
  for j in {1..20..1}; do
    convert test/val_desktop/$i.bmp -quality $(($j*5)) -sampling-factor 4:2:0 test/jpeg2000/$i/`printf "%02d" $j`.jp2
  done
done
