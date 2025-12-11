#!/usr/bin/env python3
from PIL import Image
import glob
import os

pattern = os.path.join('figures','*.png')
files = glob.glob(pattern)
if not files:
    print('No PNG files found in figures/ to upscale.')
    exit(0)

for p in files:
    try:
        img = Image.open(p)
        w,h = img.size
        new_size = (w*2, h*2)
        img2 = img.resize(new_size, Image.LANCZOS)
        # Save with DPI metadata
        img2.save(p, dpi=(300,300))
        print('Upscaled and saved', p, '->', new_size)
    except Exception as e:
        print('Failed for', p, ':', e)
