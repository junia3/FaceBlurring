#!/bin/sh
python embedding.py --label psnr --option deepface
python embedding.py --label ssim --option deepface
python embedding.py --label degree --option deepface
python embedding.py --label psnr --option facenet
python embedding.py --label ssim --option facenet
python embedding.py --label degree --option facenet
