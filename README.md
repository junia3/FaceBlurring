# Face blurring

### Generate face blurring dataset
---
# Process to create and save face blur images
Basically you need pytorch, numpy, opencv-python, matplotlib, ... libraries. I do not detailed introduction for installing.
## 1. Cloning repository and get ready to generate samples

```bash
  git clone https://github.com/junia3/FaceBlurring.git
```
You can just cloning this repo into your own computer  
And finally the directory hierarchy is configured as,  

```bash
FaceBlurring
├── config
│   ├── test.txt
├── dataset
│   ├── blur.py
│   ├── create_blurring.py
│   ├── dataset.py
│   └── utils.py
└── data
``` 
This is just a framework to create dataset. You should add your own "face dataset" into 'data' directory.
### test.txt
```txt
../data/sample_root/
```
If you add your own data samples to directory, add all data roots in the test.txt file line by line. For example, if you configured data samples like below,
```bash
...
data
├── sample_root1
│   ├── clean
│   │   ├── face_sample0.png(or .jpg)
│   │   ├── face_sample1.png(or .jpg)
│   │   ├── face_sample2.png(or .jpg)
├── sample_root2
│   ├── clean
│   │   ├── face_sample0.png(or .jpg)
│   │   ├── face_sample1.png(or .jpg)
│   │   ├── face_sample2.png(or .jpg)
└── sample_root3
```
You have to update test.txt file as
```txt
../data/sample_root1/
../data/sample_root2/
../data/sample_root3/
```

to generate blur images for all files in sample roots.  
You should first get samples and make directory like above structure(warning!!!! you cannot change directory name "clean" or "data", but you can freely change name of "sample_root")

---
## 2. create and save blur data samples
(Updated 22/08/04) I made two options to create blur images
- method from deblurGAN(https://openaccess.thecvf.com/content_cvpr_2018/html/Kupyn_DeblurGAN_Blind_Motion_CVPR_2018_paper.html) and github https://github.com/KupynOrest/DeblurGAN

- method from Defocus and Motion Blur Detection with Deep Contextual Features(https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.13567) and github https://github.com/Imalne/Defocus-and-Motion-Blur-Detection-with-Deep-Contextual-Features

I show an example command to create blurred images and save them with label information
```bash
   cd /dataset
   python create_blurring.py --blur defocus --save True --label True
```
Above command would generate blurred image with defocus method. Another option is 'deblurGAN'
```bash
   cd /dataset
   python create_blurring.py --blur deblurGAN --save True --label True
```
and simply you can just generate blur images and save them with labels, into default setting(defocus, save images and label).
```bash
   cd /dataset
   python create_blurring.py
```
---
## 3. Evaluate and Visualize samples
I created dataset module, and check how the images are labeled with psnr metric(it can be updated later with better metrics).
You can run dataset example 'after' generate blurred images.
```bash
   cd /dataset
   python dataset.py
```
And do not forget to update 'text.txt' before run this command.
