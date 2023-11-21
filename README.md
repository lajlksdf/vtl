
## 📜 Directory

- [Introduction](#-introduction)
- [Train or Test](#-train-or-test)
  - [Train](#train)
  - [Test](#test)
- [Downloads](#%EF%B8%8F-downloads)
  - [Download Models and Datasets](#download-models-and-datasets)
  - [DeepFake Dataset Samples](#deepfake-dataset-samples)
  - [Video Inpainting Dataset Samples](#video-inpainting-dataset-samples)
  - [Video Splicing Dataset Samples](#video-splicing-dataset-samples)

## ✨ Introduction

An effective and reliable method for fake video detection.


## 🔬 Train or Test

### Train

**Train ViTHash**

```apache
python train_h.py --local_rank=0 --path=../Celeb-DF --type=Celeb-DF --hash_bits=512 --pretrained=x
```

### Test

**Test ViTHash**

```apache
python test_h.py --local_rank=0 --path=../Celeb-DF --type=Celeb-DF --hash_bits=512 --pretrained=x
```

## 🚀️ Downloads

### Download Models and Datasets
> ​The key of the JSON file corresponds to the name of the source video, and the value is the hash center of the source video.​


**Pretrained models, hash centers and datasets**


> Due to some software restrictions, the original sharing link is invalid and changed to Google storage.

| Item          | Link                                                                     | Comments                                                        |
| --------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| DeepFake   | https://drive.google.com/file/d/11pPA8buSTzhzypB-D91DjJEFR4AOFVGJ/view?usp=share_link | Google drive |
| Video Splicing | https://drive.google.com/file/d/1-3mggFEiVRTPnR3PQ3yNHKsUyJ84jUkY/view?usp=share_link | Google drive |
| Similar Scenes| https://drive.google.com/file/d/1-0VBLygXSrQrYOpQqmQY8q_qv5r0uOGo/view?usp=share_link |  Google drive |
| Video Inpainting      | https://drive.google.com/file/d/1-35iQpD7RQTGGAFmJKSd3rptahnJuhmj/view?usp=share_link | Google drive. Extension of[DAVIS2016](https://davischallenge.org/) |

### DeepFake Dataset Samples

> The person with different scenes.

<div style="align-items: center;text-align: center; display: inline-block" >
    <div>
        <h3 align="center">Source Videos</h3>
        <div  align="center">
            <img src="./assets/girl_r0.gif" width="30%">
            <img src="./assets/girl_r1.gif" width="30%">
            <img src="./assets/girl_r2.gif" width="30%">
        </div>
    </div>
    <div>
        <h3 align="center">Fake Videos</h3>
        <div  align="center">
            <img src="./assets/girl_f0.gif" width="30%">
            <img src="./assets/girl_f1.gif" width="30%">
            <img src="./assets/girl_f2.gif" width="30%">
        </div>
    </div>  
</div>

> Different fake videos are made from the same source.

<div style="align-items: center;text-align: center; display: inline-block" >
    <div>
        <h3 align="center">Source Video</h3>
        <div  align="center">
            <img src="./assets/40.gif" width="30%">
        </div>
    </div>
    <div>
        <h3 align="center">Fake Videos</h3>
        <div  align="center">
            <img src="./assets/male-f1.gif" width="30%">
            <img src="./assets/male-f2.gif" width="30%">
            <img src="./assets/male-f3.gif" width="30%">
        </div>
    </div>  
</div>

### Video Inpainting Dataset Samples

> The first gif of the boat is the original video, and the remaining five videos are generated by different object inpainting methods.

<div align="center" >
    <div>
        <div  align="center">
            <img src="./assets/boat.gif" width="30%" >
            <img src="./assets/boat_f0.gif" width="30%" >
            <img src="./assets/boat_f2.gif" width="30%" >
        </div>
    </div>
    <div>
        <div  align="center">
            <img src="./assets/boat_f3.gif" width="30%" >
            <img src="./assets/boat_f4.gif" width="30%" >
            <img src="./assets/boat_f5.gif" width="30%" >
        </div>
    </div>  
</div>

### Video Splicing Dataset Samples

> The bird, person, soldier and planes are all spliced by manual.

<div align="center" >
    <div>
        <div  align="center">
            <img src="./assets/bird.gif" width="30%" >
            <img src="./assets/people.gif" width="30%" >
            <img src="./assets/soldier1.gif" width="30%" >
        </div>
    </div>
    <div>
        <div  align="center">
            <img src="./assets/plane2.gif" width="30%" >
            <img src="./assets/plane3.gif" width="30%" >
            <img src="./assets/plane4.gif" width="30%" >
        </div>
    </div>  
</div>

### Tips

_We have developed an additional tool to compare the difference between real video and fake video. Please see [tool](./ComparisonTool.md)_
