# How to train filters

## How to config
Read Angle, Strength, Coherence value in applications.media.ai.raisr-native-library/filtersX/filternotes.txt

## For filters1 and filters5
2nd pass filters same as 1st pass, simple copy and add "_2"

## For filters2, 3, 4.
```
Use ffmpeg to generate sharpened images from HR folder to create content in new folder eg. HR_Sharp
Run the following command: python3 Train.py -ohf HR_Sharp -chf HR -bits 10 -ff filter_sharp
Make sure the strength, coherence and angle values are same as the filter you are trying to train for (24 3 3 or 80 10 10)
After training is complete, rename the files to _2 and copy them to the other filter folder in which you want to use them (eg. filter2)

For 8bit image/video:
For sharpen with value 0.5: for i in `ls val2017`; do ./ffmpeg -i val2017/$i -vf unsharp=5:5:0.5:5:5:0.0 sharpen/s05/$i; done
For sharpen with value 1.0: for i in `ls val2017`; do ./ffmpeg -i val2017/$i -vf unsharp=5:5:1:5:5:0.0 sharpen/s10/$i; done
For sharpen with value 1.5: for i in `ls val2017`; do ./ffmpeg -i val2017/$i -vf unsharp=5:5:1.5:5:5:0.0 sharpen/s15/$i; done

For 10bit video:
ffmpeg sharpen filter doesn't work for 10bits(though the algo is correct, there is a bug in the implementation). Use our own unsharp_mask instead(same algo with ffmpeg).
python3 Sharpen.py -bits 10 -i input_folder -o output_folder
Note: input_folder contains frame in y4m format. 1 frame per y4m file.
```
