# face-recognition 
this is pretty self explanatory but since i do want this software to be open source i do want a good bit of contribution IF possible, if you see anything directly wrong with the code let me know and ill try my best to fix it, pull requests are kinda okay but usually ill just pick and choose what i want from them 

also side note: video is really bad and didnt work for me, let me know if it works for you 


## start 

```powershell
python -m venv .venv
pip install -r requirements.txt
python cli.py (at #3) 
```

## 1) again self explanatory you need images to feed on, they can be named whatever as long as naming is within the right folder 

```text
known/
  Alice/
    1.jpg
    2.jpg
  Bob/
    1.jpg
```

## 2) enroll your db 

```powershell
python cli.py enroll --known-dir known --db-path data/faces.db
```

## 3) simple run usage 

```powershell
python cli.py analyze-images --input-path samples/images --db-path data/faces.db --output-dir runs/images --annotate
python cli.py analyze-video --video-path samples/video.mp4 --db-path data/faces.db --output-dir runs/video --frame-step 5 --annotate-frames
```

