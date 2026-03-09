# face-recognition 
this is pretty self explanatory but since i do want this software to be open source i do want a good bit of contribution IF possible, if you see anything directly wrong with the code let me know and ill try my best to fix it, pull requests are kinda okay but usually ill just pick and choose what i want from them 


## start 

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python cli.py --help
```

## 1) add some faces, this can be bob or alice or elon,  we already have elon but no bob and alice so feel free to introduce them 

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
python cli.py scan-screen --db-path data/faces.db --monitor 1 --capture-scale 0.5 --max-fps 8 --threshold 0.45
```

