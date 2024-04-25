from imageai.Detection import VideoObjectDetection

model_path = "./models/yolo-tiny.h5"
input_path = "./input/video.mp4"
output_path = "./output/newvideo.avi"

detector = VideoObjectDetection()
detector.setModelTypeAsTinyYOLOv3()

detector.setModelPath(model_path)
detector.loadModel()
detections = detector.detectObjectsFromVideo(
    input_file_path=input_path,
    output_file_path=output_path,
    frames_per_second=30,
    log_progress=True,
    per_frame_function=None,  
    minimum_percentage_probability=30
)

for eachItem in detections:
    print(f"{eachItem['name']}: {eachItem['percentage_probability']}")





