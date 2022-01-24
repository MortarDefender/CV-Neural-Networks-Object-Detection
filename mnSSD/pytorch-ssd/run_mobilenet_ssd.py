import cv2
import json
from vision.utils.misc import Timer
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor


def run(modelPath, labelPath, VideoCaptureFileName = None):
    
    if VideoCaptureFileName is None:
        videoCapture = cv2.VideoCapture(VideoCaptureFileName)
    else:
        videoCapture = cv2.VideoCapture(0)   # capture from camera
        videoCapture.set(3, 1920)
        videoCapture.set(4, 1080)
    
    class_names = [name.strip() for name in open(labelPath).readlines()]
    
    net = create_mobilenetv1_ssd(len(class_names), is_test = True)
    
    net.load(modelPath)
    
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size = 200)
    
    timer = Timer()
    
    while True:
        _, orig_image = videoCapture.read()
        
        if orig_image is None:
            continue
        
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        timer.start()
        boxes, labels, probs = predictor.predict(image, 10, 0.4)
        interval = timer.end()
        
        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
    
            cv2.putText(orig_image, label, (box[0] + 20, box[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow('annotated', orig_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    run(config["modelPath"], config["labelPath"], config["testVideo"])
