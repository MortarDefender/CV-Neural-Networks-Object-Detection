import cv2
import sys
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd_predictor


def getModel(label_path, model_path = "savedModel.pth"):
    class_names = [name.strip() for name in open(label_path).readlines()] # get the labels names
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)          # generate the model
    net.load(model_path)                                                  # load the pretrained model
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200) # generate the model predictor

    return predictor, class_names


def detectObjectInImage(image_path, label_path, model_path = "savedModel.pth", output_path = "output.jpg"):

    predictor, class_names = getModel(label_path, model_path)

    origin_image = cv2.imread(image_path)
    image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.4)

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(origin_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(origin_image, label, (box[0] + 20, box[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imwrite(output_path, origin_image)
    print(f"Found {len(probs)} objects. The output image is {output_path}")


def detectObjectInVideo(video_path, label_path, model_path = "savedModel.pth", output_path = "results.avi"):
    
    videoWriter = None
    videoCapture = cv2.VideoCapture(video_path)
    predictor, class_names = getModel(label_path, model_path)

    while True:
        ret, orig_image = videoCapture.read()
        
        if not ret:
            break
        
        if orig_image is None:
            continue
        
        if videoWriter is None:
            height, width, layers = orig_image.shape
            videoWriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height))
        
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = predictor.predict(image, 10, 0.4)
        print('Detect Objects: {:d}.'.format(labels.size(0)))
        
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            cv2.putText(orig_image, label, (box[0]+20, box[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        
        videoWriter.write(orig_image)
    
    videoWriter.release()
    videoCapture.release()
    cv2.destroyAllWindows()


def usage():
    print("usage: python model.py <mode picture/video> <image/video item> <labels file>")


def main():

    label_file = None

    if len(sys.argv) != 4:
        if len(sys.argv) == 3:
            label_file = "labels.txt"
        else:
            usage()
    else:
        label_file = sys.argv[3]
    
    if sys.argv[1].lower() == "picture":
        detectObjectInImage(sys.argv[2], label_file)
    elif sys.argv[1].lower() == "video":
        detectObjectInVideo(sys.argv[2], label_file)
    else:
        usage()


if __name__ == '__main__':
    main()
