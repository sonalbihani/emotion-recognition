import cv2
from face_detect import find_faces
from image_commons import nparray_as_image, draw_with_alpha


def _load_emoticons(emotions):
    return [nparray_as_image(cv2.imread('graphics\\%s.png' % emotion, -1), mode=None) for emotion in emotions]


def show_webcam_and_run(model, emoticons, window_size=None):
    vc = cv2.VideoCapture(0)
    while True:
        read_value, webcam_image = vc.read()
        d,t=vc.read()
        for normalized_face, (x, y, w, h) in find_faces(webcam_image):
            prediction = model.predict(normalized_face)
            # do prediction
            prediction = prediction[0]
      

            image_to_draw = emoticons[prediction]
            draw_with_alpha(webcam_image, image_to_draw, (x, y, w, h))

        
        cv2.imshow("EMOTION", webcam_image)
        cv2.imshow("NORMAL",t)
        #read_value, webcam_image = vc.read()
        key = cv2.waitKey(10)

        if key == 27:  # exit on ESC
            break

    vc.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    emotions = ["neutral", "anger", "disgust", "happy", "surprise"]
    emoticons = _load_emoticons(emotions)

    # load model
    fisher_face = cv2.face.FisherFaceRecognizer_create()
    fisher_face.read('emotion_detection_model4.xml')

    # use learnt model
    show_webcam_and_run(fisher_face, emoticons, window_size=(500, 500))
