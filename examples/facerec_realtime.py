import tensorflow as tf
import numpy as np
import sys
import os
import detect_and_align
import re
import cv2
import argparse
import time
from scipy import misc
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC


images = {}
training = True
people = []
svm = None
pnet = None
rnet = None
onet = None
unknownImgs = np.load("./examples/web/unknown.npy")

class Face:

    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            str(self.identity),
            self.rep[0:5]
        )

def find_matching_id(id_dataset, embedding):
    threshold = 1.1
    min_dist = 10.0
    matching_id = None

    for id_data in id_dataset:
        dist = get_embedding_distance(id_data.embedding, embedding)

        if dist < threshold and dist < min_dist:
            min_dist = dist
            matching_id = id_data.name
    return matching_id, min_dist

def processFrame(sess, imgdata, identity):
        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)

        buf = np.fliplr(np.asarray(img))
        rgbFrame = np.zeros((300, 400, 3), dtype=np.uint8)
        rgbFrame[:, :, 0] = buf[:, :, 2]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 0]

        if not training:
            annotatedFrame = np.copy(buf)

        # cv2.imshow('frame', rgbFrame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     return

        identities = []

        face_patches, bbs, landmarks = detect_and_align.align_image(rgbFrame, pnet, rnet, onet)
        
        for bb in bbs:
            if len(face_patches) == 0:
                continue

            face_patches = np.stack(face_patches)
            feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
            embs = sess.run(embeddings, feed_dict=feed_dict)

            for i in range(len(embs)):
                bb = bbs[i]
                alignedFace = face_patches[i]
                phash = str(imagehash.phash(Image.fromarray(alignedFace)))
                if phash in images:
                    identity = images[phash].identity
                else:
                    rep = embs[i, :]
                    # print(rep)
                    if training:
                        images[phash] = Face(rep, identity)
                        # TODO: Transferring as a string is suboptimal.
                        # content = [str(x) for x in cv2.resize(alignedFace, (0,0),
                        # fx=0.5, fy=0.5).flatten()]
                        content = [str(x) for x in alignedFace.flatten()]
                    else:
                        if len(people) == 0:
                            identity = -1
                        elif len(people) == 1:
                            identity = 0
                        elif svm:
                            identity = svm.predict(rep)[0]
                        else:
                            print("hhh")
                            identity = -1
                        if identity not in identities:
                            identities.append(identity)

                if not training:
                    bl = (bb.left(), bb.bottom())
                    tr = (bb.right(), bb.top())
                    cv2.rectangle(annotatedFrame, bl, tr, color=(153, 255, 204),
                                  thickness=3)
                    for p in openface.AlignDlib.OUTER_EYES_AND_NOSE:
                        cv2.circle(annotatedFrame, center=landmarks[p], radius=3,
                                   color=(102, 204, 255), thickness=-1)
                    if identity == -1:
                        if len(people) == 1:
                            name = people[0]
                        else:
                            name = "Unknown"
                    else:
                        name = people[identity]
                    cv2.putText(annotatedFrame, name, (bb.left(), bb.top() - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                                color=(152, 255, 204), thickness=2)

        if not training:
            plt.figure()
            plt.imshow(annotatedFrame)
            plt.xticks([])
            plt.yticks([])
            imgdata = StringIO.StringIO()
            plt.savefig(imgdata, format='png')
            plt.close()

def load_model(model):
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def main(args):
    modelDir = os.path.join(fileDir, '..', '..', 'models')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Path to Facenet pre-trained network model.",
                        default=os.path.join('./models/facenet-1/20170511-185253/', '20170511-185253.pb'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)

    args = parser.parse_args()

    with tf.Graph().as_default():
        with tf.Session() as sess:

            pnet, rnet, onet = detect_and_align.create_mtcnn(sess, None)

            load_model(args.model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            cap = cv2.VideoCapture(0)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            show_landmarks = False
            show_bb = False
            show_id = True
            show_fps = False
            while(True):
                start = time.time()
                _, frame = cap.read()

                processFrame(sess, frame, None)
                end = time.time()

                seconds = end - start
                fps = round(1 / seconds, 2)

                if show_fps:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(fps), (0, int(frame_height) - 5), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow('frame', frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    show_landmarks = not show_landmarks
                elif key == ord('b'):
                    show_bb = not show_bb
                elif key == ord('i'):
                    show_id = not show_id
                elif key == ord('f'):
                    show_fps = not show_fps

            cap.release()
            cv2.destroyAllWindows()


