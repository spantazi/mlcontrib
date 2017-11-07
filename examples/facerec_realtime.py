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

def processFrame(self, imgdata, identity):
        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)

        buf = np.fliplr(np.asarray(img))
        rgbFrame = np.zeros((300, 400, 3), dtype=np.uint8)
        rgbFrame[:, :, 0] = buf[:, :, 2]
        rgbFrame[:, :, 1] = buf[:, :, 1]
        rgbFrame[:, :, 2] = buf[:, :, 0]

        if not self.training:
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
                if phash in self.images:
                    identity = self.images[phash].identity
                else:
                    rep = embs[i, :]
                    # print(rep)
                    if self.training:
                        self.images[phash] = Face(rep, identity)
                        # TODO: Transferring as a string is suboptimal.
                        # content = [str(x) for x in cv2.resize(alignedFace, (0,0),
                        # fx=0.5, fy=0.5).flatten()]
                        content = [str(x) for x in alignedFace.flatten()]
                    else:
                        if len(self.people) == 0:
                            identity = -1
                        elif len(self.people) == 1:
                            identity = 0
                        elif self.svm:
                            identity = self.svm.predict(rep)[0]
                        else:
                            print("hhh")
                            identity = -1
                        if identity not in identities:
                            identities.append(identity)

                if not self.training:
                    bl = (bb.left(), bb.bottom())
                    tr = (bb.right(), bb.top())
                    cv2.rectangle(annotatedFrame, bl, tr, color=(153, 255, 204),
                                  thickness=3)
                    for p in openface.AlignDlib.OUTER_EYES_AND_NOSE:
                        cv2.circle(annotatedFrame, center=landmarks[p], radius=3,
                                   color=(102, 204, 255), thickness=-1)
                    if identity == -1:
                        if len(self.people) == 1:
                            name = self.people[0]
                        else:
                            name = "Unknown"
                    else:
                        name = self.people[identity]
                    cv2.putText(annotatedFrame, name, (bb.left(), bb.top() - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                                color=(152, 255, 204), thickness=2)

        if not self.training:
            plt.figure()
            plt.imshow(annotatedFrame)
            plt.xticks([])
            plt.yticks([])
            imgdata = StringIO.StringIO()
            plt.savefig(imgdata, format='png')
            plt.close()

def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:

            pnet, rnet, onet = detect_and_align.create_mtcnn(sess, None)

            load_model(args.model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            id_dataset = id_data.get_id_data(args.id_folder[0], pnet, rnet, onet, sess, embeddings, images_placeholder, phase_train_placeholder)
            print_id_dataset_table(id_dataset)

            test_run(pnet, rnet, onet, sess, images_placeholder, phase_train_placeholder, embeddings, id_dataset, args.test_folder)

            cap = cv2.VideoCapture(0)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            show_landmarks = False
            show_bb = False
            show_id = True
            show_fps = False
            while(True):
                start = time.time()
                _, frame = cap.read()

                face_patches, padded_bounding_boxes, landmarks = detect_and_align.align_image(frame, pnet, rnet, onet)

                if len(face_patches) > 0:
                    face_patches = np.stack(face_patches)
                    feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
                    embs = sess.run(embeddings, feed_dict=feed_dict)

                    print('Matches in frame:')
                    for i in range(len(embs)):
                        bb = padded_bounding_boxes[i]

                        matching_id, dist = find_matching_id(id_dataset, embs[i, :])
                        if matching_id:
                            print('Hi %s! Distance: %1.4f' % (matching_id, dist))
                        else:
                            matching_id = 'Unkown'
                            print('Unkown! Couldn\'t fint match.')

                        if show_id:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame, matching_id, (bb[0], bb[3]), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                        if show_bb:
                            cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)

                        if show_landmarks:
                            for j in range(5):
                                size = 1
                                top_left = (int(landmarks[i, j]) - size, int(landmarks[i, j + 5]) - size)
                                bottom_right = (int(landmarks[i, j]) + size, int(landmarks[i, j + 5]) + size)
                                cv2.rectangle(frame, top_left, bottom_right, (255, 0, 255), 2)
                else:
                    print('Couldn\'t find a face')

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


