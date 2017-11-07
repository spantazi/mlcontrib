from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC

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


