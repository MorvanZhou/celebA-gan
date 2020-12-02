import matplotlib.pyplot as plt
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# data is downloaded from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    values = value if isinstance(value, (list, tuple)) else [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _img_array_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value.ravel()))


def _bytes_img_process(img_str):
    crop = [20, 0, 178, 178]
    imgs = tf.io.decode_and_crop_jpeg(img_str, crop)
    imgs = tf.image.resize(imgs, (128, 128))
    return imgs


class CelebA:
    def __init__(self, batch_size, data_dir="data"):
        self.label_path = os.path.join(data_dir, "list_attr_celeba.txt")
        self.img_dir = os.path.join(data_dir, "img_align_celeba")
        self.tfrecord_dir = os.path.join(data_dir, "tfrecord-celebA-gan")
        self.batch_size = batch_size
        self.label_names = [
            "Smiling",
            "Male",
        ]
        with open(self.label_path) as f:
            lines = f.readlines()
            self.label_names_id = []
            all_labels = lines[1].strip().split(" ")
            for label_name in self.label_names:
                self.label_names_id.append(all_labels.index(label_name))
        self.ds = None
        self.img_process_func = _bytes_img_process

    def _image_example(self, img, labels):
        feature = {
            "labels": _int64_feature(labels),
            "img": _bytes_feature(img),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _parse_img(self, example_proto):
        feature = tf.io.parse_single_example(example_proto, features={
            "labels": tf.io.FixedLenFeature([len(self.label_names)], tf.int64),
            "img": tf.io.FixedLenFeature([], tf.string)
        })
        labels = feature["labels"]
        imgs = self.img_process_func(feature["img"])
        return tf.cast(imgs, tf.float32) / 255 * 2 - 1, tf.cast(labels, tf.int32)

    def load_tf_recoder(self):
        paths = [os.path.join(self.tfrecord_dir, p) for p in os.listdir(self.tfrecord_dir)]
        raw_img_ds = tf.data.TFRecordDataset(paths, num_parallel_reads=min(4, len(paths)))
        self.ds = raw_img_ds.shuffle(2018).map(
            self._parse_img, num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ).batch(
            self.batch_size, drop_remainder=True
        ).prefetch(
            tf.data.experimental.AUTOTUNE
        )

    def to_tf_recoder(self):
        # assert not os.path.exists(path), FileExistsError
        with open(self.label_path) as f:
            lines = f.readlines()[2:]
            n = 202599//4
            chunks = [lines[i:i + n] for i in range(0, len(lines), n)]
            for i, chunk in enumerate(chunks):
                print("process {}/{}...".format(i, len(chunks)))
                path = "{}/{}.tfrecord".format(self.tfrecord_dir, i)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with tf.io.TFRecordWriter(path) as writer:
                    for line in chunk:
                        img_name, img_labels = line.split(" ", 1)
                        try:
                            img = open(os.path.join(self.img_dir, img_name), "rb").read()
                        except Exception as e:
                            break
                        label_str = img_labels.replace("  ", " ").split(" ")
                        labels = [0 if label_str[i] == "-1" else 1 for i in self.label_names_id]
                        assert len(labels) == len(self.label_names), ValueError(labels, img_labels)
                        tf_example = self._image_example(img, labels)
                        writer.write(tf_example.SerializeToString())


def show_sample(data_dir):
    d = load_celebA_tfrecord(25, data_dir)
    images, labels = next(iter(d.ds))
    images = images.numpy()
    labels = labels.numpy()
    for i in range(5):
        for j in range(5):
            n = i*5+j
            plt.subplot(5, 5, n+1)
            plt.imshow((images[n]+1)/2)
            # plt.text(129, 129, "{}".format(labels[n]))
            plt.xticks(())
            plt.xlabel("{}".format(labels[n]))
            plt.yticks(())
    plt.show()


def parse_celebA_tfreord(data_dir):
    d = CelebA(1, data_dir)
    d.to_tf_recoder()


def load_celebA_tfrecord(batch_size, data_dir):
    d = CelebA(batch_size, data_dir)
    d.load_tf_recoder()
    return d


if __name__ == "__main__":
    import time
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default="data", type=str)

    args = parser.parse_args()

    DATA_DIR = args.data_dir

    t0 = time.time()
    parse_celebA_tfreord(DATA_DIR)
    # ds = load_celebA_tfrecord(20, DATA_DIR)
    # t1 = time.time()
    # print("load time", t1-t0)
    # count = 0
    # while True:
    #     for img, label in ds:
    #         # if _ % 200 == 0:
    #         count+=1
    #         if count % 500==0: print(img.shape, label.shape)
    #         if count == 10000:
    #             break
    #     if count == 10000:
    #         break
    #
    # print("runtime", time.time()-t1)
    show_sample(DATA_DIR)
