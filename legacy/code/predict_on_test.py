import os
import json
import tensorflow as tf
from sklearn.utils import shuffle
from tf_utils import get_parser
from train_alexnet import img_generator, label_generator, get_compiled_model

def get_test(json_path):
    with open(json_path) as f:
        data = json.load(f)
        x_test  = [
                [os.path.join(os.path.dirname(json_path), c[str(i)]) for i in range(7)]
                 for c in data.get('test')
                 ]
        y_test = [p['label'] for p in data.get('test')]
    return x_test, y_test

def dataset_from_json(json_path, args):
    height, width = args.input_shape

    x_test, y_test = get_test(json_path)

    print(f"Length of test in original data:", len(x_test))
    print(f"Class imbalance in original data(test):", y_test.count(0)/y_test.count(1))

    x_test, y_test = shuffle(x_test, y_test, random_state=42)

    images = tf.data.Dataset.from_generator(generator = lambda: img_generator(x_test, args),
                                            output_types=tf.float32,
                                            output_shapes=[7, height, width])
    labels = tf.data.Dataset.from_generator(generator = lambda: label_generator(y_test),
                                            output_types = tf.int32,
                                            output_shapes = ())

    test_ds = tf.data.Dataset.zip((images, labels))

    return test_ds


if __name__=="__main__":
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    parser = get_parser()
    parser.add_argument('-json-path', '--json-path')
    parser.add_argument('-batch-size', '--batch-size', type=int, default=64)
    parser.add_argument('--stats-file')
    parser.add_argument('--modelpath')

    args = parser.parse_args()
    print(vars(args))

    test_ds = dataset_from_json(args.json_path, args=args)
    test_ds = test_ds.batch(args.batch_size)

    print("Evaluating pre-trained model")
    model = get_compiled_model(args)
    model.load_weights(args.modelpath)
    result = model.evaluate(test_ds)
    print(dict(zip(model.metrics_names, result)))

