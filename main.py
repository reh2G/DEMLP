from src.read_dataset import read_dataset
from src.prepare_dataset import prepare_dataset
from src.define_model import conv4

import tensorflow as tf

def main():

# ─── Configurations
#
    print("List of available GPUs: ", tf.config.list_physical_devices('GPU'))

    # name = "combined"
    name = "pacs"
    # name = "yildirim"
 
    imgs_path = 'data/' + name

# ─── Looping
#
    # Read dataset and split images
    X_train, y_train, X_test, y_test = read_dataset(path=imgs_path, name=name)
    X_train_kidney, y_train_kidney, X_train_normal, y_train_normal, X_test_kidney, y_test_kidney, X_test_normal, y_test_normal = prepare_dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    model = conv4()

if __name__ == "__main__":
    main()
