import tensorflow as tf

from src.dataset.read_dataset import read_dataset
from src.model.run_model import run_model
from src.dataset.utils_dataset import get_next_test_dir

def main():
    # ─── Configurations
    print("GPUs disponíveis: ", tf.config.list_physical_devices('GPU'))

    DEBUG = False
    SIMILARITY = 0.75
    KFOLD = 5
    EPOCHS = 200
    BATCH_SIZE = 32
    EARLY_STOPPING_PATIENCE = 20
    name = "combined"
    imgs_path = 'data/' + name

    # ─── Experiment Settings
    config = {'n_folds': KFOLD, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'early_stopping_patience': EARLY_STOPPING_PATIENCE, 'similarity': SIMILARITY, 'dataset_name': name}

    # ─── Output directory (auto-increment test number)
    output_dir = get_next_test_dir()
    print(f"Output: {output_dir}")

    # ─── Read dataset
    X, y, groups = read_dataset(path=imgs_path, name=name, DEBUG=DEBUG, SIMILARITY=SIMILARITY, output_dir=output_dir)

    # ─── Run experiment
    run_model(X, y, groups, config, output_dir)

if __name__ == "__main__":
    main()
