# import the necessary packages
import os

# initialize the list of class label names
CLASSES = [0.0, 1.0]
# define the minimum learning rate, maximum learning rate, batch size,
# step size, CLR method, and number of epochs
MIN_LR = 1e-4
MAX_LR = 1e-2
STEP_SIZE = 8
CLR_METHOD = "triangular"
# define the path to the output learning rate finder plot, training
# history plot and cyclical learning rate plot
LRFIND_PLOT_PATH = os.path.sep.join(["plots", "lr", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["plots", "lr", "training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["plots", "lr", "clr_plot.png"])

# Read the dataset
model_name = 'distilbert-base-uncased'
# Set up epochs and steps
# Max length of encoded string(including special tokens such as [CLS] and [SEP]):
MAX_SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 14
model_save_path = 'model/' + model_name
data_path = 'matching/transcripts/processed/cross_5.csv'
MOMENTUM = 0.9
