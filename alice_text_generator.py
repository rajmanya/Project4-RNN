import os
import numpy as np
import re
import shutil
import tensorflow as tf
import time

DATA_DIR = "./"
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
LOG_DIR = os.path.join(DATA_DIR, "logs")


def clean_logs():
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    shutil.rmtree(LOG_DIR, ignore_errors=True)

# Add this new function to create directories
def create_directories():
    """Create necessary directories for model checkpoints and logs"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

""" def download_and_read(urls):
    texts = []
    for i, url in enumerate(urls):
        p = tf.keras.utils.get_file("ex1-{:d}.txt".format(i), url,
            cache_dir=".")
        text = open(p, mode="r", encoding="utf-8").read()
        # remove byte order mark
        text = text.replace("\ufeff", "")
        # remove newlines
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', " ", text)
        # Split the text into words - key change for word-level model
        words = text.split()
        # add it to the list - now adding words instead of characters
        texts.extend(words)
    return texts """

def split_train_labels(sequence):
    input_seq = sequence[0:-1]
    output_seq = sequence[1:]
    return input_seq, output_seq


class CharGenModel(tf.keras.Model):

    def __init__(self, vocab_size, num_timesteps, 
            embedding_dim, **kwargs):
        super(CharGenModel, self).__init__(**kwargs)
        self.embedding_layer = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim
        )
        self.rnn_layer = tf.keras.layers.GRU(
            num_timesteps,
            recurrent_initializer="glorot_uniform",
            recurrent_activation="sigmoid",
            stateful=True,
            return_sequences=True
        )
        self.dense_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x = self.embedding_layer(x)
        x = self.rnn_layer(x)
        x = self.dense_layer(x)
        return x

    def reset_states(self):
        self.rnn_layer.reset_states()
def loss(labels, predictions):
    return tf.losses.sparse_categorical_crossentropy(
        labels,
        predictions,
        from_logits=True
    )

def experiment_with_temperatures(model, vocab_size, seq_length, embedding_dim, char2idx, idx2char, checkpoint_file):
    """Experiment with different temperature values for text generation"""
    print("\n==== EXPERIMENT LEAD: TEMPERATURE VARIATION EXPERIMENTS ====")
    
    # Define temperature values to test
    temperatures = [0.5, 1.0, 1.5]
    
    # Create a generation model
    gen_model = CharGenModel(vocab_size, seq_length, embedding_dim)
    gen_model.build(input_shape=(1, seq_length))
    
    # Load weights from the provided checkpoint
    gen_model.load_weights(checkpoint_file)
    print(f"Loaded weights from {checkpoint_file}")
    
    # Starting word for generation
    start_word = "Alice"
    if start_word not in char2idx:
        start_word = list(char2idx.keys())[0]  # Fallback to first word in vocabulary
    
    # Generate text with different temperatures
    print("\nGenerating text samples with different temperatures:")
    
    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        generated_text = generate_text(
            gen_model, 
            start_word, 
            char2idx, 
            idx2char,
            num_words_to_generate=50,  # Shorter for easier comparison
            temperature=temp
        )
        
        print(generated_text)
    
    # Analysis and observations
    print("\n==== TEMPERATURE EXPERIMENT ANALYSIS ====")
    print("Temperature 0.5 (Low): More deterministic, predictable text generation.")
    print("Temperature 1.0 (Medium): Balanced between predictability and creativity.")
    print("Temperature 1.5 (High): More diverse and unpredictable output, potentially less coherent.")
    
    return temperatures

# This function should be called after some training but before the end of your script
# For example, after the first 10 epochs of training

def generate_text(model, prefix_word, char2idx, idx2char,
        num_words_to_generate=100, temperature=1.0):
    # For word-level model, prefix should be a single word
    if prefix_word in char2idx:
        input = [char2idx[prefix_word]]
        input = tf.expand_dims(input, 0)
        text_generated = []
        model.reset_states()
        for i in range(num_words_to_generate):
            preds = model(input)
            preds = tf.squeeze(preds, 0) / temperature
            # predict word returned by model
            pred_id = tf.random.categorical(preds, num_samples=1)[-1, 0].numpy()
            text_generated.append(idx2char[pred_id])
            # pass the prediction as the next input to the model
            input = tf.expand_dims([pred_id], 0)
            
        return prefix_word + " " + " ".join(text_generated)
    else:
        return f"Error: '{prefix_word}' not in vocabulary. Try a different starting word."

def download_and_read(file_paths):
    texts = []
    for i, file_path in enumerate(file_paths):
        # tf.keras.utils.get_file() is commented out and replaced with direct file reading
        # p = tf.keras.utils.get_file("ex1-{:d}.txt".format(i), url, cache_dir=".")
        # Instead, directly read from the local file path
        text = open(file_path, mode="r", encoding="utf-8").read()
        # rest of preprocessing remains the same
        text = text.replace("\ufeff", "")
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', " ", text)
        words = text.split()  # Using word-level from the previous modification
        texts.extend(words)
    return texts


# Replace with local file paths:
texts = download_and_read([
    "./pg28885.txt",  # Local path to the first file
    "./12-0.txt"      # Local path to the second file
])

""" # download and read into local data structure (list of chars)
texts = download_and_read([
    "http://www.gutenberg.org/cache/epub/28885/pg28885.txt",
    "https://www.gutenberg.org/files/12/12-0.txt"
]) """
clean_logs()
create_directories() 

# create the vocabulary
vocab = sorted(set(texts))
print("vocab size: {:d}".format(len(vocab)))

# create mapping from vocab chars to ints
char2idx = {c:i for i, c in enumerate(vocab)}
idx2char = {i:c for c, i in char2idx.items()}

# define network
vocab_size = len(vocab)
embedding_dim = 256

# numericize the texts
texts_as_ints = np.array([char2idx[c] for c in texts])
data = tf.data.Dataset.from_tensor_slices(texts_as_ints)

# number of characters to show before asking for prediction
# sequences: [None, 100]
seq_length = 100
sequences = data.batch(seq_length + 1, drop_remainder=True)
sequences = sequences.map(split_train_labels)

# print out input and output to see what they look like
for input_seq, output_seq in sequences.take(1):
    print("input:[{:s}]".format(
        "".join([idx2char[i] for i in input_seq.numpy()])))
    print("output:[{:s}]".format(
        "".join([idx2char[i] for i in output_seq.numpy()])))

def experiment_with_parameters(seq_len, batch_sz):
    """Experiment with different sequence lengths and batch sizes"""
    print(f"\n--- EXPERIMENT: seq_length={seq_len}, batch_size={batch_sz} ---")
    
    # Create sequences with new sequence length
    exp_sequences = data.batch(seq_len + 1, drop_remainder=True)
    exp_sequences = exp_sequences.map(split_train_labels)
    
    # Calculate new steps per epoch
    exp_steps_per_epoch = len(texts) // seq_len // batch_sz
    print(f"Steps per epoch: {exp_steps_per_epoch}")
    
    # Create dataset with new batch size
    exp_dataset = exp_sequences.shuffle(10000).batch(batch_sz, drop_remainder=True)
    
    # Build model
    exp_model = CharGenModel(vocab_size, seq_len, embedding_dim)
    exp_model.build(input_shape=(batch_sz, seq_len))
    exp_model.compile(optimizer=tf.optimizers.Adam(), loss=loss)
    
    # Measure training time for one epoch
    start_time = time.time()
    exp_history = exp_model.fit(
        exp_dataset.repeat(),
        epochs=1,
        steps_per_epoch=exp_steps_per_epoch
    )
    end_time = time.time()
    
    training_time = end_time - start_time
    final_loss = exp_history.history['loss'][0]
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Final loss: {final_loss:.4f}")
    
    return training_time, final_loss

# Experiment with different sequence lengths and batch sizes
print("\n==== EXPERIMENT LEAD: SEQUENCE LENGTH AND BATCH SIZE EXPERIMENTS ====")

# Smaller parameters
small_seq_length = 50
small_batch_size = 32
small_time, small_loss = experiment_with_parameters(small_seq_length, small_batch_size)

# Larger parameters
large_seq_length = 150
large_batch_size = 128
large_time, large_loss = experiment_with_parameters(large_seq_length, large_batch_size)

# Results comparison
print("\n==== EXPERIMENT RESULTS ====")
print(f"Small parameters (seq_length={small_seq_length}, batch_size={small_batch_size}):")
print(f"  - Training time: {small_time:.2f} seconds")
print(f"  - Final loss: {small_loss:.4f}")
print(f"Large parameters (seq_length={large_seq_length}, batch_size={large_batch_size}):")
print(f"  - Training time: {large_time:.2f} seconds")
print(f"  - Final loss: {large_loss:.4f}")
print(f"Time ratio (large/small): {large_time/small_time:.2f}x")
print("\n==== CONTINUING WITH DEFAULT PARAMETERS ====")

# Reset to original value
seq_length = 100  # Reset to original value
batch_size = 64   # Reset to original value

# set up for training
# batches: [None, 64, 100]
batch_size = 64
steps_per_epoch = len(texts) // seq_length // batch_size
dataset = sequences.shuffle(10000).batch(batch_size, drop_remainder=True)
print(dataset)


model = CharGenModel(vocab_size, seq_length, embedding_dim)
model.build(input_shape=(batch_size, seq_length))
model.summary()

# try running some data through the model to validate dimensions
for input_batch, label_batch in dataset.take(1):
    pred_batch = model(input_batch)

print(pred_batch.shape)
assert(pred_batch.shape[0] == batch_size)
assert(pred_batch.shape[1] == seq_length)
assert(pred_batch.shape[2] == vocab_size)

model.compile(optimizer=tf.optimizers.Adam(), loss=loss)

# we will train our model for 50 epochs, and after every 10 epochs
# we want to see how well it will generate text
num_epochs = 50
for i in range(num_epochs // 10):
    model.fit(
        dataset.repeat(),
        epochs=10,
        steps_per_epoch=steps_per_epoch
        # callbacks=[checkpoint_callback, tensorboard_callback]
    )
    checkpoint_file = os.path.join(
        CHECKPOINT_DIR, "model_epoch_{:d}.weights.h5".format(i+1))
    model.save_weights(checkpoint_file)

    # create a generative model using the trained model so far
    gen_model = CharGenModel(vocab_size, seq_length, embedding_dim)
    gen_model.build(input_shape=(1, seq_length))
    gen_model.load_weights(checkpoint_file)

    print(f"after epoch: {(i+1)*10}")
    print(generate_text(gen_model, "Alice", char2idx, idx2char))
    print("---")
    # Add the temperature experiment after the first training cycle (i=0)
    if i == 0:
        # Run temperature experiments
        experiment_with_temperatures(model, vocab_size, seq_length, embedding_dim, 
                                char2idx, idx2char, checkpoint_file)
