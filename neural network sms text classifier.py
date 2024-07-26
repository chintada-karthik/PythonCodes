import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample SMS data
data = {
    'text': [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "U dun say so early hor... U c already then say...",
        "Nah I don't think he goes to usf, he lives around here though",
        "FreeMsg Hey there darling it's been 3 week's now and no word back!",
        "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!",
        "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k?",
        "I've been searching for the right words to thank you for this breather. I promise i won't take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times.",
        "I HAVE A DATE ON SUNDAY WITH WILL!!",
        "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>>",
        "Oh k...i'm watching here:)"],
    'label': ['spam', 'ham', 'ham', 'spam', 'spam', 'ham', 'ham', 'ham', 'spam', 'ham']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Tokenize and pad the sequences
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequences, maxlen=50, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=50, padding='post', truncating='post')

# Build the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=16, input_length=50),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_padded, y_train, epochs=10, batch_size=16, validation_data=(X_test_padded, y_test), verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=1)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Function to predict SMS type
def predict_sms(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)[0][0]
    return 'spam' if prediction > 0.5 else 'ham'

# Example usage
sms = "Congratulations! You've won a free ticket to Bahamas. Reply YES to claim."
prediction = predict_sms(sms)
print(f"The SMS is classified as: {prediction}")