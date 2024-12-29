import os
from keras.models import load_model
import tensorflow as tf
import numpy as np
import streamlit as st

# Set up Streamlit header
st.header('FLOWER CLASSIFICATION MODEL')
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://png.pngtree.com/thumb_back/fw800/background/20231105/pngtree-floral-and-foliage-fusion-on-a-rustic-wooden-background-image_13789995.png'); /* Replace with a valid image URL */
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Class labels and descriptions
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
flower_data = {
    'daisy': 'Daisies are cheerful, bright flowers that come in a variety of colors, most commonly white with a yellow center. They symbolize purity, innocence, and new beginnings. Daisies are often associated with simplicity and fresh beginnings, making them a popular choice in bouquets, gardens, and as a symbol in various cultures. These flowers bloom in the spring and summer, thriving in open, sunny areas. Their delicate appearance and bright, welcoming colors make them a beloved flower in many parts of the world.',
    'dandelion': 'Dandelions are bright yellow flowers known for their resilience and ability to thrive in a variety of environments. They are often seen as weeds, but they have many medicinal properties. The flowers seed head, commonly referred to as a puffball, can be blown to disperse seeds, symbolizing wishes or dreams. Dandelions are also rich in nutrients and have been used in traditional medicine for their detoxifying and healing properties.',
    'rose': ' Roses are classic, fragrant flowers that symbolize love, passion, and romance. Available in many colors, including red, white, pink, and yellow, each color holds a distinct meaning. Roses are commonly used in bouquets, gardens, and as symbols in literature and art. Their elegant petals and strong aroma make them one of the most beloved flowers worldwide.',
    'sunflower': 'Sunflowers are large, vibrant yellow flowers that symbolize adoration, loyalty, and longevity. Known for their tall stature and bright, sunny appearance, they are often associated with warmth and happiness. Sunflowers are unique for their heliotropic behavior, meaning they turn to face the sun throughout the day. They are also cultivated for their seeds, which are rich in nutrients, and for their oil, which has numerous culinary and industrial uses. Sunflowers are often seen as a symbol of positivity and growth.',
    'tulip': 'Tulips are elegant, colorful flowers with cup-shaped blooms, typically seen in vibrant shades like red, yellow, pink, and purple. They are a symbol of perfect love, prosperity, and new beginnings. Tulips bloom in the spring and are often associated with the renewal and beauty of the season. These flowers are widely cultivated in gardens and are a popular flower in bouquets and floral arrangements. The tulip is also a symbol of Dutch culture, where it holds significant historical and cultural value.'
}

# Load the trained model
model = load_model('Flower_Recog_Model.h5')

# Classification function
def classify_images(image_path):
    # Load and preprocess the image
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    # Predict using the model
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    predicted_flower = flower_names[np.argmax(result)]
    confidence_score = max(result) * 100

    # Create the outcome and description
    outcome = (
        f"The image belongs to **{predicted_flower}** "
        f"with a confidence score of **{confidence_score:.2f}%**."
    )
    description = flower_data[predicted_flower]

    return outcome, description

# File uploader
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Save the uploaded file locally
    upload_dir = 'upload'
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)

    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image and classification results
    col1, col2 = st.columns(2)

    with col1:
        st.image(file_path, caption='Uploaded Image', width=150)  # Reduced width of the image

    with col2:
        result, description = classify_images(file_path)
        st.markdown(f"### Prediction Result:\n{result}")
        st.markdown(f"### Flower Description:\n{description}")

