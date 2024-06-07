import streamlit as st
import torch
from PIL import Image
from geopy.geocoders import Nominatim
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Set the page configuration
st.set_page_config(page_title="Fire Detection", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

def get_location():
    # Use geopy to get coordinates
    geolocator = Nominatim(user_agent="streamlit_app")
    location = geolocator.geocode("Madurai")
    if location:
        city = location.raw.get('display_name', '').split(',')[0]
        latitude = location.latitude
        longitude = location.longitude
        return {'city': city, 'latitude': latitude, 'longitude': longitude}
    else:
        return None

def get_weather(latitude, longitude):
    # Replace 'YOUR_API_KEY' with your actual API key
    api_key = '12a9f83a6662d44f674807a4b7cbb15e'
    api_url = f'http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric'
    response = requests.get(api_url)
    if response.status_code == 200:
        weather_data = response.json()
        temperature = weather_data['main']['temp']
        weather_description = weather_data['weather'][0]['description']
        return temperature, weather_description
    else:
        return None, None

def send_email(sender_email, password, receiver_email, subject, body, image_path):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # Attach the body text
    msg.attach(MIMEText(body, 'plain'))

    # Attach the image
    with open(image_path, 'rb') as f:
        image = MIMEImage(f.read())
    msg.attach(image)

    # Send the email
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())

def load_model():
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="weights/best.pt", force_reload=True)
    return model

def detect_fire(image):
    # Detect fire in the image using the YOLOv5 model
    results = model(image)
    return results

st.title('Fire Detection')
st.sidebar.title('App Mode')

app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['About App', 'Detect on Image'])

if app_mode == 'About App':
    st.subheader("About")
    st.markdown(' <h5>🔥 WildfireEye: YOLO-Based Forest Fire Detection and Alert System</h5>', unsafe_allow_html=True)
    st.markdown("- <h5>Forest Fire</h5>", unsafe_allow_html=True)
    st.image("Images/Forest-Fire-Protection.jpg")
    st.markdown("- <h5>Detection system on YOLO</h5>", unsafe_allow_html=True)
    st.image("Images/10-Figure11-1.png")
    st.markdown("""
                ## Features
- Detect on Image
""")

if app_mode == 'Detect on Image':
    st.subheader("Detected Fire:")
    text = st.markdown("")

    st.sidebar.markdown("---")
    # Input for Image
    img_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if img_file:
        image = Image.open(img_file)
    else:
        st.sidebar.error("Please upload an image.")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.markdown("Original Image")
    st.sidebar.image(image)

    # Load the YOLOv5 model
    model = load_model()

    # predict the image
    results = detect_fire(image)
    length = len(results.xyxy[0])
    output = results.render()
    text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>", unsafe_allow_html=True)
    st.subheader("Output Image")
    st.image(output, use_column_width=True)

    # Get location and weather information
    location = get_location()
    if location:
        latitude = location['latitude']
        longitude = location['longitude']
        temperature, weather_description = get_weather(latitude, longitude)

        # Send email with detection results
        if length > 0:
            sender_email = "f07387005@gmail.com"
            password = "ynfr jeou yqgx gbje"
            receiver_email = "f07387005@gmail.com"
            subject = "Fire Detection Alert!"
            body = f"Fire detected!\n\nWeather: {weather_description}, Temperature: {temperature} °C"
            image_path = "detected_image.jpg"  # Replace with the path to save the detected image
            image.save(image_path)
            send_email(sender_email, password, receiver_email, subject, body, image_path)
