import cv2
import numpy as np
import torch
from PIL import Image
from geopy.geocoders import Nominatim
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import streamlit as st
import os

# Load the YOLOv5 model
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="weights/best.pt", force_reload=True)
    return model

model = load_model()

def get_user_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        city = data.get("city")
        region = data.get("region")
        country = data.get("country")

        geolocator = Nominatim(user_agent="streamlit_app")
        location = geolocator.geocode(city)
        if location:
            latitude = location.latitude
            longitude = location.longitude
        else:
            latitude = None
            longitude = None

        return {'city': city, 'region': region, 'country': country, 'latitude': latitude, 'longitude': longitude}
    except Exception as e:
        print("Error fetching location:", e)
        return None

def get_weather(latitude, longitude):
    api_key = '12a9f83a6662d44f674807a4b7cbb15e'  # Replace with your OpenWeather API key
    api_url = f'http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric'
    response = requests.get(api_url)
    if response.status_code == 200:
        weather_data = response.json()
        temperature = weather_data['main']['temp']
        weather_description = weather_data['weather'][0]['description']
        return temperature, weather_description
    else:
        return None, None

def send_email_with_attachment(sender_email, password, department_emails, subject, body, attachment_path,
                               original_attachment_path, weather_data, location):
    for department, receiver_email in department_emails.items():
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = receiver_email
        message['Subject'] = subject

        location_str = f"Location: {location['city']}, {location['region']}, {location['country']}"
        body_with_location = f"{body}\n\n{location_str}\nWeather: {weather_data[1]}, Temperature: {weather_data[0]} °C"
        message.attach(MIMEText(body_with_location, "plain"))

        attachment = open(attachment_path, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename= {attachment_path}")
        message.attach(part)

        original_attachment = open(original_attachment_path, "rb")
        original_part = MIMEBase('application', 'octet-stream')
        original_part.set_payload((original_attachment).read())
        encoders.encode_base64(original_part)
        original_part.add_header('Content-Disposition', f"attachment; filename= {original_attachment_path}")
        message.attach(original_part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()

st.title('Fire Detection')

st.sidebar.subheader("Detection Options")
option = st.sidebar.selectbox("Choose detection mode", ["Image", "Video", "Webcam"])

if option == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        results = model(image)
        detections = results.xyxy[0]
        length = len(detections)
        st.write(f"Number of fire detections: {length}")
        st.image(np.squeeze(results.render()), caption='Processed Image', use_column_width=True)

elif option == "Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if uploaded_file:
        video_bytes = uploaded_file.read()
        st.video(video_bytes)

        # Save the video temporarily
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        vid = cv2.VideoCapture(video_path)
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame)
            output = np.squeeze(results.render())
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            st.image(output, channels="BGR")

elif option == "Webcam":
    st.subheader("Webcam Fire Detection")
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.error("Failed to capture webcam.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)
        output = np.squeeze(results.render())
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        st.image(output, channels="BGR", caption='Webcam Feed')

        # Detect fire and send email
        if st.button("Detect Fire"):
            # Perform fire detection logic here
            # Example: if fire detected, send email
            st.write("Fire detected! Sending email...")

            # Get user location and weather data
            location = get_user_location()
            if location and location['latitude'] and location['longitude']:
                                temperature, weather_description = get_weather(location['latitude'], location['longitude'])
                if temperature is not None and weather_description is not None:
                    # Send email with detected image and weather information
                    sender_email = "your_email@gmail.com"
                    password = "your_email_password"
                    department_emails = {"Fire Department": "fire_department@example.com"}
                    subject = "Fire Detected!"
                    body = "Fire has been detected. Please take necessary actions."
                    attachment_path = "detected_image.jpg"  # Path to the detected image
                    original_attachment_path = "original_image.jpg"  # Path to the original image

                    send_email_with_attachment(sender_email, password, department_emails, subject, body,
                                               attachment_path, original_attachment_path,
                                               (temperature, weather_description), location)

            else:
                st.error("Failed to fetch location or weather data.")

video_capture.release()

