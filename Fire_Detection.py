import streamlit as st
import cv2
import numpy as np
import torch
import tempfile
from PIL import Image
from geopy.geocoders import Nominatim
import requests
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Function to load the YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="weights/best.pt", force_reload=True)
    return model

# Function to get the user's location
def get_user_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        city = data.get("city")
        region = data.get("region")
        country = data.get("country")

        # Convert city name to latitude and longitude
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

# Function to get weather data using latitude and longitude
def get_weather(latitude, longitude):
    # Replace 'YOUR_API_KEY' with your actual API key
    api_key = '19b1d383b69e6c595c4e1fb3e5191c96'
    api_url = f'http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric'
    response = requests.get(api_url)
    if response.status_code == 200:
        weather_data = response.json()
        temperature = weather_data['main']['temp']
        weather_description = weather_data['weather'][0]['description']
        return temperature, weather_description
    else:
        return None, None

# Function to send email with attachments
def send_email_with_attachment(sender_email, password, department_emails, subject, body, attachment_path,
                               original_attachment_path, weather_data, location):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    for department, receiver_email in department_emails.items():
        # Create the email message
        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = receiver_email
        message['Subject'] = subject

        # Construct the email body
        location_str = f"Location: {location['city']}, {location['region']}, {location['country']}"
        body_with_location = f"{body}\n\n{location_str}\nWeather: {weather_data[1]}, Temperature: {weather_data[0]} °C"
        message.attach(MIMEText(body_with_location, "plain"))

        # Attach the cropped image
        attachment = open(attachment_path, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename= {attachment_path}")
        message.attach(part)

        # Attach the original image
        original_attachment = open(original_attachment_path, "rb")
        original_part = MIMEBase('application', 'octet-stream')
        original_part.set_payload((original_attachment).read())
        encoders.encode_base64(original_part)
        original_part.add_header('Content-Disposition', f"attachment; filename= {original_attachment_path}")
        message.attach(original_part)

        # Send the email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = message.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()

# Define a video transformer class for real-time video processing
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()

    def transform(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame)
        detections = results.xyxy[0]
        length = sum(1 for d in detections if d[4] >= 0.7)  # Count detections with confidence >= 0.7
        output = np.squeeze(results.render())

        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR), length

# Set the background color
background_color = "#00ff00"  # blue color, you can use any other color code

# Set the page configuration
st.set_page_config(page_title="Fire Detection", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

# Apply the background color using the theme configuration
st.markdown(f"""
    <style>
        body {{
            background-color: {background_color};
        }}
    </style>
""", unsafe_allow_html=True)

# Display title and sidebar
st.title('Fire Detection')
st.sidebar.title('App Mode')
app_mode = st.sidebar.selectbox('Choose the App Mode', ['About App', 'Detect on Image', 'Detect on Video', 'Detect on WebCam'])

if app_mode == 'About App':
    # Display about app information
    st.subheader("About")
    st.markdown('<h5>🔥 WildfireEye: YOLO-Based Forest Fire Detection and Alert System</h5>', unsafe_allow_html=True)
    st.markdown("- <h5>Forest Fire</h5>", unsafe_allow_html=True)
    st.image("Images/Forest-Fire-Protection.jpg")
    st.markdown("- <h5>Detection system on YOLO</h5>", unsafe_allow_html=True)
    st.image("Images/10-Figure11-1.png")
    st.markdown("## Features\n- Detect on Image\n- Detect on Videos\n- Live Detection")

elif app_mode == 'Detect on Image':
    # Code for image detection
    st.subheader("Detect Fire in Image")
    st.write("Upload an image to detect fire.")

    img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if img_file is not None:
        image = np.array(Image.open(img_file))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform fire detection on the uploaded image
        results = load_model()(image)
        detections = results.xyxy[0]
        fire_count = sum(1 for d in detections if d[4] >= 0.7)  # Count detections with confidence >= 0.7

        st.write(f"Number of fires detected: {fire_count}")

elif app_mode == 'Detect on Video':
    # Code for video detection
    st.subheader("Detect Fire in Video")
    st.write("Upload a video to detect fire.")

    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

    if video_file is not None:
        # Load the YOLOv5 model
        model = load_model()

        tffile = tempfile.NamedTemporaryFile(delete=False)
        tffile.write(video_file.read())

        # Display uploaded video
        st.video(tffile.name)

        # Open video file
        vid = cv2.VideoCapture(tffile.name)
        frame_count = 0
        fire_count = 0

        while vid.isOpened():
            ret, frame = vid.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform fire detection on each frame
            results = model(frame)
            detections = results.xyxy[0]
            fire_count += sum(1 for d in detections if d[4] >= 0.7)  # Count detections with confidence >= 0.7
            frame_count += 1

        vid.release()

        st.write(f"Number of frames processed: {frame_count}")
        st.write(f"Number of fires detected: {fire_count}")


elif app_mode == 'Detect on WebCam':
    # Display settings for real-time webcam detection
    st.subheader("Real-time Fire Detection on WebCam")
    st.markdown("Please grant access to your webcam to start real-time fire detection.")

    # Display settings for email notification
    st.sidebar.subheader("Email Notification")
    sender_email = st.sidebar.text_input("Sender Email")
    password = st.sidebar.text_input("Password", type="password")
    department_emails = {
        "Fire Department": st.sidebar.text_input("Fire Department Email"),
        "Forest Department": st.sidebar.text_input("Forest Department Email"),
        "Ambulance Department": st.sidebar.text_input("Ambulance Department Email")
    }

    # Initialize video streamer
    webrtc_ctx = webrtc_streamer(key="fire-detection", video_transformer_factory=VideoTransformer)

    if webrtc_ctx.video_transformer:
        # Run the detection loop
        while True:
            frame, length = webrtc_ctx.video_transformer.recv()
            if frame is None:
                break

            # Display the output frame with detected fire count
            st.image(frame, channels="BGR", use_column_width=True)
            st.markdown(f"<h3 style='text-align: center; color:red;'>Fire Count: {length}</h3>", unsafe_allow_html=True)

            # Send email if fire is detected
            if length > 0:
                location = get_user_location()
                latitude = location['latitude']
                longitude = location['longitude']
                temperature, weather_description = get_weather(latitude, longitude)
                weather_data = (temperature, weather_description)

                detected_object_path = 'detected_object.jpg'
                original_image_path = 'original_image.jpg'
                send_email_with_attachment(sender_email, password, department_emails, "Fire Alert!",
                                           "Fire detected, please respond immediately.", detected_object_path,
                                           original_image_path, weather_data, location)
                st.write("Fire detected! Email alert sent to relevant departments.")
