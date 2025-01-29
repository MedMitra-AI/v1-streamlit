import streamlit as st
import base64
import sqlite3
import boto3
import io
import os
from datetime import datetime
from openai import OpenAI

# -----------------------
# Configuration Section
# -----------------------

# AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "YOUR_AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "YOUR_AWS_SECRET_ACCESS_KEY")
# AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
# S3_BUCKET_NAME = "ecg--images-mgumst"  # Replace with your bucket name


AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_REGION = st.secrets["AWS_REGION"]
S3_BUCKET_NAME = st.secrets["AWS_BUCKET_NAME"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

FIXED_PROMPT = (
    "This is an image of an ECG. Please list along with reason if the image exhibits signs "
    "of any of these following categories - Normal ECG, Myocardial Infarction, ST/T Change, Ventricular fibrillation, Atrial fibrillation, Ventricular Tachycardia "
    "Conduction Disturbance, Hypertrophy. Please state your reason for the output and explain "
    "with sound medical reasoning which a doctor can understand. Please dont explain each class in detail, just give the conclusion with concise reason. give me possible diagnosis and treatment plan accorindgly as well."
    "Be very careful with determining between Ventricular fibrillation and Ventricular Tachycardia, between atrial fibrillation and ventricular fibrillation and between normal ECG and atrial fibrillation"
    "NEVER mention anything along the lines of you not being a medical professional"
)

# -----------------------
# Setup S3 Client
# -----------------------
s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# -----------------------
# Setup SQLite
# -----------------------
conn = sqlite3.connect("feedback.db")
c = conn.cursor()
c.execute(
    """
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        s3_url TEXT,
        model_output TEXT,
        user_response TEXT,
        user_correction TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""
)
conn.commit()

# -----------------------
# Helper Functions
# -----------------------
def encode_image(image_bytes: bytes) -> str:
    """Encodes an image file (as bytes) to base64."""
    return base64.b64encode(image_bytes).decode("utf-8")

def upload_to_s3(file_bytes_io: io.BytesIO, filename: str) -> str:
    """
    Uploads a BytesIO object to S3 and returns the S3 URL.
    Adjust ACL or bucket policy if you need a public link.
    """
    # Reset pointer just to be safe
    file_bytes_io.seek(0)

    # Generate a unique key with date/time
    unique_key = f"ecg_images/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
    
    # Upload file to S3
    s3_client.upload_fileobj(
        file_bytes_io,
        S3_BUCKET_NAME,
        unique_key
    )
    # Construct the S3 URL (assuming standard region and no custom domain)
    s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{unique_key}"
    return s3_url

# -----------------------
# Streamlit App
# -----------------------
st.title("ECG Analysis and Classifier")

uploaded_file = st.file_uploader("Upload an ECG Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read the file into memory
    file_bytes = uploaded_file.read()
    # Create a BytesIO object from file_bytes
    file_bytes_io = io.BytesIO(file_bytes)

    # 1. Upload the file to S3
    with st.spinner("Uploading image to S3..."):
        s3_url = upload_to_s3(file_bytes_io, uploaded_file.name)
    st.success("Image successfully uploaded to S3!")

    # 2. Show the image in Streamlit from the in-memory bytes
    st.image(file_bytes, caption="Uploaded ECG Image")

    # 3. Analyze with OpenAI
    st.markdown("### Analyzing ECG...")
    with st.spinner("Analyzing the ECG image and generating a response..."):
        try:
            # Re-init an OpenAI client
            client = OpenAI(api_key=OPENAI_API_KEY)

            # Prepare the base64 encoding of the image
            base64_image = encode_image(file_bytes)

            response = client.chat.completions.create(
                model="o1",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": FIXED_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            },
                        ],
                    }
                ],
            )

            generated_response = response.choices[0].message.content
            st.success("Response generated!")
            st.write(generated_response)

        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # 4. Collect Feedback
    st.markdown("### Was this output helpful?")
    feedback = st.radio(
        "Please select:",
        options=["👍 Yes, it looks good!", "👎 No, needs improvement"],
        index=0
    )

    user_correction = ""
    if feedback == "👎 No, needs improvement":
        user_correction = st.text_area("Please provide the correct or improved response:")

    if st.button("Submit Feedback"):
        # Insert feedback record into SQLite
        c.execute(
            """
            INSERT INTO feedback (s3_url, model_output, user_response, user_correction)
            VALUES (?, ?, ?, ?)
            """,
            (s3_url, generated_response, feedback, user_correction)
        )
        conn.commit()
        st.success("Your feedback has been recorded. Thank you!")
