import os
import io
import re
import uuid
import base64
import logging
import tempfile

import pdfplumber
import requests
import streamlit as st
from PIL import Image
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import boto3
from openai import OpenAI

# -------------------------------------------------- #
#              OPENAI CLIENT CONFIG                  #
# -------------------------------------------------- #
client = OpenAI(api_key="sk-proj-aA4in0l2WCEkJXq4yeHAT3BlbkFJmwOhRnH8ypgJpolet2Nb")  # replace with your actual key

# -------------------------------------------------- #
#                   LOGGING SETUP                    #
# -------------------------------------------------- #
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - [%(levelname)s] - %(filename)s.%(funcName)s(%(lineno)d) - %(message)s',
    handlers=[
        logging.FileHandler("medmitra.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------- #
#              LOAD ENVIRONMENT VARIABLES            #
# -------------------------------------------------- #
load_dotenv()

# -------------------- ENV VARIABLES -------------------- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-aA4in0l2WCEkJXq4yeHAT3BlbkFJmwOhRnH8ypgJpolet2Nb")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set.")

DATABASE_URI = os.getenv("DATABASE_URI")  # e.g. "postgresql://user:pass@host:port/dbname"
if not DATABASE_URI:
    logger.warning("DATABASE_URI is not set.")
engine = create_engine(DATABASE_URI, echo=False)
logger.info("Database engine created.")

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
bucket_name = os.getenv("AWS_BUCKET_NAME")

s3_client = boto3.client(
    's3',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
logger.info("S3 client initialized.")

# -------------------- DEPARTMENT CONTEXTS -------------------- #
contexts = {
    "Cardiology": "Common conditions in cardiology include chest pain, palpitations, shortness of breath, hypertension, syncope.",
    "Pulmonology": "Common conditions in pulmonology include chronic cough, shortness of breath, wheezing, hemoptysis, chest pain.",
    "Gastroenterology": "Common conditions in gastroenterology include abdominal pain, diarrhea, constipation, nausea and vomiting, GERD.",
    "Neurology": "Common conditions in neurology include headache, dizziness, seizures, weakness or numbness, memory loss.",
    "Rheumatology": "Common conditions in rheumatology include joint pain, swelling, stiffness, muscle pain, fatigue.",
    "Dermatology": "Common conditions in dermatology include rashes, itching, skin lesions, hair loss, nail changes.",
    "Nephrology": "Common conditions in nephrology include edema, hypertension, hematuria, proteinuria, electrolyte imbalances.",
    "Hematology": "Common conditions in hematology include anemia, bleeding disorders, leukemia, lymphoma, thrombocytopenia.",
    "Infectious Diseases": "Common conditions in infectious diseases include fever, unexplained weight loss, lymphadenopathy, chronic cough, recurrent infections.",
    "Psychiatry": "Common conditions in psychiatry include depression, anxiety, bipolar disorder, schizophrenia, substance use disorders.",
    "Pediatrics": "Common conditions in pediatrics include growth delays, developmental disorders, frequent infections, behavioral issues, pediatric asthma.",
    "Orthopedics": "Common conditions in orthopedics include fractures, joint pain, back pain, sports injuries, osteoarthritis.",
    "Ophthalmology": "Common conditions in ophthalmology include vision loss, eye pain, red eyes, blurred vision, floaters and flashes.",
    "Otolaryngology": "Common conditions in otolaryngology include hearing loss, tinnitus, sinusitis, sore throat, vertigo.",
    "Gynecology": "Common conditions in gynecology include irregular menstruation, pelvic pain, vaginal discharge, infertility, menopause symptoms.",
    "Urology": "Common conditions in urology include urinary incontinence, erectile dysfunction, hematuria, prostate issues, kidney stones.",
    "Oncology": "Common conditions in oncology include unexplained weight loss, persistent fatigue, lump or mass, changes in skin, persistent pain.",
    "General Medicine": "Common conditions in general medicine include fever, unexplained pain, fatigue, weight changes, cough.",
    "Endocrinology": "Common conditions in endocrinology include diabetes, thyroid disorders, adrenal disorders, osteoporosis, hormone imbalances."
}

# -------------------- HELPER / UTILITY -------------------- #

def get_content_type(filename):
    """
    Return a 'Content-Type' string based on file extension.
    We'll handle pdf vs. png vs. jpg for simplicity.
    """
    fn_lower = filename.lower()
    if fn_lower.endswith(".pdf"):
        return "application/pdf"
    elif fn_lower.endswith(".png"):
        return "image/png"
    elif fn_lower.endswith(".jpg") or fn_lower.endswith(".jpeg"):
        return "image/jpeg"
    else:
        return "application/octet-stream"

def upload_to_s3(file_bytes, filename):
    """
    Upload file to S3 with the correct ContentType.
    """
    logger.debug("Uploading file to S3...")

    unique_filename = f"{uuid.uuid4()}_{filename}"
    content_type = get_content_type(filename)

    logger.debug(f"File {filename} -> S3 Key: {unique_filename}, Content-Type: {content_type}")
    logger.debug(f"Upload size: {len(file_bytes)} bytes")

    s3_client.put_object(
        Bucket=bucket_name,
        Key=unique_filename,
        Body=file_bytes,
        ContentType=content_type,
        ACL='private'
    )
    file_url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{unique_filename}"
    logger.info(f"File uploaded to S3 with key {unique_filename}")
    return file_url

def generate_presigned_url(s3_url, expiration=3600):
    """
    Generate a presigned URL to a private object in S3.
    """
    if not s3_url:
        return None
    try:
        object_key = s3_url.split('/')[-1]
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_key},
            ExpiresIn=expiration
        )
        return presigned_url
    except Exception as e:
        logger.error(f"Error generating presigned URL: {e}")
        return None

def downsample_image(file_bytes, max_size=(500, 500), quality=50):
    """
    Takes raw bytes of an uploaded image, compresses it to a smaller JPEG.
    """
    buf = io.BytesIO(file_bytes)
    img = Image.open(buf)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail(max_size)
    out_bytes = io.BytesIO()
    img.save(out_bytes, format="JPEG", quality=quality)
    out_bytes.seek(0)
    return out_bytes.read()

def encode_image(file_obj):
    """
    Read a file-like object fully, encode it in base64, then reset pointer.
    """
    file_data = file_obj.read()
    encoded = base64.b64encode(file_data).decode("utf-8")
    file_obj.seek(0)
    return encoded

def extract_text_from_pdf(pdf_file):
    """
    Given a file-like object, extract textual content using pdfplumber.
    """
    with pdfplumber.open(pdf_file) as pdf:
        all_text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"
    return all_text

# ---------- Example placeholders for analyzing image/prescriptions/lab text ---------- #
FIXED_PROMPT_IMAGE = (
    "You are a helpful medical AI. Examine the following medical image. "
    "Describe any possible findings, anomalies, or relevant clinical interpretations. "
    "Avoid disclaimers about 'consult a specialist'."
)

def analyze_medical_image(image_data_b64):
    """
    Analyze a base64-encoded image using GPT-4-turbo.
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": FIXED_PROMPT_IMAGE},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data_b64}"}
                    },
                ],
            }
        ]
    )
    return response.choices[0].message.content.strip()

def analyze_lab_report_text(lab_report_text):
    """
    Analyze lab report text using GPT-4o model.
    """
    messages = [
        {"role": "system", "content": "You are an expert doctor."},
        {
            "role": "user",
            "content": (
                "Based on the following patient's data extracted from their medical report, provide:\n"
                "1. Diagnosis\n2. Prognosis\n3. Treatment recommendations\n\n"
                f"Lab text:\n{lab_report_text}"
            ),
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=2048,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def analyze_prescription_text_or_image(prescription_content, is_pdf=False):
    """
    If PDF text is provided, we use GPT-4o. 
    If image (base64) is provided, we use GPT-4-turbo.
    """
    if is_pdf:
        # treat prescription_content as text
        messages = [
            {"role": "system", "content": "You are a medical assistant analyzing a prescription PDF text."},
            {
                "role": "user",
                "content": (
                    "Below is the text from a prescription. Summarize any relevant medications, dosages, or instructions:\n\n"
                    f"{prescription_content}"
                )
            }
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1500,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    else:
        # treat prescription_content as base64 image
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical assistant. Analyze the following image (prescription). "
                        "Extract relevant medications, dosages, or instructions. Avoid disclaimers."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{prescription_content}"}
                        },
                    ]
                }
            ]
        )
        return response.choices[0].message.content.strip()

def get_medical_advice(
    department,
    chief_complaint,
    history_presenting_illness,
    past_history,
    personal_history,
    family_history,
    obg_history="",
    image_analysis_text="",
    lab_analysis_text="",
    prescription_analysis_text=""
):
    """
    Combine all analyses + patient's textual histories into one final GPT call.
    Now updated to include drug dosages and contra-indicative drug warnings.
    """
    context = contexts.get(department, "")

    prompt_text = f"""
Department: {department}
Context: {context}

Chief Complaint: {chief_complaint}
History of Presenting Illness: {history_presenting_illness}
Past History: {past_history}
Personal History: {personal_history}
Family History: {family_history}
OBG History: {obg_history}

Medical Image Analysis (if any): {image_analysis_text}
Lab Report Analysis (if any): {lab_analysis_text}
Prescription Analysis (if any): {prescription_analysis_text}

Return your response in bullet-point style with these headings:

**Results**

**Most Likely Diagnosis**
- ...
**Other Possible Diagnoses**
- ...
**Suggested Tests**
- ...
**Prognosis**
- ...
**Suggested Treatment Plan**
- Include recommended drug dosages and provide warnings about any potential contra-indicative drugs.

**Case Summary**
(Short concluding summary)

Keep the tone clinically relevant. Avoid disclaimers about needing further specialist follow-ups.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant specialized in medical diagnosis, prognosis, and treatment planning. "
                "No disclaimers, please."
            )
        },
        {"role": "user", "content": prompt_text}
    ]

    try:
        response = client.chat.completions.create(model="gpt-4o", messages=messages)
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating medical advice: {e}")
        return "Error generating medical advice. Please try again later."

# -------------------- TEXT PARSING FUNCTIONS -------------------- #
def parse_section(full_text, section_name):
    pattern = rf"\*\*{section_name}\*\*\s*\n?(.*?)(?=\n\*\*|$)"
    match = re.search(pattern, full_text, flags=re.IGNORECASE|re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def remove_section(full_text, section_name):
    pattern = rf"\*\*{section_name}\*\*\s*\n?(.*?)(?=\n\*\*|$)"
    new_text = re.sub(pattern, "", full_text, flags=re.IGNORECASE|re.DOTALL)
    return new_text.strip()

def extract_bullet_items(section_text):
    """
    Return a list of lines that start with '- '
    """
    items = []
    for line in section_text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            item = line[2:].strip()
            if item:
                items.append(item)
    return items

# -------------------- DATABASE FUNCTIONS -------------------- #
def insert_patient_info(
    name,
    age,
    gender,
    contact_number,
    department,
    chief_complaint,
    history_presenting_illness,
    past_history,
    personal_history,
    family_history,
    obg_history,
    lab_report_url,
    medical_imaging_url,
    previous_prescription_url=None,
    medical_advice=None,
    case_summary=None
):
    logger.info(f"Inserting new patient record into DB for: {name}, age: {age}")
    insert_query = text("""
        INSERT INTO patient_info (
            patient_name,
            age,
            gender,
            contact_number,
            department,
            chief_complaint,
            history_of_presenting_illness,
            past_history,
            personal_history,
            family_history,
            obg_history,
            lab_report_url,
            medical_imaging_url,
            previous_prescription_url,
            medical_advice,
            case_summary
        ) VALUES (
            :patient_name,
            :age,
            :gender,
            :contact_number,
            :department,
            :chief_complaint,
            :history_of_presenting_illness,
            :past_history,
            :personal_history,
            :family_history,
            :obg_history,
            :lab_report_url,
            :medical_imaging_url,
            :previous_prescription_url,
            :medical_advice,
            :case_summary
        ) RETURNING id
    """)

    with engine.begin() as conn:
        result = conn.execute(insert_query, {
            'patient_name': name,
            'age': age,
            'gender': gender,
            'contact_number': contact_number,
            'department': department,
            'chief_complaint': chief_complaint,
            'history_of_presenting_illness': history_presenting_illness,
            'past_history': past_history,
            'personal_history': personal_history,
            'family_history': family_history,
            'obg_history': obg_history,
            'lab_report_url': lab_report_url,
            'medical_imaging_url': medical_imaging_url,
            'previous_prescription_url': previous_prescription_url,
            'medical_advice': medical_advice,
            'case_summary': case_summary
        })
        new_id = result.fetchone()[0]
    logger.info(f"Patient record inserted successfully with ID: {new_id}")
    return new_id

def update_patient_final_choices(
    patient_id, 
    final_diagnosis, 
    final_tests, 
    final_treatment_plan,
    case_summary=None
):
    logger.info(f"Updating final choices for patient ID: {patient_id}")
    update_query = text("""
        UPDATE patient_info
        SET final_diagnosis = :final_diagnosis,
            final_tests = :final_tests,
            final_treatment_plan = :final_treatment_plan,
            case_summary = :case_summary
        WHERE id = :id
    """)
    with engine.begin() as conn:
        conn.execute(update_query, {
            'final_diagnosis': final_diagnosis,
            'final_tests': final_tests,
            'final_treatment_plan': final_treatment_plan,
            'case_summary': case_summary,
            'id': patient_id
        })
    logger.info(f"Final diagnosis/tests/treatment updated for patient ID: {patient_id}")

def search_patients(name=None, age=None, gender=None, contact=None):
    logger.debug("Searching patient records...")
    query = "SELECT * FROM patient_info WHERE 1=1"
    params = {}

    if name:
        query += " AND patient_name ILIKE :name"
        params['name'] = f"%{name}%"
    if age and age > 0:
        query += " AND age = :age"
        params['age'] = age
    if gender and gender not in ["Select Gender", None, ""]:
        query += " AND gender = :gender"
        params['gender'] = gender
    if contact:
        query += " AND contact_number ILIKE :contact"
        params['contact'] = f"%{contact}%"

    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        records = result.mappings().all()
    logger.debug(f"Search query returned {len(records)} record(s).")
    return records

# -------------------- STREAMLIT APP -------------------- #
st.title("MedMitra AI")

tab_selection = st.sidebar.radio(
    "Navigation",
    [
        "Patient Information",
        "Diagnosis, Prognosis & Treatment",
        "Follow-Up Questions",
        "Search Patient Records"
    ]
)

department = st.sidebar.selectbox("Select Department", list(contexts.keys()))

# Keep track of ephemeral data in session state
if "patient_data" not in st.session_state:
    st.session_state["patient_data"] = {}
if "gpt_advice_text" not in st.session_state:
    st.session_state["gpt_advice_text"] = ""
if "gpt_case_summary" not in st.session_state:
    st.session_state["gpt_case_summary"] = ""
# For conversation-based follow-up
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = ""
# For suggested questions
if "suggested_questions" not in st.session_state:
    st.session_state["suggested_questions"] = ""

# -------------------- 1) PATIENT INFORMATION TAB -------------------- #
if tab_selection == "Patient Information":
    st.header(f"{department} - Patient Information")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Patient Name*", value="").strip()
        age = st.number_input("Age*", min_value=1, max_value=120, value=30)
    with col2:
        gender_sel = st.selectbox("Gender*", ["Select Gender", "Male", "Female", "Other"])
        contact_number = st.text_input("Contact Number*", value="").strip()

    chief_complaint = st.text_input("Chief Complaint*", value="")
    history_presenting_illness = st.text_input("History of Presenting Illness*", value="")
    past_history = st.text_input("Past History*", value="")
    personal_history = st.text_input("Personal History*", value="")
    family_history = st.text_input("Family History*", value="")

    obg_history = ""
    if department == "Gynecology":
        obg_history = st.text_input("OBG History", value="")

    st.write("Optional: Upload a Lab Report (PDF), Medical Imaging (PNG/JPG), or a Previous Prescription (PDF/PNG/JPG).")

    # Lab Report (PDF)
    lab_report_file = st.file_uploader("Upload Lab Report (PDF)", type=["pdf"])
    lab_report_text = ""
    lab_report_url = None
    if lab_report_file:
        pdf_bytes = lab_report_file.read()
        lab_report_url = upload_to_s3(pdf_bytes, lab_report_file.name)

        # Extract text
        lab_report_file.seek(0)
        extracted_lab_text = extract_text_from_pdf(lab_report_file)
        if extracted_lab_text:
            st.write("Extracted Text from PDF:")
            st.text_area("Lab Report Data", extracted_lab_text, height=200)
            lab_report_text = extracted_lab_text
        else:
            st.error("Could not extract text from the PDF (or no text found).")

    # Medical Imaging
    image_data_b64 = ""
    medical_imaging_url = None
    image_file = st.file_uploader("Upload Medical Imaging (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if image_file:
        raw_bytes = image_file.read()
        smaller_img_bytes = downsample_image(raw_bytes, max_size=(500, 500), quality=50)
        st.image(smaller_img_bytes, caption="Downsampled Medical Image", use_column_width=True)

        medical_imaging_url = upload_to_s3(smaller_img_bytes, image_file.name)

        # For GPT analysis
        file_like = io.BytesIO(smaller_img_bytes)
        encoded_img = base64.b64encode(file_like.read()).decode("utf-8")
        image_data_b64 = encoded_img

    # Previous Prescription
    prescription_url = None
    prescription_text_or_b64 = ""
    prescription_is_pdf = False
    prescription_file = st.file_uploader("Upload Previous Prescription (PDF/PNG/JPG)", type=["pdf", "png", "jpg", "jpeg"])
    if prescription_file:
        prescrip_bytes = prescription_file.read()
        prescription_url = upload_to_s3(prescrip_bytes, prescription_file.name)

        if prescription_file.type == "application/pdf":
            prescription_is_pdf = True
            file_like = io.BytesIO(prescrip_bytes)
            extracted_prescrip_text = extract_text_from_pdf(file_like)
            if extracted_prescrip_text:
                prescription_text_or_b64 = extracted_prescrip_text
                st.success("Prescription PDF processed successfully.")
            else:
                st.warning("No text found in prescription PDF.")
        else:
            # It's an image
            compressed_prescrip = downsample_image(prescrip_bytes, max_size=(500, 500), quality=50)
            st.image(compressed_prescrip, caption="Downsampled Previous Prescription", use_column_width=True)
            # Convert to base64
            file_like = io.BytesIO(compressed_prescrip)
            encoded_prescrip_img = base64.b64encode(file_like.read()).decode("utf-8")
            prescription_text_or_b64 = encoded_prescrip_img
            st.success("Prescription image uploaded & prepared for analysis.")

    if st.button("Save Patient Info"):
        if not name:
            st.error("Patient Name is required")
        elif age < 1:
            st.error("Please enter a valid age")
        elif gender_sel == "Select Gender":
            st.error("Please select a gender")
        elif not contact_number:
            st.error("Contact Number is required")
        elif not chief_complaint:
            st.error("Chief Complaint is required")
        elif not all([history_presenting_illness, past_history, personal_history, family_history]):
            st.error("Please fill in all the required fields.")
        else:
            st.session_state["patient_data"] = {
                "name": name,
                "age": age,
                "gender": gender_sel,
                "contact_number": contact_number,
                "department": department,
                "chief_complaint": chief_complaint,
                "history_presenting_illness": history_presenting_illness,
                "past_history": past_history,
                "personal_history": personal_history,
                "family_history": family_history,
                "obg_history": obg_history if department == "Gynecology" else "",
                "lab_report_text": lab_report_text,
                "lab_report_url": lab_report_url,
                "image_data_b64": image_data_b64,
                "medical_imaging_url": medical_imaging_url,
                "prescription_data": prescription_text_or_b64,
                "prescription_is_pdf": prescription_is_pdf,
                "previous_prescription_url": prescription_url
            }

            try:
                with st.spinner("Saving patient info..."):
                    new_id = insert_patient_info(
                        name=name,
                        age=age,
                        gender=gender_sel,
                        contact_number=contact_number,
                        department=department,
                        chief_complaint=chief_complaint,
                        history_presenting_illness=history_presenting_illness,
                        past_history=past_history,
                        personal_history=personal_history,
                        family_history=family_history,
                        obg_history=obg_history if department == "Gynecology" else "",
                        lab_report_url=lab_report_url,
                        medical_imaging_url=medical_imaging_url,
                        previous_prescription_url=prescription_url
                    )
                    st.session_state["patient_data"]["id"] = new_id
                st.success(f"Patient info saved successfully with ID: {new_id}")
            except Exception as e:
                st.error(f"Failed to save patient info: {e}")


# -------------------- 2) DIAGNOSIS, PROGNOSIS & TREATMENT TAB -------------------- #
elif tab_selection == "Diagnosis, Prognosis & Treatment":
    st.header(f"{department} - Diagnosis, Prognosis & Treatment")

    patient_data = st.session_state.get("patient_data", {})
    if not patient_data.get("name"):
        st.warning("Please fill out 'Patient Information' first.")
    else:
        if st.button("Get Medical Advice"):
            with st.spinner("Analyzing labs, images, prescriptions, then generating advice..."):
                # 1) Image analysis
                image_analysis_text = ""
                if patient_data.get("image_data_b64"):
                    image_analysis_text = analyze_medical_image(patient_data["image_data_b64"])

                # 2) Lab report analysis
                lab_analysis_text = ""
                if patient_data.get("lab_report_text"):
                    lab_analysis_text = analyze_lab_report_text(patient_data["lab_report_text"])

                # 3) Prescription analysis
                prescription_analysis_text = ""
                if patient_data.get("prescription_data"):
                    is_pdf = patient_data["prescription_is_pdf"]
                    prescription_analysis_text = analyze_prescription_text_or_image(
                        prescription_content=patient_data["prescription_data"],
                        is_pdf=is_pdf
                    )

                # 4) Final advice combining everything
                advice_text = get_medical_advice(
                    department=patient_data["department"],
                    chief_complaint=patient_data["chief_complaint"],
                    history_presenting_illness=patient_data["history_presenting_illness"],
                    past_history=patient_data["past_history"],
                    personal_history=patient_data["personal_history"],
                    family_history=patient_data["family_history"],
                    obg_history=patient_data.get("obg_history", ""),
                    image_analysis_text=image_analysis_text,
                    lab_analysis_text=lab_analysis_text,
                    prescription_analysis_text=prescription_analysis_text
                )

            # Parse out "Case Summary" if needed
            case_summary_section = parse_section(advice_text, "Case Summary")
            advice_text_no_cs = remove_section(advice_text, "Case Summary")

            st.session_state["gpt_advice_text"] = advice_text_no_cs
            st.session_state["gpt_case_summary"] = case_summary_section

            st.markdown(advice_text_no_cs)

            # Save to DB
            if "id" in patient_data:
                try:
                    update_query = text("""
                        UPDATE patient_info
                        SET medical_advice = :advice,
                            case_summary = :case_summary
                        WHERE id = :pid
                    """)
                    with engine.begin() as conn:
                        conn.execute(update_query, {
                            'advice': advice_text_no_cs,
                            'case_summary': case_summary_section,
                            'pid': patient_data["id"]
                        })
                    st.success("Advice & Case Summary saved in DB!")
                except Exception as e:
                    st.error(f"DB update error: {e}")

        # Show final bullet picks
        gpt_text = st.session_state.get("gpt_advice_text", "")
        case_summary_text = st.session_state.get("gpt_case_summary", "")

        if gpt_text:
            st.subheader("Select Your Final Choices Below")
            diag_section = parse_section(gpt_text, "Most Likely Diagnosis")
            other_diag_section = parse_section(gpt_text, "Other Possible Diagnoses")
            tests_section = parse_section(gpt_text, "Suggested Tests")
            treat_section = parse_section(gpt_text, "Suggested Treatment Plan")

            # Extract bullet items
            most_likely_items = extract_bullet_items(diag_section)
            other_diag_list = extract_bullet_items(other_diag_section)
            tests_list = extract_bullet_items(tests_section)
            treat_list = extract_bullet_items(treat_section)

            final_diagnosis_radio = None
            if most_likely_items:
                st.markdown("**Most Likely Diagnosis (choose one)**")
                final_diagnosis_radio = st.radio("", options=most_likely_items)

            selected_other_diag = None
            if other_diag_list:
                st.markdown("**Other Possible Diagnoses (pick one if desired)**")
                selected_other_diag = st.radio("", options=["(None)"] + other_diag_list)

            selected_tests = []
            if tests_list:
                st.markdown("**Suggested Tests (check all that apply)**")
                for test in tests_list:
                    checked = st.checkbox(test, value=False)
                    if checked:
                        selected_tests.append(test)

            selected_treats = []
            if treat_list:
                st.markdown("**Suggested Treatment Plan (check all that apply)**")
                for treat_item in treat_list:
                    checked = st.checkbox(treat_item, value=False)
                    if checked:
                        selected_treats.append(treat_item)

            # Combine final diagnosis
            final_dx = ""
            if final_diagnosis_radio:
                final_dx = final_diagnosis_radio
            if selected_other_diag and selected_other_diag != "(None)":
                final_dx = selected_other_diag

            final_tests_str = ", ".join(selected_tests)
            final_treatment_str = ", ".join(selected_treats)

            if st.button("Save Selections"):
                pid = patient_data.get("id")
                if not pid:
                    st.error("No patient ID found. Please ensure patient info is saved.")
                else:
                    try:
                        update_patient_final_choices(
                            patient_id=pid,
                            final_diagnosis=final_dx,
                            final_tests=final_tests_str,
                            final_treatment_plan=final_treatment_str,
                            case_summary=case_summary_text
                        )
                        st.success("Doctor's Final Choices & Case Summary Saved Successfully!")
                    except Exception as e:
                        st.error(f"DB update error: {e}")


# -------------------- 3) FOLLOW-UP QUESTIONS TAB -------------------- #
elif tab_selection == "Follow-Up Questions":
    st.header("Follow-Up Questions")

    def compile_patient_context_for_gpt():
        """
        Compile the entire relevant patient record + GPT advice + conversation history
        into one string for GPT context.
        """
        patient_data = st.session_state.get("patient_data", {})
        advice_text = st.session_state.get("gpt_advice_text", "")
        case_summary = st.session_state.get("gpt_case_summary", "")
        conversation_history = st.session_state.get("conversation_history", "")

        if not patient_data:
            return ""

        context_str = f"""
        Patient Name: {patient_data.get('name','')}
        Age: {patient_data.get('age','')}
        Gender: {patient_data.get('gender','')}
        Contact Number: {patient_data.get('contact_number','')}
        Department: {patient_data.get('department','')}
        Chief Complaint: {patient_data.get('chief_complaint','')}
        History of Presenting Illness: {patient_data.get('history_presenting_illness','')}
        Past History: {patient_data.get('past_history','')}
        Personal History: {patient_data.get('personal_history','')}
        Family History: {patient_data.get('family_history','')}
        OBG History: {patient_data.get('obg_history','')}

        Lab Report Text (if any): {patient_data.get('lab_report_text','')}

        ---- GPT Advice So Far ----
        {advice_text}

        ---- Case Summary (if any) ----
        {case_summary}

        ---- Conversation So Far ----
        {conversation_history}
        """

        return context_str.strip()

    def ask_gpt_followup_question(full_context, question):
        """
        Uses GPT-4 for the conversation, referencing entire context.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI medical assistant. You have access to the patient's entire record, "
                    "previous GPT advice, and the conversation so far. Provide clinically relevant, concise answers. "
                    "Use brand names/dosages where appropriate, and do not add disclaimers about specialist follow-ups."
                ),
            },
            {
                "role": "user",
                "content": f"Full Patient Context:\n{full_context}\n\nFollow-Up Question: {question}"
            }
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.5,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error from OpenAI: {str(e)}"

    def generate_question_suggestions(full_context):
        """
        Generate a set of 5 new follow-up questions from different angles.
        """
        prompt = f"""
        You are a helpful AI medical assistant. You've seen the entire patient context, including conversation so far.

        Please propose 5 new, non-repetitive, interesting and relevant follow-up questions a doctor might ask, from different angles
        (e.g., clarifications on symptoms, comorbid conditions, medication side effects, social or lifestyle factors, next steps in testing, etc.).
        Avoid duplicating previous questions from the conversation.

        Return them either as a numbered list or bullet list.

        FULL CONTEXT:
        {full_context}
        """

        messages = [
            {"role": "system", "content": "You are ChatGPT, a helpful AI medical assistant."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=400
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error from OpenAI: {str(e)}"

    # Ensure patient info is saved
    patient_data = st.session_state.get("patient_data", {})
    if not patient_data:
        st.warning("No patient information available. Please fill out 'Patient Information' first.")
        st.stop()

    st.write("Use this chatbot interface to ask any follow-up questions regarding the patient's history, labs, images, prescriptions, or previous GPT advice.")

    # 1) Generate suggestions
    if st.button("Generate Suggested Follow-Up Questions"):
        full_context = compile_patient_context_for_gpt()
        suggestions_text = generate_question_suggestions(full_context)
        st.session_state["suggested_questions"] = suggestions_text

    # Show suggestions
    suggestions = st.session_state.get("suggested_questions", "")
    if suggestions:
        st.markdown("### Suggested Follow-Up Questions")
        st.markdown(suggestions)

    # 2) Let user type a question or pick from suggestions
    st.write("---")
    st.subheader("Ask a Follow-Up Question")
    follow_up_question = st.text_input("Type your question here")

    # Parse suggestions from bullet or numbered lines
    suggestion_lines = []
    for line in suggestions.splitlines():
        line = line.strip()
        # If line starts with "- " or a digit + punctuation
        if line.startswith("- ") or re.match(r"^\d+[.)]\s", line):
            suggestion_lines.append(line)

    selected_suggestion = st.selectbox("...or pick one of the suggestions above", ["(None)"] + suggestion_lines)

    if st.button("Submit Follow-Up"):
        question_to_ask = follow_up_question.strip()
        if selected_suggestion != "(None)" and not question_to_ask:
            # If user didn't type anything, but selected a suggestion
            question_to_ask = re.sub(r"^(\- |\d+[.)]\s)", "", selected_suggestion).strip()

        if question_to_ask:
            full_context = compile_patient_context_for_gpt()
            response_from_gpt = ask_gpt_followup_question(full_context, question_to_ask)

            # 4) Update conversation history
            current_history = st.session_state.get("conversation_history", "")
            updated_conversation = (
                current_history
                + f"\n\nDoctor's Question: {question_to_ask}\nAI's Answer: {response_from_gpt}"
            )
            st.session_state["conversation_history"] = updated_conversation

            # 5) Show GPT’s response
            st.markdown("### Response")
            st.write(response_from_gpt)

            # Clear typed question
            st.session_state["follow_up_question"] = ""
        else:
            st.warning("Please type or select a question before clicking 'Submit Follow-Up'.")


# -------------------- 4) SEARCH PATIENT RECORDS TAB -------------------- #
elif tab_selection == "Search Patient Records":
    st.header("Search Patient Records")

    search_name = st.text_input("Search by Name").strip()
    search_age = st.number_input("Search by Age", min_value=0, value=0)
    search_gender = st.selectbox("Search by Gender", ["Select Gender", "Male", "Female", "Other"])
    search_contact = st.text_input("Search by Contact Number").strip()

    if st.button("Search"):
        records = search_patients(
            name=search_name,
            age=search_age if search_age > 0 else None,
            gender=search_gender if search_gender != "Select Gender" else None,
            contact=search_contact
        )

        if records:
            st.success(f"Found {len(records)} record(s)")
            for record in records:
                with st.expander(f"Record ID: {record['id']} - {record['patient_name']}"):
                    # Show main info
                    st.markdown(f"**Name:** {record['patient_name']}")
                    st.markdown(f"**Age:** {record['age']}")
                    st.markdown(f"**Gender:** {record['gender']}")
                    st.markdown(f"**Contact:** {record['contact_number']}")
                    st.markdown(f"**Department:** {record['department']}")
                    st.markdown(f"**Chief Complaint:** {record['chief_complaint']}")
                    st.markdown(f"**History of Presenting Illness:** {record['history_of_presenting_illness']}")
                    st.markdown(f"**Past History:** {record['past_history']}")
                    st.markdown(f"**Personal History:** {record['personal_history']}")
                    st.markdown(f"**Family History:** {record['family_history']}")
                    if record.get('obg_history'):
                        st.markdown(f"**OBG History:** {record['obg_history']}")

                    # Final doc-chosen items
                    st.markdown("**Final Diagnosis (Doc Selected):**")
                    final_dx_val = record.get('final_diagnosis', '')
                    st.write(f"- {final_dx_val}" if final_dx_val else "Not provided")

                    st.markdown("**Final Tests (Doc Selected):**")
                    final_tests_val = record.get('final_tests', '')
                    if final_tests_val:
                        splitted_tests = [x.strip() for x in final_tests_val.split(',') if x.strip()]
                        for test in splitted_tests:
                            st.write(f"- {test}")
                    else:
                        st.write("Not provided")

                    st.markdown("**Final Treatment Plan (Doc Selected):**")
                    final_treat_val = record.get('final_treatment_plan', '')
                    if final_treat_val:
                        splitted_treats = [x.strip() for x in final_treat_val.split(',') if x.strip()]
                        for tr in splitted_treats:
                            st.write(f"- {tr}")
                    else:
                        st.write("Not provided")

                    # GPT advice (minus "Case Summary")
                    st.markdown("**Medical Advice (GPT Output, minus Case Summary):**")
                    st.markdown(record.get('medical_advice','') or "No GPT advice stored.")

                    # Case Summary
                    st.markdown("**Case Summary:**")
                    if record.get('case_summary'):
                        st.write(record['case_summary'])
                    else:
                        st.write("None")

                    st.markdown(f"**Created At:** {record.get('created_at', '')}")

                    # ---------- Lab Report PDF Link / Iframe ----------
                    lab_url = record.get("lab_report_url")
                    if lab_url:
                        st.markdown("**Lab Report (PDF):**")
                        presigned_pdf_url = generate_presigned_url(lab_url)
                        if presigned_pdf_url:
                            st.write(f"[Open PDF]({presigned_pdf_url})")
                            st.markdown(
                                f"""
                                <iframe src="{presigned_pdf_url}" width="700" height="1000">
                                This browser does not support PDFs. 
                                <a href="{presigned_pdf_url}">Download PDF</a>
                                </iframe>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning("Could not generate a presigned URL for this PDF.")

                    # ---------- Medical Imaging (PNG/JPG) ----------
                    med_img_url = record.get("medical_imaging_url")
                    if med_img_url:
                        st.markdown("**Medical Imaging Preview (Image)**")
                        presigned_img_url = generate_presigned_url(med_img_url)
                        if presigned_img_url:
                            st.image(presigned_img_url, use_column_width=True)
                        else:
                            st.warning("Could not generate a presigned URL for this image.")

                    # ---------- Previous Prescription Link / Iframe ----------
                    prescription_url = record.get("previous_prescription_url")
                    if prescription_url:
                        st.markdown("**Previous Prescription:**")
                        presigned_prescription_url = generate_presigned_url(prescription_url)
                        if presigned_prescription_url:
                            if prescription_url.lower().endswith(".pdf"):
                                st.write(f"[Open PDF]({presigned_prescription_url})")
                                st.markdown(
                                    f"""
                                    <iframe src="{presigned_prescription_url}" width="700" height="1000">
                                    This browser does not support PDFs. 
                                    <a href="{presigned_prescription_url}">Download PDF</a>
                                    </iframe>
                                    """,
                                    unsafe_allow_html=True
                                )
                            else:
                                st.image(presigned_prescription_url, use_column_width=True)
                        else:
                            st.warning("Could not generate a presigned URL for this prescription.")
        else:
            st.warning("No records found matching the search criteria.")
