import streamlit as st
from datetime import time
import time as pytime
import os
import io
import re
import uuid
import base64
import logging
import tempfile
from datetime import datetime, date

import pdfplumber
import requests
from PIL import Image
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import boto3
from openai import OpenAI
 # <-- using openai library directly
import json
from sqlalchemy import text

# Additional imports for PDF styling
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.colors import black

# Load environment variables
load_dotenv()

# -------------------------------------------------- #
#             OPENAI CLIENT CONFIG                   #
# -------------------------------------------------- #
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     raise ValueError("Please set OPENAI_API_KEY in your environment or secrets.")
# openai.api_key = openai_api_key

# ---------------------------------------
#    LOGGING AND ENVIRONMENT SETUP
# ---------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - [%(levelname)s] - %(filename)s.%(funcName)s(%(lineno)d) - %(message)s',
    handlers=[logging.FileHandler("medmitra.log", mode='a'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------------------------------
#      SECRETS / CONFIG VIA Streamlit
# ---------------------------------------
# openai_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=openai_key)


# DATABASE_URI = os.getenv("DATABASE_URI", "postgresql://postgres:Medmitra123%23@patientrecords.cte8m8wug3oq.us-east-1.rds.amazonaws.com:5432/postgres")
# if not DATABASE_URI:
#     logger.warning("DATABASE_URI not set.")


# aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "YOUR_AWS_ACCESS_KEY")
# aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "YOUR_AWS_SECRET")
# aws_region = os.getenv("AWS_REGION", "us-east-1")
# bucket_name = os.getenv("AWS_BUCKET_NAME", "your-s3-bucket")

DATABASE_URI = st.secrets["DATABASE_URI"]
aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
aws_region = st.secrets["AWS_REGION"]
bucket_name = st.secrets["AWS_BUCKET_NAME"]


openai_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_key)


# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URI, echo=False)

# Create the S3 client
s3_client = boto3.client(
    's3',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# ---------------------------------------
#        DEPARTMENT CONTEXTS
# ---------------------------------------
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

# ---------------------------------------
#        HELPER / UTILITY FUNCTIONS
# ---------------------------------------
def get_content_type(filename):
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
    unique_filename = f"{uuid.uuid4()}_{filename}"
    content_type = get_content_type(filename)
    s3_client.put_object(
        Bucket=bucket_name,
        Key=unique_filename,
        Body=file_bytes,
        ContentType=content_type,
        ACL='private'
    )
    return f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{unique_filename}"

def generate_presigned_url(s3_url, expiration=3600):
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
    """Reduce size & quality to keep images smaller."""
    buf = io.BytesIO(file_bytes)
    img = Image.open(buf)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail(max_size)
    out_bytes = io.BytesIO()
    img.save(out_bytes, format="JPEG", quality=quality)
    out_bytes.seek(0)
    return out_bytes.read()

def extract_text_from_pdf(pdf_file):
    """Extract text from each page of a PDF using pdfplumber."""
    with pdfplumber.open(pdf_file) as pdf:
        all_text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"
    return all_text

# ---------------------------------------
#         DATABASE FUNCTIONS
# ---------------------------------------
def insert_patient_version(
    patient_id,
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
    obg_history=None,
    lab_report_url=None,
    medical_imaging_url=None,
    previous_prescription_url=None,
    medical_advice=None,
    case_summary=None,
    final_diagnosis=None,
    final_tests=None,
    final_treatment_plan=None,
    uhid=None,
    guardian_name=None,
    consultant_doctor=None,
    address=None,
    queue_no=None,
    registration_date=None
):
    """Inserts a new row into patient_info_versions with a local Python timestamp."""
    logger.info(f"Inserting new version row for patient {patient_id}.")
    version_query = text("""
        INSERT INTO patient_info_versions (
            patient_id,
            version_timestamp,
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
            case_summary,
            final_diagnosis,
            final_tests,
            final_treatment_plan,
            uhid,
            guardian_name,
            consultant_doctor,
            address,
            queue_no,
            registration_date
        ) VALUES (
            :pid,
            :ts,
            :pname,
            :page,
            :pgender,
            :pcontact,
            :pdept,
            :pchief,
            :phpi,
            :ppast,
            :ppers,
            :pfam,
            :pobg,
            :plab,
            :pimg,
            :ppres,
            :padvice,
            :pcase,
            :fdx,
            :ftests,
            :ftreat,
            :uhid,
            :gname,
            :cdoctor,
            :addr,
            :qno,
            :rdate
        )
    """)
    with engine.begin() as conn:
        conn.execute(version_query, {
            'pid': patient_id,
            'ts': datetime.now(),
            'pname': name,
            'page': age,
            'pgender': gender,
            'pcontact': contact_number,
            'pdept': department,
            'pchief': chief_complaint,
            'phpi': history_presenting_illness,
            'ppast': past_history,
            'ppers': personal_history,
            'pfam': family_history,
            'pobg': obg_history,
            'plab': lab_report_url,
            'pimg': medical_imaging_url,
            'ppres': previous_prescription_url,
            'padvice': medical_advice,
            'pcase': case_summary,
            'fdx': final_diagnosis,
            'ftests': final_tests,
            'ftreat': final_treatment_plan,
            'uhid': uhid,
            'gname': guardian_name,
            'cdoctor': consultant_doctor,
            'addr': address,
            'qno': queue_no,
            'rdate': registration_date
        })
    logger.info(f"Version row added for patient {patient_id}.")

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
    case_summary=None,
    uhid=None,
    guardian_name=None,
    consultant_doctor=None,
    address=None,
    queue_no=None,
    registration_date=None
):
    """
    1) Create a new 'latest' record in patient_info.
    2) Also insert the same data as the initial row in patient_info_versions.
    """
    logger.info(f"Inserting new patient record for: {name}, age: {age}")
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
            case_summary,
            uhid,
            guardian_name,
            consultant_doctor,
            address,
            queue_no,
            registration_date
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
            :case_summary,
            :uhid,
            :gname,
            :cdoctor,
            :addr,
            :qno,
            :rdate
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
            'case_summary': case_summary,
            'uhid': uhid,
            'gname': guardian_name,
            'cdoctor': consultant_doctor,
            'addr': address,
            'qno': queue_no,
            'rdate': registration_date
        })
        new_id = result.fetchone()[0]

    # Also insert into versions table
    insert_patient_version(
        patient_id=new_id,
        name=name,
        age=age,
        gender=gender,
        contact_number=contact_number,
        department=department,
        chief_complaint=chief_complaint,
        history_presenting_illness=history_presenting_illness,
        past_history=past_history,
        personal_history=personal_history,
        family_history=family_history,
        obg_history=obg_history,
        lab_report_url=lab_report_url,
        medical_imaging_url=medical_imaging_url,
        previous_prescription_url=previous_prescription_url,
        medical_advice=medical_advice,
        case_summary=case_summary,
        uhid=uhid,
        guardian_name=guardian_name,
        consultant_doctor=consultant_doctor,
        address=address,
        queue_no=queue_no,
        registration_date=registration_date
    )
    logger.info(f"Patient record inserted with ID: {new_id}")
    return new_id

def update_patient_info(
    patient_id,
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
    obg_history=None,
    lab_report_url=None,
    medical_imaging_url=None,
    previous_prescription_url=None,
    uhid=None,
    guardian_name=None,
    consultant_doctor=None,
    address=None,
    queue_no=None,
    registration_date=None
):
    """
    1) Update 'latest' in patient_info
    2) Insert a new version row (with local timestamp).
    """
    logger.info(f"Updating patient_info ID: {patient_id}")
    update_query = text("""
        UPDATE patient_info
        SET
            patient_name = :name,
            age = :age,
            gender = :gender,
            contact_number = :contact,
            department = :department,
            chief_complaint = :chief,
            history_of_presenting_illness = :hpi,
            past_history = :past,
            personal_history = :personal,
            family_history = :family,
            obg_history = :obg,
            lab_report_url = :lab_url,
            medical_imaging_url = :img_url,
            previous_prescription_url = :prescrip_url,
            uhid = :uhid,
            guardian_name = :gname,
            consultant_doctor = :cdoctor,
            address = :addr,
            queue_no = :qno,
            registration_date = :rdate
        WHERE id = :pid
    """)
    with engine.begin() as conn:
        conn.execute(update_query, {
            'name': name,
            'age': age,
            'gender': gender,
            'contact': contact_number,
            'department': department,
            'chief': chief_complaint,
            'hpi': history_presenting_illness,
            'past': past_history,
            'personal': personal_history,
            'family': family_history,
            'obg': obg_history,
            'lab_url': lab_report_url,
            'img_url': medical_imaging_url,
            'prescrip_url': previous_prescription_url,
            'uhid': uhid,
            'gname': guardian_name,
            'cdoctor': consultant_doctor,
            'addr': address,
            'qno': queue_no,
            'rdate': registration_date,
            'pid': patient_id
        })

    # Also insert new version
    insert_patient_version(
        patient_id=patient_id,
        name=name,
        age=age,
        gender=gender,
        contact_number=contact_number,
        department=department,
        chief_complaint=chief_complaint,
        history_presenting_illness=history_presenting_illness,
        past_history=past_history,
        personal_history=personal_history,
        family_history=family_history,
        obg_history=obg_history,
        lab_report_url=lab_report_url,
        medical_imaging_url=medical_imaging_url,
        previous_prescription_url=previous_prescription_url,
        uhid=uhid,
        guardian_name=guardian_name,
        consultant_doctor=consultant_doctor,
        address=address,
        queue_no=queue_no,
        registration_date=registration_date
    )

def update_patient_medical_advice(version_id, advice_text, case_summary):
    """Update the existing version row with newly generated GPT advice."""
    logger.info(f"Updating medical advice in version ID: {version_id}")
    query = text("""
        UPDATE patient_info_versions
        SET medical_advice = :madv,
            case_summary = :csum
        WHERE id = :vid
    """)
    with engine.begin() as conn:
        conn.execute(query, {
            'madv': advice_text,
            'csum': case_summary,
            'vid': version_id
        })

def update_patient_final_choices(
    patient_id,
    final_diagnosis,
    final_tests,
    final_treatment_plan,
    case_summary=None
):
    """
    1) Update final dx/tests/treatment in 'latest' patient_info
    2) Insert new version row capturing these fields
    """
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

    # Pull the new "latest" row so we can version it
    sel_query = text("SELECT * FROM patient_info WHERE id = :pid")
    with engine.begin() as conn:
        row = conn.execute(sel_query, {'pid': patient_id}).mappings().first()

    insert_patient_version(
        patient_id=patient_id,
        name=row['patient_name'],
        age=row['age'],
        gender=row['gender'],
        contact_number=row['contact_number'],
        department=row['department'],
        chief_complaint=row['chief_complaint'],
        history_presenting_illness=row['history_of_presenting_illness'],
        past_history=row['past_history'],
        personal_history=row['personal_history'],
        family_history=row['family_history'],
        obg_history=row['obg_history'],
        lab_report_url=row['lab_report_url'],
        medical_imaging_url=row['medical_imaging_url'],
        previous_prescription_url=row['previous_prescription_url'],
        medical_advice=row['medical_advice'],
        case_summary=row['case_summary'],
        final_diagnosis=final_diagnosis,
        final_tests=final_tests,
        final_treatment_plan=final_treatment_plan,
        uhid=row.get('uhid'),
        guardian_name=row.get('guardian_name'),
        consultant_doctor=row.get('consultant_doctor'),
        address=row.get('address'),
        queue_no=row.get('queue_no'),
        registration_date=row.get('registration_date')
    )

def search_patients(name=None, age=None, gender=None, contact=None, uhid=None):
    """
    Search patient_info records by optional filters: name, age, gender, contact, uhid.
    """
    logger.debug("Searching patient_info for main records...")
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
    if uhid:
        query += " AND uhid ILIKE :uhid"
        params['uhid'] = f"%{uhid}%"

    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        records = result.mappings().all()
    logger.debug(f"Search query returned {len(records)} records.")
    return records

def get_patient_versions(patient_id):
    """
    Return all versions from patient_info_versions for this patient, DESC order.
    """
    logger.debug(f"Fetching versions for patient {patient_id}...")
    sel_query = text("""
        SELECT * FROM patient_info_versions
        WHERE patient_id = :pid
        ORDER BY version_timestamp DESC
    """)
    with engine.connect() as conn:
        result = conn.execute(sel_query, {'pid': patient_id}).mappings().all()
    return result

def get_version_by_id(version_id):
    """
    Get a specific version row by its ID.
    """
    sel_query = text("SELECT * FROM patient_info_versions WHERE id = :vid")
    with engine.connect() as conn:
        row = conn.execute(sel_query, {'vid': version_id}).mappings().first()
    return row

# ---------------------------------------
#       GPT-LIKE FUNCTIONS (REAL)
# ---------------------------------------

def analyze_medical_image(image_data_b64):
    """
    Analyze a base64-encoded image using GPT-3.5/4 via conversation.
    We'll pass a system message + user message that references the image in base64.
    """
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful medical AI. "
                    "Examine the following medical image. "
                    "Describe any possible findings, anomalies, or relevant clinical interpretations. "
                    "Avoid disclaimers about consulting a specialist."
                )
            },
            {
                "role": "user",
                "content": f"Here is the image in base64:\n\ndata:image/jpeg;base64,{image_data_b64}\n"
            }
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in analyze_medical_image: {e}")
        return "Error analyzing medical image. Please try again later."

def analyze_lab_report_text(lab_report_text):
    """
    Analyze lab report text using GPT-3.5/4.
    We'll provide instructions to interpret and summarize the lab findings.
    """
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an experienced medical professional. "
                    "Analyze the following lab report text and summarize important findings, "
                    "possible diagnoses, and relevant details. Avoid disclaimers."
                )
            },
            {
                "role": "user",
                "content": lab_report_text
            }
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in analyze_lab_report_text: {e}")
        return "Error analyzing lab report. Please try again later."

def analyze_prescription_text_or_image(prescription_content, is_pdf=False):
    """
    If PDF text is provided, we treat it as text.
    If image content (base64) is provided, we treat it as an image.
    """
    if is_pdf:
        # We have textual content extracted from PDF
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a medical assistant analyzing a prescription from text. "
                        "Extract relevant medications, dosages, or instructions. Avoid disclaimers."
                    )
                },
                {
                    "role": "user",
                    "content": prescription_content
                }
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in analyze_prescription_text_or_image (PDF): {e}")
            return "Error analyzing PDF prescription."
    else:
        # We have base64 image content
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a medical assistant. Analyze the following image (prescription). "
                        "Extract relevant medications, dosages, or instructions. Avoid disclaimers."
                    )
                },
                {
                    "role": "user",
                    "content": f"data:image/jpeg;base64,{prescription_content}"
                }
            ]
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in analyze_prescription_text_or_image (Image): {e}")
            return "Error analyzing image prescription."

def get_medical_advice(
    age,
    gender,
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
    Combine textual fields + analyses to produce final advice.
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
- Provide recommended drug dosages, mention any contra-indications

**Case Summary**
(Short concluding summary)
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant specialized in medical diagnosis and treatment. "
                "No disclaimers, just your best medical reasoning."
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

def generate_prescription(diagnosis, tests, treatments, patient_info=None):
    """
    Generate a prescription using OpenAI's GPT model.

    We request a JSON structure with:
      - 'diagnosis' (string)
      - 'drugs' (array of objects with {name, strength, frequency, duration})
      - 'tests' (array of objects with {name, purpose})
      - 'follow_up' (object with {when_to_return, what_to_monitor})

    Example of the final GPT output:
    {
      "diagnosis": "Stable angina",
      "drugs": [
        {
          "name": "Aspirin",
          "strength": "75 mg",
          "frequency": "once daily",
          "duration": "long-term"
        },
        ...
      ],
      "tests": [
        {
          "name": "Exercise stress test",
          "purpose": "To assess for inducible ischemia during exertion"
        },
        ...
      ],
      "follow_up": {
        "when_to_return": "In 3 months or earlier if symptoms worsen",
        "what_to_monitor": "Chest pain, blood pressure, and any side effects of medications"
      }
    }
    """
    patient_info_str = (
        f"Patient {patient_info.get('name', 'Unknown')}, aged {patient_info.get('age', 'Unknown')}, "
        f"{patient_info.get('gender', 'Unknown')}"
        if patient_info else "No patient info provided."
    )
    
    prompt = f"""
    You are a medical expert. Based on the following details, generate a prescription:
    Diagnosis: {diagnosis}
    Patient Info: {patient_info_str}
    Recommended Tests: {tests}
    Proposed Treatments: {treatments}

    Please return valid JSON with these keys:
      - "diagnosis" (string)
      - "drugs" (array of objects, each with "name", "strength", "frequency", "duration")
      - "tests" (array of objects, each with "name" and "purpose")
      - "follow_up" (an object with "when_to_return" and "what_to_monitor")

    Example structure:
    {{
      "diagnosis": "...",
      "drugs": [
         {{ "name": "...", "strength": "...", "frequency": "...", "duration": "..." }},
         ...
      ],
      "tests": [
         {{ "name": "...", "purpose": "..." }},
         ...
      ],
      "follow_up": {{
         "when_to_return": "...",
         "what_to_monitor": "..."
      }}
    }}
    """
    try:
        messages = [
            {"role": "system", "content": "You are ChatGPT, a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1800,
            temperature=0.7
        )
        gpt_content = response.choices[0].message.content.strip()
        # Attempt to parse the JSON
        prescription = json.loads(gpt_content)
        return prescription
    except Exception as e:
        # Fallback if JSON parsing fails
        logger.error(f"Error in generate_prescription: {e}")
        return {
            "diagnosis": diagnosis,
            "drugs": [],
            "tests": [],
            "follow_up": {
                "when_to_return": "N/A",
                "what_to_monitor": "N/A"
            }
        }
# ---------------------------------------
#  Parsing GPT Advice Helpers
# ---------------------------------------
def parse_section(full_text, section_name):
    """Extract text from a named section with **section_name** pattern."""
    pattern = rf"\*\*{section_name}\*\*\s*\n?(.*?)(?=\n\*\*|$)"
    match = re.search(pattern, full_text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def remove_section(full_text, section_name):
    """Remove the entire section from text, used for case_summary extraction."""
    pattern = rf"\*\*{section_name}\*\*\s*\n?(.*?)(?=\n\*\*|$)"
    return re.sub(pattern, "", full_text, flags=re.IGNORECASE | re.DOTALL).strip()

def extract_bullet_items(section_text):
    items = []
    if not section_text:
        return items
    for line in section_text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            items.append(line[2:].strip())
    return items

# ---------------------------------------
#         STREAMLIT APP LAYOUT
# ---------------------------------------
st.title("MedMitra AI")

tab_selection = st.sidebar.radio(
    "Navigation",
    [
        "Patient Information",
        "Diagnosis, Prognosis & Treatment",
        "Search Patient Records",
        "Prescription Writing"
    ]
)

# Session placeholders
if "patient_data" not in st.session_state:
    st.session_state["patient_data"] = {}
if "gpt_advice_text" not in st.session_state:
    st.session_state["gpt_advice_text"] = ""
if "gpt_case_summary" not in st.session_state:
    st.session_state["gpt_case_summary"] = ""
if "search_results" not in st.session_state:
    st.session_state["search_results"] = []
if "view_id" not in st.session_state:
    st.session_state["view_id"] = None
if "edit_id" not in st.session_state:
    st.session_state["edit_id"] = None
if "version_id" not in st.session_state:
    st.session_state["version_id"] = None
if "prescription" not in st.session_state:
    st.session_state["prescription"] = {}

# ---------------------------------------
# 1) PATIENT INFORMATION TAB
# ---------------------------------------
if tab_selection == "Patient Information":
    st.header("Patient Information")

    existing_department = st.session_state["patient_data"].get("department", "General Medicine")

    # First row: Name, Age, Gender
    colA, colB, colC = st.columns(3)
    with colA:
        name = st.text_input("Patient Name*", value="").strip()
    with colB:
        age = st.number_input("Age*", min_value=1, max_value=120, value=30)
    with colC:
        gender_sel = st.selectbox("Gender*", ["Select Gender", "Male", "Female", "Other"])

    # Second row: UHID, Guardian, Consultant
    colD, colE, colF = st.columns(3)
    with colD:
        uhid_value = st.text_input("UHID/Reg No", value="")
    with colE:
        guardian_name = st.text_input("Guardian Name", value="")
    with colF:
        consultant_doctor = st.text_input("Consultant Doctor", value="")

    # Third row: Department, Contact, Address
    colG, colH, colI = st.columns(3)
    with colG:
        department = st.selectbox("Department*", list(contexts.keys()), index=list(contexts.keys()).index(existing_department))
    with colH:
        contact_number = st.text_input("Contact Number*", value="").strip()
    with colI:
        address = st.text_input("Address", value="")

    # Fourth row: Queue No, Registration Date
    colJ, colK = st.columns(2)
    with colJ:
        queue_no = st.text_input("Queue No", value="")
    with colK:
        registration_date = st.date_input("Registration Date", value=date.today())

    # Additional optional fields
    chief_complaint = st.text_input("Chief Complaint", value="")
    history_presenting_illness = st.text_input("History of Presenting Illness", value="")
    past_history = st.text_input("Past History", value="")
    personal_history = st.text_input("Personal History", value="")
    family_history = st.text_input("Family History", value="")

    obg_history = ""
    if department == "Gynecology":
        obg_history = st.text_input("OBG History", value="")

    st.write("Optional: Upload Lab Report (PDF), Medical Imaging (PNG/JPG), or Prescription (PDF/PNG/JPG).")

    # Lab
    lab_report_file = st.file_uploader("Lab Report (PDF)", type=["pdf"])
    lab_report_text = ""
    lab_report_url = None
    if lab_report_file:
        pdf_bytes = lab_report_file.read()
        lab_report_url = upload_to_s3(pdf_bytes, lab_report_file.name)
        lab_report_file.seek(0)
        extracted_text = extract_text_from_pdf(lab_report_file)
        if extracted_text:
            st.text_area("Extracted Lab Text", extracted_text, height=200)
            lab_report_text = extracted_text

    # Imaging
    image_data_b64 = ""
    medical_imaging_url = None
    image_file = st.file_uploader("Medical Imaging (PNG/JPG)", type=["png","jpg","jpeg"])
    if image_file:
        raw_bytes = image_file.read()
        downsized = downsample_image(raw_bytes)
        st.image(downsized, caption="Resized Imaging", use_column_width=True)
        medical_imaging_url = upload_to_s3(downsized, image_file.name)

        # base64 for GPT usage
        b64file = io.BytesIO(downsized)
        image_data_b64 = base64.b64encode(b64file.read()).decode("utf-8")

    # Prescription
    prescription_url = None
    prescription_text_or_b64 = ""
    prescription_is_pdf = False
    prescrip_file = st.file_uploader("Previous Prescription (PDF/PNG/JPG)", type=["pdf","png","jpg","jpeg"])
    if prescrip_file:
        fbytes = prescrip_file.read()
        prescription_url = upload_to_s3(fbytes, prescrip_file.name)

        if prescrip_file.type == "application/pdf":
            prescription_is_pdf = True
            pfile = io.BytesIO(fbytes)
            extracted_prescription = extract_text_from_pdf(pfile)
            if extracted_prescription:
                prescription_text_or_b64 = extracted_prescription
                st.success("Prescription PDF text extracted.")
        else:
            cpres = downsample_image(fbytes)
            st.image(cpres, caption="Prescription Image", use_column_width=True)
            b64img = base64.b64encode(cpres).decode("utf-8")
            prescription_text_or_b64 = b64img
            st.success("Prescription image prepared.")

    if st.button("Save Patient Info"):
        if not name:
            st.error("Patient Name is required")
        elif age < 1:
            st.error("Please enter a valid age")
        elif gender_sel == "Select Gender":
            st.error("Please select a gender")
        elif not contact_number:
            st.error("Contact Number is required")
        elif not department:
            st.error("Department is required")
        else:
            # Store in session
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
                "previous_prescription_url": prescription_url,
                "uhid": uhid_value,
                "guardian_name": guardian_name,
                "consultant_doctor": consultant_doctor,
                "address": address,
                "queue_no": queue_no,
                "registration_date": registration_date
            }

            try:
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
                    previous_prescription_url=prescription_url,
                    uhid=uhid_value,
                    guardian_name=guardian_name,
                    consultant_doctor=consultant_doctor,
                    address=address,
                    queue_no=queue_no,
                    registration_date=registration_date
                )
                st.session_state["patient_data"]["id"] = new_id
                st.success(f"Patient info saved successfully (ID: {new_id}).")
            except Exception as e:
                st.error(f"Failed to save patient info: {e}")


# ---------------------------------------
# 2) DIAGNOSIS, PROGNOSIS & TREATMENT
# ---------------------------------------
elif tab_selection == "Diagnosis, Prognosis & Treatment":
    dep = st.session_state["patient_data"].get("department", "General Medicine")
    st.header(f"{dep} - Diagnosis, Prognosis & Treatment")

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

                # 2) Lab analysis
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

                # 4) Final advice
                advice_text = get_medical_advice(
                    age=patient_data["age"],
                    gender=patient_data["gender"],
                    department=patient_data["department"],
                    chief_complaint=patient_data.get("chief_complaint",""),
                    history_presenting_illness=patient_data.get("history_presenting_illness",""),
                    past_history=patient_data.get("past_history",""),
                    personal_history=patient_data.get("personal_history",""),
                    family_history=patient_data.get("family_history",""),
                    obg_history=patient_data.get("obg_history",""),
                    image_analysis_text=image_analysis_text,
                    lab_analysis_text=lab_analysis_text,
                    prescription_analysis_text=prescription_analysis_text
                )

            # parse out case summary
            case_summary_section = parse_section(advice_text, "Case Summary")
            advice_text_no_cs = remove_section(advice_text, "Case Summary")

            st.session_state["gpt_advice_text"] = advice_text_no_cs
            st.session_state["gpt_case_summary"] = case_summary_section

            st.markdown(advice_text_no_cs)

            # Also store in DB (new version)
            if "id" in patient_data:
                pid = patient_data["id"]
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
                            'pid': pid
                        })

                    # Insert a new version capturing the new advice
                    insert_patient_version(
                        patient_id=pid,
                        name=patient_data["name"],
                        age=patient_data["age"],
                        gender=patient_data["gender"],
                        contact_number=patient_data["contact_number"],
                        department=patient_data["department"],
                        chief_complaint=patient_data.get("chief_complaint",""),
                        history_presenting_illness=patient_data.get("history_presenting_illness",""),
                        past_history=patient_data.get("past_history",""),
                        personal_history=patient_data.get("personal_history",""),
                        family_history=patient_data.get("family_history",""),
                        obg_history=patient_data.get("obg_history",""),
                        lab_report_url=patient_data.get("lab_report_url"),
                        medical_imaging_url=patient_data.get("medical_imaging_url"),
                        previous_prescription_url=patient_data.get("previous_prescription_url"),
                        medical_advice=advice_text_no_cs,
                        case_summary=case_summary_section,
                        uhid=patient_data.get("uhid"),
                        guardian_name=patient_data.get("guardian_name"),
                        consultant_doctor=patient_data.get("consultant_doctor"),
                        address=patient_data.get("address"),
                        queue_no=patient_data.get("queue_no"),
                        registration_date=patient_data.get("registration_date")
                    )

                    st.success("Advice & Case Summary saved in DB (new version created)!")
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
                st.markdown("**Other Possible Diagnoses**")
                selected_other_diag = st.radio("", options=["(None)"] + other_diag_list)

            selected_tests = []
            if tests_list:
                st.markdown("**Suggested Tests (check all that apply)**")
                for test in tests_list:
                    if st.checkbox(test, value=False):
                        selected_tests.append(test)

            selected_treats = []
            if treat_list:
                st.markdown("**Suggested Treatment Plan (check all that apply)**")
                for treat_item in treat_list:
                    if st.checkbox(treat_item, value=False):
                        selected_treats.append(treat_item)

            final_dx = final_diagnosis_radio if final_diagnosis_radio else ""
            if selected_other_diag and selected_other_diag != "(None)":
                final_dx = selected_other_diag

            final_tests_str = ", ".join(selected_tests)
            final_treatment_str = ", ".join(selected_treats)

            # Store the selections in session state
            st.session_state["final_diagnosis"] = final_dx
            st.session_state["selected_tests"] = selected_tests
            st.session_state["selected_treats"] = selected_treats

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


# ---------------------------------------
# 3) SEARCH PATIENT RECORDS TAB
# ---------------------------------------
elif tab_selection == "Search Patient Records":
    st.header("Search Patient Records")

    s_name = st.text_input("Search by Name").strip()
    s_age = st.number_input("Search by Age", min_value=0, value=0)
    s_gender = st.selectbox("Search by Gender", ["Select Gender", "Male", "Female", "Other"])
    s_contact = st.text_input("Search by Contact Number").strip()
    s_uhid = st.text_input("Search by UHID/Reg No").strip()  # NEW FIELD

    if st.button("Search"):
        found_records = search_patients(
            name=s_name,
            age=s_age if s_age > 0 else None,
            gender=s_gender if s_gender != "Select Gender" else None,
            contact=s_contact,
            uhid=s_uhid
        )
        if found_records:
            st.success(f"Found {len(found_records)} record(s).")
            st.session_state["search_results"] = found_records
            st.session_state["view_id"] = None
            st.session_state["edit_id"] = None
            st.session_state["version_id"] = None
        else:
            st.warning("No matching records found.")
            st.session_state["search_results"] = []
            st.session_state["view_id"] = None
            st.session_state["edit_id"] = None
            st.session_state["version_id"] = None

    # Show basic info for each found record, with a button to view versions or edit
    if st.session_state["search_results"]:
        for rec in st.session_state["search_results"]:
            st.write("---")
            st.write(f"**ID:** {rec['id']} | **Name:** {rec['patient_name']} | **Age:** {rec['age']} | **Gender:** {rec['gender']}")
            st.write(f"**Contact:** {rec['contact_number']} | **Department:** {rec['department']} | **UHID:** {rec.get('uhid','')}")

            c1, c2 = st.columns(2)
            with c1:
                if st.button(f"View Versions for #{rec['id']}", key=f"view_btn_{rec['id']}"):
                    st.session_state["view_id"] = rec["id"]
                    st.session_state["edit_id"] = None
                    st.session_state["version_id"] = None

            with c2:
                if st.button(f"Edit Patient #{rec['id']}", key=f"edit_btn_{rec['id']}"):
                    st.session_state["edit_id"] = rec["id"]
                    st.session_state["view_id"] = None
                    st.session_state["version_id"] = None

        # Viewing versions
        if st.session_state["view_id"]:
            pat_id = st.session_state["view_id"]
            st.write("---")
            st.subheader(f"Viewing Versions for Patient ID: {pat_id}")

            all_versions = get_patient_versions(pat_id)
            if not all_versions:
                st.warning("No version history for this patient.")
            else:
                version_labels = []
                for v in all_versions:
                    ts_str = v['version_timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                    version_labels.append(f"{ts_str} (vID: {v['id']})")

                chosen_label = st.selectbox("Select a version timestamp", version_labels)
                idx = version_labels.index(chosen_label)
                chosen_version = all_versions[idx]
                st.session_state["version_id"] = chosen_version["id"]

                st.markdown(f"**Version Timestamp:** {chosen_version['version_timestamp']}")
                st.markdown(f"**Name:** {chosen_version['patient_name']} | **Age:** {chosen_version['age']} | **Gender:** {chosen_version['gender']} | **Contact:** {chosen_version['contact_number']}")
                st.markdown(f"**UHID/Reg No:** {chosen_version.get('uhid','')} | **Guardian:** {chosen_version.get('guardian_name','')} | **Consultant Doctor:** {chosen_version.get('consultant_doctor','')}")
                st.markdown(f"**Address:** {chosen_version.get('address','')} | **Queue No:** {chosen_version.get('queue_no','')} | **Registration Date:** {chosen_version.get('registration_date','')}")
                st.markdown(f"**Chief Complaint:** {chosen_version.get('chief_complaint','')}")
                st.markdown(f"**HPI:** {chosen_version.get('history_of_presenting_illness','')}")
                st.markdown(f"**Past History:** {chosen_version.get('past_history','')}")
                st.markdown(f"**Personal History:** {chosen_version.get('personal_history','')}")
                st.markdown(f"**Family History:** {chosen_version.get('family_history','')}")
                if chosen_version.get('obg_history'):
                    st.markdown(f"**OBG History:** {chosen_version['obg_history']}")
                st.markdown(f"**Medical Advice:** {chosen_version.get('medical_advice','None')}")
                st.markdown(f"**Case Summary:** {chosen_version.get('case_summary','None')}")
                st.markdown(f"**Final Diagnosis:** {chosen_version.get('final_diagnosis','') or '(None)'}")
                st.markdown(f"**Final Tests:** {chosen_version.get('final_tests','') or '(None)'}")
                st.markdown(f"**Final Treatment Plan:** {chosen_version.get('final_treatment_plan','') or '(None)'}")

                if chosen_version.get("lab_report_url"):
                    pdfv = generate_presigned_url(chosen_version["lab_report_url"])
                    if pdfv:
                        st.markdown(f"**Lab Report (Version):** [Open PDF]({pdfv})")

                if chosen_version.get("medical_imaging_url"):
                    imgv = generate_presigned_url(chosen_version["medical_imaging_url"])
                    if imgv:
                        if chosen_version["medical_imaging_url"].lower().endswith((".png",".jpg",".jpeg")):
                            st.image(imgv, use_column_width=True)
                        else:
                            st.write(f"[Imaging File (Version)]({imgv})")

                if chosen_version.get("previous_prescription_url"):
                    presv = generate_presigned_url(chosen_version["previous_prescription_url"])
                    if presv:
                        if chosen_version["previous_prescription_url"].lower().endswith(".pdf"):
                            st.markdown(f"**Prescription (PDF) [Version]:** [Open PDF]({presv})")
                        else:
                            st.image(presv, use_column_width=True)

                # If there's already medical advice, show it
                existing_advice = chosen_version.get("medical_advice", "")
                if existing_advice.strip():
                    st.info("Medical advice already exists for this version.")
                else:
                    if st.button("Get Medical Advice for This Version", key=f"gpt_version_{chosen_version['id']}"):
                        with st.spinner("Generating advice for this version..."):
                            advice_text = get_medical_advice(
                                department=chosen_version["department"],
                                chief_complaint=chosen_version.get("chief_complaint",""),
                                history_presenting_illness=chosen_version.get("history_of_presenting_illness",""),
                                past_history=chosen_version.get("past_history",""),
                                personal_history=chosen_version.get("personal_history",""),
                                family_history=chosen_version.get("family_history",""),
                                obg_history=chosen_version.get("obg_history",""),
                                age=chosen_version.get("age",0),
                                gender=chosen_version.get("gender","")
                            )

                        cs_section = parse_section(advice_text, "Case Summary")
                        advice_no_cs = remove_section(advice_text, "Case Summary")

                        update_patient_medical_advice(
                            version_id=chosen_version["id"],
                            advice_text=advice_no_cs,
                            case_summary=cs_section
                        )

                        st.success("Medical advice updated in this version’s record!")
                        refreshed = get_version_by_id(chosen_version["id"])
                        if refreshed:
                            chosen_version = refreshed

                        st.write("---")
                        st.markdown(f"**Updated Medical Advice:** {chosen_version.get('medical_advice','None')}")
                        st.markdown(f"**Case Summary:** {chosen_version.get('case_summary','None')}")

                # Show final choices if there's some advice
                current_advice = chosen_version.get("medical_advice", "")
                if current_advice.strip():
                    st.write("---")
                    st.subheader("Select Your Final Choices Based on the Advice")

                    diag_section = parse_section(current_advice, "Most Likely Diagnosis")
                    other_diag_section = parse_section(current_advice, "Other Possible Diagnoses")
                    tests_section = parse_section(current_advice, "Suggested Tests")
                    treat_section = parse_section(current_advice, "Suggested Treatment Plan")

                    most_likely_items = extract_bullet_items(diag_section)
                    other_diag_list = extract_bullet_items(other_diag_section)
                    tests_list = extract_bullet_items(tests_section)
                    treat_list = extract_bullet_items(treat_section)

                    with st.form(f"final_choices_form_{chosen_version['id']}"):
                        final_diagnosis_radio = None
                        if most_likely_items:
                            st.markdown("**Most Likely Diagnosis (choose one)**")
                            final_diagnosis_radio = st.radio(
                                "Likely Diagnosis",
                                options=most_likely_items,
                                key=f"rad_{chosen_version['id']}"
                            )

                        selected_other_diag = None
                        if other_diag_list:
                            st.markdown("**Other Possible Diagnoses**")
                            selected_other_diag = st.radio(
                                "Other Diagnoses",
                                options=["(None)"] + other_diag_list,
                                key=f"othdiag_{chosen_version['id']}"
                            )

                        selected_tests = []
                        if tests_list:
                            st.markdown("**Suggested Tests (check all that apply)**")
                            for idx, test in enumerate(tests_list):
                                cb_key = f"chk_test_{chosen_version['id']}_{idx}"
                                if st.checkbox(test, key=cb_key):
                                    selected_tests.append(test)

                        selected_treats = []
                        if treat_list:
                            st.markdown("**Suggested Treatment Plan (check all that apply)**")
                            for idx, treat_item in enumerate(treat_list):
                                cb_key = f"chk_treat_{chosen_version['id']}_{idx}"
                                if st.checkbox(treat_item, key=cb_key):
                                    selected_treats.append(treat_item)

                        final_dx = final_diagnosis_radio if final_diagnosis_radio else ""
                        if selected_other_diag and selected_other_diag != "(None)":
                            final_dx = selected_other_diag

                        final_tests_str = ", ".join(selected_tests)
                        final_treatment_str = ", ".join(selected_treats)

                        submit_btn = st.form_submit_button("Save Selections for This Version")
                        if submit_btn:
                            pid = chosen_version["patient_id"]
                            try:
                                update_patient_final_choices(
                                    patient_id=pid,
                                    final_diagnosis=final_dx,
                                    final_tests=final_tests_str,
                                    final_treatment_plan=final_treatment_str,
                                    case_summary=chosen_version.get('case_summary','')
                                )
                                st.success("Doctor's Final Choices & Case Summary Saved Successfully!")
                            except Exception as e:
                                st.error(f"DB update error: {e}")

        # Editing flow
        if st.session_state["edit_id"]:
            pat_id_edt = st.session_state["edit_id"]
            st.write("---")
            st.subheader(f"Editing Patient ID: {pat_id_edt}")

            query_latest = text("SELECT * FROM patient_info WHERE id=:pid")
            with engine.connect() as conn:
                latest_info = conn.execute(query_latest, {'pid': pat_id_edt}).mappings().first()

            if not latest_info:
                st.warning("No record found in patient_info for this ID.")
            else:
                new_name = st.text_input("Name", value=latest_info['patient_name'])
                new_age = st.number_input("Age", min_value=1, max_value=120, value=latest_info['age'])
                gender_opts = ["Male", "Female", "Other"]
                try:
                    gidx = gender_opts.index(latest_info['gender'])
                except:
                    gidx = 0
                new_gender = st.selectbox("Gender", gender_opts, index=gidx)
                new_contact = st.text_input("Contact Number", value=latest_info['contact_number'])

                dept_list = list(contexts.keys())
                try:
                    didx = dept_list.index(latest_info['department'])
                except:
                    didx = 0
                new_dept = st.selectbox("Department", dept_list, index=didx)

                new_uhid = st.text_input("UHID/Reg No", value=latest_info.get('uhid',''))
                new_guardian = st.text_input("Guardian Name", value=latest_info.get('guardian_name',''))
                new_consultant = st.text_input("Consultant Doctor", value=latest_info.get('consultant_doctor',''))
                new_address = st.text_input("Address", value=latest_info.get('address',''))
                new_queue = st.text_input("Queue No", value=latest_info.get('queue_no',''))

                cur_reg_dt = latest_info.get('registration_date')
                try:
                    if isinstance(cur_reg_dt, datetime):
                        cur_reg_dt = cur_reg_dt.date()
                    elif isinstance(cur_reg_dt, str):
                        cur_reg_dt = datetime.strptime(cur_reg_dt, "%Y-%m-%d").date()
                except:
                    cur_reg_dt = date.today()

                new_reg_date = st.date_input("Registration Date", value=cur_reg_dt or date.today())

                new_chief = st.text_input("Chief Complaint", value=latest_info.get('chief_complaint',''))
                new_hpi = st.text_area("History of Presenting Illness", value=latest_info.get('history_of_presenting_illness',''))
                new_past = st.text_area("Past History", value=latest_info.get('past_history',''))
                new_pers = st.text_area("Personal History", value=latest_info.get('personal_history',''))
                new_fam = st.text_area("Family History", value=latest_info.get('family_history',''))

                new_obg = ""
                if new_dept == "Gynecology":
                    new_obg = st.text_area("OBG History", value=latest_info.get('obg_history',''))

                st.write("Optionally upload new files to be included in the new version:")

                new_lab_file = st.file_uploader("New Lab Report (PDF)", type=["pdf"], key=f"lab_{pat_id_edt}")
                updated_lab_url = None
                if new_lab_file:
                    pdfb = new_lab_file.read()
                    updated_lab_url = upload_to_s3(pdfb, new_lab_file.name)
                    st.info("New Lab PDF uploaded.")

                new_img_file = st.file_uploader("New Imaging (PNG/JPG)", type=["png","jpg","jpeg"], key=f"img_{pat_id_edt}")
                updated_img_url = None
                if new_img_file:
                    rawb = new_img_file.read()
                    dsb = downsample_image(rawb)
                    updated_img_url = upload_to_s3(dsb, new_img_file.name)
                    st.image(dsb, caption="Updated Imaging Preview")

                new_pres_file = st.file_uploader("New Prescription (PDF/PNG/JPG)", type=["pdf","png","jpg","jpeg"], key=f"pres_{pat_id_edt}")
                updated_presc_url = None
                if new_pres_file:
                    pb = new_pres_file.read()
                    updated_presc_url = upload_to_s3(pb, new_pres_file.name)
                    st.info("New prescription uploaded.")

                if st.button("Save New Version", key=f"save_{pat_id_edt}"):
                    try:
                        final_lab_url = updated_lab_url or latest_info.get("lab_report_url")
                        final_img_url = updated_img_url or latest_info.get("medical_imaging_url")
                        final_pres_url = updated_presc_url or latest_info.get("previous_prescription_url")

                        update_patient_info(
                            patient_id=pat_id_edt,
                            name=new_name,
                            age=new_age,
                            gender=new_gender,
                            contact_number=new_contact,
                            department=new_dept,
                            chief_complaint=new_chief,
                            history_presenting_illness=new_hpi,
                            past_history=new_past,
                            personal_history=new_pers,
                            family_history=new_fam,
                            obg_history=new_obg,
                            lab_report_url=final_lab_url,
                            medical_imaging_url=final_img_url,
                            previous_prescription_url=final_pres_url,
                            uhid=new_uhid,
                            guardian_name=new_guardian,
                            consultant_doctor=new_consultant,
                            address=new_address,
                            queue_no=new_queue,
                            registration_date=new_reg_date
                        )
                        st.success("New version created; 'latest' record updated successfully.")
                    except Exception as e:
                        st.error(f"Update failed: {e}")

# ---------------------------------------
# 4) PRESCRIPTION WRITING TAB
# ---------------------------------------
elif tab_selection == "Prescription Writing":
    st.header("Prescription Writing")

    patient_data = st.session_state.get("patient_data", {})
    if not patient_data.get("name"):
        st.warning("Please fill out 'Patient Information' first.")
        st.stop()

    if "prescription" not in st.session_state:
        st.session_state["prescription"] = {
            "diagnosis": "",
            "drugs": [],
            "tests": [],
            "follow_up": {"when_to_return": "", "what_to_monitor": ""}
        }

    prescription = st.session_state["prescription"]

    if st.button("Generate Prescription"):
        with st.spinner("Generating prescription..."):
            gpt_prescription = generate_prescription(
                diagnosis=st.session_state.get("final_diagnosis", ""),
                tests=st.session_state.get("selected_tests", []),
                treatments=st.session_state.get("selected_treats", []),
                patient_info={
                    "name": patient_data.get("name", ""),
                    "age": patient_data.get("age", ""),
                    "gender": patient_data.get("gender", "")
                }
            )
            st.session_state["prescription"] = gpt_prescription
            st.success("Prescription generated successfully!")
            prescription = gpt_prescription

    # Let the user manually edit JSON fields
    st.subheader("Prescription Fields")

    diagnosis_input = st.text_input("Diagnosis", value=prescription.get("diagnosis",""))

    drugs_json_str = json.dumps(prescription.get("drugs", []), indent=2)
    drugs_text = st.text_area("Drugs (JSON array)", value=drugs_json_str, height=150)

    tests_json_str = json.dumps(prescription.get("tests", []), indent=2)
    tests_text = st.text_area("Recommended Tests (JSON array)", value=tests_json_str, height=150)

    followup_json_str = json.dumps(prescription.get("follow_up", {}), indent=2)
    followup_text = st.text_area("Follow-Up Instructions (JSON object)", value=followup_json_str, height=150)

    def create_pdf(data):
        """Build a PDF with prescription details."""
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=30,
            leftMargin=30,
            topMargin=30,
            bottomMargin=30,
        )

        styles = getSampleStyleSheet()
        center_style = ParagraphStyle(
            'CenterStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            alignment=1,  # center
            textColor=black,
        )

        flowables = []
        # Title
        flowables.append(Paragraph(f"<b>Prescription for:</b> {patient_data.get('name','Unknown')}", center_style))
        flowables.append(Spacer(1, 12))

        # Diagnosis
        flowables.append(Paragraph(f"<b>Diagnosis:</b> {data.get('diagnosis','')}", center_style))
        flowables.append(Spacer(1, 8))

        # Drugs
        flowables.append(Paragraph("<b>Drugs & Dosages:</b>", center_style))
        for d in data.get("drugs", []):
            dname = d.get("name","N/A")
            dstrength = d.get("strength","N/A")
            dfreq = d.get("frequency","N/A")
            ddur = d.get("duration","N/A")
            line = f"- {dname}, {dstrength}, {dfreq}, {ddur}"
            flowables.append(Paragraph(line, center_style))
        flowables.append(Spacer(1, 8))

        # Tests
        flowables.append(Paragraph("<b>Recommended Tests:</b>", center_style))
        for t in data.get("tests", []):
            tname = t.get("name","N/A")
            tpurpose = t.get("purpose","")
            line = f"- {tname} (Purpose: {tpurpose})"
            flowables.append(Paragraph(line, center_style))
        flowables.append(Spacer(1, 8))

        # Follow-up
        fup = data.get("follow_up", {})
        ret_when = fup.get("when_to_return", "N/A")
        monitor_what = fup.get("what_to_monitor", "N/A")
        flowables.append(Paragraph("<b>Follow-Up Instructions:</b>", center_style))
        flowables.append(Paragraph(f"Next Visit: {ret_when} | Monitor: {monitor_what}", center_style))

        flowables.append(Spacer(1, 12))
        doc.build(flowables)
        pdf_value = pdf_buffer.getvalue()
        pdf_buffer.close()
        return pdf_value

    if st.button("Save Prescription"):
        try:
            updated_drugs = json.loads(drugs_text)
            updated_tests = json.loads(tests_text)
            updated_followup = json.loads(followup_text)
        except Exception as e:
            st.error(f"Invalid JSON in one of the fields: {e}")
            st.stop()

        new_prescription = {
            "diagnosis": diagnosis_input.strip(),
            "drugs": updated_drugs,
            "tests": updated_tests,
            "follow_up": updated_followup
        }
        st.session_state["prescription"] = new_prescription

        pdf_data = create_pdf(new_prescription)
        try:
            file_name = f"Prescription_{patient_data.get('name', 'Unknown')}_{int(pytime.time())}.pdf"
            presc_url = upload_to_s3(pdf_data, file_name)

            # Optionally store in DB
            if "id" in patient_data:
                update_query = text("""
                    UPDATE patient_info
                    SET previous_prescription_url = :file_url
                    WHERE id = :pid
                """)
                with engine.begin() as conn:
                    conn.execute(update_query, {
                        'file_url': presc_url,
                        'pid': patient_data["id"]
                    })

            st.success(f"Prescription PDF uploaded successfully! URL: {presc_url}")

            st.download_button(
                label="Download Prescription PDF",
                data=pdf_data,
                file_name=f"Prescription_{patient_data.get('name', 'Unknown')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error saving prescription: {e}")
