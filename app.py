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
client = OpenAI(api_key="sk-proj-aA4in0l2WCEkJXq4yeHAT3BlbkFJmwOhRnH8ypgJpolet2Nb")

# ---------------------------------------
#    LOGGING AND ENVIRONMENT SETUP
# ---------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - [%(levelname)s] - %(filename)s.%(funcName)s(%(lineno)d) - %(message)s',
    handlers=[logging.FileHandler("medmitra.log", mode='a'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

load_dotenv()

DATABASE_URI = os.getenv("DATABASE_URI", "postgresql://user:pass@localhost/dbname")
if not DATABASE_URI:
    logger.warning("DATABASE_URI not set.")
engine = create_engine(DATABASE_URI, echo=False)

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "YOUR_AWS_ACCESS_KEY")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "YOUR_AWS_SECRET")
aws_region = os.getenv("AWS_REGION", "us-east-1")
bucket_name = os.getenv("AWS_BUCKET_NAME", "your-s3-bucket")

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
    with pdfplumber.open(pdf_file) as pdf:
        all_text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text += page_text + "\n"
    return all_text

# ---------------------------------------
#        DATABASE FUNCTIONS
# ---------------------------------------
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

def update_patient_info(
    patient_id, name, age, gender, contact_number, department,
    chief_complaint, history_presenting_illness, past_history,
    personal_history, family_history, obg_history=None,
    lab_report_url=None, medical_imaging_url=None, previous_prescription_url=None
):
    query = text("""
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
            previous_prescription_url = :prescrip_url
        WHERE id = :pid
    """)
    with engine.begin() as conn:
        conn.execute(query, {
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
            'pid': patient_id
        })

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

# ---------------------------------------
#  GPT-LIKE FUNCTIONS (STUB EXAMPLES)
# ---------------------------------------
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

def parse_section(full_text, section_name):
    pattern = rf"\*\*{section_name}\*\*\s*\n?(.*?)(?=\n\*\*|$)"
    match = re.search(pattern, full_text, flags=re.IGNORECASE|re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def remove_section(full_text, section_name):
    pattern = rf"\*\*{section_name}\*\*\s*\n?(.*?)(?=\n\*\*|$)"
    return re.sub(pattern, "", full_text, flags=re.IGNORECASE|re.DOTALL).strip()

def extract_bullet_items(section_text):
    items = []
    for line in section_text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            items.append(line[2:].strip())
    return items

# ---------------------------------------
#          STREAMLIT APP LAYOUT
# ---------------------------------------
st.title("MedMitra AI")

tab_selection = st.sidebar.radio(
    "Navigation",
    [
        "Patient Information",
        "Diagnosis, Prognosis & Treatment",
        # Removed "Follow-Up Questions" for brevity
        "Search Patient Records"
    ]
)

department = st.sidebar.selectbox("Select Department", list(contexts.keys()))

# Session placeholders for ephemeral data
if "patient_data" not in st.session_state:
    st.session_state["patient_data"] = {}
if "gpt_advice_text" not in st.session_state:
    st.session_state["gpt_advice_text"] = ""
if "gpt_case_summary" not in st.session_state:
    st.session_state["gpt_case_summary"] = ""
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = ""
if "suggested_questions" not in st.session_state:
    st.session_state["suggested_questions"] = ""
if "search_results" not in st.session_state:
    st.session_state["search_results"] = []
if "edit_id" not in st.session_state:
    st.session_state["edit_id"] = None

# ---------------------------------------
# 1) PATIENT INFORMATION TAB
# ---------------------------------------
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

    st.write("Optional: Upload Lab Report (PDF), Medical Imaging (PNG/JPG), or Prescription (PDF/PNG/JPG).")

    # Lab Report
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

    # Medical Imaging
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

    # Previous Prescription
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
                st.success(f"Patient info saved successfully (ID: {new_id}).")
            except Exception as e:
                st.error(f"Failed to save patient info: {e}")

# ---------------------------------------
# 2) DIAGNOSIS, PROGNOSIS & TREATMENT
# ---------------------------------------
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

# ---------------------------------------
# 4) SEARCH PATIENT RECORDS TAB
# ---------------------------------------
elif tab_selection == "Search Patient Records":
    st.header("Search Patient Records")

    s_name = st.text_input("Search by Name").strip()
    s_age = st.number_input("Search by Age", min_value=0, value=0)
    s_gender = st.selectbox("Search by Gender", ["Select Gender", "Male", "Female", "Other"])
    s_contact = st.text_input("Search by Contact Number").strip()

    if st.button("Search"):
        found_records = search_patients(
            name=s_name,
            age=s_age if s_age>0 else None,
            gender=s_gender if s_gender!="Select Gender" else None,
            contact=s_contact
        )
        if found_records:
            st.success(f"Found {len(found_records)} record(s).")
            st.session_state["search_results"] = found_records
            st.session_state["edit_id"] = None
        else:
            st.warning("No matching records found.")
            st.session_state["search_results"] = []
            st.session_state["edit_id"] = None

    if st.session_state["search_results"]:
        for rec in st.session_state["search_results"]:
            st.write("---")
            # Show main info + entire patient history
            st.write(f"**ID:** {rec['id']} | **Name:** {rec['patient_name']} | **Age:** {rec['age']} | **Gender:** {rec['gender']}")
            st.write(f"**Contact:** {rec['contact_number']} | **Department:** {rec['department']}")

            # Show all textual histories
            st.markdown(f"**Chief Complaint:** {rec['chief_complaint']}")
            st.markdown(f"**History of Presenting Illness:** {rec['history_of_presenting_illness']}")
            st.markdown(f"**Past History:** {rec['past_history']}")
            st.markdown(f"**Personal History:** {rec['personal_history']}")
            st.markdown(f"**Family History:** {rec['family_history']}")
            if rec.get('obg_history'):
                st.markdown(f"**OBG History:** {rec['obg_history']}")

            # Doctor-chosen final items
            dx = rec.get("final_diagnosis", "") or "(No final diagnosis)"
            tests = rec.get("final_tests","") or "(No tests chosen)"
            treat = rec.get("final_treatment_plan","") or "(No treatment plan)"
            st.markdown(f"**Final Diagnosis (Doc Selected):** {dx}")
            st.markdown(f"**Final Tests (Doc Selected):** {tests}")
            st.markdown(f"**Final Treatment Plan (Doc Selected):** {treat}")

            med_adv = rec.get("medical_advice","") or "No medical advice stored."
            st.markdown(f"**Medical Advice:** {med_adv}")

            csum = rec.get("case_summary","") or "(No case summary)"
            st.markdown(f"**Case Summary:** {csum}")

            # Potential file links
            if rec.get("lab_report_url"):
                pdf_link = generate_presigned_url(rec["lab_report_url"])
                if pdf_link:
                    st.markdown(f"**Lab Report:** [Open PDF]({pdf_link})")

            if rec.get("medical_imaging_url"):
                img_link = generate_presigned_url(rec["medical_imaging_url"])
                if img_link:
                    # If it's an image
                    if rec["medical_imaging_url"].lower().endswith((".png",".jpg",".jpeg")):
                        st.image(img_link, use_column_width=True)
                    else:
                        st.write(f"[Imaging File]({img_link})")

            if rec.get("previous_prescription_url"):
                pres_link = generate_presigned_url(rec["previous_prescription_url"])
                if pres_link:
                    if rec["previous_prescription_url"].lower().endswith(".pdf"):
                        st.markdown(f"**Previous Prescription (PDF):** [Open PDF]({pres_link})")
                    else:
                        st.image(pres_link, use_column_width=True)

            # Edit button
            if st.button(f"Edit Patient #{rec['id']}", key=f"edit_btn_{rec['id']}"):
                st.session_state["edit_id"] = rec["id"]

        # Inline edit form for whichever record is selected
        if st.session_state["edit_id"]:
            st.write("---")
            st.subheader(f"Editing Patient ID: {st.session_state['edit_id']}")

            edit_record = next((r for r in st.session_state["search_results"]
                                if r["id"] == st.session_state["edit_id"]), None)
            if edit_record:
                new_name = st.text_input("Name", value=edit_record['patient_name'])
                new_age = st.number_input("Age", min_value=1, max_value=120, value=edit_record['age'])
                gender_opts = ["Male", "Female", "Other"]
                try:
                    gender_idx = gender_opts.index(edit_record['gender'])
                except:
                    gender_idx = 0
                new_gender = st.selectbox("Gender", gender_opts, index=gender_idx)
                new_contact = st.text_input("Contact Number", value=edit_record['contact_number'])

                dept_list = list(contexts.keys())
                try:
                    dept_idx = dept_list.index(edit_record['department'])
                except:
                    dept_idx = 0
                new_dept = st.selectbox("Department", dept_list, index=dept_idx)

                new_chief = st.text_input("Chief Complaint", value=edit_record['chief_complaint'])
                new_hpi = st.text_area("History of Presenting Illness", value=edit_record['history_of_presenting_illness'])
                new_past = st.text_area("Past History", value=edit_record['past_history'])
                new_pers = st.text_area("Personal History", value=edit_record['personal_history'])
                new_fam = st.text_area("Family History", value=edit_record['family_history'])

                new_obg = ""
                if new_dept == "Gynecology":
                    new_obg = st.text_area("OBG History", value=edit_record.get('obg_history',''))

                st.write("Optionally upload new files to replace existing ones:")

                # Lab
                new_lab_file = st.file_uploader("New Lab Report (PDF)", type=["pdf"], key=f"lab_{edit_record['id']}")
                updated_lab_url = edit_record.get("lab_report_url", "")
                if new_lab_file:
                    pdfb = new_lab_file.read()
                    updated_lab_url = upload_to_s3(pdfb, new_lab_file.name)
                    st.info("New lab PDF uploaded.")

                # Imaging
                new_img_file = st.file_uploader("New Imaging (PNG/JPG)", type=["png","jpg","jpeg"], key=f"img_{edit_record['id']}")
                updated_img_url = edit_record.get("medical_imaging_url", "")
                if new_img_file:
                    rawb = new_img_file.read()
                    dsb = downsample_image(rawb)
                    updated_img_url = upload_to_s3(dsb, new_img_file.name)
                    st.image(dsb, caption="Updated Imaging Preview")

                # Prescription
                new_pres_file = st.file_uploader("New Prescription (PDF/PNG/JPG)", type=["pdf","png","jpg","jpeg"], key=f"pres_{edit_record['id']}")
                updated_presc_url = edit_record.get("previous_prescription_url", "")
                if new_pres_file:
                    pb = new_pres_file.read()
                    updated_presc_url = upload_to_s3(pb, new_pres_file.name)
                    st.info("New prescription uploaded.")

                if st.button(f"Save Changes for ID {edit_record['id']}", key=f"save_{edit_record['id']}"):
                    try:
                        update_patient_info(
                            patient_id=edit_record['id'],
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
                            lab_report_url=updated_lab_url,
                            medical_imaging_url=updated_img_url,
                            previous_prescription_url=updated_presc_url
                        )
                        st.success("Record updated successfully.")
                    except Exception as e:
                        st.error(f"Update failed: {e}")
