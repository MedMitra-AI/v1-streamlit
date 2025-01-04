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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-fallback")
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

def encode_image(file_bytes):
    """
    Convert image bytes to base64 string (if needed for GPT).
    """
    return base64.b64encode(file_bytes).decode("utf-8")

def extract_text_from_pdf_bytes(pdf_data: bytes) -> str:
    """
    Use pdfplumber on an in-memory bytes object instead of a file pointer.
    This ensures we don't lose the file pointer or corrupt data.
    """
    with io.BytesIO(pdf_data) as mem:
        with pdfplumber.open(mem) as pdf:
            all_text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text += page_text + "\n"
    return all_text


# -------------------- GPT PROMPT FUNCTION -------------------- #
def get_medical_advice_descriptive(
    department,
    chief_complaint,
    history_presenting_illness,
    past_history,
    personal_history,
    family_history,
    obg_history="",
    image_data=None,
    lab_report_text=None
):
    logger.info(f"Generating medical advice for department: {department}, complaint: {chief_complaint}")
    context = contexts.get(department, "")

    prompt_text = f"""
You are a helpful medical assistant specialized in medical diagnosis, prognosis, and treatment planning.
Include brand names of recommended drugs/dosages in the treatment plan. Avoid disclaimers about specialist follow-ups.

Patient Data:
Department: {department}
Context: {context}
Chief Complaint: {chief_complaint}
History of Presenting Illness: {history_presenting_illness}
Past History: {past_history}
Personal History: {personal_history}
Family History: {family_history}
OBG History: {obg_history}

Lab Report Analysis (if any): {lab_report_text}
Medical Image Analysis (if any): {image_data}

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
- ...
**Case Summary**
(Short concluding summary)

Please ensure each heading is in this exact format to allow parsing.
"""

    messages = [
        {"role": "system", "content": "You are ChatGPT, a helpful medical assistant."},
        {"role": "user", "content": prompt_text}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1800,
            temperature=0.7
        )
        logger.debug("Received descriptive text from GPT.")
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating medical advice: {e}")
        return ""


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
    lab_report_text = None
    lab_report_url = None
    if lab_report_file:
        # read entire file
        pdf_bytes = lab_report_file.read()
        # extract text from memory (pdfplumber)
        lab_report_text = extract_text_from_pdf_bytes(pdf_bytes)

        if lab_report_text:
            st.write("Extracted Text from PDF:")
            st.text_area("Lab Report Data", lab_report_text, height=200)
        else:
            st.error("Could not extract text from the PDF (or no text found).")

        # now upload exactly those same bytes
        lab_report_url = upload_to_s3(pdf_bytes, lab_report_file.name)

    # Medical Imaging
    image_file = st.file_uploader("Upload Medical Imaging (PNG/JPG)", type=["png", "jpg", "jpeg"])
    image_data = None
    medical_imaging_url = None
    if image_file:
        raw_bytes = image_file.read()
        # Downsample
        smaller_img_bytes = downsample_image(raw_bytes, max_size=(500, 500), quality=50)
        # For GPT usage
        image_data = encode_image(smaller_img_bytes)
        st.image(smaller_img_bytes, caption="Downsampled Medical Image", use_column_width=True)
        # Upload
        medical_imaging_url = upload_to_s3(smaller_img_bytes, image_file.name)

    # Previous Prescription
    prescription_file = st.file_uploader("Upload Previous Prescription (PDF/PNG/JPG)", type=["pdf", "png", "jpg", "jpeg"])
    previous_prescription_url = None
    if prescription_file:
        prescrip_bytes = prescription_file.read()

        # check if pdf or image
        if prescription_file.type == "application/pdf":
            previous_prescription_url = upload_to_s3(prescrip_bytes, prescription_file.name)
            st.success("Prescription PDF uploaded successfully.")
        else:
            # It's an image
            compressed_prescrip = downsample_image(prescrip_bytes, max_size=(500, 500), quality=50)
            st.image(compressed_prescrip, caption="Downsampled Previous Prescription", use_column_width=True)
            previous_prescription_url = upload_to_s3(compressed_prescrip, prescription_file.name)
            st.success("Prescription image uploaded successfully.")

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
                "image_data": image_data,
                "medical_imaging_url": medical_imaging_url,
                "previous_prescription_url": previous_prescription_url
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
                        previous_prescription_url=previous_prescription_url
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
            with st.spinner("Generating advice..."):
                advice_text = get_medical_advice_descriptive(
                    department=patient_data["department"],
                    chief_complaint=patient_data["chief_complaint"],
                    history_presenting_illness=patient_data["history_presenting_illness"],
                    past_history=patient_data["past_history"],
                    personal_history=patient_data["personal_history"],
                    family_history=patient_data["family_history"],
                    obg_history=patient_data.get("obg_history", ""),
                    image_data=patient_data.get("image_data", ""),
                    lab_report_text=patient_data.get("lab_report_text", "")
                )

            # Extract "Case Summary"
            case_summary_section = parse_section(advice_text, "Case Summary")

            # Remove "Case Summary" from main text
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
            most_likely_items = extract_bullet_items(diag_section)
            final_diagnosis_radio = None
            if most_likely_items:
                st.markdown("**Most Likely Diagnosis (choose one)**")
                final_diagnosis_radio = st.radio("", options=most_likely_items)

            other_diag_section = parse_section(gpt_text, "Other Possible Diagnoses")
            other_diag_list = extract_bullet_items(other_diag_section)
            selected_other_diag = None
            if other_diag_list:
                st.markdown("**Other Possible Diagnoses (pick one if desired)**")
                selected_other_diag = st.radio("", options=["(None)"] + other_diag_list)

            tests_section = parse_section(gpt_text, "Suggested Tests")
            tests_list = extract_bullet_items(tests_section)
            selected_tests = []
            if tests_list:
                st.markdown("**Suggested Tests (check all that apply)**")
                for test in tests_list:
                    checked = st.checkbox(test, value=False)
                    if checked:
                        selected_tests.append(test)

            treat_section = parse_section(gpt_text, "Suggested Treatment Plan")
            treat_list = extract_bullet_items(treat_section)
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

    if "allow_follow_up" not in st.session_state:
        st.session_state["allow_follow_up"] = False

    # Let’s allow follow-up if GPT text was generated
    if st.session_state.get("gpt_advice_text", ""):
        st.session_state["allow_follow_up"] = True

    if st.session_state["allow_follow_up"]:
        follow_up_question = st.text_input("Enter Follow-Up Question", key="follow_up_question")
        if st.button("Submit Follow-Up"):
            if follow_up_question:
                conversation_history = st.session_state.get("conversation_history", "")

                def handle_follow_up(convo_history, question):
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant specialized in medical diagnosis, prognosis, and treatment planning. "
                                "Include brand names and dosages, no disclaimers about specialist follow-ups."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Below is the previous conversation:\n\n{convo_history}\n\nFollow-up Question: {question}"
                        }
                    ]
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=messages
                        )
                        return response.choices[0].message.content.strip()
                    except Exception as e:
                        return f"Error from OpenAI: {e}"

                updated_response = handle_follow_up(conversation_history, follow_up_question)
                updated_conversation = (
                    conversation_history
                    + f"\n\nFollow-up Question: {follow_up_question}\nResponse: {updated_response}"
                )
                st.session_state["conversation_history"] = updated_conversation
                st.write(updated_response)
            else:
                st.error("Please enter a follow-up question.")
    else:
        st.write("No initial advice generated yet. Go to 'Diagnosis, Prognosis & Treatment' tab first.")


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
