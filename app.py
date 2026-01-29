"""
Certificate Verification Module - Rule-Based OCR System
========================================================
This module performs certificate verification using OCR and keyword matching.
No machine learning models are used - purely rule-based approach.

Author: Certificate Verification System
Date: 2026
"""

import os
import re
import shutil
import uuid
import json
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import cv2
import pytesseract
from PIL import Image
import numpy as np
from datetime import datetime, timedelta

try:
    import fitz  # PyMuPDF for PDF text extraction
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# All supported document formats (lowercase for validation)
SUPPORTED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png', 'tiff', 'tif', 'webp'}
# MIME types for validation (optional)
MIME_TO_EXTENSION = {
    'application/pdf': 'pdf',
    'image/jpeg': 'jpeg',
    'image/jpg': 'jpg',
    'image/png': 'png',
    'image/tiff': 'tiff',
    'image/webp': 'webp',
}
EXTENSION_TO_MIME = {
    'pdf': 'application/pdf',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'png': 'image/png',
    'tiff': 'image/tiff',
    'tif': 'image/tiff',
    'webp': 'image/webp',
}
app.config['ALLOWED_EXTENSIONS'] = SUPPORTED_EXTENSIONS
DEFAULT_ALLOWED_FILE_TYPES = ['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'webp']

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure Tesseract path (Windows - adjust if needed)
# For Windows, you may need to set the path manually:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Uncomment and adjust the path above if Tesseract is not in your system PATH

# ============================================================================
# CERTIFICATE KEYWORD DICTIONARY
# ============================================================================
# IMPORTANT:
# - PRIMARY keywords: determine certificate type (must match at least one)
# - SECONDARY keywords: supporting terms only (MUST NOT determine certificate type)
# This design fixes conflicts like "tahsildar" appearing in both Income/Community.
CERTIFICATE_KEYWORDS = {
    'Income Certificate': {
        'primary_keywords': [
            'income certificate',
            'family income',
            'annual income',
            'certificate of income',
        ],
        'secondary_keywords': [
            'tahsildar',
            'revenue department',
            'district collector',
        ],
    },
    'Community Certificate': {
        'primary_keywords': [
            'community certificate',
            'scheduled caste',
            'scheduled tribe',
            'backward class',
            'most backward class',
        ],
        'secondary_keywords': [
            'tahsildar',
            'revenue department',
        ],
    },
    'Educational Certificate (Anna University)': {
        'primary_keywords': [
            'anna university',
            'controller of examinations',
        ],
        'secondary_keywords': [
            'degree',
            'b.e',
            'b.tech',
            'chennai',
        ],
    },
    'Aadhaar Card': {
        'primary_keywords': [
            'uidai',
            'unique identification authority of india',
            'aadhaar',
        ],
        'secondary_keywords': [
            'government of india',
        ],
    },
    'Driving License': {
        'primary_keywords': [
            'driving licence',
            'driving license',
            'dl no',
        ],
        'secondary_keywords': [
            'transport department',
            'government of india',
            'valid till',
            'date of expiry',
            'expires on',
            'valid from',
            'date of expiry',
            'dl no.',
        ],
    },
    'Bonafide Certificate': {
        'primary_keywords': [
            'bonafide certificate',
        ],
        'secondary_keywords': [
            'this is to certify',
            'student of',
            'studying in',
            'academic year',
            'institution',
        ],
    },
}

# If multiple PRIMARY matches exist, use this priority order (highest first).
# NOTE: PAN is intentionally excluded (removed from project).
CERTIFICATE_PRIORITY = [
    'Income Certificate',
    'Community Certificate',
    'Educational Certificate (Anna University)',
    'Aadhaar Card',
    'Driving License',
    'Bonafide Certificate',
]

# Mapping: certificate type -> upload field/section ID (for reassignment and audit).
# Must match frontend CERTIFICATE_TYPES keys.
CERTIFICATE_TYPE_TO_FIELD_ID = {
    'Income Certificate': 'income',
    'Community Certificate': 'community',
    'Educational Certificate (Anna University)': 'educational',
    'Aadhaar Card': 'aadhaar',
    'Driving License': 'license',
    'Bonafide Certificate': 'bonafide',
}
AUDIT_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_reassignments.log')
FIELDS_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'certificate_fields.json')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_file_extension(filename):
    """Get lowercase extension without dot."""
    if '.' in filename:
        return filename.rsplit('.', 1)[1].lower()
    return ''


def allowed_file(filename, allowed_extensions=None):
    """
    Check if the uploaded file has an allowed extension.
    If allowed_extensions is provided (e.g. from field config), only those are accepted.
    Use 'all' or None in the list to mean all supported formats.
    """
    ext = get_file_extension(filename)
    if not ext:
        return False
    supported = SUPPORTED_EXTENSIONS
    if allowed_extensions is not None and allowed_extensions and 'all' not in [x.lower() for x in allowed_extensions]:
        allowed_set = {e.lower().lstrip('.') for e in allowed_extensions}
        return ext in allowed_set and ext in supported
    return ext in supported


def log_reassignment(from_category, to_category, filename):
    """Log certificate reassignment for audit purposes."""
    try:
        with open(AUDIT_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} | Reassignment | from {from_category} | to {to_category} | file {filename}\n")
    except Exception:
        pass


def log_dl_expiry_extraction(expiry_date_str):
    """Log extracted Driving License expiry date for debugging and audit."""
    try:
        with open(AUDIT_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} | DL_EXPIRY | extracted_date: {expiry_date_str}\n")
    except Exception:
        pass


def load_fields_config():
    """Load admin-defined certificate fields from JSON file."""
    if not os.path.exists(FIELDS_CONFIG_PATH):
        # Initialize with default fields
        # Default allowed file types: all document formats (can be overridden per field)
        default_allowed = ['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'webp']
        default_fields = [
            {
                'id': 'income',
                'field_name': 'Income Certificate',
                'expected_category': 'Income Certificate',
                'expiry_validation_required': True,
                'mandatory': True,
                'enabled': True,
                'allowed_file_types': default_allowed
            },
            {
                'id': 'community',
                'field_name': 'Community Certificate',
                'expected_category': 'Community Certificate',
                'expiry_validation_required': False,
                'mandatory': True,
                'enabled': True,
                'allowed_file_types': default_allowed
            },
            {
                'id': 'license',
                'field_name': 'Driving License',
                'expected_category': 'Driving License',
                'expiry_validation_required': True,
                'mandatory': False,
                'enabled': True,
                'allowed_file_types': default_allowed
            },
            {
                'id': 'aadhaar',
                'field_name': 'Aadhaar Card',
                'expected_category': 'Aadhaar Card',
                'expiry_validation_required': False,
                'mandatory': True,
                'enabled': True,
                'allowed_file_types': default_allowed
            },
            {
                'id': 'educational',
                'field_name': 'Educational Certificate (Anna University)',
                'expected_category': 'Educational Certificate (Anna University)',
                'expiry_validation_required': False,
                'mandatory': False,
                'enabled': True,
                'allowed_file_types': default_allowed
            },
            {
                'id': 'bonafide',
                'field_name': 'Bonafide Certificate',
                'expected_category': 'Bonafide Certificate',
                'expiry_validation_required': False,
                'mandatory': False,
                'enabled': True,
                'allowed_file_types': default_allowed
            }
        ]
        save_fields_config(default_fields)
        return default_fields
    
    try:
        with open(FIELDS_CONFIG_PATH, 'r', encoding='utf-8') as f:
            fields = json.load(f)
        # Ensure all fields have allowed_file_types (migration for old configs)
        updated = False
        for field in fields:
            if 'allowed_file_types' not in field or not field['allowed_file_types']:
                field['allowed_file_types'] = DEFAULT_ALLOWED_FILE_TYPES
                updated = True
        if updated:
            save_fields_config(fields)
        return fields
    except Exception as e:
        return []


def save_fields_config(fields):
    """Save admin-defined certificate fields to JSON file."""
    try:
        with open(FIELDS_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(fields, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def get_active_fields():
    """Get only enabled certificate fields."""
    all_fields = load_fields_config()
    return [f for f in all_fields if f.get('enabled', True)]


def get_field_by_id(field_id):
    """Get a specific field by its ID."""
    fields = load_fields_config()
    for field in fields:
        if field.get('id') == field_id:
            return field
    return None


def preprocess_image(image_path):
    """
    Preprocess the certificate image for better OCR accuracy.
    
    Steps:
    1. Read image
    2. Convert to grayscale
    3. Apply thresholding (binary)
    4. Optional: Noise removal
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better text extraction
    # This helps with varying lighting conditions
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Alternative: Simple thresholding (uncomment if adaptive doesn't work well)
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh


def extract_text_ocr(image_path):
    """
    Extract text from certificate image using Tesseract OCR.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Extracted text from the image
    """
    try:
        # Preprocess image
        processed_img = preprocess_image(image_path)
        
        # Perform OCR using pytesseract
        # Using PSM mode 6: Assume uniform block of text
        # Using PSM mode 3: Fully automatic page segmentation (default)
        custom_config = r'--oem 3 --psm 6'
        extracted_text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        # Clean up extracted text
        extracted_text = extracted_text.strip()
        
        return extracted_text
    
    except Exception as e:
        raise Exception(f"OCR extraction failed: {str(e)}")


def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF. Uses PyMuPDF: text layer first; if minimal text, render page to image and OCR.
    """
    if not HAS_PYMUPDF:
        raise Exception("PDF support requires PyMuPDF. Install with: pip install PyMuPDF")
    doc = fitz.open(pdf_path)
    try:
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text = (text or "").strip()
            if len(text) < 50:
                # Likely scanned: render to image and run OCR
                import tempfile
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
                try:
                    os.close(tmp_fd)
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    pix.save(tmp_path)
                    text = extract_text_ocr(tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
            text_parts.append(text or "")
        return "\n\n".join(text_parts).strip()
    finally:
        doc.close()


def extract_text_from_document(filepath, file_extension):
    """
    Normalize document to text: PDF -> PDF extraction; image -> image OCR.
    Returns (extracted_text, original_format).
    """
    ext = (file_extension or "").lower().lstrip(".")
    if ext == "pdf":
        text = extract_text_from_pdf(filepath)
        return text, "pdf"
    if ext in ("jpg", "jpeg", "png", "tiff", "tif", "webp"):
        if ext in ("tiff", "tif", "webp"):
            # PIL reads TIFF/WebP; run Tesseract on PIL Image
            pil_img = Image.open(filepath)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(pil_img, config=custom_config)
        else:
            text = extract_text_ocr(filepath)
        return (text or "").strip(), ext
    raise ValueError(f"Unsupported format: {ext}")


def extract_date_from_text(text, certificate_type):
    """
    Extract date from OCR text based on certificate type.
    
    PRIORITY LOGIC FOR INCOME CERTIFICATE:
    1. PRIORITY 1 (HIGHEST): Detect validity range (FROM → TO)
       - Extract both start_date and end_date
       - Return as 'validity_range' type
    2. PRIORITY 2: Detect single issue date
       - Extract issue date
       - Return as 'issue_date' type (will add +1 year later)
    
    Date Formats Supported:
    - DD/MM/YYYY
    - DD-MM-YYYY
    - YYYY-MM-DD
    - DD.MM.YYYY
    
    Args:
        text (str): Extracted OCR text
        certificate_type (str): Type of certificate (Income Certificate or Driving License)
        
    Returns:
        tuple: (date_info, date_string, date_type)
               - date_info: dict with 'start_date' and 'end_date' (for range) or single datetime (for single date) or None
               - date_string: Formatted date string(s) or None
               - date_type: 'validity_range' | 'issue_date' | 'expiry_date' | None
    """
    if not text or len(text.strip()) == 0:
        return None, None, None
    
    text_lower = text.lower()
    
    # Date regex patterns
    # Pattern 1: DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    date_pattern_1 = r'\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})\b'
    # Pattern 2: YYYY-MM-DD or YYYY/MM/DD
    date_pattern_2 = r'\b(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})\b'

    def _parse_ddmmyyyy(match_tuple):
        """Parse (day, month, year) safely into datetime."""
        d, m, y = match_tuple
        return datetime(int(y), int(m), int(d))

    def _parse_yyyymmdd(match_tuple):
        """Parse (year, month, day) safely into datetime."""
        y, m, d = match_tuple
        return datetime(int(y), int(m), int(d))
    
    # ========================================================================
    # INCOME CERTIFICATE: MANDATORY PROCESSING ORDER
    # ========================================================================
    # ABSOLUTE RULE: If an explicit validity period (DATE to DATE) exists,
    # use ONLY those dates. NEVER add +1 year. NEVER use issue date.
    # ========================================================================
    if certificate_type == 'Income Certificate':
        # --- STEP 1 (MANDATORY): SEARCH FOR VALIDITY RANGE FIRST ---
        # Use the mandated regex on the ENTIRE text before any other date logic.
        # This detects "06-01-2026 to 05-01-2027" even with OCR typos like
        # "Cectineate valleity pertod" (no keyword match needed for the regex).
        # MANDATORY REGEX: (\\d{2}[-/]\\d{2}[-/]\\d{4})\\s*(to|-)\\s*(\\d{2}[-/]\\d{2}[-/]\\d{4})
        VALIDITY_RANGE_REGEX = r'(\d{2}[-/]\d{2}[-/]\d{4})\s*(?:to|-)\s*(\d{2}[-/]\d{2}[-/]\d{4})'
        range_match = re.search(VALIDITY_RANGE_REGEX, text_lower)
        
        if range_match:
            try:
                date1_str = range_match.group(1).strip()
                date2_str = range_match.group(2).strip()
                # Parse DD-MM-YYYY or DD/MM/YYYY (first 3 groups: d, m, y)
                def _parse_dd_mm_yyyy(s):
                    parts = re.split(r'[-/]', s)
                    if len(parts) == 3:
                        d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
                        return datetime(y, m, d)
                    return None
                start_date = _parse_dd_mm_yyyy(date1_str)
                end_date = _parse_dd_mm_yyyy(date2_str)
                if start_date and end_date:
                    date_info = {'start_date': start_date, 'end_date': end_date}
                    date_string = f"{start_date.strftime('%d-%m-%Y')} to {end_date.strftime('%d-%m-%Y')}"
                    # STOP: Do not extract issue date. Do not run any other date logic.
                    return date_info, date_string, 'validity_range'
            except (ValueError, AttributeError, TypeError):
                pass
            # If parse failed, fall through to issue-date logic (do not treat as range)
        
        # --- STEP 2: ONLY IF RANGE NOT FOUND - Extract issue date ---
        # When validity range is detected above, we never reach here.
        lines = text.split('\n')
        issue_keywords = ['issued on', 'issue date', 'date of issue', 'dated']
        # Explicit ignore: dates near these must NOT be used for validity.
        ignore_keywords = [
            'digitally signed', 'digital signature', 'signed on',
            'signature', 'date:',
        ]
        
        def _find_issue_date_safely(lines, issue_kw, ignore_kw):
            for line in lines:
                line_lower = line.lower()
                if any(ig in line_lower for ig in ignore_kw):
                    continue
                if any(k in line_lower for k in issue_kw):
                    for m in re.finditer(date_pattern_1, line_lower):
                        try:
                            return _parse_ddmmyyyy(m.groups())
                        except ValueError:
                            continue
                    for m in re.finditer(date_pattern_2, line_lower):
                        try:
                            return _parse_yyyymmdd(m.groups())
                        except ValueError:
                            continue
            return None
        
        date_obj = _find_issue_date_safely(lines, issue_keywords, ignore_keywords)
        if date_obj:
            return date_obj, date_obj.strftime('%d-%m-%Y'), 'issue_date'
        
        # --- STEP 3: Fallback only when neither range nor issue date found ---
        for m in re.finditer(date_pattern_1, text_lower):
            try:
                obj = _parse_ddmmyyyy(m.groups())
                return obj, obj.strftime('%d-%m-%Y'), 'issue_date'
            except ValueError:
                continue
        for m in re.finditer(date_pattern_2, text_lower):
            try:
                obj = _parse_yyyymmdd(m.groups())
                return obj, obj.strftime('%d-%m-%Y'), 'issue_date'
            except ValueError:
                continue
        
        return None, None, None
        
    # ========================================================================
    # DRIVING LICENSE: Expiry date extraction (DigiLocker, NT/TR, legacy)
    # ========================================================================
    # Keywords (authoritative for validity): Date of Expiry, Expiry Date, Valid Till, Valid Upto.
    # IGNORE: Date of Birth, Issue Date, digitally signed date, PDF timestamp.
    # Rule: Current_Date <= Date_of_Expiry -> VALID; else EXPIRED.
    # ========================================================================
    elif certificate_type == 'Driving License':
        _fmt_d = '%d-%m-%Y'

        # Normalize OCR text: collapse multiple spaces/newlines for keyword+date proximity
        text_normalized = re.sub(r'\s+', ' ', (text or '').strip())
        text_lower_norm = text_normalized.lower()
        # Also keep line-based search for "Date of Expiry : 18-07-2046" on same line
        lines = [re.sub(r'\s+', ' ', line.strip()) for line in text.split('\n') if line.strip()]
        text_lower = text_lower_norm if text_normalized else text.lower()

        def _parse_ddmm(date_s):
            """Parse DD-MM-YYYY or DD/MM/YYYY to datetime. Returns None on failure."""
            if not date_s:
                return None
            parts = re.split(r'[-/]', date_s.strip())
            if len(parts) != 3:
                return None
            try:
                d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
                return datetime(y, m, d)
            except (ValueError, IndexError):
                return None

        # Date pattern: DD-MM-YYYY or DD/MM/YYYY (1 or 2 digits for d/m)
        date_pattern_dl = r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})'

        def _extract_expiry_by_keywords(where_lower, where_raw):
            """Search for expiry keywords and return first valid date. Ignores issue/signature dates."""
            # Order: Date of Expiry (DigiLocker), Expiry Date, Valid Till, Valid Upto, Valid Until
            patterns = [
                (r'date\s+of\s+expiry\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})', 'date of expiry'),
                (r'expiry\s+date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})', 'expiry date'),
                (r'valid\s+till\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})', 'valid till'),
                (r'valid\s+upto\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})', 'valid upto'),
                (r'valid\s+up\s+to\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})', 'valid up to'),
                (r'valid\s+until\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})', 'valid until'),
            ]
            for pat, name in patterns:
                m = re.search(pat, where_lower)
                if m:
                    dt = _parse_ddmm(m.group(1))
                    if dt:
                        return dt
            return None

        # 1) Try normalized full text (handles "Date of Expiry : 18-07-2046")
        expiry_date = _extract_expiry_by_keywords(text_lower_norm, text_normalized)

        # 2) Try line-by-line (handles multi-line or spaced formatting)
        if not expiry_date:
            for line in lines:
                line_low = line.lower()
                # Skip lines that are clearly not expiry (digitally signed, PDF, etc.)
                if 'digitally signed' in line_low or 'tcpdf' in line_low or 'it act' in line_low:
                    continue
                expiry_date = _extract_expiry_by_keywords(line_low, line)
                if expiry_date:
                    break

        # 3) Legacy patterns: Validity (NT), Valid Till, Validity (TR)
        PATTERN_NT = r'validity\s*\(\s*nt\s*\)\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})'
        PATTERN_TR = r'validity\s*\(\s*tr\s*\)\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})'
        PATTERN_VALID_TILL = r'valid\s*till\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})'

        nt_match = re.search(PATTERN_NT, text_lower)
        tr_match = re.search(PATTERN_TR, text_lower)
        vt_match = re.search(PATTERN_VALID_TILL, text_lower)

        validity_nt_str = None
        validity_tr_str = None
        valid_till_str = None
        nt_dt = _parse_ddmm(nt_match.group(1)) if nt_match else None
        if nt_dt:
            validity_nt_str = nt_dt.strftime(_fmt_d)
        tr_dt = _parse_ddmm(tr_match.group(1)) if tr_match else None
        if tr_dt:
            validity_tr_str = tr_dt.strftime(_fmt_d)
        vt_dt = _parse_ddmm(vt_match.group(1)) if vt_match else None
        if vt_dt:
            valid_till_str = vt_dt.strftime(_fmt_d)

        # Use first found: explicit expiry keywords > NT > Valid Till > TR
        if not expiry_date:
            expiry_date = nt_dt or vt_dt or tr_dt

        # Issue Date (for display only; IGNORE for validity decision)
        issue_date_str = None
        issue_keywords = ['issue date', 'date of issue', 'issued on', 'issued']
        for line in text.split('\n'):
            line_low = line.lower()
            if any(k in line_low for k in issue_keywords) and 'date of birth' not in line_low and 'dob' not in line_low:
                m = re.search(date_pattern_1, line_low)
                if m:
                    try:
                        d = _parse_ddmmyyyy(m.groups())
                        issue_date_str = d.strftime(_fmt_d)
                        break
                    except ValueError:
                        pass

        date_info = {
            'expiry_date': expiry_date,
            'issue_date_str': issue_date_str,
            'validity_nt_str': validity_nt_str,
            'validity_tr_str': validity_tr_str,
            'valid_till_str': valid_till_str,
        }
        date_type = 'expiry_date' if expiry_date else None
        date_string = expiry_date.strftime(_fmt_d) if expiry_date else None
        if expiry_date:
            log_dl_expiry_extraction(date_string)
        return date_info, date_string, date_type


def check_certificate_validity(certificate_type, extracted_date, date_type):
    """
    Check if certificate is valid based on extracted date and certificate type.
    
    VALIDITY RULES FOR INCOME CERTIFICATE:
    - PRIORITY 1: If validity_range (FROM → TO) is found:
        - Use end_date directly (DO NOT add +1 year)
        - Compare current_date with end_date
    - PRIORITY 2: If single issue_date is found:
        - Valid till = issue_date + 365 days
    
    VALIDITY RULES FOR DRIVING LICENSE:
    - Valid until expiry date
    
    Args:
        certificate_type (str): Type of certificate
        extracted_date: dict with 'start_date' and 'end_date' (for range) or datetime (for single date) or None
        date_type (str): 'validity_range' | 'issue_date' | 'expiry_date' | None
        
    Returns:
        dict: {
            'is_valid': bool,
            'status': 'Accepted' | 'Rejected (Expired)' | 'Not Verified (Date Missing)',
            'message': str,
            'validity_info': str,
            'validity_start_date': str or None,
            'validity_end_date': str or None,
            'validity_source': str or None
        }
    """
    current_date = datetime.now()
    _fmt = '%d-%m-%Y'  # Output format: DD-MM-YYYY

    # Early return only when we have nothing at all. For Driving License, we may have a dict
    # with issue_date/validity_nt/validity_tr but no expiry_date—we still need to run the
    # DL branch to return those for display.
    if extracted_date is None and date_type is None:
        return {
            'is_valid': False,
            'status': 'Not Verified (Date Missing)',
            'message': 'Date could not be extracted from the certificate. Please verify manually.',
            'validity_info': 'Date extraction failed',
            'validity_start_date': None,
            'validity_end_date': None,
            'validity_source': None,
            'issue_date': None,
            'validity_nt': None,
            'validity_tr': None,
        }
    
    if certificate_type == 'Income Certificate':
        # --- EXPLICIT VALIDITY RANGE: Use ONLY these dates. NEVER +1 year. ---
        if date_type == 'validity_range':
            start_date = extracted_date['start_date']
            end_date = extracted_date['end_date']
            if current_date <= end_date:
                return {
                    'is_valid': True,
                    'status': 'Accepted',
                    'message': f'Certificate is valid. Valid from: {start_date.strftime(_fmt)}, Valid till: {end_date.strftime(_fmt)}',
                    'validity_info': f'Valid from {start_date.strftime(_fmt)} to {end_date.strftime(_fmt)}',
                    'validity_start_date': start_date.strftime(_fmt),
                    'validity_end_date': end_date.strftime(_fmt),
                    'validity_source': 'Explicit Validity Period (OCR)',
                }
            else:
                return {
                    'is_valid': False,
                    'status': 'Rejected (Expired)',
                    'message': f'Certificate has expired. Valid from: {start_date.strftime(_fmt)}, Expired on: {end_date.strftime(_fmt)}',
                    'validity_info': f'Expired on {end_date.strftime(_fmt)}',
                    'validity_start_date': start_date.strftime(_fmt),
                    'validity_end_date': end_date.strftime(_fmt),
                    'validity_source': 'Explicit Validity Period (OCR)',
                }
        
        # --- SINGLE ISSUE DATE: Only when NO explicit validity range was found. Add +1 year. ---
        if date_type == 'issue_date':
            issue_date = extracted_date
            expiry_date = issue_date + timedelta(days=365)
            if current_date <= expiry_date:
                return {
                    'is_valid': True,
                    'status': 'Accepted',
                    'message': f'Certificate is valid. Issue date: {issue_date.strftime(_fmt)}, Valid until: {expiry_date.strftime(_fmt)}',
                    'validity_info': f'Valid until {expiry_date.strftime(_fmt)}',
                    'validity_start_date': issue_date.strftime(_fmt),
                    'validity_end_date': expiry_date.strftime(_fmt),
                    'validity_source': None,
                }
            else:
                return {
                    'is_valid': False,
                    'status': 'Rejected (Expired)',
                    'message': f'Certificate has expired. Issue date: {issue_date.strftime(_fmt)}, Expired on: {expiry_date.strftime(_fmt)}',
                    'validity_info': f'Expired on {expiry_date.strftime(_fmt)}',
                    'validity_start_date': issue_date.strftime(_fmt),
                    'validity_end_date': expiry_date.strftime(_fmt),
                    'validity_source': None,
                }
        
        return {
            'is_valid': False,
            'status': 'Not Verified (Date Missing)',
            'message': 'Issue date or validity range not found in certificate.',
            'validity_info': 'Issue date/validity range missing',
            'validity_start_date': None,
            'validity_end_date': None,
            'validity_source': None,
        }
    
    elif certificate_type == 'Driving License':
        # Support new DL format (dict with NT/TR, issue_date) and legacy (plain datetime)
        if isinstance(extracted_date, dict):
            expiry_date = extracted_date.get('expiry_date')
            issue_date_str = extracted_date.get('issue_date_str')
            validity_nt_str = extracted_date.get('validity_nt_str') or 'Not Present'
            validity_tr_str = extracted_date.get('validity_tr_str') or 'Not Present'
        else:
            expiry_date = extracted_date if date_type == 'expiry_date' else None
            issue_date_str = None
            validity_nt_str = 'Not Present'
            validity_tr_str = 'Not Present'

        if date_type == 'expiry_date' and expiry_date:
            if current_date <= expiry_date:
                return {
                    'is_valid': True,
                    'status': 'Accepted',
                    'message': f'License is valid. Valid until {expiry_date.strftime(_fmt)}',
                    'validity_info': f'Valid until {expiry_date.strftime(_fmt)}',
                    'validity_start_date': None,
                    'validity_end_date': expiry_date.strftime(_fmt),
                    'validity_source': None,
                    'issue_date': issue_date_str,
                    'validity_nt': validity_nt_str,
                    'validity_tr': validity_tr_str,
                }
            else:
                return {
                    'is_valid': False,
                    'status': 'Rejected (Expired)',
                    'message': f'License has expired. Expired on {expiry_date.strftime(_fmt)}',
                    'validity_info': f'Expired on {expiry_date.strftime(_fmt)}',
                    'validity_start_date': None,
                    'validity_end_date': expiry_date.strftime(_fmt),
                    'validity_source': None,
                    'issue_date': issue_date_str,
                    'validity_nt': validity_nt_str,
                    'validity_tr': validity_tr_str,
                }
        else:
            # Re-read from dict when we didn't set them above (e.g. dict but no expiry_date)
            if isinstance(extracted_date, dict):
                _issue = extracted_date.get('issue_date_str')
                _nt = extracted_date.get('validity_nt_str') or 'Not Present'
                _tr = extracted_date.get('validity_tr_str') or 'Not Present'
            else:
                _issue = _nt = _tr = 'Not Present'
            return {
                'is_valid': False,
                'status': 'Not Verified (Date Missing)',
                'message': 'Validity (NT) or Valid Till not found in certificate.',
                'validity_info': 'Validity date missing',
                'validity_start_date': None,
                'validity_end_date': None,
                'validity_source': None,
                'issue_date': _issue,
                'validity_nt': _nt,
                'validity_tr': _tr,
            }
    
    return {
        'is_valid': True,
        'status': 'Accepted',
        'message': 'Certificate type does not require date validation.',
        'validity_info': 'N/A',
        'validity_start_date': None,
        'validity_end_date': None,
        'validity_source': None,
        'issue_date': None,
        'validity_nt': None,
        'validity_tr': None,
    }


def detect_certificate_type(text):
    """
    Detect certificate type from extracted text using keyword matching.
    
    This function checks ALL certificate types and returns the detected type.
    Used for validation against expected certificate type.
    
    Logic:
    1. Convert text to lowercase for case-insensitive matching
    2. Check each certificate category for keyword matches
    3. Return first match found (if any)
    4. If no matches, return "Unknown"
    
    Args:
        text (str): Extracted OCR text
        
    Returns:
        tuple: (detected_certificate_type, matched_primary_keywords)
               Returns ("Unknown", []) if no PRIMARY keywords match
    """
    if not text or len(text.strip()) == 0:
        return "Unknown", []
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # PRIMARY-only matching to avoid conflicts (e.g., 'tahsildar' must NOT classify)
    primary_matches = {}

    for cert_type, cert_data in CERTIFICATE_KEYWORDS.items():
        primary_keywords = cert_data.get('primary_keywords', [])
        matched_primary = [kw for kw in primary_keywords if kw in text_lower]
        if matched_primary:
            primary_matches[cert_type] = matched_primary

    # No PRIMARY matches -> Unknown
    if not primary_matches:
        return "Unknown", []

    # If multiple PRIMARY matches exist, choose by fixed priority order
    for cert_type in CERTIFICATE_PRIORITY:
        if cert_type in primary_matches:
            return cert_type, primary_matches[cert_type]

    # Fallback (should not happen if CERTIFICATE_PRIORITY includes all keys)
    first_type = next(iter(primary_matches.keys()))
    return first_type, primary_matches[first_type]


def validate_certificate_match(expected_type, detected_type, matched_primary_keywords, extracted_text=None):
    """
    Validate if the detected certificate matches the expected certificate type.
    Also performs date validity checking for Income Certificate and Driving License.
    
    STRICT VALIDATION RULES:
    1. If detected matches expected → Check date validity (if applicable)
    2. If detected does NOT match expected → Status: "Rejected"
    3. If detected is "Unknown" → Status: "Not Classified"
    
    DATE VALIDATION (if certificate type matches):
    - Income Certificate: Valid for 1 year from issue date
    - Driving License: Valid until expiry date
    
    Args:
        expected_type (str): The certificate type expected by the upload field
        detected_type (str): The certificate type detected by OCR
        matched_primary_keywords (list): PRIMARY keywords that matched during detection
        extracted_text (str): Optional OCR text for date extraction
        
    Returns:
        dict: {
            'status': 'Accepted' | 'Rejected' | 'Rejected (Expired)' | 'Not Classified' | 'Not Verified (Date Missing)',
            'message': Status message explaining the result,
            'extracted_date': str or None,
            'current_date': str,
            'validity_info': str or None,
            'validity_period': str or None,
            'matched_primary_keywords': list
        }
    """
    current_date_str = datetime.now().strftime('%d-%m-%Y')
    
    _dl_none = {'issue_date': None, 'validity_nt': None, 'validity_tr': None}

    if detected_type == "Unknown":
        return {
            'status': 'Not Classified',
            'message': 'Certificate type could not be determined. No matching keywords found.',
            'extracted_date': None,
            'current_date': current_date_str,
            'validity_info': None,
            'validity_period': None,
            'validity_start_date': None,
            'validity_end_date': None,
            'validity_source': None,
            'matched_primary_keywords': [],
            **_dl_none,
        }
    
    if detected_type != expected_type:
        return {
            'status': 'Rejected',
            'message': f'Uploaded certificate does not match the selected category. Expected: {expected_type}, Detected: {detected_type}',
            'extracted_date': None,
            'current_date': current_date_str,
            'validity_info': None,
            'validity_period': None,
            'validity_start_date': None,
            'validity_end_date': None,
            'validity_source': None,
            'matched_primary_keywords': matched_primary_keywords or [],
            **_dl_none,
        }
    
    if expected_type in ['Income Certificate', 'Driving License']:
        if extracted_text:
            date_info, date_str, date_type = extract_date_from_text(extracted_text, expected_type)
            validity_result = check_certificate_validity(expected_type, date_info, date_type)
            return {
                'status': validity_result['status'],
                'message': validity_result['message'],
                'extracted_date': date_str,
                'current_date': current_date_str,
                'validity_info': validity_result['validity_info'],
                'validity_period': '1 Year' if expected_type == 'Income Certificate' and date_type == 'issue_date' else None,
                'validity_start_date': validity_result.get('validity_start_date'),
                'validity_end_date': validity_result.get('validity_end_date'),
                'validity_source': validity_result.get('validity_source'),
                'matched_primary_keywords': matched_primary_keywords or [],
                'issue_date': validity_result.get('issue_date'),
                'validity_nt': validity_result.get('validity_nt'),
                'validity_tr': validity_result.get('validity_tr'),
            }
        else:
            return {
                'status': 'Not Verified (Date Missing)',
                'message': 'Cannot verify date validity. OCR text not available.',
                'extracted_date': None,
                'current_date': current_date_str,
                'validity_info': 'Date extraction failed - no text available',
                'validity_period': '1 Year' if expected_type == 'Income Certificate' else None,
                'validity_start_date': None,
                'validity_end_date': None,
                'validity_source': None,
                'matched_primary_keywords': matched_primary_keywords or [],
                **_dl_none,
            }
    
    return {
        'status': 'Accepted',
        'message': f'Certificate verified successfully. Detected: {detected_type}',
        'extracted_date': None,
        'current_date': current_date_str,
        'validity_info': 'N/A (Date validation not required for this certificate type)',
        'validity_period': None,
        'validity_start_date': None,
        'validity_end_date': None,
        'validity_source': None,
        'matched_primary_keywords': matched_primary_keywords or [],
        **_dl_none,
    }


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Render the main upload page (user interface)."""
    return render_template('index.html')


@app.route('/admin')
def admin():
    """Render the admin interface."""
    return render_template('admin.html')


@app.route('/api/user/fields', methods=['GET'])
def get_user_fields():
    """Get active certificate fields for user interface."""
    active_fields = get_active_fields()
    return jsonify({
        'success': True,
        'fields': active_fields
    })


@app.route('/api/admin/fields', methods=['GET'])
def get_admin_fields():
    """Get all certificate fields (admin view - includes disabled)."""
    all_fields = load_fields_config()
    return jsonify({
        'success': True,
        'fields': all_fields
    })


@app.route('/api/admin/fields', methods=['POST'])
def create_field():
    """Create a new certificate field."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required = ['field_name', 'expected_category', 'expiry_validation_required', 'mandatory']
        if not all(k in data for k in required):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Validate expected_category exists in CERTIFICATE_KEYWORDS
        if data['expected_category'] not in CERTIFICATE_KEYWORDS:
            return jsonify({
                'success': False,
                'error': f'Invalid expected_category. Must be one of: {", ".join(CERTIFICATE_KEYWORDS.keys())}'
            }), 400
        
        # Generate field ID from field_name (sanitize)
        field_id = data.get('id') or re.sub(r'[^a-z0-9]+', '_', data['field_name'].lower()).strip('_')
        
        # Check if ID already exists
        existing_fields = load_fields_config()
        if any(f.get('id') == field_id for f in existing_fields):
            field_id = f"{field_id}_{uuid.uuid4().hex[:8]}"
        
        allowed = data.get('allowed_file_types')
        if allowed is None or (isinstance(allowed, list) and 'all' in [x.lower() for x in (allowed or [])]):
            allowed = DEFAULT_ALLOWED_FILE_TYPES
        if not isinstance(allowed, list):
            allowed = DEFAULT_ALLOWED_FILE_TYPES
        
        new_field = {
            'id': field_id,
            'field_name': data['field_name'],
            'expected_category': data['expected_category'],
            'expiry_validation_required': bool(data['expiry_validation_required']),
            'mandatory': bool(data['mandatory']),
            'enabled': data.get('enabled', True),
            'allowed_file_types': allowed
        }
        
        existing_fields.append(new_field)
        if save_fields_config(existing_fields):
            return jsonify({'success': True, 'field': new_field}), 201
        else:
            return jsonify({'success': False, 'error': 'Failed to save field'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/fields/<field_id>', methods=['PUT'])
def update_field(field_id):
    """Update an existing certificate field."""
    try:
        data = request.get_json()
        fields = load_fields_config()
        
        field_index = None
        for i, field in enumerate(fields):
            if field.get('id') == field_id:
                field_index = i
                break
        
        if field_index is None:
            return jsonify({'success': False, 'error': 'Field not found'}), 404
        
        # Update field
        if 'field_name' in data:
            fields[field_index]['field_name'] = data['field_name']
        if 'expected_category' in data:
            if data['expected_category'] not in CERTIFICATE_KEYWORDS:
                return jsonify({
                    'success': False,
                    'error': f'Invalid expected_category. Must be one of: {", ".join(CERTIFICATE_KEYWORDS.keys())}'
                }), 400
            fields[field_index]['expected_category'] = data['expected_category']
        if 'expiry_validation_required' in data:
            fields[field_index]['expiry_validation_required'] = bool(data['expiry_validation_required'])
        if 'mandatory' in data:
            fields[field_index]['mandatory'] = bool(data['mandatory'])
        if 'enabled' in data:
            fields[field_index]['enabled'] = bool(data['enabled'])
        if 'allowed_file_types' in data:
            allowed = data['allowed_file_types']
            if allowed is None or (isinstance(allowed, list) and 'all' in [x.lower() for x in (allowed or [])]):
                allowed = DEFAULT_ALLOWED_FILE_TYPES
            fields[field_index]['allowed_file_types'] = allowed if isinstance(allowed, list) else DEFAULT_ALLOWED_FILE_TYPES
        
        if save_fields_config(fields):
            return jsonify({'success': True, 'field': fields[field_index]})
        else:
            return jsonify({'success': False, 'error': 'Failed to save field'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/admin/fields/<field_id>', methods=['DELETE'])
def delete_field(field_id):
    """Delete a certificate field."""
    try:
        fields = load_fields_config()
        fields = [f for f in fields if f.get('id') != field_id]
        
        if save_fields_config(fields):
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Failed to delete field'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and process STRICT certificate verification.
    
    This endpoint enforces strict validation:
    - User must upload certificate in the correct field
    - System validates if detected certificate matches expected type
    - Returns Rejected status if mismatch occurs
    
    Expected Parameters:
    - file: Image file (JPG, JPEG, PNG)
    - expected_certificate: The certificate type expected by the upload field
                           Must be one of: 'Aadhaar Card', 'Community Certificate',
                           'Income Certificate', 'Educational Certificate (Anna University)',
                           'Bonafide Certificate', 'Driving License'
    
    Returns:
        JSON response with:
        - expected_certificate: The certificate type expected
        - detected_certificate: The certificate type detected by OCR
        - status: Accepted / Rejected / Not Classified / Rejected (Expired) / Not Verified (Date Missing)
        - message: Status message
        - extracted_text: Full OCR text
        - matched_primary_keywords: List of matched PRIMARY keywords (if any)
        - extracted_date: Extracted issue/expiry date (if applicable)
        - current_date: System date (DD/MM/YYYY)
        - validity_period: For Income Certificate => '1 Year'
        - validity_info: Human-readable validity explanation
    """
    try:
        # Get expected certificate type from request
        expected_certificate = request.form.get('expected_certificate', '').strip()
        field_id = request.form.get('field_id', '').strip()
        
        # Validate field exists and is enabled (if field_id provided)
        if field_id:
            field = get_field_by_id(field_id)
            if not field:
                return jsonify({
                    'error': f'Field with ID "{field_id}" not found',
                    'expected_certificate': expected_certificate,
                    'detected_certificate': 'Unknown',
                    'status': 'Not Classified',
                    'message': 'Invalid field ID',
                    'extracted_text': '',
                    'matched_primary_keywords': [],
                    'extracted_date': None,
                    'current_date': datetime.now().strftime('%d/%m/%Y'),
                    'validity_period': None,
                    'validity_info': None
                }), 400
            if not field.get('enabled', True):
                return jsonify({
                    'error': f'Field "{field.get("field_name")}" is currently disabled',
                    'expected_certificate': expected_certificate,
                    'detected_certificate': 'Unknown',
                    'status': 'Not Classified',
                    'message': 'This field is disabled by admin',
                    'extracted_text': '',
                    'matched_primary_keywords': [],
                    'extracted_date': None,
                    'current_date': datetime.now().strftime('%d/%m/%Y'),
                    'validity_period': None,
                    'validity_info': None
                }), 400
            # Use expected_category from field if not provided
            if not expected_certificate:
                expected_certificate = field.get('expected_category', '')
        
        # Validate expected certificate type
        # Check against predefined certificate types in keyword dictionary
        valid_certificate_types = list(CERTIFICATE_KEYWORDS.keys())
        if not expected_certificate or expected_certificate not in valid_certificate_types:
            return jsonify({
                'error': f'Invalid expected_certificate. Must be one of: {", ".join(valid_certificate_types)}',
                'expected_certificate': '',
                'detected_certificate': 'Unknown',
                'status': 'Not Classified',
                'message': 'Invalid certificate type specified',
                'extracted_text': '',
                'matched_primary_keywords': [],
                'extracted_date': None,
                'current_date': datetime.now().strftime('%d/%m/%Y'),
                'validity_period': None,
                'validity_info': None
            }), 400
        
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'expected_certificate': expected_certificate,
                'detected_certificate': 'Unknown',
                'status': 'Not Classified',
                'message': 'No file uploaded',
                'extracted_text': '',
                'matched_primary_keywords': [],
                'extracted_date': None,
                'current_date': datetime.now().strftime('%d/%m/%Y'),
                'validity_period': None,
                'validity_info': None
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'expected_certificate': expected_certificate,
                'detected_certificate': 'Unknown',
                'status': 'Not Classified',
                'message': 'No file selected',
                'extracted_text': '',
                'matched_primary_keywords': [],
                'extracted_date': None,
                'current_date': datetime.now().strftime('%d/%m/%Y'),
                'validity_period': None,
                'validity_info': None
            }), 400
        
        # Allowed file types for this field (default: all document formats)
        allowed_types = DEFAULT_ALLOWED_FILE_TYPES
        if field_id and field:
            allowed_types = field.get('allowed_file_types') or DEFAULT_ALLOWED_FILE_TYPES
        
        # Check if file extension is allowed
        if not allowed_file(file.filename, allowed_types):
            allowed_str = ', '.join(sorted(set(allowed_types))) if allowed_types else 'pdf, jpg, jpeg, png, tiff, webp'
            return jsonify({
                'error': f'Invalid file type. Allowed formats: {allowed_str}',
                'expected_certificate': expected_certificate,
                'detected_certificate': 'Unknown',
                'status': 'Not Classified',
                'message': 'Unsupported file format',
                'extracted_text': '',
                'matched_primary_keywords': [],
                'extracted_date': None,
                'current_date': datetime.now().strftime('%d/%m/%Y'),
                'validity_period': None,
                'validity_info': None
            }), 400
        
        file_ext = get_file_extension(file.filename)
        if file_ext not in SUPPORTED_EXTENSIONS:
            return jsonify({
                'error': 'Unsupported file format.',
                'expected_certificate': expected_certificate,
                'detected_certificate': 'Unknown',
                'status': 'Not Classified',
                'message': 'Unsupported file format',
                'extracted_text': '',
                'matched_primary_keywords': [],
                'extracted_date': None,
                'current_date': datetime.now().strftime('%d/%m/%Y'),
                'validity_period': None,
                'validity_info': None
            }), 400
        
        # Save to temp file first; only move to permanent storage if category matches or on reassignment
        filename = secure_filename(file.filename)
        temp_name = f"temp_{uuid.uuid4().hex}_{filename}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_name)
        file.save(temp_path)
        
        try:
            # Normalize document into common OCR pipeline (PDF or image -> text)
            extracted_text, original_format = extract_text_from_document(temp_path, file_ext)
        except Exception as ocr_err:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise ocr_err
        
        # Detect certificate type from extracted text
        detected_certificate, matched_primary_keywords = detect_certificate_type(extracted_text)
        
        # Perform STRICT validation: Check if detected matches expected
        validation_result = validate_certificate_match(
            expected_certificate,
            detected_certificate,
            matched_primary_keywords,
            extracted_text
        )
        
        # Smart reassignment: category mismatch but detected is known -> offer reassignment (do not save under wrong category)
        reassignment_from = request.form.get('reassignment_from', '').strip()
        if reassignment_from:
            log_reassignment(reassignment_from, expected_certificate, filename)
        
        category_mismatch = (
            detected_certificate != expected_certificate
            and detected_certificate != "Unknown"
        )
        
        if category_mismatch:
            # Do not save file under wrong category; remove temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            return jsonify({
                'success': True,
                'category_mismatch': True,
                'expected_certificate': expected_certificate,
                'detected_certificate': detected_certificate,
                'status': validation_result['status'],
                'message': validation_result['message'],
                'extracted_text': extracted_text,
                'matched_primary_keywords': validation_result.get('matched_primary_keywords', matched_primary_keywords),
                'extracted_date': validation_result.get('extracted_date'),
                'current_date': validation_result.get('current_date'),
                'validity_info': validation_result.get('validity_info'),
                'validity_period': validation_result.get('validity_period'),
                'validity_start_date': validation_result.get('validity_start_date'),
                'validity_end_date': validation_result.get('validity_end_date'),
                'validity_source': validation_result.get('validity_source'),
                'issue_date': validation_result.get('issue_date'),
                'validity_nt': validation_result.get('validity_nt'),
                'validity_tr': validation_result.get('validity_tr'),
                'original_format': file_ext,
            })
        
        # Match or Unknown: save to permanent location (avoid duplicate filename)
        final_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(final_path):
            base, ext = os.path.splitext(filename)
            final_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_{uuid.uuid4().hex[:8]}{ext}")
            filename = os.path.basename(final_path)
        try:
            os.rename(temp_path, final_path)
        except OSError:
            shutil.move(temp_path, final_path)
        
        return jsonify({
            'success': True,
            'category_mismatch': False,
            'expected_certificate': expected_certificate,
            'detected_certificate': detected_certificate,
            'status': validation_result['status'],
            'message': validation_result['message'],
            'extracted_text': extracted_text,
            'matched_primary_keywords': validation_result.get('matched_primary_keywords', matched_primary_keywords),
            'extracted_date': validation_result.get('extracted_date'),
            'current_date': validation_result.get('current_date'),
            'validity_info': validation_result.get('validity_info'),
            'validity_period': validation_result.get('validity_period'),
            'validity_start_date': validation_result.get('validity_start_date'),
            'validity_end_date': validation_result.get('validity_end_date'),
            'validity_source': validation_result.get('validity_source'),
            'issue_date': validation_result.get('issue_date'),
            'validity_nt': validation_result.get('validity_nt'),
            'validity_tr': validation_result.get('validity_tr'),
            'original_format': file_ext,
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'expected_certificate': request.form.get('expected_certificate', ''),
            'detected_certificate': 'Unknown',
            'status': 'Not Classified',
            'message': f'Processing error: {str(e)}',
            'extracted_text': '',
            'matched_primary_keywords': [],
            'extracted_date': None,
            'current_date': datetime.now().strftime('%d/%m/%Y'),
            'validity_period': None,
            'validity_info': None
        }), 500


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Certificate Verification System - Starting...")
    print("=" * 60)
    print("\nMake sure Tesseract OCR is installed and in your system PATH.")
    print("For Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    print("\nServer will start at: http://127.0.0.1:5000")
    print("=" * 60)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
