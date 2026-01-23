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
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import cv2
import pytesseract
from PIL import Image
import numpy as np
from datetime import datetime, timedelta

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'}

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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


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
    # DRIVING LICENSE: New Indian DL format (Validity NT/TR) and legacy (Valid Till)
    # ========================================================================
    # IGNORE for validity decision: Date of Birth, Issue Date, TR (unless transport).
    # PRIORITY 1: Validity (NT); PRIORITY 2: Valid Till; PRIORITY 3: Not Verified.
    # ========================================================================
    elif certificate_type == 'Driving License':
        _fmt_d = '%d-%m-%Y'

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

        # STRICT regex patterns (case-insensitive via text_lower)
        # Validity (NT): validity\s*\(nt\)\s*(DD-MM-YYYY)
        PATTERN_NT = r'validity\s*\(\s*nt\s*\)\s*(\d{2}[-/]\d{2}[-/]\d{4})'
        # Validity (TR): validity\s*\(tr\)\s*(DD-MM-YYYY)
        PATTERN_TR = r'validity\s*\(\s*tr\s*\)\s*(\d{2}[-/]\d{2}[-/]\d{4})'
        # Valid Till (legacy): valid\s*till\s*(DD-MM-YYYY)
        PATTERN_VALID_TILL = r'valid\s*till\s*(\d{2}[-/]\d{2}[-/]\d{4})'

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

        # PRIORITY 1: Validity (NT) -> use as expiry_date for validation
        # PRIORITY 2: Valid Till (legacy) -> use as expiry_date
        # PRIORITY 3: neither -> expiry_date = None
        expiry_date = nt_dt if nt_dt else (vt_dt if vt_dt else None)

        date_info = {
            'expiry_date': expiry_date,
            'issue_date_str': issue_date_str,
            'validity_nt_str': validity_nt_str,
            'validity_tr_str': validity_tr_str,
            'valid_till_str': valid_till_str,
        }
        date_type = 'expiry_date' if expiry_date else None
        date_string = expiry_date.strftime(_fmt_d) if expiry_date else None
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
    """Render the main upload page."""
    return render_template('index.html')


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
        
        # Check if file extension is allowed
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type. Please upload JPG, JPEG, or PNG files.',
                'expected_certificate': expected_certificate,
                'detected_certificate': 'Unknown',
                'status': 'Not Classified',
                'message': 'Invalid file format',
                'extracted_text': '',
                'matched_primary_keywords': [],
                'extracted_date': None,
                'current_date': datetime.now().strftime('%d/%m/%Y'),
                'validity_period': None,
                'validity_info': None
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text using OCR
        extracted_text = extract_text_ocr(filepath)
        
        # Detect certificate type from extracted text
        detected_certificate, matched_primary_keywords = detect_certificate_type(extracted_text)
        
        # Perform STRICT validation: Check if detected matches expected
        # Also performs date validity checking for Income Certificate and Driving License
        validation_result = validate_certificate_match(
            expected_certificate,
            detected_certificate,
            matched_primary_keywords,
            extracted_text  # Pass text for date extraction
        )
        
        # Clean up uploaded file (optional - remove if you want to keep files)
        # os.remove(filepath)
        
        # Return results with strict validation and date information
        return jsonify({
            'success': True,
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
