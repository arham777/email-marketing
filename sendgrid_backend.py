#!/usr/bin/env python3
"""
SendGrid Email API Backend

A FastAPI backend for sending personalized emails through SendGrid.
This API provides endpoints for uploading recipient lists, creating email content,
and sending personalized emails with support for images and HTML content.

Usage:
uvicorn sendgrid_backend:app --reload

Environment Variables (store in .env file):
- SENDGRID_API_KEY: Your SendGrid API key
- FROM_EMAIL: Default sender email address (optional)
"""

import os
import csv
import base64
import mimetypes
import re
import time
import logging
import sys
import io
import json
import tempfile
import urllib.parse
import uuid
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

# FastAPI
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Query, Body, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, validator

# SendGrid
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Mail, Attachment, FileContent, FileName,
    FileType, Disposition, ContentId
)

# Utils
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('email_api.log')
    ]
)
logger = logging.getLogger('email_api')

# Load environment variables
load_dotenv()
SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY')
DEFAULT_FROM_EMAIL = os.getenv('FROM_EMAIL')

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 2
EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="SendGrid Email API",
    description="API for sending personalized emails via SendGrid",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class EmailAddress(BaseModel):
    email: EmailStr
    name: Optional[str] = None

class ImageData(BaseModel):
    path: str
    content_id: str
    is_url: Optional[bool] = False
    is_base64: Optional[bool] = False

class EmailContent(BaseModel):
    subject: str
    html_content: str
    images: Optional[List[ImageData]] = []

class Recipient(BaseModel):
    email: EmailStr
    personalization: Optional[Dict[str, Any]] = {}

class SendEmailRequest(BaseModel):
    from_email: Union[EmailStr, str]
    recipients: List[Recipient]
    content: EmailContent
    test_mode: Optional[bool] = False

class CSVColumnInfo(BaseModel):
    columns: List[str]
    email_column: Optional[str] = None

class CSVProcessResult(BaseModel):
    recipients: List[Dict[str, Any]]
    valid_count: int
    invalid_count: int
    personalization_fields: List[str]
    invalid_emails: Optional[List[Dict[str, Any]]] = []

class PreviewEmailRequest(BaseModel):
    content: EmailContent
    recipient: Optional[Dict[str, Any]] = None

class ValidationResult(BaseModel):
    valid: bool
    message: Optional[str] = None

class EmailValidationRequest(BaseModel):
    emails: List[str]

class EmailValidationResult(BaseModel):
    valid_emails: List[str]
    invalid_emails: List[Dict[str, str]]
    valid_count: int
    invalid_count: int

class SendResponse(BaseModel):
    success: bool
    message: str
    total: int
    success_count: int
    failed_count: int
    failed_emails: Optional[List[Dict[str, str]]] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    api_key_configured: bool
    from_email_configured: bool

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int

# --- Helper Functions ---

def validate_email(email: str) -> bool:
    """Validates email format using regex."""
    return bool(re.match(EMAIL_REGEX, email))

def find_email_column(headers: List[str]) -> Optional[str]:
    """Try to automatically find the email column in CSV headers."""
    email_keywords = ['email', 'e-mail', 'mail', 'recipient', 'to']
    
    # First check for exact matches
    for header in headers:
        if header.lower() in email_keywords:
            return header
    
    # Then check for partial matches
    for header in headers:
        for keyword in email_keywords:
            if keyword in header.lower():
                return header
    
    return None

def process_csv_file(file_content: bytes) -> CSVProcessResult:
    """Process a CSV file and extract recipient data."""
    recipients = []
    invalid_emails = []
    email_column = None
    
    try:
        # Read CSV from file content
        csv_text = file_content.decode('utf-8-sig')
        csv_file = io.StringIO(csv_text)
        reader = csv.reader(csv_file)
        
        # Get headers
        headers = next(reader)
        
        # Find email column
        email_column = find_email_column(headers)
        if not email_column:
            # If we can't find it automatically, use the first column
            email_column = headers[0]
        
        # Create a DictReader for the data
        csv_file.seek(0)
        next(csv.reader(csv_file))  # Skip header
        dict_reader = csv.DictReader(csv_file, fieldnames=headers)
        
        # Process rows
        for row_num, row in enumerate(dict_reader, start=2):
            if row and email_column in row:
                email_address = row[email_column].strip()
                if email_address:
                    # Validate email format
                    if not validate_email(email_address):
                        invalid_emails.append({
                            "row": row_num,
                            "email": email_address,
                            "reason": "Invalid email format"
                        })
                        continue
                    
                    # Store all row data for personalization
                    recipient_data = row.copy()
                    # Store email under standard EMAIL key for consistency
                    recipient_data['EMAIL'] = email_address
                    recipients.append(recipient_data)
        
        # Extract personalization fields
        personalization_fields = set()
        for recipient in recipients:
            personalization_fields.update(recipient.keys())
        
        # Remove email fields from personalization fields
        personalization_fields = [f for f in personalization_fields 
                                  if f.lower() != email_column.lower() and f != 'EMAIL']
        
        return CSVProcessResult(
            recipients=recipients,
            valid_count=len(recipients),
            invalid_count=len(invalid_emails),
            personalization_fields=personalization_fields,
            invalid_emails=invalid_emails
        )
        
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing CSV file: {str(e)}"
        )

def create_attachment(image_data: ImageData) -> Optional[Attachment]:
    """Creates a SendGrid Attachment object from image data."""
    if not image_data.path:
        return None

    logger.info(f"Processing image: '{image_data.path}'")
    
    try:
        # Handle base64 encoded image
        if image_data.is_base64 or (image_data.path.startswith(('data:image/', 'data:application/')) and 
                                   ';base64,' in image_data.path):
            try:
                # Extract content type and data
                _, data_part = image_data.path.split(',', 1)
                content_type = image_data.path.split(',')[0].split(':')[1].split(';')[0]
                
                # Decode base64 data
                data = base64.b64decode(data_part)
                
                # Generate a filename
                extension = content_type.split('/')[-1]
                filename = f"image_{image_data.content_id}.{extension}"
                
            except Exception as e:
                logger.error(f"Error processing base64 image: {e}")
                return None
                
        # Handle URL
        elif image_data.is_url or image_data.path.startswith(('http://', 'https://')):
            try:
                response = requests.get(image_data.path, timeout=10)
                if response.status_code != 200:
                    logger.error(f"Failed to download image: {response.status_code}")
                    return None
                
                data = response.content
                
                # Get filename from URL
                filename = os.path.basename(urllib.parse.urlparse(image_data.path).path)
                if not filename:
                    filename = f"image_{image_data.content_id}"
                
                content_type = response.headers.get('Content-Type')
                
            except Exception as e:
                logger.error(f"Error downloading image: {e}")
                return None
        else:
            # Handle file path - could be a previously uploaded file
            file_path = Path(image_data.path)
            if not file_path.is_absolute():
                file_path = UPLOAD_DIR / image_data.path
                
            if not file_path.is_file():
                logger.error(f"Image file not found: {file_path}")
                return None

            # Read image data
            with open(file_path, 'rb') as f:
                data = f.read()
                
            # Use the filename from the path
            filename = file_path.name
            content_type = None

        # Check file size
        file_size_kb = len(data) / 1024
        if file_size_kb > 10240:  # 10MB limit
            logger.warning(f"Image too large: {file_size_kb:.2f}KB - SendGrid limit is 10MB")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image too large: {file_size_kb:.2f}KB (exceeds 10MB limit)"
            )

        # Encode to Base64
        encoded_file = base64.b64encode(data).decode()

        # Determine MIME type if needed
        if content_type is None:
            file_type, _ = mimetypes.guess_type(filename)
            if not file_type:
                # Try to determine from extension
                _, ext = os.path.splitext(filename)
                if ext.lower() in ['.jpg', '.jpeg']:
                    file_type = 'image/jpeg'
                elif ext.lower() == '.png':
                    file_type = 'image/png'
                elif ext.lower() == '.gif':
                    file_type = 'image/gif'
                else:
                    file_type = 'application/octet-stream'
        else:
            file_type = content_type

        # Create Attachment object
        attachment = Attachment(
            FileContent(encoded_file),
            FileName(filename),
            FileType(file_type),
            Disposition('inline'),
            ContentId(image_data.content_id)
        )
        logger.info(f"Image processed successfully: {image_data.content_id}")
        return attachment

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

def process_personalization(html_content: str, recipient_data: Dict[str, Any]) -> str:
    """Replace personalization placeholders with recipient data."""
    if not html_content or not recipient_data:
        return html_content
        
    personalized_content = html_content
    
    # Find all placeholders
    placeholders = set(re.findall(r'\{\{([^}]+)\}\}', personalized_content))
    
    # Replace placeholders with data
    for key, value in recipient_data.items():
        if not value:  # Skip empty values
            continue
            
        # Try different case variations
        for key_format in [key, key.upper(), key.lower(), key.title(), key.capitalize()]:
            placeholder = "{{" + key_format + "}}"
            if placeholder in personalized_content:
                personalized_content = personalized_content.replace(placeholder, str(value))
    
    # Try variants with spaces/underscores
    for key, value in recipient_data.items():
        if not value:
            continue
            
        # Try replacing underscores with spaces and vice versa
        key_variants = []
        if '_' in key:
            key_variants.append(key.replace('_', ' '))
        elif ' ' in key:
            key_variants.append(key.replace(' ', '_'))
            
        for variant in key_variants:
            for var_format in [variant, variant.upper(), variant.lower(), variant.title(), variant.capitalize()]:
                placeholder = "{{" + var_format + "}}"
                if placeholder in personalized_content:
                    personalized_content = personalized_content.replace(placeholder, str(value))
    
    return personalized_content

def enhance_html_content(html_content: str) -> str:
    """Add proper HTML structure if not present."""
    if not html_content:
        return html_content
        
    enhanced_content = html_content
    
    # Check if content has HTML structure
    if "<html" not in enhanced_content.lower() and "</html>" not in enhanced_content.lower():
        enhanced_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
{enhanced_content}
</body>
</html>"""

    return enhanced_content

def initialize_sendgrid() -> Optional[SendGridAPIClient]:
    """Initialize SendGrid client."""
    if not SENDGRID_API_KEY:
        logger.error("SendGrid API key not found")
        return None
    
    try:
        logger.info("Initializing SendGrid")
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        return sg
    except Exception as e:
        logger.error(f"SendGrid initialization failed: {e}")
        return None

def get_sg_client() -> SendGridAPIClient:
    """Get SendGrid client or raise exception if not available."""
    sg_client = initialize_sendgrid()
    if not sg_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SendGrid API key not configured or invalid"
        )
    return sg_client

def send_emails_async(
    from_email: str,
    recipients: List[Dict[str, Any]],
    subject: str,
    html_content: str,
    attachments: Optional[List[Attachment]] = None,
    test_mode: bool = False
) -> Dict[str, Any]:
    """Send personalized emails to recipients with attachments (async)."""
    total_recipients = len(recipients)
    success_count = 0
    failed_emails = []
    
    logger.info(f"Starting email sending process {'(TEST MODE)' if test_mode else ''}")
    
    # Check for sender name format "Name <email@example.com>"
    from_name = None
    if " <" in from_email and from_email.endswith(">"):
        from_name = from_email.split(" <")[0]
        from_email = from_email.split(" <")[1].rstrip(">")
    
    # Initialize SendGrid
    try:
        sg_client = get_sg_client()
    except HTTPException as e:
        return {
            "success": False,
            "message": str(e.detail),
            "total": total_recipients,
            "success_count": 0,
            "failed_count": total_recipients,
            "failed_emails": [{"email": r.get("EMAIL", "unknown"), "reason": str(e.detail)} 
                              for r in recipients]
        }
    
    start_time = time.time()
    
    # Process each recipient
    for i, recipient in enumerate(recipients, 1):
        email = recipient.get('EMAIL')
        
        # Skip invalid emails
        if not email or not validate_email(email):
            logger.warning(f"Skipping invalid email: {email}")
            failed_emails.append({"email": email, "reason": "Invalid email format"})
            continue
        
        try:
            # Create personalized content
            personalized_html = process_personalization(html_content, recipient)
            personalized_subject = process_personalization(subject, recipient)
            
            # Create message
            message = Mail(
                from_email=from_email,
                to_emails=email,
                subject=personalized_subject,
                html_content=personalized_html
            )
            
            # Add from name if provided
            if from_name:
                message.from_email = (from_name, from_email)
            
            # Add attachments if provided
            if attachments:
                for attachment in attachments:
                    message.add_attachment(attachment)
            
            # Send the email
            if not test_mode:
                retries = 0
                success = False
                
                while not success and retries <= MAX_RETRIES:
                    try:
                        if retries > 0:
                            logger.info(f"Retry #{retries} for {email}")
                            time.sleep(RETRY_DELAY * retries)
                            
                        response = sg_client.send(message)
                        status_code = response.status_code
                        
                        if 200 <= status_code < 300:  # Success
                            logger.info(f"Email sent to {email}")
                            success_count += 1
                            success = True
                        else:
                            logger.error(f"Failed to send to {email}: {status_code}")
                            retries += 1
                    except Exception as e:
                        logger.error(f"Error sending to {email}: {str(e)}")
                        retries += 1
                
                if not success:
                    failed_emails.append({"email": email, "reason": f"Failed after {retries} attempts"})
            else:
                # Test mode - just log
                logger.info(f"TEST MODE - Would send to: {email}")
                success_count += 1
            
            # Small pause to avoid rate limits
            time.sleep(0.2)
            
        except Exception as e:
            logger.error(f"Error processing {email}: {str(e)}")
            failed_emails.append({"email": email, "reason": str(e)})
    
    # Calculate duration
    duration = time.time() - start_time
    logger.info(f"Email sending completed in {duration:.1f} seconds. Success: {success_count}/{total_recipients}")
    
    return {
        "success": success_count > 0,
        "message": "Emails sent successfully" if success_count > 0 else "Failed to send emails",
        "total": total_recipients,
        "success_count": success_count,
        "failed_count": len(failed_emails),
        "failed_emails": failed_emails if failed_emails else None
    }

# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API and SendGrid connection are working."""
    sg_client = initialize_sendgrid()
    
    if sg_client:
        return {
            "status": "ok",
            "message": "SendGrid API connected successfully",
            "api_key_configured": bool(SENDGRID_API_KEY),
            "from_email_configured": bool(DEFAULT_FROM_EMAIL)
        }
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "message": "Could not connect to SendGrid API",
                "api_key_configured": bool(SENDGRID_API_KEY),
                "from_email_configured": bool(DEFAULT_FROM_EMAIL)
            }
        )

@app.post("/validate-emails", response_model=EmailValidationResult)
async def validate_emails(request: EmailValidationRequest):
    """Validate a list of email addresses."""
    valid_emails = []
    invalid_emails = []
    
    for email in request.emails:
        if validate_email(email):
            valid_emails.append(email)
        else:
            invalid_emails.append({
                "email": email,
                "reason": "Invalid email format"
            })
    
    return {
        "valid_emails": valid_emails,
        "invalid_emails": invalid_emails,
        "valid_count": len(valid_emails),
        "invalid_count": len(invalid_emails)
    }

@app.post("/process-csv", response_model=CSVProcessResult)
async def process_csv(file: UploadFile = File(...)):
    """Upload and process a CSV file with recipient data."""
    try:
        contents = await file.read()
        return process_csv_file(contents)
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing CSV file: {str(e)}"
        )

@app.post("/upload-image", response_model=List[Dict[str, str]])
async def upload_image(images: List[UploadFile] = File(...)):
    """Upload multiple images for use in emails.
    
    This endpoint accepts multiple images and processes them for use in emails.
    Each image is saved and assigned a unique content ID that can be used in the HTML content.
    
    Returns a list of image information including:
    - filename: The saved filename
    - path: Path to the saved file
    - content_id: Unique ID to reference in HTML
    - html_tag: Ready-to-use HTML img tag
    """
    try:
        # Check if images list is empty
        if not images:
            return []
        
        results = []
        
        # Process each image
        for image in images:
            # Generate unique filename
            content_id = f"image_{uuid.uuid4().hex[:8]}"
            extension = Path(image.filename).suffix
            filename = f"{content_id}{extension}"
            
            # Save file
            file_path = UPLOAD_DIR / filename
            with open(file_path, "wb") as f:
                f.write(await image.read())
            
            # Add result
            results.append({
                "filename": filename,
                "path": str(file_path),
                "content_id": content_id,
                "html_tag": f'<img src="cid:{content_id}" alt="Image">'
            })
        
        return results
    except Exception as e:
        logger.error(f"Error uploading images: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error uploading images: {str(e)}"
        )

@app.post("/preview-email")
async def preview_email(request: PreviewEmailRequest):
    """Preview an email with optional personalization applied."""
    try:
        # Enhance HTML content
        html_content = enhance_html_content(request.content.html_content)
        
        # Apply personalization if recipient data provided
        if request.recipient:
            html_content = process_personalization(html_content, request.recipient)
        
        # Process images to ensure they exist
        image_statuses = []
        for image in request.content.images:
            attachment = create_attachment(image)
            image_statuses.append({
                "content_id": image.content_id,
                "status": "valid" if attachment else "invalid",
                "path": image.path
            })
        
        return {
            "html_content": html_content,
            "subject": request.content.subject,
            "images": image_statuses,
            "personalization_applied": bool(request.recipient)
        }
    except Exception as e:
        logger.error(f"Error previewing email: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error previewing email: {str(e)}"
        )

@app.post("/send-email", response_model=SendResponse)
async def send_email(request: SendEmailRequest, background_tasks: BackgroundTasks):
    """Send personalized emails to recipients."""
    try:
        # Get from email (use default if not provided)
        from_email = request.from_email or DEFAULT_FROM_EMAIL
        if not from_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="From email address is required"
            )
        
        # Convert recipients to the format expected by send_emails
        recipients = []
        for r in request.recipients:
            recipient_data = r.personalization.copy() if r.personalization else {}
            recipient_data["EMAIL"] = r.email
            recipients.append(recipient_data)
        
        if not recipients:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No recipients provided"
            )
        
        # Process images
        attachments = []
        for image in request.content.images:
            attachment = create_attachment(image)
            if attachment:
                attachments.append(attachment)
        
        # Enhance HTML content
        html_content = enhance_html_content(request.content.html_content)
        
        # Send emails in background
        background_tasks.add_task(
            send_emails_async,
            from_email=from_email,
            recipients=recipients,
            subject=request.content.subject,
            html_content=html_content,
            attachments=attachments,
            test_mode=request.test_mode
        )
        
        # Return immediate response
        return {
            "success": True,
            "message": f"Processing {len(recipients)} emails {'(test mode)' if request.test_mode else ''}",
            "total": len(recipients),
            "success_count": 0,  # Will be updated in background task
            "failed_count": 0,   # Will be updated in background task
            "failed_emails": None
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error sending email: {str(e)}"
        )

@app.post("/test-email", response_model=SendResponse)
async def test_email(request: SendEmailRequest):
    """Test email sending without actually sending emails."""
    # Force test mode
    request.test_mode = True
    return await send_email(request, BackgroundTasks())

@app.get("/")
async def root():
    """API root / documentation redirect."""
    return {"message": "SendGrid Email API", "docs": "/docs"}

# --- Error Handlers ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR
        }
    )

# --- Main ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sendgrid_backend:app", host="0.0.0.0", port=8000, reload=True) 