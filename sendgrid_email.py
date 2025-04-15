#!/usr/bin/env python3
"""
SendGrid Email Sender - Simple Version

This script provides an easy way to send personalized emails through SendGrid.
Just follow the step-by-step prompts to send emails to multiple recipients.

Usage:
python sendgrid_email.py [--test]

Options:
--test    Run in test mode (no emails will actually be sent)

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
import urllib.parse
import urllib.request
import argparse
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('email_sender.log')
    ]
)
logger = logging.getLogger('email_sender')

# Try to import optional libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Check for SendGrid
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import (
        Mail, Attachment, FileContent, FileName,
        FileType, Disposition, ContentId
    )
except ImportError as e:
    logger.error(f"Failed to import SendGrid: {e}")
    print("\nERROR: SendGrid library not found. Please install it with:")
    print("pip install sendgrid")
    sys.exit(1)

# Load environment variables
load_dotenv()
SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY')
DEFAULT_FROM_EMAIL = os.getenv('FROM_EMAIL')

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 2
EMAIL_REGEX = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

# --- Helper Functions ---

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Send personalized emails using SendGrid')
    parser.add_argument('--test', action='store_true', help='Run in test mode (no emails will be sent)')
    return parser.parse_args()

def validate_email(email):
    """Validates email format using regex."""
    return bool(re.match(EMAIL_REGEX, email))

def get_from_email():
    """Get the sender email address from environment or user input."""
    from_email = DEFAULT_FROM_EMAIL
    
    if not from_email:
        print("\nWhat email address do you want to send from?")
        from_email = input("Your email: ").strip()
        
        while not from_email or not validate_email(from_email):
            print("Please enter a valid email address.")
            from_email = input("Your email: ").strip()
    
    print(f"\nWill send emails from: {from_email}")
    return from_email

def get_csv_file():
    """Get the CSV file containing recipient data."""
    print("\nWhere is your contact list? (CSV file with email addresses)")
    csv_file = input("File path: ").strip()
    
    while not csv_file or not os.path.isfile(csv_file):
        if csv_file:
            print(f"Sorry, couldn't find the file: '{csv_file}'")
        print("Please enter a valid file path.")
        csv_file = input("File path: ").strip()
    
    print(f"\nFound your contact list: {csv_file}")
    return csv_file

def find_email_column(headers):
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

def detect_csv_columns(filename):
    """Detect and map column names in a CSV file."""
    try:
        with open(filename, mode='r', newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile)
            fieldnames = next(reader)  # Get the header row
            
            # Find email column automatically
            email_column = find_email_column(fieldnames)
            
            if not email_column:
                print("\nI couldn't automatically find the email column in your CSV file.")
                print("Here are the columns in your file:")
                for i, field in enumerate(fieldnames, 1):
                    print(f"  {i}. {field}")
                
                choice = input("\nWhich column has the email addresses? (enter the number): ").strip()
                try:
                    column_index = int(choice) - 1
                    if 0 <= column_index < len(fieldnames):
                        email_column = fieldnames[column_index]
                    else:
                        print("Invalid choice. Please run the script again.")
                        return None
                except ValueError:
                    # If they entered the name instead of a number, try to use that
                    if choice in fieldnames:
                        email_column = choice
                    else:
                        print("Invalid choice. Please run the script again.")
                        return None
            
            print(f"\nUsing '{email_column}' column for email addresses.")
            logger.info(f"Selected email column: '{email_column}'")
            return email_column
            
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        print(f"There was a problem reading your CSV file: {str(e)}")
        return None

def get_recipients_from_csv(filename, email_column):
    """Read recipient data from CSV file."""
    recipients = []
    invalid_emails = []
    
    try:
        with open(filename, mode='r', newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            
            if email_column not in reader.fieldnames:
                print(f"Error: Column '{email_column}' not found in the CSV file.")
                print(f"Available columns: {', '.join(reader.fieldnames)}")
                return None
            
            for row_num, row in enumerate(reader, start=2):
                if row and email_column in row:
                    email_address = row[email_column].strip()
                    if email_address:
                        # Validate email format
                        if not validate_email(email_address):
                            invalid_emails.append((row_num, email_address))
                            continue
                        
                        # Store all row data for personalization
                        recipient_data = row.copy()
                        # Store email under standard EMAIL key for consistency
                        recipient_data['EMAIL'] = email_address
                        recipients.append(recipient_data)
            
            # Report on findings
            valid_count = len(recipients)
            invalid_count = len(invalid_emails)
            
            print(f"\nFound {valid_count} valid email addresses in your contact list.")
            
            if invalid_count > 0:
                print(f"Note: {invalid_count} email addresses appear to be invalid and will be skipped.")
            
            # Display personalization fields available
            if valid_count > 0:
                fields = set()
                for recipient in recipients:
                    fields.update(recipient.keys())
                
                fields = [f for f in fields if f != email_column and f != 'EMAIL']
                
                if fields:
                    print("\nYou can personalize your email with these fields:")
                    for field in sorted(fields):
                        print(f"  • {field} - Use {{{{PLACEHOLDER}}}} in your message")
                    print("\nExample: 'Hello {{FIRST_NAME}}' will be customized for each recipient")
                
            return recipients if valid_count > 0 else None
            
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        print(f"There was a problem reading your contact list: {str(e)}")
        return None

def show_html_help():
    """Show helpful examples for HTML content."""
    print("\nHTML FORMATTING TIPS:")
    print("-----------------")
    print("• For a new paragraph: <p>Your text here</p>")
    print("• For bold text: <b>Bold text</b>")
    print("• For a link: <a href=\"https://example.com\">Click here</a>")
    print("• For an image: <img src=\"cid:image_name\" alt=\"Description\">")
    print("• For a heading: <h1>Your heading</h1>")
    print("• For a list:")
    print("  <ul>")
    print("    <li>Item 1</li>")
    print("    <li>Item 2</li>")
    print("  </ul>")
    print("-----------------")

def get_email_content():
    """Get email subject, body, and images from user input."""
    print("\n===== STEP 3: CREATE YOUR EMAIL =====")
    
    # Get subject
    subject = input("\nEmail subject line: ").strip()
    while not subject:
        print("Subject cannot be empty.")
        subject = input("Email subject line: ").strip()
    
    # Offer HTML help
    print("\nDo you want to see some HTML formatting tips? (y/n)")
    if input().strip().lower() in ['y', 'yes']:
        show_html_help()
    
    # Get HTML content
    print("\nEnter your email content below.")
    print("You can use HTML formatting for styling.")
    print("When you're finished, press Enter twice (on an empty line).")
    print("\nSTART TYPING YOUR EMAIL CONTENT:")
    html_lines = []
    
    try:
        line = input()
        while True:
            html_lines.append(line)
            line = input()
            if line == "":
                break
    except EOFError:
        pass
    
    html_content = "\n".join(html_lines)
    
    while not html_content.strip():
        print("Email content cannot be empty.")
        print("Please type your email content (press Enter twice when finished):")
        html_lines = []
        try:
            line = input()
            while True:
                html_lines.append(line)
                line = input()
                if line == "":
                    break
        except EOFError:
            pass
        html_content = "\n".join(html_lines)
    
    # Preview the content
    print("\n----- Email Content Preview -----")
    print(html_content)
    print("---------------------------------")
    
    # Get images
    images = []
    print("\nDo you want to add images to your email? (y/n)")
    add_images = input().strip().lower() in ['y', 'yes']
    
    if add_images:
        print("\nADDING IMAGES:")
        print("For each image, you'll need to provide the file path or URL.")
        print("Each image will get a unique ID to use in your HTML.")
        
        image_count = 0
        while True:
            image_count += 1
            print(f"\nIMAGE #{image_count}:")
            print("Enter file path or URL (or leave empty to finish):")
            image_path = input().strip()
            
            if not image_path:
                break
            
            image_cid = f"image_{image_count}"
            
            # Check if path is valid
            is_url = image_path.startswith(('http://', 'https://'))
            if not is_url and not os.path.isfile(image_path):
                print(f"Warning: File '{image_path}' not found.")
                retry = input("Try a different path? (y/n): ").strip().lower()
                if retry == 'y':
                    image_count -= 1  # Don't count this attempt
                    continue
            
            images.append({
                'path': image_path,
                'content_id': image_cid
            })
            
            print(f"✓ Image added with ID: {image_cid}")
            print(f"  To show this image in your email, add this HTML:")
            print(f"  <img src=\"cid:{image_cid}\" alt=\"Image {image_count}\">")
            
            if image_count >= 1:
                print(f"\nAdd another image? (y/n)")
                add_another = input().strip().lower() in ['y', 'yes']
                if not add_another:
                    break
    
    if images:
        print(f"\nAdded {len(images)} images to your email.")
        
        # Check if the HTML doesn't use the images and suggest edits
        missing_images = []
        for img in images:
            if f"cid:{img['content_id']}" not in html_content:
                missing_images.append(img)
        
        if missing_images:
            print("\nWARNING: Some images are not referenced in your content!")
            print("Do you want to update your email content to include these images? (y/n)")
            if input().strip().lower() in ['y', 'yes']:
                print("\nEnter your updated email content below.")
                print("Make sure to include image tags for each image:")
                for img in images:
                    print(f"  <img src=\"cid:{img['content_id']}\" alt=\"Image\">")
                
                print("\nSTART TYPING YOUR UPDATED EMAIL CONTENT:")
                html_lines = []
                try:
                    line = input()
                    while True:
                        html_lines.append(line)
                        line = input()
                        if line == "":
                            break
                except EOFError:
                    pass
                
                new_content = "\n".join(html_lines)
                if new_content.strip():
                    html_content = new_content
    else:
        print("\nNo images added to your email.")
    
    return {
        'subject': subject,
        'html_content': html_content,
        'images': images
    }

def create_attachment(image_path, content_id):
    """Creates a SendGrid Attachment object from an image file path or URL."""
    if not image_path:
        return None

    logger.info(f"Processing image: '{image_path}'")
    
    # Check if the path is a URL
    is_url = image_path.startswith(('http://', 'https://'))
    
    # Check if the data is a base64 string
    is_base64 = False
    if (image_path.startswith(('data:image/', 'data:application/')) and 
        ';base64,' in image_path):
        is_base64 = True
    
    try:
        if is_base64:
            # Handle base64 encoded image
            try:
                # Extract content type and data
                _, data_part = image_path.split(',', 1)
                content_type = image_path.split(',')[0].split(':')[1].split(';')[0]
                
                # Decode base64 data
                data = base64.b64decode(data_part)
                
                # Generate a filename
                extension = content_type.split('/')[-1]
                filename = f"image_{content_id}.{extension}"
                
            except Exception as e:
                logger.error(f"Error processing base64 image: {e}")
                return None
                
        elif is_url:
            # Download the image from URL
            try:
                if REQUESTS_AVAILABLE:
                    response = requests.get(image_path, timeout=10)
                    if response.status_code != 200:
                        logger.error(f"Failed to download image: {response.status_code}")
                        return None
                    data = response.content
                    
                    # Get filename from URL
                    filename = os.path.basename(urllib.parse.urlparse(image_path).path)
                    if not filename:
                        filename = f"image_{content_id}"
                    
                    content_type = response.headers.get('Content-Type')
                else:
                    # Fallback to urllib
                    with urllib.request.urlopen(image_path) as response:
                        data = response.read()
                    filename = os.path.basename(urllib.parse.urlparse(image_path).path)
                    if not filename:
                        filename = f"image_{content_id}"
                    content_type = response.headers.get('Content-Type')
            except Exception as e:
                logger.error(f"Error downloading image: {e}")
                return None
        else:
            # Handle local file path
            if not os.path.isfile(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None

            # Read image data
            with open(image_path, 'rb') as f:
                data = f.read()
                
            # Use the filename from the path
            filename = os.path.basename(image_path)
            content_type = None

        # Check file size
        file_size_kb = len(data) / 1024
        if file_size_kb > 10240:  # 10MB limit
            logger.warning(f"Image too large: {file_size_kb:.2f}KB - SendGrid limit is 10MB")
            return None
        elif file_size_kb > 5120:  # 5MB warning
            logger.warning(f"Large image warning: {file_size_kb:.2f}KB - may cause delivery issues")

        # Encode to Base64
        encoded_file = base64.b64encode(data).decode()

        # Determine MIME type if needed
        if content_type is None:
            file_type, _ = mimetypes.guess_type(image_path)
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
            ContentId(content_id)
        )
        logger.info(f"Image processed successfully: {content_id}")
        return attachment

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

def process_personalization(html_content, recipient_data):
    """Replace personalization placeholders with recipient data."""
    if not html_content or not recipient_data:
        return html_content
        
    personalized_content = html_content
    
    # Find all placeholders
    placeholders = set(re.findall(r'\{\{([^}]+)\}\}', personalized_content))
    replaced = set()
    
    # Replace placeholders with data
    for key, value in recipient_data.items():
        if not value:  # Skip empty values
            continue
            
        # Try different case variations
        for key_format in [key, key.upper(), key.lower(), key.title(), key.capitalize()]:
            placeholder = "{{" + key_format + "}}"
            if placeholder in personalized_content:
                personalized_content = personalized_content.replace(placeholder, str(value))
                replaced.add(key_format)
    
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
                    replaced.add(var_format)
    
    return personalized_content

def enhance_html_content(html_content):
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

def send_emails(sg_client, from_email, recipients, subject, html_content, attachments=None, test_mode=False):
    """Send personalized emails to recipients with attachments."""
    total_recipients = len(recipients)
    success_count = 0
    failed_emails = []
    
    logger.info(f"Starting email sending process {'(TEST MODE)' if test_mode else ''}")
    
    # Check for sender name format "Name <email@example.com>"
    from_name = None
    if " <" in from_email and from_email.endswith(">"):
        from_name = from_email.split(" <")[0]
        from_email = from_email.split(" <")[1].rstrip(">")
    
    print(f"\nSending emails to {total_recipients} recipients...")
    if test_mode:
        print("(Test mode: No actual emails will be sent)")
    start_time = time.time()
    
    # Process each recipient
    for i, recipient in enumerate(recipients, 1):
        email = recipient.get('EMAIL')
        
        # Skip invalid emails
        if not email or not validate_email(email):
            logger.warning(f"Skipping invalid email: {email}")
            failed_emails.append((email, "Invalid email format"))
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
                            
                            # Progress update
                            if total_recipients > 10 and i % 10 == 0:
                                print(f"Progress: {i}/{total_recipients} emails sent")
                        else:
                            logger.error(f"Failed to send to {email}: {status_code}")
                            retries += 1
                    except Exception as e:
                        logger.error(f"Error sending to {email}: {str(e)}")
                        retries += 1
                
                if not success:
                    failed_emails.append((email, f"Failed after {retries} attempts"))
            else:
                # Test mode - just log
                logger.info(f"TEST MODE - Would send to: {email}")
                success_count += 1
                
                # Progress update
                if total_recipients > 10 and i % 10 == 0:
                    print(f"Progress: {i}/{total_recipients} processed (TEST MODE)")
            
            # Small pause to avoid rate limits
            time.sleep(0.2)
            
        except Exception as e:
            logger.error(f"Error processing {email}: {str(e)}")
            failed_emails.append((email, str(e)))
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Report results
    print("\n===== EMAIL SENDING COMPLETE =====")
    print(f"Total recipients: {total_recipients}")
    print(f"Successfully sent: {success_count}")
    print(f"Failed: {len(failed_emails)}")
    print(f"Time taken: {duration:.1f} seconds")
    
    if failed_emails:
        print("\nFailed emails:")
        for email, reason in failed_emails[:5]:
            print(f"  • {email}: {reason}")
        if len(failed_emails) > 5:
            print(f"  • ... and {len(failed_emails) - 5} more")
    
    if test_mode:
        print("\nTest completed successfully. No real emails were sent.")
    
    return success_count, failed_emails

def initialize_sendgrid():
    """Initialize SendGrid client."""
    if not SENDGRID_API_KEY:
        print("\nERROR: SendGrid API key not found.")
        print("Please create a .env file with your API key:")
        print("SENDGRID_API_KEY=your_api_key_here")
        return None
    
    try:
        logger.info("Initializing SendGrid")
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        return sg
    except Exception as e:
        logger.error(f"SendGrid initialization failed: {e}")
        print(f"\nERROR: Could not connect to SendGrid: {str(e)}")
        return None

def main():
    """Main function to run the email sending process."""
    # Parse command line arguments
    args = parse_args()
    test_mode = args.test
    
    print("\n==================================")
    print("       Email Sender Tool          ")
    print("==================================")
    
    if test_mode:
        print("\nRunning in TEST MODE - No actual emails will be sent")
    
    # Step 1: Initialize SendGrid
    print("\n===== STEP 1: CHECKING CONNECTION =====")
    sg_client = initialize_sendgrid()
    if not sg_client:
        return False
    
    # Step 2: Get sender email
    print("\n===== STEP 2: SENDER INFORMATION =====")
    from_email = get_from_email()
    
    # Step 3: Get CSV file with recipients
    print("\n===== STEP 3: RECIPIENT LIST =====")
    csv_file = get_csv_file()
    
    # Step 4: Detect CSV columns
    email_column = detect_csv_columns(csv_file)
    if not email_column:
        return False
    
    # Step 5: Get recipient data
    recipients = get_recipients_from_csv(csv_file, email_column)
    if not recipients:
        print("No valid recipients found in your contact list.")
        return False
    
    # Step 6: Get email content
    content = get_email_content()
    subject = content['subject']
    html_content = content['html_content']
    images = content['images']
    
    # Step 7: Prepare email
    print("\n===== STEP 4: PREPARING EMAIL =====")
    
    # Enhance HTML content
    html_content = enhance_html_content(html_content)
    
    # Process images
    print("Processing images...")
    attachments = []
    for image in images:
        print(f"  • Processing image: {image['path']}")
        attachment = create_attachment(image['path'], image['content_id'])
        if attachment:
            attachments.append(attachment)
            print(f"    ✓ Image processed successfully")
        else:
            print(f"    ✗ Failed to process image")
    
    # Step 8: Show email preview
    print("\n===== EMAIL PREVIEW =====")
    print(f"From: {from_email}")
    print(f"To: {len(recipients)} recipients")
    print(f"Subject: {subject}")
    print(f"Images: {len(attachments)}/{len(images)}")
    
    # Preview content (just the beginning)
    content_preview = html_content.replace("<html>", "").replace("</html>", "")
    content_preview = re.sub(r'<[^>]*>', ' ', content_preview)  # Remove HTML tags
    content_preview = re.sub(r'\s+', ' ', content_preview).strip()  # Clean up whitespace
    preview_length = min(100, len(content_preview))
    print(f"Content: {content_preview[:preview_length]}...")
    
    # Check for personalization
    if "{{" in html_content and "}}" in html_content:
        placeholders = re.findall(r'\{\{([^}]+)\}\}', html_content)
        print("\nPersonalization: Using the following fields:")
        for placeholder in sorted(set(placeholders)):
            print(f"  • {placeholder}")
    
    # Confirm before sending (only if not in test mode)
    if not test_mode:
        print("\nReady to send emails to real recipients.")
        print(f"This will send to {len(recipients)} email addresses.")
        print("Do you want to proceed? (y/n)")
        confirm = input().strip().lower() in ['y', 'yes']
        if not confirm:
            print("Operation cancelled. No emails were sent.")
            return True
    
    # Send emails
    success_count, failed_emails = send_emails(
        sg_client=sg_client,
        from_email=from_email,
        recipients=recipients,
        subject=subject,
        html_content=html_content,
        attachments=attachments,
        test_mode=test_mode
    )
    
    # If in test mode, ask about sending for real
    if test_mode and success_count > 0:
        print("\nTest completed successfully.")
        print("Would you like to send the actual emails now? (y/n)")
        send_real = input().strip().lower() in ['y', 'yes']
        if send_real:
            print("\nSending actual emails...")
            success_count, failed_emails = send_emails(
                sg_client=sg_client,
                from_email=from_email,
                recipients=recipients,
                subject=subject,
                html_content=html_content,
                attachments=attachments,
                test_mode=False
            )
    
    print("\n==================================")
    print("          All Done!              ")
    print("==================================")
    
    return True

def simple_main(from_email, csv_file_path, subject, html_body, image_paths=None, test_mode=True):
    """
    Simplified function to send emails with all parameters provided directly.
    
    Args:
        from_email (str): Sender email address
        csv_file_path (str): Path to CSV file with recipient data
        subject (str): Email subject line
        html_body (str): Email content in HTML format
        image_paths (list): List of paths to images to attach
        test_mode (bool): If True, emails won't actually be sent
    
    Returns:
        bool: True if process completed successfully
    """
    print(f"\nRunning email sender with predefined parameters {'(TEST MODE)' if test_mode else ''}")
    
    # Initialize SendGrid
    sg_client = initialize_sendgrid()
    if not sg_client:
        return False
    
    # Detect CSV columns
    email_column = detect_csv_columns(csv_file_path)
    if not email_column:
        return False
    
    # Get recipient data
    recipients = get_recipients_from_csv(csv_file_path, email_column)
    if not recipients:
        print("No valid recipients found in your contact list.")
        return False
    
    # Prepare images
    attachments = []
    images_info = []
    
    if image_paths:
        for i, image_path in enumerate(image_paths, 1):
            image_cid = f"image_{i}"
            images_info.append({
                'path': image_path,
                'content_id': image_cid
            })
            
            print(f"Processing image: {image_path}")
            attachment = create_attachment(image_path, image_cid)
            if attachment:
                attachments.append(attachment)
                print(f"✓ Image processed with ID: {image_cid}")
                print(f"  To reference this image in your email: <img src=\"cid:{image_cid}\" alt=\"Image {i}\">")
    
    # Enhance HTML content
    html_content = enhance_html_content(html_body)
    
    # Send emails
    success_count, failed_emails = send_emails(
        sg_client=sg_client,
        from_email=from_email,
        recipients=recipients,
        subject=subject,
        html_content=html_content,
        attachments=attachments,
        test_mode=test_mode
    )
    
    return True

if __name__ == "__main__":
    try:
        # Run the interactive version by default
        # main()
        
        # Use the simple_main function with direct parameters
        simple_main(
            from_email="email@example.com",
            csv_file_path="recipients.csv",
            subject="Your Important Email",
            html_body="<p>Hello {{FIRST_NAME}},</p><p>This is a test email with <b>HTML formatting</b>.</p><p>Check out this image: </p>",
            image_paths=[
            "https://images.unsplash.com/photo-1741462166411-b94730c55171?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxmZWF0dXJlZC1waG90b3MtZmVlZHw5fHx8ZW58MHx8fHx8", 
            "https://images.unsplash.com/photo-1741762764258-8f9348bdf186?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxmZWF0dXJlZC1waG90b3MtZmVlZHwxMXx8fGVufDB8fHx8fA%3D%3D"
            ],
            test_mode=False  # Set to False to actually send emails
        )
    except KeyboardInterrupt:
        logger.info("User cancelled operation")
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {str(e)}")
        print("Check the email_sender.log file for details.")
        sys.exit(1) 