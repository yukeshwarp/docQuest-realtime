import fitz
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.file_conversion import convert_office_to_pdf
from utils.llm_interaction import summarize_page, get_image_explanation, generate_system_prompt, llm_extract_sections_paragraphs_tables
import io
import base64
import logging
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords once
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Set up logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s [%(levelname)s] %(message)s")
generated_system_prompt = None

def remove_stopwords_and_blanks(text):
    """Preprocess text by removing stopwords, punctuation, and extra blank spaces."""
    # Remove punctuation and extra spaces
    text = text.translate(str.maketrans('', '', string.punctuation)).strip()
    filtered_text = ' '.join([word for word in text.split() if word.lower() not in stop_words and word])
    return filtered_text

def sanitize_text(text):
    """Sanitize text to ensure it doesn't contain problematic characters."""
    text = text.replace("'", "\\'").replace('"', '\\"')
    return text

def detect_ocr_images_and_vector_graphics_in_pdf(page, ocr_text_threshold=0.4):
    """Detect OCR images or vector graphics on a given PDF page."""
    try:
        images = page.get_images(full=True)
        text_blocks = page.get_text("blocks")
        vector_graphics_detected = bool(page.get_drawings())

        # Calculate text coverage
        page_area = page.rect.width * page.rect.height
        text_area = sum((block[2] - block[0]) * (block[3] - block[1]) for block in text_blocks)
        text_coverage = text_area / page_area if page_area > 0 else 0

        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        base64_image = base64.b64encode(img_data).decode("utf-8")
        pix = None  # Free up memory for pixmap

        if (images or vector_graphics_detected) and text_coverage < ocr_text_threshold:
            return base64_image  # Return image data if OCR image or vector graphics detected

    except Exception as e:
        logging.error(f"Error detecting OCR images/graphics on page {page.number}: {e}")
    
    return None

def process_page_batch(pdf_document, batch, system_prompt, ocr_text_threshold=0.4):
    """Process a batch of PDF pages and extract summaries, full text, and image analysis."""
    previous_summary = ""
    batch_data = []

    for page_number in batch:
        try:
            page = pdf_document.load_page(page_number)
            text = page.get_text("text").strip()
            logging.debug(f"Extracted text from page {page_number + 1}: {text[:100]}...")  # Log the first 100 characters
            summary = ""
            
            # Summarize the page after preprocessing
            if text != "":
                preprocessed_text = remove_stopwords_and_blanks(text)
                sanitized_text = sanitize_text(preprocessed_text)  # Sanitize the text
                summary = summarize_page(sanitized_text, previous_summary, page_number + 1, system_prompt)
                previous_summary = summary
            
            # Detect images or vector graphics
            image_data = detect_ocr_images_and_vector_graphics_in_pdf(page, ocr_text_threshold)
            image_analysis = []
            if image_data:
                image_explanation = get_image_explanation(image_data)
                image_analysis.append({"page_number": page_number + 1, "explanation": image_explanation})

            # Extract sections, headings, paragraphs, and tables using the LLM
            structured_data = llm_extract_sections_paragraphs_tables(sanitized_text)  # Assuming this includes tables and figures

            # Store the extracted data, including the structured JSON
            batch_data.append({
                "page_number": page_number + 1,
                "full_text": text,
                "text_summary": summary,
                "structured_data": structured_data,  # Include structured data
                "image_analysis": image_analysis
            })

        except Exception as e:
            logging.error(f"Error processing page {page_number + 1}: {str(e)} (Type: {type(e).__name__})")
            batch_data.append({
                "page_number": page_number + 1,
                "full_text": "",  # Include empty text in case of an error
                "text_summary": "Error in processing this page",
                "structured_data": {},  # Empty structured data in case of an error
                "image_analysis": []
            })

    return batch_data

def process_pdf_pages(uploaded_file, first_file=False):
    """Process the PDF pages in batches and extract summaries and image analysis."""
    global generated_system_prompt
    file_name = uploaded_file.name
    
    try:
        # Check if the uploaded file is a PDF
        if file_name.lower().endswith('.pdf'):
            pdf_stream = io.BytesIO(uploaded_file.read())  # Directly read PDF
        else:
            # Convert Office files to PDF if necessary
            pdf_stream = convert_office_to_pdf(uploaded_file)
        
        # Process the PDF document
        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
        document_data = {"document_name": file_name, "pages": []}  # Add document_name at the top
        total_pages = len(pdf_document)
        full_text = ""
        
        # If it's the first file, generate the system prompt
        if first_file and generated_system_prompt is None:
            for page_number in range(total_pages):
                page = pdf_document.load_page(page_number)
                full_text += page.get_text("text").strip() + " "  # Concatenate all text
                if len(full_text.split()) >= 200:
                    break
            # Use the first 200 words for the system prompt
            first_200_words = ' '.join(full_text.split()[:200])
            generated_system_prompt = generate_system_prompt(first_200_words)

        # Batch size of 5 pages
        batch_size = 5
        page_batches = [range(i, min(i + batch_size, total_pages)) for i in range(0, total_pages, batch_size)]
        
        # Use ThreadPoolExecutor to process batches concurrently
        with ThreadPoolExecutor() as executor:
            future_to_batch = {executor.submit(process_page_batch, pdf_document, batch, generated_system_prompt): batch for batch in page_batches}
            for future in as_completed(future_to_batch):
                try:
                    batch_data = future.result()  # Get the result of processed batch
                    document_data["pages"].extend(batch_data)
                except Exception as e:
                    logging.error(f"Error processing batch: {str(e)} (Type: {type(e).__name__})")

        # Close the PDF document after processing
        pdf_document.close()

        # Sort pages by page_number to ensure correct order
        document_data["pages"].sort(key=lambda x: x["page_number"])
        return document_data

    except Exception as e:
        logging.error(f"Error processing PDF file {file_name}: {str(e)} (Type: {type(e).__name__})")
        raise ValueError(f"Unable to process the file {file_name}. Error: {str(e)}")
