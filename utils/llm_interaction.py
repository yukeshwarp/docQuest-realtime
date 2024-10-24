import os
from dotenv import load_dotenv
import requests
from utils.config import azure_endpoint, api_key, api_version, model
import logging
import time
import requests
import random
import re
import nltk
from nltk.corpus import stopwords
import json
import concurrent.futures

logging.basicConfig(level=logging.ERROR, format="%(asctime)s [%(levelname)s] %(message)s")



nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    text = text.lower()    
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

def get_headers():
    """Generate common headers for the API requests."""
    return {
        "Content-Type": "application/json",
        "api-key": api_key
    }


def get_image_explanation(base64_image, retries=5, initial_delay=2):
    """Get image explanation from OpenAI API with exponential backoff."""
    headers = get_headers()
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that responds in Markdown."},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": "Explain the contents and figures or tables if present of this image of a document page. The explanation should be concise and semantically meaningful. Do not make assumptions about the specification and be accurate in your explanation."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]}
        ],
        "temperature": 0.0
    }

    url = f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}"

    
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)  
            response.raise_for_status()  
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', "No explanation provided.")
        
        except requests.exceptions.Timeout as e:
            if attempt < retries - 1:
                wait_time = initial_delay * (2 ** attempt)  
                logging.warning(f"Timeout error. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(wait_time)
            else:
                logging.error(f"Request failed after {retries} attempts due to timeout: {e}")
                return f"Error: Request timed out after {retries} retries."

        except requests.exceptions.RequestException as e:
            logging.error(f"Error requesting image explanation: {e}")
            return f"Error: Unable to fetch image explanation due to network issues or API error."

    return "Error: Max retries reached without success."

import requests
import logging

def llm_extract_sections_paragraphs_tables(text):
    headers = get_headers()
    preprocessed_text = preprocess_text(text)
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert in document structure analysis. Don't provide any other addon text to the output provide only the structure and dont provide the format name of the structure in the response."},
            {"role": "user", "content":
             f"""
             You are provided with the following document text. Based on its content, extract and identify the following details:
             Text: {preprocessed_text}

             1. Divide the text into sections.
             2. Identify headings, paragraphs, and tables.
             3. Return the output as a structured JSON format with sections, headings, paragraphs, and tables.

             Example JSON structure:
             {{
                 "sections": [
                     {{
                         "heading": "Heading Title",
                         "paragraphs": ["Paragraph 1 content", "Paragraph 2 content"],
                         "tables": ["Table content as text"]
                     }},
                     ...
                 ]
             }}
             """
            }
        ],
        "temperature": 0.5  
    }

    try:
        response = requests.post(
                f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
                headers=headers,
                json=data,
                timeout=60
            )
        
        response.raise_for_status()  
        prompt_response = response.json().get('choices', [{}])[0].get('message', {}).get('content', "")
        prompt_res = prompt_response.strip()
        json_data = json.loads(prompt_res)
        return json_data
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error during LLM extraction of sections and tables: {e}")
        return {
            "sections": [],
            "tables": []
        }


def generate_system_prompt(document_content):
    """
    Generate a system prompt based on the expertise, tone, and voice needed 
    to summarize the document content.
    """
    headers = get_headers()
    preprocessed_content = preprocess_text(document_content)
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that serves the task given."},
            {"role": "user", "content":
             f"""You are provided with a document. Based on its content, extract and identify the following details:
            Document_content: {preprocessed_content}

            1. **Domain**: Identify the specific domain or field of expertise the document is focused on. Examples include technology, finance, healthcare, law, etc.
            2. **Subject Matter**: Determine the main topic or focus of the document. This could be a detailed concept, theory, or subject within the domain.
            3. **Experience**: Based on the content, infer the level of experience required to understand or analyze the document (e.g., novice, intermediate, expert).
            4. **Expertise**: Identify any specialized knowledge, skills, or proficiency in a particular area that is necessary to evaluate the content.
            5. **Educational Qualifications**: Infer the level of education or qualifications expected of someone who would need to review or write the document (e.g., PhD, Master's, Bachelor's, or certification in a field).
            6. **Style**: Describe the writing style of the document. Is it formal, technical, conversational, academic, or instructional?
            7. **Tone**: Identify the tone used in the document. For example, is it neutral, authoritative, persuasive, or informative?
            8. **Voice**: Analyze whether the voice is active, passive, first-person, third-person, or impersonal, and whether it's personal or objective.

            After extracting this information, use it to fill in the following template:
    
            ---

            You are now assuming a persona based on the content of the provided document. Your persona should reflect the <domain> and <subject matter> of the content, with the requisite <experience>, <expertise>, and <educational qualifications> to analyze the document effectively. Additionally, you should adopt the <style>, <tone> and <voice> present in the document. Your expertise includes:
    
            <Domain>-Specific Expertise:
            - In-depth knowledge and experience relevant to the <subject matter> of the document.
            - Familiarity with the key concepts, terminology, and practices within the <domain>.
            
            Analytical Proficiency:
            - Skilled in interpreting and evaluating the content, structure, and purpose of the document.
            - Ability to assess the accuracy, clarity, and completeness of the information presented.
    
            Style, Tone, and Voice Adaptation:
            - Adopt the writing <style>, <tone>, and <voice> used in the document to ensure consistency and coherence.
            - Maintain the level of formality, technicality, or informality as appropriate to the document’s context.
            
            Your analysis should include:
            - A thorough evaluation of the content, ensuring it aligns with <domain>-specific standards and practices.
            - An assessment of the clarity and precision of the information and any accompanying diagrams or illustrations.
            - Feedback on the strengths and potential areas for improvement in the document.
            - A determination of whether the document meets its intended purpose and audience requirements.
            - Proposals for any necessary amendments or enhancements to improve the document’s effectiveness and accuracy.
        
            ---

            Generate a response filling the template with appropriate details based on the content of the document and return the filled in template as response."""}
        ],
        "temperature": 0.5  
    }

    try:
        response = requests.post(
            f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
            headers=headers,
            json=data,
            timeout=20
        )
        response.raise_for_status()
        prompt_response = response.json().get('choices', [{}])[0].get('message', {}).get('content', "")
        return prompt_response.strip()

    except requests.exceptions.RequestException as e:
        logging.error(f"Error generating system prompt: {e}")
        return f"Error: Unable to generate system prompt due to network issues or API error."


def summarize_page(page_text, previous_summary, page_number, system_prompt, max_retries=5, base_delay=1, max_delay=32):
    """
    Summarize a single page's text using LLM, and generate a system prompt based on the document content.
    Implements exponential backoff with jitter to handle timeout errors.
    """
    headers = get_headers()
    preprocessed_page_text = preprocess_text(page_text)
    preprocessed_previous_summary = preprocess_text(previous_summary)
    
    system_prompt = system_prompt
    
    prompt_message = (
        f"Please rewrite the following page content from (Page {page_number}) along with context from the previous page summary "
        f"to make them concise and well-structured. Maintain proper listing and referencing of the contents if present."
        f"Do not add any new information or make assumptions. Keep the meaning accurate and the language clear.\n\n"
        f"Previous page summary: {preprocessed_previous_summary}\n\n"
        f"Current page content:\n{preprocessed_page_text}\n"
    )

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_message}
        ],
        "temperature": 0.0
    }
    
    attempt = 0
    while attempt < max_retries:
        try:
            response = requests.post(
                f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
                headers=headers,
                json=data,
                timeout=50
            )
            response.raise_for_status()
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', "No summary provided.").strip()
        
        except requests.exceptions.RequestException as e:
            attempt += 1
            if attempt >= max_retries:
                logging.error(f"Error summarizing page {page_number}: {e}")
                return f"Error: Unable to summarize page {page_number} due to network issues or API error."

            
            delay = min(max_delay, base_delay * (2 ** attempt))  
            jitter = random.uniform(0, delay)  
            logging.warning(f"Retrying in {jitter:.2f} seconds (attempt {attempt}) due to error: {e}")
            time.sleep(jitter)


def llm_check_relevance(prompt):
    """Check the relevance of a given text content to a question using the LLM."""
    headers = get_headers()  
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a relevance-checking assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }

    try:
        response = requests.post(
            f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
            headers=headers,
            json=data,
            timeout=60  
        )
        response.raise_for_status()  
        relevance_result = response.json().get('choices', [{}])[0].get('message', {}).get('content', "").strip()        
        logging.info(f"Relevance check result: {relevance_result}")

        return "yes" in relevance_result.lower()  

    except requests.exceptions.RequestException as e:
        logging.error(f"Error checking relevance: {e}")
        return False  


def fetch_page(doc_data, question="basic placeholder"):
    relevant_content = []

    def process_page(page):
        page_number = page["page_number"]
        structured_data = page.get("structured_data", {})

        if 'sections' in structured_data.keys():
            for section in structured_data['sections']:
                heading = section.get("heading", "No Heading")
                for paragraph in section.get("paragraphs", []):
                    relevance_prompt = f"Is the following paragraph relevant to the question: '{question}'? Paragraph: '{paragraph}'"
                    if llm_check_relevance(relevance_prompt):
                        return {
                            "page_number": int(page_number),
                            "heading": heading,
                            "paragraph": paragraph
                        }
        return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_page, page) for page in doc_data["pages"]]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                relevant_content.append(result)

    return relevant_content


def fetch_sections(doc_data, question):
    """Check relevance of page summaries and return full text if relevant."""
    relevant_texts = []

    def process_page(page):
        page_number = page["page_number"]
        page_summary = page.get("text_summary", "")
        relevance_prompt = f"Is the following page summary relevant to the question: '{question}'? Summary: '{page_summary}'"
        
        if llm_check_relevance(relevance_prompt):
            return {
                "page_number": int(page_number),
                "full_text": page.get("full_text", "")
            }
        return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_page, page) for page in doc_data["pages"]]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                relevant_texts.append(result)

    return relevant_texts


def fetch_table(doc_data, question):
    relevant_tables = []

    def process_page(page):
        page_number = page["page_number"]
        structured_data = page.get("structured_data", {})
        
        if 'sections' in structured_data.keys():
            for section in structured_data['sections']:
                for table in section.get("tables", []):
                    relevance_prompt = f"Does this table contain relevant data for the question: '{question}'? Table: {table}"
                    if llm_check_relevance(relevance_prompt):
                        return {
                            "page_number": int(page_number),
                            "table": table
                        }
        return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_page, page) for page in doc_data["pages"]]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                relevant_tables.append(result)

    return relevant_tables


def fetch_figures(doc_data, question):
    """Check relevance of image explanations and return page number and image explanation if relevant."""
    relevant_figures = []

    def process_page(page):
        page_number = page["page_number"]
        image_analysis = page.get("image_analysis", [])

        for image in image_analysis:
            relevance_prompt = f"Is this image explanation relevant to the question: '{question}'? Explanation: '{image['explanation']}'"
            if llm_check_relevance(relevance_prompt):
                return {
                    "page_number": int(page_number),
                    "explanation": image["explanation"]
                }
        return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_page, page) for page in doc_data["pages"]]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                relevant_figures.append(result)

    return relevant_figures

def ask_question(documents, question, chat_history):
    """Answer a question based on relevant content from multiple PDFs and chat history."""
    structured_relevant_content = {
        "pages": {}
    }

    # Iterate through each document
    for doc_name, doc_data in documents.items():
        
        # Fetch relevant content for each aspect
        relevant_pages = fetch_page(doc_data, question)
        relevant_sections = fetch_sections(doc_data, question)
        relevant_tables = fetch_table(doc_data, question)
        relevant_figures = fetch_figures(doc_data, question)

        # Combine all fetched results
        for page in relevant_pages:
            page_number = page["page_number"]
            if page_number not in structured_relevant_content["pages"]:
                structured_relevant_content["pages"][page_number] = {
                    "headings_and_paragraphs": [],
                    "summaries": [],
                    "tables": [],
                    "figures": []
                }
            structured_relevant_content["pages"][page_number]["headings_and_paragraphs"].append({
                "heading": page.get("heading", "No Heading"),
                "paragraph": page.get("paragraph", "No Paragraph")
            })

        for section in relevant_sections:
            page_number = section["page_number"]
            if page_number not in structured_relevant_content["pages"]:
                structured_relevant_content["pages"][page_number] = {
                    "headings_and_paragraphs": [],
                    "summaries": [],
                    "tables": [],
                    "figures": []
                }
            structured_relevant_content["pages"][page_number]["summaries"].append(section.get("full_text", "No Full Text"))

        for table in relevant_tables:
            page_number = table["page_number"]
            if page_number not in structured_relevant_content["pages"]:
                structured_relevant_content["pages"][page_number] = {
                    "headings_and_paragraphs": [],
                    "summaries": [],
                    "tables": [],
                    "figures": []
                }
            structured_relevant_content["pages"][page_number]["tables"].append(table.get("table", "No Table Data"))

        for figure in relevant_figures:
            page_number = figure["page_number"]
            if page_number not in structured_relevant_content["pages"]:
                structured_relevant_content["pages"][page_number] = {
                    "headings_and_paragraphs": [],
                    "summaries": [],
                    "tables": [],
                    "figures": []
                }
            structured_relevant_content["pages"][page_number]["figures"].append(figure.get("explanation", "No Explanation"))

    # Sort content by page number
    sorted_pages = sorted(structured_relevant_content["pages"].items())

    # Construct the prompt with ordered content
    prompt_message = "You have the following relevant content to answer the question:\n\n---\n"
    
    for page_number, content in sorted_pages:
        prompt_message += f"Page {page_number}:\n"
        
        if content["summaries"]:
            prompt_message += f"  Summaries: {content['summaries']}\n"
        
        if content["headings_and_paragraphs"]:
            prompt_message += f"  Headings and Paragraphs:\n"
            for item in content["headings_and_paragraphs"]:
                prompt_message += f"    - Heading: {item['heading']}\n"
                prompt_message += f"      Paragraph: {item['paragraph']}\n"
        
        if content["tables"]:
            prompt_message += f"  Tables: {content['tables']}\n"
        
        if content["figures"]:
            prompt_message += f"  Figures: {content['figures']}\n"
        
        prompt_message += "\n"
    
    prompt_message += "---\n"
    prompt_message += f"Question: {question}\n"

    # Send the constructed prompt to the LLM
    headers = get_headers()
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an assistant that answers questions based on the provided relevant content."},
            {"role": "user", "content": prompt_message}
        ],
        "temperature": 0.0
    }

    try:
        response = requests.post(
            f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()

        return response.json().get('choices', [{}])[0].get('message', {}).get('content', "No answer provided.").strip()

    except requests.exceptions.RequestException as e:
        logging.error(f"Error answering question '{question}': {e}")
        raise Exception(f"Unable to answer the question due to network issues or API error.")
