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
# Set up logging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s [%(levelname)s] %(message)s")


# Ensure NLTK stopwords are downloaded
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """
    Preprocess the text by removing blank spaces, stopwords, and punctuations.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # Remove multiple spaces and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
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

    # Exponential backoff retry mechanism
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)  # Adjusted timeout
            response.raise_for_status()  # Raise HTTPError for bad responses
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', "No explanation provided.")
        
        except requests.exceptions.Timeout as e:
            if attempt < retries - 1:
                wait_time = initial_delay * (2 ** attempt)  # Exponential backoff
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
    """
    Use Azure LLM to extract sections, headings, paragraphs, and tables from the given text.
    
    :param text: The full text content of a PDF page or a section of a document.
    :return: A structured JSON object containing sections, headings, paragraphs, and tables.
    """
    headers = get_headers()  # This function should return the necessary headers for Azure API requests.
    preprocessed_text = preprocess_text(text)  # Optional: preprocess text before sending to the model.
    
    data = {
        "model": model,  # Use your Azure OpenAI model deployment ID.
        "messages": [
            {"role": "system", "content": "You are an expert in document structure analysis."},
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
        "temperature": 0.5  # Adjust as needed for creativity and structure.
    }

    try:
        response = requests.post(
            f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
            headers=headers,
            json=data,
            timeout=20
        )
        response.raise_for_status()  # Check for HTTP errors.
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
        "temperature": 0.5  # Adjust as needed to generate creative but relevant system prompts
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
    # Generate the system prompt based on the document content
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

            # Calculate exponential backoff with jitter
            delay = min(max_delay, base_delay * (2 ** attempt))  # Exponential backoff
            jitter = random.uniform(0, delay)  # Add jitter for randomness
            logging.warning(f"Retrying in {jitter:.2f} seconds (attempt {attempt}) due to error: {e}")
            time.sleep(jitter)


import concurrent.futures
import requests
import logging

import requests
import logging

def llm_check_relevance(prompt):
    """Check the relevance of a given text content to a question using the LLM."""
    headers = get_headers()  # Function to retrieve headers for the API request
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
            timeout=60  # Add timeout for API request
        )
        response.raise_for_status()  # Raise HTTPError for bad responses

        # Extract the relevance determination from the response
        relevance_result = response.json().get('choices', [{}])[0].get('message', {}).get('content', "").strip()

        # Log the relevance check result
        logging.info(f"Relevance check result: {relevance_result}")

        # Determine if the content is relevant
        return "yes" in relevance_result.lower()  # Return True if the response indicates relevance

    except requests.exceptions.RequestException as e:
        logging.error(f"Error checking relevance: {e}")
        return False  # Return False in case of an error


def fetch_page(doc_data, question):
    """Check relevance of paragraphs in each page and retrieve relevant headings and paragraphs."""
    relevant_content = []

    for page in doc_data["pages"]:
        page_number = page["page_number"]
        structured_data = page["structured_data"]

        for section in structured_data["sections"]:
            heading = section["heading"]
            for paragraph in section["paragraphs"]:
                # Construct prompt for LLM to check relevance
                relevance_prompt = f"Is the following paragraph relevant to the question: '{question}'? Paragraph: '{paragraph}'"

                # Call LLM with relevance check
                if llm_check_relevance(relevance_prompt):
                    relevant_content.append({
                        "page_number": page_number,
                        "heading": heading,
                        "paragraph": paragraph
                    })

    return relevant_content

def fetch_sections(doc_data, question):
    """Check relevance of page summaries and return full text if relevant."""
    relevant_texts = []

    for page in doc_data["pages"]:
        page_number = page["page_number"]
        page_summary = page["text_summary"]

        # Construct prompt for LLM to check relevance
        relevance_prompt = f"Is the following page summary relevant to the question: '{question}'? Summary: '{page_summary}'"

        # Call LLM with relevance check
        if llm_check_relevance(relevance_prompt):
            relevant_texts.append({
                "page_number": page_number,
                "full_text": page["full_text"]  # Return the full text of the page
            })

    return relevant_texts

def fetch_table(doc_data, question):
    """Check relevance of tables and return page number and table content if relevant."""
    relevant_tables = []

    for page in doc_data["pages"]:
        page_number = page["page_number"]
        structured_data = page["structured_data"]

        for table in structured_data["tables"]:
            # Construct prompt for LLM to check relevance
            relevance_prompt = f"Does this table contain relevant data for the question: '{question}'? Table: {table}"

            # Call LLM with relevance check
            if llm_check_relevance(relevance_prompt):
                relevant_tables.append({
                    "page_number": page_number,
                    "table": table  # Return the table content
                })

    return relevant_tables

def fetch_figures(doc_data, question):
    """Check relevance of image explanations and return page number and image explanation if relevant."""
    relevant_figures = []

    for page in doc_data["pages"]:
        page_number = page["page_number"]
        image_analysis = page["image_analysis"]

        for image in image_analysis:
            # Construct prompt for LLM to check relevance
            relevance_prompt = f"Is this image explanation relevant to the question: '{question}'? Explanation: '{image['explanation']}'"

            # Call LLM with relevance check
            if llm_check_relevance(relevance_prompt):
                relevant_figures.append({
                    "page_number": page_number,
                    "explanation": image["explanation"]  # Return the image explanation
                })

    return relevant_figures

def ask_question(documents, question, chat_history):
    """Answer a question based on relevant content from multiple PDFs and chat history."""
    combined_content = []
    structured_relevant_content = {
        "page_numbers": [],
        "summaries": [],
        "headings_and_paragraphs": [],
        "tables": [],
        "figures": []
    }

    # Combine relevant content from each document
    for doc_name, doc_data in documents.items():
        # Fetch relevant sections, tables, and figures
        relevant_pages = fetch_page(doc_data, question)
        relevant_sections = fetch_sections(doc_data, question)
        relevant_tables = fetch_table(doc_data, question)
        relevant_figures = fetch_figures(doc_data, question)

        # Collect relevant content
        for page in relevant_pages:
            structured_relevant_content["page_numbers"].append(page["page_number"])
            structured_relevant_content["headings_and_paragraphs"].append({
                "heading": page["heading"],
                "paragraph": page["paragraph"]
            })

        for section in relevant_sections:
            structured_relevant_content["page_numbers"].append(section["page_number"])
            structured_relevant_content["summaries"].append(section["full_text"])

        for table in relevant_tables:
            structured_relevant_content["page_numbers"].append(table["page_number"])
            structured_relevant_content["tables"].append(table["table"])

        for figure in relevant_figures:
            structured_relevant_content["page_numbers"].append(figure["page_number"])
            structured_relevant_content["figures"].append(figure["explanation"])

    # Construct prompt message using the structured relevant content
    prompt_message = (
        f"""
    You have the following relevant content to answer the question:

    ---
    Page Numbers: {structured_relevant_content['page_numbers']}
    
    Summaries: {structured_relevant_content['summaries']}
    
    Headings and Paragraphs: {structured_relevant_content['headings_and_paragraphs']}
    
    Tables: {structured_relevant_content['tables']}
    
    Figures: {structured_relevant_content['figures']}
    ---
    
    Question: {question}
    """
    )

    # LLM interaction to get the answer
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
