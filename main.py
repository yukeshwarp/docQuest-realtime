import streamlit as st
import json
from utils.pdf_processing import process_pdf_pages
from utils.llm_interaction import ask_question
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import io
import tiktoken  # For token counting

# Initialize session state variables
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Token counting function
def count_tokens(text, model="gpt-4o"):
    """Count the tokens in a given text."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

# Function to handle user question and count tokens
def handle_question(prompt):
    if prompt:
        try:
            # Count tokens for the user prompt and documents (input)
            input_tokens = count_tokens(prompt)
            document_tokens = count_tokens(json.dumps(st.session_state.documents))
            total_input_tokens = input_tokens + document_tokens

            with st.spinner('Thinking...'):
                answer = ask_question(
                    st.session_state.documents, prompt, st.session_state.chat_history
                )
            
            # Count tokens for the response (output)
            output_tokens = count_tokens(answer)

            # Store the question, answer, and token counts
            st.session_state.chat_history.append({
                "question": prompt,
                "answer": answer,
                "input_tokens": total_input_tokens,
                "output_tokens": output_tokens
            })

            # Display the updated chat history with token counts
            display_chat()
        except Exception as e:
            st.error(f"Error processing question: {e}")

# Function to reset session data when files are changed
def reset_session():
    st.session_state.documents = {}
    st.session_state.chat_history = []
    st.session_state.uploaded_files = []

# Function to display chat history with token counts
def display_chat():
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            user_message = f"""
            <div style='padding:10px; border-radius:10px; margin:5px 0; text-align:right;'> 
            {chat['question']}
            <small style='color:grey;'>Tokens: {chat['input_tokens']}</small></div>
            """
            assistant_message = f"""
            <div style='padding:10px; border-radius:10px; margin:5px 0; text-align:left;'> 
            {chat['answer']}
            <small style='color:grey;'>Tokens: {chat['output_tokens']}</small></div>
            """
            st.markdown(user_message, unsafe_allow_html=True)
            st.markdown(assistant_message, unsafe_allow_html=True)

# Sidebar for file upload and document information
with st.sidebar:
    uploaded_files = st.file_uploader(
        "",
        type=["pdf", "docx", "xlsx", "pptx"],
        accept_multiple_files=True,
        help="Supports PDF, DOCX, XLSX, and PPTX formats.",
    )

    if uploaded_files:
        new_files = []
        for index, uploaded_file in enumerate(uploaded_files):
            if uploaded_file.name not in st.session_state.documents:
                new_files.append(uploaded_file)
            else:
                st.info(f"{uploaded_file.name} is already uploaded.")

        if new_files:
            # Use a placeholder to show progress
            progress_text = st.empty()
            progress_bar = st.progress(0)
            total_files = len(new_files)

            # Spinner while processing documents
            with st.spinner("Learning about your document(s)..."):
                # Process files in pairs using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_to_file = {
                        executor.submit(process_pdf_pages, uploaded_file, first_file=(index == 0)): uploaded_file 
                        for index, uploaded_file in enumerate(new_files)
                    }

                    for i, future in enumerate(as_completed(future_to_file)):
                        uploaded_file = future_to_file[future]
                        try:
                            # Get the result from the future
                            document_data = future.result()
                            st.session_state.documents[uploaded_file.name] = document_data

                            st.success(f"{uploaded_file.name} processed successfully!")
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")

                        # Update progress bar
                        progress_bar.progress((i + 1) / total_files)
                    
            progress_text.text("Processing complete.")
            progress_bar.empty()
            
    if st.session_state.documents:
        download_data = json.dumps(st.session_state.documents, indent=4)
        st.download_button(
            label="Download Document Analysis",
            data=download_data,
            file_name="document_analysis.json",
            mime="application/json",
        )

# Main Page - Chat Interface
st.image("logoD.png", width=200)
st.title("docQuest")
st.subheader("Unveil the Essence, Compare Easily, Analyze Smartly", divider="orange")

if st.session_state.documents:

    # Chat input field using st.chat_input
    prompt = st.chat_input("Ask me anything about your documents", key="chat_input")

    # Check if the prompt has been updated
    if prompt:
        handle_question(prompt)  # Call the function to handle the question

# Sidebar to display total tokens used
total_input_tokens = sum(chat["input_tokens"] for chat in st.session_state.chat_history)
total_output_tokens = sum(chat["output_tokens"] for chat in st.session_state.chat_history)
st.sidebar.write(f"Total Input Tokens: {total_input_tokens}")
st.sidebar.write(f"Total Output Tokens: {total_output_tokens}")
