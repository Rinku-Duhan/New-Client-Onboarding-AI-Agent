# New Client Onboarding AI Assistant

## Overview
The **New Client Onboarding AI Assistant** is an AI-powered chatbot designed to help new employees quickly get acquainted with company policies, procedures, and internal regulations. 

This project is built for the **Umbrella Corporation** and combines:

- **Employee Data:** Personalized information such as name, role, department, skills, hire date, and supervisor details.
- **Company Policies:** Extracted from a PDF containing internal regulations, stored in a vector database for fast retrieval.
- **Interactive Chatbot Interface:** Built using **Streamlit** and **LangChain**. The assistant answers questions using both employee-specific data and company policies.

The assistant can handle common onboarding queries like:

- "What are the lab safety protocols?"
- "How do I request leave?"
- "What benefits am I eligible for as a new employee?"

---

## Features
- Displays employee information in the sidebar in a user-friendly format.
- Provides answers augmented with company policies from a vector store.
- Streams responses in real-time for a natural chat experience.
- Caches employee data and conversation history to maintain context.
- Easily extendable to include new documents or policies.

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd client-onboarding-rag-demo-solution
```
### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```
### 3. Install required dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure environment variables
Create a .env file in the project root and add the required API key as mentioned in .env.example file

### 5. Run the application
```bash
streamlit run app.py
```
## Usage
1. The Streamlit application opens in your default web browser.
2. Employee information is displayed in the sidebar for context.
3. Type your questions in the chat input field.
4. The assistant provides real-time responses, using:
5. Employee-specific data for personalization.
6. Policy documents from the vector store for accurate answers.
7. Employee data and chat history are cached during the session for consistent responses.

## Notes
1. Make sure umbrella_corp_policies.pdf is present in the data/ directory.
2. The chatbot uses LangChain to manage prompts, retrieval, and language model interactions.
3. Streamlit session state is used to store employee data and chat history for the session.
4. You can replace the LLM or embeddings provider as needed for customization.
