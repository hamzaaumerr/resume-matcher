import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from uuid import uuid4
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_cohere import ChatCohere
import random

# Streamlit app
st.title("Resume Matcher")

cohere_api_key = st.secrets["API_KEY"]

# Generate or retrieve a unique user ID
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(random.randint(1000000000, 9999999999))

user_id = st.session_state["user_id"]

# Sidebar for user input
with st.sidebar:
    with st.form(key="user_details_form", clear_on_submit=True):

        # Check if name and contact are already in session state
        if "name" not in st.session_state:
            st.session_state["name"] = ""
        if "contact" not in st.session_state:
            st.session_state["contact"] = ""
        
        st.header("Enter Your Details")
        name = st.text_input("Name", st.session_state["name"])
        contact = st.text_input("Contact - Email, Number, links, etc", st.session_state["contact"])
        skills = st.text_area("Skills - Technical, Soft, etc (one per line)", "")
        experience = st.text_area("Experience - Company Name, Duration, Role, Responsibilities etc (one per line)", "")
        education = st.text_area("Education - Degree Name, University Name, Duration etc (one per line)", "")
        project = st.text_area("Projects - Name, Features etc (one per line)", "")

        submit_button = st.form_submit_button(label="Save")
    
    if submit_button:
        if all([name, contact, skills, experience, education, project]):
            st.session_state["name"] = name
            st.session_state["contact"] = contact
            st.session_state["saved"] = True
        else:
            st.warning("Please fill out all fields.")

# Check if the user has saved the details
if st.session_state.get("saved", False):
    # Convert inputs to lists (split by new lines)
    skills_list = [s.strip() for s in skills.splitlines() if s.strip()]
    experience_list = [e.strip() for e in experience.splitlines() if e.strip()]
    education_list = [e.strip() for e in education.splitlines() if e.strip()]
    project_list = [p.strip() for p in project.splitlines() if p.strip()]

    # Initialize Cohere Embeddings
    embedding_model = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-light-v3.0")

    # Initialize Chroma vector store
    vector_store = Chroma(
        collection_name="user_data",
        embedding_function=embedding_model,
        persist_directory="chroma"
    )

    # Create documents for skills, experience, education, and projects
    skill_documents = [
        Document(
            page_content=skill,
            metadata={"user_id": user_id, "type": "skill"},
            id=str(uuid4())
        )
        for skill in skills_list
    ]

    experience_documents = [
        Document(
            page_content=exp,
            metadata={"user_id": user_id, "type": "experience"},
            id=str(uuid4())
        )
        for exp in experience_list
    ]

    education_documents = [
        Document(
            page_content=edu,
            metadata={"user_id": user_id, "type": "education"},
            id=str(uuid4())
        )
        for edu in education_list
    ]

    project_documents = [
        Document(
            page_content=proj,
            metadata={"user_id": user_id, "type": "project"},
            id=str(uuid4())
        )
        for proj in project_list
    ]

    # Add documents to the vector store
    vector_store.add_documents(documents=skill_documents)
    vector_store.add_documents(documents=experience_documents)
    vector_store.add_documents(documents=education_documents)
    vector_store.add_documents(documents=project_documents)

    # Main app layout for job description and resume generation
    job_description = st.text_area("Job Description", "")

    if st.button("Generate Resume"):
        def remove_duplicates(results):
            seen = set()
            unique_results = []
            for result in results:
                content = result.page_content
                if content not in seen:
                    unique_results.append(result)
                    seen.add(content)
            return unique_results
        
        # Retrieve relevant information
        relevant_skills = vector_store.similarity_search_by_vector(
            embedding=embedding_model.embed_query(job_description), k=10, filter={"type": "skill"}
        )

        relevant_experience = vector_store.similarity_search_by_vector(
            embedding=embedding_model.embed_query(job_description), k=10, filter={"type": "experience"}
        )
        
        relevant_education = vector_store.similarity_search_by_vector(
            embedding=embedding_model.embed_query(job_description), k=2, filter={"type": "education"}
        )

        relevant_project = vector_store.similarity_search_by_vector(
            embedding=embedding_model.embed_query(job_description), k=2, filter={"type": "project"}
        )

        # Format the relevant information
        skills_content = "\n".join([doc.page_content for doc in relevant_skills])
        experience_content = "\n".join([doc.page_content for doc in relevant_experience])
        education_content = "\n".join([doc.page_content for doc in relevant_education])
        project_content = "\n".join([doc.page_content for doc in relevant_project])

        # Create a system prompt to generate a resume
        system_prompt = f"""
        Generate a resume with the following details:

        Guidelines:
        Be specific and use active voice.
        Avoid errors, passive language, and personal pronouns.
        Ensure consistency and readability.

        Avoid:
        Spelling/grammar errors, missing contact info, poor organization.

        Action Verbs Examples:
        Leadership: Led, Managed
        Communication: Presented, Promoted
        Technical: Engineered, Programmed
        Organizational: Organized, Implemented

        Details:
        Job Description: {job_description}
        Name: {st.session_state['name']}
        Contact: {st.session_state['contact']}
        Projects: {project_content}
        Skills: {skills_content}
        Experience: {experience_content}
        Education: {education_content}

        Formatting:
        Start with name and contact.
        List experience, education, projects and skills in order.
        Use headings and bullet points.

        Instruction:
        Do not add any extra text or headings from yourself; use only the provided details.
        """

        # Initialize ChatCohere
        chat_model = ChatCohere(cohere_api_key=cohere_api_key)

        # Generate the resume
        resume_output = chat_model.invoke(input=system_prompt)

        # Display the generated resume
        st.subheader("Generated Resume")
        st.write(resume_output.content)

else:
    st.info("Please fill out your details in the sidebar and click Save to proceed.")
