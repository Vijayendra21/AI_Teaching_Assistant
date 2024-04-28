import os
import streamlit as st
from PyPDF2 import PdfReader
#from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

import httpx
from langchain_community.llms import Cohere
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

#load_dotenv()

llm = Cohere(cohere_api_key=st.secrets["COHERE_API_KEY"] , verbose=False)
chain = load_qa_chain(llm, chain_type="stuff")

deepgram = DeepgramClient(st.secrets["DEEPGRAM_API_KEY"])
options = PrerecordedOptions(
    model="nova-2",
    smart_format=True,
)


st.title("AI Teacher Bot based on your Resource!")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


file = st.file_uploader("Upload your Resource", type=["pdf", "mp3"])

if file is not None:
    if "myFile" not in st.session_state or st.session_state.myFile == file:
        st.session_state.myFile = file
        if "kb" not in st.session_state:
            if file.type == "application/pdf":
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

            elif file.type == "audio/mpeg":
                pass
                audio_bytes = file.read()
                payload: FileSource = {
                    "buffer": audio_bytes,
                }
                response = deepgram.listen.prerecorded.v("1").transcribe_file(
                    payload, options, timeout=httpx.Timeout(300.0, connect=10.0))
                text = response["results"]["channels"][0]['alternatives'][0]['transcript']

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            # st.write(chunks)

            embeddings = CohereEmbeddings(
                cohere_api_key="YGE5KbwA6SwirK8OozaIcD89SoT1Ml04XsB0zyp2")
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            st.session_state.kb = knowledge_base
            docs = st.session_state.kb.similarity_search('')
            response = chain.run(
                input_documents=docs, question="give me detailed notes on this document")
            if st.session_state.messages == []:
                st.session_state.messages.append(
                    {"role": "assistant", "content": response})

    elif st.session_state.myFile != file:
        st.session_state.myFile = file
        st.session_state.messages = []

        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

        elif file.type == "audio/mpeg":
            pass
            audio_bytes = file.read()
            payload: FileSource = {
                "buffer": audio_bytes,
            }
            response = deepgram.listen.prerecorded.v("1").transcribe_file(
                payload, options, timeout=httpx.Timeout(300.0, connect=10.0))
            text = response["results"]["channels"][0]['alternatives'][0]['transcript']
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        # st.write(chunks)

        embeddings = CohereEmbeddings(
            cohere_api_key="YGE5KbwA6SwirK8OozaIcD89SoT1Ml04XsB0zyp2")
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.session_state.kb = knowledge_base
        docs = st.session_state.kb.similarity_search('')
        response = chain.run(input_documents=docs,
                             question="give me detailed notes on this document")
        if st.session_state.messages == []:
            st.session_state.messages.append(
                {"role": "assistant", "content": response})


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    if "kb" in st.session_state:

        # docs = knowledge_base.similarity_search(prompt)

        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        if 'quiz' in prompt.lower():
            docs = st.session_state.kb.similarity_search(' ')
            response = chain.run(input_documents=docs, question="generate as many multiple choice questions with 4 options for each as you can from the given data to test my understanding. make sure each option is in a new line to make it more readable. i want the answer key to be shown at the end, after all the questions are shown. if the number of possible options are not 4, dont give me that question")
            print(response[len(response)-10:len(response)-1])
        else:
            docs = st.session_state.kb.similarity_search(prompt)
            response = chain.run(input_documents=docs, question=prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
