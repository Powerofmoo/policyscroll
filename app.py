import os
import getpass
import requests

import sentence_transformers

import streamlit as st

VECTOR_DB ="bbf2ef09-875b-4737-a793-499409a108b0"

IBM_API_KEY = os.getenv("IBM_API_KEY")
IBM_PROJECT_ID = "a0659778-f4ce-4da1-ba01-43b4f43a026f"

IBM_URL_TOKEN = "https://iam.cloud.ibm.com/identity/token"
IBM_URL_CHAT = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-10-25"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

##############################################
##
##   IBM API
##
##############################################
def IBM_token():
    # Define the headers
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    # Define the data payload
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": IBM_API_KEY
    }
    
    # Make the POST request
    response = requests.post(IBM_URL_TOKEN, headers=headers, data=data)
    st.session_state.IBM_ACCESS_TOKEN = response.json().get("access_token", "")


def IBM_chat (messages):
    body = {
        "model_id": "ibm/granite-3-8b-instruct",
        "project_id": IBM_PROJECT_ID,
        "messages": messages,
        "max_tokens": 10000,
        "temperature": 0.3,
        "time_limit": 10000
    }
    headers = {
    	"Accept": "application/json",
    	"Content-Type": "application/json",
    	"Authorization": "Bearer " + st.session_state.IBM_ACCESS_TOKEN
    }    
    response = requests.post(
    	IBM_URL_CHAT,
    	headers=headers,
    	json=body
    )
    
    if response.status_code != 200:
    	raise Exception("Non-200 response: " + str(response.text))
    
    response = response.json()
    return response["choices"][0]["message"]["content"]

## get token
IBM_token()

def get_credentials():
	return {
		"url" : "https://us-south.ml.cloud.ibm.com",
		"apikey" : os.getenv("IBM_API_KEY")
	}

model_id = "ibm/granite-3-8b-instruct"

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "min_new_tokens": 0,
    "repetition_penalty": 1
}

project_id = os.getenv("IBM_PROJECT_ID")
space_id = os.getenv("IBM_SPACE_ID")

from ibm_watsonx_ai.foundation_models import ModelInference

model = ModelInference(
	model_id = model_id,
	params = parameters,
	credentials = get_credentials(),
	project_id = project_id,
	space_id = space_id
	)

from ibm_watsonx_ai.client import APIClient

wml_credentials = get_credentials()
client = APIClient(credentials=wml_credentials, project_id=project_id)  #, space_id=space_id)

vector_index_id = VECTOR_DB
vector_index_details = client.data_assets.get_details(vector_index_id)
vector_index_properties = vector_index_details["entity"]["vector_index"]

top_n = 20 if vector_index_properties["settings"].get("rerank") else int(vector_index_properties["settings"]["top_k"])

def rerank( client, documents, query, top_n ):
    from ibm_watsonx_ai.foundation_models import Rerank

    reranker = Rerank(
        model_id="cross-encoder/ms-marco-minilm-l-12-v2",
        api_client=client,
        params={
            "return_options": {
                "top_n": top_n
            },
            "truncate_input_tokens": 512
        }
    )

    reranked_results = reranker.generate(query=query, inputs=documents)["results"]

    new_documents = []
    
    for result in reranked_results:
        result_index = result["index"]
        new_documents.append(documents[result_index])
        
    return new_documents

from ibm_watsonx_ai.foundation_models.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings

emb = SentenceTransformerEmbeddings('sentence-transformers/all-MiniLM-L6-v2')

import subprocess
import gzip
import json
import chromadb
import random
import string

def hydrate_chromadb():
    data = client.data_assets.get_content(vector_index_id)
    content = gzip.decompress(data)
    stringified_vectors = str(content, "utf-8")
    vectors = json.loads(stringified_vectors)
    
    #chroma_client = chromadb.Client()
    #chroma_client = chromadb.InMemoryClient()
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # make sure collection is empty if it already existed
    collection_name = "my_collection"
    try:
        collection = chroma_client.delete_collection(name=collection_name)
    except:
        print("Collection didn't exist - nothing to do.")
    collection = chroma_client.create_collection(name=collection_name)

    vector_embeddings = []
    vector_documents = []
    vector_metadatas = []
    vector_ids = []

    for vector in vectors:
        vector_embeddings.append(vector["embedding"])
        vector_documents.append(vector["content"])
        metadata = vector["metadata"]
        lines = metadata["loc"]["lines"]
        clean_metadata = {}
        clean_metadata["asset_id"] = metadata["asset_id"]
        clean_metadata["asset_name"] = metadata["asset_name"]
        clean_metadata["url"] = metadata["url"]
        clean_metadata["from"] = lines["from"]
        clean_metadata["to"] = lines["to"]
        vector_metadatas.append(clean_metadata)
        asset_id = vector["metadata"]["asset_id"]
        random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        id = "{}:{}-{}-{}".format(asset_id, lines["from"], lines["to"], random_string)
        vector_ids.append(id)

    collection.add(
        embeddings=vector_embeddings,
        documents=vector_documents,
        metadatas=vector_metadatas,
        ids=vector_ids
    )
    return collection

chroma_collection = hydrate_chromadb()

def proximity_search( question ):
    query_vectors = emb.embed_query(question)
    query_result = chroma_collection.query(
        query_embeddings=query_vectors,
        n_results=top_n,
        include=["documents", "metadatas", "distances"]
    )

    documents = list(reversed(query_result["documents"][0]))

    if vector_index_properties["settings"].get("rerank"):
        documents = rerank(client, documents, question, vector_index_properties["settings"]["top_k"])

    return "\n".join(documents)

# Streamlit UI
st.title("🔍 Policy Scroll")
st.subheader("AI-Powered Project Matching")
st.write("Explore the Lab Lab Library to find relevant past projects that align with your policy or new initiative.")

# Suggested search queries as buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Solarpunk projects to connect with"):
        st.session_state["user_input"] = "Solarpunk projects to connect with"

with col2:
    if st.button("How to implement DEI?"):
        st.session_state["user_input"] = "How to implement DEI?"
        
# User input in Streamlit
user_input = st.chat_input("Describe your policy or project to find relevant Lab Lab projects...")

if st.session_state["user_input"]:

    # Display user message
    #st.chat_message("user").markdown(st.session_state["user_input"])

    grounding = proximity_search(st.session_state["user_input"])

    # add the submissions as context (only in prompt, not in history)
    prompt = st.session_state["user_input"] + ". If a project is mentioned, share the image. The context for the question: " + grounding;
    messages = st.session_state.messages.copy()
    messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "user", "content": st.session_state["user_input"]})

    # Get response from IBM
    with st.spinner("Thinking..."):
        assistant_reply = IBM_chat(messages)

    # Display assistant message
    st.chat_message("assistant").markdown(assistant_reply)
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
