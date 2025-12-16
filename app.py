#import libraries
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
from typing import List,Dict,Any

#load environment variables
load_dotenv()

#App configuration
st.set_page_config(page_title="Bitte Rag ChatBot", 
page_icon="material/chat_bubble:", #speech bubble icon
layout="centered")

#Add a title to the app include a bot emoji
st.title("🤖 Bitte RAG ChatBot")

#Add a description to the app
st.markdown("***Your Intelligent assistant powered by GPT-5 and RAG technology***")
st.divider()

#add a collapsible section
with st.expander("About the Bitte RAG dataset",expanded=False):
    st.markdown("""
    **Bitte RAG ChatBot**
    
    - **Model:** GPT-5 via OpenAI responses API
    - **RAG:** File Search tool using your pre-built vector store
    - **Features:** multi-turn chat, image inputs, clear conversation
    - **Secrets:** reads OPENAI_API_KEY and VECTOR_STORE_ID from Streamlit secrets or environment variables

    **How it works:**
    
    Your message and (optional) images go to the Response API along with a system prompt.
     """)

# Retrive the credentials from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
vector_store_id = os.getenv("VECTOR_STORE_ID") or st.secrets["VECTOR_STORE_ID"]

#set the OpenAi key in the os
os.environ["OPENAI_API_KEY"] = openai_api_key

#initialize the OpenAI client
client = OpenAI()

# warn if OpenAI key or the vector store id is not set:
if not openai_api_key:
    st.warning("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable or Streamlit secrets.")
if not vector_store_id:
    st.warning("Vector store ID is not set. Please set the VECTOR_STORE_ID environment variable or Streamlit secrets.")

# Configuration of the system prompt:
system_prompt = """You are a toxic CEO who loves things like pre-revenue or cash burn ratio.
Always respond in English, even when user input or retrieved documents are in other languages. Translate context as needed but produce English-only answers."""

#Store the previous response id
if "messages" not in st.session_state:
    st.session_state.messages = []
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None

#Create a sidebar with user controls
with st.sidebar:
    st.header("User Controls")
    st.divider()
    # Clear the conversation history - reset chat history and context
    if st.button("Clear Conversation History",use_container_width=True):
        st.session_state.messages = []
        st.session_state.previous_response_id = None
        #reset the page
        st.rerun()

# Helper functions
def build_input_parts(text: str,images:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """
    Build the input parts array for the OpenAI from text and images.

    Args:
        text: The text to be sent to OpenAI
        images: The images to be sent to OpenAI

    Returns:
        A list of input parts compatible with openAI responses API    
    """

    content = []
    if text and text.strip():
        content.append({
            "type":"input_text",
            "text":text.strip()
        }
        )
    #since images can be multiple we need a loop
    for img in images:
        content.append({
            "type":"input_image",
            "image_url":img["data_url"]
        }
        )
    # Responses API expects a list of messages with role and content parts.
    return [{"role": "user", "content": content}] if content else []      

# Function to generate  a response from the OpenAI responses API
def call_responses(parts:List[Dict[str,Any]],previous_response_id:str = None):
    """
    Call the OpenAI responses API with the input parts.

    Args:
        parts: The input parts to be sent to the OpenAI
        previous_response_id: The previous response id to be sent to the OpenAI

    """
    tools = [
        {
            "type":"file_search",
            "vector_store_ids": [vector_store_id],
            "max_num_results":20
        }
    ]
    response=client.responses.create(
        model="gpt-5-nano",
        input=parts,
        instructions=system_prompt,
        tools=tools,
        previous_response_id=previous_response_id
    )

    return response

#function to get the text output
def get_text_output(response:Any) -> str:
    """
    Get the text output from the OpenAI responses API
    """
    return response.output_text




#render all previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        #Extract text content from the message structure
        if isinstance(m["content"],list):
            #Handle structured messages (user input)
            for part in m["content"]:
                for content_item in part.get("content",[]):
                    ctype = content_item.get("type")
                    if ctype == "input_text":
                        st.markdown(content_item.get("text",""))
                    elif ctype == "input_image":
                        st.image(content_item.get("image_url"),width=100)
        else:
            #Handle simple text messages (assistant response)    
            st.markdown(m["content"])


# User interface - upload imagaes
uploaded = st.file_uploader(
    "Upload images",
    type=["png","jpg","jpeg","webp"],
    accept_multiple_files=True,
    key=f"file_uploader_{len(st.session_state.messages)}"
    )
#user interface - chat input
prompt = st.chat_input("Type your message here....")

if prompt is not None:
    # Process the images into an API-compatible format for the API
    images=[]
    if uploaded:
        images = [
            {
                "mime_type": f"image/{f.type.split('/')[-1]}" if f.type else "image/png",
                "data_url": f"data:{f.type or 'image/png'};base64,{base64.b64encode(f.read()).decode('utf-8')}",
            }
            for f in uploaded 
        ]
    # Build the input parts for the responses API
    parts = build_input_parts(prompt,images)

    # Store the messages
    st.session_state.messages.append({"role": "user", "content": parts})
    # Display the user's message
    with st.chat_message("user"):
        for p in parts:
            # Each part is already a message dict with role/content
            for content_item in p.get("content", []):
                ctype = content_item.get("type")
                if ctype == "input_text":
                    st.markdown(content_item.get("text", ""))
                elif ctype == "input_image":
                    #display the image in a smaller size
                    st.image(content_item['image_url'],width=100)
                else:
                    st.error(f"Unknown content type: {ctype}")

    #Generate the AI response:
    with st.chat_message("assistant"):
        with st.spinner("Thinking...."):
            try:
                response=call_responses(parts,st.session_state.previous_response_id)
                output_text = get_text_output(response)

                #Display the AI's response
                st.markdown(output_text)
                st.session_state.messages.append({"role":"assistant","content":output_text})

                #Retrive the ID if available
                if hasattr(response, "id"):
                    st.session_state.previous_response_id=response.id
            except Exception as e:
                st.error(f"Error generating response: {e}")        

