import pypdf
import streamlit as st

import main

# Set the title and page layout
st.title("DocReader chatbot")
st.sidebar.title("Chatbot Options")
st.sidebar.markdown("Customize your Chatbot")
# upload file button
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['pdf'])

if uploaded_file is not None:
    # convert PDF to list
    pdf = pypdf.PdfReader(uploaded_file)
    txt = main.prepareDocForUpload(pdf)
    print("going to load to DB !!!!!")
    db = main.loadTextoDB(txt)
    prompt = main.init(db)

# Create a text input for user input
user_input = st.text_input("Load a pdf file to begin....", "")


# init prompt
# prompt = main.init()

# Create a function to generate bot responses
def generate_response(user_input):
    # Add your chatbot logic here
    # Process the user input and generate a response
    # For demonstration purposes, a simple response is provided
    if prompt:
        response = main.chatPrompt(prompt, user_input)
        return response["answer"]


# Check if the user has entered any input
if user_input:
    # Generate the bot response based on user input
    bot_response = generate_response(user_input)
    # Display the bot response
    st.text_area("Bot Response", value=bot_response, height=200)

    with st.expander('Conversation History'):
        st.info("testing")

