import langchain
import pypdf
import streamlit
import streamlit as st

import main

# Set the title and page layout
st.title("DocReader GPT")
st.image('docReader.png')
st.sidebar.title("Options Menu")
# upload file button
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['pdf'])

if uploaded_file is not None:
    # convert PDF to list
    pdf = pypdf.PdfReader(uploaded_file)
    txt = main.prepareDocForUpload(pdf)
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
        return main.chatPrompt(prompt, user_input)


# Check if the user has entered any input
if user_input:
    # Generate the bot response based on user input
    bot_response = generate_response(user_input)
    # Display the bot response
    st.text_area("Bot Response", value=bot_response["answer"], height=200)

    with st.expander('Conversation History'):
        for messages in bot_response['chat_history']:
            if isinstance(messages, langchain.schema.HumanMessage):
                st.divider()
            st.info(messages.content)




