import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Streamlit app
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# set groq api key and url (YT or website) to summarise

with st.sidebar:
    groq_api = st.text_input("Groq api key", value="", type='password')

generic_url = st.text_input("URL", label_visibility="collapsed")
llm =ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api)

prompt_template="""
Provide summary of following content in 300 words:
Content:{text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['text'])


if st.button("Summarise the content"):
    # validate all inputs
    if not groq_api.strip() or not generic_url.strip():
        st.error("Please provide the info to get started")
    elif not validators.url(generic_url):
        st.error("Plz provide valid url")
    else:
        try:
            with st.spinner("Waiting.."):
                ## load website and yt oyutube data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                                                   # headers are given to tell from which browser u should try to execute it
                                                   )
                data=loader.load()

                ## Chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary=chain.run(data)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")

