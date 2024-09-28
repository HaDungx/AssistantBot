import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

def load_model():
    # Thiết lập API key của GROQ
    os.environ["GROQ_API_KEY"] = "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"


    llm = ChatGroq(temperature=0, model="llama3-8b-8192")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói tôi không biết, tìm thông tin liên quan nhất và nói tôi không chắc\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    db = FAISS.load_local("faiss_index_folder", embedding_model)


    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=None,  # Thay thế bằng retriever của bạn
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )
    return llm_chain

def response_generator(user_input):
    # Gọi mô hình LLM để lấy phản hồi
    user_input += ' dịch sang tiếng việt' 
    response = st.session_state['model'].invoke({"query": user_input})
    answer = response['result']

    for word in answer.split():
        yield word + " "
        time.sleep(0.05)


def main():
    st.title("Chatbot LLM với Streamlit")
    st.write("Hãy đặt câu hỏi về thời khóa biểu hoặc các thông tin liên quan:")

    # Khởi tạo lịch sử cuộc hội thoại
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Hiển thị cuộc trò chuyện khi rerun  
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Lấy câu hỏi từ người dùng
    user_input = st.chat_input("Nhập promt")
    if user_input:      
        # Lưu cuộc hội thoại người dùng
        st.session_state['messages'].append({"role": "user", "content": user_input})

        # Hiển thị cuộc hội thoại người dùng
        with st.chat_message("user"):
            st.markdown(user_input)

        # Hiển thị cuộc hội thoại AI
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator())

        # Lưu cuộc hội thoại AI  
        st.session_state.messages.append({"role": "assistant", "content": response})    

if __name__ == '__main__':
    if not 'model' in st.session_state:
        st.session_state['model'] = load_model()
    main()
