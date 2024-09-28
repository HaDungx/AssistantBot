import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

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

def main():
    st.title("Chatbot LLM với Streamlit")
    st.write("Hãy đặt câu hỏi về thời khóa biểu hoặc các thông tin liên quan:")

    # Lưu lịch sử cuộc hội thoại
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Lấy câu hỏi từ người dùng
    user_input = st.text_input("Bạn: ", "")

    if user_input:
        # Gọi mô hình LLM để lấy phản hồi
        response = llm_chain.invoke({"query": user_input})
        answer = response['result']

        # Lưu cuộc hội thoại
        st.session_state['messages'].append({"role": "user", "content": user_input})
        st.session_state['messages'].append({"role": "assistant", "content": answer})

    # Hiển thị cuộc hội thoại
    for message in st.session_state['messages']:
        if message['role'] == 'user':
            st.write(f"**Bạn:** {message['content']}")
        else:
            st.write(f"**AI:** {message['content']}")

if __name__ == '__main__':
    main()
