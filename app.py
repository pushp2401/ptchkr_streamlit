import streamlit as st
import os
from PyPDF2 import PdfReader
import pymupdf
import numpy as np
import cv2
import shutil
import imageio
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext
import qdrant_client
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import SimpleMultiModalQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core import load_index_from_storage, get_response_synthesizer
import tempfile
from qdrant_client import QdrantClient, models
import getpass



curr_user = getpass.getuser()
# from langchain.vectorstores import Chroma
# To connect to the same event-loop,
# allows async events to run on notebook

# import nest_asyncio

# nest_asyncio.apply()

from dotenv import load_dotenv
load_dotenv()


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ''
    for page in reader.pages:
        text = page.extract_text()
        full_text += text
    return full_text

def extract_images_from_pdf(pdf_path, img_save_path):
    doc = pymupdf.open(pdf_path)
    for page in doc:
        img_number = 0
        for block in page.get_text("dict")["blocks"]:
            if block['type'] == 1:
                name = os.path.join(img_save_path, f"img{page.number}-{img_number}.{block['ext']}")
                out = open(name, "wb")
                out.write(block["image"])
                out.close()
                img_number += 1

def is_empty(img_path):
    image = cv2.imread(img_path, 0)
    std_dev = np.std(image)
    return std_dev < 1

def move_images(source_folder, dest_folder):
    image_files = [f for f in os.listdir(source_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    os.makedirs(dest_folder, exist_ok=True)
    moved_count = 0
    for file in image_files:
        src_path = os.path.join(source_folder, file)
        if not is_empty(src_path):
            shutil.move(src_path, os.path.join(dest_folder, file))
            moved_count += 1
    return moved_count

def remove_low_size_images(data_path):
    images_list = os.listdir(data_path)
    low_size_photo_list = []
    for one_image in images_list:
        image_path = os.path.join(data_path, one_image)
        try:
            pic = imageio.imread(image_path)
            size = pic.size
            if size < 100:
                low_size_photo_list.append(one_image)
        except:
            pass
    for one_image in low_size_photo_list[1:]:
        os.remove(os.path.join(data_path, one_image))

def calc_diff(img1 , img2) :
    i1 = Image.open(img1)
    i2 = Image.open(img2)
    h1 = imagehash.phash(i1)
    h2 = imagehash.phash(i2)
    return h1 - h2

def remove_duplicate_images(data_path) :
    image_files = os.listdir(data_path)
    only_images = []
    for one_image in image_files : 
        if one_image.endswith('jpeg') or one_image.endswith('png') or one_image.endswith('jpg') :
            only_images.append(one_image)
    only_images1 = sorted(only_images) 
    for one_image in only_images1 :
        for another_image in only_images1 :
            try :
                if one_image == another_image :
                    continue
                else :
                    diff = calc_diff(os.path.join(data_path ,one_image) , os.path.join(data_path ,another_image))
                    if diff ==0  :
                        os.remove(os.path.join(data_path , another_image))
            except Exception as e:
                print(e)
                pass
# from langchain_chroma import Chroma
# import chromadb
def initialize_qdrant(temp_dir , file_name , user):
    client = qdrant_client.QdrantClient(path=f"qdrant_mm_db_pipeline_{user}_{file_name}")
    # client = qdrant_client.QdrantClient(url = "http://localhost:2452")    
    # client = qdrant_client.QdrantClient(url="4b0af7be-d5b3-47ac-b215-128ebd6aa495.europe-west3-0.gcp.cloud.qdrant.io:6333", api_key="CO1sNGLmC6R_Q45qSIUxBSX8sxwHud4MCm4as_GTI-vzQqdUs-bXqw",)
    # client = qdrant_client.AsyncQdrantClient(location = ":memory:")
        
    if "vectordatabase" not in st.session_state or not st.session_state.vectordatabase:
            
        # text_store = client.create_collection(f"text_collection_pipeline_{user}_{file_name}"  ) 
        # image_store = client.create_collection(f"image_collection_pipeline_{user}_{file_name}"  ) 


        text_store = QdrantVectorStore( client = client , collection_name=f"text_collection_pipeline_{user}_{file_name}" )
        image_store = QdrantVectorStore(client = client , collection_name=f"image_collection_pipeline_{user}_{file_name}")
        storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)
        documents = SimpleDirectoryReader(os.path.join(temp_dir, f"my_own_data_{user}_{file_name}")).load_data()
        index = MultiModalVectorStoreIndex.from_documents(documents, storage_context=storage_context)

        st.session_state.vectordatabase = index
    else :
        index = st.session_state.vectordatabase
    retriever_engine = index.as_retriever(similarity_top_k=1, image_similarity_top_k=1)
    return retriever_engine

def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)
            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            images_shown += 1
            if images_shown >= 6:
                break

def retrieve_and_query(query, retriever_engine):
    retrieval_results = retriever_engine.retrieve(query)
    
    qa_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information , "
        "answer the query in detail.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_tmpl = PromptTemplate(qa_tmpl_str)

    llm = OpenAI(model="gpt-4o", temperature=0)
    response_synthesizer = get_response_synthesizer(response_mode="refine", text_qa_template=qa_tmpl, llm=llm)

    response = response_synthesizer.synthesize(query, nodes=retrieval_results)
    
    retrieved_image_path_list = []
    for node in retrieval_results:
        if (node.metadata['file_type'] == 'image/jpeg') or (node.metadata['file_type'] == 'image/png'):
            if node.score > 0.25:
                retrieved_image_path_list.append(node.metadata['file_path'])
    
    return response, retrieved_image_path_list
#tmpnimvp35m , tmpnimvp35m , tmpydpissmv


def img_display(img_path_list) :
    ##################### new image display function ###################################
    for one_img in img_path_list :
        image = Image.open(one_img) 
        st.image(image)

def process_pdf(pdf_file):
    temp_dir = tempfile.TemporaryDirectory()
    unique_folder_name = temp_dir.name.split('/')[-1]
    temp_pdf_path = os.path.join(temp_dir.name, pdf_file.name)
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_file.getvalue())

    data_path = os.path.join(temp_dir.name, f"my_own_data_{unique_folder_name}_{os.path.splitext(pdf_file.name)[0]}")
    os.makedirs(data_path , exist_ok=True) 
    img_save_path = os.path.join(temp_dir.name, f"extracted_images_{unique_folder_name}_{os.path.splitext(pdf_file.name)[0]}")
    os.makedirs(img_save_path , exist_ok=True) 

    extracted_text = extract_text_from_pdf(temp_pdf_path)
    with open(os.path.join(data_path, "content.txt"), "w") as file:
        file.write(extracted_text)

    extract_images_from_pdf(temp_pdf_path, img_save_path)
    moved_count = move_images(img_save_path, data_path)
    remove_low_size_images(data_path)
    remove_duplicate_images(data_path)
    retriever_engine = initialize_qdrant(temp_dir.name , os.path.splitext(pdf_file.name)[0] , unique_folder_name)

    return temp_dir, retriever_engine

def main():
    st.title("PDF Vector Database Query Tool")
    st.markdown("This tool creates a vector database from a PDF and allows you to query it.")
    
    if "retriever_engine" not in st.session_state:
        st.session_state.retriever_engine = None
    if "vectordatabase" not in st.session_state:
        st.session_state.vectordatabase = None

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is None:
        st.info("Please upload a PDF file.")
    else:
        st.info(f"Uploaded PDF: {uploaded_file.name}")
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                temp_dir, st.session_state.retriever_engine = process_pdf(uploaded_file)
        
                st.success("PDF processed successfully!")

    if st.session_state.retriever_engine :
        query = st.text_input("Enter your question:")
        
        
        if st.button("Ask Question"):
            print("running")
            try:

                with st.spinner("Retrieving information..."):
                    # import pdb; pdb.set_trace()
                    response, retrieved_image_path_list = retrieve_and_query(query, st.session_state.retriever_engine)
                    print(retrieved_image_path_list)
                st.write("Retrieved Context:")
                for node in response.source_nodes:
                    st.code(node.node.get_text())
                
                st.write("\nRetrieved Images:")
                # plot_images(retrieved_image_path_list)
                img_display(retrieved_image_path_list)
                # st.pyplot()

                st.write("\nFinal Answer:")
                st.code(response.response)
            
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
