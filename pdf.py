import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
# from llama_index.readers import PDFReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index=load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )
        
    return index

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)


# pdf_path = 'data/United_States.pdf'
# us_pdf = PDFReader().load_data(file=pdf_path)

us_index = get_index(documents, "us")
us_engine = us_index.as_query_engine()