import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import requests
from unrealspeech import UnrealSpeechAPI, play


load_dotenv()
embeddings = OpenAIEmbeddings()
speech_api = UnrealSpeechAPI(api_key=os.getenv('UNREAL_SPPEECH_APIKEY'))

# text = 'Unreal Speech is a state-of-speech technology for creating text-to-speech applications with few lines of code'
# doc_embeddings = embeddings.embed_documents([text])

# loader = TextLoader('documents/summary.txt')
loader = DirectoryLoader('documents', glob="**/*.txt")
text_splitter = CharacterTextSplitter(chunk_size=2500,  chunk_overlap=0)
documents = loader.load()

texts = text_splitter.split_documents(documents)
# chroma vector  store
vecstore = Chroma.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff',
                                 retriever=vecstore.as_retriever())


def query(q):
    try:
        print("Query: ", q)
        answer = qa.run(q)
        audio_data = speech_api.stream(answer, 'Will')
        play(audio_data)

        print('Answer: ', answer)
    except Exception as e:
        print("An error occurred: ", e)


query("What did Australia’s Minister emphasized on?")
