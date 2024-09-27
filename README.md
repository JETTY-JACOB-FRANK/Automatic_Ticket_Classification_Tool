AUTOMATIC TICKET CLASSIFICATION


Overview:

The Automatic Ticket Classification tool is a web application designed to help users classify their service tickets based on the content they provide. Using advanced machine learning models and embeddings from the SentenceTransformer, the system automatically predicts the department (HR, IT, or Transport) that a ticket should be routed to. The application integrates with Pinecone to retrieve relevant information for user questions and leverages a pre-trained model for classification.

The tool has been developed using Streamlit for the web interface, with machine learning support from models loaded through joblib. The system can create and submit tickets to appropriate departments based on user input.


Features:


•	Real-time ticket classification based on input text.

•	Integration with Pinecone for similarity search and question-answering.

•	Embeddings creation using SentenceTransformer.

•	Automatic routing to departments (HR, IT, or Transport).

•	Streamlit UI for user interaction.


Technologies Used:


•	Python

•	Streamlit

•	Pinecone

•	LangChain

•	OpenAI API

•	Hugging Face Transformers (Sentence Transformers)

•	Scikit-learn

•	Dotenv

•	YAML

•	Git
