### Arabic Question Answering App using AraBERT:
 <br/> 
 
  <br/> 
This project provides an interactive Arabic Question Answering (QA) web application built with Streamlit and powered by AraBERT, a transformer-based language model fine-tuned on Arabic QA datasets such as the Arabic Reading Comprehension Dataset (ARCD).

The application allows users to input an Arabic context and a question, and returns the most relevant answer extracted from the context.
<br/>

### **Features**:
  <br/> 

⦁	Supports Arabic input and output.

⦁	Utilizes AraBERT for accurate extractive QA in Arabic.

⦁	Built on the Hugging Face Transformers library.

⦁	Provides a simple and user-friendly Streamlit interface.

⦁	Can be executed locally once the model is downloaded.

  <br/> 

### **Technology Stack**:
  <br/> 

Python 3.10 or higher

Streamlit

Hugging Face Transformers

PyTorch

AraBERT model (aubmindlab/bert-base-arabertv2)

Accelerate (for GPU/CPU optimization)

  <br/> 


### **Installation**:
  <br/> 

```bash
# Clone the repository
git clone https://github.com/your-username/arabert-qa-app.git
cd arabert-qa-app

# Install the required packages
pip install streamlit transformers torch accelerate

# If necessary, install the full Transformers package for PyTorch support
pip install transformers[torch]
```

  <br/> 


### **Usage**:
  <br/> 

Run the Streamlit application:
```bash
streamlit run app.py

```
Open the provided local URL in your browser. The interface provides:

A text area to enter the context (paragraph)

An input box to enter the question

A button to execute the QA model

The model outputs:

The answer extracted from the context

The confidence score indicating the model's certainty


The application uses custom CSS for:

Styled buttons for "Get Answer" and "Regenerate".

Colored boxes to display answers, confidence scores, warnings, and errors.

Responsive text areas for context and question input.

Stores previous context, question, answer, and score in Streamlit session state to facilitate regeneration without re-entering the inputs.


  <br/> 

### **Example**:
  <br/> 

```json
Context:

تأسست جامعة الدول العربية في القاهرة عام 1945، وتضم دولاً عربية تهدف إلى توثيق العلاقات بينها وتعزيز التعاون المشترك.

Question:

متى تأسست جامعة الدول العربية؟
```
Output:
```json
Answer: عام 1945

Confidence Score: 0.98

```
  <br/> 

### **Notes**
  <br/> 

⦁	The application will automatically use GPU if available for faster inference.

⦁	Confidence scores may be lower if the question is ambiguous or the context is insufficient.

⦁	Users can modify the context or question and click "Get Answer" to regenerate a new response.
