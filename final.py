# TODO: Import streamlit as st
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

# --- Page Configuration ---
# TODO: Set the page config with a title and icon
st.set_page_config(
    page_title="QA- ChatBot",
    page_icon="🤖",
    layout="wide"
)

# --- Model Loading ---
# TODO: Add the streamlit decorator to cache the model loading
@st.cache_resource
def load_model():
    # Path to your fine-tuned model
    model_path = "./arabert_qa_results"
    output_dir = "./arabert_qa_results"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        # Use GPU if available
        device = 0 if torch.cuda.is_available() else -1

        # TODO: Create a question-answering pipeline
        return pipeline(
            "question-answering",  # نوع المهمة: QA
            model=output_dir,      # المسار للنموذج الذي دربته
            tokenizer=output_dir   # المسار للـ tokenizer
        )

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# TODO: Call the load_model function to get the pipeline
qa_pipeline =  load_model()

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1E3A8A;
        margin-bottom: 20px;
    }
    .sub-header {
        text-align: center;
        color: #2563EB;
        margin: 20px 0px;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .get-answer-btn {
        background: linear-gradient(45deg, #10B981, #059669);
        color: white;
        border: none;
    }
    .get-answer-btn:hover {
        background: linear-gradient(45deg, #059669, #047857);
        transform: translateY(-2px);
    }
    .regenerate-btn {
        background: linear-gradient(45deg, #F59E0B, #D97706);
        color: white;
        border: none;
    }
    .regenerate-btn:hover {
        background: linear-gradient(45deg, #D97706, #B45309);
        transform: translateY(-2px);
    }
    .success-box {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(45deg, #D1FAE5, #A7F3D0);
        border-left: 5px solid #10B981;
        margin: 10px 0px;
    }
    .info-box {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(45deg, #DBEAFE, #93C5FD);
        border-left: 5px solid #3B82F6;
        margin: 10px 0px;
    }
    .warning-box {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(45deg, #FEF3C7, #FCD34D);
        border-left: 5px solid #F59E0B;
        margin: 10px 0px;
    }
    .error-box {
        padding: 15px;
        border-radius: 10px;
        background: linear-gradient(45deg, #FEE2E2, #FCA5A5);
        border-left: 5px solid #EF4444;
        margin: 10px 0px;
    }
    .section-divider {
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        height: 3px;
        border-radius: 2px;
        margin: 20px 0px;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #E5E7EB;
    }
    .stTextArea textarea:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- App Title and Description ---
st.markdown("<h1 class='main-header'>🤖 Arabic QA ChatBot</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='sub-header'>Enter context and ask your question to get answers!</h4>", unsafe_allow_html=True)
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# --- User Input ---
st.markdown("<h4 class='sub-header'> 📄 Context</h4>", unsafe_allow_html=True)
context = st.text_area("", height=200, placeholder="ادخل النص", label_visibility="collapsed")

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

st.markdown("<h4 class='sub-header'> ❓ Question</h4>", unsafe_allow_html=True)
question = st.text_area("", height=70, placeholder="ادخل سؤالك هنا", label_visibility="collapsed")

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# --- Buttons Layout ---
col1, col2, col3, col4 = st.columns([1,1,1,1])

with col2:
    get_answer_btn = st.button("Get Answer", key="get_answer", use_container_width=True)

with col3:
    regenerate_btn = st.button("Regenerate", key="regenerate", use_container_width=True)

# Add custom CSS for button colors
st.markdown("""
<script>
    // Color the buttons
    document.querySelector('[data-testid="baseButton-secondary"][kind="secondary"]').classList.add('get-answer-btn');
    document.querySelectorAll('[data-testid="baseButton-secondary"][kind="secondary"]')[1].classList.add('regenerate-btn');
</script>
""", unsafe_allow_html=True)

# Store previous inputs and results in session state
if 'previous_context' not in st.session_state:
    st.session_state.previous_context = ""
if 'previous_question' not in st.session_state:
    st.session_state.previous_question = ""
if 'previous_answer' not in st.session_state:
    st.session_state.previous_answer = ""
if 'previous_score' not in st.session_state:
    st.session_state.previous_score = 0.0

# --- Prediction ---
# Check if either button is clicked
if get_answer_btn or regenerate_btn:

    # Check if all inputs are provided
    if qa_pipeline and context and question:

        # TODO: Use st.spinner to show a loading message
        with st.spinner("🔍 Finding the answer... | ...جاري البحث عن الإجابة"):
            try:
                # TODO: Call the qa_pipeline with the question and context
                result = qa_pipeline(question=question, context=context)

                # Store results in session state
                st.session_state.previous_context = context
                st.session_state.previous_question = question
                st.session_state.previous_answer = result['answer']
                st.session_state.previous_score = result['score']

                # TODO: Display the answer using custom success box
                st.markdown(f"""
                <div class="success-box">
                    <h4 style="color: #065F46; margin:0;">✅ Answer</h4>
                    <p style="color: #065F46; font-size: 16px; margin: 8px 0px 0px 0px;"><strong>{result['answer']}</strong></p>
                </div>
                """, unsafe_allow_html=True)

                # TODO: Display the score using custom info box
                st.markdown(f"""
                <div class="info-box">
                    <h4 style="color: #1E40AF; margin:0;">📊 Confidence Score</h4>
                    <p style="color: #1E40AF; font-size: 16px; margin: 8px 0px 0px 0px;"><strong>{result['score']:.4f}</strong></p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <h4 style="color: #7F1D1D; margin:0;">❌ Error</h4>
                    <p style="color: #7F1D1D; margin: 8px 0px 0px 0px;">An error occurred during prediction: {e}</p>
                </div>
                """, unsafe_allow_html=True)

    elif not qa_pipeline:
        st.markdown(f"""
        <div class="error-box">
            <h4 style="color: #7F1D1D; margin:0;">❌ Model Error</h4>
            <p style="color: #7F1D1D; margin: 8px 0px 0px 0px;">Model could not be loaded. Please check the logs.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # TODO: Add a warning if context or question is missing
        st.markdown(f"""
        <div class="warning-box">
            <h4 style="color: #92400E; margin:0;">⚠️ Missing Information</h4>
            <p style="color: #92400E; margin: 8px 0px 0px 0px;">Please provide both Context and Question to get an answer.</p>
        </div>
        """, unsafe_allow_html=True)

# Display previous answer if Regenerate was clicked but no new inputs
elif regenerate_btn and st.session_state.previous_answer:
    st.markdown(f"""
    <div class="success-box">
        <h4 style="color: #065F46; margin:0;">✅ Answer (Regenerated)</h4>
        <p style="color: #065F46; font-size: 16px; margin: 8px 0px 0px 0px;"><strong>{st.session_state.previous_answer}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        <h4 style="color: #1E40AF; margin:0;">📊 Confidence Score</h4>
        <p style="color: #1E40AF; font-size: 16px; margin: 8px 0px 0px 0px;"><strong>{st.session_state.previous_score:.4f}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="warning-box">
        <h4 style="color: #92400E; margin:0;">🔄 Using Previous Inputs</h4>
        <p style="color: #92400E; margin: 8px 0px 0px 0px;">To get a new answer, modify the context or question and click 'Get Answer'.</p>
    </div>
    """, unsafe_allow_html=True)

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)