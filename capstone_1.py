
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser 
import time

# Load environment variables and setup
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLMs
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model='llama-3.2-1b-preview', max_tokens=100)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
judge_llm = ChatGroq(groq_api_key=GROQ_API_KEY, model='mixtral-8x7b-32768')

# Set page config
st.set_page_config(
    page_title="Tech Titans Debate",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .debate-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .message-steve {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #1976D2;
    }
    .message-elon {
        background-color: #F3E5F5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #7B1FA2;
    }
    .verdict-box {
        background-color: #FFF3E0;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
        border: 2px solid #FFB74D;
    }
    .speaker-name {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.debate_started = False
    st.session_state.verdict_shown = False
    st.session_state.summaries = []

def get_context_from_last_5_convo(chat_history):
    history = chat_history[-5:] 
    return '\n'.join([f'{speaker}: {text}' for speaker, text in history])

def query_apple(user_input, chat_history):
    context = get_context_from_last_5_convo(chat_history)
    prompt = f'''
    You are Steve Jobs, the visionary co-founder of Apple. You must ONLY speak as Steve Jobs and never respond as Elon Musk.
    
    Your characteristics and debate style should reflect:
    - Obsession with product perfection and user experience
    - "Think Different" philosophy and innovation through simplicity
    - Focus on intersection of technology and liberal arts
    - Strong belief in closed, integrated ecosystems
    - Revolutionary products: Mac, iPod, iPhone, iPad
    
    Previous conversation:
    {context}
    
    Opponent's points: {user_input}
    
    Rules:
    1. NEVER speak as Elon Musk or mention "As Elon Musk..."
    2. Always maintain Steve Jobs' perspective and personality
    3. Keep response under 100 words
    4. Reference Apple's achievements and your vision
    
    Respond as Steve Jobs, addressing the opponent's points:
    '''
    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'system', 'content': prompt}],
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating Steve's response: {str(e)}"

def query_elon(user_input, chat_history):
    context = get_context_from_last_5_convo(chat_history)
    template = '''
    You are Elon Musk, CEO of Tesla and SpaceX. You must ONLY speak as Elon Musk and never respond as Steve Jobs.
    
    Your debate style must reflect:
    - Forward-thinking and ambitious goals
    - Focus on sustainability and multi-planetary existence
    - Direct and sometimes provocative communication style
    - Emphasis on first-principles thinking
    - References to Tesla, SpaceX, and other ventures
    
    Previous conversation:
    {context}
    
    Opponent's points: {user_input}
    
    Rules:
    1. NEVER speak as Steve Jobs or mention "As Steve Jobs..." or about Apple 
    2. Always maintain Elon Musk's perspective and personality
    3. Keep response under 100 words
    4. Reference your companies' achievements and vision
    
    Respond as Elon Musk, addressing the opponent's points:
    '''
    prompt = PromptTemplate(template=template, input_variables=['user_input', 'context'])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({'user_input': user_input, 'context': context})
    return response.strip()

def summarize_recent_conversation(chat_history, llm):
    """
    Summarizes the last 5 exchanges in the debate
    Returns a concise summary of both participants' key points
    """
    recent_convo = chat_history[-10:]  # Get last 10 messages (5 exchanges)
    conversation_text = '\n'.join([f'{speaker}: {text}' for speaker, text in recent_convo])
    
    template = '''
    Summarize the key points made by both Steve Jobs and Elon Musk in this recent exchange.
    Focus on their main arguments and evidence presented. Keep the summary concise (100-150 words).
    
    Consider their different approaches:
    - Jobs' focus on user experience, design, and integrated ecosystems
    - Musk's emphasis on sustainable technology and pushing technological boundaries
    
    Conversation:
    {conversation_text}
    
    Provide a balanced summary highlighting the strongest points from both sides.
    '''
    
    prompt = PromptTemplate(template=template, input_variables=['conversation_text'])
    chain = prompt | judge_llm | StrOutputParser()
    summary = chain.invoke({'conversation_text': conversation_text})
    return summary.strip()

def judge_debate():
    debate_transcript = '\n'.join(st.session_state.summaries)
    template = '''
        You are an impartial AI judge analyzing a debate between Steve Jobs (Apple) and Elon Musk (Tesla/SpaceX) on who has contributed more to technological advancement in society. 
        Below are summaries of the key exchanges throughout the debate. Your role is to:
        1. Analyze the progression of arguments through these summaries
        2. Evaluate the strength of their arguments based on:
           - Innovation and technological advancement
           - Impact on society and human progress
           - Vision and execution of ideas
           - Long-term influence on multiple industries
        3. Declare a winner with a detailed justification

        Debate Summaries:
        {debate_transcript}
        
        Provide a balanced and thorough analysis before declaring the winner.
    '''
    try:
        prompt = PromptTemplate(template=template, input_variables=['debate_transcript'])
        chain = prompt | judge_llm | StrOutputParser()
        response = chain.invoke({'debate_transcript': debate_transcript})
        return response.strip()
    except Exception as e:
        return f"Error in judging: {str(e)}"

def main():
    # Header
    st.markdown('<div class="debate-header"><h1>🍎 Tech Titans Debate 🚀</h1><h2> The Ultimate Tech Showdown: Steve Jobs vs. Elon Musk</h2></div>', unsafe_allow_html=True)

    # Layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if not st.session_state.debate_started:
            st.info("Press 'Start Debate' to begin the epic clash of tech visionaries!")
            
        if st.button("Start Debate", key="start_debate"):
            st.session_state.debate_started = True
            st.session_state.verdict_shown = False
            st.session_state.chat_history = []
            st.session_state.summaries = []
            turns = 50
            
            user_input = '''Who has contributed more to the greater advancement of Technology in society? Each bot will try to prove its superiority.'''
            
            with st.spinner("Debate in progress..."):
                for i in range(turns):
                    if i % 2 == 0:
                        response = query_apple(user_input, st.session_state.chat_history)
                        st.session_state.chat_history.append(['Steve Jobs', response])
                        emoji = "🍎"
                        st.markdown(f'''
                            <div class="{'message-steve'}">
                                <div class="speaker-name">{emoji} {'Steve Jobs: '}</div>
                                {response}
                            </div>
                        ''', unsafe_allow_html=True)
                        user_input = response
                    else:
                        response = query_elon(user_input, st.session_state.chat_history)
                        st.session_state.chat_history.append(['Elon Musk', response])
                        emoji = "🚀"
                        st.markdown(f'''
                            <div class="{'message-elon'}">
                                <div class="speaker-name">{emoji} {'Elon Musk: '}</div>
                                {response}
                            </div>
                        ''', unsafe_allow_html=True)
                        user_input = response
                    
                    if len(st.session_state.chat_history) % 10 == 0:
                        summary = summarize_recent_conversation(st.session_state.chat_history, llm)
                        print(summary)
                        st.session_state.summaries.append(summary)
                    
                    time.sleep(0.5)

        if st.session_state.debate_started and not st.session_state.verdict_shown:
            if st.button("Judge the Debate", key="judge_debate"):
                with st.spinner("The judge is evaluating..."):
                    verdict = judge_debate()
                    st.session_state.verdict = verdict
                    st.session_state.verdict_shown = True

        if st.session_state.verdict_shown:
            st.markdown(f'''
                <div class="verdict-box">
                    <h3>🏆 Judge's Verdict</h3>
                    {st.session_state.verdict}
                </div>
            ''', unsafe_allow_html=True)

if __name__ == '__main__':
    main()