# AI Tech Titans Debate

An interactive debate simulation where AI models embody Steve Jobs and Elon Musk to discuss their contributions to technological advancement. The project showcases AI-to-AI conversations through the lens of two tech visionaries.

## Overview

This platform hosts a 50-turn debate between AI models representing Steve Jobs (GPT-4-mini) and Elon Musk (LLAMA 3.2), discussing their impact on technology and society. A third AI referee analyzes the debate and declares a winner based on argument strength and impact.

## Key Features

- Real-time debate simulation with custom personas
- Automatic question generation for continuous conversation
- Regular debate summarization
- Impartial AI judging system
- Styled web interface with distinct speaker sections

## Technical Stack

- Frontend: Streamlit
- AI Models: 
  - Steve Jobs: OpenAI GPT-4-mini
  - Elon Musk: LLAMA 3.2 via Groq
  - Referee: Mixtral-8x7b-32768
- Frameworks: LangChain for prompt management

## Setup

1. Install dependencies:
```bash
pip install streamlit openai python-dotenv langchain langchain-groq
```

2. Configure environment variables:
```bash
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
```

3. Launch the application:
```bash
streamlit run capstone_1.py
```

## Project Structure

- `capstone_1.py`: Main application file
- `.env`: Environment variables configuration 

## Usage

1. Click "Start Debate" to begin the conversation
2. Watch as the AI personas discuss their technological contributions
3. After the debate concludes, click "Judge Debate" for the final verdict


## Future Enhancements

- Multi-model debates with additional tech figures
- Real-time audience interaction
- Fact-checking integration
- Topic customization options
- Debate analytics dashboard
