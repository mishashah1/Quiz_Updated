import os
import streamlit as st
import pandas as pd
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator

# Load environment variables from .env file
load_dotenv()

# Define data model for Multiple Choice Questions using Pydantic
class MCQQuestion(BaseModel):
    question: str = Field(description="The question text")
    options: List[str] = Field(description="List of 4 possible answers")
    correct_answer: str = Field(description="The correct answer from the options")

    @validator('question', pre=True)
    def clean_question(cls, v):
        if isinstance(v, dict):
            return v.get('description', str(v))
        return str(v)

# Define data model for Fill in the Blank Questions using Pydantic
class FillBlankQuestion(BaseModel):
    question: str = Field(description="The question text with '_____' for the blank")
    answer: str = Field(description="The correct word or phrase for the blank")

    @validator('question', pre=True)
    def clean_question(cls, v):
        if isinstance(v, dict):
            return v.get('description', str(v))
        return str(v)

class QuizManager:
    def __init__(self):
        self.questions = []
        self.user_answers = []
        self.results = []
        self.current_topic = None
        self.current_difficulty = None
        self.current_quiz_id = None

    def reset_state(self):
        self.questions = []
        self.user_answers = []
        self.results = []

    def generate_quiz_id(self, topic, question_type, difficulty):
        import time
        import hashlib
        timestamp = str(time.time())
        quiz_str = f"{topic}_{question_type}_{difficulty}_{timestamp}"
        return hashlib.md5(quiz_str.encode()).hexdigest()[:8]

    def load_QuanBank_questions(self, file_key, difficulty, num_questions):
        """Load questions from QuanBank files based on structured module distributions (T1 or T2)."""
        QuanBank_files = {
            'T1': ('T1.xlsx', {1: 0.3, 2: 0.3, 3: 0.4}),
            'T2': ('T2.xlsx', {4: 0.5, 5: 0.5})
        }

        file_info = QuanBank_files.get(file_key)
        if not file_info:
            st.error(f"Invalid file key: {file_key}")
            return [], False

        file_path, module_distribution = file_info
        if not os.path.exists(file_path):
            st.error(f"QuanBank file not found: {file_path}")
            return [], False

        try:
            df = pd.read_excel(file_path)

            # Define fallback difficulty levels
            difficulty_map = {
                'Easy': ['B', 'I', 'E'],
                'Medium': ['I', 'B', 'E'],
                'Hard': ['E', 'I', 'B']
            }
            fallback_levels = difficulty_map.get(difficulty, ['I', 'B', 'E'])

            all_selected_questions = pd.DataFrame()
            raw_counts = {m: num_questions * p for m, p in module_distribution.items()}
            int_counts = {m: int(c) for m, c in raw_counts.items()}
            total_allocated = sum(int_counts.values())
            remaining = num_questions - total_allocated

            remainder_order = sorted(
                ((m, raw_counts[m] - int_counts[m]) for m in module_distribution),
                key=lambda x: x[1], reverse=True
            )

            for i in range(remaining):
                int_counts[remainder_order[i][0]] += 1


            for module, count in int_counts.items():
                module_df = df[df['MODULE'] == module]

                selected_module_questions = pd.DataFrame()
                for level in fallback_levels:
                    available = module_df[module_df['DIFFICULTY LEVEL'] == level]
                    n_needed = count - len(selected_module_questions)
                    if n_needed <= 0:
                        break
                    if not available.empty:
                        selected_module_questions = pd.concat([
                            selected_module_questions,
                            available.sample(min(n_needed, len(available)))
                        ], ignore_index=True)

                all_selected_questions = pd.concat([all_selected_questions, selected_module_questions], ignore_index=True)
                

            # If total exceeds desired due to rounding, sample down
            if len(all_selected_questions) > num_questions:
                all_selected_questions = all_selected_questions.sample(n=num_questions)

            # Shuffle the whole thing once after all modules added and downsampled
            all_selected_questions = all_selected_questions.sample(frac=1).reset_index(drop=True)

            if all_selected_questions.empty:
                st.warning("No questions found matching the criteria.")
                return [], False

            # Construct question list
            questions = []
            for _, row in all_selected_questions.iterrows():
                options = [row['OPTION1'], row['OPTION2'], row['OPTION3'], row['OPTION4']]
                correct_answer = options[ord(row['CORRECT ANSWER'].lower()) - ord('a')]
                questions.append({
                    'type': 'MCQ',
                    'question': row['QUESTION TEXT'],
                    'options': options,
                    'correct_answer': correct_answer,
                    'topic': row['TOPIC NAME'],
                    'quiz_id': self.current_quiz_id
                })

            return questions, True
        except Exception as e:
            st.error(f"Error reading QuanBank file: {e}")
            return [], False

    def generate_questions(self, generator, topic, question_type, difficulty, num_questions, source='AutoGen', module=None):
        fallback_used = False
        self.reset_state()
        self.current_topic = topic
        self.current_difficulty = difficulty
        self.current_quiz_id = self.generate_quiz_id(topic, question_type, difficulty)
        
        try:
            if source == 'QuanBank' and question_type == 'Multiple Choice':
                questions, success = self.load_QuanBank_questions(module, difficulty, num_questions)
                if success:
                    self.questions = questions
                else:
                    # Fallback to AutoGen with Python topic
                    source = 'AutoGen'
                    topic = "Python"
                    self.current_topic = topic
                    self.current_quiz_id = self.generate_quiz_id(topic, question_type, difficulty)
                    return True, fallback_used
            
            if source == 'AutoGen' or question_type != 'Multiple Choice':
                for _ in range(num_questions):
                    if question_type == 'Multiple Choice':
                        question = generator.generate_mcq(topic, difficulty.lower())
                        self.questions.append({
                            'type': 'MCQ',
                            'question': question.question,
                            'options': question.options,
                            'correct_answer': question.correct_answer,
                            'topic': topic,
                            'quiz_id': self.current_quiz_id
                        })
                    else:
                        question = generator.generate_fill_blank(topic, difficulty.lower())
                        self.questions.append({
                            'type': 'Fill in the Blank',
                            'question': question.question,
                            'correct_answer': question.answer,
                            'topic': topic,
                            'quiz_id': self.current_quiz_id
                        })
            
            self.user_answers = ["" if q['type'] == 'MCQ' else "" for q in self.questions]
            return True, fallback_used
        except Exception as e:
            st.error(f"Error generating questions: {e}")
            return False, fallback_used

    def attempt_quiz(self):
        for i, q in enumerate(self.questions):
            st.markdown(f"""
            <div class="question-card">
                <div class="question-number">Question {i+1}</div>
                <div class="question-text">{q['question']}</div>
            </div>
            """, unsafe_allow_html=True)

            if q['type'] == 'MCQ':
                radio_key = f"mcq_radio_{self.current_quiz_id}_{i}"
                selected_option = st.radio(
                    f"Select the correct answer for Question {i+1}:",
                    options=q['options'],
                    index=None,
                    key=radio_key,
                    label_visibility="visible"
                )
                self.user_answers[i] = selected_option if selected_option is not None else ""
            else:
                answer_key = f"fill_blank_{self.current_quiz_id}_{i}"
                if answer_key in st.session_state:
                    self.user_answers[i] = st.session_state[answer_key]
                
                user_answer = st.text_input(
                    f"Fill in the blank for Question {i+1}",
                    key=answer_key,
                    label_visibility="collapsed",
                    placeholder="Type your answer here..."
                )
                self.user_answers[i] = user_answer

    def evaluate_quiz(self):
        self.results = []
        for i, (q, user_ans) in enumerate(zip(self.questions, self.user_answers)):
            if q['type'] == 'MCQ':
                is_correct = user_ans == q['correct_answer'] if user_ans else False
                formatted_user_answer = user_ans if user_ans else "No selection"
                
                result_dict = {
                    'question_number': i + 1,
                    'question': q['question'],
                    'question_type': q['type'],
                    'topic': q['topic'],
                    'quiz_id': q['quiz_id'],
                    'user_answer': formatted_user_answer,
                    'correct_answer': q['correct_answer'],
                    'is_correct': is_correct,
                    'options': q['options']
                }
            else:
                is_correct = user_ans.strip().lower() == q['correct_answer'].strip().lower() if user_ans else False
                result_dict = {
                    'question_number': i + 1,
                    'question': q['question'],
                    'question_type': q['type'],
                    'topic': q['topic'],
                    'quiz_id': q['quiz_id'],
                    'user_answer': user_ans if user_ans else "No answer provided",
                    'correct_answer': q['correct_answer'],
                    'is_correct': is_correct,
                    'options': q['options'] if 'options' in q else []
                }
            self.results.append(result_dict)

    def generate_result_dataframe(self):
        return pd.DataFrame(self.results)

    def save_to_csv(self):
        try:
            if not self.results:
                st.warning("No results to save.")
                return None
            
            df = self.generate_result_dataframe()
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"quiz_results_{self.current_topic}_{timestamp}.csv"
            
            os.makedirs('results', exist_ok=True)
            full_path = os.path.join('results', unique_filename)
            df.to_csv(full_path, index=False)
            
            st.success(f"Results saved successfully!")
            return full_path
        except Exception as e:
            st.error(f"Failed to save results: {e}")
            return None

    def clear_session_state(self):
        keys_to_remove = []
        for key in st.session_state:
            if self.current_quiz_id and self.current_quiz_id in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del st.session_state[key]

class QuestionGenerator:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'), 
            model="llama-3.1-8b-instant",
            temperature=0.9
        )

    def generate_mcq(self, topic: str, difficulty: str = 'medium') -> MCQQuestion:
        mcq_parser = PydanticOutputParser(pydantic_object=MCQQuestion)
        prompt = PromptTemplate(
            template=(
                "Generate a {difficulty} multiple-choice question about {topic}.\n\n"
                "Return ONLY a JSON object with these exact fields:\n"
                "- 'question': A clear, specific question\n"
                "- 'options': An array of exactly 4 possible answers\n"
                "- 'correct_answer': One of the options that is the correct answer\n\n"
                "Example format:\n"
                '{{\n'
                '    "question": "What is the capital of France?",\n'
                '    "options": ["London", "Berlin", "Paris", "Madrid"],\n'
                '    "correct_answer": "Paris"\n'
                '}}\n\n'
                "Your response:"
            ),
            input_variables=["topic", "difficulty"]
        )

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = self.llm.invoke(prompt.format(topic=topic, difficulty=difficulty))
                parsed_response = mcq_parser.parse(response.content)
                if not parsed_response.question or len(parsed_response.options) != 4 or not parsed_response.correct_answer:
                    raise ValueError("Invalid question format")
                if parsed_response.correct_answer not in parsed_response.options:
                    raise ValueError("Correct answer not in options")
                return parsed_response
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate valid MCQ after {max_attempts} attempts: {str(e)}")
                continue

    def generate_fill_blank(self, topic: str, difficulty: str = 'medium') -> FillBlankQuestion:
        fill_blank_parser = PydanticOutputParser(pydantic_object=FillBlankQuestion)
        prompt = PromptTemplate(
            template=(
                "Generate a {difficulty} fill-in-the-blank question about {topic}.\n\n"
                "Return ONLY a JSON object with these exact fields:\n"
                "- 'question': A sentence with '_____' marking where the blank should be\n"
                "- 'answer': The correct word or phrase that belongs in the blank\n\n"
                "Example format:\n"
                '{{\n'
                '    "question": "The capital of France is _____.",\n'
                '    "answer": "Paris"\n'
                '}}\n\n'
                "Your response:"
            ),
            input_variables=["topic", "difficulty"]
        )

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = self.llm.invoke(prompt.format(topic=topic, difficulty=difficulty))
                parsed_response = fill_blank_parser.parse(response.content)
                if not parsed_response.question or not parsed_response.answer:
                    raise ValueError("Invalid question format")
                if "_____" not in parsed_response.question:
                    parsed_response.question = parsed_response.question.replace("___", "_____")
                    if "_____" not in parsed_response.question:
                        raise ValueError("Question missing blank marker '_____'")
                return parsed_response
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to generate valid fill-in-the-blank question after {max_attempts} attempts: {str(e)}")
                continue