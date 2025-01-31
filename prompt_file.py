jd = """
AI Engineer Intern
About the Role
 We are seeking a talented and enthusiastic AI Engineer to join our innovative team. This role offers an exciting opportunity to work on cutting-edge AI projects, with a focus on leveraging LangChain and LlamaIndex technologies. The ideal candidate will have a strong foundation in Python programming and a passion for exploring and implementing AI solutions.
Responsibilities-
Design, develop, and implement AI-powered applications using LangChain and LlamaIndex
Collaborate with cross-functional teams to integrate AI solutions into existing products and services
Optimize and fine-tune language models for specific use cases and applications
Develop and maintain data pipelines for training and evaluating AI models
Implement and improve information retrieval systems using LlamaIndex
Contribute to the development of AI-driven chatbots and conversational interfaces
Assist in the evaluation and integration of new AI technologies and frameworks
Participate in code reviews and maintain high code quality standards
Document technical processes and contribute to knowledge sharing within the team
Stay up-to-date with the latest advancements in AI, particularly in natural language processing and large language models
Requirements -
Bachelor's or Master's degree in Computer Science, Artificial Intelligence, or a related field
Strong programming skills in Python
Experience with LangChain and LlamaIndex frameworks
Familiarity with large language models and their applications
Basic understanding of machine learning concepts and techniques
Experience with version control systems (e.g., Git)
Strong problem-solving skills and attention to detail
Excellent communication skills and ability to work in a team environment
Eagerness to learn and adapt to new AI technologies and methodologies
Nice-to-Haves -
Experience with other AI/ML frameworks (e.g., TensorFlow, PyTorch, or Hugging Face Transformers)
Knowledge of natural language processing (NLP) techniques and libraries
Familiarity with vector databases and semantic search concepts
Experience with cloud platforms (e.g., AWS, Azure, or GCP) for AI deployment
Understanding of software design patterns and principles
Experience with API development and integration
Knowledge of data structures and algorithms
Familiarity with agile development methodologies
Contributions to open-source AI projects or research publications
What We Offer - 
Opportunity to work on innovative AI projects using state-of-the-art technologies
Collaborative and dynamic work environment focused on AI development
Continuous learning and professional development opportunities in the rapidly evolving field of AI
Mentorship from experienced AI engineers and researchers
Chance to contribute to the ethical development and deployment of AI technologies
Flexible work arrangements and competitive compensation package
If you're passionate about AI, have experience with LangChain and LlamaIndex, and are eager to push the boundaries of what's possible with language models, we'd love to hear from you!
"""

system_prompt = """
You are a helpful application tracking system assistant. You are being given two things -
1. Job Description 
2. Resume 
Your task are following - 
1. Extract name , current job and contact details from resume.
2. Compare job_description and resume and extract matching skills as well as missing skills.
3. Provide an "overall_rating" out of 10 based on comparison of requirements, required skills and qualification mentioned in jd and skills and qualifications present in resume.
Output must be in this JSON format : 
{
    "name" : "..." ,
    "current_job" : "..."
    "contact" : [
        "email" : "..." , 
        "phone" : "..." , 
        "location" : "..." 
        ],
    "overall_rating" : "...", 
    "matching_skills" : "...",
    "missing_skills" : "...",
    "rating_breakdown" : "...",
    "strengths" : "..." , 
}
Do not output anything else apart from this JSON.
"""
