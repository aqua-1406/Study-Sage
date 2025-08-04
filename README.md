# Study-Sage
📚 StudySage – AI-Powered Learning Assistant
    StudySage is an AI tool that helps students understand textbooks better by turning a book (PDF) into:

		 Chapter-wise summaries
		
		 Flashcards for quick revision
		
		 Quizzes to test understanding


🔍 **Features**

		 Upload any textbook (PDF format)
		
		 Detect and split content chapter-wise
		
		 Generate clean summaries using HuggingFace models (T5, BART)
		
		 Create flashcards for key points
		
		 Auto-generate quizzes from chapters
		
		 Streamlit-based user interface

		 Fully offline, free, and open-source

🛠️ **Tech Stack**

			Python 3.10+
			
			Streamlit – for the frontend UI
			
			PyPDF2 / pdfminer / fitz (PyMuPDF) – for PDF extraction
			
			Regex – for chapter splitting
			
			HuggingFace Transformers (t5-small, bart-base) – for text summarization
			
			Scikit-learn / NLTK / spaCy (optional) – for future quiz/flashcard generation
			
			Jupyter Notebook – for experimentation and model testing
