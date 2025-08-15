# CourseGuide

CourseGuide is a chatbot-powered web application designed to help users navigate course sequences, graduation requirements, and school organizations. It features a React frontend and a Flask backend, with support for local LLMs.

## Features
- Interactive chatbot interface 
- Course sequence and graduation requirement lookup
- School clubs and organizations info
- Powered by local language models

## Project Structure
```
CourseGuide/
├── App.jsx
├── CourseGuide-env.yml
├── req.txt
├── Backend/
│   ├── app.py
│   ├── backend.py
│   ├── backker.py
│   └── ...
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── ...
│   ├── public/
│   └── ...
├── Llama/
│   └── ...
├── Model/
│   └── ...
```

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js & npm
- (Optional) Conda for environment management

### Backend Setup
1. Navigate to `Backend` folder:
   ```cmd
   cd Backend
   ```
2. Install Python dependencies:
   ```cmd
   pip install -r ../req.txt
   ```
3. Start the Flask server:
   ```cmd
   python app.py
   ```

### Frontend Setup
1. Navigate to `frontend` folder:
   ```cmd
   cd frontend
   ```
2. Install Node dependencies:
   ```cmd
   npm install
   ```
3. Start the frontend:
   ```cmd
   npm run dev
   ```

### Usage
- Open your browser and go to the frontend URL (usually `http://localhost:5173`)
- Interact with the chatbot to get course info and more

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Improvements to Address
1. Implement caching for chat history (currently not stored).
2. Prevent the database from reinitializing on each query.

## License
MIT
