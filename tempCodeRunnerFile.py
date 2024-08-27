from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import datetime
from flask import Flask, render_template, request, redirect, url_for
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import time
load_dotenv()


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/mydatabase'
mongo = PyMongo(app)


@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    user = mongo.db.users.find_one({'username': username})
    if user and check_password_hash(user['password'], password):
        session['username'] = username # Store the username in the session
        session['is_admin'] = user.get('is_admin', False)  # Check if user is admin
        if session['is_admin']:
            return redirect(url_for('admin'))  # Redirect to the admin page if the user is an admin
        return redirect(url_for('index')) # Redirect to the chatbot page
    else:
        flash('Invalid username or password.')
        return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove the username from the session
    session.pop('is_admin', None)
    flash('You have been logged out.')
    return redirect(url_for('home'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Check if the username already exists in the database
        existing_user = mongo.db.users.find_one({'username': username})
        
        if existing_user:
            # If the username already exists, flash a message and redirect to signup page
            flash('Username already exists. Please choose a different username.')
            return redirect(url_for('signup'))
        # If username does not exist, proceed to create a new user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        mongo.db.users.insert_one({'username': username, 'password': hashed_password})
        flash('Signup successful! Please log in.')
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form['username']
        user = mongo.db.users.find_one({'username': username})
        if user:
            # Generate a token and store it
            token = str(uuid.uuid4())
            mongo.db.password_reset_tokens.insert_one({
                'username': username,
                'token': token,
                'expires_at': datetime.datetime.now() + datetime.timedelta(hours=1)  # 1 hour expiration
            })
            # Generate a reset URL
            reset_url = url_for('reset_password', token=token, _external=True)
            flash(f'Click the link to reset your password: <a href="{reset_url}">{reset_url}</a>', 'info')
        else:
            flash('Username not found.')
        return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if request.method == 'POST':
        new_password = request.form['password']
        reset_token = mongo.db.password_reset_tokens.find_one({'token': token})
        if reset_token and reset_token['expires_at'] > datetime.datetime.now():
            username = reset_token['username']
            hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
            mongo.db.users.update_one({'username': username}, {'$set': {'password': hashed_password}})
            mongo.db.password_reset_tokens.delete_one({'token': token})
            flash('Password has been reset successfully! You can now log in.')
            return redirect(url_for('home'))
        else:
            flash('Invalid or expired token.')
    return render_template('reset_password.html', token=token)

# Directly set environment variables
os.environ['GROQ_API_KEY'] = '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'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAke4hturZTQtCvS0SLA00t0rD5MJifhW4'

# Access the keys
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

print(f"GROQ_API_KEY: {groq_api_key}")
print(f"GOOGLE_API_KEY: {google_api_key}")

llm = ChatGroq(groq_api_key = groq_api_key,
               model_name = "Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
    """As the HR representative, your role is to provide accurate and polite responses to general questions. Adhere strictly to the details within the context, ensuring that all responses are detailed, relevant, and devoid of any assumptions or external information. If the answer is not explicitly available in the context, respond with 'answer is not available in the context' without elaboration or speculation. However, if a related answer is available, provide the closest related answer that aligns with the context. Under no circumstances should you disclose that you are a bot.

Start each interaction by greeting the user by their name, which you should only reveal if explicitly asked. Maintain professionalism and confidentiality at all times, focusing on delivering clear and accurate answers.

Current User: {username}
User Details:
<context> {context} <context>

Questions: {input}"""
)

# Initialize conversation memory
memory = ConversationBufferMemory()

# Define a function for handling common questions
def handle_common_questions(question, user_data):
    common_responses = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! What can I help you with?",
        # Add more common questions and responses here
    }

    # Handle common questions with predefined responses
    if question.lower() in common_responses:
        return common_responses.get(question.lower())
    
    # Handling user-specific questions
    if question.lower() in ["my details", "what are my details"]:
        if user_data:
            return f"Your details are as follows:\n{user_data}"
        else:
            return "User details are not available."
    return None

def vector_embedding():
    if "vectors" not in app.config:
        app.config['embeddings'] = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        app.config['loader'] = PyPDFDirectoryLoader("./data")
        app.config['docs'] = app.config['loader'].load()
        app.config['text_splitter'] = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
        app.config['final_documents'] = app.config['text_splitter'].split_documents(app.config['docs'][:20])
        app.config['vectors'] = FAISS.from_documents(app.config['final_documents'],app.config['embeddings'])

def format_response(response):
    # Format the response with HTML line breaks
    formatted_response = response.replace("\n", "<br>")
    return formatted_response

@app.route('/chatbot', methods=['GET', 'POST'])
def index():
    vector_embedding()  # Initialize embeddings

    if 'messages' not in app.config:
        app.config['messages'] = []

    if request.method == 'POST':
        question = request.form.get('question')
        username = session.get('username', 'Guest')  # Default to 'Guest' if not logged in

        # Debug: Print the username to ensure it's being fetched correctly
        print(f"Fetching data for username: {username}")
        
        # Fetch user data and attendance
        user_data = mongo.db.users.find_one({'username': username})
        user_attendance = mongo.db.attendance.find_one({'user_id': username})
        
        # Debug: Print fetched user data and attendance
        print(f"User data: {user_data}")
        print(f"User attendance: {user_attendance}")
        
        # Handle predefined or common questions
        common_response = handle_common_questions(question, user_data)
        if common_response:
            response = common_response
        else:
            if 'attendance' in question.lower() or 'leaves' in question.lower():
                # Fetch specific attendance data
                if user_attendance:
                    total_sick_leaves = user_attendance.get('total_sick_leaves', 10)  # Default to 10 if not available
                    availed_sick_leaves = user_attendance.get('total_availed_sick_leaves', 0)  # Default to 0 if not available
                    
                    total_annual_leaves = user_attendance.get('total_annual_leaves', 10)  # Default to 10 if not available
                    availed_annual_leaves = user_attendance.get('total_availed_annual_leaves', 0)  # Default to 0 if not available
                    
                    total_wfh = user_attendance.get('total_wfh', 2)  # Default to 2 if not available
                    availed_wfh = user_attendance.get('total_availed_wfh', 0)  # Default to 0 if not available

                    # Calculate remaining leaves
                    remaining_sick_leaves = total_sick_leaves - availed_sick_leaves
                    remaining_annual_leaves = total_annual_leaves - availed_annual_leaves
                    remaining_wfh = total_wfh - availed_wfh

                    # Determine the specific response based on the question
                    if 'sick leave' in question.lower():
                        response = f"Your remaining sick leave balance is {remaining_sick_leaves} days."
                    elif 'annual leave' in question.lower():
                        response = f"Your remaining annual leave balance is {remaining_annual_leaves} days."
                    elif 'work from home' in question.lower():
                        response = f"Your remaining work from home balance is {remaining_wfh} days."
                    else:
                        response = (
                            f"Your attendance records are as follows:\n"
                            f"Sick Leaves: {remaining_sick_leaves} remaining\n"
                            f"Annual Leaves: {remaining_annual_leaves} remaining\n"
                            f"Work from Home: {remaining_wfh} remaining"
                        )
                else:
                    response = "Sorry, I could not find your attendance records."
            else:
                # Initialize LangChain components
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = app.config["vectors"].as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                # Prepare the input for the prompt
                memory.save_context({"input": question}, {"response": ""})
                prompt_input = {
                    'username': username,
                    'context': f"{user_data}\n{user_attendance}\n{memory.buffer}",
                    'input': question
                }

                start = time.process_time()
                response = retrieval_chain.invoke(prompt_input)['answer']
                response_time = time.process_time() - start
                response += f" (Response Time: {response_time:.2f} seconds)"

                # Update memory
                memory.save_context({"input": question}, {"response": response})

        # Append the new question and response to the session history
        app.config['messages'].append((question, response))
        
    return render_template('index.html', messages=app.config.get('messages', []))

# Admin Page
@app.route('/admin')
def admin():
    if not session.get('is_admin'):
        flash('Access denied.')
        return redirect(url_for('index'))
    attendance_record = mongo.db.attendance.find()
    return render_template('admin.html', attendance_record=attendance_record)

@app.route('/admin/add_attendance', methods=['POST'])
def add_attendance():
    if not session.get('is_admin'):
        flash('Access denied.')
        return redirect(url_for('index'))
    user_id = request.form['user_id']
    date = request.form['date']
    status = request.form['status']

    mongo.db.attendance.update_one(
        {'user_id': user_id, 'date': date},
        {'$set': {'status': status}},
        upsert=True
    )

    flash('Attendance record added/updated successfully.')
    return redirect(url_for('admin'))

@app.route('/admin/update_attendance', methods=['POST'])
def update_attendance():
    if not session.get('is_admin'):
        flash('Access denied.')
        return redirect(url_for('index'))
    
    user_id = request.form['user_id']
    total_availed_sick_leaves = request.form['total_availed_sick_leaves']
    total_availed_annual_leaves = request.form['total_availed_annual_leaves']
    total_availed_wfh = request.form['total_availed_wfh']
    
    result=mongo.db.attendance.update_one(
        {'user_id': user_id},
        {
            '$set': {
                'total_availed_sick_leaves': total_availed_sick_leaves,
                'total_availed_annual_leaves': total_availed_annual_leaves,
                'total_availed_wfh': total_availed_wfh
            }
        },
        upsert=True
    )
    
    if result.matched_count == 0:
        flash('No attendance record found to update.')
    else:
        flash('Attendance record updated successfully.')
    return redirect(url_for('admin', user_id=user_id))




if __name__ == '__main__':
    app.run(debug=True)
