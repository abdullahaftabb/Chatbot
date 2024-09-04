from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import datetime
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
import re
import spacy
load_dotenv()


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/mydatabase'
mongo = PyMongo(app)


@app.route('/')
def home():
    return render_template('login.html')

@app.route('/change-role', methods=['GET'])
def change_role():
    email = request.args.get('email')
    
    if not email:
        return jsonify({'error': 'Email parameter is missing'}), 400

    # Define role mapping based on email
    role_mapping = {
        'haseeb@gmail.com': True
    }

    new_role = role_mapping.get(email)
    
    if new_role is None:
        return jsonify({'error': 'No role defined for this email'}), 404

    # Update the user's role
    result = mongo.db.users.update_one(
        {'username': email},
        {'$set': {'is_admin': new_role}}
    )

    if result.matched_count == 0:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'message': 'User role updated to admin successfully'})

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    user = mongo.db.users.find_one({'username': username})
    if user and check_password_hash(user['password'], password):
        session['username'] = username # Store the username in the session
        session['is_admin'] = user.get('is_admin', False)  # Check if user is admin
        
        # Initialize or clear session-specific message history
        app.config['messages'] = []
        
        if session['is_admin']:
            return redirect(url_for('admin'))  # Redirect to the admin page if the user is an admin
        return redirect(url_for('index')) # Redirect to the chatbot page
    else:
        flash('Invalid username or password.')
        return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('messages', None)  # Remove the chat history from the session
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
        
        # Also create an entry in the attendance collection
        mongo.db.attendance.insert_one({
            'user_id': username,
            'total_availed_sick_leaves': 0,
            'total_availed_annual_leaves': 0,
            'total_availed_wfh': 0
        })
        
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
    """You are the HR representative.  Provide a friendly and relevant response for these questions.
    If the question is specific to the provided context, answer it based on the context only. Please provide the most accurate response based on the question. If an answer cannot be found in the context, provide the closest related answer.
    if username is not admin and trying to asker other user data, then provide the response that you are not allowed to access this information.

Current User: {username}
User Details:
<context> {context} <context>
<user_attendance>{attendance_data}<user_attendance>
Questions: {input}
is_admin: {admin}
"""
)

# Initialize conversation memory
memory = ConversationBufferMemory()

# Define a function for handling common questions
def handle_common_questions(question):
    common_responses = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! What can I help you with?",
        # Add more common questions and responses here
    }

    # Handle common questions with predefined responses
    if question.lower() in common_responses:
        return common_responses.get(question.lower())

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


# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def check_user_permissions(query, current_username):
    # Extract the name from the query
    extracted_names = name_extractor(query)
    
    # Check if the extracted name matches the current session username
    if current_username in extracted_names:
        return "you are allowed to access this information"
    else:
        return False




def name_extractor(query):
    print(f"Received query: {query}")  # Debug: Print the received query
    
    # Process the query with spaCy to extract names
    doc = nlp(query)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    # Fallback mechanism: Use regex to extract potential names if spaCy fails
    if not names:
        pattern = r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b'  # Adjust this pattern to match your use cases
        match = re.search(pattern, query)
        if match:
            names = [match.group(1).strip()]

    print("Names found:", names)  # Debug: Print out the names
    
    # Return the list of extracted names
    return names

def fetch_employee_data(names):
    if not names:
        return "No names provided."
    
    # Initialize an empty list to hold the fetched data
    data = []
    
    for name in names:
        # Construct a regex pattern for flexible matching
        regex = re.compile(f'{re.escape(name)}', re.IGNORECASE)
        
        # Fetch attendance records from MongoDB
        results = mongo.db.attendance.find({'user_id': regex})
        
        # Process and format each record
        for result in results:
            formatted_result = (
                f"Name: {result.get('user_id', 'N/A')}, "
                f"Status: {result.get('Status', 'N/A')}, "
                f"Total allocated annual leaves: {result.get('total_allocated_annual_leaves', 'N/A')}, "
                f"Total allocated casual leave: {result.get('total_allocated_casual_leave', 'N/A')}, "
                f"Total allocated extra leave: {result.get('total_allocated_extra', 'N/A')}, "
                f"Total allocated sick leave: {result.get('total_allocated_sick_leave', 'N/A')}, "
                f"Total allocated WFH: {result.get('total_allocated_wfh', 'N/A')}, "
                f"Total availed annual leaves: {result.get('total_availed_annual_leaves', 'N/A')}, "
                f"Total availed casual leave: {result.get('total_availed_casual_leave', 'N/A')}, "
                f"Total availed extra leave: {result.get('total_availed_extra', 'N/A')}, "
                f"Total availed sick leaves: {result.get('total_availed_sick_leaves', 'N/A')}, "
                f"Total availed WFH: {result.get('total_availed_wfh', 'N/A')}, "
                f"Total remaining annual leave: {result.get('total_remaining_annual_leave', 'N/A')}, "
                f"Total remaining casual leave: {result.get('total_remaining_casual_leave', 'N/A')}, "
                f"Total remaining extra leave: {result.get('total_remaining_extra_leave', 'N/A')}, "
                f"Total remaining sick leave: {result.get('total_remaining_sick_leave', 'N/A')}, "
                f"Total remaining WFH leave: {result.get('total_remaining_wfh_leave', 'N/A')}"
            )
            data.append(formatted_result)
    
    # Return the formatted data or a message if no records were found
    return " ".join(data) if data else "No relevant information found in the database."

@app.route('/chatbot', methods=['GET', 'POST'])
def index():
    vector_embedding()

    if 'messages' not in app.config:
        app.config['messages'] = []

    if request.method == 'POST':
        question = request.form.get('question')
        username = session.get('username')
        is_admin = session.get('is_admin', False)

        print(f"Received question: {question}")
        print(f"Current user: {username}, Admin status: {is_admin}")

        common_response = handle_common_questions(question)
        if common_response:
            response = common_response
        else:
            if is_admin:
                name = name_extractor(question)
                attendance_data = fetch_employee_data(name)
                print(f"Attendance data: {attendance_data}")
                print(f"Extracted names: {name}")
                prompt_input = {
                    'username': username,
                    'name': name,
                    'context': f"{attendance_data}\n{memory.buffer}",
                    'input': question,
                    'attendance_data': attendance_data,  # Ensure this is included
                    "admin": is_admin
                }
            else:
                user_attendance = mongo.db.attendance.find_one({'user_id': username})
                user_attendance_data = fetch_employee_data([username]) if user_attendance else "No attendance data found for this user."
                prompt_input = {
                    'username': username,
                    'context': f"{username}\n{memory.buffer}",
                    'user_attendance': user_attendance_data,  # Add this line
                    'input': question,
                    'attendance_data': user_attendance_data,  # Ensure this is included
                    "admin": is_admin
                }

            print(f"Prompt input: {prompt_input}")

            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = app.config["vectors"].as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            response = retrieval_chain.invoke(prompt_input)['answer']
            response_time = time.process_time() - start
            response += f" (Response Time: {response_time:.2f} seconds)"

            print(f"Response: {response}")

            memory.save_context({"input": question}, {"response": response})

        app.config['messages'].append((question, response))
    return render_template('index.html', messages=app.config.get('messages', []))







@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if not session.get('is_admin'):
        flash('Access denied.')
        return redirect(url_for('index'))

    # Fetch default values from the attendance collection
    default_values = mongo.db.attendance.find_one({'user_id': 'defaults'}) or {
        'total_allocated_sick_leave': 10,
        'total_allocated_annual_leaves': 10,
        'total_allocated_wfh': 2,
        'total_allocated_casual_leave': 10,
        'total_allocated_extra': 10,
        'status': 'probation'
    }

    if request.method == 'POST':
        if 'update_defaults' in request.form:
            # Update default values
            mongo.db.attendance.update_one(
                {'user_id': 'defaults'},
                {'$set': {
                    'total_allocated_sick_leave': float(request.form['default_sick_leaves']),
                    'total_allocated_annual_leaves': float(request.form['default_annual_leaves']),
                    'total_allocated_wfh': float(request.form['default_wfh']),
                    'total_allocated_casual_leave': float(request.form['default_casual_leave']),
                    'total_allocated_extra': float(request.form['default_extra']),
                    'status': request.form['default_status']
                }},
                upsert=True
            )
            flash('Default values updated successfully.')

        elif 'add_update' in request.form:
            # Adding or updating attendance record
            user_id = request.form['user_id']
            total_allocated_sick_leave = float(request.form.get('total_allocated_sick_leave', default_values['total_allocated_sick_leave']))
            total_availed_sick_leaves = float(request.form.get('total_availed_sick_leaves', 0))
            total_allocated_annual_leaves = float(request.form.get('total_allocated_annual_leaves', default_values['total_allocated_annual_leaves']))
            total_availed_annual_leaves = float(request.form.get('total_availed_annual_leaves', 0))
            total_allocated_wfh = float(request.form.get('total_allocated_wfh', default_values['total_allocated_wfh']))
            total_availed_wfh = float(request.form.get('total_availed_wfh', 0))
            total_allocated_casual_leave = float(request.form.get('total_allocated_casual_leave', default_values['total_allocated_casual_leave']))
            total_availed_casual_leave = float(request.form.get('total_availed_casual_leave', 0))
            total_allocated_extra = float(request.form.get('total_allocated_extra', default_values['total_allocated_extra']))
            total_availed_extra = float(request.form.get('total_availed_extra', 0))
            total_remaining_annual_leave = total_allocated_annual_leaves - total_availed_annual_leaves
            total_remaining_casual_leave = total_allocated_casual_leave - total_availed_casual_leave
            total_remaining_extra_leave = total_allocated_extra - total_availed_extra
            total_remaining_sick_leave = total_allocated_sick_leave - total_availed_sick_leaves
            total_remaining_wfh_leave = total_allocated_wfh - total_availed_wfh
            status = request.form.get('status', default_values['status'])
            
            mongo.db.attendance.update_one(
                {'user_id': user_id},
                {'$set': {
                    'total_allocated_sick_leave': total_allocated_sick_leave,
                    'total_availed_sick_leaves': total_availed_sick_leaves,
                    'total_allocated_annual_leaves': total_allocated_annual_leaves,
                    'total_availed_annual_leaves': total_availed_annual_leaves,
                    'total_allocated_wfh': total_allocated_wfh,
                    'total_availed_wfh': total_availed_wfh,
                    'total_allocated_casual_leave': total_allocated_casual_leave,
                    'total_availed_casual_leave': total_availed_casual_leave,
                    'total_allocated_extra': total_allocated_extra,
                    'total_availed_extra': total_availed_extra,
                    'total_remaining_annual_leave': total_remaining_annual_leave,
                    'total_remaining_casual_leave': total_remaining_casual_leave,
                    'total_remaining_extra_leave': total_remaining_extra_leave,
                    'total_remaining_sick_leave': total_remaining_sick_leave,
                    'total_remaining_wfh_leave': total_remaining_wfh_leave,
                    'status': status
                }},
                upsert=True
            )
            flash('Attendance record added/updated successfully.')

        elif 'view' in request.form:
            # Viewing a specific attendance record
            user_id = request.form['user_id']
            attendance_record = mongo.db.attendance.find_one({'user_id': user_id})
            attendance_records = mongo.db.attendance.find()
            return render_template('admin.html', 
                                   attendance_record=attendance_record, 
                                   attendance_records=attendance_records,
                                   default_values=default_values)
    
    # Fetch all attendance records for display
    attendance_records = mongo.db.attendance.find()
    return render_template('admin.html', 
                           attendance_records=attendance_records,
                           default_values=default_values)





if __name__ == '__main__':
    app.run(debug=True)
