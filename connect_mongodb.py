import pymongo
import pprint

# Connect to MongoDB
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['mydatabase']

# Fetch collections
users_collection = db['users']  # Corrected collection name
attendance_collection = db['attendance']

# Fetch documents from collections
users = list(users_collection.find())
attendances = list(attendance_collection.find())

# Print documents
print("Users:")
pprint.pprint(users)

print("\nAttendances Sample:")
pprint.pprint(attendances)

# Flatten the attendance collection if necessary
attendance_records = []
for doc in attendances:
    # Check if the document contains numeric keys
    if isinstance(doc, dict) and any(key.isdigit() for key in doc.keys()):
        # Extract individual records from the dictionary with numeric keys
        for key in doc:
            if key != '_id':  # Skip the _id field
                attendance_records.append(doc[key])
    else:
        # If not numeric keys, treat the document as a single record
        attendance_records.append(doc)

# Extracting user data for processing
user_attendance = []
for user_doc in users:
    user_id = user_doc.get("username")  # Use 'username' for matching
    if user_id:
        attendance_doc = next((item for item in attendance_records if item.get("user_id") == user_id), None)
        if attendance_doc:
            user_attendance.append({
                "user_id": user_id,
                "password": user_doc.get("password"),  # Password from the users collection
                **attendance_doc
            })

print("\nUser Attendance:")
pprint.pprint(user_attendance)
