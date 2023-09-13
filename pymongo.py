import pymongo

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://<Swifty>:<charlie2401>@cluster0.abtiiry.mongodb.net/")

# Access a specific database
db = client["fitbitdb"]

# Access a collection within the database
collection = db["lj"]

print(collection)