import os
from pymongo import MongoClient
from pymongo.collection import Collection

_client: MongoClient | None = None


def get_collection() -> Collection:
    """
    Returns the brains collection.
    Connection is created once and reused (MongoClient is thread-safe).
    """
    global _client
    if _client is None:
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise RuntimeError(
                "MONGODB_URI not set. Run with: doppler run -- uvicorn ..."
            )
        _client = MongoClient(uri)

    db = _client["contextbridge"]
    return db["brains"]


def save_brain(document: dict) -> str:
    """
    Saves a brain document to MongoDB.
    Returns the inserted document's string ID.
    """
    collection = get_collection()
    result = collection.insert_one(document)
    return str(result.inserted_id)


def get_brain(brain_id: str) -> dict | None:
    """
    Fetches a single brain by its MongoDB ObjectId string.
    Returns None if not found.
    """
    from bson import ObjectId

    collection = get_collection()
    doc = collection.find_one({"_id": ObjectId(brain_id)})
    if doc:
        doc["_id"] = str(doc["_id"])   
    return doc


def get_all_brains(user_id: str | None = None) -> list[dict]:
    """
    Returns all brains, optionally filtered by user_id.
    Sorted newest first.
    """
    collection = get_collection()
    query = {"user_id": user_id} if user_id else {}
    docs = collection.find(query).sort("created_at", -1).limit(50)

    results = []
    for doc in docs:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results