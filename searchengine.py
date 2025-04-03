from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Union
from pydantic import BaseModel
import os
import uvicorn
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import pymongo
from bson import ObjectId
import boto3
from dotenv import load_dotenv
from botocore.client import Config
import json
import datetime

# Load environment variables
load_dotenv()

app = FastAPI(title="Image Search API", description="API for searching images using CLIP embeddings")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MONGODB_URL = os.getenv("MONGODB_URL")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
BUCKET = os.getenv("BUCKET")
REGION = os.getenv("REGION")
EMBEDDING_DIMENSION = 512  # Dimension of CLIP embeddings

# Models
class SearchResult(BaseModel):
    image_id: str
    image_url: str
    similarity_score: float
    metadata: dict

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str

# Initialize MongoDB client
mongo_client = pymongo.MongoClient(MONGODB_URL)
db = mongo_client["pixelmind"]  # Use your actual database name
posts_collection = db["userposts"]  # Collection containing posts with images
comments_collection = db["comments"]  # Collection containing comments
replies_collection = db["forumreplies"]  # Collection containing replies
embeddings_collection = db["image_embeddings"]  # Collection for storing embeddings

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY,
    region_name=REGION
)

# SQS client for receiving S3 events
sqs_client = boto3.client(
    'sqs',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY,
    region_name=REGION
)

# Initialize CLIP model
model = SentenceTransformer('clip-ViT-B-32')

# Create indexes on MongoDB collections
@app.on_event("startup")
async def startup_event():
    # Create index on embedding vectors for faster similarity search
    embeddings_collection.create_index([("url", pymongo.ASCENDING)], unique=True)
    embeddings_collection.create_index([("image_id", pymongo.ASCENDING)])
    embeddings_collection.create_index([("collection", pymongo.ASCENDING)])
    
    # Create index on imageUrl for faster lookups
    posts_collection.create_index([("imgKey", pymongo.ASCENDING)])
    comments_collection.create_index([("mediaAttachments", pymongo.ASCENDING)])
    replies_collection.create_index([("mediaAttachments", pymongo.ASCENDING)])

def extract_images_from_mongodb():
    """Extract all images from MongoDB collections that don't have embeddings yet"""
    image_data = []
    
    # Extract from posts (user posts)
    post_images = posts_collection.find({"imgKey": {"$exists": True, "$ne": None}})
    for post in post_images:
        # Check if embedding already exists
        if not embeddings_collection.find_one({"image_id": str(post["_id"])}):
            # For posts, the imgKey is directly the S3 key
            image_data.append({
                "id": str(post["_id"]),
                "url": f"https://{BUCKET}.s3.{REGION}.amazonaws.com/{post['imgKey']}",
                "key": post['imgKey'],  # Store the direct key for easier access
                "collection": "posts"
            })
    
    # Extract from comments with media attachments
    comment_images = comments_collection.find({"mediaAttachments": {"$exists": True, "$ne": []}})
    for comment in comment_images:
        if comment.get("mediaAttachments") and len(comment["mediaAttachments"]) > 0:
            for attachment in comment["mediaAttachments"]:
                if "fileUrl" in attachment and attachment.get("fileType", "").startswith("image/"):
                    # Use attachment ID as unique identifier
                    attachment_id = str(attachment.get("_id", ""))
                    if not embeddings_collection.find_one({"image_id": attachment_id}):
                        # Extract the key from fileName if available
                        key = attachment.get("fileName", "")
                        image_data.append({
                            "id": attachment_id,
                            "url": attachment["fileUrl"],
                            "key": key,  # Store the direct key if available
                            "parent_id": str(comment["_id"]),
                            "collection": "comments"
                        })
    
    # Extract from forum replies with media attachments
    reply_images = replies_collection.find({"mediaAttachments": {"$exists": True, "$ne": []}})
    for reply in reply_images:
        if reply.get("mediaAttachments") and len(reply["mediaAttachments"]) > 0:
            for attachment in reply["mediaAttachments"]:
                if "fileUrl" in attachment and attachment.get("fileType", "").startswith("image/"):
                    # Use attachment ID as unique identifier
                    attachment_id = str(attachment.get("_id", ""))
                    if not embeddings_collection.find_one({"image_id": attachment_id}):
                        # Extract the key from fileName if available
                        key = attachment.get("fileName", "")
                        image_data.append({
                            "id": attachment_id,
                            "url": attachment["fileUrl"],
                            "key": key,  # Store the direct key if available
                            "parent_id": str(reply["_id"]),
                            "collection": "replies"
                        })
    
    return image_data

async def download_image_from_s3(image_url, direct_key=None):
    """Download image from S3 bucket"""
    try:
        # Use direct key if provided
        if direct_key:
            key = direct_key
        else:
            # Extract key from URL
            # Handle different URL formats
            if "/" + BUCKET + ".s3" in image_url:
                # Format: https://bucketname.s3.region.amazonaws.com/key
                key = image_url.split(BUCKET + ".s3")[1].lstrip(".")
                if key.startswith("/"):
                    key = key[1:]  # Remove leading slash
                if "amazonaws.com/" in key:
                    key = key.split("amazonaws.com/")[1]
            elif "amazonaws.com/" + BUCKET + "/" in image_url:
                # Format: https://s3.region.amazonaws.com/bucketname/key
                key = image_url.split(BUCKET + "/")[1]
            else:
                # Direct key format
                key = image_url.split("/")[-1]
        
        # Remove any query parameters
        if "?" in key:
            key = key.split("?")[0]
            
        print(f"Extracted S3 key: {key} from URL: {image_url}")
        
        # Download file to memory
        response = s3_client.get_object(Bucket=BUCKET, Key=key)
        image_data = response['Body'].read()
        
        # Convert to PIL Image
        return Image.open(io.BytesIO(image_data))
    except Exception as e:
        print(f"Error downloading image from S3: {str(e)} for URL: {image_url}")
        return None

async def generate_embedding_for_image(img_data):
    """Generate CLIP embedding for a single image and store it in MongoDB"""
    try:
        # Skip if embedding already exists
        existing = embeddings_collection.find_one({"image_id": img_data["id"]})
        if existing:
            return True
        
        # Download image from S3
        direct_key = img_data.get("key")
        image = await download_image_from_s3(img_data["url"], direct_key)
        
        if not image:
            print(f"Failed to download image for {img_data['id']}")
            return False
        
        # Generate embedding
        embedding = model.encode(image)
        
        # Store in MongoDB
        embeddings_collection.insert_one({
            "image_id": img_data["id"],
            "url": img_data["url"],
            "collection": img_data["collection"],
            "parent_id": img_data.get("parent_id"),
            "embedding": embedding.tolist(),
            "created_at": datetime.datetime.utcnow()
        })
        
        print(f"Generated embedding for image {img_data['id']} from {img_data['collection']}")
        return True
    except Exception as e:
        print(f"Error processing image {img_data['url']}: {str(e)}")
        return False

async def process_new_s3_object(key, bucket):
    """Process a new image added to S3"""
    try:
        # Generate S3 URL
        image_url = f"https://{bucket}.s3.{REGION}.amazonaws.com/{key}"
        
        # Check if this image exists in our collections
        post = posts_collection.find_one({"imgKey": key})
        if post:
            img_data = {
                "id": str(post["_id"]),
                "url": image_url,
                "collection": "posts"
            }
            await generate_embedding_for_image(img_data)
            return True
            
        comment = comments_collection.find_one({"mediaAttachments.fileUrl": image_url})
        if comment:
            for attachment in comment["mediaAttachments"]:
                if attachment["fileUrl"] == image_url:
                    img_data = {
                        "id": str(attachment["_id"]),
                        "url": image_url,
                        "collection": "comments"
                    }
                    await generate_embedding_for_image(img_data)
                    return True
            
        reply = replies_collection.find_one({"mediaAttachments.fileUrl": image_url})
        if reply:
            for attachment in reply["mediaAttachments"]:
                if attachment["fileUrl"] == image_url:
                    img_data = {
                        "id": str(attachment["_id"]),
                        "url": image_url,
                        "collection": "replies"
                    }
                    await generate_embedding_for_image(img_data)
                    return True
            
        # If image not found in any collection, it might be new and not yet linked
        # Store the embedding with a temporary ID
        image = await download_image_from_s3(image_url)
        if image:
            embedding = model.encode(image)
            embeddings_collection.insert_one({
                "image_id": f"temp_{key.replace('/', '_')}",
                "url": image_url,
                "collection": "pending",
                "embedding": embedding.tolist(),
                "created_at": datetime.datetime.utcnow()
            })
            print(f"Stored embedding for new image {key}")
            return True
            
        return False
    except Exception as e:
        print(f"Error processing new S3 object {key}: {str(e)}")
        return False

@app.post("/webhooks/s3-event")
async def process_s3_event(event: dict):
    """Webhook endpoint to process S3 events for new images"""
    try:
        for record in event.get("Records", []):
            if record.get("eventName", "").startswith("ObjectCreated:"):
                bucket = record.get("s3", {}).get("bucket", {}).get("name")
                key = record.get("s3", {}).get("object", {}).get("key")
                
                if bucket and key:
                    success = await process_new_s3_object(key, bucket)
                    if success:
                        print(f"Successfully processed new image: {key}")
        
        return {"message": f"Processed {len(event.get('Records', []))} S3 events"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing S3 event: {str(e)}")

@app.post("/index/build")
async def build_index():
    """Build embeddings for all images in MongoDB that don't have embeddings yet"""
    try:
        # Extract images from MongoDB
        image_data = extract_images_from_mongodb()
        
        # Generate embeddings for each image
        success_count = 0
        for img_data in image_data:
            if await generate_embedding_for_image(img_data):
                success_count += 1
        
        return {"message": f"Generated embeddings for {success_count} images"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building index: {str(e)}")

@app.post("/index/update-single")
async def update_single_image(image_id: str, collection: str = "posts"):
    """Generate embedding for a single image by ID"""
    try:
        # Find the image
        if collection == "posts":
            doc = posts_collection.find_one({"_id": ObjectId(image_id)})
        elif collection == "comments":
            doc = comments_collection.find_one({"_id": ObjectId(image_id)})
        elif collection == "replies":
            doc = replies_collection.find_one({"_id": ObjectId(image_id)})
        else:
            raise HTTPException(status_code=400, detail="Invalid collection")
        
        if not doc or (collection == "posts" and "imgKey" not in doc):
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Generate embedding
        if collection == "posts":
            img_data = {
                "id": image_id,
                "url": f"https://{BUCKET}.s3.{REGION}.amazonaws.com/{doc['imgKey']}",
                "collection": collection
            }
        else:
            for attachment in doc["mediaAttachments"]:
                if str(attachment["_id"]) == image_id:
                    img_data = {
                        "id": image_id,
                        "url": attachment["fileUrl"],
                        "collection": collection
                    }
                    break
            else:
                raise HTTPException(status_code=404, detail="Image not found")
        
        if await generate_embedding_for_image(img_data):
            return {"message": f"Embedding generated for image {image_id}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate embedding")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

async def search_similar_images(query_features, limit=5):
    """Search for similar images using vector similarity"""
    # Find similar images
    results = []
    
    # Convert to list for MongoDB query
    query_vector = query_features.tolist()
    
    # Use MongoDB aggregation to find similar vectors
    pipeline = [
        {
            "$addFields": {
                "similarity": {
                    "$reduce": {
                        "input": {"$zip": {"inputs": ["$embedding", query_vector]}},
                        "initialValue": 0,
                        "in": {"$add": ["$$value", {"$multiply": [{"$arrayElemAt": ["$$this", 0]}, {"$arrayElemAt": ["$$this", 1]}]}]}
                    }
                }
            }
        },
        {"$sort": {"similarity": -1}},
        {"$limit": limit}
    ]
    
    similar_docs = list(embeddings_collection.aggregate(pipeline))
    
    # Process results
    for doc in similar_docs:
        image_id = doc["image_id"]
        collection_name = doc["collection"]
        
        # Skip temporary or pending images
        if image_id.startswith("temp_") or collection_name == "pending":
            continue
        
        # Get metadata
        metadata = await get_image_metadata(image_id)
        
        if metadata:
            results.append(SearchResult(
                image_id=image_id,
                image_url=metadata.get("image_url", doc["url"]),
                similarity_score=float(doc["similarity"]),
                metadata=metadata
            ))
    
    return results

@app.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    limit: int = Query(5, description="Number of results to return")
):
    """Search images by uploading an image"""
    try:
        # Read uploaded image
        image_data = await file.read()
        query_image = Image.open(io.BytesIO(image_data))
        
        # Encode query image
        query_features = model.encode(query_image)
        
        # Search in MongoDB
        results = await search_similar_images(query_features, limit)
        
        return SearchResponse(results=results, query=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

@app.get("/search/text")
async def search_by_text(
    query: str = Query(..., description="Text query to search for"),
    limit: int = Query(5, description="Number of results to return")
):
    """Search images by text query"""
    try:
        # Encode query
        query_features = model.encode(query)
        
        # Search
        results = await search_similar_images(query_features, limit)
        
        return SearchResponse(results=results, query=query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

async def get_image_metadata(image_id):
    """Get metadata for an image"""
    # First check in embeddings collection to get the source collection
    embedding_doc = embeddings_collection.find_one({"image_id": image_id})
    
    if not embedding_doc:
        return None
    
    collection_name = embedding_doc.get("collection")
    parent_id = embedding_doc.get("parent_id")
    
    if collection_name == "posts":
        # Search in posts
        post = posts_collection.find_one({"_id": ObjectId(image_id)})
        if post:
            return {
                "collection": "posts", 
                "data": post,
                "image_url": f"https://{BUCKET}.s3.{REGION}.amazonaws.com/{post.get('imgKey', '')}"
            }
    
    elif collection_name == "comments":
        # For comments, we need to find the parent comment
        if parent_id:
            comment = comments_collection.find_one({"_id": ObjectId(parent_id)})
            if comment:
                # Find the specific attachment
                for attachment in comment.get("mediaAttachments", []):
                    if str(attachment.get("_id", "")) == image_id:
                        return {
                            "collection": "comments", 
                            "data": comment,
                            "image_url": attachment.get("fileUrl", "")
                        }
    
    elif collection_name == "replies":
        # For replies, we need to find the parent reply
        if parent_id:
            reply = replies_collection.find_one({"_id": ObjectId(parent_id)})
            if reply:
                # Find the specific attachment
                for attachment in reply.get("mediaAttachments", []):
                    if str(attachment.get("_id", "")) == image_id:
                        return {
                            "collection": "replies", 
                            "data": reply,
                            "image_url": attachment.get("fileUrl", "")
                        }
    
    return None

@app.get("/check-sqs-messages")
async def check_sqs_messages(
    queue_url: str = Query(..., description="SQS Queue URL for S3 events")
):
    """Check and process messages from SQS queue (for S3 events)"""
    try:
        response = sqs_client.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=5
        )
        
        messages = response.get('Messages', [])
        processed_count = 0
        
        for message in messages:
            body = json.loads(message['Body'])
            # If S3 event notification is wrapped in SNS message
            if 'Message' in body:
                body = json.loads(body['Message'])
            
            # Process the S3 event
            for record in body.get('Records', []):
                if record.get('eventSource') == 'aws:s3' and record.get('eventName', '').startswith('ObjectCreated:'):
                    bucket = record.get('s3', {}).get('bucket', {}).get('name')
                    key = record.get('s3', {}).get('object', {}).get('key')
                    
                    if bucket and key:
                        await process_new_s3_object(key, bucket)
                        processed_count += 1
            
            # Delete the message
            sqs_client.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=message['ReceiptHandle']
            )
        
        return {"message": f"Processed {processed_count} new images from SQS queue"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing SQS messages: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("searchengine:app", host="0.0.0.0", port=8000, reload=True)