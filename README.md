


# Image Search API

A powerful image search API that uses CLIP embeddings to enable semantic search across images in MongoDB collections. This API allows searching by text queries or by uploading an image to find similar images.

## Features

- **Text-to-Image Search**: Find images that match a text description
- **Image-to-Image Search**: Upload an image to find similar images
- **Auto-indexing**: Automatically indexes new images added to MongoDB or S3
- **MongoDB Integration**: Works with existing MongoDB collections
- **S3 Storage**: Retrieves images from Amazon S3

## Setup

### Prerequisites

- Python 3.8+
- MongoDB database with collections for posts, comments, and forum replies
- Amazon S3 bucket for image storage
- AWS credentials with access to S3

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
MONGODB_URL="your-mongodb-connection-string"
S3_ACCESS_KEY="your-aws-access-key"
S3_SECRET_ACCESS_KEY="your-aws-secret-key"
BUCKET="your-s3-bucket-name"
REGION="your-aws-region"
```

### Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python searchengine.py
```

The API will start on `http://0.0.0.0:8000`.

## API Endpoints

### Build Index

Build or rebuild the CLIP embeddings index for all images in the database.

```
POST /index/build
```

**Response:**

```json
{
  "message": "Index built successfully with X images"
}
```

### Generate Embedding for a Specific Image

Generate an embedding for a specific image by ID and collection.

```
POST /embedding/generate/{collection}/{image_id}
```

**Path Parameters:**
- `collection`: Collection name (posts, comments, forumreplies)
- `image_id`: MongoDB ID of the document containing the image

**Response:**

```json
{
  "message": "Embedding generated for image {image_id}"
}
```

### Text Search

Search for images using a text query.

```
GET /search/text?query={text_query}&limit={limit}
```

**Query Parameters:**
- `query`: Text description to search for
- `limit` (optional): Number of results to return (default: 5)

**Response:**

```json
{
  "results": [
    {
      "image_id": "image_id",
      "image_url": "https://bucket.s3.region.amazonaws.com/image_key",
      "similarity_score": 0.85,
      "metadata": {
        "collection": "collection_name",
        "data": { /* document data */ }
      }
    }
  ],
  "query": "your search query"
}
```

### Image Search

Search for similar images by uploading an image.

```
POST /search/image?limit={limit}
```

**Query Parameters:**
- `limit` (optional): Number of results to return (default: 5)

**Request Body:**
- Form data with a file field named "file" containing the image to search for

**Response:**

```json
{
  "results": [
    {
      "image_id": "image_id",
      "image_url": "https://bucket.s3.region.amazonaws.com/image_key",
      "similarity_score": 0.85,
      "metadata": {
        "collection": "collection_name",
        "data": { /* document data */ }
      }
    }
  ],
  "query": "uploaded_image_filename"
}
```

### Check SQS Messages

Process new images from SQS queue (if configured).

```
GET /sqs/check
```

**Response:**

```json
{
  "message": "Processed X new images from SQS queue"
}
```

## Data Structure

The API works with the following MongoDB collections:

### userposts

Posts with images stored in S3:

```json
{
  "_id": "ObjectId",
  "userId": "user_id",
  "desc": "post description",
  "imgKey": "s3_image_key",
  "likes": [],
  "dislikes": []
}
```

### comments

Comments with attached images:

```json
{
  "_id": "ObjectId",
  "postId": "post_id",
  "depth": 1,
  "parentId": null,
  "author": {},
  "commentText": "comment text",
  "mediaAttachments": [
    {
      "fileName": "file_name",
      "fileType": "image/png",
      "fileUrl": "https://bucket.s3.region.amazonaws.com/file_key",
      "fileSize": 123456,
      "uploadedAt": "timestamp",
      "_id": "attachment_id"
    }
  ],
  "likes": [],
  "dislikes": []
}
```

### forumreplies

Forum replies with attached images:

```json
{
  "_id": "ObjectId",
  "content": "reply content",
  "topicId": "topic_id",
  "userId": "user_id",
  "userName": "user_name",
  "isAnswer": false,
  "parentReplyId": null,
  "mediaAttachments": [
    {
      "fileName": "file_name",
      "fileType": "image/jpeg",
      "fileUrl": "https://bucket.s3.region.amazonaws.com/file_key",
      "fileSize": 0,
      "uploadedAt": "timestamp",
      "_id": "attachment_id"
    }
  ],
  "likes": [],
  "dislikes": []
}
```

## How It Works

1. The API extracts images from MongoDB collections (userposts, comments, forumreplies)
2. It downloads images from S3 and generates CLIP embeddings
3. Embeddings are stored in the `image_embeddings` collection in MongoDB
4. When searching, the API converts the query (text or image) to a CLIP embedding
5. It finds the most similar embeddings using vector similarity
6. Results are returned with metadata from the original documents

## Troubleshooting

- If the index is not building, check your MongoDB and S3 credentials
- Make sure the collections exist in your MongoDB database
- Verify that images are accessible in your S3 bucket
- Check the console logs for any error messages during startup
