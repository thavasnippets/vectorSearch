import sqlite3
import numpy as np
from PIL import Image

def cosine_similarity(a, b):
    # Ensure that both vectors have the same length
    min_length = min(len(a), len(b))
    a = a[:min_length]
    b = b[:min_length] 

    # Calculate dot product and magnitudes 
    # Dot Product (Scalar Product) The dot product of two vectors is a scalar quantity obtained by
    # multiplying the corresponding components of the vectors and summing the results.
    # It provides a measure of the alignment between two vectors.
    
    # If you have two vectors A and B:        
    #     Vector A: [A₁, A₂, A₃]        
    #     Vector B: [B₁, B₂, B₃]        
    #     The dot product is calculated as:        
    #     A · B = A₁ \* B₁ + A₂ \* B₂ + A₃ \* B₃

    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    # Calculate cosine similarity
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    similarity = dot_product / (magnitude_a * magnitude_b)
    return similarity


def find_similar_vectors(query_vector, threshold, database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id,file_name,image_data FROM images")
    similar_vectors = []

    for row in cursor.fetchall():
        vector_blob = row[2]      
        # print(vector_blob )
        stored_vector = np.frombuffer(vector_blob, dtype=np.float32)
        
        similarity = cosine_similarity(query_vector, stored_vector)      
        if similarity >= threshold:
            similar_vectors.append((row[0],row[1],similarity))

    similar_vectors.sort(key=lambda x: x[2], reverse=True)
    conn.close()
    return similar_vectors


image = Image.open("./test_images/find.png")
query_vector = np.array(image,dtype=np.float32).flatten()
threshold = 0.9
database_path = 'image_database.db'
similar_vectors = find_similar_vectors(query_vector, threshold, database_path)

print("Similar Vectors:")
for vec_id, file_name,similarity in similar_vectors:
    print(f"Vector ID: {vec_id}, file_name: {file_name}, Similarity: {similarity}")