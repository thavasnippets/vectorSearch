import sqlite3
from PIL import Image
import io
import numpy as np
import os

def load_images():
    # Connect to the database or create a new one
    db_connection = sqlite3.connect("image_database.db")
    cursor = db_connection.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, file_name varchar(100), image_data BLOB)")
    folder_path = "./training_images/" 
    file_names = os.listdir(folder_path)

    # Read all the images and covert into  vector using numpy
    for file_name in file_names:       
        image_path = f"{folder_path}{file_name}" 
        image = Image.open(image_path)

        # Convert the image to grayscale
        gray_image = image.convert('L')
        # gray_image.save(f'gray_{file_name}')
       
        image_vector = np.array(gray_image,dtype=np.float32).flatten()
        cursor.execute("INSERT INTO images (file_name,image_data) VALUES (?,?)",(file_name,image_vector.tobytes()) )
        # break        

    db_connection.commit()
    db_connection.close()



load_images()