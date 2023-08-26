Let's start from the basics

**Vector:** Quantity with a _magnitude_ and a _direction_

Magnitude is the size of a mathematical object, a property which determines whether the object is larger or smaller than other objects of the same kind.

**Magnitude** : It's the size of an object, a property which determines whether the object is larger or smaller than other objects of the same kind.

**Direction** : Vectors have a specific direction in space. This direction can be represented by an angle with respect to a reference axis or by specifying the coordinates of the vector's endpoint in a coordinate system.

Representation: Vectors can be represented in various ways,

- Geometrically using arrows
- Using ordered sets of numbers
- Using column matrices

**Vector Arithmetic Operations:**

- **Addition and Subtraction:** Vectors can be added or subtracted component-wise.

    Let's say you have three vectors A, B, and C:
    
    Vector A: [A₁, A₂, A₃]
    
    Vector B: [B₁, B₂, B₃]
    
    Vector C: [C₁, C₂, C₃]
    
    To add these vectors together, you add their corresponding components:
    
    Resultant Vector R: [A₁ + B₁ + C₁, A₂ + B₂ + C₂, A₃ + B₃ + C₃]
    
    Similarly for the subtraction

- **Multiplication:**

    1. **Scalar Multiplication:** Vectors can be multiplied by scalars (real numbers), resulting in a vector with the same direction but a scaled magnitude. Scalar multiplication affects the length of the vector but not its direction.
        
          If you have a vector A and a scalar k:
          
          Vector A: [A₁, A₂, A₃]
          
          Scalar k
          
          The scalar multiplication result is:
          
          k \* A = [k \* A₁, k \* A₂, k \* A₃]
    
    1. **Dot Product (Scalar Product):** The dot product of two vectors is a scalar quantity obtained by multiplying the corresponding components of the vectors and summing the results. It provides a measure of the alignment between two vectors.
    
        If you have two vectors A and B:
        
        Vector A: [A₁, A₂, A₃]
        
        Vector B: [B₁, B₂, B₃]
        
        The dot product is calculated as:
        
        A · B = A₁ \* B₁ + A₂ \* B₂ + A₃ \* B₃
    
    1. **Cross Product (Vector Product):** The cross product of two vectors results in a third vector that is perpendicular to both input vectors. (to Find the momentum)
    
        If you have two vectors A and B:
        
        Vector A: [A₁, A₂, A₃]
        
        Vector B: [B₁, B₂, B₃]
        
        The cross product is calculated as:
        
        A × B = [A₂ \* B₃ - A₃ \* B₂, A₃ \* B₁ - A₁ \* B₃, A₁ \* B₂ - A₂ \* B₁]
    
    1. **Unit Vector:** A unit vector is a vector with a magnitude of 1 that points in a specific direction. It is commonly used to describe directions in space.
       

Now I am sure you might have fair re-collection or understanding of vectors

A **vector database** is a type of database that stores and retrieves data using vectors as the fundamental data representation (Data points in a multi-dimensional space).

Each dimension in the space corresponds to a specific feature or attribute of the data.

Vector databases are particularly useful for working with high-dimensional data, such as images, audio, text, and other complex data types.

In this, each data item is represented as a vector of numerical values, where each value corresponds to a feature of the data.

These vectors can then be indexed and queried for various purposes, such as similarity search, clustering, classification, and recommendation systems.

This makes them suitable for applications such as:

- Content-based Retrieval: image and video search
- Natural Language Processing (NLP): Find similar words or sentences based on their semantic meanings.
- Recommendation Systems: For efficient recommendation of items based on similarity to a user's preferences.
- Anomaly Detection: By identifying deviations from normal patterns.
- Vector databases are engineered to perform high-speed similarity searches in massive datasets

  # Demo
  Lets create application for images matching
  
  **Steps**
  - Data preparation 
      1.  Read all the images and covert into  vector using numpy
      2.  Save the flatten array in the database
  - Search by Image
      1. Read the Image and convert it into array
      2. Find the matching using dot product and cosine similarity
   
    
