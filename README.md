### \# Hierarchical Multi-Label Text Classification



#### \*\*Final Project for DATA304: Big Data Analysis\*\*



###### This project implements a Hierarchical Multi-Label Text Classification system for Amazon product reviews. 

###### It utilizes \*\*Sentence-BERT (SBERT)\*\* for semantic embedding and applies \*\*DAG (Directed Acyclic Graph)\*\* constraints to ensure topological consistency in predictions.



##### 

##### 📂 Project Structure



\- `sbert.py`: The main executable script containing the entire pipeline (Data Loading, Model, Prediction).

\- `requirements.txt`: List of Python dependencies.

\- `submission.csv`: The final output file generated after running the script.



##### 🛠 Prerequisites



\- Python 3.8+

\- PyTorch (CUDA support recommended for faster inference)



##### &nbsp;📦 Installation



1\. Clone this repository:

&nbsp;  ```bash

&nbsp;  git clone \[https://github.com/sangmogh/20252R0136DATA30400.git](https://github.com/sangmogh/20252R0136DATA30400.git)

&nbsp;  cd 20252R0136DATA30400


2. Install the required packages:

&nbsp;  pip install -r requirements.txt





##### &nbsp;🚀 How to Run



1\. Prepare Data: ensure the following dataset files are placed in the root directory of the project:

&nbsp;  classes.txt
   class\_hierarchy.txt

&nbsp;  class\_related\_keywords.txt

&nbsp;  test\_corpus.txt



2\. Execute the Script: Run the main python script. It will automatically detect the device (CPU/GPU) and process the data.

&nbsp;   python sbert.py



3\. Check Output: After the process completes, the prediction results will be saved as submission.csv.

&nbsp;   Format: id,labels

&nbsp;   Example: 0,10,25,30 (IDs are comma-separated)



##### 🧠 Method



This approach combines semantic search with graph-theoretic post-processing.



###### 1\. Semantic Embedding (Neural Part)

&nbsp;    Model: We used the `all-MiniLM-L6-v2` pre-trained model from Sentence-Transformers. It is chosen for its optimal balance between performance and inference speed.
&nbsp;    Label Representation: We embed the Class Names directly into the vector space.

&nbsp;    *Note:* While the report analyzes the potential impact of keyword enrichment, this submission script implements the streamlined Class-Name based inference for maximum efficiency.

&nbsp;    Review Representation: The input product reviews (Product Name + Review Text) are embedded into the same vector space.

&nbsp;    Scoring: We calculate the Cosine Similarity between the review vector and all class vectors.



###### 2\. Hierarchical Constraint (Symbolic Part)

To respect the taxonomy structure (DAG), we applied a Bottom-Up Inference strategy with the True Path Rule:

&nbsp;  1. Anchor Selection: Select the class with the highest cosine similarity score as the 'Anchor'.

&nbsp;  2. True Path Rule: If an anchor is selected, all its ancestor nodes (parent, grandparent, etc.) are mandatory included in the prediction set.

&nbsp;  3. Expansion: If the prediction set has fewer than 3 labels, we select the next best candidate anchor and repeat the process.

&nbsp;  4. Truncation (Pruning): If the prediction set exceeds 3 labels:
        We iteratively remove nodes to strictly maintain a maximum of 3 labels.
        Removal Criteria: We remove Leaf Nodes (nodes with no children in the current set) that have the lowest original similarity score. 
                                             This ensures the hierarchical structure remains unbroken while keeping the most semantically relevant nodes.



##### 📝 Reproducibility



To ensure reproducibility, random seeds for numpy, torch, and python are strictly fixed to 42 at the beginning of the execution.

