## Getting Started

To replicate the experiments, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/xuanxhe/T2IEval
   cd T2IEval
   ```

2. **Create Environment and Install Dependencies**:

   ```bash
   conda create -n T2IEval python=3.10
   pip install -r requirements.txt
   ```

3. **Download the Dataset and Preprocess the Data**:

   ```bash
   # Download the dataset from Huggingface
   sh scripts/download.sh
   
   # Average the annotation scores and calculate the variance of the alignment scores for different image-text pairs corresponding to the same prompt
   python3 process/process_train.py
   
   # Corresponds the element split from the prompt to a specific index in the prompt
   python3 process/element2mask.py
   
   
   ```

4. **Run the Training Scripts**:

   ```bash
   sh scripts/train.sh
   ```

5. **Evaluate the Models**:

   ```bash
   sh scripts/eval.sh
   ```

6. **Download Weights and Test Sets**:

   You can download the pre-trained model weights from[T2IEval](https://huggingface.co/xuanxhe/T2IEval/tree/main)and Test Sets from[Testsets](https://huggingface.co/datasets/xuanxhe/T2IEval_Testsets/tree/main)

   Create a new folder "checkpoints" under the home directory and place the model weights in that folder.

   Unzip the test set in the home directory

7. **Test the Models**:

   ```bash
   sh scripts/test.sh
   ```

