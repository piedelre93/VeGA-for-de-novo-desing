# VeGA: Installation and Usage Guide

This document will guide you through installing and using VeGA, a new open-source framework for de novo molecular design. By following these steps, you will be able to pre-train, fine-tune, and train your own Transformer model.

## 1. Installation

First, download the codebase. Then, use conda to set up a new environment for VeGA. If you're new to conda, we recommend checking out [this tutorial](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) before proceeding.

```bash
conda env create -f enviroment.yml  
conda activate vega
python -m pip install tensorflow[and-cuda]  
conda install -c conda-forge jupyter notebook
```

Now, you can launch one of the three notebooks (e.g., `gen.ipynb`) using Jupyter Notebook.

## 2. Generation (gen.ipynb)

After opening `gen.ipynb`, go to:
**Kernel → Restart Kernel and Clear Output**

Before running the code, modify **Cell 1** by specifying the paths to the following files:
- `model_path` 
- `char2idx_path` 
- `idx2char_path` 
- `vocab_path` 

These files can be found in the repository under the `model_ChEMBLDB` and `model_COCONUT` folders.

- `max_length` corresponds to the maximum sequence length for generation and varies depending on the model. 
- `training_file` should point to the dataset used during training (required for novelty calculations). 

**Cell 2** contains the model architecture and can be executed without modifications.

**Cell 3** allows you to set generation parameters:
- `num_batches` (number of batches to generate) 
- `batch_size` (molecules generated per batch) 
- `temperature` (controls randomness in generation) 
- `out_csv_path` (output file for generated molecules) 

**Cell 4** starts the generation process. Upon completion, a report will display the following metrics:
- Total number generated 
- Validity 
- Average QED 
- Average Synthetic Accessibility (SA) 
- Novelty 
- Originality 

### 📝 Notes:

#### 1. Error Handling:
If running Cell 4 results in:
```
ValueError: Input 0 of layer "functional_4" is incompatible with the layer: expected shape=(None, 193), found shape=(32, 20)
```
Replace the `max_length` value with the one in the error message (e.g., 193).

#### 2. Model Options:
You can use the pre-trained models included in the repository:
- `model_ChEMBLDB` (pre-trained on ChEMBL) 
- `model_COCONUT` (trained on natural compounds)

Alternatively, you can train or fine-tune your own model by following the instructions in the next sections.

## 3. Training from Scratch (train_new_model.ipynb)

### Cell 1: GPU Availability Check
After opening `train_new_model.ipynb`, go to:
**Kernel → Restart Kernel and Clear Output**

Running this cell tests whether a GPU is available on your workstation.

### Cell 2: Training Settings
This cell contains key parameters for managing the training process. We recommend keeping the default settings unless adjustments are necessary. For detailed explanations of each parameter, refer to the reference publication.

**User modifications:**
- `SMILES_FILE`: Specify the path to your training dataset. 
- `Epochs`: You may reduce the number of training epochs if needed. 

### Cell 3: Import Required Libraries
This cell loads all necessary Python libraries. No modifications are required.

### Cell 4: Transformer Architecture
This cell contains the core Transformer model structure. Execute as-is; no changes needed.

### Cell 5: SMILES Preprocessing
This cell processes the SMILES strings from the training set, performing the following steps:
- Validates and corrects SMILES strings. 
- Filters out invalid SMILES. 
- Canonicalizes valid SMILES. 
- Generates the character vocabulary. 
- Determines the maximum sequence length. 
- Saves mapping dictionaries in the Jupyter notebook's working directory: 
  - `char2idx.pkl` (character-to-index mapping) 
  - `idx2char.pkl` (index-to-character mapping) 
  - `vocab.json` (vocabulary) 
- Splits the dataset into training (80%) and validation (20%) sets by default. 

### Cell 6: Start Training
This cell launches the training process. You can monitor:
- Training progress 
- Training & validation loss 

The model is automatically saved whenever the loss improves. You can stop training once the metrics meet your expectations.

### Cell 7: Model Evaluation
This cell assesses the model's learning by generating molecules and checking their validity. For a more comprehensive evaluation, use the `gen.ipynb` notebook (as described in the previous section).

## 4. Fine-Tuning (Fine_tuning.ipynb)

This notebook allows you to fine-tune a pre-trained Transformer model on your own dataset. Below is a step-by-step guide for configuring and running the fine-tuning process.

### 🔹 Cell 1: Set File Paths
Before executing this cell, update the following file paths according to your local setup:

- `model_path` Path to the pre-trained base model to start fine-tuning from 
- `char2idx_path` Path to the character-to-index mapping pickle file 
- `idx2char_path` Path to the index-to-character mapping pickle file 
- `vocab_path` Path to the vocabulary JSON file 
- `model_out_path` Destination path where the fine-tuned model will be saved 

✅ **Make sure all these files are correctly set before running the cell, otherwise the training may fail.**

### 🔹 Cell 2: Training Configuration
This cell defines the main hyperparameters and settings used during training.

📘 **Recommendation:**
Leave the parameters with their default values unless you have a specific reason to change them.
For detailed descriptions of each parameter, refer to the VeGA research publication.

🛠️ **User-defined modifications:**

- `SMILES_FILE` Path to the .smi or .txt file containing your custom training dataset |
- `Epochs` Number of training epochs. You can reduce this for quicker testing |

### 🔹 Cell 3: Transformer Architecture
This cell defines the core architecture of the Transformer model.

Simply execute the cell — no modifications are needed.
This loads the model structure and prepares it for continued training using your dataset and pre-trained weights.

### 🔹 Cell 4: Start Fine-Tuning
This cell initiates the fine-tuning process. While training progresses, you will be able to monitor:
- Training loss 
- Validation loss 

The model will automatically be saved to disk whenever a new minimum validation loss is achieved — this ensures you won't lose the best checkpoint.

🛑 **You can safely stop training at any time if the performance metrics meet your expectations.**

### 📝 Additional Notes
- The fine-tuned model will be saved at the location specified in `model_out_path`. You can later use it for generation or further training via the appropriate notebooks (`gen.ipynb`, etc.). 
- Ensure that the `char2idx`, `idx2char`, and `vocab` files match the tokenizer used when the base model was originally trained. Mismatches may cause errors or degrade performance.



## 📚 Citation
If you use VeGA in your research, please cite:
```bibtex
@article{xx,
  title={xx},
  author={xx},
  year={xx}
}
```


## 📞 Contact
For questions or issues, please contact: 
```
pietro.delre@unina.it
```
