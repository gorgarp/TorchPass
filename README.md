
# TorchPass

TorchPass is a password generation program that leverages modern deep learning techniques to generate "human-like" password lists. While inspired by the concepts introduced in PassGAN, TorchPass is a complete rewrite that incorporates current best practices in deep learning and natural language processing using PyTorch.

It now only takes ~10 hours to run 100 epochs against the Rockyou dataset with a single NVIDIA 3070. This is a dramatic improvement over PassGAN.

Update 08/02/2024: Added support for multiple GPU hosts.
Update 08/04/2024 Added batching for generation to speed up output.

## Features

### 1. **Device Agnostic**
   - Automatically detects and utilizes CUDA-enabled GPUs for accelerated training and generation. Falls back to CPU if GPU is unavailable.
   - Supports multi-GPU training using `nn.DataParallel` for faster computation.

### 2. **Custom Dataset Handling**
   - Supports custom datasets through a user-defined list of passwords. The dataset is preprocessed to ensure compatibility with the model, including character-to-index mapping and padding.

### 3. **Neural Network Model**
   - Utilizes an LSTM-based architecture with embedding layers, layer normalization, and dropout for regularization.
   - Supports mixed precision training for faster computations on compatible hardware.

### 4. **Training Features**
   - Implements gradient clipping and learning rate scheduling to stabilize training and prevent overfitting.
   - Includes early stopping based on validation loss to avoid overfitting and save computational resources.
   - Automatically saves the best model based on validation performance.

### 5. **Password Generation**
   - Generates passwords of varying lengths (configurable in code) using temperature scaling to control the randomness and diversity of the generated outputs.
   - Can generate a specified number of passwords in a single run, making it easy to generate large datasets.

### 6. **Multi-Process Data Loading**
   - Supports multi-process data loading for faster training, especially on large datasets, by utilizing multiple CPU cores.

## Installation

To install TorchPass, ensure you have Python 3.7+ and PyTorch installed. Then, clone the repository and install the required dependencies. For optimal performance, it is recommended to have CUDA installed for GPU acceleration.

```bash
# Clone the repository
git clone https://github.com/yourusername/torchpass.git

# Navigate to the TorchPass directory
cd torchpass

# Install required dependencies
pip install -r requirements.txt
```

Note: If you intend to use a GPU for training and generation, ensure that you have CUDA installed and configured properly on your system.

## Usage

TorchPass can operate in two main modes: training (`train`) and generation (`generate`). The functionality can be controlled using command-line arguments.

### Training Mode

To train the model, use the following command:

```bash
python torchpass.py --mode train --input /path/to/passwords.txt --model model.pth --epochs 50 --batch 256 --workers 4
```

#### Arguments

- `--mode train`: (Required) Specifies that the program should run in training mode.
- `--input`: (Required) Specifies the path to the input file containing the passwords for training.
- `--model`: (Optional, default: `'model.pth'`) Path to save or load the model. If the file exists, the model will continue training from the saved state.
- `--epochs`: (Optional, default: `50`) Specifies the number of training epochs.
- `--batch`: (Optional, default: `256`) Specifies the batch size for training.
- `--worker`: (Optional, default: `4`) Specifies the number of worker processes for data loading.

### Generation Mode

To generate passwords using a pre-trained model, use the following command:

```bash
python torchpass.py --mode generate --model model.pth --output generated_passwords.txt --num_pass 100 --temp 1.0
```

#### Arguments

- `--mode generate`: (Required) Specifies that the program should run in generation mode.
- `--model model.pth`: (Required) Path to the trained model file.
- `--output`: (Required) Specifies the output file to save the generated passwords.
- `--num_pass`: (Optional, default: `100`) Specifies the number of passwords to generate.
- `--temp`: (Optional, default: `1.0`) Specifies the temperature for generation, controlling the randomness. Lower values make the output more deterministic, while higher values increase randomness.

### Example Commands

- **Training a new model:**
  ```bash
  python torchpass.py --mode train --input my_passwords.txt --model new_model.pth --epochs 30 --batch 128 --workers 2
  ```

- **Generating passwords:**
  ```bash
  python torchpass.py --mode generate --model best_model.pth --output my_generated_passwords.txt --num_pass 200 --temp 0.8
  ```

## Notes
At this time, I've not cracked how to memory map the dataset beyond the RAM limit. The issue sits with degredation of performance when reading from SSD instead of RAM. For the time being, split the dataset to fit into your RAM.

NPUs will not be supported in the near future, as ```aten::_thnn_fused_lstm_cell``` is not supported by DirectML at this time. If that changes, I will revisit.

## Contributing
Feel welcome to submit pull requests! This was started as a mental exercise so I am very open to improvements!

## License

TorchPass is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

TorchPass is inspired by PassGAN and other works in the field of password generation using neural networks. Special thanks to the PyTorch community for providing a powerful and flexible deep learning framework.

## Legal Disclaimer

This program is intended for educational and security research purposes only. The use of this software for illegal activities, including unauthorized access to computer systems, is strictly prohibited. The creators and contributors of TorchPass are not responsible for any misuse of the software.
