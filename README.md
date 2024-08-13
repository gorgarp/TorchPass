
# TorchPass

TorchPass is a password generation program leveraging advanced deep learning techniques to generate "human-like" password lists. Inspired by PassGAN, TorchPass is a complete rewrite that incorporates modern best practices in deep learning and natural language processing using PyTorch.

With a single NVIDIA 3070, TorchPass can now train for 100 epochs on the Rockyou dataset in approximately <8 hours, a significant improvement over PassGAN.

## Updates

- **08/13/2024:** Major update with numerous optimizations, including multiprocessing CUDA streams for generation and data loader memory tuning.
- **08/04/2024:** Added batching for generation and numerous performance upgrades. Uploaded a sample model trained on a custom dataset.
- **08/02/2024:** Added support for multiple GPU hosts.

## Features

### 1. **GPU-Optimized**
   - Automatically detects and utilizes CUDA-enabled GPUs for accelerated training and generation. No support for CPU-only execution to ensure maximum performance.
   - Multi-GPU support for training and generation, allowing faster computation across multiple devices.

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
   - Efficiently generates large datasets with batch processing and multi-GPU support.

### 6. **Multi-Process Data Loading**
   - Supports multi-process data loading for faster training, especially on large datasets, by utilizing multiple CPU cores.

## Installation

To install TorchPass, ensure you have Python 3.7+ and PyTorch installed. Then, clone the repository and install the required dependencies. For optimal performance, CUDA must be installed for GPU acceleration.

```bash
# Clone the repository
git clone https://github.com/yourusername/torchpass.git

# Navigate to the TorchPass directory
cd torchpass

# Install required dependencies
pip install -r requirements.txt
```

## Usage

TorchPass can operate in two main modes: training (`train`) and generation (`generate`). The functionality can be controlled using command-line arguments.

A sample model has been uploaded to `Sample/model.pth`. It was trained over many hours using a custom dataset.

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
- `--workers`: (Optional, default: `4`) Specifies the number of worker processes for data loading.

### Generation Mode

To generate passwords using a pre-trained model, use the following command:

```bash
python torchpass.py --mode generate --model model.pth --output generated_passwords.txt --num_pass 100 --temp 1.0 --workers 4
```

#### Arguments

- `--mode generate`: (Required) Specifies that the program should run in generation mode.
- `--model model.pth`: (Required) Path to the trained model file.
- `--output`: (Required) Specifies the output file to save the generated passwords.
- `--num_pass`: (Optional, default: `100`) Specifies the number of passwords to generate.
- `--temp`: (Optional, default: `1.0`) Specifies the temperature for generation, controlling the randomness. Lower values make the output more deterministic, while higher values increase randomness.
- `--workers`: (Optional, default: `4`) Specifies the number of worker processes for password generation.

### Example Commands

- **Training a new model:**
  ```bash
  python torchpass.py --mode train --input my_passwords.txt --model new_model.pth --epochs 30 --batch 128 --workers 2
  ```

- **Generating passwords:**
  ```bash
  python torchpass.py --mode generate --model best_model.pth --output my_generated_passwords.txt --num_pass 200 --temp 0.8 --workers 4
  ```

## Notes

- Dataset handling is optimized for systems with sufficient RAM. For large datasets exceeding available memory, performance may degrade when reading from SSD. Consider splitting the dataset to fit into your system's RAM.
- NPUs are currently unsupported due to incompatibility with the `aten::_thnn_fused_lstm_cell` operation in DirectML. This may be revisited if support is added in the future.

## Contributing

Feel free to submit pull requests! This project started as a mental exercise, and I'm open to improvements and new ideas!

## License

TorchPass is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

TorchPass is inspired by PassGAN and other works in the field of password generation using neural networks. Special thanks to the PyTorch community for providing a powerful and flexible deep learning framework.

## Legal Disclaimer

This program is intended for educational and security research purposes only. The use of this software for illegal activities, including unauthorized access to computer systems, is strictly prohibited. The creators and contributors of TorchPass are not responsible for any misuse of the software.
