# InnoTSR: Real-Time Traffic Sign Recognition System with Voice Feedback 

**InnoTSR** project is a Python-based real-time Traffic Sign Recognition (TSR) system with voice feedback control using deep learning providing real-time information to users for traffic monitoring and management systems.


## Key Features
- **Hign Accuracy**: Utilizes cutting-edge deep learning algorithms for accurate traffic sign detection and recognition.
- **Voice Command Alerts**: Built-in instant voice alert to notify the user of detected traffic sign information in real-time.
- **Enhanced User Experience**: Provides a seamless and intuitive graphical user interface experience.


## Installation

**Requirements**: 
- Python 3.11 or higher

**Prerequisition**: 
- install the necessary Python packages:

```bash
pip install -r requirements.txt
``` 

To install and run the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/OCR-tech/InnoTSR.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd InnoTSR
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

## Usage

To use the traffic sign recognition system, follow these instructions:
1. **Prepare your environment**: Ensure your system has a camera or input method to capture traffic signs.
2. **Start the application**: Run the main script to initiate the traffic sign recognition system.
3. **Voice Command Alerts**: The system will provide real-time voice alerts for detected traffic signs.


for usages, run the following commands:
1. **Train the model**: Ensure you have a dataset of traffic sign images in the data/raw directory.
   ```bash
   python scripts/train.py
   ```

2. **Run the dection script**:
   ```bash
   python scripts/detect.py
   ```

3. **Run the user interface:**:
   ```bash
   python ui/main.py
   ```



## Project Structure

The repository contains the following main files and folders:

- `main.py`: The main script to run the application.
- `requirements.txt`: A list of dependencies required to run the project.
- `models/`: Directory containing pre-trained models for traffic sign recognition.
- `data/`: Directory for storing training and test datasets.
- `utils/`: Utility functions and helper scripts.

## Contributing

We welcome contributions to enhance the InnoTSR project. To contribute, please follow these steps:

1. **Fork the repository**.
2. **Create a new branch**:
   ```bash
   git checkout -b feature-branch
   ```
3. **Make your changes and commit them**:
   ```bash
   git commit -m "Description of changes"
   ```
4. **Push to the branch**:
   ```bash
   git push origin feature-branch
   ```
5. **Create a pull request**.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact the project owner at [OCR-tech](https://github.com/OCR-tech).