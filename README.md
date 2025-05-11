# InnoTSR: Real-Time Traffic Sign Recognition System with Voice Feedback

**InnoTSR** project is a Python-based real-time Traffic Sign Recognition (TSR) system using deep learning with voice feedback providing real-time information to users for personalized alert systems.


<br/>
<p align="center">
<img src="docs/img/img1a.png" style="width:35%; height:auto;">&emsp;
</p>


## Key Features
- **Hign Accuracy**: Utilizes cutting-edge deep learning algorithms for accurate traffic sign detection and recognition.
- **Voice Command Alerts**: Built-in instant voice alert to notify the user of detected traffic sign information in real-time.
- **Enhanced User Experience**: Provides a seamless and intuitive graphical user interface experience.


## Installation

**Requirements**:
- Python 3.11 or higher
- Tensorflow 2.18 or higher
- SSD MobileNet V2 model
- OpenCV for video capturing and processing
- SpeechRecognition for voice command processing

To install this project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/your-repository.git
   ```

2. Navigate to the project directory:
   ```sh
   cd your-repository
   ```

3. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

To use the traffic sign recognition system, follow these instructions:
1. **Prepare your environment**: Ensure your system has a camera or input method to capture traffic signs.
2. **Start the application**: Run the main script to initiate the traffic sign recognition system.
3. **Voice Command Alerts**: The system will provide real-time voice alerts for detected traffic signs.


Run the following commands:
1. Train the model: Ensure you have a dataset of traffic sign images in the `dataset/raw_data`.
   ```bash
   python scripts/train.py
   ```

<!-- 2. **Run the detection script**:
   ```bash
   python scripts/detect.py
   ``` -->

<!-- 2. **Run the user interface:**:
   ```bash
   python ui/main.py
   ``` -->

2. Run the application:
   ```sh
   python main.py
   ```

<!--
## Project Structure

The repository contains the following main files and folders:

- `main.py`: The main script to run the application.
- `requirements.txt`: A list of dependencies required to run the project.
- `models/`: Directory containing pre-trained and saved models for traffic sign recognition.
- `dataset/`: Directory for storing training and test datasets.
- `scripts/`: Utility functions and helper scripts.
- `utils/`: Utility functions and helper scripts. 
-->

## Contributing

For contributions, please follow these steps:
For contributions, please follow the steps below:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions, please contact us at: [OCR-tech](https://github.com/OCR-tech).