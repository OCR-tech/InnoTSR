# InnoTSR: Real-Time Traffic Sign Recognition System with Voice Feedback

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-blue)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=OCR-tech.InnoTSR)
<!-- ![GitHub repo size](https://img.shields.io/github/repo-size/OCR-tech/InnoTSR)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.18%2B-blue)
![GitHub last commit](https://img.shields.io/github/last-commit/OCR-tech/InnoTSR)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/OCR-tech/InnoTSR)
![GitHub contributors](https://img.shields.io/github/contributors-anon/OCR-tech/InnoTSR) -->

**InnoTSR** project is a Python-based real-time Traffic Sign Recognition (TSR) system using deep learning with voice feedback providing real-time information to users for personalized alert systems.

<br/>
<p align="center">
<img src="docs/img/img1a.png" style="width:35%; height:auto;">&emsp;
</p>

## Key Features

- **High Accuracy**: Utilizes cutting-edge deep learning algorithms for accurate traffic sign detection and recognition.
- **Voice Command and Alerts**: Built-in instant voice command and alert to notify the user of detected traffic sign information in real-time.
- **Enhanced User Experience**: Provides a seamless and intuitive graphical user interface experience.

## Installation

Requirements:

- Python 3.11 or higher
- OpenCV for video capturing and processing
- SpeechRecognition for voice command processing
  <!-- - TensorFlow 2.18 or higher -->
  <!-- - SSD MobileNet V2 model -->

To install this project, please follow these steps:

1. Clone the repository:

   ```sh
   git clone https://github.com/OCR-tech/InnoTSR.git
   cd InnoTSR
   ```

2. Create a virtual environment:

   ```sh
   python -m venv .venv
   .venv\Scripts\Activate
   ```

3. Install the dependencies:

   ```sh
   pip install -r requirements.txt
   ```

<!-- # ssd-mobilenet-v2-tensorflow2-fpnlite-320x320-v1.tar -->
<!-- 4. Download the [SSD MobileNet V2 TensorFlow 2 model](https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1) and extract the files into `app/models/pretrained_model/`.
   - Ensure the directory contains files like `saved_model.pb` and the `saved_model` folder. -->

## Usage

To use the traffic sign recognition system, follow these instructions:

1. **Prepare your environment**: Ensure your system has a camera or input method to capture traffic signs.
2. **Start the application**: Run the main script to initiate the traffic sign recognition system.
3. **Voice Command Alerts**: The system will provide real-time voice alerts for detected traffic signs.

Run the following commands:

<!-- 1. Train the model: Ensure you have a dataset of traffic sign images in the `app/dataset/raw_data`.
   ```sh
   python scripts/train.py
   ``` -->

<!-- 2. **Run the detection script**:
   ```sh
   python scripts/detect.py
   ``` -->

<!-- 2. **Run the user interface:**:
   ```sh
   python ui/main.py
   ``` -->

1. Run the application:
   ```sh
   python app/main.py
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

For contributions, please follow the steps below:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have questions, suggestions, or would like to contribute, feel free to reach out:

- **Email**: ocrtech.mail@gmail.com
- **Website**: [https://ocr-tech.github.io/InnoTSR](https://ocr-tech.github.io/InnoTSR/)
- **GitHub**: [https://github.com/OCR-tech](https://github.com/OCR-tech)
