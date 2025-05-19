# Radiology Assistant

A supportive tool for radiologists that enhances the interpretation of brain MRI scans by highlighting potential tumor areas and providing classification suggestions.

## Features

- Upload and analyze brain MRI scans
- View sample MRI scans with synthetic tumors
- Get tumor detection with visual highlighting
- Receive classification suggestions (Likely Benign, Possibly Malignant, Likely Malignant)
- See confidence levels and recommendations

## Technical Implementation

- Flask web framework for the backend
- Bootstrap for responsive UI
- Matplotlib and NumPy for image processing and visualization
- PIL for image manipulation
- TensorFlow integration (ready for future ML model integration)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/radiology-assistant.git
cd radiology-assistant

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Usage

1. Start the application with `python app.py`
2. Open your browser and navigate to `http://localhost:8080`
3. Upload an MRI scan or use one of the sample images
4. View the analysis results with highlighted tumor regions

## Screenshots

(Screenshots will be added here)

## Future Improvements

- Implement a proper deep learning model for more accurate tumor detection
- Add more detailed reporting features
- Include 3D visualization of tumor regions
- Support for different types of medical imaging (CT, X-ray, etc.)

## License

MIT