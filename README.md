#Resume Screening AI

A robust and intelligent web application for automated resume analysis and job category prediction using machine learning.

##  Features

- **Multi-format Support**: Upload resumes in PDF, DOCX, or TXT formats
- **Batch Processing**: Process multiple files simultaneously
- **Real-time Analytics**: Interactive charts and statistics
- **Confidence Scoring**: Get prediction confidence levels
- **Error Handling**: Comprehensive error logging and user feedback
- **Export Results**: Download analysis results as CSV
- **Modern UI**: Beautiful, responsive interface with custom styling
- **Session Management**: Persistent data across app sessions

## Quick Start

### Prerequisites
- Python 3.8 or higher
- All required model files (`clf.pkl`, `tfidf.pkl`, `encoder.pkl`)

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd Resume-Screening-App
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## Usage Guide

### Uploading Resumes
1. Click on the file upload area or drag and drop files
2. Supported formats: PDF, DOCX, TXT
3. Maximum file size: 10MB per file
4. You can upload multiple files at once

### Viewing Results
- **Dashboard**: Check real-time statistics in the sidebar
- **Category Distribution**: Interactive pie chart showing job categories
- **Results Table**: Detailed analysis with confidence scores
- **Error Log**: View any processing errors in the expandable section

### Exporting Data
- Click "Download Results CSV" to export analysis results
- Use "Clear All Data" to reset the session

## Technical Details

### Machine Learning Pipeline
- **Text Extraction**: Robust extraction from multiple file formats
- **Text Preprocessing**: Cleaning and normalization
- **Feature Engineering**: TF-IDF vectorization
- **Classification**: Support Vector Machine (SVM) model
- **Post-processing**: Confidence scoring and category mapping

### Error Handling
- File size validation
- Format validation
- Text extraction error handling
- Model loading error handling
- Comprehensive error logging

### Performance Features
- Cached model loading
- Session state management
- Efficient batch processing
- Memory optimization

##Supported Job Categories

The model can predict various job categories including:
- Data Science
- Software Development
- Marketing
- Sales
- Human Resources
- Finance
- Healthcare
- And more...

##  Customization

### Adding New File Formats
1. Add the format to the `handle_file_upload()` function
2. Create a corresponding extraction function
3. Update the file uploader type list

### Modifying the UI
- Edit the CSS styles in the `st.markdown()` section
- Modify the layout structure in the `main()` function
- Add new visualizations using Plotly

### Model Updates
- Replace the pickle files with new trained models
- Ensure compatibility with the existing preprocessing pipeline

##Troubleshooting

### Common Issues

**"Model file not found" error**
- Ensure `clf.pkl`, `tfidf.pkl`, and `encoder.pkl` are in the project directory
- Check file permissions

**"File size too large" error**
- Reduce file size to under 10MB
- Compress PDF files if necessary

**"Extracted text is too short" error**
- Ensure the file contains readable text
- Check if the file is corrupted or password-protected

**Import errors**
- Install all dependencies: `pip install -r requirements.txt`
- Check Python version compatibility

### Performance Tips
- Use SSD storage for faster file processing
- Close other applications to free up memory
- Process files in smaller batches for large datasets

## 📈 Future Enhancements

- [ ] OCR support for image-based resumes
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] API endpoint for integration
- [ ] User authentication and data persistence
- [ ] Custom model training interface
- [ ] Resume scoring and ranking
- [ ] Skills extraction and matching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Streamlit for the web interface
- Powered by scikit-learn for machine learning
- Uses Plotly for interactive visualizations
- Inspired by the need for efficient resume screening

---

**Built with ❤️ using Streamlit and Machine Learning**