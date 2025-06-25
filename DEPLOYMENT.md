# 🚀 Deployment Guide for Resume Screening AI

This guide provides step-by-step instructions to deploy your Resume Screening AI application on different platforms.

## 📋 Prerequisites

Before deploying, ensure you have:
- All model files: `clf.pkl`, `tfidf.pkl`, `encoder.pkl`, `model_metadata.pkl`
- All application files: `app.py`, `requirements.txt`
- Git repository set up

## 🌐 Option 1: Streamlit Cloud (Recommended - Free)

### Steps:
1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and branch
   - Set main file path: `app.py`
   - Click "Deploy"

### Advantages:
- ✅ Free hosting
- ✅ Automatic deployments
- ✅ Easy setup
- ✅ Built for Streamlit apps

---

## ☁️ Option 2: Heroku

### Steps:
1. **Install Heroku CLI**
   ```bash
   # macOS
   brew install heroku/brew/heroku
   
   # Or download from: https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

4. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

5. **Open the app**
   ```bash
   heroku open
   ```

### Advantages:
- ✅ Free tier available
- ✅ Scalable
- ✅ Good for production

---

## 🐳 Option 3: Docker

### Local Docker Deployment:
```bash
# Build the image
docker build -t resume-screening-ai .

# Run the container
docker run -p 8501:8501 resume-screening-ai
```

### Docker Hub Deployment:
```bash
# Tag your image
docker tag resume-screening-ai your-username/resume-screening-ai

# Push to Docker Hub
docker push your-username/resume-screening-ai
```

### Advantages:
- ✅ Portable
- ✅ Consistent environment
- ✅ Easy scaling

---

## 🔧 Option 4: Railway

### Steps:
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Railway will automatically detect and deploy your app
4. Set environment variables if needed

### Advantages:
- ✅ Free tier
- ✅ Automatic deployments
- ✅ Easy setup

---

## 🌍 Option 5: Google Cloud Platform (GCP)

### Steps:
1. **Install Google Cloud SDK**
2. **Create a project**
   ```bash
   gcloud init
   ```

3. **Deploy to App Engine**
   ```bash
   gcloud app deploy
   ```

### Advantages:
- ✅ Highly scalable
- ✅ Production-ready
- ✅ Good integration with other Google services

---

## 📊 Option 6: AWS

### Steps:
1. **Create an EC2 instance**
2. **Install dependencies**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip
   pip3 install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py --server.port=8501 --server.address=0.0.0.0
   ```

### Advantages:
- ✅ Highly scalable
- ✅ Production-ready
- ✅ Many services available

---

## 🔍 Troubleshooting

### Common Issues:

1. **Model files not found**
   - Ensure all `.pkl` files are in the repository
   - Check file paths in `app.py`

2. **Dependencies not installed**
   - Verify `requirements.txt` is complete
   - Check for system dependencies in `packages.txt`

3. **Port issues**
   - Ensure port 8501 is exposed
   - Check firewall settings

4. **Memory issues**
   - Model files are large (~234MB)
   - Consider using a platform with sufficient memory

### Performance Tips:

1. **Optimize model loading**
   - Use `@st.cache_resource` decorator
   - Consider model compression

2. **File size limits**
   - Keep upload size under 10MB
   - Compress large files

3. **Caching**
   - Enable Streamlit caching for better performance

---

## 📈 Monitoring & Maintenance

### After Deployment:

1. **Monitor performance**
   - Check response times
   - Monitor error rates
   - Track usage statistics

2. **Regular updates**
   - Update dependencies
   - Retrain models periodically
   - Monitor for security updates

3. **Backup strategy**
   - Backup model files
   - Version control all changes
   - Document deployment procedures

---

## 🎯 Recommended Deployment Strategy

### For Development/Testing:
- **Streamlit Cloud** - Quick and easy

### For Production:
- **Heroku** or **Railway** - Good balance of features and ease
- **AWS/GCP** - For enterprise-level applications

### For Learning/Demo:
- **Streamlit Cloud** - Perfect for showcasing

---

## 📞 Support

If you encounter issues:
1. Check the platform's documentation
2. Review error logs
3. Test locally first
4. Ensure all files are properly committed

---

**Happy Deploying! 🚀** 