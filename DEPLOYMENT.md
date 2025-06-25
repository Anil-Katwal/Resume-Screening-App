# Deployment Guide for Resume Screening AI

This guide provides step-by-step instructions to deploy your Resume Screening AI application on different platforms.

## Prerequisites

Before deploying, ensure you have:
- All model files: `clf.pkl`, `tfidf.pkl`, `encoder.pkl`, `model_metadata.pkl`
- All application files: `app.py`, `requirements.txt`
- Git repository set up
- 
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

##  Monitoring & Maintenance

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

##  Recommended Deployment Strategy

### For Development/Testing:
- **Streamlit Cloud** - Quick and easy

### For Production:
- **Heroku** or **Railway** - Good balance of features and ease
- **AWS/GCP** - For enterprise-level applications

### For Learning/Demo:
- **Streamlit Cloud** - Perfect for showcasing

---

## Support

If you encounter issues:
1. Check the platform's documentation
2. Review error logs
3. Test locally first
4. Ensure all files are properly committed

---


