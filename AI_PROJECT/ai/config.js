// Gemini API Configuration
// यहाँ अपना Gemini API Key डालें
// Get your API key from: https://makersuite.google.com/app/apikey
// या https://aistudio.google.com/app/apikey

const GEMINI_API_KEY = 'AIzaSyB1I6N5aIaHMX34hKD4Mr_xD8EwxVAFVj8';

// Gemini API Endpoint - Using current models that support vision
// gemini-1.5-flash: Fast model with vision support (recommended)
// gemini-1.5-pro: More capable model with vision support
const GEMINI_MODEL = 'gemini-1.5-flash';
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`;

// Custom Model API Configuration
// Your trained EfficientNetB0 model backend
const CUSTOM_MODEL_API_URL = 'http://localhost:5000/predict-detailed';
const USE_CUSTOM_MODEL_FIRST = true; // Set to true to use your model first, then Gemini for enhancement

// Export configuration
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { GEMINI_API_KEY, GEMINI_API_URL, GEMINI_MODEL, CUSTOM_MODEL_API_URL, USE_CUSTOM_MODEL_FIRST };
}

