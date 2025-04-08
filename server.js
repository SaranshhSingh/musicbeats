const express = require('express');
const cors = require('cors');
const axios = require('axios');
const bodyParser = require('body-parser');
const dotenv = require('dotenv');
const path = require('path');

// Load environment variables
dotenv.config();

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

// Serve the main HTML file
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// API route for emotion detection
app.post('/api/detect-emotion', async (req, res) => {
  try {
    const { text, provider, apiKey, model } = req.body;
    
    if (!text) {
      return res.status(400).json({ error: 'Text is required' });
    }
    
    if (!apiKey) {
      // Fallback to basic keyword detection if no API key is provided
      const emotion = detectEmotionFromText(text);
      return res.json({ emotion });
    }
    
    let detectedEmotion;
    
    switch (provider) {
      case 'openai':
        detectedEmotion = await callOpenAI(text, apiKey, model);
        break;
      case 'huggingface':
        detectedEmotion = await callHuggingFace(text, apiKey, model);
        break;
      case 'cohere':
        detectedEmotion = await callCohere(text, apiKey, model);
        break;
      case 'palm':
        detectedEmotion = await callPalm(text, apiKey, model);
        break;
      default:
        detectedEmotion = detectEmotionFromText(text);
    }
    
    res.json({ emotion: detectedEmotion });
    
  } catch (error) {
    console.error('Error in emotion detection:', error);
    res.status(500).json({ 
      error: 'Error processing request', 
      message: error.message 
    });
  }
});

// OpenAI API call
async function callOpenAI(text, apiKey, model) {
  try {
    const response = await axios.post(
      'https://api.openai.com/v1/chat/completions',
      {
        model: model || 'gpt-3.5-turbo',
        messages: [
          {
            role: 'system',
            content: 'You are an emotion detection system. Analyze the text and return only one of these emotions: happy, sad, energetic, relaxed, focused, or mixed. Return just the emotion word and nothing else.'
          },
          {
            role: 'user',
            content: text
          }
        ],
        temperature: 0.3,
        max_tokens: 10
      },
      {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        }
      }
    );
    
    const emotion = response.data.choices[0].message.content.trim().toLowerCase();
    // Ensure we return one of our valid emotions
    return validateEmotion(emotion);
    
  } catch (error) {
    console.error('OpenAI API error:', error.response?.data || error.message);
    // Fallback to basic detection
    return detectEmotionFromText(text);
  }
}

// HuggingFace API call
async function callHuggingFace(text, apiKey, model) {
  try {
    const response = await axios.post(
      `https://api-inference.huggingface.co/models/${model || 'distilbert-base-uncased-finetuned-emotion'}`,
      { inputs: text },
      {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        }
      }
    );
    
    // Map HuggingFace's emotion classification to our categories
    // This is a simplified mapping and might need adjustment based on the model used
    const result = response.data[0];
    let topEmotion = Object.keys(result).reduce((a, b) => result[a] > result[b] ? a : b);
    
    // Map HuggingFace emotions to our categories
    const emotionMap = {
      'joy': 'happy',
      'happiness': 'happy',
      'sadness': 'sad',
      'anger': 'energetic',
      'fear': 'sad',
      'surprise': 'energetic',
      'disgust': 'sad',
      'neutral': 'relaxed'
    };
    
    return emotionMap[topEmotion] || 'mixed';
    
  } catch (error) {
    console.error('HuggingFace API error:', error.response?.data || error.message);
    return detectEmotionFromText(text);
  }
}

// Cohere API call
async function callCohere(text, apiKey, model) {
  try {
    const response = await axios.post(
      'https://api.cohere.ai/v1/classify',
      {
        model: model || 'large',
        inputs: [text],
        examples: [
          { text: "I feel so happy and excited today!", label: "happy" },
          { text: "Everything is going great, I'm on top of the world", label: "happy" },
          { text: "I'm sad and feeling blue", label: "sad" },
          { text: "Nothing is going right, I feel depressed", label: "sad" },
          { text: "I'm full of energy and ready to take on the world", label: "energetic" },
          { text: "Let's go! I'm pumped up and motivated", label: "energetic" },
          { text: "I'm feeling so relaxed and peaceful", label: "relaxed" },
          { text: "Just chilling and taking it easy", label: "relaxed" },
          { text: "I need to concentrate and get work done", label: "focused" },
          { text: "I'm in the zone and really productive right now", label: "focused" }
        ]
      },
      {
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        }
      }
    );
    
    const prediction = response.data.classifications[0];
    return prediction.prediction; // This should be one of our emotion categories
    
  } catch (error) {
    console.error('Cohere API error:', error.response?.data || error.message);
    return detectEmotionFromText(text);
  }
}

// Google PaLM API call
async function callPalm(text, apiKey, model) {
  try {
    const palmModel = model || 'text-bison';
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${palmModel}:generateText?key=${apiKey}`;
    
    const response = await axios.post(
      apiUrl,
      {
        prompt: {
          text: `Analyze the following text and return only one of these emotions: happy, sad, energetic, relaxed, focused, or mixed. Only return the emotion word, nothing else.
          
          Text: "${text}"`
        },
        temperature: 0.3,
        maxOutputTokens: 10
      }
    );
    
    const generatedText = response.data.candidates[0].output.trim().toLowerCase();
    return validateEmotion(generatedText);
    
  } catch (error) {
    console.error('PaLM API error:', error.response?.data || error.message);
    return detectEmotionFromText(text);
  }
}

// Basic emotion detection from text (fallback method)
function detectEmotionFromText(text) {
  text = text.toLowerCase();
  
  // Emotion keywords mapping
  const emotionKeywords = {
    happy: ["happy", "joy", "joyful", "excited", "cheerful", "upbeat", "delighted", "glad", "pleased", "content", "elated"],
    sad: ["sad", "down", "depressed", "unhappy", "blue", "gloomy", "melancholy", "heartbroken", "upset", "sorrowful"],
    energetic: ["energetic", "pumped", "motivated", "active", "dynamic", "charged", "peppy", "lively", "vigorous", "spirited"],
    relaxed: ["relaxed", "calm", "peaceful", "tranquil", "serene", "chill", "mellow", "soothing", "easy-going", "laid-back"],
    focused: ["focused", "concentrated", "studying", "working", "determined", "attentive", "productive", "driven", "busy", "studious"]
  };
  
  // Check for each emotion keyword
  for (const [emotion, keywords] of Object.entries(emotionKeywords)) {
    for (const keyword of keywords) {
      if (text.includes(keyword)) {
        return emotion;
      }
    }
  }
  
  // Default to mixed if no clear emotion is detected
  return "mixed";
}

// Validate emotion to ensure it's one of our valid categories
function validateEmotion(emotion) {
  const validEmotions = ['happy', 'sad', 'energetic', 'relaxed', 'focused', 'mixed'];
  
  if (validEmotions.includes(emotion)) {
    return emotion;
  }
  
  // Additional checks for variations
  if (emotion.includes('happ') || emotion.includes('joy')) return 'happy';
  if (emotion.includes('sad') || emotion.includes('depress')) return 'sad';
  if (emotion.includes('energ') || emotion.includes('excit')) return 'energetic';
  if (emotion.includes('relax') || emotion.includes('calm')) return 'relaxed';
  if (emotion.includes('focus') || emotion.includes('concentr')) return 'focused';
  
  // Default fallback
  return 'mixed';
}

// Start the server
app.listen(PORT, () => {
  console.log(`AI DJ Server running on port ${PORT}`);
});