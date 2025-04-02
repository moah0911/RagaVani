"""
AI Service Module for RagaVani application

This module provides AI-powered features for the application, including
explanations of Indian classical music concepts, analysis of audio recordings,
and generation of personalized recommendations.
"""

import time
import json
import requests
import logging
from typing import Dict, List, Optional, Union, Any

import google.generativeai as genai
import numpy as np

# Import environment utilities
from utils.env_utils import get_environment_variable, load_environment_variables

# Configure logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_environment_variables()

def initialize_gemini():
    """Initialize the Gemini API client"""
    # Get API key from environment variables
    api_key = get_environment_variable("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    
    # Set default parameters for models
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
    
    logger.info("Gemini API initialized successfully")
    return generation_config

def get_ai_response(prompt: str, analysis_type: str = "general_query") -> str:
    """
    Get response from AI model
    
    Parameters:
        prompt (str): The prompt/query to send to the AI
        analysis_type (str): Type of analysis or query
    
    Returns:
        str: AI response text
    """
    try:
        # Initialize the Gemini client
        generation_config = initialize_gemini()
        
        # Select the appropriate model
        model = genai.GenerativeModel("gemini-2.0-flash", generation_config=generation_config)
        
        # Construct the full prompt with instructions based on analysis type
        full_prompt = construct_prompt(prompt, analysis_type)
        
        # Get response
        response = model.generate_content(full_prompt)
        
        return response.text
    except Exception as e:
        error_message = f"Error with AI service: {str(e)}"
        logger.error(error_message)
        
        # Return a friendly error message
        return f"I encountered an issue while processing your request. {error_message}"

def construct_prompt(user_query: str, analysis_type: str) -> str:
    """
    Construct a specialized prompt based on the type of analysis
    
    Parameters:
        user_query (str): The user's query
        analysis_type (str): Type of analysis or query
    
    Returns:
        str: Constructed prompt
    """
    base_prompt = f"You are an expert in Indian classical music, with deep knowledge of both Hindustani and Carnatic traditions.\n\nUser query: {user_query}\n\n"
    
    if analysis_type == "raga_explanation":
        prompt = base_prompt + """
        Please provide a comprehensive explanation of this raga, including:
        1. Its historical origins and development
        2. Its melodic structure (notes, vadi, samvadi)
        3. Its emotional qualities and time of day for performance
        4. Notable compositions and performers
        5. Similar ragas and how to distinguish them
        
        Use proper musical terminology where appropriate, and explain any technical terms.
        """
    
    elif analysis_type == "tala_explanation":
        prompt = base_prompt + """
        Please provide a comprehensive explanation of this tala (rhythm cycle), including:
        1. Its structure and organization of beats
        2. Its usage in different forms of compositions
        3. Notable compositions that use this tala
        4. Variations and related talas
        5. Performance techniques and challenges
        
        Use proper musical terminology where appropriate, and explain any technical terms.
        """
    
    elif analysis_type == "instrument_explanation":
        prompt = base_prompt + """
        Please provide a comprehensive explanation of this instrument, including:
        1. Its historical development and construction
        2. Playing techniques and sound production
        3. Role in Indian classical music
        4. Notable performers and styles
        5. Key maintenance and practice considerations
        
        Use proper musical terminology where appropriate, and explain any technical terms.
        """
    
    elif analysis_type == "technique_explanation":
        prompt = base_prompt + """
        Please provide a comprehensive explanation of this technique, including:
        1. How it is executed and practiced
        2. Its role and importance in Indian classical music
        3. Variations across different instruments or vocal styles
        4. Notable examples in performances
        5. Tips for developing proficiency
        
        Use proper musical terminology where appropriate, and explain any technical terms.
        """
    
    elif analysis_type == "recommendation":
        prompt = base_prompt + """
        Based on the user's interests or preferences, please provide thoughtful recommendations for:
        1. Specific recordings or performances to listen to
        2. Artists who excel in this area
        3. Related ragas, talas, or compositions to explore
        4. Resources for deeper understanding
        
        Explain the reasoning behind your recommendations to help the user understand the connections.
        """
    
    elif analysis_type == "audio_analysis":
        prompt = base_prompt + """
        The user has shared an audio analysis result. Please interpret these results in a helpful way:
        1. Explain what the detected features indicate about the music
        2. Provide context for any identified ragas or talas
        3. Note any interesting patterns or unusual elements
        4. Suggest what the user might listen for in similar recordings
        
        Use proper musical terminology where appropriate, and explain any technical terms.
        """
    
    else:  # General query
        prompt = base_prompt + """
        Please provide a clear, accurate, and educational response about Indian classical music. 
        Include historical context, musical concepts, and practical aspects where relevant.
        Use proper musical terminology where appropriate, and explain any technical terms.
        If the question is ambiguous, address the most likely interpretation and note other possibilities.
        """
    
    return prompt

def get_ai_analysis(analysis_type: str, **kwargs) -> str:
    """
    Get AI analysis based on the specified type
    
    Parameters:
        analysis_type (str): Type of analysis to perform
        **kwargs: Additional parameters specific to the analysis type
    
    Returns:
        str: Analysis result
    """
    try:
        if analysis_type == "general_query":
            # For general queries, extract the query and pass it to the AI
            query = kwargs.get("query", "")
            if not query:
                return "No query provided. Please ask a specific question about Indian classical music."
            
            return get_ai_response(query, analysis_type)
        
        elif analysis_type == "raga_analysis":
            # For raga analysis, extract raga info and format a prompt
            raga_name = kwargs.get("raga_name", "")
            raga_info = kwargs.get("raga_info", {})
            
            if not raga_name:
                return "No raga specified for analysis."
            
            prompt = f"Please analyze Raga {raga_name}. "
            
            if raga_info:
                # Format the raga information
                prompt += "Here are some details about the raga:\n"
                
                if "notes" in raga_info:
                    aroha = " ".join(raga_info["notes"].get("aroha", []))
                    avaroha = " ".join(raga_info["notes"].get("avaroha", []))
                    prompt += f"Aroha (ascending scale): {aroha}\n"
                    prompt += f"Avaroha (descending scale): {avaroha}\n"
                
                if "vadi" in raga_info:
                    prompt += f"Vadi (important note): {raga_info['vadi']}\n"
                
                if "samvadi" in raga_info:
                    prompt += f"Samvadi (second important note): {raga_info['samvadi']}\n"
                
                if "pakad" in raga_info:
                    prompt += f"Pakad (characteristic phrase): {raga_info['pakad']}\n"
                
                if "time" in raga_info:
                    prompt += f"Time of performance: {raga_info['time']}\n"
                
                if "mood" in raga_info:
                    prompt += f"Mood: {', '.join(raga_info['mood'])}\n"
            
            prompt += "\nPlease explain the characteristics, emotional qualities, and performance considerations for this raga. Also, suggest some notable recordings or artists known for performing this raga."
            
            return get_ai_response(prompt, "raga_explanation")
        
        elif analysis_type == "tala_analysis":
            # For tala analysis, extract tala info and format a prompt
            tala_name = kwargs.get("tala_name", "")
            tala_info = kwargs.get("tala_info", {})
            
            if not tala_name:
                return "No tala specified for analysis."
            
            prompt = f"Please analyze Tala {tala_name}. "
            
            if tala_info:
                # Format the tala information
                prompt += "Here are some details about the tala:\n"
                
                if "beats" in tala_info:
                    prompt += f"Total beats: {tala_info['beats']}\n"
                
                if "vibhags" in tala_info:
                    prompt += f"Vibhag (section) structure: {'+'.join(str(v) for v in tala_info['vibhags'])}\n"
                
                if "clap_pattern" in tala_info:
                    prompt += f"Clap pattern: {' | '.join(tala_info['clap_pattern'])}\n"
                
                if "pattern_description" in tala_info:
                    prompt += f"Pattern description: {tala_info['pattern_description']}\n"
            
            prompt += "\nPlease explain the structure, usage, and performance considerations for this tala. Include information about compositions that commonly use this tala and any special techniques associated with it."
            
            return get_ai_response(prompt, "tala_explanation")
        
        elif analysis_type == "audio_feature_analysis":
            # For audio analysis, extract features and format a prompt
            features = kwargs.get("features", {})
            
            if not features:
                return "No audio features provided for analysis."
            
            prompt = "Please analyze the following audio features:\n\n"
            
            if "detected_raga" in features:
                prompt += f"Detected Raga: {features['detected_raga']} (confidence: {features.get('raga_confidence', 'unknown')}%)\n"
            
            if "detected_tala" in features:
                prompt += f"Detected Tala: {features['detected_tala']} (confidence: {features.get('tala_confidence', 'unknown')}%)\n"
            
            if "ornaments" in features and features["ornaments"]:
                ornament_counts = {}
                for ornament in features["ornaments"]:
                    orn_type = ornament.get("type", "unknown")
                    ornament_counts[orn_type] = ornament_counts.get(orn_type, 0) + 1
                
                prompt += "Detected ornaments:\n"
                for orn_type, count in ornament_counts.items():
                    prompt += f"- {orn_type}: {count} instances\n"
            
            prompt += "\nPlease interpret these features and explain what they indicate about the music. Provide context about the raga and tala, if detected, and explain the significance of the ornaments and their role in the performance."
            
            return get_ai_response(prompt, "audio_analysis")
        
        elif analysis_type == "learning_recommendation":
            # For learning recommendations, extract user interests and level
            interests = kwargs.get("interests", [])
            level = kwargs.get("level", "beginner")
            
            if not interests:
                return "No interests specified for recommendations."
            
            prompt = f"I'm a {level} student of Indian classical music with interest in {', '.join(interests)}. "
            prompt += "Please recommend resources, practice approaches, and pieces to study that would be appropriate for my level and interests."
            
            return get_ai_response(prompt, "recommendation")
        
        else:
            return "Unknown analysis type. Please specify a valid analysis type."
    
    except Exception as e:
        error_message = f"Error performing AI analysis: {str(e)}"
        logger.error(error_message)
        return f"I encountered an issue while analyzing: {error_message}"

# Initialize the model when the module is imported
try:
    initialize_gemini()
    logger.info("Gemini API initialized successfully")
except Exception as e:
    logger.warning(f"Could not initialize Gemini API: {e}")
    logger.warning("The AI assistant will not be available without a valid API key.")
    