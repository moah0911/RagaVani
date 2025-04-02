"""
Tala Knowledge Base for RagaVani application

This module provides information about Indian classical talas (rhythm cycles),
including their structure, patterns, and classifications.
"""

# Database of talas with their properties
TALA_DATABASE = {
    "Teentaal": {
        "beats": 16,
        "vibhags": [4, 4, 4, 4],
        "clap_pattern": ["X", "2", "0", "3"],
        "pattern_description": "4+4+4+4 beats",
        "description": "Teentaal (or Trital) is the most common tala in Hindustani classical music. It consists of 16 beats divided into four equal sections. It is versatile and can be played at various tempos, making it suitable for a wide range of compositions and improvisations, from slow elaborate alaap to fast taans.",
        "tempo_range": ["Slow (vilambit)", "Medium (madhya)", "Fast (drut)"],
        "instruments": ["Tabla", "Pakhawaj"],
        "variations": ["Sitarkhani", "Chautala"]
    },
    "Ektaal": {
        "beats": 12,
        "vibhags": [2, 2, 2, 2, 2, 2],
        "clap_pattern": ["X", "0", "2", "0", "3", "4"],
        "pattern_description": "2+2+2+2+2+2 beats",
        "description": "Ektaal consists of 12 beats divided into six equal sections of 2 beats each. It is commonly used for slower, more lyrical compositions like thumri, dadra, and some dhrupad compositions. The rhythmic structure allows for a graceful flow and is well-suited for expressing emotional content.",
        "tempo_range": ["Slow (vilambit)", "Medium (madhya)"],
        "instruments": ["Tabla", "Pakhawaj"],
        "variations": ["Choutaal", "Farodast"]
    },
    "Jhaptaal": {
        "beats": 10,
        "vibhags": [2, 3, 2, 3],
        "clap_pattern": ["X", "2", "0", "3"],
        "pattern_description": "2+3+2+3 beats",
        "description": "Jhaptaal consists of 10 beats arranged in an asymmetrical pattern of 2+3+2+3. This creates a distinctive rhythmic feel that is both engaging and challenging. It is commonly used in instrumental music, particularly for compositions that require a more complex rhythmic structure.",
        "tempo_range": ["Medium (madhya)", "Fast (drut)"],
        "instruments": ["Tabla", "Pakhawaj"],
        "variations": []
    },
    "Rupak Taal": {
        "beats": 7,
        "vibhags": [3, 2, 2],
        "clap_pattern": ["0", "2", "3"],
        "pattern_description": "3+2+2 beats",
        "description": "Rupak Taal is unique in that it begins with the khali (empty) beat rather than the sam (first beat). It consists of 7 beats arranged in a 3+2+2 pattern. This creates a distinctive 'limping' feel that is particularly well-suited for certain types of compositions, especially those with a lilting, graceful character.",
        "tempo_range": ["Medium (madhya)", "Fast (drut)"],
        "instruments": ["Tabla"],
        "variations": []
    },
    "Dadra": {
        "beats": 6,
        "vibhags": [3, 3],
        "clap_pattern": ["X", "0"],
        "pattern_description": "3+3 beats",
        "description": "Dadra is a light tala consisting of 6 beats divided into two equal sections of 3 beats each. It is commonly used in semi-classical forms like thumri, dadra, ghazal, and light classical songs. Its simple structure makes it accessible and popular for more lyrical, emotion-based compositions.",
        "tempo_range": ["Medium (madhya)"],
        "instruments": ["Tabla", "Dholak"],
        "variations": []
    },
    "Keherwa": {
        "beats": 8,
        "vibhags": [4, 4],
        "clap_pattern": ["X", "0"],
        "pattern_description": "4+4 beats",
        "description": "Keherwa is a popular tala in folk and light classical music, consisting of 8 beats divided into two equal sections. It has a lively, bouncy feel and is commonly used in bhajans, ghazals, and folk songs. Its symmetric structure makes it easy to follow and suited for dance accompaniment.",
        "tempo_range": ["Medium (madhya)", "Fast (drut)"],
        "instruments": ["Tabla", "Dholak", "Khol"],
        "variations": ["Bhajani"]
    },
    "Dhamar": {
        "beats": 14,
        "vibhags": [5, 2, 3, 4],
        "clap_pattern": ["X", "2", "0", "3"],
        "pattern_description": "5+2+3+4 beats",
        "description": "Dhamar Taal is a complex tala of 14 beats, arranged in an asymmetrical pattern of 5+2+3+4. It is primarily associated with the dhamar genre of dhrupad singing and is traditionally played on the pakhawaj rather than the tabla. The asymmetric grouping creates a sophisticated rhythmic cycle used for specific compositional forms.",
        "tempo_range": ["Medium (madhya)"],
        "instruments": ["Pakhawaj", "Tabla"],
        "variations": []
    },
    "Chautaal": {
        "beats": 12,
        "vibhags": [2, 2, 2, 2, 2, 2],
        "clap_pattern": ["X", "2", "0", "3", "0", "4"],
        "pattern_description": "2+2+2+2+2+2 beats",
        "description": "Chautaal is a 12-beat tala divided into six sections of 2 beats each. It is frequently used in dhrupad singing and pakhawaj playing. Its symmetric structure allows for complex rhythmic elaborations while maintaining a steady pulse that supports the melodic development in dhrupad.",
        "tempo_range": ["Slow (vilambit)", "Medium (madhya)"],
        "instruments": ["Pakhawaj", "Tabla"],
        "variations": ["Ektaal"]
    },
    "Jhoomra": {
        "beats": 14,
        "vibhags": [3, 4, 3, 4],
        "clap_pattern": ["X", "2", "0", "3"],
        "pattern_description": "3+4+3+4 beats",
        "description": "Jhoomra is a 14-beat tala arranged in a pattern of 3+4+3+4. It is often used in khayal singing and has a distinctive swing that gives it a graceful, flowing quality. The alternating groups of 3 and 4 create a rhythmic tension and release that is particularly effective for certain melodic phrases.",
        "tempo_range": ["Slow (vilambit)", "Medium (madhya)"],
        "instruments": ["Tabla"],
        "variations": []
    },
    "Tilwada": {
        "beats": 16,
        "vibhags": [4, 4, 4, 4],
        "clap_pattern": ["X", "2", "0", "3"],
        "pattern_description": "4+4+4+4 beats",
        "description": "Tilwada is a 16-beat tala that, while having the same structure as Teentaal (4+4+4+4), has a different feel and is played with distinctive bols (rhythmic syllables). It is typically used for slower, more elaborate compositions where the rhythmic intricacies can be fully explored and appreciated.",
        "tempo_range": ["Slow (vilambit)"],
        "instruments": ["Tabla"],
        "variations": ["Teentaal"]
    },
    "Adha Chautaal": {
        "beats": 7,
        "vibhags": [2, 2, 1, 2],
        "clap_pattern": ["X", "2", "0", "3"],
        "pattern_description": "2+2+1+2 beats",
        "description": "Adha Chautaal is a 7-beat tala arranged in a pattern of 2+2+1+2. The name 'adha' means 'half', suggesting it is half of a larger tala. It has an interesting asymmetric rhythm due to the single beat in the third section, creating a distinctive lilt that works well for certain types of compositions.",
        "tempo_range": ["Medium (madhya)"],
        "instruments": ["Tabla", "Pakhawaj"],
        "variations": []
    },
    "Deepchandi": {
        "beats": 14,
        "vibhags": [3, 4, 3, 4],
        "clap_pattern": ["X", "2", "0", "3"],
        "pattern_description": "3+4+3+4 beats",
        "description": "Deepchandi is a 14-beat tala commonly used in thumri, dadra, and other semi-classical forms. It has the same structure as Jhoomra (3+4+3+4) but with different bols and a different feel. It has a romantic, expressive quality that makes it particularly suitable for the emotional content of thumri singing.",
        "tempo_range": ["Medium (madhya)"],
        "instruments": ["Tabla"],
        "variations": ["Jhoomra"]
    }
}

def get_tala_info(tala_name):
    """
    Get information about a tala
    
    Parameters:
        tala_name (str): Name of the tala
    
    Returns:
        dict or None: Dictionary containing tala information, or None if not found
    """
    # Try exact match first
    if tala_name in TALA_DATABASE:
        return TALA_DATABASE[tala_name]
    
    # Try case-insensitive match
    for name, info in TALA_DATABASE.items():
        if name.lower() == tala_name.lower():
            return info
    
    return None

def get_all_talas():
    """
    Get a list of all available talas
    
    Returns:
        list: List of tala names
    """
    return list(TALA_DATABASE.keys())

def get_tala_by_beats(num_beats):
    """
    Get talas with a specific number of beats
    
    Parameters:
        num_beats (int): Number of beats to search for
    
    Returns:
        list: List of tala names with the specified number of beats
    """
    matching_talas = []
    
    for tala_name, tala_info in TALA_DATABASE.items():
        if "beats" in tala_info and tala_info["beats"] == num_beats:
            matching_talas.append(tala_name)
    
    return matching_talas

def get_tala_clap_pattern(tala_name):
    """
    Get the clap pattern for a tala
    
    Parameters:
        tala_name (str): Name of the tala
    
    Returns:
        list or None: List of strings representing the clap pattern, or None if not found
    """
    tala_info = get_tala_info(tala_name)
    if tala_info and "clap_pattern" in tala_info:
        return tala_info["clap_pattern"]
    
    return None

def compare_talas(tala1, tala2):
    """
    Compare two talas and highlight similarities and differences
    
    Parameters:
        tala1 (str): Name of first tala
        tala2 (str): Name of second tala
    
    Returns:
        dict: Dictionary containing comparison results
    """
    info1 = get_tala_info(tala1)
    info2 = get_tala_info(tala2)
    
    if not info1 or not info2:
        return None
    
    comparison = {
        "tala1": tala1,
        "tala2": tala2,
        "similarities": {},
        "differences": {}
    }
    
    # Compare beats
    if info1.get("beats") == info2.get("beats"):
        comparison["similarities"]["beats"] = info1.get("beats")
    else:
        comparison["differences"]["beats"] = {
            "tala1": info1.get("beats"),
            "tala2": info2.get("beats")
        }
        
        # Calculate mathematical relationship between beat counts
        if info1.get("beats") and info2.get("beats"):
            relationship = calculate_mathematical_relationship(info1["beats"], info2["beats"])
            comparison["differences"]["beat_relationship"] = relationship
    
    # Compare vibhags (sections)
    if info1.get("vibhags") == info2.get("vibhags"):
        comparison["similarities"]["vibhags"] = info1.get("vibhags")
    else:
        comparison["differences"]["vibhags"] = {
            "tala1": info1.get("vibhags"),
            "tala2": info2.get("vibhags")
        }
    
    # Compare clap patterns
    if info1.get("clap_pattern") == info2.get("clap_pattern"):
        comparison["similarities"]["clap_pattern"] = info1.get("clap_pattern")
    else:
        comparison["differences"]["clap_pattern"] = {
            "tala1": info1.get("clap_pattern"),
            "tala2": info2.get("clap_pattern")
        }
    
    # Compare instruments
    if "instruments" in info1 and "instruments" in info2:
        instruments1 = set(info1["instruments"])
        instruments2 = set(info2["instruments"])
        
        common_instruments = instruments1.intersection(instruments2)
        if common_instruments:
            comparison["similarities"]["instruments"] = list(common_instruments)
        
        unique_instruments1 = instruments1 - instruments2
        unique_instruments2 = instruments2 - instruments1
        
        if unique_instruments1 or unique_instruments2:
            comparison["differences"]["instruments"] = {
                "tala1": list(unique_instruments1) if unique_instruments1 else [],
                "tala2": list(unique_instruments2) if unique_instruments2 else []
            }
    
    # Compare tempo ranges
    if "tempo_range" in info1 and "tempo_range" in info2:
        tempo1 = set(info1["tempo_range"])
        tempo2 = set(info2["tempo_range"])
        
        common_tempos = tempo1.intersection(tempo2)
        if common_tempos:
            comparison["similarities"]["tempo_range"] = list(common_tempos)
        
        unique_tempos1 = tempo1 - tempo2
        unique_tempos2 = tempo2 - tempo1
        
        if unique_tempos1 or unique_tempos2:
            comparison["differences"]["tempo_range"] = {
                "tala1": list(unique_tempos1) if unique_tempos1 else [],
                "tala2": list(unique_tempos2) if unique_tempos2 else []
            }
    
    return comparison

def calculate_mathematical_relationship(beats1, beats2):
    """
    Calculate the mathematical relationship between two tala beat counts
    
    Parameters:
        beats1 (int): Beats in first tala
        beats2 (int): Beats in second tala
    
    Returns:
        dict: Dictionary containing mathematical relationship information
    """
    relationship = {}
    
    # Calculate ratio in simplest form
    common_divisor = gcd(beats1, beats2)
    ratio_1 = beats1 // common_divisor
    ratio_2 = beats2 // common_divisor
    
    relationship["ratio"] = f"{ratio_1}:{ratio_2}"
    
    # Calculate if one is a multiple of the other
    if beats1 % beats2 == 0:
        relationship["multiple"] = f"{beats1} is {beats1 // beats2} times {beats2}"
    elif beats2 % beats1 == 0:
        relationship["multiple"] = f"{beats2} is {beats2 // beats1} times {beats1}"
    
    # Check for common rhythmic cycles
    if common_divisor > 1:
        relationship["common_divisor"] = common_divisor
        relationship["comment"] = f"Both talas have a common rhythmic unit of {common_divisor} beats"
    
    return relationship

def gcd(a, b):
    """Calculate greatest common divisor of two numbers"""
    while b:
        a, b = b, a % b
    return a